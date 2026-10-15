"""Real CUDA/FSDP2 integration checks; run with torchrun --nproc-per-node=2.

Set DMUON_TEST_REPLICAS=2 and use four ranks to exercise HSDP.
"""

import copy
import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from transformers.utils import ModelOutput

from veomni.arguments.arguments_types import MixedPrecisionConfig, OptimizerConfig
from veomni.distributed.parallel_state import get_parallel_state, init_parallel_state
from veomni.optim.dmuon import build_dmuon, clip_dmuon_grad_norm, dedicate_model
from veomni.trainer.base import BaseTrainer


@dataclass
class Losses(ModelOutput):
    ce: torch.Tensor | None = None
    flow: torch.Tensor | None = None


class PairedBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.native = nn.Linear(32, 32, bias=False)
        self.action = nn.Linear(32, 32, bias=False)
        self.norm = nn.LayerNorm(32)

    def forward(self, hidden, action, ki, robot):
        hidden = torch.tanh(self.native(self.norm(hidden)))
        if robot:
            prefix = hidden.detach() if ki else hidden
            action = torch.tanh(self.action.forward(action + prefix))
        else:
            action = action + self.action.weight.sum() * 0
        return hidden, action


class BranchedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(64, 32)
        self.layers = nn.ModuleList([PairedBlock(), PairedBlock()])
        self.action_input = nn.Linear(32, 32, bias=False)
        self.lm_head = nn.Linear(32, 32, bias=False)
        self.register_buffer("scale", torch.tensor(0.5))

    def forward(self, ids, actions, ki=False, robot=True):
        hidden = self.embed_tokens(ids)
        # Match MoT projections read directly rather than via module __call__.
        action = actions @ self.action_input.weight.T
        for layer in self.layers:
            hidden, action = layer(hidden, action, ki, robot)
        ce = self.lm_head(hidden).square().mean() * self.scale
        flow = action.square().mean() if robot else action.sum() * 0
        return Losses(ce=ce, flow=flow)


def make_model(initial, backend, coefficients, bf16, reshard):
    model = copy.deepcopy(initial)
    mp = MixedPrecisionConfig(enable=bf16, param_dtype="bfloat16", reduce_dtype="bfloat16")
    dedicate_model(model, OptimizerConfig(type="dmuon"), mp, ["PairedBlock"], reshard)
    ps = get_parallel_state()
    policy = (
        MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)
        if bf16
        else MixedPrecisionPolicy()
    )
    for block in model.layers:
        fully_shard(block, mesh=ps.fsdp_mesh, mp_policy=policy, reshard_after_forward=reshard)
    fully_shard(model, mesh=ps.fsdp_mesh, mp_policy=policy, reshard_after_forward=reshard)
    import dmuon

    before = dmuon.get_model_state_dict(model, cpu_offload=True, rank0_only=False)
    rng = torch.cuda.get_rng_state()
    cpu_rng = torch.get_rng_state()
    optimizer = build_dmuon(
        model,
        1e-3,
        (0.9, 0.95),
        1e-8,
        0.1,
        None,
        [],
        [],
        {
            "lr": 5e-3,
            "dmuon_ns_backend": backend,
            "dmuon_ns_coefficients": coefficients,
        },
    )
    assert not optimizer.state
    assert_state_equal(dmuon.get_model_state_dict(model, cpu_offload=True, rank0_only=False), before)
    assert torch.equal(torch.cuda.get_rng_state(), rng)
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    return model, optimizer


def check_graph_replay_with_pinning(initial):
    from dmuon.optim import ns_graph

    if not ns_graph.enabled():
        return
    model, optimizer = make_model(initial, "gram", "polar_express", True, False)
    errors = []
    started, stop = threading.Event(), threading.Event()

    def pin():
        try:
            while not stop.is_set():
                torch.empty(16 * 1024 * 1024, dtype=torch.uint8, pin_memory=True)
                started.set()
        except Exception as error:
            errors.append(error)

    worker = threading.Thread(target=pin)
    worker.start()
    try:
        assert started.wait(timeout=30), "Pinned-memory worker did not start"
        for param in optimizer._dedicated_params:
            if id(param) not in optimizer._dp_to_muon_group_idx:
                continue
            example = param._owned_data.detach()
            for shift in range(1, 4):
                value = example.roll(shift, dims=1).contiguous()
                expected = optimizer._ns.local(value, optimizer._ns_steps)
                torch.testing.assert_close(optimizer._ns_compute(value, is_tp=False), expected, rtol=0, atol=0)
                assert torch.randn(32, device="cuda").isfinite().all()
    finally:
        stop.set()
        worker.join()
    assert not errors
    assert not optimizer.state
    if dist.get_rank() == 0:
        print("PASS warmed graph replay with concurrent pinned-memory allocation and subsequent RNG", flush=True)


def check_gradients(initial, backend, coefficients, bf16, reshard, ki):
    import dmuon

    model, optimizer = make_model(initial, backend, coefficients, bf16, reshard)
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(700 + rank)
    ids = torch.randint(64, (4,), device="cuda")
    actions = torch.randn(4, 32, device="cuda")
    golden = copy.deepcopy(initial).cuda()
    all_ids, all_actions = (
        [torch.empty_like(ids) for _ in range(world)],
        [torch.empty_like(actions) for _ in range(world)],
    )
    dist.all_gather(all_ids, ids)
    dist.all_gather(all_actions, actions)
    for worker in range(world):
        losses = golden(all_ids[worker], all_actions[worker], ki=ki, robot=worker % 2 == 0)
        ((losses.ce + losses.flow) / world).backward()
    losses = model(ids, actions, ki=ki, robot=rank % 2 == 0)
    (losses.ce + losses.flow).backward()
    optimizer._ensure_grads_ready()
    golden_grads = dict(golden.named_parameters())
    fqns = {id(module): name for name, module in model.named_modules()}
    tolerance = dict(rtol=0.04, atol=0.003) if bf16 else dict(rtol=1e-4, atol=1e-6)
    for param in optimizer._dedicated_params:
        name = f"{fqns[id(param.module)]}.{param.param_name}".lstrip(".")
        torch.testing.assert_close(param._reduced_grad.float(), golden_grads[name].grad.float(), **tolerance)
    fsdp_ids = {id(param) for param in optimizer._fsdp_params}
    for name, param in model.named_parameters():
        if id(param) in fsdp_ids:
            grad = param.grad.full_tensor()
            torch.testing.assert_close(grad.float(), golden_grads[name].grad.float(), **tolerance)
    expected = torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in golden.parameters()]))
    actual = clip_dmuon_grad_norm(model, 0.05, error_if_nonfinite=True)
    torch.testing.assert_close(actual, expected, **tolerance)
    optimizer.step()
    optimizer.zero_grad()
    snapshot = dmuon.get_model_state_dict(model, cpu_offload=True, rank0_only=False)
    assert all(torch.isfinite(tensor).all() for tensor in snapshot.values())
    if rank == 0:
        print(f"PASS gradients+clip+step {backend}/{coefficients} bf16={bf16} reshard={reshard} ki={ki}", flush=True)


def check_accumulation(initial):
    import dmuon
    from dmuon.optim.newton_schulz import direct_newton_schulz

    rank, world = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(900 + rank)
    ids = torch.randint(64, (4,), device="cuda")
    actions = torch.randn(4, 32, device="cuda")
    all_ids = [torch.empty_like(ids) for _ in range(world)]
    all_actions = [torch.empty_like(actions) for _ in range(world)]
    dist.all_gather(all_ids, ids)
    dist.all_gather(all_actions, actions)
    snapshots = []
    gradient_snapshots = []
    dedicated_names = set()
    for accumulate in (False, True):
        model, optimizer = make_model(initial, "direct", "classic", False, True)
        slices = [slice(0, 2), slice(2, 4)] if accumulate else [slice(None)]
        golden = copy.deepcopy(initial).cuda()
        for worker in range(world):
            for portion in slices:
                losses = golden(all_ids[worker][portion], all_actions[worker][portion], ki=True)
                ((losses.ce + losses.flow) / (world * len(slices))).backward()
        for index, portion in enumerate(slices):
            trainer = SimpleNamespace(
                model=model, args=SimpleNamespace(train=SimpleNamespace(optimizer=OptimizerConfig(type="dmuon")))
            )
            with BaseTrainer.gradient_sync_context(trainer, index, len(slices)):
                losses = model(ids[portion], actions[portion], ki=True)
                ((losses.ce + losses.flow) / len(slices)).backward()
        optimizer._ensure_grads_ready()
        fqns = {id(module): name for name, module in model.named_modules()}
        golden_params = dict(golden.named_parameters())
        gradients = {}
        owner_names = {}
        for param in optimizer._dedicated_params:
            name = f"{fqns[id(param.module)]}.{param.param_name}".lstrip(".")
            owner_names[id(param)] = name
            dedicated_names.add(name)
            torch.testing.assert_close(param._reduced_grad, golden_params[name].grad, rtol=1e-4, atol=1e-6)
            gradients[name] = param._reduced_grad.clone()
        fsdp_ids = {id(param) for param in optimizer._fsdp_params}
        for name, param in model.named_parameters():
            if id(param) in fsdp_ids:
                grad = param.grad.full_tensor()
                torch.testing.assert_close(grad, golden_params[name].grad, rtol=1e-4, atol=1e-6)
                gradients[name] = grad
        gradient_snapshots.append(gradients)
        clip_dmuon_grad_norm(model, 1.0)
        # Official direct NS casts to FP16. Small summation differences can
        # cross rounding boundaries, so validate each update against its own
        # exact gradient instead of loosening the full/micro weight tolerance.
        expected = {}
        for param in optimizer._dedicated_params:
            group = optimizer.param_groups[optimizer._dp_to_muon_group_idx[id(param)]]
            grad = param._reduced_grad
            update = direct_newton_schulz(
                grad.add(grad, alpha=group["momentum"]),
                coefficients=[[3.4445, -4.775, 2.0315]] * 5,
            )
            weight = param._owned_data.clone()
            if group["weight_decay"] > 0:
                weight.mul_(1 - group["lr"] * group["weight_decay"])
            weight.add_(update, alpha=-group["lr"] * 0.2 * max(weight.shape) ** 0.5)
            expected[id(param)] = weight
        optimizer.step()
        for param in optimizer._dedicated_params:
            torch.testing.assert_close(param._owned_data, expected[id(param)], rtol=0, atol=0)
        optimizer.zero_grad()
        snapshots.append(dmuon.get_model_state_dict(model, cpu_offload=True, rank0_only=False))
    for name in gradient_snapshots[0]:
        torch.testing.assert_close(gradient_snapshots[0][name], gradient_snapshots[1][name], rtol=1e-4, atol=1e-6)
    all_dedicated_names = [None] * world
    dist.all_gather_object(all_dedicated_names, dedicated_names)
    dedicated_names = set().union(*all_dedicated_names)
    for name in snapshots[0]:
        if name not in dedicated_names:
            torch.testing.assert_close(snapshots[0][name], snapshots[1][name], rtol=1e-4, atol=1e-6)
    if rank == 0:
        print("PASS accumulation: golden/full/micro gradients and exact official FP16 NS updates", flush=True)


def assert_state_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_state_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_state_equal(left, right)
    else:
        assert actual == expected


def check_checkpoint_continuation(initial, backend, coefficients, bf16, warmup_steps):
    import dmuon

    from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer

    paths = [None]
    if dist.get_rank() == 0:
        root = Path(os.environ.get("DMUON_TEST_OUTPUT", "/tmp/veomni-dmuon-tests"))
        root.mkdir(parents=True, exist_ok=True)
        paths[0] = tempfile.mkdtemp(
            prefix=f"world{dist.get_world_size()}-{backend}-{coefficients}-{warmup_steps}-", dir=root
        )
    dist.broadcast_object_list(paths, src=0)
    path = Path(paths[0])
    model, optimizer = make_model(initial, backend, coefficients, bf16, False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    torch.manual_seed(1100 + dist.get_rank())
    ids = torch.randint(64, (4,), device="cuda")
    actions = torch.randn(4, 32, device="cuda")

    def step(current_model, current_optimizer, current_scheduler):
        losses = current_model(ids, actions, ki=True, robot=dist.get_rank() % 2 == 0)
        loss = losses.ce + losses.flow
        loss.backward()
        clip_dmuon_grad_norm(current_model, 0.1)
        current_optimizer.step()
        current_scheduler.step()
        current_optimizer.zero_grad()
        current_model.scale.add_(0.01)
        return loss.detach().clone()

    for _ in range(warmup_steps):
        step(model, optimizer, scheduler)
    DistributedCheckpointer.save(
        str(path),
        {
            "model": model,
            "optimizer": optimizer,
            "extra_state": {"scheduler": scheduler.state_dict(), "global_steps": warmup_steps},
        },
        save_async=False,
    )
    assert (path / ".metadata").is_file()
    assert (path / "dmuon_optimizer.pt").is_file()
    saved_model = dmuon.get_model_state_dict(model, cpu_offload=True, rank0_only=False)
    saved_optimizer = dmuon.get_optimizer_state_dict(model, optimizer, cpu_offload=True, rank0_only=False)
    losses = [step(model, optimizer, scheduler) for _ in range(3)]
    expected_model = dmuon.get_model_state_dict(model, cpu_offload=True, rank0_only=False)
    expected_optimizer = dmuon.get_optimizer_state_dict(model, optimizer, cpu_offload=True, rank0_only=False)

    resumed_model, resumed_optimizer = make_model(initial, backend, coefficients, bf16, False)
    resumed_scheduler = torch.optim.lr_scheduler.StepLR(resumed_optimizer, step_size=1, gamma=0.9)
    state = DistributedCheckpointer.load(
        str(path), {"model": resumed_model, "optimizer": resumed_optimizer, "extra_state": {}}
    )
    for param_state in resumed_optimizer.state.values():
        for value in param_state.values():
            if isinstance(value, torch.Tensor):
                # Restored moments must not retain the full parameter's storage.
                assert value.untyped_storage().nbytes() == value.numel() * value.element_size()
    assert state["extra_state"]["global_steps"] == warmup_steps
    assert_state_equal(dmuon.get_model_state_dict(resumed_model, cpu_offload=True, rank0_only=False), saved_model)
    assert_state_equal(
        dmuon.get_optimizer_state_dict(resumed_model, resumed_optimizer, cpu_offload=True, rank0_only=False),
        saved_optimizer,
    )
    resumed_scheduler.load_state_dict(state["extra_state"]["scheduler"])
    for expected_loss in losses:
        assert_state_equal(step(resumed_model, resumed_optimizer, resumed_scheduler), expected_loss)
    assert_state_equal(dmuon.get_model_state_dict(resumed_model, cpu_offload=True, rank0_only=False), expected_model)
    assert_state_equal(
        dmuon.get_optimizer_state_dict(resumed_model, resumed_optimizer, cpu_offload=True, rank0_only=False),
        expected_optimizer,
    )
    assert_state_equal(resumed_scheduler.state_dict(), scheduler.state_dict())
    if dist.get_rank() == 0:
        print(
            f"PASS synchronous DCP + 3 exact continuation steps {backend}/{coefficients} warmup={warmup_steps}",
            flush=True,
        )


def check_trainable_only(initial):
    import dmuon

    from veomni.checkpoint.dcp_checkpointer import ModelState

    initial = copy.deepcopy(initial)
    initial.embed_tokens.weight.requires_grad_(False)
    model, _ = make_model(initial, "direct", "classic", False, True)
    assert dmuon.get_dedicated_params(model)
    saved = ModelState(model, trainable_only=True).state_dict()
    assert "embed_tokens.weight" not in saved
    initial.embed_tokens.weight.add_(1)
    with torch.no_grad():
        initial.lm_head.weight.add_(1)
        initial.layers[0].native.weight.add_(1)
        initial.scale.add_(1)
    resumed_model, _ = make_model(initial, "direct", "classic", False, True)
    ModelState(resumed_model, trainable_only=True).load_state_dict(saved)
    actual = ModelState(resumed_model).state_dict()
    assert_state_equal(actual.pop("embed_tokens.weight"), initial.embed_tokens.weight)
    assert_state_equal(actual, saved)
    if dist.get_rank() == 0:
        print("PASS trainable-only restore: dedicated, ordinary FSDP, frozen embedding and buffer", flush=True)


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    replicas = int(os.environ.get("DMUON_TEST_REPLICAS", "1"))
    world = dist.get_world_size()
    init_parallel_state(dp_size=world, dp_replicate_size=replicas, dp_shard_size=world // replicas)
    torch.manual_seed(42)
    initial = BranchedModel()
    check_graph_replay_with_pinning(initial)
    for backend, coefficients, bf16, reshard, ki in (
        ("direct", "classic", False, True, False),
        ("direct", "classic", False, True, True),
        ("gram", "polar_express", True, False, True),
    ):
        check_gradients(initial, backend, coefficients, bf16, reshard, ki)
    check_accumulation(initial)
    # Include uneven embedding shards and, with eight ranks, empty bias shards.
    torch.manual_seed(43)
    checkpoint_initial = copy.deepcopy(initial)
    checkpoint_initial.embed_tokens = nn.Embedding(67, 32)
    checkpoint_initial.lm_head = nn.Linear(32, 3)
    check_checkpoint_continuation(checkpoint_initial, "direct", "classic", False, 3)
    check_checkpoint_continuation(checkpoint_initial, "gram", "polar_express", True, 3)
    check_checkpoint_continuation(checkpoint_initial, "direct", "classic", False, 0)
    check_trainable_only(checkpoint_initial)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
