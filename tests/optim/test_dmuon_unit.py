"""Adapter input tests; distributed tests verify official DMuon numerics."""

import os
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from veomni.optim.dmuon import build_dmuon


@pytest.mark.parametrize("enabled, capture_ok", [(False, True), (True, True), (True, False)])
def test_ns_warmup_respects_opt_out_and_rejects_failed_capture(monkeypatch, enabled, capture_ok):
    from veomni.optim.dmuon import warmup_dmuon_graphs

    monkeypatch.setitem(sys.modules, "dmuon.optim", SimpleNamespace(ns_graph=SimpleNamespace(enabled=lambda: enabled)))
    monkeypatch.setattr("veomni.optim.dmuon.synchronize", lambda: None)
    params = [SimpleNamespace(_owned_data=torch.ones(shape)) for shape in [(4, 4), (4, 4), (4, 8), (8, 8)]]
    calls = []
    cache = SimpleNamespace(_graphs={})

    def compute(value, *, is_tp):
        assert not is_tp
        calls.append(tuple(value.shape))
        key = (tuple(value.shape), value.dtype, value.device.index)
        cache._graphs[key] = SimpleNamespace(graph=object() if capture_ok else None)

    optimizer = SimpleNamespace(
        _dedicated_params=params,
        _dp_to_muon_group_idx={id(p): 0 for p in params[:3]},
        _ns_compute=compute,
        _ns_graph_cache=cache,
    )
    if enabled and not capture_ok:
        with pytest.raises(RuntimeError, match="DMuon NS graph warmup failed"):
            warmup_dmuon_graphs(optimizer)
    else:
        warmup_dmuon_graphs(optimizer)
    expected = [] if not enabled else [(4, 4), (4, 4)]
    if enabled and capture_ok:
        expected += [(4, 8), (4, 8)]
    assert calls == expected
    assert all(torch.equal(p._owned_data, torch.ones_like(p._owned_data)) for p in params)


@pytest.mark.parametrize("kind", ["adamw", "muon", "dmuon"])
def test_trainer_forwards_dedication_and_ns_options_only_for_dmuon(monkeypatch, kind):
    from veomni.arguments.arguments_types import OptimizerConfig
    from veomni.trainer import base

    config = OptimizerConfig(type=kind, dmuon_ns_backend="direct", dmuon_ns_coefficients="classic")
    options = base._collect_muon_kwargs(config)
    if kind == "dmuon":
        assert options["dmuon_ns_backend"] == "direct"
        assert options["dmuon_ns_coefficients"] == "classic"
    else:
        assert not any(key.startswith("dmuon_") for key in options)
    args = SimpleNamespace(
        model=SimpleNamespace(model_path="unused", lora_config={}, basic_modules=[]),
        train=SimpleNamespace(
            optimizer=config,
            init_device="cpu",
            broadcast_model_weights_from_rank0=False,
            gradient_checkpointing=SimpleNamespace(enable=False, enable_reentrant=False),
            accelerator=SimpleNamespace(
                fsdp_config=SimpleNamespace(
                    reshard_after_forward=True,
                    mixed_precision=object(),
                    forward_prefetch=False,
                    offload=False,
                    max_load_broadcast_size=1024,
                )
            ),
        ),
    )
    calls = {}

    def parallelize(model, **kwargs):
        calls.update(kwargs)
        return model

    monkeypatch.setattr(base, "build_parallelize_model", parallelize)
    trainer = SimpleNamespace(args=args, model=nn.Linear(4, 4))
    base.BaseTrainer._build_parallelized_model(trainer)
    assert trainer.model.training
    if kind == "dmuon":
        assert calls["optimizer_config"] is config
    else:
        assert "optimizer_config" not in calls


@pytest.mark.parametrize("broadcast", [None, False, True])
def test_dmuon_rejects_rank0_broadcast_before_model_setup(monkeypatch, broadcast):
    from veomni.distributed import torch_parallelize

    optimizer_config = object()
    monkeypatch.setattr(
        torch_parallelize,
        "get_parallel_state",
        lambda: SimpleNamespace(
            dp_mode="fsdp2", fsdp_enabled=True, tp_enabled=False, sp_enabled=False, any_extra_parallel_enabled=False
        ),
    )
    model = nn.Linear(4, 4, dtype=torch.bfloat16)
    options = {
        "weights_path": "checkpoint",
        "mixed_precision": SimpleNamespace(enable=True),
        "enable_gradient_checkpointing": False,
        "optimizer_config": optimizer_config,
    }
    if broadcast is not None:
        options["broadcast_model_weights_from_rank0"] = broadcast
    if broadcast:
        with pytest.raises(ValueError, match="DMuon integration does not support broadcast_model_weights_from_rank0"):
            torch_parallelize.build_parallelize_model(model, **options)
        assert model.weight.dtype is torch.bfloat16
    else:
        calls = {}

        def parallelize_model_fsdp2(**kwargs):
            calls.update(kwargs)
            return kwargs["model"]

        monkeypatch.setattr(torch_parallelize, "parallelize_model_fsdp2", parallelize_model_fsdp2)
        assert torch_parallelize.build_parallelize_model(model, **options) is model
        assert model.weight.dtype is torch.float32
        assert calls["optimizer_config"] is optimizer_config
        assert not calls.get("broadcast_model_weights_from_rank0", False)


@pytest.mark.parametrize("kind", ["adamw", "muon", "dmuon"])
@pytest.mark.parametrize("micro_steps", [1, 3])
def test_trainer_uses_scoped_official_no_sync(monkeypatch, kind, micro_steps):
    from veomni.trainer.base import BaseTrainer

    active = []
    calls = []
    model = object()

    @contextmanager
    def no_sync(actual_model):
        assert actual_model is model
        calls.append(actual_model)
        active.append(True)
        try:
            yield
        finally:
            active.pop()

    monkeypatch.setitem(sys.modules, "dmuon", SimpleNamespace(no_sync=no_sync))
    trainer = SimpleNamespace(
        args=SimpleNamespace(train=SimpleNamespace(optimizer=SimpleNamespace(type=kind))), model=model
    )
    for micro_step in range(micro_steps):
        with BaseTrainer.gradient_sync_context(trainer, micro_step, micro_steps):
            assert bool(active) == (kind == "dmuon" and micro_step < micro_steps - 1)
        assert not active
    assert len(calls) == (micro_steps - 1 if kind == "dmuon" else 0)
    if kind == "dmuon" and micro_steps > 1:
        with pytest.raises(RuntimeError, match="backward failed"):
            with BaseTrainer.gradient_sync_context(trainer, 0, micro_steps):
                raise RuntimeError("backward failed")
        assert not active


@pytest.fixture
def construction(monkeypatch):
    calls = {}
    monkeypatch.setattr("veomni.optim.dmuon.warmup_dmuon_graphs", lambda optimizer: calls.update(warmup=optimizer))

    def newton_schulz(**kwargs):
        calls["ns"] = kwargs
        return object()

    def muon(model, **kwargs):
        calls["optimizer"] = kwargs
        return SimpleNamespace()

    monkeypatch.setitem(sys.modules, "dmuon", SimpleNamespace(NewtonSchulz=newton_schulz, Muon=muon))
    model = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4), nn.Linear(4, 4))
    model[2].requires_grad_(False)
    model._dedicated_comm_ctx = object()

    def build(**overrides):
        arguments = dict(
            model=model,
            lr=1e-3,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.1,
            param_groups=None,
            no_decay_modules=["LayerNorm"],
            no_decay_params=["bias"],
            muon_kwargs={"lr": 0.02, "weight_decay": 0.2},
        )
        arguments.update(overrides)
        return build_dmuon(**arguments)

    return model, calls, build


def test_generator_groups_keep_decay_exclusions_frozen_parameters_and_lr_ratio(construction):
    model, calls, build = construction
    semantic_groups = ({"params": (p for p in model.parameters()), "lr": 2e-4, "name": "vision"} for _ in range(1))
    optimizer = build(param_groups=semantic_groups)
    groups = calls["optimizer"]["param_groups"]
    assert len(groups) == 2
    assert {id(p) for group in groups for p in group["params"]} == {
        id(p) for p in model.parameters() if p.requires_grad
    }
    assert groups[0]["params"][0] is model[0].weight
    assert all(group["name"] == "vision" for group in groups)
    assert all(group["muon_lr"] == pytest.approx(0.004) for group in groups)
    assert all(group["adamw_lr"] == 2e-4 for group in groups)
    assert [group["muon_weight_decay"] for group in groups] == [0.2, 0.0]
    assert [group["adamw_weight_decay"] for group in groups] == [0.1, 0.0]
    assert model._veomni_dmuon_optimizer is optimizer
    assert calls["warmup"] is optimizer


def test_explicit_zero_decay_and_zero_lr_are_preserved(construction):
    model, calls, build = construction
    build(param_groups=[{"params": model.parameters(), "lr": 0, "weight_decay": 0}])
    assert all(group["muon_lr"] == group["adamw_lr"] == 0 for group in calls["optimizer"]["param_groups"])
    assert all(
        group["muon_weight_decay"] == group["adamw_weight_decay"] == 0 for group in calls["optimizer"]["param_groups"]
    )


@pytest.mark.parametrize("groups", [[], iter(())])
def test_explicit_empty_groups_do_not_fall_back_to_all_parameters(construction, groups):
    _, calls, build = construction
    with pytest.raises(ValueError, match="must not be empty"):
        build(param_groups=groups)
    assert "optimizer" not in calls


@pytest.mark.parametrize("lr", [0, -1, float("nan"), float("inf")])
def test_invalid_base_lr_fails_before_upstream_construction(construction, lr):
    _, calls, build = construction
    with pytest.raises(ValueError, match="positive and finite"):
        build(lr=lr)
    assert "optimizer" not in calls


@pytest.mark.parametrize("lr", [-1, float("nan"), float("inf")])
def test_invalid_semantic_lr(construction, lr):
    model, _, build = construction
    with pytest.raises(ValueError, match="learning rates must be finite and nonnegative"):
        build(param_groups=[{"params": model.parameters(), "lr": lr}])


@pytest.mark.parametrize(
    "options, message",
    [
        ({"lr": float("nan")}, "learning rate"),
        ({"lr": -1}, "learning rate"),
        ({"weight_decay": float("inf")}, "weight decay"),
        ({"weight_decay": -1}, "weight decay"),
        ({"momentum": float("nan")}, "momentum"),
        ({"momentum": 1.1}, "momentum"),
        ({"nesterov": "false"}, "boolean"),
        ({"ns_steps": 0}, "NS steps"),
        ({"ns_steps": 100}, "NS steps"),
        ({"ns_steps": 5.0}, "NS steps"),
        ({"ns_steps": True}, "NS steps"),
        ({"ns_coefficients": [1, 2]}, "three finite"),
        ({"ns_coefficients": [1, 2, float("nan")]}, "three finite"),
        ({"eps": 1e-8}, "normalization epsilon"),
        ({"adjust_lr_fn": "original"}, "match_rms_adamw"),
        ({"dmuon_ns_backend": "unknown"}, "Unknown DMuon NS"),
        ({"dmuon_ns_coefficients": "unknown"}, "Unknown DMuon NS"),
        ({"ns_steps": 4}, "five NS iterations"),
        ({"momentuum": 0.9}, "Unknown DMuon options"),
    ],
)
def test_invalid_muon_options_are_rejected(construction, options, message):
    _, calls, build = construction
    with pytest.raises(ValueError, match=message):
        build(muon_kwargs=options)
    assert "optimizer" not in calls


def test_custom_classic_coefficients_and_steps_reach_upstream(construction):
    _, calls, build = construction
    build(
        muon_kwargs={
            "dmuon_ns_backend": "direct",
            "dmuon_ns_coefficients": "classic",
            "ns_steps": 3,
            "ns_coefficients": (3.4, -4.7, 2.0),
        }
    )
    assert calls["ns"] == {"backend": "direct", "coefficients": [[3.4, -4.7, 2.0]] * 3, "restart_iterations": []}
    assert calls["optimizer"]["ns_steps"] == 3


def test_foreign_trainable_parameter_has_clear_error(construction):
    _, _, build = construction
    with pytest.raises(ValueError, match="outside the model"):
        build(param_groups=[{"params": [nn.Parameter(torch.randn(4, 4))]}])


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Official DMuon requires CUDA")
@pytest.mark.parametrize(
    "style, checkpoints",
    [
        ("constant", {0: 0.0, 1: 0.5, 2: 1.0, 5: 1.0, 8: 1.0}),
        ("linear", {0: 0.0, 1: 0.5, 2: 1.0, 5: 0.5, 8: 1e-4}),
        ("cosine", {0: 0.0, 1: 0.5, 2: 1.0, 5: 0.50005, 8: 1e-4}),
    ],
)
def test_real_dmuon_scheduler_preserves_semantic_and_route_lr_ratios(monkeypatch, tmp_path, style, checkpoints):
    """Exercise VeOmni scheduling on upstream DMuon's actual lowered groups."""
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    from veomni.optim import build_lr_scheduler

    dmuon = pytest.importorskip("dmuon")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        pytest.skip("This single-rank test cannot run under torchrun")
    if dist.is_initialized():
        pytest.skip("This single-rank test owns its process group")
    monkeypatch.setenv("DMUON_NS_CUDA_GRAPH", "0")
    monkeypatch.setenv("DMUON_NS_KERNEL", "cublas")
    dist.init_process_group("nccl", init_method=(tmp_path / "init").as_uri(), rank=0, world_size=1)
    try:
        mesh = init_device_mesh("cuda", (1,), mesh_dim_names=("dp_shard",))
        model = nn.Sequential()
        model.add_module("vision", nn.Linear(8, 8))
        model.add_module("backbone", nn.Linear(8, 8))
        model.cuda()
        dmuon.dedicate_params(model, mesh, predicate=lambda name, param: param.ndim == 2, compute_dtype=torch.float32)
        fully_shard(model, mesh=mesh)
        optimizer = build_dmuon(
            model,
            lr=1e-3,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.1,
            param_groups=[
                {"params": model.vision.parameters(), "lr": 2e-4, "name": "vision"},
                {"params": model.backbone.parameters(), "lr": 1e-3, "name": "backbone"},
            ],
            no_decay_modules=[],
            no_decay_params=["bias"],
            muon_kwargs={"lr": 0.02, "dmuon_ns_backend": "direct", "dmuon_ns_coefficients": "classic"},
        )
        expected_rates = {
            ("vision", "muon"): 0.004,
            ("vision", "adamw"): 2e-4,
            ("backbone", "muon"): 0.02,
            ("backbone", "adamw"): 1e-3,
        }
        assert {(g["semantic_group_name"], g["subgroup_type"]) for g in optimizer.param_groups} == set(expected_rates)
        for group in optimizer.param_groups:
            assert group["lr"] == pytest.approx(expected_rates[group["semantic_group_name"], group["subgroup_type"]])
        scheduler = build_lr_scheduler(
            optimizer, train_steps=8, lr=1e-3, lr_decay_style=style, lr_warmup_ratio=0.25, lr_min=1e-7
        )
        inputs = torch.ones(2, 8, device="cuda")
        for step in range(9):
            if step in checkpoints:
                for group in optimizer.param_groups:
                    base = expected_rates[group["semantic_group_name"], group["subgroup_type"]]
                    assert group["lr"] == pytest.approx(base * checkpoints[step])
            if step < 8:
                optimizer.zero_grad()
                model(inputs).square().mean().backward()
                optimizer.step()
                scheduler.step()
        torch.cuda.synchronize()
    finally:
        dist.destroy_process_group()
