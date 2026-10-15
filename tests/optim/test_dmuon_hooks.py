"""CPU autograd checks for DMuon multi-branch boundaries and checkpointing."""

import copy
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils._pytree import tree_flatten
from torch.utils.checkpoint import checkpoint
from transformers.utils import ModelOutput


pytest.importorskip("dmuon")
from dmuon._core.state import DedicatedState  # noqa: E402

from veomni.optim.dmuon_hooks import adapt_multibranch_hooks  # noqa: E402


@dataclass
class _Output(ModelOutput):
    loss: torch.Tensor = None
    flow: object = None


class _Group:
    def __init__(self, model, events):
        self.params = ()
        self.model = model
        self.events = events
        self._post_backward_fired = False
        self.grad_snapshots = []

    def _pre_forward_wait(self):
        pass

    def unshard(self, **kwargs):
        self.events.append("unshard")

    def wait_for_unshard(self):
        self.events.append("wait_unshard")

    def _backward_prefetch(self):
        self.events.append("prefetch")

    def _record_post_forward(self):
        pass

    def reshard(self):
        self.events.append("reshard")

    def reduce_grads(self):
        self.events.append("reduce")
        self.grad_snapshots.append([None if p.grad is None else p.grad.clone() for p in self.model.parameters()])

    def wait_for_reduce(self):
        self.events.append("wait_reduce")


def _attach(model, boundary=None):
    events = []
    comm = SimpleNamespace(
        all_states=[],
        last_reduced_group=None,
        post_backward_final_callback_queued=False,
        reset_post_forward_order=lambda: None,
    )
    boundary = model if boundary is None else boundary
    group = _Group(boundary, events)
    state = DedicatedState(boundary, group, comm)
    boundary._dedicated_state = state
    adapt_multibranch_hooks(model)
    return state, group, events


def test_nested_model_output_and_aliases_preserve_type():
    state, _, _ = _attach(nn.Linear(2, 2))
    x, y = torch.randn(2, requires_grad=True), torch.randn(2, requires_grad=True)
    constant = torch.ones(2)
    output = _Output(loss=x, flow={"nested": [y, (None, x, constant)]})
    wrapped = state._register_pre_backward(output)
    assert type(wrapped) is _Output
    assert isinstance(wrapped.flow, dict)
    assert isinstance(wrapped.flow["nested"], list)
    assert isinstance(wrapped.flow["nested"][1], tuple)
    assert wrapped.loss is wrapped.flow["nested"][1][1]
    assert wrapped.loss.grad_fn is wrapped.flow["nested"][0].grad_fn
    assert wrapped.flow["nested"][1][2] is constant

    args, kwargs = state._register_post_backward(({"prefix": [x]},), {"suffix": (y,)})
    assert args[0]["prefix"][0].grad_fn is kwargs["suffix"][0].grad_fn


def test_no_grad_preserves_identity_and_adaptation_is_instance_local():
    original_pre = DedicatedState._register_pre_backward
    original_post = DedicatedState._register_post_backward
    model = nn.Linear(2, 2)
    state, _, _ = _attach(model)
    untouched_model = nn.Linear(2, 2)
    comm = SimpleNamespace(all_states=[])
    untouched = DedicatedState(untouched_model, _Group(untouched_model, []), comm)
    tree = {"ids": torch.ones(2, dtype=torch.int64), "none": None}
    assert state._register_pre_backward(tree) is tree
    args, kwargs = (tree,), {"mask": torch.ones(2)}
    returned_args, returned_kwargs = state._register_post_backward(args, kwargs)
    assert returned_args is args and returned_kwargs is kwargs
    assert DedicatedState._register_pre_backward is original_pre
    assert DedicatedState._register_post_backward is original_post
    assert untouched._register_pre_backward.__func__ is original_pre
    assert state._run_post_backward.__func__ is DedicatedState._run_post_backward


class _TwoBranch(nn.Module):
    def __init__(self, detach_prefix=False):
        super().__init__()
        self.prefix = nn.Linear(3, 3, bias=False)
        self.suffix = nn.Linear(3, 3, bias=False)
        self.detach_prefix = detach_prefix

    def forward(self, inputs, *, suffix):
        hidden = self.prefix(inputs["prefix"])
        action = self.suffix(suffix)
        action = action + (hidden.detach() if self.detach_prefix else hidden)
        return _Output(loss=hidden.square().sum(), flow={"loss": action.square().sum()})


@pytest.mark.parametrize("detach_prefix", [False, True])
@pytest.mark.parametrize("use_checkpoint", [False, True])
@pytest.mark.parametrize("branch", ["both", "prefix", "suffix"])
def test_branch_gradients_and_reduce_order(detach_prefix, use_checkpoint, branch):
    torch.manual_seed(11)
    model = _TwoBranch(detach_prefix)
    reference = copy.deepcopy(model)
    state, group, events = _attach(model)
    for name, param in model.named_parameters():
        param.register_hook(lambda grad, name=name: events.append(f"grad:{name}"))
    prefix = torch.randn(2, 3, requires_grad=True)
    suffix = torch.randn(2, 3, requires_grad=True)
    reference_prefix = prefix.detach().clone().requires_grad_()
    reference_suffix = suffix.detach().clone().requires_grad_()

    def compute(module, p, s):
        return module({"prefix": p}, suffix=s)

    actual = (
        checkpoint(compute, model, prefix, suffix, use_reentrant=False)
        if use_checkpoint
        else compute(model, prefix, suffix)
    )
    expected = compute(reference, reference_prefix, reference_suffix)
    events.clear()

    def loss(output):
        return (output.loss if branch != "suffix" else 0) + (output.flow["loss"] if branch != "prefix" else 0)

    loss(expected).backward()
    loss(actual).backward()
    for actual_param, reference_param in zip(model.parameters(), reference.parameters()):
        # A shared output node may materialize a zero gradient for the unused
        # branch; its mathematical update remains identical.
        expected_grad = torch.zeros_like(actual_param) if reference_param.grad is None else reference_param.grad
        torch.testing.assert_close(actual_param.grad, expected_grad)
    for actual_input, reference_input in [(prefix, reference_prefix), (suffix, reference_suffix)]:
        expected_grad = torch.zeros_like(actual_input) if reference_input.grad is None else reference_input.grad
        torch.testing.assert_close(actual_input.grad, expected_grad)
    assert events.index("wait_unshard") < min(i for i, event in enumerate(events) if event.startswith("grad:"))
    assert events.count("reduce") == 1
    assert events.index("reduce") > max(i for i, event in enumerate(events) if event.startswith("grad:"))
    for snapshot, param in zip(group.grad_snapshots[0], model.parameters()):
        torch.testing.assert_close(snapshot, param.grad)
    assert not state.comm_ctx.post_backward_final_callback_queued


def test_root_callback_handles_inputs_without_grad():
    model = _TwoBranch(detach_prefix=True)
    _, group, events = _attach(model)
    output = model({"prefix": torch.randn(2, 3)}, suffix=torch.randn(2, 3))
    leaves, _ = tree_flatten(output)
    sum(leaf for leaf in leaves if isinstance(leaf, torch.Tensor)).backward()
    assert events.count("reduce") == 1
    assert all(grad is not None for grad in group.grad_snapshots[0])


class _ReusedBranch(nn.Module):
    def __init__(self, use_checkpoint):
        super().__init__()
        self.visual = nn.Linear(3, 3, bias=False)
        self.use_checkpoint = use_checkpoint

    def forward(self, first, second):
        def call(inputs):
            return checkpoint(self.visual, inputs, use_reentrant=False) if self.use_checkpoint else self.visual(inputs)

        return _Output(loss=call(first).square().sum(), flow=call(second).square().sum())


@pytest.mark.parametrize("use_checkpoint", [False, True])
@pytest.mark.parametrize("input_grad", [False, True])
@pytest.mark.parametrize("branch", ["both", "first", "second", "dummy"])
def test_reused_module_reduces_only_after_every_call(use_checkpoint, input_grad, branch):
    torch.manual_seed(19)
    model = _ReusedBranch(use_checkpoint)
    reference = copy.deepcopy(model)
    state, group, events = _attach(model, model.visual)
    model.visual.weight.register_hook(lambda grad: events.append("grad:visual"))
    for iteration in range(2):
        model.zero_grad()
        reference.zero_grad()
        first = torch.randn(2, 3, requires_grad=input_grad)
        second = torch.randn(2, 3, requires_grad=input_grad)
        reference_first = first.detach().clone().requires_grad_(input_grad)
        reference_second = second.detach().clone().requires_grad_(input_grad)
        actual = model(first, second)
        expected = reference(reference_first, reference_second)
        events.clear()
        group.grad_snapshots.clear()

        def loss(output):
            return (output.loss if branch != "second" else 0) + (
                output.flow * (0 if branch == "dummy" else 1) if branch != "first" else 0
            )

        loss(expected).backward()
        loss(actual).backward()
        torch.testing.assert_close(model.visual.weight.grad, reference.visual.weight.grad)
        assert events.count("reduce") == 1, (iteration, events)
        assert events.index("reduce") > max(i for i, event in enumerate(events) if event.startswith("grad:"))
        torch.testing.assert_close(group.grad_snapshots[0][0], reference.visual.weight.grad)
        assert not state.comm_ctx.post_backward_final_callback_queued
