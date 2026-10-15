# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Synchronous optimizer DCP restore must preserve the next training steps."""

import copy

import pytest
import torch
import torch.distributed as dist
from torch import nn

from veomni.checkpoint import dcp_checkpointer
from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer, OptimizerState
from veomni.optim import build_lr_scheduler, build_optimizer


class _MixedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(8, 8)
        self.conv = nn.Conv1d(8, 8, 3, padding=1, groups=8)
        self.norm = nn.LayerNorm(8)
        self.unused = nn.Linear(8, 8)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, use_unused):
        if use_unused == "adamw_only":
            return self.dropout(self.norm.bias * x)
        if use_unused == "muon_only":
            # Keep AdamW completely inactive after convolution routing changes.
            return self.dropout(nn.functional.linear(x, self.hidden.weight))
        x = self.hidden(x) + self.conv(x.transpose(1, 2)).transpose(1, 2)
        if use_unused:
            x = x + self.unused(x)
        return self.dropout(self.norm(x))


def _build(optimizer_type):
    model = _MixedModel()
    optimizer = build_optimizer(
        model,
        lr=1e-3,
        weight_decay=0.01,
        optimizer_type=optimizer_type,
        muon_kwargs={"lr": 5e-3, "adjust_lr_fn": "match_rms_adamw"},
    )
    scheduler = build_lr_scheduler(optimizer, train_steps=8, lr=1e-3, lr_decay_style="cosine", lr_warmup_ratio=0.125)
    return model, optimizer, scheduler


def _assert_equal(actual, expected):
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_equal(actual[key], expected[key])
    elif isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            _assert_equal(a, b)
    else:
        assert actual == expected


def _step(model, optimizer, scheduler, use_unused):
    optimizer.zero_grad()
    x = torch.randn(4, 5, 8)
    target = torch.randn_like(x)
    loss = (model(x, use_unused) - target).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    return {
        "loss": loss.detach().clone(),
        "model": copy.deepcopy(model.state_dict()),
        "optimizer": {name: copy.deepcopy(child.state_dict()) for name, child in _children(optimizer).items()},
        "scheduler": copy.deepcopy(scheduler.state_dict()),
        "rng": torch.get_rng_state().clone(),
    }


def _children(optimizer):
    return optimizer.optimizers_dict if hasattr(optimizer, "_is_multi_optimizer") else {"adamw": optimizer}


@pytest.fixture
def cpu_process_group(tmp_path, monkeypatch):
    assert not dist.is_initialized()
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1)
    # CPU tests exercise real DCP I/O without initializing an accelerator.
    monkeypatch.setattr(dcp_checkpointer, "empty_cache", lambda: None)
    monkeypatch.setattr(dcp_checkpointer, "synchronize", lambda: None)
    try:
        yield
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "optimizer_type,warmup_path",
    [
        ("adamw", False),
        ("adamw", "no_steps"),
        ("muon", False),
        ("muon", "adamw_only"),
        ("muon", "muon_only"),
        ("muon", "no_steps"),
    ],
)
def test_dcp_five_step_continuation(tmp_path, cpu_process_group, optimizer_type, warmup_path):
    torch.manual_seed(41)
    model, optimizer, scheduler = _build(optimizer_type)
    warmup_steps = 0 if warmup_path == "no_steps" else 3
    for _ in range(warmup_steps):
        _step(model, optimizer, scheduler, use_unused=warmup_path)
    if not warmup_steps:
        # Saving an empty optimizer must preserve already-computed gradients.
        model(torch.randn(4, 5, 8), False).square().mean().backward()
    assert model.unused.weight.grad is None
    assert model.unused.bias.grad is None

    children = _children(optimizer)
    if optimizer_type == "muon":
        if warmup_path == "muon_only":
            assert children["muon"].state
            assert not children["adamw"].state
        elif warmup_path == "adamw_only":
            assert children["adamw"].state
            assert not children["muon"].state
        elif warmup_path == "no_steps":
            assert all(not child.state for child in children.values())
    optimizer_before_save = {name: copy.deepcopy(child.state_dict()) for name, child in children.items()}
    model_before_save = copy.deepcopy(model.state_dict())
    grads_before_save = {
        name: None if param.grad is None else param.grad.clone() for name, param in model.named_parameters()
    }
    rng_before_save = torch.get_rng_state().clone()
    state = {
        "model": model,
        "optimizer": optimizer,
        "extra_state": {
            "scheduler": scheduler.state_dict(),
            "rng": torch.get_rng_state(),
            "global_steps": warmup_steps,
        },
    }
    checkpoint = str(tmp_path / "checkpoint")
    DistributedCheckpointer.save(checkpoint, state, save_async=False)
    for name, child in children.items():
        _assert_equal(child.state_dict(), optimizer_before_save[name])
    _assert_equal(model.state_dict(), model_before_save)
    _assert_equal({name: param.grad for name, param in model.named_parameters()}, grads_before_save)
    _assert_equal(torch.get_rng_state(), rng_before_save)

    saved_state = OptimizerState(model, optimizer, fill_missing_optimizer_states=True).state_dict()
    _assert_equal(model.state_dict(), model_before_save)
    _assert_equal({name: param.grad for name, param in model.named_parameters()}, grads_before_save)
    _assert_equal(torch.get_rng_state(), rng_before_save)
    if optimizer_type == "muon":
        assert "state.unused.weight.momentum_buffer" in saved_state
        assert "state.unused.weight.exp_avg" not in saved_state
        assert "state.unused.bias.exp_avg" in saved_state
        assert "state.unused.bias.momentum_buffer" not in saved_state
        assert torch.count_nonzero(saved_state["state.unused.weight.momentum_buffer"]) == 0
        assert saved_state["state.unused.bias.step"] == 0
    else:
        assert saved_state["state"]["unused.weight"]["step"] == 0

    # First use of both missing-state parameters happens after checkpointing.
    expected = [_step(model, optimizer, scheduler, use_unused=True) for _ in range(5)]
    resumed_model, resumed_optimizer, resumed_scheduler = _build(optimizer_type)
    resumed = DistributedCheckpointer.load(
        checkpoint, {"model": resumed_model, "optimizer": resumed_optimizer, "extra_state": {}}
    )
    resumed_scheduler.load_state_dict(resumed["extra_state"]["scheduler"])
    torch.set_rng_state(resumed["extra_state"]["rng"])
    assert resumed["extra_state"]["global_steps"] == warmup_steps
    for reference in expected:
        _assert_equal(_step(resumed_model, resumed_optimizer, resumed_scheduler, use_unused=True), reference)
