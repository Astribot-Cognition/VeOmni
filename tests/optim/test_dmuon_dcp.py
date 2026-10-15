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

"""Test DCP routing with a mocked DMuon API and real synchronous checkpoint I/O."""

import copy
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch import nn

from veomni.checkpoint import dcp_checkpointer
from veomni.checkpoint.dcp_checkpointer import (
    DistributedCheckpointer,
    ModelState,
    _dmuon_optimizer_layout,
    _validate_dmuon_optimizer_state,
)


class _BufferedModel(nn.Module):
    def __init__(self, dedicated=True):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.linear.register_buffer("offset", torch.randn(4))
        self.register_buffer("counter", torch.tensor(0))
        self.register_buffer("transient", torch.tensor(123), persistent=False)
        if dedicated:
            self._dedicated_comm_ctx = object()

    def forward(self, x):
        self.counter.add_(1)
        return self.linear(x) + self.linear.offset


@pytest.fixture
def dmuon_api(tmp_path, monkeypatch):
    assert not dist.is_initialized()
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1)
    calls = []

    def get_model(model, *, cpu_offload, rank0_only):
        calls.append(("model", cpu_offload, rank0_only))
        return {name: param.detach().cpu().clone() for name, param in model.named_parameters()}

    def set_model(model, state):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in state:
                    param.copy_(state[name])

    def get_optimizer(model, optimizer, *, cpu_offload, rank0_only):
        calls.append(("optimizer", cpu_offload, rank0_only))
        return {
            "fsdp": {
                name: copy.deepcopy(optimizer.state[param])
                for name, param in model.named_parameters()
                if param in optimizer.state
            },
            "dedicated": {},
            "dedicated_adamw": {},
            "param_groups": [
                {key: copy.deepcopy(value) for key, value in group.items() if key != "params"}
                for group in optimizer.param_groups
            ],
        }

    def set_optimizer(model, optimizer, state):
        for name, param in model.named_parameters():
            if name in state["fsdp"]:
                entry = copy.deepcopy(state["fsdp"][name])
                assert isinstance(entry["step"], int)
                entry["step"] = torch.tensor(float(entry["step"]))
                optimizer.state[param] = entry
        for current, saved in zip(optimizer.param_groups, state["param_groups"]):
            current.update(saved)

    monkeypatch.setitem(
        sys.modules,
        "dmuon",
        SimpleNamespace(
            get_model_state_dict=get_model,
            set_model_state_dict=set_model,
            get_dedicated_params=lambda model: [],
            get_optimizer_state_dict=get_optimizer,
            set_optimizer_state_dict=set_optimizer,
        ),
    )
    monkeypatch.setattr(dcp_checkpointer, "empty_cache", lambda: None)
    monkeypatch.setattr(dcp_checkpointer, "synchronize", lambda: None)
    try:
        yield calls
    finally:
        dist.destroy_process_group()


def _build(dedicated=True):
    model = _BufferedModel(dedicated)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    if dedicated:
        optimizer._veomni_dmuon = True
        optimizer._fsdp_params = list(model.parameters())
        optimizer._dp_to_muon_group_idx = {}
        optimizer._dp_to_adamw_group_idx = {}
        optimizer._adamw_param_to_group_idx = {id(param): 0 for param in model.parameters()}
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    return model, optimizer, scheduler


def _step(model, optimizer, scheduler):
    optimizer.zero_grad()
    loss = model(torch.randn(2, 4)).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss.detach().clone()


def test_dmuon_dcp_continuation_preserves_buffers_optimizer_and_extra_state(tmp_path, dmuon_api):
    torch.manual_seed(17)
    model, optimizer, scheduler = _build()
    for _ in range(3):
        _step(model, optimizer, scheduler)
    checkpoint = str(tmp_path / "checkpoint")
    DistributedCheckpointer.save(
        checkpoint,
        {
            "model": model,
            "optimizer": optimizer,
            "extra_state": {"rng": torch.get_rng_state(), "scheduler": scheduler.state_dict(), "global_steps": 3},
        },
    )
    assert (tmp_path / "checkpoint" / "dmuon_optimizer.pt").is_file()
    metadata = dcp_checkpointer.FileSystemReader(checkpoint).read_metadata()
    assert "model.linear.offset" in metadata.state_dict_metadata
    assert "model.counter" in metadata.state_dict_metadata
    assert "model.transient" not in metadata.state_dict_metadata
    assert not any(name.startswith("optimizer.") for name in metadata.state_dict_metadata)
    expected_loss = _step(model, optimizer, scheduler)
    expected_model = copy.deepcopy(model.state_dict())
    expected_rng = torch.get_rng_state().clone()

    resumed_model, resumed_optimizer, resumed_scheduler = _build()
    resumed_model.transient.fill_(456)
    state = DistributedCheckpointer.load(
        checkpoint, {"model": resumed_model, "optimizer": resumed_optimizer, "extra_state": {}}
    )
    assert state["extra_state"]["global_steps"] == 3
    resumed_scheduler.load_state_dict(state["extra_state"]["scheduler"])
    torch.set_rng_state(state["extra_state"]["rng"])
    torch.testing.assert_close(
        _step(resumed_model, resumed_optimizer, resumed_scheduler), expected_loss, rtol=0, atol=0
    )
    for name, tensor in resumed_model.state_dict().items():
        torch.testing.assert_close(tensor, expected_model[name], rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
    assert resumed_scheduler.state_dict() == scheduler.state_dict()
    assert resumed_model.transient.item() == 456
    assert ("model", True, False) in dmuon_api
    assert ("optimizer", True, True) in dmuon_api


@pytest.mark.parametrize("dedicated_checkpoint", [True, False])
def test_cross_optimizer_resume_rejected_but_weights_only_allowed(tmp_path, dmuon_api, dedicated_checkpoint):
    model, optimizer, _ = _build(dedicated_checkpoint)
    checkpoint = str(tmp_path / "checkpoint")
    DistributedCheckpointer.save(checkpoint, {"model": model, "optimizer": optimizer})
    other_model, other_optimizer, _ = _build(not dedicated_checkpoint)
    with pytest.raises(ValueError, match="Load model weights only"):
        DistributedCheckpointer.load(checkpoint, {"model": other_model, "optimizer": other_optimizer})
    DistributedCheckpointer.load(checkpoint, {"model": other_model})
    for name, tensor in other_model.state_dict().items():
        torch.testing.assert_close(tensor, model.state_dict()[name], rtol=0, atol=0)


def test_dmuon_async_save_rejected_before_writing(tmp_path, dmuon_api):
    model, optimizer, _ = _build()
    checkpoint = tmp_path / "checkpoint"
    with pytest.raises(ValueError, match="synchronous saving only"):
        DistributedCheckpointer.save(str(checkpoint), {"model": model, "optimizer": optimizer}, save_async=True)
    assert not checkpoint.exists()


@pytest.mark.parametrize("operation", ["save", "load"])
@pytest.mark.parametrize("dedicated_model", [True, False])
def test_dmuon_model_optimizer_mismatch_rejected(tmp_path, dmuon_api, operation, dedicated_model):
    model, optimizer, _ = _build(dedicated_model)
    optimizer._veomni_dmuon = not dedicated_model
    with pytest.raises(ValueError, match="dedicated DMuon model"):
        getattr(DistributedCheckpointer, operation)(
            str(tmp_path / "checkpoint"), {"model": model, "optimizer": optimizer}
        )


def test_dmuon_trainable_only_keeps_frozen_weights_and_persistent_buffers(dmuon_api):
    model = _BufferedModel()
    model.linear.weight.requires_grad_(False)
    wrapper = ModelState(model, trainable_only=True)
    state = wrapper.state_dict()
    assert "linear.weight" not in state
    assert "linear.bias" in state
    assert "linear.offset" in state
    assert "counter" in state
    assert "transient" not in state
    frozen_weight = model.linear.weight.detach().clone()
    state["linear.bias"].fill_(1)
    state["linear.offset"].fill_(2)
    wrapper.load_state_dict(state)
    torch.testing.assert_close(model.linear.weight, frozen_weight, rtol=0, atol=0)
    assert torch.all(model.linear.bias == 1)
    assert torch.all(model.linear.offset == 2)


def test_dmuon_strict_model_restore_checks_missing_buffers(dmuon_api):
    wrapper = ModelState(_BufferedModel())
    state = wrapper.state_dict()
    del state["linear.offset"]
    with pytest.raises(RuntimeError, match="missing keys.*linear.offset"):
        wrapper.load_state_dict(state)


@pytest.mark.parametrize("corruption", ["unknown_name", "shape"])
def test_dmuon_strict_model_restore_rejects_invalid_tensors(dmuon_api, corruption):
    wrapper = ModelState(_BufferedModel())
    state = wrapper.state_dict()
    if corruption == "unknown_name":
        state["typo.weight"] = state["linear.weight"]
    else:
        state["linear.weight"] = state["linear.weight"][:, :2]
    with pytest.raises(RuntimeError, match="unexpected keys|invalid shape"):
        wrapper.load_state_dict(state)


@pytest.mark.parametrize("first_dedicated,second_dedicated", [(True, False), (True, True), (False, True)])
def test_checkpoint_overwrite_cannot_leave_stale_dmuon_sidecar(tmp_path, dmuon_api, first_dedicated, second_dedicated):
    checkpoint = str(tmp_path / "checkpoint")
    model, optimizer, _ = _build(first_dedicated)
    DistributedCheckpointer.save(checkpoint, {"model": model, "optimizer": optimizer})
    metadata_before = (tmp_path / "checkpoint" / ".metadata").read_bytes()
    second_model, second_optimizer, _ = _build(second_dedicated)
    with pytest.raises(ValueError, match="use a new directory"):
        DistributedCheckpointer.save(checkpoint, {"model": second_model, "optimizer": second_optimizer})
    assert (tmp_path / "checkpoint" / ".metadata").read_bytes() == metadata_before


@pytest.mark.parametrize(
    "corruption",
    ["section", "group_count", "group_membership", "group_structure", "unknown_param", "state_key", "shape", "step"],
)
def test_dmuon_optimizer_validation_rejects_partial_or_wrong_restore(dmuon_api, corruption):
    model, optimizer, scheduler = _build()
    _step(model, optimizer, scheduler)
    state = sys.modules["dmuon"].get_optimizer_state_dict(model, optimizer, cpu_offload=True, rank0_only=True)
    state["_veomni_param_groups"] = _dmuon_optimizer_layout(model, optimizer)[0]
    if corruption == "section":
        del state["dedicated"]
    elif corruption == "group_count":
        state["param_groups"].append(copy.deepcopy(state["param_groups"][0]))
    elif corruption == "group_membership":
        state["_veomni_param_groups"][0].pop()
    elif corruption == "group_structure":
        state["param_groups"][0]["subgroup_type"] = "muon"
    elif corruption == "unknown_param":
        state["fsdp"]["typo.weight"] = state["fsdp"]["linear.weight"]
    elif corruption == "state_key":
        del state["fsdp"]["linear.weight"]["exp_avg"]
    elif corruption == "shape":
        state["fsdp"]["linear.weight"]["exp_avg"] = torch.zeros(4, 2)
    elif corruption == "step":
        state["fsdp"]["linear.weight"]["step"] = torch.tensor([1, 2])
    with pytest.raises(ValueError, match="DMuon"):
        _validate_dmuon_optimizer_state(model, optimizer, state)


def test_dmuon_scalar_tensor_step_normalized_before_upstream_setter(dmuon_api):
    model, optimizer, scheduler = _build()
    _step(model, optimizer, scheduler)
    state = sys.modules["dmuon"].get_optimizer_state_dict(model, optimizer, cpu_offload=True, rank0_only=True)
    state["_veomni_param_groups"] = _dmuon_optimizer_layout(model, optimizer)[0]
    _validate_dmuon_optimizer_state(model, optimizer, state)
    assert all(isinstance(entry["step"], int) for entry in state["fsdp"].values())


def test_dmuon_dcp_metadata_shape_check_runs_before_partial_loading(tmp_path, dmuon_api):
    original = _BufferedModel()
    checkpoint = str(tmp_path / "checkpoint")
    DistributedCheckpointer.save(checkpoint, {"model": original})
    incompatible = _BufferedModel()
    incompatible.linear = nn.Linear(4, 2)
    incompatible.linear.register_buffer("offset", torch.zeros(4))
    weights_before = incompatible.linear.weight.detach().clone()
    with pytest.raises(ValueError, match="tensor shape differs for linear.weight"):
        DistributedCheckpointer.load(checkpoint, {"model": incompatible})
    torch.testing.assert_close(incompatible.linear.weight, weights_before, rtol=0, atol=0)


def test_dmuon_empty_optimizer_resume_clears_existing_live_state(tmp_path, dmuon_api):
    checkpoint = str(tmp_path / "checkpoint")
    model, optimizer, scheduler = _build()
    DistributedCheckpointer.save(checkpoint, {"model": model, "optimizer": optimizer})
    _step(model, optimizer, scheduler)
    assert optimizer.state
    DistributedCheckpointer.load(checkpoint, {"model": model, "optimizer": optimizer})
    assert not optimizer.state
