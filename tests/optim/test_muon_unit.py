# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Single-process unit tests for ``veomni.optim.muon``."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn


muon_module = pytest.importorskip("torch.optim._muon")
from torch.optim import Muon as UpstreamMuon  # noqa: E402
from torch.optim._muon import _zeropower_via_newtonschulz as upstream_ns  # noqa: E402

from veomni.optim.muon import (  # noqa: E402
    DEFAULT_NS_COEFFICIENTS,
    DEFAULT_NS_STEPS,
    DistributedMuon,
    batched_newton_schulz,
    split_muon_adamw_params,
)


def _toy_model() -> nn.Module:
    """Tiny model mixing 2D linears with 1D and embedding params."""
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Embedding(8, 4),  # 2D weight, force-AdamW by name
        nn.LayerNorm(4),  # 1D weight + 1D bias
        nn.Linear(4, 8, bias=True),  # 2D weight (Muon), 1D bias (AdamW)
        nn.Linear(8, 4, bias=False),  # 2D weight (Muon), no bias
    )


def _sample_2d(M: int, K: int, dtype: torch.dtype = torch.float32, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(M, K, generator=g, dtype=dtype)


class TestSplit:
    def test_split_partitions_correctly(self):
        model = _toy_model()
        muon, adamw, muon_names, adamw_names = split_muon_adamw_params(model)

        assert sorted(muon_names) == sorted(["2.weight", "3.weight"])

        assert "0.weight" in adamw_names  # embedding
        assert "1.weight" in adamw_names  # LN weight (1D)
        assert "1.bias" in adamw_names  # LN bias (1D)
        assert "2.bias" in adamw_names  # Linear bias (1D)

        assert set(muon_names).isdisjoint(set(adamw_names))
        assert len(muon) == len(muon_names)
        assert len(adamw) == len(adamw_names)

    def test_no_decay_modules_overrides_to_adamw(self):
        model = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
        muon, adamw, muon_names, adamw_names = split_muon_adamw_params(
            model,
            no_decay_modules=["Linear"],
        )
        assert muon_names == []
        assert "0.weight" in adamw_names
        assert "0.bias" in adamw_names
        assert "1.weight" in adamw_names

    def test_frozen_params_excluded(self):
        model = nn.Linear(4, 4)
        for p in model.parameters():
            p.requires_grad_(False)
        muon, adamw, *_ = split_muon_adamw_params(model)
        assert muon == [] and adamw == []


class TestParamShapeEligibility:
    """``DistributedMuon`` must accept 2D/3D and reject 1D/4D+."""

    @pytest.mark.parametrize(
        "shape, accepted",
        [
            ((4,), False),  # 1D bias / norm
            ((4, 8), True),  # 2D dense linear
            ((4, 8, 16), True),  # 3D MoE expert stack
            ((2, 4, 8, 16), False),  # 4D conv weight
        ],
    )
    def test_eligibility(self, shape, accepted):
        p = nn.Parameter(torch.randn(*shape))
        if accepted:
            opt = DistributedMuon([p], lr=1e-3)
            p.grad = torch.randn_like(p)
            opt.step()
            assert torch.isfinite(p).all()
        else:
            with pytest.raises(ValueError, match="2D and 3D"):
                DistributedMuon([p], lr=1e-3)


class TestNumerics:
    """Plain-tensor parity with upstream ``torch.optim.Muon``."""

    def test_single_step_matches_upstream(self):
        torch.manual_seed(123)
        w = torch.randn(8, 16, dtype=torch.float32)
        a = nn.Parameter(w.clone())
        b = nn.Parameter(w.clone())

        opt_up = UpstreamMuon(
            [a],
            lr=1e-2,
            weight_decay=0.05,
            momentum=0.9,
            nesterov=True,
            adjust_lr_fn="match_rms_adamw",
        )
        opt_ours = DistributedMuon(
            [b],
            lr=1e-2,
            weight_decay=0.05,
            momentum=0.9,
            nesterov=True,
            adjust_lr_fn="match_rms_adamw",
        )

        torch.manual_seed(7)
        grad = torch.randn_like(a)
        a.grad = grad.clone()
        b.grad = grad.clone()

        opt_up.step()
        opt_ours.step()

        torch.testing.assert_close(a.detach(), b.detach(), atol=0.0, rtol=0.0)

    def test_state_dict_roundtrip(self):
        """Step + serialize + restore preserves both Muon and AdamW state."""
        from veomni.optim import build_optimizer

        model_a = _toy_model()
        opt_a = build_optimizer(
            model_a,
            lr=1e-4,
            weight_decay=0.01,
            optimizer_type="muon",
            muon_kwargs={"lr": 1e-2, "adjust_lr_fn": "match_rms_adamw"},
        )
        for p in model_a.parameters():
            if p.requires_grad:
                p.grad = torch.randn_like(p)
        opt_a.step()
        sd_before = opt_a.state_dict()

        flat_keys = set(sd_before.keys())
        assert any("momentum_buffer" in k for k in flat_keys), (
            f"expected Muon momentum_buffer in state_dict; got {sorted(flat_keys)[:8]}"
        )
        assert any("exp_avg" in k for k in flat_keys), (
            f"expected AdamW exp_avg in state_dict; got {sorted(flat_keys)[:8]}"
        )

        model_b = _toy_model()
        model_b.load_state_dict(model_a.state_dict())
        opt_b = build_optimizer(
            model_b,
            lr=1e-4,
            weight_decay=0.01,
            optimizer_type="muon",
            muon_kwargs={"lr": 1e-2, "adjust_lr_fn": "match_rms_adamw"},
        )
        opt_b.load_state_dict(sd_before)

        sd_after = opt_b.state_dict()
        assert set(sd_before.keys()) == set(sd_after.keys())
        for k, v_before in sd_before.items():
            v_after = sd_after[k]
            if isinstance(v_before, torch.Tensor):
                torch.testing.assert_close(v_before, v_after, atol=0.0, rtol=0.0, msg=f"state_dict mismatch at {k}")


class TestBuildOptimizer:
    def test_build_returns_multi_optimizer(self):
        from veomni.optim import build_optimizer
        from veomni.optim.optimizer import MultiOptimizer

        model = _toy_model()
        opt = build_optimizer(
            model,
            lr=1e-4,
            weight_decay=0.01,
            optimizer_type="muon",
            muon_kwargs={"lr": 5e-3, "adjust_lr_fn": "match_rms_adamw"},
        )
        assert isinstance(opt, MultiOptimizer)
        assert "muon" in opt.optimizers_dict and "adamw" in opt.optimizers_dict
        assert isinstance(opt.optimizers_dict["muon"], DistributedMuon)
        assert isinstance(opt.optimizers_dict["adamw"], torch.optim.AdamW)

    def test_build_no_2d_params_raises(self):
        from veomni.optim import build_optimizer

        model = nn.LayerNorm(4)
        with pytest.raises(ValueError, match="no eligible 2D/3D parameters"):
            build_optimizer(model, optimizer_type="muon", muon_kwargs={"lr": 1e-3})


class TestSemanticGroups:
    @staticmethod
    def build(model, groups, **kwargs):
        from veomni.optim import build_optimizer

        return build_optimizer(
            model,
            lr=1e-3,
            weight_decay=0.1,
            optimizer_type="muon",
            param_groups=groups,
            muon_kwargs={"lr": 5e-3, "weight_decay": 0.2},
            **kwargs,
        )

    def test_learning_rates_and_decay_match_independent_optimizers(self):
        from veomni.optim import build_lr_scheduler

        model = nn.ModuleDict({"visual": nn.Linear(4, 4), "backbone": nn.Linear(4, 4)})
        reference = nn.ModuleDict({"visual": nn.Linear(4, 4), "backbone": nn.Linear(4, 4)})
        reference.load_state_dict(model.state_dict())
        groups = [
            {"params": list(model["visual"].parameters()), "lr": 2e-4, "weight_decay": 0.0},
            {"params": list(model["backbone"].parameters())},
        ]
        optimizer = self.build(model, groups, no_decay_params=["bias"])
        muon = optimizer.optimizers_dict["muon"]
        adamw = optimizer.optimizers_dict["adamw"]
        assert [g["lr"] for g in muon.param_groups] == pytest.approx([1e-3, 5e-3])
        assert [g["lr"] for g in adamw.param_groups] == pytest.approx([2e-4, 1e-3])
        assert [g["weight_decay"] for g in muon.param_groups] == [0.0, 0.2]
        assert [g["weight_decay"] for g in adamw.param_groups] == [0.0, 0.0]
        assert "lr" not in groups[1]  # Caller-owned group dictionaries stay unchanged.

        references = [
            UpstreamMuon([reference["visual"].weight], lr=1e-3, weight_decay=0.0, adjust_lr_fn="match_rms_adamw"),
            UpstreamMuon([reference["backbone"].weight], lr=5e-3, weight_decay=0.2, adjust_lr_fn="match_rms_adamw"),
            torch.optim.AdamW([reference["visual"].bias], lr=2e-4, weight_decay=0.0, betas=(0.9, 0.95)),
            torch.optim.AdamW([reference["backbone"].bias], lr=1e-3, weight_decay=0.0, betas=(0.9, 0.95)),
        ]
        for _ in range(3):
            for actual, expected in zip(model.parameters(), reference.parameters()):
                grad = torch.randn_like(actual)
                actual.grad, expected.grad = grad.clone(), grad.clone()
            optimizer.step()
            for child in references:
                child.step()
            for actual, expected in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        scheduler = build_lr_scheduler(optimizer, train_steps=10, lr=1e-3, lr_decay_style="cosine")
        optimizer.step()
        scheduler.step()
        assert muon.param_groups[0]["lr"] / muon.param_groups[1]["lr"] == pytest.approx(0.2)
        assert adamw.param_groups[0]["lr"] / adamw.param_groups[1]["lr"] == pytest.approx(0.2)

    def test_fallback_respects_no_decay_modules_and_generator_groups(self):
        model = nn.ModuleDict({"hidden": nn.Linear(4, 4), "norm": nn.LayerNorm(4), "lookup": nn.Embedding(4, 4)})
        model["hidden"].bias.requires_grad_(False)
        optimizer = self.build(
            model,
            [
                {"params": model["hidden"].parameters(), "lr": 0.0},
                {"params": model["norm"].parameters(), "lr": 2e-4, "weight_decay": 0.3},
                {"params": model["lookup"].parameters(), "lr": 4e-4, "weight_decay": 0.3},
            ],
            no_decay_modules=["LayerNorm"],
        )
        assert optimizer.optimizers_dict["muon"].param_groups[0]["lr"] == 0.0
        fallback = optimizer.optimizers_dict["adamw"].param_groups
        assert [(g["lr"], g["weight_decay"]) for g in fallback] == [(2e-4, 0.0), (4e-4, 0.3)]
        assigned = [
            p for child in optimizer.optimizers_dict.values() for group in child.param_groups for p in group["params"]
        ]
        assert {id(p) for p in assigned} == {id(p) for p in model.parameters() if p.requires_grad}
        assert len(assigned) == len({id(p) for p in assigned})

    def test_flattened_optimizer_state_restore_keeps_semantic_groups(self):
        model = nn.ModuleDict({"visual": nn.Linear(4, 4), "hidden": nn.Linear(4, 4)})

        def build(target):
            return self.build(
                target,
                [
                    {"params": target["visual"].parameters(), "lr": 2e-4, "weight_decay": 0.0},
                    {"params": target["hidden"].parameters()},
                ],
            )

        optimizer = build(model)
        for parameter in model.parameters():
            parameter.grad = torch.randn_like(parameter)
        optimizer.step()
        restored_model = copy.deepcopy(model)
        restored = build(restored_model)
        restored.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        for name in optimizer.optimizers_dict:
            actual = restored.optimizers_dict[name]
            expected = optimizer.optimizers_dict[name]
            assert [g["lr"] for g in actual.param_groups] == [g["lr"] for g in expected.param_groups]
            assert [g["weight_decay"] for g in actual.param_groups] == [
                g["weight_decay"] for g in expected.param_groups
            ]
        for parameter, resumed_parameter in zip(model.parameters(), restored_model.parameters()):
            grad = torch.randn_like(parameter)
            parameter.grad, resumed_parameter.grad = grad.clone(), grad.clone()
        optimizer.step()
        restored.step()
        for parameter, resumed_parameter in zip(model.parameters(), restored_model.parameters()):
            torch.testing.assert_close(parameter, resumed_parameter, rtol=0, atol=0)

    @pytest.mark.parametrize("invalid", ["duplicate", "missing", "foreign", "negative_lr", "nan_lr"])
    def test_invalid_groups_rejected(self, invalid):
        model = nn.Linear(4, 4)
        groups = [{"params": list(model.parameters())}]
        expected = "duplicate"
        if invalid == "duplicate":
            groups.append({"params": [model.weight]})
        elif invalid == "missing":
            groups[0]["params"] = [model.weight]
            expected = "every trainable"
        elif invalid == "foreign":
            groups[0]["params"].append(nn.Parameter(torch.ones(4, 4)))
            expected = "outside the model"
        else:
            groups[0]["lr"] = -1.0 if invalid == "negative_lr" else float("nan")
            expected = "finite and nonnegative"
        with pytest.raises(ValueError, match=expected):
            self.build(model, groups)


class TestBatchedNS:
    """Math primitive parity + contract checks."""

    def test_2d_byte_parity_with_upstream(self):
        """2D batched NS must equal upstream's 2D NS bit-for-bit."""
        x = _sample_2d(8, 16, dtype=torch.float32, seed=42)
        out_upstream = upstream_ns(x.clone(), DEFAULT_NS_COEFFICIENTS, DEFAULT_NS_STEPS, eps=1e-7)
        out_ours = batched_newton_schulz(
            x.clone(),
            ns_coefficients=DEFAULT_NS_COEFFICIENTS,
            ns_steps=DEFAULT_NS_STEPS,
            eps=1e-7,
            compute_dtype=torch.bfloat16,
        )
        torch.testing.assert_close(out_ours.to(torch.bfloat16), out_upstream, atol=0.0, rtol=0.0)

    @pytest.mark.parametrize(
        "shape",
        [
            (3, 8, 16),  # tall expert stack
            (4, 16, 8),  # wide expert stack (transpose path)
            (2, 32, 32),  # square expert stack
        ],
    )
    def test_3d_matches_per_slice(self, shape):
        """3D batched NS must match a Python loop calling 2D NS per slice."""
        N, M, K = shape
        g = torch.Generator(device="cpu").manual_seed(123)
        x = torch.randn(N, M, K, generator=g, dtype=torch.float32)

        per_slice = torch.stack(
            [upstream_ns(x[i].clone(), DEFAULT_NS_COEFFICIENTS, DEFAULT_NS_STEPS, eps=1e-7) for i in range(N)],
            dim=0,
        )
        batched = batched_newton_schulz(
            x.clone(),
            ns_coefficients=DEFAULT_NS_COEFFICIENTS,
            ns_steps=DEFAULT_NS_STEPS,
            eps=1e-7,
            compute_dtype=torch.bfloat16,
        )
        torch.testing.assert_close(batched.to(torch.bfloat16), per_slice, atol=5e-3, rtol=5e-3)

    @pytest.mark.parametrize(
        "case",
        [
            {"ns_steps": 150, "match": "less than 100"},
            {"ns_coefficients": (1.0, 2.0), "match": "exactly 3"},
        ],
        ids=["too_many_steps", "bad_coefficients"],
    )
    def test_contracts_reject_invalid(self, case):
        kwargs = {k: v for k, v in case.items() if k != "match"}
        with pytest.raises(ValueError, match=case["match"]):
            batched_newton_schulz(_sample_2d(4, 8), **kwargs)

    @pytest.mark.parametrize("shape", [(4, 8), (3, 4, 8)], ids=["2d", "3d"])
    def test_zero_input_safe(self, shape):
        out = batched_newton_schulz(torch.zeros(*shape))
        assert torch.isfinite(out).all()
        assert out.shape == tuple(shape)


def assert_partition(model, muon, adamw, muon_names, adamw_names):
    canonical = dict(model.named_parameters())
    assigned = [id(p) for p in muon + adamw]
    assert len(assigned) == len(set(assigned))
    assert set(assigned) == {id(p) for p in model.parameters() if p.requires_grad}
    assert len(muon) == len(muon_names) and len(adamw) == len(adamw_names)
    assert all(canonical[name] is parameter for name, parameter in zip(muon_names, muon))
    assert all(canonical[name] is parameter for name, parameter in zip(adamw_names, adamw))


@pytest.mark.parametrize("conv_type", [nn.Conv1d, nn.ConvTranspose1d])
def test_convolution_falls_back_without_losing_3d_experts(conv_type):
    model = nn.ModuleDict(
        {
            "temporal_mixer": conv_type(8, 8, 4, groups=8),
            "projection": nn.Linear(8, 8),
            "experts": nn.ParameterList([nn.Parameter(torch.randn(4, 8, 16))]),
        }
    )
    muon, adamw, muon_names, adamw_names = split_muon_adamw_params(model)
    assert set(muon_names) == {"projection.weight", "experts.0"}
    assert set(adamw_names) == {"temporal_mixer.weight", "temporal_mixer.bias", "projection.bias"}
    assert_partition(model, muon, adamw, muon_names, adamw_names)


@pytest.mark.parametrize("conv_type", [nn.Conv1d, nn.ConvTranspose1d])
@pytest.mark.parametrize("alias_position", ["before", "after", "root"])
@pytest.mark.parametrize("frozen", [False, True])
def test_tied_convolution_uses_parameter_identity(conv_type, alias_position, frozen):
    conv = conv_type(8, 8, 4, groups=8)
    conv.weight.requires_grad_(not frozen)
    alias = nn.Module()
    alias.register_parameter("shared", conv.weight)
    model = nn.Module()
    if alias_position == "root":
        model.register_parameter("shared", conv.weight)
        model.add_module("temporal_mixer", conv)
    else:
        children = [("alias", alias), ("temporal_mixer", conv)]
        if alias_position == "after":
            children.reverse()
        for name, child in children:
            model.add_module(name, child)
    model.add_module("projection", nn.Linear(8, 8))
    model.add_module("experts", nn.ParameterList([nn.Parameter(torch.randn(4, 8, 16))]))
    muon, adamw, muon_names, adamw_names = split_muon_adamw_params(model)
    assert all(p is not conv.weight for p in muon)
    assert sum(p is conv.weight for p in adamw) == (not frozen)
    assert set(muon_names) == {"projection.weight", "experts.0"}
    assert_partition(model, muon, adamw, muon_names, adamw_names)


@pytest.mark.parametrize("module_type", [nn.Embedding, nn.Linear])
@pytest.mark.parametrize("alias_first", [False, True])
def test_existing_module_fallback_also_uses_parameter_identity(module_type, alias_first):
    owner = module_type(8, 8)
    alias = nn.Module()
    alias.register_parameter("shared", owner.weight)
    children = [("alias", alias), ("owner", owner)]
    if not alias_first:
        children.reverse()
    model = nn.ModuleDict(children)
    kwargs = {"no_decay_modules": ["Linear"]} if module_type is nn.Linear else {}
    muon, adamw, muon_names, adamw_names = split_muon_adamw_params(model, **kwargs)
    assert all(p is not owner.weight for p in muon)
    assert sum(p is owner.weight for p in adamw) == 1
    assert_partition(model, muon, adamw, muon_names, adamw_names)
