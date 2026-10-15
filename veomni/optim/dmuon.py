"""Thin FSDP2 integration for the optional official DMuon package."""

import math

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate

from ..distributed.parallel_state import get_parallel_state
from ..utils.device import synchronize
from .muon import split_muon_adamw_params


def dedicate_model(model, config, mixed_precision, basic_modules, reshard_after_forward):
    """Take ownership before fully_shard; parameters must already be loaded."""
    import dmuon

    ps = get_parallel_state()
    if ps.tp_enabled or ps.sp_enabled or ps.any_extra_parallel_enabled:
        raise ValueError("DMuon integration currently supports dense FSDP2/HSDP without TP/SP/EP")
    params, _, names, _ = split_muon_adamw_params(model, config.no_decay_modules, config.no_decay_params)
    if not params or any(p.ndim != 2 for p in params):
        raise ValueError("DMuon requires eligible dense 2D matrices")
    selected = set(names)
    boundaries = set(basic_modules) | {"Qwen3_5VisionPatchMerger"}
    # Root action projections are used directly on VLM-only ranks. Paired
    # decoder boundaries cover native.forward calls to the action expert.
    assignment = dmuon.dedicate_params(
        model,
        ps.dp_shard_mesh,
        predicate=lambda name, param: name in selected,
        replicate_mesh=ps.dp_replicate_mesh if ps.dp_replicate_size > 1 else None,
        reshard_after_forward=reshard_after_forward,
        param_policy={
            "defaults": dict(
                param_dtype=getattr(torch, mixed_precision.param_dtype) if mixed_precision.enable else torch.float32,
                grad_dtype=getattr(torch, mixed_precision.reduce_dtype) if mixed_precision.enable else torch.float32,
                master_dtype=torch.float32,
                optim_dtype=torch.float32,
                cast_forward_inputs=False,
            )
        },
        hook_boundary_predicate=lambda module: module is model or module.__class__.__name__ in boundaries,
    )
    if len(assignment) != len(params):
        raise RuntimeError("DMuon did not take ownership of all selected matrices")
    from .dmuon_hooks import adapt_multibranch_hooks

    adapt_multibranch_hooks(model)
    model._veomni_dmuon_names = selected


def build_dmuon(model, lr, betas, eps, weight_decay, param_groups, no_decay_modules, no_decay_params, muon_kwargs):
    """Preserve semantic learning-rate groups and AdamW exclusions."""
    import dmuon

    from .optimizer import get_parameter_names

    if not hasattr(model, "_dedicated_comm_ctx"):
        raise ValueError("DMuon must be selected before FSDP2 parallelization")
    options = dict(muon_kwargs or {})
    supported = {
        "lr",
        "momentum",
        "nesterov",
        "weight_decay",
        "ns_steps",
        "ns_coefficients",
        "eps",
        "adjust_lr_fn",
        "dmuon_ns_backend",
        "dmuon_ns_coefficients",
    }
    if options.keys() - supported:
        raise ValueError(f"Unknown DMuon options: {sorted(options.keys() - supported)}")
    muon_lr = options.get("lr", 2e-2)
    muon_decay = options.get("weight_decay", 0.0)
    momentum = options.get("momentum", 0.95)
    for name, value in (
        ("Muon learning rate", muon_lr),
        ("Muon weight decay", muon_decay),
        ("AdamW weight decay", weight_decay),
        ("AdamW epsilon", eps),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"DMuon {name} must be finite and nonnegative")
    if not math.isfinite(momentum) or not 0 <= momentum < 1:
        raise ValueError("DMuon momentum must be finite and in [0, 1)")
    if not isinstance(options.get("nesterov", True), bool):
        raise ValueError("DMuon nesterov must be a boolean")
    if len(betas) != 2 or any(not math.isfinite(beta) or not 0 <= beta < 1 for beta in betas):
        raise ValueError("DMuon AdamW betas must be finite and in [0, 1)")
    if options.get("eps", 1e-7) != 1e-7:
        raise ValueError("Official DMuon fixes its NS normalization epsilon at 1e-7")
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("DMuon AdamW base learning rate must be positive and finite")
    if options.get("adjust_lr_fn", "match_rms_adamw") != "match_rms_adamw":
        raise ValueError("DMuon uses match_rms_adamw learning-rate scaling")
    backend = options.get("dmuon_ns_backend", "gram")
    coefficients = options.get("dmuon_ns_coefficients", "polar_express")
    steps = options.get("ns_steps", 5)
    if type(steps) is not int or not 0 < steps < 100:
        raise ValueError("DMuon NS steps must be an integer in [1, 99]")
    classic_coefficients = options.get("ns_coefficients", (3.4445, -4.775, 2.0315))
    if len(classic_coefficients) != 3 or any(not math.isfinite(value) for value in classic_coefficients):
        raise ValueError("DMuon NS coefficients must contain exactly three finite values")
    if coefficients == "polar_express" and steps != 5:
        raise ValueError("DMuon Polar Express uses exactly five NS iterations")
    if coefficients not in {"classic", "polar_express"} or backend not in {"direct", "gram"}:
        raise ValueError("Unknown DMuon NS backend or coefficients")
    ns = dmuon.NewtonSchulz(
        backend=backend,
        coefficients=[list(classic_coefficients)] * steps if coefficients == "classic" else None,
        restart_iterations=[] if coefficients == "classic" else None,
    )
    decay_names = set(get_parameter_names(model, no_decay_modules, no_decay_params))
    names = {id(param): name for name, param in model.named_parameters()}
    groups = []
    if isinstance(param_groups, dict):
        raise TypeError("DMuon parameter groups must be an iterable of dictionaries")
    semantic_groups = list(param_groups) if param_groups is not None else [{"params": model.parameters()}]
    if not semantic_groups:
        raise ValueError("DMuon parameter groups must not be empty")
    for group in semantic_groups:
        if not isinstance(group, dict) or "params" not in group:
            raise ValueError("DMuon parameter groups must be dictionaries containing 'params'")
        raw_params = group["params"]
        params = [raw_params] if isinstance(raw_params, torch.Tensor) else list(raw_params)
        if any(not isinstance(param, torch.Tensor) for param in params):
            raise TypeError("DMuon parameter groups must contain tensors")
        params = [param for param in params if param.requires_grad]
        if any(id(param) not in names for param in params):
            raise ValueError("DMuon parameter groups contain a parameter outside the model")
        group_lr = group.get("lr", lr)
        scaled_muon_lr = muon_lr * group_lr / lr
        if not math.isfinite(group_lr) or group_lr < 0 or not math.isfinite(scaled_muon_lr):
            raise ValueError("DMuon parameter-group learning rates must be finite and nonnegative")
        group_decay = group.get("weight_decay", weight_decay)
        if not math.isfinite(group_decay) or group_decay < 0:
            raise ValueError("DMuon parameter-group weight decay must be finite and nonnegative")
        for decay in (True, False):
            subset = [p for p in params if (names[id(p)] in decay_names) == decay]
            if not subset:
                continue
            groups.append(
                {
                    **group,
                    "params": subset,
                    "muon_lr": scaled_muon_lr,
                    "muon_weight_decay": 0.0 if not decay or group.get("weight_decay") == 0.0 else muon_decay,
                    "adamw_lr": group_lr,
                    "adamw_weight_decay": group_decay if decay else 0.0,
                }
            )
    optimizer = dmuon.Muon(
        model,
        lr=muon_lr,
        momentum=momentum,
        nesterov=options.get("nesterov", True),
        weight_decay=muon_decay,
        ns_steps=steps,
        ns_backend=ns,
        adamw_lr=lr,
        adamw_betas=betas,
        adamw_eps=eps,
        adamw_weight_decay=weight_decay,
        param_groups=groups,
        first_step_progress_log_limit=2,
    )
    optimizer._veomni_dmuon = True
    model._veomni_dmuon_optimizer = optimizer
    warmup_dmuon_graphs(optimizer)
    return optimizer


@torch.no_grad()
def warmup_dmuon_graphs(optimizer):
    """Capture dense FP32 NS shapes before DataLoader pinned-memory threads start."""
    from dmuon.optim import ns_graph

    if not ns_graph.enabled():
        return
    seen = set()
    for param in optimizer._dedicated_params:
        if id(param) not in optimizer._dp_to_muon_group_idx:
            continue
        example = param._owned_data.detach()
        key = (tuple(example.shape), example.dtype, example.device.index)
        if key in seen:
            continue
        # NS is a pure function: exercise its official eager/capture path
        # without an optimizer step, momentum update, or training RNG draw.
        optimizer._ns_compute(example, is_tp=False)
        optimizer._ns_compute(example, is_tp=False)
        if optimizer._ns_graph_cache._graphs[key].graph is None:
            raise RuntimeError(f"DMuon NS graph warmup failed for {key}; inspect the upstream capture error")
        seen.add(key)
    synchronize()


@torch.no_grad()
def clip_dmuon_grad_norm(model, max_norm, norm_type=2.0, error_if_nonfinite=False):
    """Use one global clipping coefficient for dedicated and FSDP grads."""
    max_norm, norm_type = float(max_norm), float(norm_type)
    if max_norm < 0 or math.isnan(max_norm) or norm_type <= 0 or math.isnan(norm_type):
        raise ValueError("DMuon gradient clipping requires max_norm >= 0 and norm_type > 0")
    optimizer = model._veomni_dmuon_optimizer
    optimizer._ensure_grads_ready()
    grads = [(dp._reduced_grad, 1) for dp in optimizer._dedicated_params if dp._reduced_grad is not None]
    for param in optimizer._fsdp_params:
        grad = param.grad
        if grad is None:
            continue
        replicas = 1
        if isinstance(grad, DTensor):
            replicas = math.prod(
                grad.device_mesh.size(i)
                for i, placement in enumerate(grad.placements)
                if isinstance(placement, Replicate)
            )
            grad = grad.to_local()
        grads.append((grad, replicas))
    total = torch.zeros((), device=next(model.parameters()).device, dtype=torch.float32)
    for grad, replicas in grads:
        norm = torch.linalg.vector_norm(grad.detach().float(), norm_type)
        total = torch.maximum(total, norm) if math.isinf(norm_type) else total + norm.pow(norm_type) / replicas
    dist.all_reduce(
        total,
        op=dist.ReduceOp.MAX if math.isinf(norm_type) else dist.ReduceOp.SUM,
        group=get_parallel_state().dp_group,
    )
    if not math.isinf(norm_type):
        total = total.pow(1 / norm_type)
    if error_if_nonfinite and not torch.isfinite(total):
        raise RuntimeError("Non-finite DMuon global gradient norm")
    coefficient = (max_norm / (total + 1e-6)).clamp(max=1.0)
    for grad, _ in grads:
        grad.mul_(coefficient)
    return total
