"""Instance-local DMuon hooks for modules with multiple autograd branches."""

from types import MethodType

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten


def _wrap_grad_tensors(state, tree, backward_node):
    leaves, spec = tree_flatten(tree)
    tensors, indices = [], {}
    for leaf in leaves:
        if isinstance(leaf, torch.Tensor) and leaf.requires_grad and id(leaf) not in indices:
            indices[id(leaf)] = len(tensors)
            tensors.append(leaf)
    if not tensors:
        return tree
    # One node synchronizes every branch before unshard/reduce, including
    # branches reachable through nested containers or ModelOutput fields.
    wrapped = backward_node.apply(state, *tensors)
    return tree_unflatten([wrapped[indices[id(leaf)]] if id(leaf) in indices else leaf for leaf in leaves], spec)


def _register_post_backward(self, args, kwargs):
    from dmuon._core.state import _DedicatedPostBackward

    return _wrap_grad_tensors(self, (args, kwargs), _DedicatedPostBackward)


def _register_pre_backward(self, output):
    from dmuon._core.state import _DedicatedPreBackward

    return _wrap_grad_tensors(self, output, _DedicatedPreBackward)


def _run_visual_post_backward(self):
    from dmuon._core.state import DedicatedState

    # Image/video (including dummy) forwards reuse the same ViT weights.
    # Input-edge hooks may fire before the shared AccumulateGrad node, so
    # reduce only once the entire graph has finished on every rank.
    if getattr(self.comm_ctx, "_in_root_post_backward", False):
        DedicatedState._run_post_backward(self)


def adapt_multibranch_hooks(model):
    """Preserve official lifecycle logic while wrapping all differentiable edges."""
    from dmuon.utils import iter_dedicated_states

    visual_modules = {
        id(child)
        for name, module in model.named_modules()
        if name.rsplit(".", 1)[-1] == "visual"
        for child in module.modules()
    }
    for state in iter_dedicated_states(model):
        state._register_post_backward = MethodType(_register_post_backward, state)
        state._register_pre_backward = MethodType(_register_pre_backward, state)
        if id(state.module) in visual_modules:
            state._run_post_backward = MethodType(_run_visual_post_backward, state)
