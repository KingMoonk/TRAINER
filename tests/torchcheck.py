"""Test helpers for the PyTorch engine: autograd stand-ins for the old hand-written Layer.backward."""
import numpy as np
import torch

from studio_slots import Ctx


def to_np(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def tensor(value, dtype=torch.float64):
    """A tensor of `value`: integer arrays become int64 ids, everything else the given float dtype."""
    if isinstance(value, torch.Tensor):
        return value
    array = np.asarray(value)
    if np.issubdtype(array.dtype, np.integer):
        return torch.as_tensor(array, dtype=torch.long)
    return torch.as_tensor(array, dtype=dtype)


def fill(parameter, values):
    """In-place assignment of NumPy values into a parameter tensor (the old `value[...] = ...`)."""
    with torch.no_grad():
        parameter.copy_(torch.as_tensor(np.asarray(values), dtype=parameter.dtype, device=parameter.device))


def layer_backward(layer, x, dy, ctx=None, embedding=None):
    """Run forward + autograd for one layer: ADDS parameter gradients into layer.grads (and embedding.grads)
    like the old backward() did, and returns d(sum(forward · dy))/dx, or None for integer inputs."""
    layers = [layer] + ([embedding] if embedding is not None else [])
    ctx = ctx if ctx is not None else Ctx(hooks=True, embedding=embedding)
    saved = {}
    for owner in layers:
        for name, parameter in owner.params.items():
            saved[(id(owner), name)] = parameter
            owner.params[name] = parameter.detach().requires_grad_(True)
    xt = tensor(x, dtype=layer.dtype)
    if xt.is_floating_point():
        xt = xt.detach().clone().requires_grad_(True)
    try:
        with torch.enable_grad():
            y = layer.forward(xt, ctx)
            total = (y * torch.as_tensor(np.asarray(dy), dtype=y.dtype)).sum()
            inputs = [owner.params[name] for owner in layers for name in owner.params]
            if xt.requires_grad:
                inputs.append(xt)
            grads = torch.autograd.grad(total, inputs, allow_unused=True)
    finally:
        for owner in layers:
            for name in owner.params:
                owner.params[name] = saved[(id(owner), name)]
    index = 0
    with torch.no_grad():
        for owner in layers:
            for name in owner.params:
                if grads[index] is not None:
                    owner.grads[name].add_(grads[index])
                index += 1
    return grads[-1].detach() if xt.requires_grad else None
