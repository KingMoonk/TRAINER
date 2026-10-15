"""How a precision module turns into a forward-pass policy: autocast dtype, loss scaling, and the device fallback.

A precision module's build() returns the dtype of the stored weights (the master copy, which the optimizer also updates).
A mixed-precision module additionally sets `compute` (the dtype matrix multiplies run in under torch.autocast) and,
for float16, `scaled = True` (gradients are produced under a loss scale, see LossScaler).
"""
from __future__ import annotations

from functools import lru_cache

import torch


@lru_cache(maxsize=None)
def autocast_works(device, dtype):
    """True when `device` really runs a matrix multiply in `dtype` under torch.autocast (probed once, not assumed)."""
    try:
        with torch.autocast(device_type=device, dtype=dtype):
            ones = torch.ones((8, 8), device=device)
            product = ones @ ones
        return product.dtype == dtype and float(product.float().sum()) == 512.
    except Exception:
        return False


def plan(module, device):
    """(autocast dtype or None, note). The note says why a mixed-precision module falls back to float32 on this device."""
    compute = getattr(module, "compute", None)
    if compute is None or autocast_works(device, compute):
        return compute, ""
    label = {"cpu": "the CPU", "cuda": "this GPU", "mps": "Apple GPUs"}.get(device, device)
    return None, f"{module.name} is not available on {label} with this PyTorch; training in float32 instead."


def likely_device(requested):
    """The device a run asking for `requested` ("auto" included) will probably land on, for checks made before it starts."""
    if requested in ("cpu", "cuda", "mps"):
        return requested
    from studio_backend import gpu_status
    status = gpu_status()
    return status["kind"] if status["available"] else "cpu"


class LossScaler:
    """Dynamic loss scaling for float16: scale up the loss so small gradients survive, halve the scale on an overflow.

    The compiled model unscales every gradient as it accumulates it, so `scale` only protects the backward pass.
    """

    def __init__(self, scale=2.0 ** 15, growth_interval=1000):
        self.scale, self.growth_interval, self.good_steps = float(scale), growth_interval, 0

    def update(self, overflowed):
        if overflowed:
            self.scale, self.good_steps = max(1., self.scale / 2), 0
        else:
            self.good_steps += 1
            if self.good_steps >= self.growth_interval:
                self.scale, self.good_steps = min(self.scale * 2, 2.0 ** 24), 0
