"""How a precision module turns into a forward-pass policy: autocast dtype, loss scaling, and the device fallback.

A precision module's build() returns the dtype of the stored weights (the master copy, which the optimizer also updates).
A mixed-precision module additionally sets `compute` (the dtype matrix multiplies run in under torch.autocast) and,
for float16, `scaled = True` (gradients are produced under a loss scale, see LossScaler). A module with `fp8 = True` also runs the
linear layers' multiplies in 8-bit floats where fp8_ops.supported(device) proves they work (uses_fp8), and is bfloat16 mixed elsewhere.
"""
from __future__ import annotations

import sys
import time
from functools import lru_cache

import torch


def _best_time(fn, runs=3):
    fn()
    best = float("inf")
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_bf16_crashes():
    """True on a Windows machine whose OS has switched Intel AMX on: PyTorch's oneDNN then runs bfloat16 matrix multiplies (anything
    from about 32x32) through AMX kernels that die with an illegal instruction (0xc000001d) on some of those machines (seen on Intel
    family 6 model 207 CI runners; model 173 runs the same kernels fine, and `ONEDNN_MAX_CPU_ISA=AVX512_CORE_BF16` avoids the crash).
    A crash cannot be caught, and a speed probe would itself run the failing multiply, so this is read from the OS without multiplying.
    Conservative: it also covers AMX machines that would work."""
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        kernel = ctypes.windll.kernel32
        kernel.GetEnabledXStateFeatures.restype = ctypes.c_uint64
        return bool(kernel.GetEnabledXStateFeatures() >> 17 & 3)   # XSTATE bits 17 and 18: AMX tile configuration and tile data
    except Exception:
        return False


def _cpu_half_is_fast(dtype):
    """A CPU without half-precision matrix kernels runs them hundreds of times slower than float32, so probe the speed too."""
    a, b = torch.randn(256, 256), torch.randn(256, 256)

    def mixed():
        with torch.autocast("cpu", dtype=dtype):
            return a @ b

    return _best_time(mixed) <= 2 * _best_time(lambda: a @ b)


@lru_cache(maxsize=None)
def autocast_works(device, dtype):
    """True when `device` really runs a matrix multiply in `dtype` under torch.autocast, and on a CPU not slower than float32 (probed once, not assumed)."""
    if device == "cpu" and dtype == torch.bfloat16 and cpu_bf16_crashes():
        return False
    try:
        with torch.autocast(device_type=device, dtype=dtype):
            ones = torch.ones((8, 8), device=device)
            product = ones @ ones
        if product.dtype != dtype or float(product.float().sum()) != 512.:
            return False
        return device != "cpu" or _cpu_half_is_fast(dtype)
    except Exception:
        return False


def uses_fp8(module, device):
    """True when `module` asks for float8 linear layers and `device` really runs them (and the rest of its policy, bfloat16)."""
    from studio_modules.precision import fp8_ops
    return bool(getattr(module, "fp8", False)) and fp8_ops.supported(device) and autocast_works(device, module.compute)


def unavailable(module, device):
    """Why `module` cannot be picked for a run on `device`, or "" when it can. Only float8 needs hardware that may be missing;
    the other modules fall back with a note (plan) instead of being refused."""
    if not getattr(module, "fp8", False):
        return ""
    from studio_modules.precision import fp8_ops
    reason = fp8_ops.unsupported_reason(device)
    if not reason and not autocast_works(device, module.compute):
        reason = f"{module.name} needs bfloat16 matrix multiplies, which this device does not run."
    return reason


def plan(module, device):
    """(autocast dtype or None, note). The note says why a mixed-precision module falls back on this device: float8 to the
    rest of its policy, bfloat16, the others to float32."""
    compute = getattr(module, "compute", None)
    if getattr(module, "fp8", False):
        reason = unavailable(module, device)
        if not reason:
            return compute, ""
        fallback = compute if autocast_works(device, compute) else None
        return fallback, f"{reason} Training in {'bfloat16' if fallback else 'float32'} instead."
    if compute is None or autocast_works(device, compute):
        return compute, ""
    label = {"cpu": "the CPU", "cuda": "this GPU", "mps": "Apple GPUs"}.get(device, device)
    return None, f"{module.name} is not supported, or far slower than float32, on {label} with this PyTorch; training in float32 instead."


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
