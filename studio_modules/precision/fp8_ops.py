"""8-bit float matrix multiplies for the precision/fp8 module: per-tensor scaling, zero padding to multiples of 16, and the autograd
function the linear layers use (studio_modules.ops.linear).

A tensor is scaled so its largest value lands on the format's largest finite number, cast to float8, and the inverse scale travels
beside it; torch._scaled_mm multiplies the two float8 operands and the inverse scales back in while accumulating in float32. The
forward pass uses e4m3 (more precision) for activations and weights, the backward pass e5m2 (more range) for gradients.
The weights, their gradients and the optimizer state stay in the wide type: only the multiply sees 8 bits.
"""
from __future__ import annotations

import contextlib
import contextvars
from functools import lru_cache

import torch

E4M3, E5M2 = getattr(torch, "float8_e4m3fn", None), getattr(torch, "float8_e5m2", None)
ALIGN = 16            # torch._scaled_mm needs every matrix dimension divisible by 16
MIN_DIM = 16          # a layer with a dimension narrower than this (an SSM's low-rank step) stays in the compute dtype
MIN_CAPABILITY = (8, 9)   # Ada (RTX 40 series), Hopper and newer have float8 tensor cores

_active = contextvars.ContextVar("fp8_active", default=False)


def padded(n):
    """`n` rounded up to the next multiple of ALIGN."""
    return -(-n // ALIGN) * ALIGN


def quantize(x, dtype):
    """(q, inverse scale): the 2-D tensor x scaled so its largest magnitude is dtype's largest finite value, cast to `dtype`, and
    zero-padded on both sides to multiples of ALIGN (zeros change no product). The inverse scale is a float32 scalar on x's device.
    An all-zero tensor gets scale 1."""
    top = torch.finfo(dtype).max
    amax = x.detach().abs().amax().float()
    scale = torch.where(amax > 0, top / amax.clamp(min=torch.finfo(torch.float32).tiny), torch.ones_like(amax))
    scaled = (x.float() * scale).clamp(-top, top)
    rows, cols = x.shape
    pad_rows, pad_cols = padded(rows) - rows, padded(cols) - cols
    if pad_rows or pad_cols:
        scaled = torch.nn.functional.pad(scaled, (0, pad_cols, 0, pad_rows))
    return scaled.to(dtype).contiguous(), scale.reciprocal()


def dequantize(q, inverse_scale, shape=None):
    """float32 values of a quantized tensor, cropped to `shape` (the size before padding) when given."""
    out = q.float() * inverse_scale
    return out if shape is None else out[:shape[0], :shape[1]]


def scaled_mm(a, a_scale, b_t, b_scale, out_dtype):
    """a [M, K] @ b_t.T for quantized, padded, contiguous a and b_t [N, K]: [M, N] in `out_dtype`, still padded."""
    return torch._scaled_mm(a, b_t.t(), scale_a=a_scale, scale_b=b_scale, out_dtype=out_dtype)


class Fp8Linear(torch.autograd.Function):
    """y = x @ w for x [M, K] and w [K, N], the three products (forward, input gradient, weight gradient) in float8.

    x is kept for the backward pass as its float8 copy, which is what makes the saved activations smaller than bfloat16's."""

    @staticmethod
    def forward(ctx, x, w, out_dtype):
        (rows, inner), width = x.shape, w.shape[1]
        xq, x_scale = quantize(x, E4M3)
        wq_t, w_scale = quantize(w.t(), E4M3)           # [N, K]: the second operand of _scaled_mm is read transposed
        ctx.save_for_backward(xq, x_scale, wq_t, w_scale)
        ctx.shapes, ctx.dtypes = (rows, inner, width), (x.dtype, w.dtype)
        return scaled_mm(xq, x_scale, wq_t, w_scale, out_dtype)[:rows, :width]

    @staticmethod
    def backward(ctx, grad):
        xq, x_scale, wq_t, w_scale = ctx.saved_tensors
        (rows, inner, width), (x_dtype, w_dtype) = ctx.shapes, ctx.dtypes
        gq, g_scale = quantize(grad.reshape(rows, width), E5M2)
        grad_x = grad_w = None
        if ctx.needs_input_grad[0]:     # g [M, N] @ w.T: the second operand is w itself, [K, N]
            grad_x = scaled_mm(gq, g_scale, wq_t.t().contiguous(), w_scale, x_dtype)[:rows, :inner]
        if ctx.needs_input_grad[1]:     # x.T [K, M] @ g [M, N]: the second operand is g.T, [N, M]
            grad_w = scaled_mm(xq.t().contiguous(), x_scale, gq.t().contiguous(), g_scale, w_dtype)[:inner, :width]
        return grad_x, grad_w, None


def linear_matmul(x, w):
    """x @ w with float8 products. x is [..., K] (any float dtype), w [K, N]; the result is in x's dtype (or the autocast dtype)."""
    out_dtype = torch.get_autocast_dtype(x.device.type) if torch.is_autocast_enabled(x.device.type) else x.dtype
    lead = x.shape[:-1]
    out = Fp8Linear.apply(x.reshape(-1, x.shape[-1]), w, out_dtype)
    return out.reshape(*lead, w.shape[1])


def applies(x, w):
    """True when linear() should run this product in float8: the policy is on, gradients are being recorded (scoring and chat stay
    in the compute dtype: quantizing every weight for each decoded token costs more than it saves), the operands are on a GPU,
    and neither dimension of the layer is narrower than MIN_DIM."""
    return _active.get() and x.is_cuda and torch.is_grad_enabled() and min(w.shape) >= MIN_DIM


@contextlib.contextmanager
def enabled():
    """While open, studio_modules.ops.linear runs its eligible products in float8."""
    token = _active.set(True)
    try:
        yield
    finally:
        _active.reset(token)


@lru_cache(maxsize=None)
def supported(device):
    """True when `device` really runs a float8 product (probed once, not assumed): a CUDA GPU of compute capability 8.9 or newer
    whose torch._scaled_mm, through the same quantize and padded multiply the layers use, matches the wide product."""
    if device != "cuda" or E4M3 is None or E5M2 is None or not hasattr(torch, "_scaled_mm"):
        return False
    try:
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() < MIN_CAPABILITY:
            return False
        x = torch.linspace(-1, 1, 20 * 24, device="cuda").reshape(20, 24).requires_grad_(True)
        w = torch.linspace(1, -1, 24 * 40, device="cuda").reshape(24, 40).requires_grad_(True)
        y = Fp8Linear.apply(x, w, torch.float32)
        y.sum().backward()
        want = x.detach() @ w.detach()
        close = (y - want).abs().max() <= 0.1 * want.abs().max()
        return bool(close) and bool(torch.isfinite(x.grad).all()) and bool(torch.isfinite(w.grad).all())
    except Exception:
        return False


def unsupported_reason(device):
    """Why `device` cannot run float8 products, or "" when it can."""
    if supported(device):
        return ""
    if device != "cuda":
        label = {"cpu": "the CPU", "mps": "Apple GPUs"}.get(device, device)
        return f"8-bit float needs an NVIDIA GPU of the RTX 40 series, H100 or newer; this run would use {label}."
    return "8-bit float needs an NVIDIA GPU of the RTX 40 series, H100 or newer (compute capability 8.9) with a PyTorch that has float8 support; this GPU does not run it."
