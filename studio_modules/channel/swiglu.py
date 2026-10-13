"""SiLU-gated channel mixer with three bias-free projections."""
from __future__ import annotations

from dataclasses import dataclass, field

from studio_modules.ops import activate, activate_derivative, linear, linear_back
from studio_registry import ModuleSpec, register
from studio_slots import Estimate, Layer


@dataclass
class SwiGLUConfig:
    d_ff: int = field(default=0, metadata=dict(min=0, max=8192, help="Hidden width; 0 rounds 8d/3 up to a multiple of 16."))


def width(cfg, dims):
    return cfg.d_ff or ((8 * dims.d_model + 47) // 48) * 16


class SwiGLULayer(Layer):
    def __init__(self, cfg, dims):
        super().__init__(dims)
        f = width(cfg, dims)
        self.add("gate.w", (dims.d_model, f))
        self.add("up.w", (dims.d_model, f))
        self.add("down.w", (f, dims.d_model), "residual_out")

    def forward(self, x, ctx):
        pre = ctx.emit("swiglu.pre", linear(self, x, "gate"))
        gate, sigmoid = activate(self.xp, pre, "silu")
        up = linear(self, x, "up")
        act = ctx.emit("swiglu.act", gate * up)
        self.cache = (x, pre, sigmoid, gate, up, act)
        return linear(self, act, "down")

    def backward(self, dy):
        x, pre, sigmoid, gate, up, act = self.cache
        dact = linear_back(self, dy, act, "down")
        dx = linear_back(self, dact * up * activate_derivative(pre, sigmoid, "silu"), x, "gate")
        return dx + linear_back(self, dact * gate, x, "up")


@register
class SwiGLU(ModuleSpec):
    id = "channel/swiglu"
    slot = "channel"
    name = "SwiGLU"
    blurb = "SiLU gate times an up projection, then a down projection."
    ref = "Shazeer 2020"
    config = SwiGLUConfig
    hooks = (("swiglu.pre", "B T F"), ("swiglu.act", "B T F"))
    glyph3d = {"kind": "tower", "hooks": ["swiglu.act"]}

    def symbols(self, cfg, dims):
        return {"F": width(cfg, dims)}

    def estimate(self, cfg, dims):
        f = width(cfg, dims)
        return Estimate(params=3 * dims.d_model * f, flops_per_token=6 * dims.d_model * f,
                        activation_bytes=(4 * f + dims.d_model) * dims.bytes)

    def build(self, cfg, dims, rng):
        return SwiGLULayer(cfg, dims)
