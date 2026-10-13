"""SiLU-gated channel mixer with three bias-free projections."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch.nn.functional as F

from studio_modules.ops import linear
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
        act = ctx.emit("swiglu.act", F.silu(pre) * linear(self, x, "up"))
        return linear(self, act, "down")


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
