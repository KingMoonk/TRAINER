"""Causal depthwise convolution followed by a pointwise output projection."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch.nn.functional as F

from studio_modules.ops import causal_conv, linear
from studio_registry import ModuleSpec, register
from studio_slots import Estimate, Layer


@dataclass
class ConvConfig:
    kernel: int = field(default=4, metadata=dict(min=1, max=32))


class CausalConv(Layer):
    elementwise = frozenset({"conv.depth.w"})

    def __init__(self, cfg, dims):
        super().__init__(dims)
        self.kernel = cfg.kernel
        self.add("conv.depth.w", (cfg.kernel, dims.d_model))
        self.add("conv.depth.b", (dims.d_model,), "bias")
        self.add("conv.proj.w", (dims.d_model, dims.d_model), "residual_out")
        self.add("conv.proj.b", (dims.d_model,), "bias")

    def forward(self, x, ctx):
        pre = ctx.emit("conv.pre", causal_conv(x, self.params["conv.depth.w"], self.params["conv.depth.b"]))
        act = ctx.emit("conv.act", F.silu(pre))
        return linear(self, act, "conv.proj")


@register
class Conv(ModuleSpec):
    id = "mixer/conv"
    slot = "mixer"
    name = "Causal convolution"
    blurb = "A local depthwise filter followed by a pointwise projection."
    ref = "causal depthwise convolution"
    config = ConvConfig
    provides = frozenset({"ordered"})
    hooks = (("conv.pre", "B T d"), ("conv.act", "B T d"))
    glyph3d = {"kind": "ribbon", "hooks": ["conv.act"]}

    def estimate(self, cfg, dims):
        d = dims.d_model
        return Estimate(params=d * cfg.kernel + d + d * d + d,
                        flops_per_token=2 * d * (cfg.kernel + d),
                        activation_bytes=3 * d * dims.bytes,
                        kv_bytes=(cfg.kernel - 1) * d * dims.bytes)

    def build(self, cfg, dims, rng):
        return CausalConv(cfg, dims)
