"""Causal depthwise convolution followed by a pointwise output projection."""
from __future__ import annotations

from dataclasses import dataclass, field

from studio_modules.ops import activate, activate_derivative, linear, linear_back
from studio_registry import ModuleSpec, register
from studio_slots import Estimate, Layer


@dataclass
class ConvConfig:
    kernel: int = field(default=4, metadata=dict(min=1, max=32))


class CausalConv(Layer):
    def __init__(self, cfg, dims):
        super().__init__(dims)
        self.kernel = cfg.kernel
        self.add("conv.depth.w", (cfg.kernel, dims.d_model))
        self.add("conv.depth.b", (dims.d_model,), "bias")
        self.add("conv.proj.w", (dims.d_model, dims.d_model), "residual_out")
        self.add("conv.proj.b", (dims.d_model,), "bias")

    def forward(self, x, ctx):
        pre = self.xp.broadcast_to(self.params["conv.depth.b"], x.shape).copy()
        for lag in range(self.kernel):
            if lag >= x.shape[1]:
                break
            pre[:, lag:] += x[:, :x.shape[1] - lag] * self.params["conv.depth.w"][lag]
        pre = ctx.emit("conv.pre", pre)
        act, saved = activate(self.xp, pre, "silu")
        act = ctx.emit("conv.act", act)
        self.cache = (x, pre, saved, act)
        return linear(self, act, "conv.proj")

    def backward(self, dy):
        x, pre, saved, act = self.cache
        dpre = linear_back(self, dy, act, "conv.proj") * activate_derivative(pre, saved, "silu")
        self.grads["conv.depth.b"] += dpre.sum(axis=(0, 1))
        dx = self.xp.zeros_like(x)
        for lag in range(self.kernel):
            if lag >= x.shape[1]:
                break
            self.grads["conv.depth.w"][lag] += (x[:, :x.shape[1] - lag] * dpre[:, lag:]).sum(axis=(0, 1))
            dx[:, :x.shape[1] - lag] += dpre[:, lag:] * self.params["conv.depth.w"][lag]
        return dx


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
