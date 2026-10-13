"""LayerNorm: subtract the mean, divide by the standard deviation, then scale and shift."""
from __future__ import annotations

from dataclasses import dataclass, field

from studio_registry import ModuleSpec, register
from studio_slots import Estimate, Layer


@dataclass
class LayerNormConfig:
    eps: float = field(default=1e-5, metadata=dict(min=1e-12, max=1e-2, help="Added to the variance so a constant input does not divide by zero."))
    placement: str = field(default="pre", metadata=dict(choices=["pre", "post"], help="pre: x + f(norm(x)), GPT-2 onward. post: norm(x + f(x)), the 2017 transformer and GPT-1."))


class LayerNorm(Layer):
    def __init__(self, cfg, dims):
        super().__init__(dims)
        self.eps = cfg.eps
        self.add("g", (dims.d_model,), "gain")
        self.add("b", (dims.d_model,), "bias")

    def forward(self, x, ctx):
        centered = x - x.mean(axis=-1, keepdims=True)
        inv = 1. / self.xp.sqrt((centered * centered).mean(axis=-1, keepdims=True) + self.eps)
        normalized = centered * inv
        self.cache = (normalized, inv)
        return normalized * self.params["g"] + self.params["b"]

    def backward(self, dy):
        normalized, inv = self.cache
        axes = tuple(range(dy.ndim - 1))
        self.grads["g"] += (dy * normalized).sum(axis=axes)
        self.grads["b"] += dy.sum(axis=axes)
        dx = dy * self.params["g"]
        return inv * (dx - dx.mean(axis=-1, keepdims=True) - normalized * (dx * normalized).mean(axis=-1, keepdims=True))


@register
class LayerNormSpec(ModuleSpec):
    id = "norm/layernorm"
    slot = "norm"
    name = "LayerNorm"
    blurb = "Subtract mean, divide by std, scale and shift."
    ref = "Ba et al. 2016"
    config = LayerNormConfig

    def estimate(self, cfg, dims):
        return Estimate(params=2 * dims.d_model, flops_per_token=4 * dims.d_model, activation_bytes=dims.d_model * dims.bytes)

    def build(self, cfg, dims, rng):
        return LayerNorm(cfg, dims)
