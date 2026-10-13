"""Root mean square normalization with a learned scale."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from studio_registry import ModuleSpec, register
from studio_slots import Estimate, Layer


@dataclass
class RMSNormConfig:
    eps: float = field(default=1e-5, metadata=dict(min=1e-12, max=1e-2, help="Added to the mean square for numerical stability."))
    placement: str = field(default="pre", metadata=dict(choices=["pre", "post"], help="pre: x + f(norm(x)); post: norm(x + f(x))."))


class RMSNorm(Layer):
    def __init__(self, cfg, dims):
        super().__init__(dims)
        self.eps = cfg.eps
        self.add("g", (dims.d_model,), "gain")

    def forward(self, x, ctx):
        inv = 1.0 / torch.sqrt((x * x).mean(dim=-1, keepdim=True) + self.eps)
        return x * inv * self.params["g"]


@register
class RMSNormSpec(ModuleSpec):
    id = "norm/rmsnorm"
    slot = "norm"
    name = "RMSNorm"
    blurb = "Scale each token by its root mean square, then apply a learned gain."
    ref = "Zhang & Sennrich 2019"
    config = RMSNormConfig

    def estimate(self, cfg, dims):
        return Estimate(params=dims.d_model, flops_per_token=3 * dims.d_model, activation_bytes=2 * dims.d_model * dims.bytes)

    def build(self, cfg, dims, rng):
        return RMSNorm(cfg, dims)
