"""Learned position embedding: one trainable vector per position, added to the stream."""
from __future__ import annotations

from studio_registry import ModuleSpec, register
from studio_slots import Estimate, Layer


class LearnedPosition(Layer):
    cacheable = True

    def __init__(self, dims):
        super().__init__(dims)
        self.add("position", (dims.n_ctx, dims.d_model), "embedding")

    def forward(self, x, ctx):
        return x + self.params["position"][ctx.offset:ctx.offset + x.shape[1]]


@register
class Learned(ModuleSpec):
    id = "position/learned"
    slot = "position"
    name = "Learned"
    blurb = "A trainable vector per position, added to x. Fixed maximum length."
    ref = "GPT-2"

    def estimate(self, cfg, dims):
        return Estimate(params=dims.n_ctx * dims.d_model)

    def build(self, cfg, dims, rng):
        return LearnedPosition(dims)
