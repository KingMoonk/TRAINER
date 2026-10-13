"""Tied output head: reuse the embedding table, transposed, to read logits out of the stream."""
from __future__ import annotations

from studio_registry import ModuleSpec, register
from studio_slots import Layer


class TiedHead(Layer):
    def __init__(self, dims):
        super().__init__(dims)
        self.logit_scale = 1.0  # μP: set by init/mup if width_mult != 1

    def forward(self, x, ctx):
        self.cache = (ctx.embedding, x)
        return ctx.embedding.unembed(x) * self.logit_scale

    def backward(self, dlogits):
        embedding, x = self.cache
        return embedding.unembed_back(dlogits * self.logit_scale, x)  # the gradient lands in the embedding's own table


@register
class Tied(ModuleSpec):
    id = "head/tied"
    slot = "head"
    name = "Tied to embedding"
    blurb = "Reuses the embedding matrix, transposed. No new parameters."
    ref = "Press & Wolf 2017"
    requires = frozenset({"unembed"})  # of the embedding

    def build(self, cfg, dims, rng):
        return TiedHead(dims)
