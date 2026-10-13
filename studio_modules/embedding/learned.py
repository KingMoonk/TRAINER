"""Learned token embedding: one trainable d-vector per vocabulary entry."""
from __future__ import annotations

from studio_registry import ModuleSpec, register
from studio_slots import Estimate, Layer


class LearnedEmbedding(Layer):
    def __init__(self, dims):
        super().__init__(dims)
        self.add("token", (dims.vocab, dims.d_model), "embedding")

    def forward(self, ids, ctx):
        return self.params["token"][ids]

    # a tied output head reads the stream back out through the same table
    def unembed(self, hidden):
        return hidden @ self.params["token"].t()


@register
class Learned(ModuleSpec):
    id = "embedding/learned"
    slot = "embedding"
    name = "Learned table"
    blurb = "One trainable d-vector per vocabulary entry."
    ref = "standard"
    provides = frozenset({"unembed"})

    def estimate(self, cfg, dims):
        return Estimate(params=dims.vocab * dims.d_model, activation_bytes=dims.d_model * dims.bytes)

    def build(self, cfg, dims, rng):
        return LearnedEmbedding(dims)
