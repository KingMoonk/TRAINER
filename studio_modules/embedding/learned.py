"""Learned token embedding: one trainable d-vector per vocabulary entry."""
from __future__ import annotations

from studio_registry import ModuleSpec, register
from studio_slots import Estimate, Layer


class LearnedEmbedding(Layer):
    def __init__(self, dims):
        super().__init__(dims)
        self.add("token", (dims.vocab, dims.d_model), "embedding")

    def forward(self, ids, ctx):
        self.cache = ids
        return self.params["token"][ids]

    def backward(self, dx):
        ids = self.cache
        self.xp.add.at(self.grads["token"], ids.reshape(-1), dx.reshape(-1, dx.shape[-1]))  # a token used twice gets both gradients
        return None

    # a tied output head reads the stream back out through the same table
    def unembed(self, hidden):
        return hidden @ self.params["token"].T

    def unembed_back(self, dlogits, hidden):
        width = hidden.shape[-1]
        self.grads["token"] += dlogits.reshape(-1, dlogits.shape[-1]).T @ hidden.reshape(-1, width)
        return dlogits @ self.params["token"]


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
