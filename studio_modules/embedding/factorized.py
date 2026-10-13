"""Factorized token embedding with a compatible tied unembedding."""
from __future__ import annotations

from dataclasses import dataclass, field

from studio_registry import ModuleSpec, register
from studio_slots import Estimate, Layer


@dataclass
class FactorizedConfig:
    rank: int = field(default=0, metadata=dict(min=0, max=2048, help="Embedding rank; 0 uses one quarter of d_model."))


def rank_of(cfg, dims):
    return cfg.rank or max(1, dims.d_model // 4)


class FactorizedEmbedding(Layer):
    def __init__(self, cfg, dims):
        super().__init__(dims)
        rank = rank_of(cfg, dims)
        self.add("token.low", (dims.vocab, rank), "embedding")
        self.add("token.up", (rank, dims.d_model), "weight")

    def forward(self, ids, ctx):
        self.cache = ids
        return self.params["token.low"][ids] @ self.params["token.up"]

    def backward(self, dx):
        ids = self.cache
        low = self.params["token.low"][ids]
        self.grads["token.up"] += low.reshape(-1, low.shape[-1]).T @ dx.reshape(-1, dx.shape[-1])
        self.xp.add.at(self.grads["token.low"], ids.reshape(-1),
                       (dx @ self.params["token.up"].T).reshape(-1, low.shape[-1]))
        return None

    def unembed(self, hidden):
        return (hidden @ self.params["token.up"].T) @ self.params["token.low"].T

    def unembed_back(self, dlogits, hidden):
        middle = hidden @ self.params["token.up"].T
        self.grads["token.low"] += dlogits.reshape(-1, dlogits.shape[-1]).T @ middle.reshape(-1, middle.shape[-1])
        dmiddle = dlogits @ self.params["token.low"]
        self.grads["token.up"] += dmiddle.reshape(-1, dmiddle.shape[-1]).T @ hidden.reshape(-1, hidden.shape[-1])
        return dmiddle @ self.params["token.up"]


@register
class Factorized(ModuleSpec):
    id = "embedding/factorized"
    slot = "embedding"
    name = "Factorized table"
    blurb = "A rank-r token table projected to model width; the tied head reuses both factors."
    ref = "ALBERT, Lan et al. 2020"
    config = FactorizedConfig
    provides = frozenset({"unembed"})
    glyph3d = {"kind": "fan", "hooks": ["embed"]}

    def estimate(self, cfg, dims):
        rank = rank_of(cfg, dims)
        return Estimate(params=dims.vocab * rank + rank * dims.d_model,
                        flops_per_token=2 * rank * dims.d_model,
                        activation_bytes=(rank + dims.d_model) * dims.bytes)

    def build(self, cfg, dims, rng):
        return FactorizedEmbedding(cfg, dims)
