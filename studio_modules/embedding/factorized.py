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
        return self.params["token.low"][ids] @ self.params["token.up"]

    def unembed(self, hidden):
        return (hidden @ self.params["token.up"].t()) @ self.params["token.low"].t()


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
