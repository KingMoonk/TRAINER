"""Attention with linear biases: head-specific penalties for distant keys."""
from __future__ import annotations

import math

from studio_registry import ModuleSpec, register
from studio_slots import Estimate, Layer


def slopes(heads):
    """The paper's geometric slopes, including non-power-of-two head counts."""
    def powers(count):
        start = 2 ** (-2 ** -(math.log2(count) - 3))
        return [start ** (i + 1) for i in range(count)]

    power = 2 ** int(math.floor(math.log2(heads)))
    if power == heads:
        return powers(heads)
    return powers(power) + slopes(2 * power)[0::2][:heads - power]


class ALiBiPosition(Layer):
    def forward(self, x, ctx):
        return x

    def backward(self, dx):
        return dx

    def score_bias(self, length, heads):
        return self.score_bias_rows(0, length, length, heads)

    def score_bias_rows(self, start, end, length, heads):
        xp = self.xp
        queries = xp.arange(start, end, dtype=self.dtype)
        keys = xp.arange(length, dtype=self.dtype)
        distance = queries[:, None] - keys[None, :]
        rate = xp.asarray(slopes(heads), dtype=self.dtype)[:, None, None]
        return -rate * distance[None, :, :]


@register
class ALiBi(ModuleSpec):
    id = "position/alibi"
    slot = "position"
    name = "ALiBi"
    blurb = "Adds a head-specific linear distance penalty to attention scores."
    ref = "Press et al. 2022"
    requires = frozenset({"scores"})
    glyph3d = {"kind": "ribbon", "hooks": ["attn.scores"]}

    def estimate(self, cfg, dims):
        return Estimate()

    def build(self, cfg, dims, rng):
        return ALiBiPosition(dims)
