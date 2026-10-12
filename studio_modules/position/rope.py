"""Rotary position embedding on each query/key head's adjacent channel pairs."""
from __future__ import annotations

from dataclasses import dataclass, field

from studio_registry import ModuleSpec, register
from studio_slots import Estimate, Layer


@dataclass
class RoPEConfig:
    base: float = field(default=10000.0, metadata=dict(min=2.0, max=1000000.0))


class RotaryPosition(Layer):
    def __init__(self, cfg, dims):
        super().__init__(dims)
        self.base = cfg.base

    def forward(self, x, ctx):
        return x

    def backward(self, dx):
        return dx

    def _angles(self, x):
        depth = x.shape[-1]
        xp = self.xp
        positions = xp.arange(x.shape[-2], dtype=self.dtype)
        inv = self.base ** (-2 * xp.arange(depth // 2, dtype=self.dtype) / depth)
        angle = positions[:, None] * inv[None, :]
        return xp.cos(angle)[None, None], xp.sin(angle)[None, None]

    def _turn(self, x, reverse=False):
        if x.shape[-1] % 2:
            raise ValueError("RoPE requires an even query/key head width.")
        cosine, sine = self._angles(x)
        if reverse:
            sine = -sine
        even, odd = x[..., 0::2], x[..., 1::2]
        return self.xp.stack((even * cosine - odd * sine, even * sine + odd * cosine), axis=-1).reshape(x.shape)

    def rotate(self, q, k):
        return self._turn(q), self._turn(k)

    def rotate_back(self, dq, dk):
        return self._turn(dq, True), self._turn(dk, True)


@register
class RoPE(ModuleSpec):
    id = "position/rope"
    slot = "position"
    name = "Rotary (RoPE)"
    blurb = "Rotates each query and key pair by its token position; adds no learned weights."
    ref = "Su et al. 2021"
    config = RoPEConfig
    requires = frozenset({"qk"})
    glyph3d = {"kind": "ribbon", "hooks": ["attn.q", "attn.k"]}

    def estimate(self, cfg, dims):
        return Estimate()

    def validate(self, cfg, dims):
        # The position spec cannot see each block's head count; the mixer validates its own depth.
        return []

    def build(self, cfg, dims, rng):
        return RotaryPosition(cfg, dims)
