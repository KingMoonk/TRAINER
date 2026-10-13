"""Fixed sinusoidal position vectors added to the residual stream."""
from __future__ import annotations

import numpy as np

from studio_registry import ModuleSpec, register
from studio_slots import Estimate, Layer


class SinusoidalPosition(Layer):
    def __init__(self, dims):
        super().__init__(dims)
        positions = np.arange(dims.n_ctx, dtype=np.float64)[:, None]
        channels = np.arange(dims.d_model, dtype=np.float64)[None, :]
        frequencies = np.exp(-(np.floor(channels / 2) * (2.0 / dims.d_model)) * np.log(10000.0))
        angles = positions * frequencies
        encoding = np.where((channels.astype(np.int64) % 2) == 0, np.sin(angles), np.cos(angles))
        self.encoding = self.xp.asarray(encoding, dtype=self.dtype)

    def forward(self, x, ctx):
        length = x.shape[1]
        if length > self.encoding.shape[0]:
            raise ValueError(f"Sequence length {length} exceeds positional capacity {self.encoding.shape[0]}.")
        return x + self.encoding[:length]

    def backward(self, dx):
        return dx


@register
class Sinusoidal(ModuleSpec):
    id = "position/sinusoidal"
    slot = "position"
    name = "Sinusoidal"
    blurb = "Fixed sine and cosine position vectors, added to the stream."
    ref = "Vaswani et al. 2017"

    def estimate(self, cfg, dims):
        return Estimate(activation_bytes=dims.d_model * dims.bytes)

    def build(self, cfg, dims, rng):
        return SinusoidalPosition(dims)
