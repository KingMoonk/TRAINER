"""Causal long short-term memory."""
from __future__ import annotations

import torch

from studio_registry import ModuleSpec, register
from studio_slots import Estimate, Layer


class LSTMLayer(Layer):
    def __init__(self, dims):
        super().__init__(dims)
        d = dims.d_model
        self.add("lstm.x", (d, 4 * d))
        self.add("lstm.h", (d, 4 * d))
        self.add("lstm.bx", (4 * d,), "bias")
        self.add("lstm.bh", (4 * d,), "bias")

    def forward(self, x, ctx):
        p = self.params
        projected = x @ p["lstm.x"] + p["lstm.bx"]
        h = torch.zeros((x.shape[0], x.shape[2]), dtype=x.dtype, device=x.device)
        c = torch.zeros_like(h)
        states = []
        for t in range(x.shape[1]):
            raw = projected[:, t] + h @ p["lstm.h"] + p["lstm.bh"]
            i, f, g, o = raw.chunk(4, dim=-1)
            i, f, g, o = torch.sigmoid(i), torch.sigmoid(f), torch.tanh(g), torch.sigmoid(o)
            c = f * c + i * g
            h = o * torch.tanh(c)
            states.append(h)
        return ctx.emit("lstm.state", torch.stack(states, dim=1))


@register
class LSTM(ModuleSpec):
    id = "mixer/lstm"
    slot = "mixer"
    name = "LSTM"
    blurb = "A gated hidden state and cell state carry information across tokens."
    ref = "Hochreiter & Schmidhuber 1997"
    provides = frozenset({"ordered"})
    hooks = (("lstm.state", "B T d"),)
    glyph3d = {"kind": "ribbon", "hooks": ["lstm.state"]}

    def estimate(self, cfg, dims):
        d = dims.d_model
        return Estimate(params=4 * (2 * d * d + 2 * d), flops_per_token=16 * d * d,
                        activation_bytes=9 * d * dims.bytes, kv_bytes=2 * d * dims.bytes)

    def build(self, cfg, dims, rng):
        return LSTMLayer(dims)
