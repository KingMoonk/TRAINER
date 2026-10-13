"""Causal gated recurrent unit."""
from __future__ import annotations

import torch

from studio_registry import ModuleSpec, register
from studio_slots import Estimate, Layer


class GRULayer(Layer):
    def __init__(self, dims):
        super().__init__(dims)
        d = dims.d_model
        for gate in ("z", "r", "n"):
            self.add(f"gru.{gate}.x", (d, d))
            self.add(f"gru.{gate}.h", (d, d))
            self.add(f"gru.{gate}.bx", (d,), "bias")
            self.add(f"gru.{gate}.bh", (d,), "bias")

    def forward(self, x, ctx):
        p = self.params
        # the input projections for every position at once; only the recurrence runs step by step
        xz = x @ p["gru.z.x"] + p["gru.z.bx"]
        xr = x @ p["gru.r.x"] + p["gru.r.bx"]
        xn = x @ p["gru.n.x"] + p["gru.n.bx"]
        h = torch.zeros((x.shape[0], x.shape[2]), dtype=x.dtype, device=x.device)
        states = []
        for t in range(x.shape[1]):
            z = torch.sigmoid(xz[:, t] + h @ p["gru.z.h"] + p["gru.z.bh"])
            r = torch.sigmoid(xr[:, t] + h @ p["gru.r.h"] + p["gru.r.bh"])
            n = torch.tanh(xn[:, t] + r * (h @ p["gru.n.h"] + p["gru.n.bh"]))
            h = (1 - z) * n + z * h
            states.append(h)
        return ctx.emit("gru.state", torch.stack(states, dim=1))


@register
class GRU(ModuleSpec):
    id = "mixer/gru"
    slot = "mixer"
    name = "GRU"
    blurb = "A gated recurrent state passes information forward through tokens."
    ref = "Cho et al. 2014"
    provides = frozenset({"ordered"})
    hooks = (("gru.state", "B T d"),)
    glyph3d = {"kind": "ribbon", "hooks": ["gru.state"]}

    def estimate(self, cfg, dims):
        d = dims.d_model
        return Estimate(params=3 * (2 * d * d + 2 * d), flops_per_token=12 * d * d,
                        activation_bytes=7 * d * dims.bytes, kv_bytes=d * dims.bytes)

    def build(self, cfg, dims, rng):
        return GRULayer(dims)
