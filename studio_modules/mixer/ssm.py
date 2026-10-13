"""Selective state-space mixer with input-dependent delta, B and C.

This is the transparent sequential Mamba-1 recurrence, written as a plain scan over positions.
It does not claim a fused-kernel speedup.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from studio_modules.ops import causal_conv, linear
from studio_registry import ModuleSpec, register
from studio_slots import Estimate, Layer


@dataclass
class SSMConfig:
    state: int = field(default=16, metadata=dict(min=1, max=128))
    expand: int = field(default=2, metadata=dict(min=1, max=8))
    kernel: int = field(default=4, metadata=dict(min=1, max=32))
    dt_rank: int = field(default=16, metadata=dict(min=1, max=128))


class SelectiveSSM(Layer):
    elementwise = frozenset({"ssm.conv.w"})

    def __init__(self, cfg, dims):
        super().__init__(dims)
        d, e, n, rank = dims.d_model, cfg.expand * dims.d_model, cfg.state, cfg.dt_rank
        self.kernel, self.state_size, self.rank = cfg.kernel, n, rank
        self.add("ssm.in.w", (d, 2 * e))
        self.add("ssm.conv.w", (cfg.kernel, e))
        self.add("ssm.conv.b", (e,), "bias")
        self.add("ssm.x.w", (e, rank + 2 * n))
        self.add("ssm.dt.w", (rank, e))
        self.add("ssm.dt.b", (e,), value=torch.full((e,), math.log(math.expm1(.01)), dtype=self.dtype))
        a = torch.log(torch.arange(1, n + 1, dtype=self.dtype))
        self.add("ssm.A_log", (e, n), value=a.repeat(e, 1))
        self.add("ssm.D", (e,), value=torch.ones((e,), dtype=self.dtype))
        self.add("ssm.out.w", (e, d), "residual_out")

    def forward(self, x, ctx):
        p = self.params
        e, n = p["ssm.A_log"].shape
        u, z = linear(self, x, "ssm.in").chunk(2, dim=-1)
        a = ctx.emit("ssm.conv", F.silu(causal_conv(u, p["ssm.conv.w"], p["ssm.conv.b"])))
        low_dt, b, c = torch.split(linear(self, a, "ssm.x"), (self.rank, n, n), dim=-1)
        dt = F.softplus(linear(self, low_dt, "ssm.dt"))
        A = -torch.exp(p["ssm.A_log"])
        gate = F.silu(z)
        state = torch.zeros((x.shape[0], e, n), dtype=x.dtype, device=x.device)
        outputs = []
        for t in range(x.shape[1]):
            alpha = torch.exp(dt[:, t, :, None] * A[None])
            state = state * alpha + dt[:, t, :, None] * a[:, t, :, None] * b[:, t, None, :]
            state = ctx.emit("ssm.state", state)
            y = (state * c[:, t, None, :]).sum(dim=-1) + a[:, t] * p["ssm.D"]
            outputs.append(y * gate[:, t])
        inner = ctx.emit("ssm.out", torch.stack(outputs, dim=1))
        return linear(self, inner, "ssm.out")


@register
class SSM(ModuleSpec):
    id = "mixer/ssm"
    slot = "mixer"
    name = "Selective state space"
    blurb = "Input-dependent state update with causal convolution and a gated output."
    ref = "Gu & Dao 2024"
    config = SSMConfig
    provides = frozenset({"ordered"})
    hooks = (("ssm.conv", "B T E"), ("ssm.state", "B E N"), ("ssm.out", "B T E"))
    glyph3d = {"kind": "ribbon", "hooks": ["ssm.state"]}

    def symbols(self, cfg, dims):
        return {"E": cfg.expand * dims.d_model, "N": cfg.state}

    def estimate(self, cfg, dims):
        d, e, n, rank = dims.d_model, cfg.expand * dims.d_model, cfg.state, cfg.dt_rank
        params = d * 2 * e + e * cfg.kernel + e + e * (rank + 2 * n) + rank * e + e + e * n + e + e * d
        return Estimate(params=params, flops_per_token=2 * params + 4 * e * n,
                        activation_bytes=(8 * e + 3 * e * n) * dims.bytes,
                        kv_bytes=(e * n + (cfg.kernel - 1) * e) * dims.bytes)

    def build(self, cfg, dims, rng):
        return SelectiveSSM(cfg, dims)
