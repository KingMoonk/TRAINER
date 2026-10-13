"""Selective state-space mixer with input-dependent delta, B and C.

This is the transparent sequential Mamba-1 recurrence, implemented with NumPy/CuPy
operations and an explicit reverse scan. It does not claim a fused-kernel speedup.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from studio_modules.ops import activate, activate_derivative, linear, linear_back
from studio_registry import ModuleSpec, register
from studio_slots import Estimate, Layer


@dataclass
class SSMConfig:
    state: int = field(default=16, metadata=dict(min=1, max=128))
    expand: int = field(default=2, metadata=dict(min=1, max=8))
    kernel: int = field(default=4, metadata=dict(min=1, max=32))
    dt_rank: int = field(default=16, metadata=dict(min=1, max=128))


def softplus(x, xp):
    return xp.maximum(x, 0) + xp.log1p(xp.exp(-xp.abs(x)))


class SelectiveSSM(Layer):
    def __init__(self, cfg, dims):
        super().__init__(dims)
        d, e, n, rank = dims.d_model, cfg.expand * dims.d_model, cfg.state, cfg.dt_rank
        self.kernel, self.state_size, self.rank = cfg.kernel, n, rank
        self.add("ssm.in.w", (d, 2 * e))
        self.add("ssm.conv.w", (cfg.kernel, e))
        self.add("ssm.conv.b", (e,), "bias")
        self.add("ssm.x.w", (e, rank + 2 * n))
        self.add("ssm.dt.w", (rank, e))
        self.add("ssm.dt.b", (e,), value=self.xp.full((e,), math.log(math.expm1(.01)), dtype=self.dtype))
        a = self.xp.log(self.xp.arange(1, n + 1, dtype=self.dtype))
        self.add("ssm.A_log", (e, n), value=self.xp.tile(a, (e, 1)))
        self.add("ssm.D", (e,), value=self.xp.ones((e,), dtype=self.dtype))
        self.add("ssm.out.w", (e, d), "residual_out")

    def forward(self, x, ctx):
        xp = self.xp
        e, n = self.params["ssm.A_log"].shape
        projected = linear(self, x, "ssm.in")
        u, z = xp.split(projected, 2, axis=-1)
        pre = xp.broadcast_to(self.params["ssm.conv.b"], u.shape).copy()
        for lag in range(self.kernel):
            if lag >= u.shape[1]:
                break
            pre[:, lag:] += u[:, :u.shape[1] - lag] * self.params["ssm.conv.w"][lag]
        a, conv_sigmoid = activate(xp, pre, "silu")
        a = ctx.emit("ssm.conv", a)
        selected = linear(self, a, "ssm.x")
        rank, n = self.rank, self.state_size
        low_dt, b, c = xp.split(selected, (rank, rank + n), axis=-1)
        dt_pre = linear(self, low_dt, "ssm.dt")
        dt = softplus(dt_pre, xp)
        A = -xp.exp(self.params["ssm.A_log"])
        gate, gate_sigmoid = activate(xp, z, "silu")
        state = xp.zeros((x.shape[0], e, n), dtype=x.dtype)
        steps, outputs = [], []
        for t in range(x.shape[1]):
            previous = state
            alpha = xp.exp(dt[:, t, :, None] * A[None])
            state = previous * alpha + dt[:, t, :, None] * a[:, t, :, None] * b[:, t, None, :]
            state = ctx.emit("ssm.state", state)
            y = xp.sum(state * c[:, t, None, :], axis=-1) + a[:, t] * self.params["ssm.D"]
            outputs.append(y * gate[:, t])
            steps.append((previous, state, alpha, y))
        inner = ctx.emit("ssm.out", xp.stack(outputs, axis=1))
        self.cache = (x, u, z, pre, conv_sigmoid, a, low_dt, b, c, dt_pre, dt, A,
                      gate, gate_sigmoid, steps, inner)
        return linear(self, inner, "ssm.out")

    def backward(self, dy):
        xp = self.xp
        (x, u, z, pre, conv_sigmoid, a, low_dt, b, c, dt_pre, dt, A,
         gate, gate_sigmoid, steps, inner) = self.cache
        dinner = linear_back(self, dy, inner, "ssm.out")
        da = xp.zeros_like(a)
        db, dc = xp.zeros_like(b), xp.zeros_like(c)
        ddt = xp.zeros_like(dt)
        dz = xp.zeros_like(z)
        dA = xp.zeros_like(A)
        dstate = xp.zeros_like(steps[0][0])
        for t in reversed(range(x.shape[1])):
            previous, state, alpha, y = steps[t]
            dy_t = dinner[:, t] * gate[:, t]
            dz[:, t] = dinner[:, t] * y * activate_derivative(z[:, t], gate_sigmoid[:, t], "silu")
            dstate = dstate + dy_t[:, :, None] * c[:, t, None, :]
            dc[:, t] = xp.sum(dy_t[:, :, None] * state, axis=1)
            self.grads["ssm.D"] += xp.sum(dy_t * a[:, t], axis=0)
            da[:, t] += dy_t * self.params["ssm.D"]
            dalpha = dstate * previous
            dA += xp.sum(dalpha * alpha * dt[:, t, :, None], axis=0)
            ddt[:, t] = xp.sum(dalpha * alpha * A[None]
                                  + dstate * a[:, t, :, None] * b[:, t, None, :], axis=-1)
            da[:, t] += xp.sum(dstate * dt[:, t, :, None] * b[:, t, None, :], axis=-1)
            db[:, t] = xp.sum(dstate * dt[:, t, :, None] * a[:, t, :, None], axis=1)
            dstate = dstate * alpha
        self.grads["ssm.A_log"] += dA * A
        ddt_pre = ddt / (1 + xp.exp(-dt_pre))  # softplus' sigmoid derivative
        dlow_dt = linear_back(self, ddt_pre, low_dt, "ssm.dt")
        selected_grad = xp.concatenate((dlow_dt, db, dc), axis=-1)
        da += linear_back(self, selected_grad, a, "ssm.x")
        dpre = da * activate_derivative(pre, conv_sigmoid, "silu")
        self.grads["ssm.conv.b"] += dpre.sum(axis=(0, 1))
        du = xp.zeros_like(u)
        for lag in range(self.kernel):
            if lag >= u.shape[1]:
                break
            self.grads["ssm.conv.w"][lag] += (u[:, :u.shape[1] - lag] * dpre[:, lag:]).sum(axis=(0, 1))
            du[:, :u.shape[1] - lag] += dpre[:, lag:] * self.params["ssm.conv.w"][lag]
        return linear_back(self, xp.concatenate((du, dz), axis=-1), x, "ssm.in")


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
