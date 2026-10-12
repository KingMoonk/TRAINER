"""Causal gated recurrent unit with explicit reverse-time gradients."""
from __future__ import annotations

from studio_registry import ModuleSpec, register
from studio_slots import Estimate, Layer


def sigmoid(x, xp):
    return 1 / (1 + xp.exp(-x))


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
        xp = self.xp
        h = xp.zeros((x.shape[0], x.shape[2]), dtype=x.dtype)
        states, saved = [], []
        for t in range(x.shape[1]):
            xt, previous = x[:, t], h
            z = sigmoid(xt @ self.params["gru.z.x"] + self.params["gru.z.bx"]
                        + previous @ self.params["gru.z.h"] + self.params["gru.z.bh"], xp)
            r = sigmoid(xt @ self.params["gru.r.x"] + self.params["gru.r.bx"]
                        + previous @ self.params["gru.r.h"] + self.params["gru.r.bh"], xp)
            nh = previous @ self.params["gru.n.h"] + self.params["gru.n.bh"]
            n = xp.tanh(xt @ self.params["gru.n.x"] + self.params["gru.n.bx"] + r * nh)
            h = (1 - z) * n + z * previous
            states.append(h)
            saved.append((xt, previous, z, r, n, nh))
        out = ctx.emit("gru.state", xp.stack(states, axis=1))
        self.cache = saved
        return out

    def backward(self, dy):
        xp = self.xp
        dx = xp.empty_like(dy)
        dh = xp.zeros_like(dy[:, 0])
        for t in reversed(range(dy.shape[1])):
            xt, previous, z, r, n, nh = self.cache[t]
            total = dy[:, t] + dh
            dn = total * (1 - z)
            dz = total * (previous - n) * z * (1 - z)
            dh = total * z
            dnp = dn * (1 - n * n)
            dr = dnp * nh * r * (1 - r)
            dnh = dnp * r
            dx_t = xp.zeros_like(xt)
            for gate, derivative in (("z", dz), ("r", dr), ("n", dnp)):
                self.grads[f"gru.{gate}.x"] += xt.T @ derivative
                self.grads[f"gru.{gate}.bx"] += derivative.sum(axis=0)
                dx_t += derivative @ self.params[f"gru.{gate}.x"].T
            for gate, derivative in (("z", dz), ("r", dr), ("n", dnh)):
                self.grads[f"gru.{gate}.h"] += previous.T @ derivative
                self.grads[f"gru.{gate}.bh"] += derivative.sum(axis=0)
                dh += derivative @ self.params[f"gru.{gate}.h"].T
            dx[:, t] = dx_t
        return dx


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
