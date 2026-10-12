"""Causal long short-term memory with explicit reverse-time gradients."""
from __future__ import annotations

from studio_modules.mixer.gru import sigmoid
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
        xp = self.xp
        h = xp.zeros((x.shape[0], x.shape[2]), dtype=x.dtype)
        c = xp.zeros_like(h)
        states, saved = [], []
        for t in range(x.shape[1]):
            xt, hp, cp = x[:, t], h, c
            raw = xt @ self.params["lstm.x"] + self.params["lstm.bx"] + hp @ self.params["lstm.h"] + self.params["lstm.bh"]
            i, f, g, o = xp.split(raw, 4, axis=-1)
            i, f, g, o = sigmoid(i, xp), sigmoid(f, xp), xp.tanh(g), sigmoid(o, xp)
            c = f * cp + i * g
            h = o * xp.tanh(c)
            states.append(h)
            saved.append((xt, hp, cp, i, f, g, o, c))
        out = ctx.emit("lstm.state", xp.stack(states, axis=1))
        self.cache = saved
        return out

    def backward(self, dy):
        xp = self.xp
        dx = xp.empty_like(dy)
        dh = xp.zeros_like(dy[:, 0])
        dc = xp.zeros_like(dh)
        for t in reversed(range(dy.shape[1])):
            xt, hp, cp, i, f, g, o, c = self.cache[t]
            total = dy[:, t] + dh
            tanh_c = xp.tanh(c)
            do = total * tanh_c * o * (1 - o)
            dc = dc + total * o * (1 - tanh_c * tanh_c)
            di = dc * g * i * (1 - i)
            df = dc * cp * f * (1 - f)
            dg = dc * i * (1 - g * g)
            draw = xp.concatenate((di, df, dg, do), axis=-1)
            self.grads["lstm.x"] += xt.T @ draw
            self.grads["lstm.h"] += hp.T @ draw
            self.grads["lstm.bx"] += draw.sum(axis=0)
            self.grads["lstm.bh"] += draw.sum(axis=0)
            dx[:, t] = draw @ self.params["lstm.x"].T
            dh = draw @ self.params["lstm.h"].T
            dc = dc * f
        return dx


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
