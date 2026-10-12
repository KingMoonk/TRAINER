"""Top-k mixture of biased GELU MLP experts with differentiable router weights."""
from __future__ import annotations

from dataclasses import dataclass, field

from studio_modules.ops import activate, activate_derivative, softmax
from studio_registry import ModuleSpec, register
from studio_slots import Estimate, Layer


@dataclass
class MoEConfig:
    experts: int = field(default=8, metadata=dict(min=2, max=32))
    top_k: int = field(default=2, metadata=dict(min=1, max=8))
    d_ff: int = field(default=0, metadata=dict(min=0, max=8192))
    aux_weight: float = field(default=.01, metadata=dict(min=0, max=1))


def width(cfg, dims):
    return cfg.d_ff or 4 * dims.d_model


class MoELayer(Layer):
    def __init__(self, cfg, dims):
        super().__init__(dims)
        self.experts, self.top_k, self.aux_weight = cfg.experts, cfg.top_k, cfg.aux_weight
        d, f = dims.d_model, width(cfg, dims)
        self.add("moe.router.w", (d, cfg.experts))
        for expert in range(cfg.experts):
            prefix = f"moe.expert{expert}"
            self.add(prefix + ".f1.w", (d, f))
            self.add(prefix + ".f1.b", (f,), "bias")
            self.add(prefix + ".f2.w", (f, d), "residual_out")
            self.add(prefix + ".f2.b", (d,), "bias")

    def forward(self, x, ctx):
        xp = self.xp
        batch, length, d = x.shape
        flat = x.reshape(-1, d)
        logits = ctx.emit("moe.router.logits", x @ self.params["moe.router.w"])
        probabilities = ctx.emit("moe.router.prob", softmax(logits, xp))
        top = ctx.emit("moe.router.top", xp.argsort(-probabilities, axis=-1)[..., :self.top_k])
        selected = xp.take_along_axis(probabilities, top, axis=-1)
        weights = selected / xp.maximum(selected.sum(axis=-1, keepdims=True), 1e-12)
        chosen = top.reshape(-1, self.top_k)
        weights = weights.reshape(-1, self.top_k)
        output = xp.zeros_like(flat)
        saved = []
        for expert in range(self.experts):
            locations = xp.flatnonzero(xp.any(chosen == expert, axis=-1))
            if not locations.size:
                saved.append(None)
                continue
            prefix = f"moe.expert{expert}"
            inputs = flat[locations]
            pre = inputs @ self.params[prefix + ".f1.w"] + self.params[prefix + ".f1.b"]
            act, derivative = activate(xp, pre, "gelu")
            values = act @ self.params[prefix + ".f2.w"] + self.params[prefix + ".f2.b"]
            weight = xp.sum(xp.where(chosen[locations] == expert, weights[locations], 0), axis=-1)
            output[locations] += weight[:, None] * values
            saved.append((locations, inputs, pre, derivative, act, values, weight))
        load = xp.stack([(chosen == expert).sum() for expert in range(self.experts)]).astype(x.dtype)
        load /= chosen.shape[0] * self.top_k
        if self.aux_weight:
            ctx.aux += float(self.aux_weight * self.experts * xp.sum(probabilities.mean(axis=(0, 1)) * load))
        self.cache = (x, flat, probabilities.reshape(-1, self.experts), chosen, weights, load, saved, ctx)
        return output.reshape(batch, length, d)

    def backward(self, dy):
        xp = self.xp
        x, flat, probabilities, chosen, weights, load, saved, ctx = self.cache
        dflat = xp.zeros_like(flat)
        dweights = xp.zeros_like(weights)
        upstream = dy.reshape(flat.shape)
        for expert, item in enumerate(saved):
            if item is None:
                continue
            locations, inputs, pre, derivative, act, values, weight = item
            prefix = f"moe.expert{expert}"
            dvalue = upstream[locations] * weight[:, None]
            self.grads[prefix + ".f2.w"] += act.T @ dvalue
            self.grads[prefix + ".f2.b"] += dvalue.sum(axis=0)
            dact = dvalue @ self.params[prefix + ".f2.w"].T
            dpre = dact * activate_derivative(pre, derivative, "gelu")
            self.grads[prefix + ".f1.w"] += inputs.T @ dpre
            self.grads[prefix + ".f1.b"] += dpre.sum(axis=0)
            dflat[locations] += dpre @ self.params[prefix + ".f1.w"].T
            dweight = xp.sum(upstream[locations] * values, axis=-1)
            for route in range(self.top_k):
                matching = chosen[locations, route] == expert
                dweights[locations, route] += xp.where(matching, dweight, 0)
        centered = dweights - xp.sum(dweights * weights, axis=-1, keepdims=True)
        dselected = weights * centered
        dlogits = xp.zeros_like(probabilities)
        rows = xp.arange(chosen.shape[0])[:, None]
        xp.add.at(dlogits, (xp.broadcast_to(rows, chosen.shape), chosen), dselected)
        if self.aux_weight:
            average_load = xp.sum(probabilities * load[None], axis=-1, keepdims=True)
            dlogits += (ctx.scale * self.aux_weight * self.experts / chosen.shape[0]
                        * probabilities * (load[None] - average_load))
        self.grads["moe.router.w"] += flat.T @ dlogits
        dflat += dlogits @ self.params["moe.router.w"].T
        return dflat.reshape(x.shape)


@register
class MoE(ModuleSpec):
    id = "channel/moe"
    slot = "channel"
    name = "Top-k mixture of experts"
    blurb = "A router selects k GELU MLP experts per token and adds a load-balancing loss."
    ref = "Shazeer et al. 2017; Fedus et al. 2022"
    config = MoEConfig
    hooks = (("moe.router.logits", "B T X"), ("moe.router.prob", "B T X"),
             ("moe.router.top", "B T K"))
    glyph3d = {"kind": "fan", "hooks": ["moe.router.prob"]}

    def symbols(self, cfg, dims):
        return {"X": cfg.experts, "K": cfg.top_k}

    def estimate(self, cfg, dims):
        d, f = dims.d_model, width(cfg, dims)
        expert = 2 * d * f + f + d
        router = d * cfg.experts
        return Estimate(params=router + cfg.experts * expert,
                        active_params=router + cfg.top_k * expert,
                        flops_per_token=2 * (router + cfg.top_k * expert),
                        activation_bytes=(cfg.top_k * 2 * f + cfg.experts + d) * dims.bytes)

    def validate(self, cfg, dims):
        from studio_slots import Issue
        if cfg.top_k > cfg.experts:
            return [Issue("ERROR", "MoE top_k cannot exceed the number of experts.")]
        return []

    def build(self, cfg, dims, rng):
        return MoELayer(cfg, dims)
