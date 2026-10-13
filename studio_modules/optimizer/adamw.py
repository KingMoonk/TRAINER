"""AdamW: adaptive per-parameter step sizes with decoupled weight decay."""
from __future__ import annotations

from dataclasses import dataclass, field

from studio_registry import ModuleSpec, register
from studio_slots import Optimizer


@dataclass
class AdamWConfig:
    lr: float = field(default=1e-3, metadata=dict(min=1e-7, max=.1, help="Peak learning rate; the schedule scales it."))
    beta1: float = field(default=.9, metadata=dict(min=0, max=.9999, help="Memory of the gradient average."))
    beta2: float = field(default=.999, metadata=dict(min=0, max=.99999, help="Memory of the squared-gradient average."))
    weight_decay: float = field(default=.01, metadata=dict(min=0, max=1, help="Shrinks weight matrices a little every step. Gains and biases are never decayed."))
    eps: float = field(default=1e-8, metadata=dict(min=1e-12, max=1e-3))


class AdamW(Optimizer):
    states = ("m", "v")

    def apply(self, key, parameter, gradient, lr):
        cfg = self.cfg
        self.m[key] *= cfg.beta1
        self.m[key] += (1 - cfg.beta1) * gradient
        self.v[key] *= cfg.beta2
        self.v[key] += (1 - cfg.beta2) * gradient * gradient
        m = self.m[key] / (1 - cfg.beta1 ** self.steps)  # undo the bias from starting both averages at zero
        v = self.v[key] / (1 - cfg.beta2 ** self.steps)
        if self.decays(key):
            parameter *= 1 - lr * cfg.weight_decay
        parameter -= lr * m / (self.xp.sqrt(v) + cfg.eps)


@register
class AdamWSpec(ModuleSpec):
    id = "optimizer/adamw"
    slot = "optimizer"
    name = "AdamW"
    blurb = "Adaptive per-parameter step sizes with decoupled weight decay."
    ref = "Loshchilov & Hutter 2019"
    config = AdamWConfig
    backward = "Update rule in NumPy · checked against a hand-computed step"

    def build(self, cfg, dims, rng):
        return AdamW(cfg)
