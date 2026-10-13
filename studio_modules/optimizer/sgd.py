"""Stochastic gradient descent with optional momentum and role-aware L2 decay."""
from __future__ import annotations

from dataclasses import dataclass, field

from studio_registry import ModuleSpec, register
from studio_slots import Optimizer


@dataclass
class SGDConfig:
    lr: float = field(default=1e-2, metadata=dict(min=1e-7, max=1.0, help="Peak learning rate; the schedule scales it."))
    momentum: float = field(default=0.0, metadata=dict(min=0, max=.9999, help="Fraction of the previous update retained."))
    weight_decay: float = field(default=0.0, metadata=dict(min=0, max=1, help="L2 penalty applied to decayed weights, excluding gains and biases."))


class SGD(Optimizer):
    states = ("velocity",)

    def apply(self, key, parameter, gradient, lr):
        grad = gradient
        if self.decays(key) and self.cfg.weight_decay:
            grad = grad + self.cfg.weight_decay * parameter
        velocity = self.velocity[key]
        velocity.mul_(self.cfg.momentum).add_(grad)
        parameter.sub_(lr * velocity)


@register
class SGDSpec(ModuleSpec):
    id = "optimizer/sgd"
    slot = "optimizer"
    name = "SGD"
    blurb = "Gradient descent with optional momentum and role-aware weight decay."
    ref = "standard"
    config = SGDConfig
    backward = "Update rule checked against a hand-computed reference"

    def build(self, cfg, dims, rng):
        return SGD(cfg)
