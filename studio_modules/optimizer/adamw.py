"""AdamW: adaptive per-parameter step sizes with decoupled weight decay."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

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
        self.m[key].mul_(cfg.beta1).add_(gradient, alpha=1 - cfg.beta1)
        self.v[key].mul_(cfg.beta2).addcmul_(gradient, gradient, value=1 - cfg.beta2)
        m = self.m[key] / (1 - cfg.beta1 ** self.steps)  # undo the bias from starting both averages at zero
        v = self.v[key] / (1 - cfg.beta2 ** self.steps)
        if self.decays(key):
            parameter.mul_(1 - lr * cfg.weight_decay)
        parameter.sub_(lr * m / (torch.sqrt(v) + cfg.eps))


    @torch.no_grad()
    def apply_all(self, parameters, gradients, multiplier, rates):
        """apply() for every parameter at once with foreach kernels: the same arithmetic, a handful of launches instead of ten per tensor."""
        cfg = self.cfg
        keys = list(parameters)
        grads = torch._foreach_mul([gradients[k] for k in keys], multiplier)
        m, v = [self.m[k] for k in keys], [self.v[k] for k in keys]
        torch._foreach_mul_(m, cfg.beta1)
        torch._foreach_add_(m, grads, alpha=1 - cfg.beta1)
        torch._foreach_mul_(v, cfg.beta2)
        torch._foreach_addcmul_(v, grads, grads, value=1 - cfg.beta2)
        mh = torch._foreach_div(m, 1 - cfg.beta1 ** self.steps)
        denom = torch._foreach_sqrt(torch._foreach_div(v, 1 - cfg.beta2 ** self.steps))
        torch._foreach_add_(denom, cfg.eps)
        for rate in {rates[k] for k in keys}:   # a learning-rate scale is per parameter: one pass per distinct rate
            at = [i for i, k in enumerate(keys) if rates[k] == rate]
            params = [parameters[keys[i]] for i in at]
            decayed = [parameters[keys[i]] for i in at if self.decays(keys[i])]
            if decayed:
                torch._foreach_mul_(decayed, 1 - rate * cfg.weight_decay)
            step = torch._foreach_div(torch._foreach_mul([mh[i] for i in at], rate), [denom[i] for i in at])
            torch._foreach_sub_(params, step)


@register
class AdamWSpec(ModuleSpec):
    id = "optimizer/adamw"
    slot = "optimizer"
    name = "AdamW"
    blurb = "Adaptive per-parameter step sizes with decoupled weight decay."
    ref = "Loshchilov & Hutter 2019"
    config = AdamWConfig
    backward = "Update rule in PyTorch · checked against a hand-computed step"

    def build(self, cfg, dims, rng):
        return AdamW(cfg)
