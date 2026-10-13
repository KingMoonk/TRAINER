"""A constant learning rate schedule."""
from __future__ import annotations

from studio_registry import ModuleSpec, register


class Constant:
    def rate(self, step, total, lr):
        return lr


@register
class ConstantSpec(ModuleSpec):
    id = "schedule/constant"
    slot = "schedule"
    name = "Constant"
    blurb = "Keep the learning rate at its configured peak for every step."
    ref = "standard"
    backward = "No backward pass"

    def build(self, cfg, dims, rng):
        return Constant()
