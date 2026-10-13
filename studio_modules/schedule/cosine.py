"""Linear warmup, then cosine decay to a floor."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from studio_registry import ModuleSpec, register


@dataclass
class CosineConfig:
    warmup: int = field(default=20, metadata=dict(min=0, max=100000, help="Steps over which the rate climbs from 0 to its peak."))
    min_lr: float = field(default=.1, metadata=dict(min=0, max=1, help="Floor, as a fraction of the peak rate."))


class Cosine:
    def __init__(self, cfg):
        self.cfg = cfg

    def rate(self, step, total, lr):
        """Learning rate for optimizer step `step` (1-based) of `total`."""
        warmup = min(self.cfg.warmup, total - 1)
        if warmup and step <= warmup:
            return lr * step / warmup
        progress = (step - warmup) / max(1, total - warmup)
        floor = self.cfg.min_lr
        return lr * (floor + (1 - floor) * .5 * (1 + math.cos(math.pi * progress)))


@register
class CosineSpec(ModuleSpec):
    id = "schedule/cosine"
    slot = "schedule"
    name = "Warmup + cosine"
    blurb = "Linear warmup, then cosine decay to a floor."
    ref = "standard"
    config = CosineConfig
    backward = "No backward pass"

    def build(self, cfg, dims, rng):
        return Cosine(cfg)
