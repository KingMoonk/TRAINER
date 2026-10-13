"""Warmup, a stable-rate interval, then linear decay to zero."""
from __future__ import annotations

from dataclasses import dataclass, field

from studio_registry import ModuleSpec, register


@dataclass
class WSDConfig:
    warmup: int = field(default=20, metadata=dict(min=0, max=100000, help="Steps over which the rate climbs linearly from zero to its peak."))
    stable_fraction: float = field(default=.8, metadata=dict(min=0, max=1, help="Fraction of post-warmup steps held at peak before linear decay."))


class WSD:
    def __init__(self, cfg):
        self.cfg = cfg

    def rate(self, step, total, lr):
        """Learning rate for a 1-based step; decay reaches zero at `total`."""
        warmup = min(self.cfg.warmup, max(0, total - 1))
        if warmup and step <= warmup:
            return lr * step / warmup
        remaining = total - warmup
        decay_start = warmup + self.cfg.stable_fraction * remaining
        if step <= decay_start:
            return lr
        span = total - decay_start
        progress = 1.0 if span <= 0 else min(1.0, (step - decay_start) / span)
        return lr * (1.0 - progress)


@register
class WSDSpec(ModuleSpec):
    id = "schedule/wsd"
    slot = "schedule"
    name = "Warmup · stable · decay"
    blurb = "Linear warmup, a stable peak-rate interval, then linear decay to zero."
    ref = "standard"
    config = WSDConfig
    backward = "No backward pass"

    def build(self, cfg, dims, rng):
        return WSD(cfg)
