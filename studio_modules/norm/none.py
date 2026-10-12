"""No normalization."""
from __future__ import annotations

from dataclasses import dataclass, field

from studio_registry import ModuleSpec, register


@dataclass
class NoNormConfig:
    placement: str = field(default="pre", metadata=dict(choices=["pre", "post"]))


@register
class NoNorm(ModuleSpec):
    id = "norm/none"
    slot = "norm"
    name = "None"
    blurb = "Identity. Useful to see why normalization matters."
    ref = "—"
    config = NoNormConfig
    skip = True
    backward = "No backward pass"
