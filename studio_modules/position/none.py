"""No position signal."""
from __future__ import annotations

from studio_registry import ModuleSpec, register


@register
class NoPosition(ModuleSpec):
    id = "position/none"
    slot = "position"
    name = "None (NoPE)"
    blurb = "No position signal. Order-aware mixers don’t need one; attention can infer some order from the causal mask."
    ref = "Kazemnejad et al. 2023"
    skip = True
    backward = "No backward pass"
