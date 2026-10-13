"""No token mixer: the block keeps only its channel mixer."""
from __future__ import annotations

from studio_registry import ModuleSpec, register


@register
class NoMixer(ModuleSpec):
    id = "mixer/none"
    slot = "mixer"
    name = "None"
    blurb = "No token mixer in this block: positions do not exchange information here."
    ref = "—"
    skip = True
    provides = frozenset({"qk", "scores", "ordered"})  # nothing to rotate or bias, so it never blocks a position module
    backward = "No backward pass"
