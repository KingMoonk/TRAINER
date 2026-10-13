"""No channel mixer: the block keeps only its token mixer."""
from __future__ import annotations

from studio_registry import ModuleSpec, register


@register
class NoChannel(ModuleSpec):
    id = "channel/none"
    slot = "channel"
    name = "None"
    blurb = "No channel mixer. Common for SSM blocks; weak for attention stacks."
    ref = "—"
    skip = True
    backward = "No backward pass"
