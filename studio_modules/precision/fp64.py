"""float64 weights and activations (CPU and CUDA only; Apple's MPS has no float64)."""
from __future__ import annotations

import torch

from studio_registry import ModuleSpec, register


@register
class FP64(ModuleSpec):
    id = "precision/fp64"
    slot = "precision"
    name = "float64"
    blurb = "Double precision. Twice the memory, used for gradient checks; not available on Apple GPUs."
    ref = "—"
    backward = "No backward pass"

    def build(self, cfg, dims, rng):
        return torch.float64
