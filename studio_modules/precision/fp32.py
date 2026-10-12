"""float32 weights and activations."""
from __future__ import annotations

import numpy as np

from studio_registry import ModuleSpec, register


@register
class FP32(ModuleSpec):
    id = "precision/fp32"
    slot = "precision"
    name = "float32"
    blurb = "Standard for training. Half the memory of float64."
    ref = "—"
    backward = "No backward pass"

    def build(self, cfg, dims, rng):
        return np.float32
