"""8-bit float training: the linear layers' matrix multiplies in float8, everything else as in bfloat16 mixed precision."""
from __future__ import annotations

import torch

from studio_registry import ModuleSpec, register


@register
class FP8(ModuleSpec):
    id = "precision/fp8"
    slot = "precision"
    name = "float8 mixed"
    blurb = ("The linear layers' matrix multiplies run in 8-bit floats (e4m3 forward, e5m2 for gradients, one scale per tensor) and their "
             "inputs are kept in 8 bits for the backward pass. Aimed at large models on GPUs with 8-bit cores; scaling each tensor costs time, "
             "so small models and laptop GPUs can run slower than with bfloat16, and the loss can end higher. Weights, optimizer, "
             "embeddings, norms, attention scores, softmax and loss stay wide. Sizes that are not multiples of 16 are padded with zeros; a "
             "layer narrower than 16 stays in bfloat16. Scoring and chat run in bfloat16. Needs an NVIDIA GPU of the RTX 40 series, H100 or newer.")
    ref = "Micikevicius et al. 2022"
    backward = "No backward pass"
    compute = torch.bfloat16     # whatever is not a linear layer runs as bfloat16 mixed precision
    fp8 = True
    tradeoff = "8-bit matrix multiplies and smaller saved linear-layer inputs; small models can run slower, and the loss may end higher than bfloat16's"

    def build(self, cfg, dims, rng):
        return torch.float32
