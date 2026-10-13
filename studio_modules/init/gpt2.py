"""GPT-2 initialization: Normal(0, std) weights, with residual output layers scaled down by depth."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from studio_registry import ModuleSpec, register


@dataclass
class GPT2InitConfig:
    std: float = field(default=.02, metadata=dict(min=1e-4, max=1, help="Standard deviation of the starting weights."))
    scale_residual: bool = field(default=True, metadata=dict(help="Divide the std of layers that write into the residual stream by √(2 · layers), so a deep stack starts with a stream of the same size."))


def fill_normal(parameter, rng, std):
    """Draw from the CPU NumPy stream (portable across devices) into the tensor."""
    parameter.copy_(torch.as_tensor(rng.normal(0, std, tuple(parameter.shape)), dtype=parameter.dtype))


class GPT2Init:
    def __init__(self, cfg):
        self.cfg = cfg

    @torch.no_grad()
    def apply(self, layer, rng, dims):
        """Fill the layer's weights by role, drawing from rng in the order the layer registered them."""
        for name, role in layer.roles.items():
            parameter = layer.params[name]
            if role in ("weight", "embedding", "residual_out"):
                std = self.cfg.std / math.sqrt(2 * dims.n_layer) if role == "residual_out" and self.cfg.scale_residual else self.cfg.std
                fill_normal(parameter, rng, std)
            elif role == "gain":
                parameter.fill_(1)
            elif role == "bias":
                parameter.fill_(0)


@register
class GPT2InitSpec(ModuleSpec):
    id = "init/gpt2"
    slot = "init"
    name = "GPT-2 scaled"
    blurb = "Normal(0, 0.02), with residual output layers scaled by 1/√(2·layers)."
    ref = "Radford et al. 2019"
    config = GPT2InitConfig
    backward = "No backward pass"

    def build(self, cfg, dims, rng):
        return GPT2Init(cfg)
