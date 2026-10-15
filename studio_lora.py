"""LoRA (Hu et al. 2021, https://arxiv.org/abs/2106.09685): train a small low-rank correction instead of the whole model.

A chosen linear layer's weight W (in x out) stays frozen and gets a learned update  W + (alpha / rank) * A @ B  with
A (in x rank) and B (rank x out). B starts at zero, so the model starts out exactly the model it was. Only A and B train.

How it sits in the engine: adapters are ordinary entries of `model.p` / `model.g` named `lora.<weight>.a` and `lora.<weight>.b`,
the compiled forward (studio_compile) swaps in the effective weight, and `Trainable` is the view of the model an optimizer
steps. A checkpoint stores the frozen base under `weight::` and the adapters under `lora::`; `load_checkpoint` folds them
together (merge_arrays), so chat, export, evaluation and a later stage's `init_from` all see one plain merged model.
"""
from __future__ import annotations

import math

import numpy as np
import torch

DEFAULTS = {"rank": 8, "alpha": 16, "targets": ["qkv", "proj"]}
PREFIX = "lora."       # parameter names in model.p
FILE_PREFIX = "lora::"  # array names in a checkpoint


def lora_settings(raw):
    """A validated {"rank", "alpha", "targets"} (defaults filled in) from a train.run.lora value; None stays None."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("LoRA settings must be an object with rank, alpha and targets.")
    unknown = set(raw) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown LoRA setting: {sorted(unknown)[0]}.")
    out = {**DEFAULTS, **raw}
    rank = out["rank"]
    if isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= 256:
        raise ValueError("LoRA rank must be a whole number from 1 to 256.")
    alpha = out["alpha"]
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or not 0 < alpha <= 4096:
        raise ValueError("LoRA alpha must be a number above 0 and at most 4096.")
    targets = out["targets"]
    if (not isinstance(targets, (list, tuple)) or not targets or len(set(targets)) != len(targets)
            or not all(isinstance(name, str) and name.strip() for name in targets)):
        raise ValueError("LoRA targets must be a list of layer names, such as qkv and proj.")
    return {"rank": rank, "alpha": alpha, "targets": [name.strip() for name in targets]}


def scale_of(settings):
    return settings["alpha"] / settings["rank"]


def adapter_names(base):
    return PREFIX + base + ".a", PREFIX + base + ".b"


class Trainable:
    """The part of a model an optimizer steps: the same tensors as the model's, restricted to `names`. Anything else is read from the model."""

    def __init__(self, model, names):
        self.model = model
        names = list(names)
        self.p = {key: model.p[key] for key in names}
        self.g = {key: model.g[key] for key in names}
        self.roles = {key: model.roles[key] for key in names}
        self.owner = {key: model.owner[key] for key in names}
        self.group = {key: model.group[key] for key in names}
        self.lr_scale = {key: value for key, value in model.lr_scale.items() if key in self.p}
        self.elementwise = {key for key in model.elementwise if key in self.p}

    def __getattr__(self, name):  # spec, dims, backend...
        return getattr(self.model, name)


def attach_lora(model, settings):
    """Freeze `model` and add adapters to the weights the settings target. Returns the Trainable view of the adapters.

    Call it once, after the base weights are in place (a transplant or checkpoint load) and before the optimizer is built.
    """
    settings = lora_settings(settings)
    if model.adapters:
        raise ValueError("This model already has LoRA adapters.")
    wanted = set(settings["targets"])
    scale, rank = scale_of(settings), settings["rank"]
    found = set()
    base_names = list(model.p)
    for full in base_names:
        layer, name = model._slots[full]
        slot, block = model.owner[full]
        if slot not in ("mixer", "channel") or not name.endswith(".w") or name[:-2] not in wanted or model.p[full].ndim != 2:
            continue
        found.add(name[:-2])
        n_in, n_out = model.p[full].shape
        a_name, b_name = adapter_names(full)
        bound = 1 / math.sqrt(n_in)
        like = model.p[full]
        a = torch.as_tensor(model.rng.uniform(-bound, bound, (n_in, rank)), dtype=like.dtype, device=like.device)
        b = torch.zeros((rank, n_out), dtype=like.dtype, device=like.device)
        for key, value in ((a_name, a), (b_name, b)):
            model.p[key], model.g[key] = value, torch.zeros_like(value)
            model.roles[key], model.owner[key], model.group[key] = "adapter", ("lora", block), model.group[full]
        model.adapters[full] = (a_name, b_name, scale)
    missing = sorted(wanted - found)
    if missing:
        for full in list(model.adapters):
            for key in model.adapters.pop(full)[:2]:
                for table in (model.p, model.g, model.roles, model.owner, model.group):
                    del table[key]
        raise ValueError(f"LoRA found no weight called {', '.join(missing)} in this model's attention or MLP layers.")
    model.adapter_names = {key for a, b, _ in model.adapters.values() for key in (a, b)}
    model.frozen = set(base_names)
    return Trainable(model, sorted(model.adapter_names))


def merge_arrays(weights, adapters, scale):
    """Fold adapters into base weights. weights: {name: array}; adapters: {"<weight>.a" / "<weight>.b": array}. Returns a new dict of just the base weights."""
    out = dict(weights)
    for key in adapters:
        if not key.endswith(".a"):
            continue
        base = key[:-2]
        if base not in out or base + ".b" not in adapters:
            raise ValueError(f"The LoRA adapter for {base} has no matching weight or B matrix.")
        delta = adapters[key].astype(np.float64) @ adapters[base + ".b"].astype(np.float64)
        out[base] = (out[base].astype(np.float64) + scale * delta).astype(out[base].dtype)
    return out


@torch.no_grad()
def merge_lora(model):
    """Fold the model's adapters into its base weights in place, drop the adapters and unfreeze. The output is unchanged."""
    for full, (a, b, scale) in list(model.adapters.items()):
        model.p[full].add_(scale * (model.p[a] @ model.p[b]))
        for key in (a, b):
            for table in (model.p, model.g, model.roles, model.owner, model.group):
                del table[key]
    model.adapters, model.adapter_names, model.frozen = {}, set(), set()
    return model
