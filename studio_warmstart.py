"""Warm start: carry a trained version's weights into a changed design (numpy dicts and ModelSpecs only).

    params, report = warm_start(old_params, old_spec, new_spec)

`params` is a full parameter dict for `CompiledModel.load_weights` (empty when the word list changed:
the new version has to start from scratch). `report` = {copied, fresh, zeroed, dropped, preserving, from_scratch, reason}.
New parts are initialised by a throwaway CompiledModel(new_spec); only numpy leaves this module.
"""
from __future__ import annotations

import re

import numpy as np

from studio_compile import CompiledModel
from studio_spec import block_at, validate_spec

# Residual-out projections and their biases: zero them and a new block adds nothing to the residual stream.
_RESIDUAL_OUT = ("proj.w", "proj.b", "f2.w", "f2.b", "down.w", "down.b", "ssm.out.w", "ssm.out.b")
_HIDDEN_OUT = ("f2.w", "down.w")                      # [hidden, d]: new rows are zeroed when the MLP widens
_RESIZABLE = ("f1.w", "f1.b", "f2.w", "gate.w", "up.w", "down.w")
_BLOCK = re.compile(r"block(\d+)\.(.*)")

NOTES = {
    "scratch": "The word list changed, so the old word table doesn’t fit: the new version has to start from scratch.",
    "heads": "The old weights are reused, split into a different number of heads, so error jumps at first and then recovers.",
    "other": "Some parts changed type or size and start fresh, so error jumps at first and then recovers.",
    "removed": "Removed parts are dropped from the old weights; error rises a little, then recovers.",
    "same": "New parts start at zero, so the new version begins exactly where the old one is and catches up from there.",
}


def _is_zero_out(name, model):
    return model.roles.get(name) == "residual_out" or name.endswith(_RESIDUAL_OUT)


def _overlap(old, new):
    """Copy the part of `old` that fits into `new`'s shape; return (array, trimmed)."""
    cut = tuple(slice(0, min(a, b)) for a, b in zip(old.shape, new.shape))
    new[cut] = old[cut]
    return new, any(a > b for a, b in zip(old.shape, new.shape))


def warm_start(old_params, old_spec, new_spec, seed=42):
    old_spec, new_spec = validate_spec(old_spec), validate_spec(new_spec)
    if old_spec["tokenizer"] != new_spec["tokenizer"]:
        return {}, dict(copied=[], fresh=[], zeroed=[], dropped=sorted(old_params), preserving=False, from_scratch=True, reason=NOTES["scratch"])
    old_n, new_n = old_spec["dims"]["n_layer"], new_spec["dims"]["n_layer"]
    model = CompiledModel(new_spec, int(old_params["token"].shape[0]), seed=seed, device="cpu")
    new = {k: np.array(model.backend.to_cpu(v), dtype=np.float32) for k, v in model.p.items()}
    copied, fresh, zeroed, trimmed = [], [], [], []
    swapped = resized_ctx = False
    for name, arr in new.items():
        old = old_params.get(name)
        block = _BLOCK.match(name)
        index = int(block.group(1)) if block else None
        added = index is not None and index >= old_n
        if old is not None and tuple(old.shape) == arr.shape:
            arr[...] = old
            copied.append(name)
        elif old is not None and (name == "position" or (block and block.group(2).endswith(_RESIZABLE))):
            arr, cut = _overlap(np.asarray(old), arr)
            if block and block.group(2).endswith(_HIDDEN_OUT) and arr.shape[0] > old.shape[0]:
                arr[old.shape[0]:] = 0
                zeroed.append(name)
            copied.append(name)
            if cut:
                trimmed.append(name)
            if name == "position" or "fh" in name:
                resized_ctx = True
            new[name] = arr
        elif added and _is_zero_out(name, model):
            arr[...] = 0
            zeroed.append(name)
        else:
            fresh.append(name)
            swapped = swapped or not added
    dropped = sorted(set(old_params) - set(new)) + trimmed
    heads = any(block_at(old_spec, i)["mixer"]["id"] == block_at(new_spec, i)["mixer"]["id"]
                and block_at(old_spec, i)["mixer"]["config"].get("heads") != block_at(new_spec, i)["mixer"]["config"].get("heads")
                for i in range(min(old_n, new_n)))
    post = any(block_at(new_spec, i)["norm"]["config"].get("placement") == "post" for i in range(old_n, new_n))
    other = swapped or resized_ctx or post
    kind = "heads" if heads else "other" if other else "removed" if dropped else "same"
    report = dict(copied=copied, fresh=fresh, zeroed=zeroed, dropped=dropped, preserving=kind == "same",
                  from_scratch=False, reason=NOTES[kind])
    return new, report
