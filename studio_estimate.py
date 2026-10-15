"""Parameters, compute and memory of a ModelSpec, summed from each module's own estimate()."""
from __future__ import annotations

import torch

from studio_slots import BLOCK_SLOTS, SLOTS, Estimate
from studio_spec import block_at, dims_of, module_of, vocab_of

VOCAB_SLOTS = ("embedding", "position", "head")  # left out of the compute estimate, following Kaplan et al. 2020


def estimate(spec, vocab=None):
    """{"total", "active", "embedding_params", "block_params", "train_flops_per_token", "kv_bytes_per_token",
        "activation_bytes_per_token", "compute_dtype", "checkpointing", "rows": [{"slot", "block", "id", "name", "label", "params", ...}]}

    Activations and the KV cache are counted in the precision's compute dtype (bfloat16 under mixed precision; the weights,
    gradients and optimizer state stay in the weights' dtype). With checkpointing on, a block keeps only its input between the
    forward and backward passes, and one block's activations are alive at a time."""
    vocab = vocab or vocab_of(spec)
    precision, precision_cfg = module_of("precision", spec["train"]["precision"])
    dtype = precision.build(precision_cfg, None, None)
    compute = precision.compute or dtype
    run = spec["train"].get("run") or {}
    rows = []

    def add(slot, entry, block=None, label=None):
        module, cfg = module_of(slot, entry)
        e = Estimate() if module.skip else module.estimate(cfg, dims_of(spec, vocab, dtype=compute, layer=block or 0))
        rows.append({"slot": slot, "block": block, "id": module.short, "name": module.name, "label": label or SLOTS[slot][0],
                     "params": e.params, "active_params": e.params if e.active_params is None else e.active_params,
                     "flops_per_token": 0 if slot in VOCAB_SLOTS else e.flops_per_token,
                     "activation_bytes": e.activation_bytes, "kv_bytes": e.kv_bytes, "skip": module.skip})

    add("embedding", spec["embedding"])
    add("position", spec["position"])
    for i in range(spec["dims"]["n_layer"]):
        block = block_at(spec, i)
        mixer, channel = (module_of(slot, block[slot])[0] for slot in ("mixer", "channel"))
        for slot, owner, part in (("norm", mixer, "norm 1"), ("mixer", None, "token mixer"), ("norm", channel, "norm 2"), ("channel", None, "channel mixer")):
            if owner is not None and owner.skip:
                continue  # a skipped branch has no norm either
            add(slot, block[slot], i, f"{block['label']} · {part}")
    add("norm", spec["block"]["norm"], None, "Final norm")
    add("head", spec["head"])
    total = sum(row["params"] for row in rows)
    vocabulary = sum(row["params"] for row in rows if row["slot"] in VOCAB_SLOTS and row["block"] is None)
    activations = sum(row["activation_bytes"] for row in rows)
    if run.get("checkpointing"):
        inside = {}
        for row in rows:
            if row["block"] is not None:
                inside[row["block"]] = inside.get(row["block"], 0) + row["activation_bytes"]
        residual = spec["dims"]["d_model"] * torch.empty(0, dtype=dtype).element_size()   # the residual stream stays in the weights' dtype
        activations = activations - sum(inside.values()) + len(inside) * residual + max(inside.values(), default=0)
    return {"total": total, "active": sum(row["active_params"] for row in rows),
            "embedding_params": vocabulary, "block_params": total - vocabulary,
            # forward + backward costs about three forward passes
            "train_flops_per_token": 3 * sum(row["flops_per_token"] for row in rows),
            "kv_bytes_per_token": sum(row["kv_bytes"] for row in rows),
            "activation_bytes_per_token": activations, "compute_dtype": str(compute).removeprefix("torch."),
            "checkpointing": bool(run.get("checkpointing")), "rows": rows}


def per_block(result):
    """Parameters per block index, from estimate()'s rows."""
    blocks = {}
    for row in result["rows"]:
        if row["block"] is not None:
            blocks[row["block"]] = blocks.get(row["block"], 0) + row["params"]
    return [blocks[i] for i in sorted(blocks)]


__all__ = ["estimate", "per_block", "BLOCK_SLOTS"]
