"""The recipe a new design starts from: the modern preset, and the older defaults it replaces.

A stored spec lists every module id, so changing what a *new* design starts from never touches an existing version or
checkpoint. `studio_spec.DEFAULTS` stays the old recipe on purpose: it is what a spec that omits a slot is read as, so
a spec written before the preset existed keeps loading to the same network. `modern_spec` is only used to start a design.

Modern means the recipe small open models share (Llama, Qwen, SmolLM): RMSNorm, SwiGLU, rotary positions, grouped-query
attention without biases, a tied embedding, AdamW with betas .9/.95 and decay .1, warmup-stable-decay, no dropout (one
pass over plenty of text does not overfit), and bfloat16 mixed precision on a GPU that runs it. The SwiGLU width is searched
so the whole network has the parameter count the older recipe has at the same size.
"""
from __future__ import annotations

DEFAULT_DIMS = {"d_model": 64, "n_layer": 2, "n_ctx": 256}


def attention_heads(d_model):
    """(query heads, key/value heads): heads of 16 to 32 dimensions, an even depth for rotary positions, and shared
    key/value heads once there are enough query heads to share them."""
    depth = 16 if d_model < 128 else 32
    if d_model % depth == 0:
        heads = d_model // depth
    else:
        heads = next((h for h in range(min(64, d_model // 8), 0, -1) if d_model % h == 0 and (d_model // h) % 2 == 0), 1)
    heads = max(1, min(64, heads))
    kv = heads // 4 if heads >= 8 and heads % 4 == 0 else heads // 2 if heads % 2 == 0 else heads
    return heads, kv


def precision_for(device):
    """(precision id, note) for the device a run will use: bfloat16 mixed where it runs, float32 otherwise, with the reason."""
    import torch
    from studio_modules.precision.policy import autocast_works
    if device == "cpu":
        return "fp32", "On the CPU, bfloat16 is not faster than float32, so this design trains in float32."
    if device in ("cuda", "mps") and autocast_works(device, torch.bfloat16):
        return "bf16", ""
    label = {"cuda": "this GPU", "mps": "this Apple GPU"}.get(device, device)
    return "fp32", f"bfloat16 does not run on {label} with this PyTorch, so this design trains in float32."


def legacy_spec(name="My network", dims=None):
    """The 2019 recipe (layernorm, GELU MLP, learned positions, float32, cosine): what the registry defaults describe."""
    from studio_spec import validate_spec
    return validate_spec({"schema": 1, "name": name, "dims": {**DEFAULT_DIMS, "dropout": .1, **(dims or {})}})


def modern_spec(name="My network", dims=None, device="auto"):
    """(spec, note): the modern preset at the given size, and why it is not bfloat16 when it is not (empty otherwise)."""
    from studio_estimate import estimate
    from studio_modules.precision.policy import likely_device
    from studio_spec import validate_spec
    dims = {**DEFAULT_DIMS, "dropout": 0., **(dims or {})}
    heads, kv = attention_heads(dims["d_model"])
    precision, note = precision_for(likely_device(device))
    spec = {"schema": 1, "name": name, "dims": dims,
            "position": {"id": "rope"},
            "block": {"norm": {"id": "rmsnorm"},
                      "mixer": {"id": "gqa", "config": {"heads": heads, "kv_heads": kv, "bias": False}},
                      "channel": {"id": "swiglu"}},
            "head": {"id": "tied"},
            "train": {"optimizer": {"id": "adamw", "config": {"beta1": .9, "beta2": .95, "weight_decay": .1}},
                      "schedule": {"id": "wsd"}, "init": {"id": "gpt2"}, "precision": {"id": precision}}}
    target = estimate(legacy_spec(name, dims))["total"]
    best = None
    for d_ff in range(8, 8193, 8):   # the total grows with the width, so stop at the first one past the target
        spec["block"]["channel"] = {"id": "swiglu", "config": {"d_ff": d_ff}}
        total = estimate(validate_spec(spec))["total"]
        if best is None or abs(total - target) < abs(best[0] - target):
            best = (total, d_ff)
        if total >= target:
            break
    spec["block"]["channel"] = {"id": "swiglu", "config": {"d_ff": best[1]}}
    return validate_spec(spec), note


GATED = ("swiglu",)   # channel modules with three matrices, so the same room costs 2/3 of the width of a plain MLP


def room_ratio(channel, d_model):
    """The thinking room of a channel entry as a multiple of d_model, counted as a plain two-matrix MLP would (a gated
    channel of width f holds 1.5 f of them). 4 is the usual room; a gated channel, sized to match parameters rather than
    exactly 8/3, reads as the nearest of 2, 4 and 8, so a design is not called wide for a few extra units."""
    if channel["id"] in GATED:
        return (2, 4, 8)[room_choice(channel, d_model)]
    return (channel["config"].get("d_ff") or 4 * d_model) / d_model


def room_choice(channel, d_model):
    """0 narrow, 1 normal, 2 wide. A plain MLP is normal at exactly 4 d_model; a gated channel is the nearest of 2, 4 and 8."""
    if channel["id"] in GATED:
        ratio = 1.5 * (channel["config"].get("d_ff") or ((8 * d_model + 47) // 48) * 16) / d_model
        return 0 if ratio < 2.83 else 2 if ratio > 5.66 else 1
    ratio = (channel["config"].get("d_ff") or 4 * d_model) / d_model
    return 0 if ratio < 4 else 1 if ratio == 4 else 2


def room_width(channel_id, multiple, d_model):
    """d_ff that gives `multiple` times d_model of thinking room in this channel module."""
    if channel_id in GATED:
        return max(8, round(multiple * d_model / 1.5 / 8) * 8)
    return multiple * d_model
