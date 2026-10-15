"""Architecture specs: per-block transformer descriptions that map to parameter tensors.

A spec is a plain JSON dict (see the Network Builder contract). A legacy uniform
config (layers/heads/expansion) is the special case where every block has attention
plus a one-layer GELU MLP, and produces exactly the parameter names of old checkpoints.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import uuid

FORMAT = "groundwork-arch/1"
ACTIVATIONS = ("gelu", "relu")
TOKENIZERS = ("byte", "character")
BLOCK_ID = re.compile(r"^blk_[0-9a-f]{6}\Z")
TRAINING_DEFAULTS = {"steps": 300, "batch_size": 4, "accumulation": 1, "learning_rate": 0.001, "warmup": 20,
                     "weight_decay": 0.01, "eval_every": 25, "validation": 0.1, "seed": 42,
                     "objective": "all", "device": "auto"}
_INT_RANGES = {"steps": (1, 1000000), "batch_size": (1, 128), "accumulation": (1, 64), "warmup": (0, 100000),
               "eval_every": (1, 100000), "seed": (0, 2147483647)}
_FLOAT_RANGES = {"learning_rate": (1e-7, .1), "weight_decay": (0, 1), "validation": (0, .4)}


def _new_id():
    return "blk_" + uuid.uuid4().hex[:6]


def new_block(heads=4, expansion=4, attention=True, mlp=True, layers=1, activation="gelu"):
    return {"id": _new_id(), "label": "Block",
            "attention": {"enabled": bool(attention), "heads": heads},
            "mlp": {"enabled": bool(mlp), "expansion": expansion, "layers": layers, "activation": activation}}


def default_spec(name="My network"):
    blocks = [new_block(), new_block()]
    for i, block in enumerate(blocks):
        block["label"] = f"Block {i + 1}"
    return validate_spec({"format": FORMAT, "name": name,
                          "embedding": {"tokenizer": "byte", "width": 64, "context": 256, "dropout": 0.1},
                          "blocks": blocks, "training": dict(TRAINING_DEFAULTS, objective="assistant")})


def _gpt3(name, width, layers, heads, lr):
    """A GPT-3 paper shape (Brown et al. 2020, table 2.1) with its recipe on the spec: AdamW betas .9/.95,
    eps 1e-8, weight decay .1, cosine decay to 10% after warmup. Gradient clipping is the engine's fixed 1.0."""
    return {"schema": 1, "name": name,
            "dims": {"d_model": width, "n_layer": layers, "n_ctx": 2048, "dropout": .1},
            "block": {"mixer": {"id": "attention", "config": {"heads": heads}},
                      "channel": {"id": "mlp_gelu", "config": {"d_ff": 4 * width}}},
            "train": {"optimizer": {"id": "adamw", "config": {"lr": lr, "beta1": .9, "beta2": .95, "eps": 1e-8, "weight_decay": .1}},
                      "schedule": {"id": "cosine", "config": {"min_lr": .1}}}}


def model_spec_presets():
    """Ready-to-edit ModelSpecs built only from modules registered in this checkout.

    The GPT-2- and GPT-3-style presets use the local byte tokenizer until GPT-2 BPE is installed;
    they describe the network shape (and for GPT-3 its optimizer recipe), not pretrained weights or vocabulary.
    """
    import studio_registry as registry
    from studio_spec import validate_spec as validate_model_spec

    presets = {
        "starter": validate_model_spec({"schema": 1, "name": "Starter decoder",
                                         "dims": {"d_model": 64, "n_layer": 2, "n_ctx": 256, "dropout": .1}}),
        "gpt2_small_style": validate_model_spec({"schema": 1, "name": "GPT-2 small style · byte tokenizer",
                                                  "dims": {"d_model": 768, "n_layer": 12, "n_ctx": 1024, "dropout": .1},
                                                  "block": {"mixer": {"id": "attention", "config": {"heads": 12}},
                                                            "channel": {"id": "mlp_gelu", "config": {"d_ff": 3072}}}}),
        "gpt3_small_style": validate_model_spec(_gpt3("GPT-3 Small style · byte tokenizer", 768, 12, 12, 6e-4)),
        "gpt3_medium_style": validate_model_spec(_gpt3("GPT-3 Medium style · byte tokenizer", 1024, 24, 16, 3e-4)),
    }
    llama_modules = (("norm", "rmsnorm"), ("position", "rope"), ("mixer", "gqa"), ("channel", "swiglu"))
    try:
        for slot, short in llama_modules:
            registry.get(slot, short)
    except ValueError:
        pass
    else:
        presets["tiny_llama_style"] = validate_model_spec({
            "schema": 1, "name": "Tiny Llama style · byte tokenizer",
            "dims": {"d_model": 256, "n_layer": 4, "n_ctx": 256, "dropout": 0},
            "position": {"id": "rope"},
            "block": {"norm": {"id": "rmsnorm"},
                      "mixer": {"id": "gqa", "config": {"heads": 8, "kv_heads": 2, "bias": False}},
                      "channel": {"id": "swiglu"}},
        })
    return presets


def _int(value, label, low, high):
    if isinstance(value, bool) or (isinstance(value, float) and (math.isinf(value) or math.isnan(value) or value != int(value))):
        raise ValueError(f"{label} must be a whole number.")
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{label} must be a whole number.") from None
    if not low <= number <= high:
        raise ValueError(f"{label} must be between {low} and {high}.")
    return number


def _float(value, label, low, high):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a number.") from None
    if math.isinf(number) or math.isnan(number):
        raise ValueError(f"{label} must be a number.")
    if not low <= number <= high:
        raise ValueError(f"{label} must be between {low} and {high}.")
    return number


def _flag(value, label):
    if not isinstance(value, (bool, int)) or value not in (0, 1):
        raise ValueError(f"{label} must be true or false.")
    return bool(value)


def validate_spec(spec):
    """Return a normalized deep copy of a spec, or raise ValueError with a human message."""
    if not isinstance(spec, dict):
        raise ValueError("An architecture must be a JSON object.")
    if spec.get("format", FORMAT) != FORMAT:
        raise ValueError(f"Unsupported architecture format: {spec.get('format')!r}.")
    embedding = spec.get("embedding")
    if not isinstance(embedding, dict):
        raise ValueError("The architecture needs an embedding section.")
    tokenizer = embedding.get("tokenizer", "character")
    if tokenizer not in TOKENIZERS:
        raise ValueError("Tokenizer must be byte or character.")
    width = _int(embedding.get("width", 64), "Embedding width", 16, 512)
    context = _int(embedding.get("context", 128), "Context length", 32, 2048)
    dropout = _float(embedding.get("dropout", .1), "Dropout", 0, .5)
    blocks_in = spec.get("blocks")
    if not isinstance(blocks_in, list) or not 1 <= len(blocks_in) <= 24:
        raise ValueError("A network needs between 1 and 24 blocks.")
    blocks, seen = [], set()
    explicit = {raw.get("id") for raw in blocks_in if isinstance(raw, dict)}
    for i, raw in enumerate(blocks_in):
        if not isinstance(raw, dict):
            raise ValueError(f"Block {i + 1} must be an object.")
        label = str(raw.get("label") or f"Block {i + 1}")
        where = f"Block {i + 1}"
        block_id = raw.get("id")
        if block_id is None:  # derived from the index so spec hashes and transplant matching are deterministic
            n = i
            block_id = f"blk_{n:06x}"
            while block_id in seen or block_id in explicit:
                n += 1
                block_id = f"blk_{n:06x}"
        elif not isinstance(block_id, str) or not BLOCK_ID.match(block_id):
            raise ValueError(f"{where} has an invalid id {block_id!r}.")
        if block_id in seen:
            raise ValueError(f"{where} repeats the block id {block_id}.")
        seen.add(block_id)
        attention, mlp = raw.get("attention") or {}, raw.get("mlp") or {}
        if not isinstance(attention, dict) or not isinstance(mlp, dict):
            raise ValueError(f"{where} attention and mlp must be objects.")
        attention_on = _flag(attention.get("enabled", True), f"{where} attention.enabled")
        mlp_on = _flag(mlp.get("enabled", True), f"{where} mlp.enabled")
        if not attention_on and not mlp_on:
            raise ValueError(f"{where} must enable attention or the MLP (or both).")
        heads = _int(attention.get("heads", 4), f"{where} attention heads", 1, 16)
        if attention_on and width % heads:
            raise ValueError(f"{where}: embedding width {width} must be divisible by its {heads} attention heads.")
        expansion = _int(mlp.get("expansion", 4), f"{where} MLP expansion", 1, 8)
        layers = _int(mlp.get("layers", 1), f"{where} MLP layers", 1, 4)
        activation = mlp.get("activation", "gelu")
        if activation not in ACTIVATIONS:
            raise ValueError(f"{where} activation must be gelu or relu.")
        blocks.append({"id": block_id, "label": label,
                       "attention": {"enabled": attention_on, "heads": heads},
                       "mlp": {"enabled": mlp_on, "expansion": expansion, "layers": layers, "activation": activation}})
    training_in = spec.get("training") or {}
    if not isinstance(training_in, dict):
        raise ValueError("training must be an object.")
    training = {}
    for key, default in TRAINING_DEFAULTS.items():
        value = training_in.get(key, default)
        if key in _INT_RANGES:
            training[key] = _int(value, key.replace("_", " ").capitalize(), *_INT_RANGES[key])
        elif key in _FLOAT_RANGES:
            training[key] = _float(value, key.replace("_", " ").capitalize(), *_FLOAT_RANGES[key])
    if training_in.get("objective", "all") not in {"all", "assistant"}:
        raise ValueError("Objective must be all or assistant.")
    if training_in.get("device", "auto") not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("Device must be auto, cpu, cuda, or mps.")
    training["objective"] = training_in.get("objective", "all")
    training["device"] = training_in.get("device", "auto")
    return {"format": FORMAT, "name": str(spec.get("name") or "My network"),
            "embedding": {"tokenizer": tokenizer, "width": width, "context": context, "dropout": dropout},
            "blocks": blocks, "training": training}


def spec_from_config(cfg):
    """Legacy uniform config -> spec with deterministic block ids blk_000000, blk_000001, ..."""
    blocks = [{"id": f"blk_{i:06x}", "label": f"Block {i + 1}",
               "attention": {"enabled": True, "heads": cfg.get("heads", 4)},
               "mlp": {"enabled": True, "expansion": cfg.get("expansion", 4), "layers": 1, "activation": "gelu"}}
              for i in range(int(cfg.get("layers", 2)))]
    return validate_spec({
        "format": FORMAT, "name": cfg.get("name") or "My network",
        "embedding": {"tokenizer": cfg.get("tokenizer", "character"), "width": cfg.get("width", 64),
                      "context": cfg.get("context", 128), "dropout": cfg.get("dropout", .1)},
        "blocks": blocks, "training": {key: cfg.get(key, default) for key, default in TRAINING_DEFAULTS.items()}})


def config_from_spec(spec):
    """Full studio_model config for a spec (legacy display values come from the first block)."""
    spec = validate_spec(spec)
    first, embedding = spec["blocks"][0], spec["embedding"]
    return {"name": spec["name"], "tokenizer": embedding["tokenizer"], "width": embedding["width"],
            "layers": len(spec["blocks"]), "heads": first["attention"]["heads"], "context": embedding["context"],
            "expansion": first["mlp"]["expansion"], "dropout": embedding["dropout"], **spec["training"],
            "architecture": copy.deepcopy(spec)}


def parameter_names(spec, vocab=263):
    """Ordered {name: shape} for a spec. Order is the legacy insertion order (also the init/draw order)."""
    spec = validate_spec(spec)
    d, context = spec["embedding"]["width"], spec["embedding"]["context"]
    names = {"token": (vocab, d), "position": (context, d)}
    for i, block in enumerate(spec["blocks"]):
        prefix = f"block{i}"
        attention, mlp = block["attention"], block["mlp"]
        if attention["enabled"]:
            names[prefix + ".n1.g"] = (d,)
            names[prefix + ".n1.b"] = (d,)
        if mlp["enabled"]:
            names[prefix + ".n2.g"] = (d,)
            names[prefix + ".n2.b"] = (d,)
        if attention["enabled"]:
            names[prefix + ".qkv.w"] = (d, 3 * d)
            names[prefix + ".proj.w"] = (d, d)
        if mlp["enabled"]:
            hidden, layers = d * mlp["expansion"], mlp["layers"]
            names[prefix + ".f1.w"] = (d, hidden)
            for j in range(1, layers):
                names[f"{prefix}.fh{j}.w"] = (hidden, hidden)
            names[prefix + ".f2.w"] = (hidden, d)
            names[prefix + ".f1.b"] = (hidden,)
            for j in range(1, layers):
                names[f"{prefix}.fh{j}.b"] = (hidden,)
            names[prefix + ".f2.b"] = (d,)
    names["final.g"] = (d,)
    names["final.b"] = (d,)
    return names


def _size(shape):
    total = 1
    for n in shape:
        total *= n
    return total


def parameter_breakdown(spec, vocab=263):
    """{"total": int, "components": [{"id","label","kind","params"}]}; the output head is tied to the embedding."""
    spec = validate_spec(spec)
    names = parameter_names(spec, vocab)
    sizes = {name: _size(shape) for name, shape in names.items()}
    components = [{"id": "embedding", "label": "Embedding", "kind": "embedding",
                   "params": sizes["token"] + sizes["position"]}]
    for i, block in enumerate(spec["blocks"]):
        prefix = f"block{i}."
        if block["attention"]["enabled"]:
            params = sum(n for name, n in sizes.items() if name.startswith(prefix) and name.split(".")[1] in {"n1", "qkv", "proj"})
            components.append({"id": block["id"] + ":attention", "label": f"{block['label']} · attention",
                               "kind": "attention", "params": params})
        if block["mlp"]["enabled"]:
            params = sum(n for name, n in sizes.items() if name.startswith(prefix) and name.split(".")[1] not in {"n1", "qkv", "proj"})
            components.append({"id": block["id"] + ":mlp", "label": f"{block['label']} · MLP",
                               "kind": "mlp", "params": params})
    components.append({"id": "final", "label": "Final norm", "kind": "norm", "params": sizes["final.g"] + sizes["final.b"]})
    components.append({"id": "output", "label": "Output head (tied to embedding)", "kind": "output", "params": 0})
    return {"total": sum(sizes.values()), "components": components}


def _group_of(name):
    """Which (block-relative) module a parameter belongs to."""
    part = name.split(".")[1]
    return "attention" if part in {"n1", "qkv", "proj"} else "mlp"


def transplant(old_spec, old_params, new_spec, new_params):
    """Copy matching tensors from old_params into new_params in place; return {"copied": [...], "fresh": [...]}.

    Blocks match by id. A module (attention or MLP) whose tensor shapes changed, e.g. a new
    expansion, is left freshly initialised as a whole; added hidden layers are fresh.
    """
    old_spec, new_spec = validate_spec(old_spec), validate_spec(new_spec)
    old_index = {block["id"]: i for i, block in enumerate(old_spec["blocks"])}
    pairs = {}
    for name in new_params:
        if name.startswith("block"):
            prefix, rest = name.split(".", 1)
            j = int(prefix[5:])
            i = old_index.get(new_spec["blocks"][j]["id"])
            source = f"block{i}.{rest}" if i is not None else None
        else:
            source = name
        pairs[name] = source
    broken = set()
    for name, source in pairs.items():
        if name.startswith("block") and source in old_params and tuple(old_params[source].shape) != tuple(new_params[name].shape):
            broken.add((name.split(".")[0], _group_of(name)))
    copied, fresh = [], []
    for name, source in pairs.items():
        group = (name.split(".")[0], _group_of(name)) if name.startswith("block") else None
        if source in old_params and group not in broken and tuple(old_params[source].shape) == tuple(new_params[name].shape):
            new_params[name][...] = old_params[source]
            copied.append(name)
        else:
            fresh.append(name)
    return {"copied": copied, "fresh": fresh}


def structure_key(spec):
    """Canonical JSON of everything that shapes the computation (ignores names, labels and ids)."""
    spec = validate_spec(spec)
    return json.dumps({"embedding": {k: spec["embedding"][k] for k in ("tokenizer", "width", "context")},
                       "blocks": [{"attention": b["attention"] if b["attention"]["enabled"] else None,
                                   "mlp": b["mlp"] if b["mlp"]["enabled"] else None} for b in spec["blocks"]]},
                      sort_keys=True)


def spec_hash(spec):
    return hashlib.sha256(json.dumps(validate_spec(spec), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
