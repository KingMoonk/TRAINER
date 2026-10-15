"""ModelSpec: a model and its training recipe as plain JSON, one registered module per slot.

    {"schema": 1, "name": "...",
     "dims": {"d_model": 64, "n_layer": 2, "n_ctx": 256, "dropout": 0.1},
     "tokenizer": {"id": "byte"}, "embedding": {"id": "learned"}, "position": {"id": "learned"},
     "block": {"norm": {"id": "layernorm"}, "mixer": {"id": "attention", "config": {"heads": 4}}, "channel": {"id": "mlp_gelu"}},
     "blocks": [{"index": 1, "mixer": {"id": "gru"}}],          # per-block overrides (a hybrid stack)
     "head": {"id": "tied"},
     "train": {"objective": {"id": "ce"}, "optimizer": {"id": "adamw"}, "schedule": {"id": "cosine"},
               "init": {"id": "gpt2"}, "precision": {"id": "fp32"}}}

validate_spec() returns the normalized form: every slot entry is {"id", "config"} with the full config.
Older architectures ("groundwork-arch/1", and plain layers/heads/expansion configs) are converted by from_legacy().
"""
from __future__ import annotations

import copy
import hashlib
import json
import math

import studio_registry as registry
from studio_lora import lora_settings
from studio_slots import BLOCK_SLOTS, TRAIN_SLOTS, Dims, config_dict

SCHEMA = 1
DIM_RANGES = {"d_model": (16, 1024), "n_layer": (1, 24), "n_ctx": (32, 2048)}
MODEL_SLOTS = ("tokenizer", "embedding", "position", "head")
DEFAULTS = {"tokenizer": "byte", "embedding": "learned", "position": "learned", "norm": "layernorm", "mixer": "attention",
            "channel": "mlp_gelu", "head": "tied", "objective": "ce", "optimizer": "adamw", "schedule": "cosine",
            "init": "gpt2", "precision": "fp32"}
# Run settings an architecture may carry along (older designs stored them); the Training setup page supplies the rest.
RUN_INTS = {"steps": (1, 1000000), "batch_size": (1, 128), "accumulation": (1, 64), "eval_every": (1, 100000), "seed": (0, 2147483647)}
RUN_CHOICES = {"objective": ("all", "assistant"), "device": ("auto", "cpu", "cuda", "mps")}


def is_model_spec(value):
    return isinstance(value, dict) and "schema" in value


def default_spec(name="My network"):
    return validate_spec({"schema": SCHEMA, "name": name, "dims": {"d_model": 64, "n_layer": 2, "n_ctx": 256, "dropout": .1}})


def _whole(value, label, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value != int(value):
        raise ValueError(f"{label} must be a whole number.")
    if not low <= int(value) <= high:
        raise ValueError(f"{label} must be between {low} and {high}.")
    return int(value)


def _entry(slot, raw, where=""):
    """Normalize one slot entry to {"id", "config"} with a complete, validated config."""
    if raw is None:
        raw = {}
    if isinstance(raw, str):
        raw = {"id": raw}
    if not isinstance(raw, dict):
        raise ValueError(f"{where}{slot} must be an object with an id.")
    module = registry.get(slot, raw.get("id", DEFAULTS[slot]))
    config = raw.get("config")
    if config is not None and not isinstance(config, dict):
        raise ValueError(f"{where}{slot} config must be an object.")
    return {"id": module.short, "config": config_dict(module.parse(config))}


def validate_spec(spec):
    """Return a normalized deep copy, or raise ValueError with a message a person can act on.

    This checks shape, ranges and that every module and setting exists. Whether the pieces fit together
    (heads divide d_model, RoPE has q and k to rotate) is studio_validate.validate()'s job.
    """
    if not isinstance(spec, dict):
        raise ValueError("A model spec must be a JSON object.")
    if spec.get("schema", SCHEMA) != SCHEMA:
        raise ValueError(f"Unsupported model spec schema: {spec.get('schema')!r}.")
    dims_in = spec.get("dims") or {}
    if not isinstance(dims_in, dict):
        raise ValueError("dims must be an object.")
    defaults = {"d_model": 64, "n_layer": 2, "n_ctx": 256}
    dims = {key: _whole(dims_in.get(key, defaults[key]), key, *DIM_RANGES[key]) for key in DIM_RANGES}
    dropout = dims_in.get("dropout", .1)
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not 0 <= dropout <= .5:
        raise ValueError("Dropout must be between 0 and 0.5.")
    dims["dropout"] = float(dropout)
    out = {"schema": SCHEMA, "name": str(spec.get("name") or "My network"), "dims": dims}
    for slot in MODEL_SLOTS[:3]:
        out[slot] = _entry(slot, spec.get(slot))
    block_in = spec.get("block") or {}
    if not isinstance(block_in, dict):
        raise ValueError("block must be an object.")
    out["block"] = {slot: _entry(slot, block_in.get(slot), "block ") for slot in BLOCK_SLOTS}
    overrides, seen = [], set()
    blocks_in = spec.get("blocks") or []
    if not isinstance(blocks_in, list):
        raise ValueError("blocks must be a list of per-block overrides.")
    for raw in blocks_in:
        if not isinstance(raw, dict):
            raise ValueError("Each per-block override must be an object.")
        index = _whole(raw.get("index"), "Block override index", 0, 23)
        if index >= dims["n_layer"]:
            continue  # the stack got shorter; the override has nothing to apply to
        if index in seen:
            raise ValueError(f"Block {index + 1} is overridden twice.")
        seen.add(index)
        entry = {"index": index}
        for key, default in (("id", f"blk_{index:06x}"), ("label", f"Block {index + 1}")):
            if raw.get(key) and str(raw[key]) != default:
                entry[key] = str(raw[key])
        for slot in BLOCK_SLOTS:
            if slot in raw:
                entry[slot] = _entry(slot, raw[slot], f"Block {index + 1} ")
                if entry[slot] == out["block"][slot]:
                    del entry[slot]
        if len(entry) > 1:
            overrides.append(entry)
    out["blocks"] = sorted(overrides, key=lambda e: e["index"])
    out["head"] = _entry("head", spec.get("head"))
    train_in = spec.get("train") or {}
    if not isinstance(train_in, dict):
        raise ValueError("train must be an object.")
    out["train"] = {slot: _entry(slot, train_in.get(slot)) for slot in TRAIN_SLOTS}
    run_in = train_in.get("run") or {}
    if not isinstance(run_in, dict):
        raise ValueError("train.run must be an object.")
    run = {}
    for key, value in run_in.items():
        if key in RUN_INTS:
            run[key] = _whole(value, key.replace("_", " ").capitalize(), *RUN_INTS[key])
        elif key in RUN_CHOICES:
            if value not in RUN_CHOICES[key]:
                raise ValueError(f"{key.capitalize()} must be one of {', '.join(RUN_CHOICES[key])}.")
            run[key] = value
        elif key == "lora":
            if value is not None:
                run[key] = lora_settings(value)
        elif key == "validation":
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= .4:
                raise ValueError("Validation must be between 0 and 0.4.")
            run[key] = float(value)
        elif key == "checkpointing":
            if not isinstance(value, bool):
                raise ValueError("Checkpointing must be true or false.")
            if value:
                run[key] = True   # opt-in: recompute each block's activations in backward to save memory
    if run:
        out["train"]["run"] = run
    return out


def block_at(spec, index):
    """The norm, mixer and channel entries of block `index`, with its override applied, plus its id and label."""
    block = dict(spec["block"])
    block.update(id=f"blk_{index:06x}", label=f"Block {index + 1}")
    for override in spec.get("blocks") or []:
        if override["index"] == index:
            block.update({key: value for key, value in override.items() if key != "index"})
    return block


def set_block(spec, index, slot, entry):
    """Set one slot of one block in a draft spec (creating or updating its override)."""
    overrides = spec.setdefault("blocks", [])
    for override in overrides:
        if override.get("index") == index:
            override[slot] = entry
            return
    overrides.append({"index": index, slot: entry})


def module_of(slot, entry):
    """(module, parsed config) for a slot entry."""
    module = registry.get(slot, entry["id"])
    return module, module.parse(entry.get("config"))


def dims_of(spec, vocab, xp=None, dtype=None, layer=0):
    d = spec["dims"]
    extra = {key: value for key, value in (("xp", xp), ("dtype", dtype)) if value is not None}
    return Dims(d["d_model"], d["n_layer"], d["n_ctx"], vocab, d["dropout"], layer, **extra)


def vocab_of(spec, fallback=263):
    """Vocabulary size the tokenizer fixes in advance, or `fallback` when it depends on the data."""
    module, cfg = module_of("tokenizer", spec["tokenizer"])
    return module.vocab(cfg) or fallback


def spec_hash(spec):
    return hashlib.sha256(json.dumps(validate_spec(spec), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def structure_key(spec):
    """Canonical JSON of everything that shapes the weights and the computation (not names, labels or training)."""
    spec = validate_spec(spec)
    dims = {key: spec["dims"][key] for key in DIM_RANGES}
    blocks = [{slot: block_at(spec, i)[slot] for slot in BLOCK_SLOTS} for i in range(dims["n_layer"])]
    return json.dumps({"dims": dims, "blocks": blocks, **{slot: spec[slot] for slot in MODEL_SLOTS},
                       "precision": spec["train"]["precision"]}, sort_keys=True)


# ---------------------------------------------------------------------------------------------- older formats

def from_legacy(arch, cfg=None):
    """Convert a "groundwork-arch/1" architecture (validated by studio_arch) to a ModelSpec.

    `cfg`, when given, is the validated training config it runs under: its learning rate, weight decay,
    warmup and dropout win over the values stored in the architecture.
    """
    embedding, training = arch["embedding"], dict(arch.get("training") or {})
    cfg = cfg or {}
    width = embedding["width"]

    def mixer(block):
        return {"id": "attention", "config": {"heads": block["attention"]["heads"], "bias": False}} if block["attention"]["enabled"] else {"id": "none"}

    def channel(block):
        mlp = block["mlp"]
        if not mlp["enabled"]:
            return {"id": "none"}
        return {"id": "mlp_gelu", "config": {"d_ff": width * mlp["expansion"], "bias": True, "layers": mlp["layers"], "activation": mlp["activation"]}}

    blocks = arch["blocks"]
    first = blocks[0]
    overrides = []
    for i, block in enumerate(blocks):
        entry = {"index": i, "id": block["id"], "label": block["label"]}
        if i:
            entry.update(mixer=mixer(block), channel=channel(block))
        overrides.append(entry)
    run = {key: training[key] for key in (*RUN_INTS, *RUN_CHOICES, "validation") if key in training}
    return validate_spec({
        "schema": SCHEMA, "name": arch.get("name"),
        "dims": {"d_model": width, "n_layer": len(blocks), "n_ctx": embedding["context"], "dropout": cfg.get("dropout", embedding["dropout"])},
        "tokenizer": {"id": embedding["tokenizer"]}, "embedding": {"id": "learned"}, "position": {"id": "learned"},
        "block": {"norm": {"id": "layernorm"}, "mixer": mixer(first), "channel": channel(first)},
        "blocks": overrides, "head": {"id": "tied"},
        "train": {"objective": {"id": "ce"},
                  "optimizer": {"id": "adamw", "config": {"lr": cfg.get("learning_rate", training.get("learning_rate", .001)),
                                                          "weight_decay": cfg.get("weight_decay", training.get("weight_decay", .01))}},
                  "schedule": {"id": "cosine", "config": {"warmup": cfg.get("warmup", training.get("warmup", 20))}},
                  "init": {"id": "gpt2", "config": {"scale_residual": False}}, "precision": {"id": "fp32"}, "run": run}})


def to_legacy_view(spec):
    """The old attention/MLP block description, for the parts of the app that still draw that shape (the 3D viewer)."""
    dims = spec["dims"]
    blocks = []
    for i in range(dims["n_layer"]):
        block = block_at(spec, i)
        mixer, channel = registry.get("mixer", block["mixer"]["id"]), block["channel"]
        d_ff = channel["config"].get("d_ff") or 4 * dims["d_model"]
        blocks.append({"id": block["id"], "label": block["label"],
                       "attention": {"enabled": "attention" in mixer.provides, "heads": block["mixer"]["config"].get("heads", 1)},
                       "mlp": {"enabled": channel["id"] != "none", "expansion": max(1, min(8, round(d_ff / dims["d_model"]))),
                               "layers": channel["config"].get("layers", 1), "activation": channel["config"].get("activation", "gelu")}})
    return {"format": "groundwork-arch/1", "name": spec["name"],
            "embedding": {"tokenizer": spec["tokenizer"]["id"], "width": dims["d_model"], "context": dims["n_ctx"], "dropout": dims["dropout"]},
            "blocks": blocks, "training": dict(spec["train"].get("run") or {})}


def upgrade(spec):
    """Any stored architecture -> normalized ModelSpec."""
    if is_model_spec(spec):
        return validate_spec(spec)
    from studio_arch import validate_spec as validate_legacy
    return from_legacy(validate_legacy(spec))


def model_spec(cfg):
    """The ModelSpec a validated studio_model config trains: its architecture, with the config's run settings applied."""
    arch = cfg.get("architecture")
    if is_model_spec(arch):
        spec = copy.deepcopy(arch)
    else:
        from studio_arch import spec_from_config
        return from_legacy(arch or spec_from_config(cfg), cfg)
    spec["dims"]["dropout"] = cfg["dropout"]
    for slot, key, source in (("optimizer", "lr", "learning_rate"), ("optimizer", "weight_decay", "weight_decay"), ("schedule", "warmup", "warmup")):
        if key in spec["train"][slot]["config"] and source in cfg:
            spec["train"][slot]["config"][key] = cfg[source]
    return validate_spec(spec)


def config_from_spec(spec):
    """Full studio_model config for a ModelSpec. Display fields (heads, expansion) come from the first block."""
    spec = validate_spec(spec)
    dims, block, train = spec["dims"], spec["block"], spec["train"]
    cfg = {"name": spec["name"], "tokenizer": spec["tokenizer"]["id"], "width": dims["d_model"], "layers": dims["n_layer"],
           "context": dims["n_ctx"], "dropout": dims["dropout"], "heads": block["mixer"]["config"].get("heads", 1),
           "expansion": max(1, min(8, round((block["channel"]["config"].get("d_ff") or 4 * dims["d_model"]) / dims["d_model"]))),
           **(train.get("run") or {}), "architecture": spec}
    for slot, key, target in (("optimizer", "lr", "learning_rate"), ("optimizer", "weight_decay", "weight_decay"), ("schedule", "warmup", "warmup")):
        if key in train[slot]["config"]:
            cfg[target] = train[slot]["config"][key]
    return cfg
