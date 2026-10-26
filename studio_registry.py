"""The module registry: the only place the designer, estimator, validator and compiler look modules up."""
from __future__ import annotations

import dataclasses
import importlib
import inspect
import pkgutil
import threading
from pathlib import Path

from studio_slots import SLOTS, ModuleSpec, config_dict

_MODULES = {}   # (slot, short id) -> ModuleSpec instance, in registration order
_loaded = False
_load_lock = threading.RLock()   # the trainer thread and the UI thread can both be first to ask


def register(cls):
    """Class decorator: add a ModuleSpec subclass to the registry."""
    module = cls()
    slot, _, short = module.id.partition("/")
    if slot != module.slot or slot not in SLOTS or not short:
        raise ValueError(f"{cls.__name__}: id must be '<slot>/<name>' with a known slot, got id={module.id!r} slot={module.slot!r}.")
    if (slot, short) in _MODULES and type(_MODULES[slot, short]).__module__ != cls.__module__:
        raise ValueError(f"Module {module.id} is registered twice.")
    _MODULES[slot, short] = module
    return cls


def load():
    """Import every file under studio_modules/ once, so each module registers itself."""
    global _loaded
    if _loaded:
        return
    with _load_lock:
        if _loaded:
            return
        import studio_modules
        for info in pkgutil.walk_packages(studio_modules.__path__, "studio_modules."):
            importlib.import_module(info.name)
        _loaded = True


def get(slot, short):
    load()
    try:
        return _MODULES[slot, short]
    except KeyError:
        raise ValueError(f"Unknown {SLOTS[slot][0].lower() if slot in SLOTS else slot} module: {short!r}.") from None


# Display order of the library cards; a module not listed here is shown after these, alphabetically.
ORDER = {"tokenizer": ["bpe_gpt2", "bpe_fast", "bpe_custom", "word", "byte", "character"], "embedding": ["learned", "factorized"],
         "position": ["learned", "sinusoidal", "rope", "alibi", "none"], "norm": ["layernorm", "rmsnorm", "none"],
         "mixer": ["attention", "gqa", "linear", "ssm", "gru", "lstm", "conv", "none"],
         "channel": ["mlp_gelu", "swiglu", "moe", "none"], "head": ["tied", "untied"], "objective": ["ce", "ce_ls", "ce_z"],
         "optimizer": ["adamw", "sgd", "lion", "muon"], "schedule": ["cosine", "wsd", "constant"], "init": ["gpt2", "mup"],
         "precision": ["fp32", "bf16", "fp16", "fp8", "fp64"]}


def for_slot(slot):
    load()
    order = ORDER.get(slot, [])
    found = [module for (s, _), module in _MODULES.items() if s == slot]
    return sorted(found, key=lambda m: (order.index(m.short) if m.short in order else len(order), m.short))


def every():
    load()
    return list(_MODULES.values())


def source_path(module):
    """Repo-relative path of the file a module lives in, for the Sandbox's contract panel."""
    path = Path(inspect.getfile(type(module)))
    return "/".join(path.parts[-3:])


def fields(module):
    """Config fields as plain dicts, for the generated form: name, type, default, min, max, choices, help."""
    if module.config is None:
        return []
    out = []
    for f in dataclasses.fields(module.config):
        kind = "bool" if isinstance(f.default, bool) else "int" if isinstance(f.default, int) else "float" if isinstance(f.default, float) else "str"
        out.append({"name": f.name, "type": kind, "default": f.default, **{k: f.metadata[k] for k in ("min", "max", "choices", "help", "label") if k in f.metadata}})
    return out


def to_json(module):
    return {"id": module.id, "slot": module.slot, "short": module.short, "name": module.name, "blurb": module.blurb, "ref": module.ref,
            "fields": fields(module), "defaults": config_dict(module.parse()), "provides": sorted(module.provides),
            "requires": sorted(module.requires), "hooks": [list(h) for h in module.hooks], "skip": module.skip,
            "backward": module.backward, "path": source_path(module)}


__all__ = ["register", "get", "for_slot", "every", "fields", "to_json", "source_path", "ModuleSpec"]
