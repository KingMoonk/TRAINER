"""Does a ModelSpec fit together? Module checks, requires/provides, and cross-module advice."""
from __future__ import annotations

import copy

import studio_registry as registry
from studio_slots import BLOCK_SLOTS, TRAIN_SLOTS, Issue
from studio_spec import block_at, dims_of, module_of, set_block, validate_spec, vocab_of

NEEDS = {"qk": "has no q and k to rotate.", "scores": "has no attention scores to bias."}


def validate(spec, vocab=None):
    """Issues for a normalized spec, errors first. An ERROR blocks training."""
    vocab = vocab or vocab_of(spec)
    issues = []

    def check(slot, entry, block=None):
        module, cfg = module_of(slot, entry)
        for issue in module.validate(cfg, dims_of(spec, vocab, layer=block or 0)) if not module.skip else []:
            issue.slot, issue.block = slot, block
            issues.append(issue)
        return module

    embedding = check("embedding", spec["embedding"])
    position = check("position", spec["position"])
    head = check("head", spec["head"])
    for slot in TRAIN_SLOTS:
        check(slot, spec["train"][slot])
    tokenizer, tokenizer_cfg = module_of("tokenizer", spec["tokenizer"])
    for issue in tokenizer.validate(tokenizer_cfg, dims_of(spec, vocab)):
        issue.slot = "tokenizer"
        issues.append(issue)
    for need in sorted(head.requires - embedding.provides):
        issues.append(Issue("ERROR", f"{head.name} needs an embedding it can read back out of; {embedding.name} does not offer that ({need}).", "head"))
    layers = spec["dims"]["n_layer"]
    mixers, channels, norms = [], [], []
    for i in range(layers):
        block = block_at(spec, i)
        norm, mixer, channel = (check(slot, block[slot], i) for slot in BLOCK_SLOTS)
        mixers.append(mixer)
        channels.append(channel)
        norms.append(norm)
        if mixer.skip and channel.skip:
            issues.append(Issue("ERROR", f"Block {i + 1} needs a token mixer or a channel mixer.", "mixer", i))
        for need in sorted(position.requires - mixer.provides):
            issues.append(Issue("ERROR", f"Position is {position.name}, but block {i + 1}'s {mixer.name} {NEEDS.get(need, 'does not provide ' + need + '.')}", "position", i))
        heads = block["mixer"]["config"].get("heads", 1)
        if position.short == "rope" and "qk" in mixer.provides and spec["dims"]["d_model"] % (2 * heads):
            issues.append(Issue("ERROR", f"RoPE needs an even head width; block {i + 1} has d_model {spec['dims']['d_model']} and {heads} heads.", "position", i))
    real = [m for m in mixers if not m.skip]
    attention = [m for m in real if "ordered" not in m.provides]
    if position.short == "none" and attention:
        issues.append(Issue("WARN", "Attention without a position signal can only infer order from the causal mask. It trains, usually worse. A good experiment.", "position"))
    if position.short in ("learned", "sinusoidal") and real and not attention:
        issues.append(Issue("INFO", "Every mixer already sees token order, so the position embedding is redundant.", "position"))
    if attention and any(c.skip for c, m in zip(channels, mixers) if "ordered" not in m.provides and not m.skip):
        issues.append(Issue("WARN", "Attention blocks without a channel mixer are much weaker per parameter.", "channel"))
    if any(m.short == "ssm" and not c.skip for m, c in zip(mixers, channels)):
        issues.append(Issue("INFO", "Mamba blocks normally have no MLP. Set the channel mixer to None to match the paper.", "channel"))
    if any(n.skip for n in norms):
        issues.append(Issue("WARN", "Without normalization, deep stacks diverge easily. Lower the learning rate.", "norm"))
    if len({m.short for m in mixers}) > 1:
        issues.append(Issue("INFO", "Hybrid stack: " + " → ".join(m.name.split(" ")[0] for m in mixers) + ". Deep views appear only for blocks whose mixer supports them.", "mixer"))
    if any(m.short in ("gru", "lstm") for m in mixers):
        issues.append(Issue("INFO", "Recurrent mixers train step by step over T and will be noticeably slower.", "mixer"))
    return sorted(issues, key=lambda issue: ("ERROR", "WARN", "INFO").index(issue.level))


def errors(spec, vocab=None):
    return [issue.text for issue in validate(spec, vocab) if issue.level == "ERROR"]


def why_blocked(spec, slot, short, block=None):
    """Why module `short` cannot go into `slot` of this spec ("" when it can). `block` targets one block's override."""
    draft = copy.deepcopy(spec)
    entry = {"id": short}
    if slot in BLOCK_SLOTS:
        if block is None:
            draft["block"][slot] = entry
            for override in draft.get("blocks") or []:
                override.pop(slot, None)
        else:
            set_block(draft, block, slot, entry)
    elif slot in TRAIN_SLOTS:
        draft["train"][slot] = entry
    else:
        draft[slot] = entry
    try:
        draft = validate_spec(draft)
    except ValueError as error:
        return str(error)
    before = {issue.text for issue in validate(spec) if issue.level == "ERROR"}
    new = [issue.text for issue in validate(draft) if issue.level == "ERROR" and issue.text not in before]
    return new[0] if new else ""


def hooks_of(spec):
    """Every hook name a compiled model of this spec emits, with its symbolic shape."""
    names = {"embed": "B T d", "final": "B T d", "logits": "B T V"}
    for i in range(spec["dims"]["n_layer"]):
        block = block_at(spec, i)
        for name in ("resid.pre", "resid.mid", "resid.post"):
            names[f"block{i}.{name}"] = "B T d"
        for slot in ("mixer", "channel"):
            for name, shape in registry.get(slot, block[slot]["id"]).hooks:
                names[f"block{i}.{name}"] = shape
    return names


def missing_hooks(spec, needed):
    """For a deep view that needs block-level hook names (e.g. "attn.scores"): why it cannot run, or "".

    A view is available when at least one block provides every hook it needs.
    """
    available = hooks_of(spec)
    lacking = []
    for i in range(spec["dims"]["n_layer"]):
        missing = [name for name in needed if f"block{i}.{name}" not in available]
        if not missing:
            return ""
        lacking.append((i, missing[0]))
    i, name = lacking[0]
    return f"needs {name}; block {i + 1} uses {registry.get('mixer', block_at(spec, i)['mixer']['id']).name}"
