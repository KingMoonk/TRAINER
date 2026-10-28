"""The Builder's design maths, Tk-free: what a draft design is (read_spec/write_spec), which modules it can take and
why not, what it costs and what to expect of it. The API (studio_api.routers.design) calls these; nothing here touches a widget or holds state.
"""
from __future__ import annotations

import copy
import math
import uuid
from pathlib import Path
import os
import sys

import studio_registry as registry
from studio_arch import TRAINING_DEFAULTS, model_spec_presets
from studio_compute import MAX_EPOCHS, TOKENS_PER_PARAM, plan, step_seconds, throughput
from studio_estimate import estimate
from studio_glossary import MODULE_TERMS, glossary_text
from studio_preset import room_choice, room_width
from studio_slots import BLOCK_SLOTS
from studio_spec import block_at, config_from_spec, set_block, validate_spec
from studio_validate import errors, validate
from studio_versions import OTHER_PARTS, _facts, spec_diff
from studio_warmstart import NOTES
from studio_ia.attribution import attribute, blocks_change_text, run_changes, signed_move
from studio_ia import pieces as cut
from studio_ia.services import Job
from studio_ia.expectation import expectation


T = {
    "h1": "Build your model",
    "lead": "Change the picture. See what it costs and what it does.",
    "blocks": "Blocks",
    "queued": "QUEUED ON THIS COMPUTER",
    "training": "{new_version} {status}",
    "cost_slower": "slower",
    "cost_faster": "faster",
    "cost_more": "more",
    "cost_less": "less",
    "about_same": "about the same",
    "about_multiplier": "about {mult}×",
    "change_something": "Change something in the picture to see what to expect.",
    "part_embedding": "Word table",
    "part_position": "Position",
    "part_norm": "Even out",
    "part_mixer": "Look back",
    "part_channel": "Thinking room",
    "part_head": "Word scores",
    "final_norm": "Even out",
    "final_norm_tip": "One more evening-out after the last block. It is the same kind the blocks use.",
    "part_change": "{part}: {name}",
    "block_part_change": "block {n} {part}: {name}",
    "same_as_others": "same as the others",
    "part_expect": "{part} becomes {name}: {blurb}",
    "was": "was {name}",
    "preset_size": "{params} parameters · {memory} of weights and optimizer state",
    "preset_fits": "Fits this computer: about {need} at {rows} rows per step, of {ram}.",
    "preset_fits_fewer": "Too big for this computer at {rows} rows per step (about {need}; it has {ram}). It fits at {fit} rows per step.",
    "preset_no_fit": "Won't fit this computer: even one row per step needs about {need}, and it has {ram}.",
    "preset_fit_unknown": "This computer did not report its memory size, so whether it fits is unknown.",
    "preset_speed_gpu": "With the GPU, one step takes {step}.",
    "preset_speed_cpu": "Without a GPU, one step takes {step}.",
    "preset_speed_estimate": " That is an estimate: this design has not been timed on this computer yet, and the first run times it on each device.",
    "preset_speed_measured": "Measured on this computer: {device} is faster for this design ({speed:,.0f} tokens/s against {other:,.0f} on the {other_device}); one step takes {step}.",
    "preset_speed_measured_one": "Measured on this computer: only the {device} could be timed ({speed:,.0f} tokens/s); one step takes {step}.",
    "preset_slow": "Very slow: the {steps:,} steps of a catch-up take {total}.",
    "recipe_change": "training recipe: {recipe}",
    "compute_supports": "Your text supports about {supported} parameters at {rule} word pieces each; this design has {params}",
    "compute_under": ", far more than your text can train: a smaller design or more text would do better.",
    "compute_repeats": ", so your text would be repeated to train it fully.",
    "compute_reread": ", so training it fully reads your text about {times} times. Up to about {most} times works nearly as well as fresh text.",
    "compute_enough": ", so there is enough text for it.",
    "compute_roomy": ", so your text could train a bigger one.",
    "compute_steps": "Suggested length: about {steps:,} steps of {rows} rows ({passes} over your text).",
    "compute_time": " At this computer's speed that is {time}.",
    "compute_repeat": "Reaching {rule} word pieces per parameter would mean reading your text {times} times. Past about {most} times, repeating it adds little.",
    "compute_counting": "Counting your text…",
    "compute_none": "Add some text in Your data to see how much this design needs.",
    "lr_result": "Suggested learning rate: {rate}. The error fell fastest near {steepest}, and the suggestion is a tenth of that. This design uses {current}.",
    "lr_blew_up": "The error blew up past {rate}, so the test stopped there.",
    "lr_blew_up_early": "The error blew up near {rate} before it fell, so there is no suggestion.",
    "lr_flat": "The error did not fall during the test, so there is no suggestion. More text may help.",
    "lr_stopped": "Stopped. Nothing was saved.",
    "lr_failed": "The test could not run: {error}",
    # the learning loop: guess, train, check (#83)
    "predict_text": "Guess which way each one goes, then train it and see. Skip any you'd rather not guess.",
    "predict_error": "Held-back error",
    "predict_step": "Time per step",
    "predict_memory": "Memory while training",
    "predict_lower": "Lower",
    "predict_higher": "Higher",
    "predict_faster": "Faster",
    "predict_slower": "Slower",
    "predict_less": "Less",
    "predict_more": "More",
    "predict_save": "Save with my guesses",
    "predict_skip": "Save without guessing",
    "predict_back": "Back to editing",
    "predict_split_save": "Save {count} versions and queue them",
    "predict_step_title": "Version {n}: {change}",
    "guess_lower": "lower error",
    "guess_higher": "higher error",
    "guess_faster": "faster steps",
    "guess_slower": "slower steps",
    "guess_less": "less memory",
    "guess_more": "more memory",
    "guess_expected": "You expected {what}.",
    "guess_hit": "A hit.",
    "guess_miss": "A miss.",
    "guess_waiting": "It is checked once version {n} has trained.",
    "guess_no_score": "Version {n} has no held-back score to check it against yet.",
    "result_error": "It went from {a:.2f} to {b:.2f} bits per character.",
    "result_error_flat": "It stayed at {b:.2f} bits per character.",
    "result_step": "A step went from {a} to {b}.",
    "result_step_estimated": "A step takes {say} (estimated from the design; no measured speed for both runs).",
    "result_memory": "It needs {say} (estimated from the design).",
    "guess_kicker": "YOUR GUESS · VERSION {n}",
    "guess_page_said": "Before training, the page said: {text}",
    "split": "Split into {count} versions",
    "notebook_tip": "What each version changed, what that did to held-back error, step time and memory, and whether your guess was right. Error is measured; memory is estimated from the design; step time is measured when both runs logged their speed.",
    "notebook_empty": "Save and train a version to start the notebook.",
    "notebook_reading": "Reading the runs…",
    "notebook_version": "Version {n}",
    "notebook_not_trained": "not trained yet",
    "notebook_no_score": "no held-back score",
    "notebook_error": "error {a:.2f} → {b:.2f}",
    "notebook_error_only": "error {b:.2f}",
    "notebook_step": "steps {say}",
    "notebook_memory": "memory {say}",
    "notebook_estimated": " (est.)",
    "notebook_guessed": "you guessed {guesses}",
    "notebook_no_guess": "no guess",
    "notebook_many": "{count} changes at once",
    "notebook_also": "also differed: {names}",
    "notebook_asked": "asked “{question}”",
    # click a part to learn it (#83)
    "explain_in_block": "{part} · block {n}",
    "explain_now": "Now: {name}",
    "example_table_pieces": "“{sentence}” becomes {count} piece{cs}: {pieces}. Each piece looks up its own row of {d} numbers in a table with {upto}{rows} rows, one per entry in the word list.",
    "example_table": "Each piece of your text looks up its own row of {d} numbers in a table with {upto}{rows} rows, one per entry in the word list.",
    "example_position": "Up to {t} pieces are read at once. Position tells the model where each one sits, so “dog bites man” and “man bites dog” read differently.",
    "example_mixer_heads_pieces": "Reading “{last}”, each of the {heads} heads looks back over the {before} piece{s} before it and takes in what it finds useful, {size} numbers per head.",
    "example_mixer_heads": "Each piece looks back over the pieces before it through {heads} heads of {size} numbers each.",
    "example_mixer_pieces": "Reading “{last}”, it mixes in what the {before} piece{s} before it say{v}.",
    "example_mixer": "Each piece mixes in what the pieces before it say.",
    "example_channel_room": "Each piece is worked on alone, without looking at the others: its {d} numbers widen to {room} and come back to {d}.",
    "example_channel": "Each piece is worked on alone, without looking at the others.",
    "example_norm": "Each piece's {d} numbers are brought to a steady size before the next step, so none run away as blocks stack up.",
    "example_skipped": "This design skips it here: the numbers pass through unchanged.",
    "example_head_pieces": "After “{last}”, it turns that piece's {d} numbers into a score for each of the {upto}{rows} entries in the word list. The highest scores are its best guesses at the next piece.",
    "example_head": "It turns the last piece's {d} numbers into a score for each of the {upto}{rows} entries in the word list. The highest scores are its best guesses at the next piece.",
    "up_to": "up to ",
    # where the parameters go (#83)
    # guided experiments (#83)
    "exp_heads": "Do more look-back heads help?",
    "exp_words": "Letters vs words",
    "exp_depth": "Is depth or width better?",
    "exp_try_heads": "Try it",
    "exp_try_words": "Try it",
    "exp_try_depth": "Try deeper",
    "exp_try_width": "Try wider",
    "exp_no_heads": "This look-back has no heads to change.",
    "exp_no_room": "This thinking room has no width to change.",
    "predict_question": "Your question: “{question}”",
    "check_question": "You asked: “{question}”.",
    # progressive disclosure (#83)
}


def say(x, y, more, less):
    """Describe a cost ratio using the same wording in Builder and Quick test."""
    r = y / x
    if abs(r - 1) < 0.05:
        return T["about_same"]
    return T["about_multiplier"].format(mult=f"{r if r > 1 else 1 / r:.1f}") + " " + (more if r > 1 else less)

FF = (2, 4, 8)  # thinking-room width as a multiple of d_model: narrow / normal / wide
TOKENIZERS = ("byte", "bpe_fast", "bpe_gpt2")  # the small word list is the fast learned one
VS = (256, 8192, 50257)  # word-list sizes behind letters / small / large (the cap; less on little data)
WORDS = ("Letters", "Small word list", "Large word list")
WIDTHS = ("Narrow", "Normal", "Wide")
HEADS = (1, 4, 8, 12)  # 12 rarely divides the width, so picking it can round the width (width_for)
PART_SLOTS = ("embedding", "position", "head")  # the model-wide parts the tower shows; every block has BLOCK_SLOTS
PART_TERMS = dict(embedding="embedding", position="position", norm="norm", mixer="token_mixer", channel="mlp", head="output_head")
WORD_TERMS = ("bytes", "bpe", "gpt2_bpe")  # GLOSSARY entries behind WORDS
STEPS, ROWS = 1000, 16  # a catch-up job here; the Pretraining page is where length, rows and data are chosen
EXPECT = dict(
    more="More blocks: each word gets more rounds of processing, and every training step takes longer. With little data it may memorize rather than learn; watch the test line.",
    fewer="Fewer blocks: smaller and faster. It may not get its error as low.",
    heads="Different look-back heads: same size, same speed. Each block splits its looking into {how} views. Whether that helps at this size is something to test, not assume.",
    heads_width="{heads} heads need a width that splits evenly into them, so each word's width goes from {d} to {w}: {way}.",
    wider="More thinking room: more to learn in every block, so bigger and slower.",
    narrower="Less thinking room: smaller and faster, with less capacity in every block.",
    letters="Letters: the word table becomes tiny, but the model has to learn spelling, and your text turns into about 5 times as many pieces, so each step covers less text.",
    small="Small word list: built from your own data. A much smaller table; text splits into somewhat more pieces.",
    large="Large word list: GPT-2’s list. A big table, short sequences.",
)
WARM = dict(  # kind -> (chip, colour); the sentence itself is studio_warmstart.NOTES[kind]
    same=("weights carry over exactly", "live"), heads=("weights carry over, not exactly", "attention"),
    removed=("weights carry over, minus what was removed", "attention"), other=("weights carry over, changed parts start fresh", "attention"),
    scratch=("starts from scratch", "needs_training"))


def module_name(slot, short):
    return registry.get(slot, short).name


def module_tip(slot, short):
    """Hover help for a module: what it is and why you would pick it (studio_glossary), then the registry's exact description."""
    return f"{glossary_text(MODULE_TERMS[slot][short])}\n\n{registry.get(slot, short).blurb}"


def part_tip(slot):
    return glossary_text(PART_TERMS[slot])


def fields_of(slot, short):
    return {f["name"] for f in registry.fields(registry.get(slot, short))}


def knob_applies(s, key):
    """Whether look-back heads (H) or thinking room (W) mean anything for the all-blocks look-back / thinking-room choice."""
    slot, field = {"H": ("mixer", "heads"), "W": ("channel", "d_ff")}[key]
    return field in fields_of(slot, s["M"][slot])


def width_for(d, heads, even=False):
    """d_model `d`, or the nearest width (a tie goes wider) that splits into `heads` heads, of an even size when `even`
    (rotary positions rotate pairs), within the engine's 16..1024."""
    step = heads * (2 if even else 1)
    if d % step == 0:
        return d
    lo, hi = d // step * step, -(-d // step) * step
    return min((w for w in (hi, lo) if 16 <= w <= 1024), key=lambda w: abs(w - d))


def draft_width(s):
    """The width the draft `s` is written with: its own, or the one its heads round it to (write_spec)."""
    return width_for(s["d"], s["H"], s["M"]["position"] == "rope") if knob_applies(s, "H") else s["d"]


def block_parts(s, i):
    """{slot: module id} of block i: the all-blocks choice, or the block's own."""
    return {slot: s["O"].get((i, slot), s["M"][slot]) for slot in BLOCK_SLOTS}


def own_slots(s, i):
    """The slots block i has its own module in."""
    return [slot for slot in BLOCK_SLOTS if (i, slot) in s["O"]]


def choose(s, slot, short, block=None):
    """The choices after picking module `short` for `slot`: for every block (which drops that slot's per-block picks),
    or for block `block` only (picking the all-blocks module there drops its own pick)."""
    parts, own = dict(s["M"]), dict(s["O"])
    if block is None or slot not in BLOCK_SLOTS:
        parts[slot] = short
        own = {key: value for key, value in own.items() if key[1] != slot}
    elif short == parts[slot]:
        own.pop((block, slot), None)
    else:
        own[block, slot] = short
    return dict(s, M=parts, O=own)


def resize(s, n):
    """The choices with n blocks: per-block picks of blocks that are gone go with them."""
    return dict(s, L=n, O={key: value for key, value in s["O"].items() if key[0] < n})


def entry_for(slot, short, old, s):
    """Slot entry for module `short`: settings it shares with the module it replaces carry over; heads and a non-default
    thinking room follow the knobs."""
    names = fields_of(slot, short)
    config = {key: value for key, value in (old or {}).get("config", {}).items() if key in names}
    if "heads" in names and "heads" not in config:
        config["heads"] = s["H"]
    if "d_ff" in names and "d_ff" not in config and s["W"] != 1:
        config["d_ff"] = room_width(short, FF[s["W"]], s["d"])
    return {"id": short, "config": config}


def sizes(spec, v):
    """Size, time and memory of a design under word list v, summed from every module's own estimate (studio_estimate).
    params · table: the word table, position and head · time: training work per row of T pieces · memory: weights, gradients
    and Adam state at 16 bytes per parameter · activations: bytes kept per row for the backward pass."""
    vocab = tokenizer_vocab_size(spec, v)
    e = estimate(spec, vocab)
    d, t = spec["dims"]["d_model"], spec["dims"]["n_ctx"]
    # the estimate leaves the vocabulary matrices out of compute (Kaplan et al.); a step still pays for the output scores
    return dict(params=e["total"], table=e["embedding_params"], time=(e["train_flops_per_token"] + 6 * vocab * d) * t,
                memory=16 * e["total"], activations=e["activation_bytes_per_token"] * t, parts=param_parts(e["rows"]))


def tokenizer_vocab_size(spec, v):
    """The table cap selected in the tokenizer, or its fixed vocabulary size."""
    entry = spec["tokenizer"]
    configured = (entry.get("config") or {}).get("vocab_size")
    if configured:
        return configured
    from studio_spec import vocab_of
    return vocab_of(spec, fallback=VS[v])


PARAM_PARTS = (("table", "violet"), ("position", "dim"), ("blocks", "blue"), ("head", "pink"))  # bar order and colour


def param_parts(rows):
    """{table, position, blocks, head}: parameters per part, from studio_estimate's rows. head is the final norm and the word scores."""
    out = dict.fromkeys((key for key, _ in PARAM_PARTS), 0)
    for r in rows:
        key = "blocks" if r["block"] is not None else {"embedding": "table", "position": "position"}.get(r["slot"], "head")
        out[key] += r["params"]
    return out


RUN_SETTINGS = ("batch_size", "accumulation", "eval_every", "seed")  # train.run: absent means the engine's default (TRAINING_DEFAULTS)
TRAINING_SETTINGS = (*RUN_SETTINGS, "learning_rate", "weight_decay", "warmup", "dropout", "precision")
TRAINING_NAMES = dict(batch_size="batch size", accumulation="gradient accumulation", learning_rate="learning rate",
                      weight_decay="weight decay", warmup="warm-up steps", dropout="dropout", eval_every="evaluate every",
                      seed="seed", precision="precision")


def training_of(spec):
    """The training settings of a validated spec, as the choices hold them. warmup is None when its schedule has no warm-up."""
    train, run = spec["train"], spec["train"].get("run") or {}
    return dict({key: run.get(key, TRAINING_DEFAULTS[key]) for key in RUN_SETTINGS},
                learning_rate=train["optimizer"]["config"]["lr"], weight_decay=train["optimizer"]["config"]["weight_decay"],
                warmup=train["schedule"]["config"].get("warmup"), dropout=spec["dims"]["dropout"], precision=train["precision"]["id"])


def read_spec(spec):
    """ModelSpec -> the editable choices: L blocks, H heads, W thinking room, V word list, d and T, M the module of every part
    ({slot: id}; the block slots hold the all-blocks choice), O the per-block picks ({(block index, slot): id}) and the
    training settings (TRAINING_SETTINGS)."""
    spec = validate_spec(spec)
    f, block = _facts(spec), spec["block"]
    word = {"letters": 0, "small word list": 1, "large word list": 2}.get(f["words"], 1)
    parts = {slot: spec[slot]["id"] for slot in PART_SLOTS} | {slot: block[slot]["id"] for slot in BLOCK_SLOTS}
    own = {(o["index"], slot): o[slot]["id"] for o in spec["blocks"] for slot in BLOCK_SLOTS if slot in o}
    run = spec["train"].get("run") or {}
    return dict(L=f["blocks"], H=block["mixer"]["config"].get("heads") or 4, W=room_choice(block["channel"], f["width"]), V=word,
                d=f["width"], T=f["context"], U=tokenizer_vocab_size(spec, word),
                token_limit_millions=run.get("token_limit_millions", 0), M=parts, O=own, **training_of(spec))


def write_spec(base, s, was=None):
    """`base` with only the choices that differ between `was` (default: base's own) and `s` rewritten."""
    spec = copy.deepcopy(validate_spec(base))
    was = {**read_spec(spec), **(was or {})}
    vocab_size = s.get("U", was["U"])
    token_limit = s.get("token_limit_millions", was["token_limit_millions"])
    if s["L"] != was["L"]:
        spec["dims"]["n_layer"] = s["L"]
    for slot in PART_SLOTS:
        if s["M"][slot] != was["M"][slot]:
            spec[slot] = entry_for(slot, s["M"][slot], spec[slot], s)
    for slot in BLOCK_SLOTS:
        if s["M"][slot] != was["M"][slot]:
            spec["block"][slot] = entry_for(slot, s["M"][slot], spec["block"][slot], s)
    resized = False
    if s["d"] != was["d"]:
        spec["dims"]["d_model"] = s["d"]
        resized = True
    if s["T"] != was["T"]:
        spec["dims"]["n_ctx"] = s["T"]
    if s["H"] != was["H"] and "heads" in (mixer := spec["block"]["mixer"]["config"]):
        mixer["heads"] = s["H"]
        if mixer.get("kv_heads") and s["H"] % mixer["kv_heads"]:
            mixer["kv_heads"] = max(k for k in range(1, mixer["kv_heads"] + 1) if s["H"] % k == 0)
        d = spec["dims"]["d_model"]
        spec["dims"]["d_model"] = width_for(d, s["H"], spec["position"]["id"] == "rope")
        resized = spec["dims"]["d_model"] != d
    if (s["W"] != was["W"] or resized) and "d_ff" in spec["block"]["channel"]["config"]:
        spec["block"]["channel"]["config"]["d_ff"] = room_width(spec["block"]["channel"]["id"], FF[s["W"]], spec["dims"]["d_model"])
    for key in sorted(set(s["O"]) | set(was["O"])):
        if s["O"].get(key) == was["O"].get(key):
            continue
        index, slot = key
        if key in s["O"]:
            set_block(spec, index, slot, entry_for(slot, s["O"][key], block_at(spec, index)[slot], s))
        else:
            for override in spec["blocks"]:
                if override["index"] == index:
                    override.pop(slot, None)
    v = min(s["V"], len(TOKENIZERS) - 1)
    if s["V"] != was["V"]:
        spec["tokenizer"] = {"id": TOKENIZERS[v], "config": tokenizer_config(TOKENIZERS[v], spec)}
    tokenizer_id = TOKENIZERS[v]
    if vocab_size != was["U"] and tokenizer_id == "bpe_fast":
        spec["tokenizer"].setdefault("config", {})["vocab_size"] = vocab_size
    run = dict((spec["train"].get("run") or {}))
    if token_limit:
        run["token_limit_millions"] = token_limit
    else:
        run.pop("token_limit_millions", None)
    for key in RUN_SETTINGS:  # only what the person changed, so a default the spec never set is not written into it
        if s.get(key, was[key]) != was[key]:
            run[key] = s[key]
    if run:
        spec["train"]["run"] = run
    else:
        spec["train"].pop("run", None)
    train = spec["train"]
    for key, slot, name in (("learning_rate", "optimizer", "lr"), ("weight_decay", "optimizer", "weight_decay"), ("warmup", "schedule", "warmup")):
        if s.get(key, was[key]) != was[key] and name in train[slot]["config"]:
            train[slot]["config"][name] = s[key]
    if s.get("dropout", was["dropout"]) != was["dropout"]:
        spec["dims"]["dropout"] = s["dropout"]
    if s.get("precision", was["precision"]) != was["precision"]:
        train["precision"] = {"id": s["precision"], "config": {}}
        why = precision_unavailable(s["precision"], run.get("device", "auto"))
        if why:
            raise ValueError(why)
    return validate_spec(spec)


def precision_unavailable(short, device="auto"):
    """Why the precision module `short` cannot be picked for a run on `device` ("auto": where it will probably land), or ""."""
    from studio_modules.precision.policy import likely_device, unavailable
    return unavailable(registry.get("precision", short), likely_device(device))


def tokenizer_config(short, spec):
    """The settings a newly picked word list starts with: the learned one gets a vocabulary sized to the model, so the word table
    does not dominate a small design."""
    if short != "bpe_fast":
        return {}
    from studio_modules.tokenizer.bpe_fast import suggest_vocab
    return {"vocab_size": suggest_vocab(spec["dims"]["d_model"], spec["dims"]["n_layer"])}


def why_not(base, s, was, slot, short, block=None):
    """Why picking module `short` for `slot` (all blocks, or just `block`) cannot work in this design, or "" when it can.
    The 3D viewer's swaps get the same checks (validate_spec, then studio_validate's errors); only problems the pick adds count."""
    try:
        before = set(errors(write_spec(base, s, was)))
    except ValueError:
        before = set()
    try:
        after = write_spec(base, choose(s, slot, short, block), was)
    except ValueError as error:
        return str(error)
    return next((text for text in errors(after) if text not in before), "")


def part_options(base, s, was, slot, block=None):
    """[(module id, name, blurb, why it cannot be picked or "")] for one part's menu, in the registry's order."""
    current = s["O"].get((block, slot), s["M"][slot]) if block is not None else s["M"][slot]
    return [(m.short, m.name, m.blurb, "" if m.short == current else why_not(base, s, was, slot, m.short, block))
            for m in registry.for_slot(slot)]


def part_changes(cur, draft):
    """Plain-words chips for every module the draft changes: all-blocks choices first, then each block's own."""
    out = [T["part_change"].format(part=T["part_" + slot].lower(), name=module_name(slot, draft["M"][slot]))
           for slot in ("embedding", "position", *BLOCK_SLOTS, "head") if draft["M"][slot] != cur["M"][slot]]
    for i in range(draft["L"]):
        for slot in BLOCK_SLOTS:
            a, b = cur["O"].get((i, slot)), draft["O"].get((i, slot))
            if a != b and (b or i < cur["L"]):
                out.append(T["block_part_change"].format(n=i + 1, part=T["part_" + slot].lower(),
                                                         name=module_name(slot, b) if b else T["same_as_others"]))
    return out


def change_chips(old_spec, new_spec, cur, draft):
    """The not-saved chips: spec_diff's sentences, with its catch-all replaced by the parts that changed."""
    plain = spec_diff(old_spec, new_spec)
    out = [c for c in plain if c != OTHER_PARTS and not (cur["d"] != draft["d"] and c.startswith(("wider words", "narrower words")))] + part_changes(cur, draft)
    if cur["d"] != draft["d"]:
        out.append(f"hidden size {cur['d']:,} → {draft['d']:,}")
    if cur["U"] != draft["U"]:
        out.append(f"tokenizer size {cur['U']:,} → {draft['U']:,}")
    if cur["token_limit_millions"] != draft["token_limit_millions"]:
        cap = f"{draft['token_limit_millions']:,} million tokens" if draft["token_limit_millions"] else "off"
        out.append(f"training token limit {cap}")
    out += training_changes(cur, draft)
    if without_settings(recipe_of(validate_spec(old_spec))) != without_settings(recipe_of(new_spec)):
        out.append(T["recipe_change"].format(recipe=recipe_text(new_spec)))
    return out or plain


def training_changes(cur, draft):
    """One chip for every training setting the draft changes, with its old and new value."""
    out = []
    for key in TRAINING_SETTINGS:
        if key in cur and key in draft and cur[key] != draft[key]:
            a, b = cur[key], draft[key]
            both_whole = isinstance(a, int) and isinstance(b, int)
            out.append(f"{TRAINING_NAMES[key]} {a:,} → {b:,}" if both_whole else f"{TRAINING_NAMES[key]} {a} → {b}")
            if key == "precision" and (tradeoff := registry.get("precision", b).tradeoff):
                out.append(f"{b}: {tradeoff}")
    return out


def without_settings(recipe):
    """The recipe without the settings the training chips already name (learning rate, weight decay, warm-up)."""
    out = copy.deepcopy(recipe)
    for slot, key in (("optimizer", "lr"), ("optimizer", "weight_decay"), ("schedule", "warmup")):
        out[slot]["config"].pop(key, None)
    return out


def parts_changed(cur, draft):
    """True when a module changed anywhere both designs have it (blocks only one of them has are added or removed)."""
    return any(cur["M"][slot] != draft["M"][slot] for slot in PART_SLOTS) or \
        any(block_parts(cur, i) != block_parts(draft, i) for i in range(min(cur["L"], draft["L"])))


def new_issues(old_spec, new_spec):
    """Advice (warnings and notes, not errors) the new design gets that the old one did not."""
    old = {issue.text for issue in validate(old_spec)}
    return [issue.text for issue in validate(new_spec) if issue.level != "ERROR" and issue.text not in old]


def warm_kind(cur, draft):
    """How the old weights carry into the draft: scratch | heads | other | removed | same (the studio_warmstart rules)."""
    if (draft["V"], draft["U"]) != (cur["V"], cur["U"]):
        return "scratch"
    if draft["H"] != cur["H"]:
        return "heads"
    if parts_changed(cur, draft) or draft["d"] != cur["d"]:
        return "other"
    if draft["L"] < cur["L"] or draft["W"] < cur["W"]:
        return "removed"
    return "same"


def warm_outlook(cur, draft):
    """None when nothing changed, else {kind, carries, preserving, chip, state, text} for the cost panel."""
    if all(cur[k] == draft[k] for k in ("L", "H", "W", "V", "d", "T", "U", "token_limit_millions", "M", "O")) and not training_changes(cur, draft):
        return None
    kind = warm_kind(cur, draft)
    chip, state = WARM[kind]
    return dict(kind=kind, carries=kind != "scratch", preserving=kind == "same", chip=chip, state=state, text=NOTES[kind])


def ram_bytes():
    """Physical memory of this computer, or None when the OS does not say."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def fits(c, ram=None):
    """True/False: weights, gradients, optimizer state and activations for 16 rows fit in physical memory; None if unknown.
    `c` is a sizes() dict."""
    ram = ram or ram_bytes()
    if ram is None:
        return None
    return c["memory"] + ROWS * c["activations"] <= ram


# ---------------------------------------------------------------------------------------------- ready-made designs

PRESETS = (("modern", "modern_model"), ("classic", "classic_model"), ("starter", "starter_model"), ("tiny_llama_style", "llama_model"), ("gpt2_small_style", "gpt2_model"),
           ("gpt3_small_style", "gpt3_model"), ("gpt3_medium_style", "gpt3_model"))  # studio_arch key, GLOSSARY entry
SLOW_STEP = 30  # seconds per step past which a design is called very slow


def gpt2_words_ready():
    """True when GPT-2's word list is already on this computer (nothing is downloaded to find out)."""
    from studio_modules.tokenizer.bpe_gpt2 import GPT2Config, asset_paths
    return all(path.exists() for path in asset_paths(GPT2Config()))


def preset_specs(gpt2_words=False):
    """[(key, spec)] of studio_arch's presets in PRESETS order (Tiny Llama only when its modules are registered).
    With gpt2_words the GPT-2 and GPT-3 designs read text with GPT-2's word list, as the originals did."""
    from studio_preset import legacy_spec, modern_spec
    presets = dict(model_spec_presets(), modern=modern_spec("Modern preset")[0], classic=legacy_spec("Classic preset"))
    out = []
    for key, _term in PRESETS:
        if key not in presets:
            continue
        spec = presets[key]
        if gpt2_words and key.startswith("gpt"):
            spec = validate_spec(dict(spec, tokenizer={"id": "bpe_gpt2"}, name=spec["name"].replace("byte tokenizer", "GPT-2 word list")))
        out.append((key, spec))
    return out


def gpu_kind():
    """"cuda" or "mps" when this process already found a GPU, else "". It only reads an earlier check: importing torch or
    running a Metal/CUDA calculation from the Builder (on a worker thread, while Tk runs) crashes the app on macOS."""
    backend = sys.modules.get("studio_backend")
    if backend is None or not backend.gpu_status.cache_info().currsize:
        return ""
    status = backend.gpu_status()
    return status["kind"] if status["available"] else ""


def cached_probe(spec, rows=None):
    """The device probe (studio_probe) stored for this design on this machine at `rows` rows per step, or None: no GPU seen yet, or
    the design was never timed here, or the stored one was made with another torch or GPU (it is keyed by them). Never measures, and
    never touches torch before something else has."""
    if not gpu_kind():
        return None
    try:
        import studio_probe
        from studio_backend import gpu_status
        from studio_model import vocabulary_guess
        return studio_probe.cached(validate_spec(spec), vocabulary_guess({"architecture": spec}), ROWS if rows is None else rows, gpu_status())
    except Exception:  # noqa: BLE001 - a plan without a measurement is still a plan
        return None


def preset_term(key):
    return dict(PRESETS)[key]


def amount(n):
    """1.4 GB, 86.8M style: one decimal, the unit a person reads."""
    for size, unit in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if n >= size:
            return f"{n / size:.1f}{unit}"
    return str(round(n))


def gigabytes(n):
    return f"{n / 1e9:.1f} GB" if n >= 1e8 else f"{n / 1e6:.0f} MB"


def duration(seconds):
    """'about 5 minutes': a rough time, in the largest unit that keeps it a readable number."""
    if seconds < 1:
        return "under a second"
    for size, unit, limit in ((1, "seconds", 90), (60, "minutes", 5400), (3600, "hours", 172800), (86400, "days", float("inf"))):
        if seconds < limit:
            return f"about {seconds / size:.0f} {unit}"


def preset_cost(spec, ram=None, gpu=None, probe=None):
    """What a design costs before it is loaded: {params, need, fits, rows, step, slow, lines}. `lines` are the card's sentences:
    size, whether it fits this computer (and at how many rows per step it would), and how long a step takes.
    ram: bytes (None = ask the OS); gpu: "cuda"/"mps" when training will use one, "" or None for the processor alone.
    probe: `cached_probe(spec)`. When given, the step time is the measured one on the device `auto` really picks, and the line says which
    device is faster and by how much; otherwise the step is the fitted estimate and the line says so. `device` and `measured` report which."""
    s = read_spec(spec)
    c = sizes(spec, s["V"])
    ram = ram or ram_bytes()
    need = c["memory"] + ROWS * c["activations"]
    ok = None if ram is None else need <= ram
    rows = ROWS if ok or ram is None else max(0, int((ram - c["memory"]) // max(1, c["activations"])))
    kind = probe["device"] if probe else gpu or "cpu"
    step = step_seconds({"batch_size": ROWS, "context": s["T"], "layers": s["L"]}, c["params"], kind, probe=probe)
    lines = [T["preset_size"].format(params=amount(c["params"]), memory=gigabytes(c["memory"]))]
    if ram is None:
        lines.append(T["preset_fit_unknown"])
    elif ok:
        lines.append(T["preset_fits"].format(need=gigabytes(need), rows=ROWS, ram=gigabytes(ram)))
    elif rows:
        lines.append(T["preset_fits_fewer"].format(rows=ROWS, need=gigabytes(need), ram=gigabytes(ram), fit=rows))
    else:
        lines.append(T["preset_no_fit"].format(need=gigabytes(c["memory"] + c["activations"]), ram=gigabytes(ram)))
    if probe:
        speed = probe["tokens_per_second"]
        other = next((d for d in speed if d != kind), None)
        lines.append(T["preset_speed_measured"].format(device=kind.upper() if kind == "cpu" else "GPU", speed=speed[kind], step=duration(step),
                                                      other=speed[other], other_device="GPU" if other != "cpu" else "CPU")
                     if other else T["preset_speed_measured_one"].format(device=kind.upper() if kind == "cpu" else "GPU", speed=speed[kind],
                                                                         step=duration(step)))
    else:
        lines.append(T["preset_speed_gpu" if gpu else "preset_speed_cpu"].format(step=duration(step)) + (T["preset_speed_estimate"] if gpu else ""))
    if step > SLOW_STEP:
        lines[-1] += " " + T["preset_slow"].format(steps=STEPS, total=duration(step * STEPS))
    return dict(params=c["params"], need=need, fits=ok, rows=rows, step=step, slow=step > SLOW_STEP, lines=lines, device=kind, measured=bool(probe))


def passes_text(passes):
    """"one pass", "under one pass" or "2.5 passes": how many times a run reads the text."""
    if passes < .95:
        return "under one pass"
    return "one pass" if passes <= 1.05 else f"{passes:.1f} passes" if passes < 10 else f"{passes:.0f} passes"


def compute_lines(params, context, layers, preview, gpu="", probe=None):
    """[(sentence, bad)] for the cost panel: how much text a design of `params` parameters needs, against the word pieces
    `preview` (services.pretrain_preview; None while it is being counted) found, and how long a run of ROWS rows should be.
    Speed is the newest run's measured one, else studio_compute.throughput (the Pretraining page's time model)."""
    if preview is None:
        return [(T["compute_counting"], False)]
    pieces = preview.get("pieces")
    if not pieces:
        return [(T["compute_none"], False)]
    speed = preview.get("tokens_per_second") or throughput(ROWS, context, params, layers, probe["device"] if probe else gpu or "cpu", probe=probe)
    p = plan(params, pieces, tokens_per_second=speed, batch_tokens=ROWS * context)
    tail = (T["compute_under"] if p["under_trained"] else T["compute_repeats"] if p["over_repeated"]
            else T["compute_reread"].format(times=f"{p['epochs_for_optimal']:.1f}", most=MAX_EPOCHS) if params > p["data_params"]
            else T["compute_roomy"] if p["data_params"] >= 4 * params else T["compute_enough"])
    lines = [(T["compute_supports"].format(supported=amount(p["data_params"]), rule=TOKENS_PER_PARAM, params=amount(params)) + tail,
              p["under_trained"]),
             (T["compute_steps"].format(steps=p["steps"], rows=ROWS, passes=passes_text(p["epochs"]))
              + T["compute_time"].format(time=duration(p["seconds_needed"])), False)]
    if p["over_repeated"]:
        lines.append((T["compute_repeat"].format(rule=TOKENS_PER_PARAM, times=f"{p['epochs_for_optimal']:.0f}", most=MAX_EPOCHS), True))
    return lines


def lr_text(result, current=None):
    """The sentence under "Find a learning rate" for a finished (or failed or stopped) range test; "" before one has run.
    `current` is the rate the design trains with now."""
    if not result:
        return ""
    if result.get("error"):
        return T["lr_failed"].format(error=result["error"])
    if result.get("cancelled"):
        return T["lr_stopped"]
    failed = result.get("failed_at") or (result["rates"][-1] if result.get("rates") else None)
    if not result.get("suggestion"):
        return T["lr_blew_up_early"].format(rate=f"{failed:.2g}") if result.get("diverged") and failed else T["lr_flat"]
    text = T["lr_result"].format(rate=f"{result['suggestion']:.2g}", steepest=f"{result['rates'][result['steepest']]:.2g}",
                                 current=f"{current:.2g}" if current else "its own default")
    if result.get("diverged"):
        text += " " + T["lr_blew_up"].format(rate=f"{failed:.2g}")
    return text


def recipe_text(spec):
    """The optimizer and schedule a spec carries, in one line: "AdamW · learning rate 0.0006 · β2 0.95 · … · cosine to 10%"."""
    optimizer, schedule = spec["train"]["optimizer"], spec["train"]["schedule"]
    parts = [registry.get("optimizer", optimizer["id"]).name]
    for key, label in (("lr", "learning rate"), ("beta1", "β1"), ("beta2", "β2"), ("weight_decay", "weight decay")):
        if key in optimizer["config"]:
            parts.append(f"{label} {optimizer['config'][key]:g}")
    sched = registry.get("schedule", schedule["id"]).name
    if "min_lr" in schedule["config"]:
        sched += f" to {schedule['config']['min_lr']:.0%}"
    return " · ".join(parts + [sched])


def recipe_of(spec):
    """The parts of the training recipe a design carries (not the run settings the Pretraining page chooses)."""
    return {slot: spec["train"][slot] for slot in ("objective", "optimizer", "schedule", "init")}


def blocks_change_pair(versions):
    """(older, newer, blocks before, blocks after) for the latest block-count change in the history, else None. Reads no files."""
    vs = [v for v in versions if v.spec]
    for i in range(len(vs) - 1, 0, -1):
        a, b = _facts(vs[i - 1].spec)["blocks"], _facts(vs[i].spec)["blocks"]
        if a != b:
            return vs[i - 1], vs[i], a, b
    return None


def last_blocks_change(versions, facts=None, compares=None):
    """(from, to, v_from, v_to, attribution) for the latest block-count change in the history, else None. The attribution
    (studio_ia.attribution) has the signed move in held-back error and what else differed. `facts(number)` and `compares()`
    give the runs' facts and the saved Compare runs (Services.run_facts and Services.compare_runs, which read files: the page
    gathers them on a worker). With no `facts` they are not known yet, and the attribution says it is still checking."""
    pair = blocks_change_pair(versions)
    if pair is None:
        return None
    old, new, a, b = pair
    known = facts is not None
    return a, b, old.number, new.number, attribute(old, new, facts(old.number) if known else None, facts(new.number) if known else None,
                                                   compares() if compares else ())


def design_expectation(s, spec):
    """(headline, details) for the choices in `s` and their spec: what a design this size and word list can be expected to write."""
    c = sizes(spec, s["V"])
    return expectation(c["params"], TOKENIZERS[s["V"]], embedding_params=c["table"])


def expectations(cur, draft, versions=(), issues=(), facts=None, compares=None):
    """What to expect from each change, then the advice (`issues`, from new_issues) the new design gets."""
    a, b, out = cur, draft, []
    if b["L"] > a["L"]:
        out.append(EXPECT["more"])
    if b["L"] < a["L"]:
        out.append(EXPECT["fewer"])
    if b["L"] != a["L"] and (last := last_blocks_change(versions, facts, compares)):
        out.append(blocks_change_text(*last))
    if b["H"] != a["H"]:
        out.append(EXPECT["heads"].format(how="more, narrower" if b["H"] > a["H"] else "fewer, wider"))
        if (w := draft_width(b)) != b["d"]:
            out.append(EXPECT["heads_width"].format(heads=b["H"], d=b["d"], w=w, way="a little bigger and slower" if w > b["d"] else "a little smaller and faster"))
    if b["W"] > a["W"]:
        out.append(EXPECT["wider"])
    if b["W"] < a["W"]:
        out.append(EXPECT["narrower"])
    if b["V"] != a["V"]:
        out.append(EXPECT[("letters", "small", "large")[b["V"]]])
    for slot in ("embedding", "position", *BLOCK_SLOTS, "head"):
        if b["M"][slot] != a["M"][slot]:
            module = registry.get(slot, b["M"][slot])
            out.append(T["part_expect"].format(part=T["part_" + slot], name=module.name, blurb=module.blurb))
    out.extend(issues)
    return out or [T["change_something"]]


# Guess, train, check: shared desktop and web learning loop.

GUESSES = dict(error=("lower", "higher"), step=("faster", "slower"), memory=("less", "more"))  # dimension -> its two directions


SAME = .05  # a step time or memory within 5% is the same (the cost panel's "about the same")


def _positive(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0


def train_memory(c):
    """Bytes training needs at ROWS rows per step: weights, gradients and Adam state, plus activations. `c` is a sizes() dict."""
    return c["memory"] + ROWS * c["activations"]


def prediction_questions(old_spec, new_spec):
    """The dimensions worth a guess for the change from old_spec to new_spec: held-back error always, step time and memory
    only when the design's estimate says they move (a guess the cost panel already answers "about the same" is no question)."""
    a, b = sizes(old_spec, read_spec(old_spec)["V"]), sizes(new_spec, read_spec(new_spec)["V"])
    out = ["error"]
    if abs(b["time"] / a["time"] - 1) >= SAME:
        out.append("step")
    if abs(train_memory(b) / train_memory(a) - 1) >= SAME:
        out.append("memory")
    return out


def direction(key, before, after):
    """Which way dimension `key` moved: error by the signed move in bits per character (a move under 0.005 is "same"), step
    time and memory by ratio (within SAME is "same"). One of GUESSES[key], or "same"; None when either number is missing."""
    if key == "error":
        change = signed_move(before, after)
        if change is None:
            return None
        return "same" if round(abs(change), 2) == 0 else "lower" if change < 0 else "higher"
    if not (_positive(before) and _positive(after)):
        return None
    if abs(after / before - 1) < SAME:
        return "same"
    return GUESSES[key][after > before]


def scored(v):
    """A version's held-back bits per character, or None."""
    bpc = (v.scorecard or {}).get("bpc") if v else None
    return bpc if isinstance(bpc, (int, float)) and not isinstance(bpc, bool) else None


def trained(v):
    return bool(v and (v.best_checkpoint or scored(v) is not None))


def results(old, new, speeds=None):
    """{dimension: {before, after, way, measured}} for the change from version `old` to `new`. Error is each version's held-back
    score (two scoring methods are not a move); step time is measured when `speeds` ({number: seconds per step}) has both,
    else the design's estimate; memory is always the design's estimate."""
    speeds = speeds or {}
    same_method = (old.scorecard or {}).get("bpc_method") == (new.scorecard or {}).get("bpc_method")
    a, b = (scored(old), scored(new)) if same_method else (None, scored(new))
    out = {"error": dict(before=a, after=b, way=direction("error", a, b), measured=True)}
    try:
        ca, cb = sizes(old.spec, read_spec(old.spec)["V"]), sizes(new.spec, read_spec(new.spec)["V"])
    except (KeyError, TypeError, ValueError):
        ca = cb = None
    sa, sb = speeds.get(old.number), speeds.get(new.number)
    if _positive(sa) and _positive(sb):
        out["step"] = dict(before=sa, after=sb, way=direction("step", sa, sb), measured=True)
    elif ca:
        out["step"] = dict(before=ca["time"], after=cb["time"], way=direction("step", ca["time"], cb["time"]), measured=False)
    if ca:
        ma, mb = train_memory(ca), train_memory(cb)
        out["memory"] = dict(before=ma, after=mb, way=direction("memory", ma, mb), measured=False)
    return out


def seconds_text(s):
    return f"{s * 1000:.0f} ms" if s < 1 else f"{s:.1f} s"


def ratio_text(key, r):
    """'about 1.4× slower', 'about the same'."""
    return say(r["before"], r["after"], *({"step": (T["cost_slower"], T["cost_faster"]), "memory": (T["cost_more"], T["cost_less"])}[key]))


def outcome(key, guess, r, is_trained):
    """"hit", "miss" or None (not checked yet) for one guess against its result `r` (a results() entry, or None)."""
    if not is_trained or not r or r["way"] is None:
        return None
    return "hit" if r["way"] == guess else "miss"


def result_sentence(key, r):
    if key == "error":
        if r["before"] is None:
            return ""
        return T["result_error_flat" if r["way"] == "same" else "result_error"].format(a=r["before"], b=r["after"])
    if key == "step" and r["measured"]:
        return T["result_step"].format(a=seconds_text(r["before"]), b=seconds_text(r["after"]))
    return T["result_step_estimated" if key == "step" else "result_memory"].format(say=ratio_text(key, r))


def guess_check(old, new, speeds=None):
    """The check of version `new`'s stored guess: {lines, hits, misses, state}, or None when it has none (or no version before it).
    Each line is "You expected lower error. It went from 2.31 to 2.18 bits per character. A hit."; before training it says when
    it will be checked. state is a StateTray state: live (all hits), attention (a miss), neutral (nothing checked yet)."""
    guesses = {k: g for k, g in (getattr(new, "prediction", None) or {}).items() if g in GUESSES.get(k, ())}
    if not guesses or old is None:
        return None
    rs, done = results(old, new, speeds), trained(new)
    lines, hits, misses = [], 0, 0
    for key in GUESSES:
        if key not in guesses:
            continue
        guess, r = guesses[key], rs.get(key)
        line = T["guess_expected"].format(what=T["guess_" + guess])
        result = outcome(key, guess, r, done)
        if not done:
            line += " " + T["guess_waiting"].format(n=new.number)
        elif result is None:
            line += " " + T["guess_no_score"].format(n=new.number)
        else:
            line += " " + " ".join(filter(None, (result_sentence(key, r), T["guess_" + result])))
        hits, misses = hits + (result == "hit"), misses + (result == "miss")
        lines.append(line)
    return dict(lines=lines, hits=hits, misses=misses, state="attention" if misses else "live" if hits else "neutral")


def page_said(old, new):
    """The first "what to expect" sentence for the change from version `old` to `new` (what the page said before it trained)."""
    try:
        return expectations(read_spec(old.spec), read_spec(new.spec))[0]
    except (KeyError, TypeError, ValueError):
        return ""


def check_text(old, new, speeds=None):
    """(state, kicker, text) for the guess tray of version `new`, or None when it has no guess to show."""
    check = guess_check(old, new, speeds)
    if check is None:
        return None
    said = page_said(old, new)
    question = (getattr(new, "prediction", None) or {}).get("question")
    text = " ".join(([T["check_question"].format(question=question)] if question else []) + check["lines"]
                    + ([T["guess_page_said"].format(text=said)] if said and said != T["change_something"] else []))
    return check["state"], T["guess_kicker"].format(n=new.number), text


def split_steps(cur, draft):
    """The draft as a chain of one-change choices [s1, …, sk]: each is the one before with one more of the draft's choices
    applied, and the last has all of them. Word list, blocks, heads and thinking room first (the knobs act on the modules they
    find), then each part, then each block's own pick. A knob the draft's module has no use for (heads left over from before a
    swap) is no step of its own: it rides along with the last one."""
    steps, s = [], dict(cur)

    def take(patch):
        nonlocal s
        s = dict(s, **patch)
        steps.append(s)
    if draft["V"] != s["V"]:
        take(dict(V=draft["V"], U=draft["U"]))
    if draft["L"] != s["L"]:
        take(resize(s, draft["L"]))
    for key in ("H", "W"):
        if draft[key] != s[key] and knob_applies(draft, key):
            take({key: draft[key]})
    for key in ("d", "T", "U", "token_limit_millions"):
        if draft[key] != s[key]:
            take({key: draft[key]})
    if patch := {key: draft[key] for key in TRAINING_SETTINGS if key in draft and draft[key] != s.get(key)}:
        take(patch)  # the training settings are one step: they say how to train, not what the network is
    for slot in (*PART_SLOTS, *BLOCK_SLOTS):
        if draft["M"][slot] != s["M"][slot]:
            take(dict(M=dict(s["M"], **{slot: draft["M"][slot]})))
    for key in sorted(set(s["O"]) | set(draft["O"])):
        if draft["O"].get(key) != s["O"].get(key):
            own = dict(s["O"])
            if key in draft["O"]:
                own[key] = draft["O"][key]
            else:
                own.pop(key)
            take(dict(O=own))
    if steps:
        steps[-1] = dict(draft)
    return steps


def split_plan(base, was, cur, draft):
    """[(choices, spec, chips)] saving the draft as one version per change, each written onto the one before; [] when that is
    not possible: fewer than two changes, or a step on the way would be a design that cannot work. A step that changes nothing
    in the spec (heads on a look-back without heads) is folded into the next."""
    out, prev_s, prev_spec = [], was, write_spec(base, was, was)
    try:
        for s in split_steps(cur, draft):
            spec = write_spec(base, s, was)
            if spec == prev_spec:
                continue
            if errors(spec):
                return []
            out.append((s, spec, change_chips(prev_spec, spec, prev_s, s)))
            prev_s, prev_spec = s, spec
    except ValueError:
        return []
    return out if len(out) > 1 and out[-1][1] == write_spec(base, draft, was) else []


def split_jobs(start, plan, mode, parent_checkpoint, datasets, run_dir, new_id=lambda: uuid.uuid4().hex[:8]):
    """The pretrain jobs for split versions start, start + 1, …, chained: the first starts as `mode` from `parent_checkpoint`,
    each later one starts from the best checkpoint the one before writes (its output folder is set here, so the path is known
    before it runs; the queue runs them in order). A step that changes the word list trains from scratch."""
    jobs, before = [], parent_checkpoint
    for i, (s, spec, _chips) in enumerate(plan):
        prev = plan[i - 1][0] if i else None
        step_mode = "from_scratch" if (i == 0 and mode == "from_scratch") or (prev and (s["V"], s["U"]) != (prev["V"], prev["U"])) else "catch_up"
        job_id = new_id()
        job = catchup_job(job_id, start + i, spec, step_mode, before, datasets)
        out = str(Path(run_dir) / f"queue_{job_id}") if run_dir else None
        job.params["output_dir"] = out
        jobs.append(job)
        before = str(Path(out) / "best.npz") if out else None
    return jobs


def notebook(versions, speeds=None, facts=None):
    """One row per version, oldest first: {number, change, effects, guesses, note, state, question}. effects says what the version did to
    held-back error, step time and memory against the version before it; guesses is the stored guess with ✓ or ✗; note says
    when the comparison is not about one change (several changes at once, or the runs also differed in data, steps, learning
    rate or extra training: studio_ia.attribution.run_changes on Services.run_facts). Reads no files: `speeds` and `facts`
    ({number: …}) are what a worker read."""
    speeds, facts, rows, old = speeds or {}, facts or {}, [], None
    for v in versions:
        row = dict(number=v.number, change=v.change or "", effects="", guesses="", note="", state="neutral",
                   question=(getattr(v, "prediction", None) or {}).get("question") or "")
        done = trained(v)
        if old is None or not (old.spec and v.spec):
            row["effects"] = T["notebook_error_only"].format(b=scored(v)) if scored(v) is not None else \
                T["notebook_not_trained" if not done else "notebook_no_score"]
        else:
            rs = results(old, v, speeds)
            err = rs["error"]
            parts = [T["notebook_not_trained"] if not done else T["notebook_no_score"] if err["after"] is None
                     else T["notebook_error"].format(a=err["before"], b=err["after"]) if err["before"] is not None
                     else T["notebook_error_only"].format(b=err["after"])]
            for key in ("step", "memory"):
                if key in rs:
                    parts.append(T["notebook_" + key].format(say=ratio_text(key, rs[key]))
                                 + ("" if rs[key]["measured"] else T["notebook_estimated"]))
            row["effects"] = ", ".join(parts)
            check = guess_check(old, v, speeds)
            if check:
                marks = {"hit": " ✓", "miss": " ✗"}
                guesses = getattr(v, "prediction", None) or {}
                said = [T["guess_" + guesses[k]] + marks.get(outcome(k, guesses[k], rs.get(k), done), "")
                        for k in GUESSES if guesses.get(k) in GUESSES[k]]
                row["guesses"] = T["notebook_guessed"].format(guesses=", ".join(said))
                row["state"] = check["state"]
            elif v.number > 1:
                row["guesses"] = T["notebook_no_guess"]
            try:
                many = len(change_chips(old.spec, v.spec, read_spec(old.spec), read_spec(v.spec)))
            except (KeyError, TypeError, ValueError):
                many = 0
            ran = run_changes(facts[old.number], facts[v.number])[0] if facts.get(old.number) and facts.get(v.number) else []
            ran = [name for name in ran if name not in ("starting weights", "tokens")]  # a catch-up always starts from the version before
            row["note"] = "; ".join(filter(None, (T["notebook_many"].format(count=many) if many > 1 else "",
                                                  T["notebook_also"].format(names=", ".join(ran)) if ran else "")))
        rows.append(row)
        old = v
    return rows


def guesses_from(picked):
    """{dimension: direction} from the save-time chips ({dimension: the label picked, or None}); unpicked ones are left out."""
    out = {}
    for key, label in picked.items():
        labels = [T["predict_" + d] for d in GUESSES[key]]
        if label in labels:
            out[key] = GUESSES[key][labels.index(label)]
    return out


def notebook_lines(rows):
    """The notebook card's lines: one per version, or why there are none (still reading: rows is None)."""
    if rows is None:
        return [T["notebook_reading"]]
    return [notebook_line(row) for row in rows] or [T["notebook_empty"]]


def notebook_line(row):
    """'Version 3 · more blocks (2 → 4) → error 2.31 → 2.18, steps about 1.4× slower · asked “…” · you guessed lower error ✓'."""
    head = f"{T['notebook_version'].format(n=row['number'])} · {row['change']}" if row["change"] else T["notebook_version"].format(n=row["number"])
    asked = T["notebook_asked"].format(question=row["question"]) if row.get("question") else ""
    return " · ".join(filter(None, (f"{head} → {row['effects']}", asked, row["guesses"], row["note"])))


MISSING = {1: "Built from your text: add some in Your data to see these pieces.",
           2: "GPT-2’s list could not be downloaded. It is fetched the first time you train with it."}


def pieces_for(v, lists):
    """([(label, state)], count line) for the example sentence under word list v. `lists` = services.word_lists(), or None
    while it is still being read. The count line puts every list's count side by side, in the order of the buttons."""
    if lists is None:
        lists = {"tokenizers": (cut.Tokenizer("byte"), None, None), "sentence": cut.EXAMPLE}
        note = "Reading your text…"
    else:
        note = MISSING.get(v)
    line = " · ".join("–" if n is None else f"{n:,}" for n in cut.counts(lists)) + " pieces"
    tok = lists["tokenizers"][v]
    if tok is None:
        return [(note, "neutral")], line
    chips = [(label, "pieces" if known else "attention") for label, known in cut.chips(tok, lists["sentence"])]
    return chips, line


def pill_text(job, base):
    """Header pill: what is training now, else where the current version stands."""
    if job:
        n = job.params.get("version", base.number if base else 0)
        if not n:
            return "Training (no version yet)"
        scratch = job.params.get("mode") == "from_scratch"
        return f"Version {n} {'is training' if scratch or job.kind != 'pretrain' else 'is catching up'}"
    return f"Version {base.number} is {base.status}" if base else "No saved version yet"


def catchup_job(job_id, number, spec, mode, parent_checkpoint, datasets):
    """The pretrain job for a saved version, in the shape the Pretraining page queues."""
    cfg_ = {"batch_size": ROWS, **config_from_spec(spec), "steps": STEPS, "validation": .1}  # the version's own batch size when it has one
    scratch = mode == "from_scratch"
    return Job(job_id, "pretrain", f"Version {number} {'from scratch' if scratch else 'catch-up'}",
               params={"config": cfg_, "datasets": list(datasets), "mode": mode, "init_from": None if scratch else parent_checkpoint,
                       "output_dir": None, "user_text": False, "version": number,
                       **({"token_budget": cfg_["token_limit_millions"] * 1_000_000} if cfg_.get("token_limit_millions") else {})})


# ---------------------------------------------------------------------------------------------- learn the parts (#83)

def example_pieces(v, lists):
    """The example sentence cut by word list v ([label]), or None while the lists are read or that list is not built."""
    tok = lists["tokenizers"][v] if lists else None
    return [label for label, _known in cut.chips(tok, lists["sentence"])] if tok is not None else None


def example_text(part, s, short, lists=None):
    """What part `part` (running module `short`) does to the example sentence under the draft's choices `s`."""
    return example_for(part, s, short, example_pieces(s["V"], lists), lists["sentence"] if lists else None)


def example_for(part, s, short, pieces, sentence):
    """example_text with the example sentence already cut by the draft's word list: `pieces` ([label] or None) of `sentence`."""
    d, rows = draft_width(s), s["U"]
    upto = T["up_to"] if s["V"] == 1 else ""  # built from your text: the size is a cap
    last, before = (pieces[-1].strip() or pieces[-1], len(pieces) - 1) if pieces and len(pieces) > 1 else (None, 0)
    slot = "norm" if part == "final_norm" else part
    if slot == "norm" or slot == "mixer" or slot == "channel":
        if registry.get(slot, short).skip:
            return T["example_skipped"]
    if part == "embedding":
        if pieces:
            return T["example_table_pieces"].format(sentence=sentence, count=len(pieces), cs="" if len(pieces) == 1 else "s", pieces=" · ".join(pieces), d=d,
                                                    upto=upto, rows=f"{rows:,}")
        return T["example_table"].format(d=d, upto=upto, rows=f"{rows:,}")
    if part == "position":
        return T["example_position"].format(t=s["T"])
    if part == "mixer":
        heads = s["H"] if "heads" in fields_of("mixer", short) else None
        if heads:
            key = "example_mixer_heads_pieces" if last else "example_mixer_heads"
            return T[key].format(last=last, before=before, s="" if before == 1 else "s", heads=heads, size=d // heads)
        return T["example_mixer_pieces" if last else "example_mixer"].format(last=last, before=before, s="" if before == 1 else "s",
                                                                          v="s" if before == 1 else "")
    if part == "channel":
        if "d_ff" in fields_of("channel", short):
            return T["example_channel_room"].format(d=d, room=FF[s["W"]] * d)
        return T["example_channel"]
    if slot == "norm":
        return T["example_norm"].format(d=d)
    return T["example_head_pieces" if last else "example_head"].format(last=last, d=d, upto=upto, rows=f"{rows:,}")


def explain(part, s, block=None, lists=None):
    """What the side panel says about one part of the draft `s`: {title, what, module, module_text, example}. `part` is a
    part key (embedding, position, mixer, channel, norm, final_norm, head); `block` is the block a sub-layer was clicked in (None: a model-wide part or the all-blocks choice).
    `lists` is services.word_lists() (None while it is read)."""
    return explain_for(part, s, block, example_pieces(s["V"], lists), lists["sentence"] if lists else None)


def explain_for(part, s, block=None, pieces=None, sentence=None):
    """explain() with the example sentence already cut: `pieces` ([label] or None while the lists are read) of `sentence`."""
    slot = "norm" if part == "final_norm" else part
    short = block_parts(s, block)[slot] if block is not None and slot in BLOCK_SLOTS else s["M"][slot]
    title = T["final_norm"] if part == "final_norm" else T["part_" + slot]
    if block is not None:
        title = T["explain_in_block"].format(part=title, n=block + 1)
    what = f"{T['final_norm_tip']}\n\n{part_tip('norm')}" if part == "final_norm" else part_tip(slot)
    return dict(title=title, what=what, module=T["explain_now"].format(name=module_name(slot, short)), module_text=module_tip(slot, short),
                example=example_for(part, s, short, pieces, sentence))


EXPERIMENTS = (("heads", "exp_heads"), ("words", "exp_words"), ("depth", "exp_depth"), ("width", "exp_depth"))  # depth and width ask one question


def experiment_draft(cur, key):
    """(choices, "") with the one change that puts question `key` to the current version's choices `cur`, or (None, why not)."""
    if key == "heads":
        if not knob_applies(cur, "H"):
            return None, T["exp_no_heads"]
        up = [h for h in HEADS if h > cur["H"]]
        return dict(cur, H=up[0] if up else max(h for h in HEADS if h < cur["H"])), ""
    if key == "words":
        return dict(cur, V=1 if cur["V"] == 0 else 0), ""
    if key == "depth":
        return resize(cur, cur["L"] + 1 if cur["L"] < 12 else cur["L"] - 1), ""
    if not knob_applies(cur, "W"):
        return None, T["exp_no_room"]
    return dict(cur, W=cur["W"] + 1 if cur["W"] < 2 else cur["W"] - 1), ""


def experiment_rows(cur):
    """[(question, [(key, label, why it cannot be tried or "")])] for the starting-questions card, in EXPERIMENTS order."""
    out = {}
    for key, question in EXPERIMENTS:
        out.setdefault(T[question], []).append((key, T["exp_try_" + key], experiment_draft(cur, key)[1]))
    return list(out.items())


def held_question(experiment, draft):
    """The question a starting question set up, while the draft is still exactly that one change; else None.
    `experiment` = (question, the draft it set up) or None."""
    return experiment[0] if experiment and experiment[1] == draft else None
