"""Build your model: the picture is the editor. Draft edits live on this page until saved."""
from __future__ import annotations

import copy
import math
import os
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox
import uuid
from pathlib import Path

import studio_registry as registry
from studio_arch import model_spec_presets
from studio_compute import MAX_EPOCHS, TOKENS_PER_PARAM, plan, throughput
from studio_estimate import estimate
from studio_glossary import MODULE_TERMS, glossary_text
from studio_slots import BLOCK_SLOTS
from studio_spec import block_at, config_from_spec, default_spec, set_block, validate_spec
from studio_ux import step_seconds
from studio_validate import errors, validate
from studio_versions import OTHER_PARTS, _facts, spec_diff
from studio_warmstart import NOTES
from studio_ia import pieces as cut
from studio_ia.attribution import attribute, blocks_change_text, run_changes, signed_move
from studio_ia import threads
from studio_ia.copy import BUILDER as T
from studio_ia.expectation import expectation
from studio_ia.pages.base import Page
from studio_ia.services import Job, if_added, plain_error, queued_outlook
from studio_ia.theme import C, GUTTER, font, tk_label
from studio_ia.widgets import BpcPlot, Card, Chip, GuessChips, Kicker, Progress, Segmented, StateTray, Toggle, _activate, button, info_icon, tip

FF = (2, 4, 8)  # thinking-room width as a multiple of d_model: narrow / normal / wide
TOKENIZERS = ("byte", "bpe_custom", "bpe_gpt2", "word")
VS = (256, 8192, 50257, 8192)  # word-list sizes behind letters / small / large / whole words (the cap; less on little data)
WORDS = ("Letters", "Small word list", "Large word list", "Whole words")
WIDTHS = ("Narrow", "Normal", "Wide")
HEADS = (1, 4, 8)
PART_SLOTS = ("embedding", "position", "head")  # the model-wide parts the tower shows; every block has BLOCK_SLOTS
PART_TERMS = dict(embedding="embedding", position="position", norm="norm", mixer="token_mixer", channel="mlp", head="output_head")
WORD_TERMS = ("bytes", "bpe", "gpt2_bpe", "whole_words")  # GLOSSARY entries behind WORDS
PROMPT = "The largest planet in our solar system is"
STEPS, ROWS = 1000, 16  # a catch-up job here; the Pretraining page is where length, rows and data are chosen
LINKS = (("compare", "compare_runs"), ("quicktest", "quick_test"))  # the bottom row; the look-inside pages hang off the parts (PART_LINKS)
EXPECT = dict(
    more="More blocks: each word gets more rounds of processing, and every training step takes longer. With little data it may memorize rather than learn; watch the test line.",
    fewer="Fewer blocks: smaller and faster. It may not get its error as low.",
    heads="Different look-back heads: same size, same speed. Each block splits its looking into {how} views. Whether that helps at this size is something to test, not assume.",
    wider="More thinking room: more to learn in every block, so bigger and slower.",
    narrower="Less thinking room: smaller and faster, with less capacity in every block.",
    letters="Letters: the word table becomes tiny, but the model has to learn spelling, and your text turns into about 5 times as many pieces, so each step covers less text.",
    small="Small word list: built from your own data. A much smaller table; text splits into somewhat more pieces.",
    large="Large word list: GPT-2’s list. A big table, short sequences.",
    words="Whole words: one entry per frequent word in your own data. Short sequences, but a word that is not in the list is lost.",
)
WARM = dict(  # kind -> (chip, colour); the sentence itself is studio_warmstart.NOTES[kind]
    same=("weights carry over exactly", "live"), heads=("weights carry over, not exactly", "attention"),
    removed=("weights carry over, minus what was removed", "attention"), other=("weights carry over, changed parts start fresh", "attention"),
    scratch=("starts from scratch", "needs_training"))
TRAINING_TEXT = "Chat switches to {new} only when it beats version {n} on held-back text, or when you pick it."
NO_TEXT = "There is no plain text to train on yet. Add some in Your data, then queue it."
NO_PARENT = "Version {n} has no trained weights to start from, so the next version trains from scratch."
NO_CURVE = "No error curve yet: version {n} has not trained on this computer. It appears here once it has."


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
        config["d_ff"] = FF[s["W"]] * s["d"]
    return {"id": short, "config": config}


def sizes(spec, v):
    """Size, time and memory of a design under word list v, summed from every module's own estimate (studio_estimate).
    params · table: the word table, position and head · time: training work per row of T pieces · memory: weights, gradients
    and Adam state at 16 bytes per parameter · activations: bytes kept per row for the backward pass."""
    e = estimate(spec, VS[v])
    d, t = spec["dims"]["d_model"], spec["dims"]["n_ctx"]
    # the estimate leaves the vocabulary matrices out of compute (Kaplan et al.); a step still pays for the output scores
    return dict(params=e["total"], table=e["embedding_params"], time=(e["train_flops_per_token"] + 6 * VS[v] * d) * t,
                memory=16 * e["total"], activations=e["activation_bytes_per_token"] * t, parts=param_parts(e["rows"]))


PARAM_PARTS = (("table", "violet"), ("position", "dim"), ("blocks", "blue"), ("head", "pink"))  # bar order and colour


def param_parts(rows):
    """{table, position, blocks, head}: parameters per part, from studio_estimate's rows. head is the final norm and the word scores."""
    out = dict.fromkeys((key for key, _ in PARAM_PARTS), 0)
    for r in rows:
        key = "blocks" if r["block"] is not None else {"embedding": "table", "position": "position"}.get(r["slot"], "head")
        out[key] += r["params"]
    return out


def param_bars(a, b, width=300):
    """([(part, pixels)] now, [(part, pixels)] draft): each design's parts stacked, both drawn to the scale of the larger one."""
    top = max(sum(a.values()), sum(b.values()), 1)
    return tuple([(key, round(parts[key] / top * width)) for key, _ in PARAM_PARTS] for parts in (a, b))


def param_shares(parts):
    """[(part, 'word table 12%')] in bar order."""
    total = sum(parts.values()) or 1
    return [(key, T["params_share"].format(name=T["params_" + key], share=f"{parts[key] / total:.0%}")) for key, _ in PARAM_PARTS]


def param_note(parts):
    """A sentence when the word table is most of the model, else ""."""
    return T["params_table_heavy"] if parts["table"] >= .5 * (sum(parts.values()) or 1) else ""


def say(x, y, more, less):
    r = y / x
    if abs(r - 1) < 0.05:
        return T["about_same"]
    return T["about_multiplier"].format(mult=f"{r if r > 1 else 1 / r:.1f}") + " " + (more if r > 1 else less)


def bar(x, y):
    m = max(x, y)
    return round(x / m * 100), round(y / m * 100)


def read_spec(spec):
    """ModelSpec -> the editable choices: L blocks, H heads, W thinking room, V word list, d and T, M the module of every part
    ({slot: id}; the block slots hold the all-blocks choice) and O the per-block picks ({(block index, slot): id})."""
    spec = validate_spec(spec)
    f, block = _facts(spec), spec["block"]
    word = {"letters": 0, "small word list": 1, "large word list": 2, "whole words": 3}.get(f["words"], 1)
    room = (block["channel"]["config"].get("d_ff") or 4 * f["width"]) / f["width"]
    parts = {slot: spec[slot]["id"] for slot in PART_SLOTS} | {slot: block[slot]["id"] for slot in BLOCK_SLOTS}
    own = {(o["index"], slot): o[slot]["id"] for o in spec["blocks"] for slot in BLOCK_SLOTS if slot in o}
    return dict(L=f["blocks"], H=block["mixer"]["config"].get("heads") or 4, W=0 if room < 4 else 1 if room == 4 else 2, V=word,
                d=f["width"], T=f["context"], M=parts, O=own)


def write_spec(base, s, was=None):
    """`base` with only the choices that differ between `was` (default: base's own) and `s` rewritten."""
    spec = copy.deepcopy(validate_spec(base))
    was = was or read_spec(spec)
    if s["L"] != was["L"]:
        spec["dims"]["n_layer"] = s["L"]
    for slot in PART_SLOTS:
        if s["M"][slot] != was["M"][slot]:
            spec[slot] = entry_for(slot, s["M"][slot], spec[slot], s)
    for slot in BLOCK_SLOTS:
        if s["M"][slot] != was["M"][slot]:
            spec["block"][slot] = entry_for(slot, s["M"][slot], spec["block"][slot], s)
    if s["H"] != was["H"] and "heads" in spec["block"]["mixer"]["config"]:
        spec["block"]["mixer"]["config"]["heads"] = s["H"]
    if s["W"] != was["W"] and "d_ff" in spec["block"]["channel"]["config"]:
        spec["block"]["channel"]["config"]["d_ff"] = FF[s["W"]] * s["d"]
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
    if s["V"] != was["V"]:
        spec["tokenizer"] = {"id": TOKENIZERS[s["V"]], "config": {}}
    return validate_spec(spec)


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
    out = [c for c in plain if c != OTHER_PARTS] + part_changes(cur, draft)
    if recipe_of(validate_spec(old_spec)) != recipe_of(new_spec):
        out.append(T["recipe_change"].format(recipe=recipe_text(new_spec)))
    return out or plain


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
    if draft["V"] != cur["V"]:
        return "scratch"
    if draft["H"] != cur["H"]:
        return "heads"
    if parts_changed(cur, draft):
        return "other"
    if draft["L"] < cur["L"] or draft["W"] < cur["W"]:
        return "removed"
    return "same"


def warm_outlook(cur, draft):
    """None when nothing changed, else {kind, carries, preserving, chip, state, text} for the cost panel."""
    if all(cur[k] == draft[k] for k in ("L", "H", "W", "V", "M", "O")):
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

PRESETS = (("starter", "starter_model"), ("tiny_llama_style", "llama_model"), ("gpt2_small_style", "gpt2_model"),
           ("gpt3_small_style", "gpt3_model"), ("gpt3_medium_style", "gpt3_model"))  # studio_arch key, GLOSSARY entry
SLOW_STEP = 30  # seconds per step past which a design is called very slow


def gpt2_words_ready():
    """True when GPT-2's word list is already on this computer (nothing is downloaded to find out)."""
    from studio_modules.tokenizer.bpe_gpt2 import GPT2Config, asset_paths
    return all(path.exists() for path in asset_paths(GPT2Config()))


def preset_specs(gpt2_words=False):
    """[(key, spec)] of studio_arch's presets in PRESETS order (Tiny Llama only when its modules are registered).
    With gpt2_words the GPT-2 and GPT-3 designs read text with GPT-2's word list, as the originals did."""
    presets = model_spec_presets()
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


def preset_cost(spec, ram=None, gpu=None):
    """What a design costs before it is loaded: {params, need, fits, rows, step, slow, lines}. `lines` are the card's sentences:
    size, whether it fits this computer (and at how many rows per step it would), and how long a step takes.
    ram: bytes (None = ask the OS); gpu: "cuda"/"mps" when training will use one, "" or None for the processor alone."""
    s = read_spec(spec)
    c = sizes(spec, s["V"])
    ram = ram or ram_bytes()
    need = c["memory"] + ROWS * c["activations"]
    ok = None if ram is None else need <= ram
    rows = ROWS if ok or ram is None else max(0, int((ram - c["memory"]) // max(1, c["activations"])))
    step = step_seconds({"batch_size": ROWS, "context": s["T"], "layers": s["L"]}, c["params"], gpu or "cpu")
    lines = [T["preset_size"].format(params=amount(c["params"]), memory=gigabytes(c["memory"]))]
    if ram is None:
        lines.append(T["preset_fit_unknown"])
    elif ok:
        lines.append(T["preset_fits"].format(need=gigabytes(need), rows=ROWS, ram=gigabytes(ram)))
    elif rows:
        lines.append(T["preset_fits_fewer"].format(rows=ROWS, need=gigabytes(need), ram=gigabytes(ram), fit=rows))
    else:
        lines.append(T["preset_no_fit"].format(need=gigabytes(c["memory"] + c["activations"]), ram=gigabytes(ram)))
    lines.append(T["preset_speed_gpu" if gpu else "preset_speed_cpu"].format(step=duration(step)))
    if step > SLOW_STEP:
        lines[-1] += " " + T["preset_slow"].format(steps=STEPS, total=duration(step * STEPS))
    return dict(params=c["params"], need=need, fits=ok, rows=rows, step=step, slow=step > SLOW_STEP, lines=lines)


def passes_text(passes):
    """"one pass", "under one pass" or "2.5 passes": how many times a run reads the text."""
    if passes < .95:
        return "under one pass"
    return "one pass" if passes <= 1.05 else f"{passes:.1f} passes" if passes < 10 else f"{passes:.0f} passes"


def compute_lines(params, context, layers, preview, gpu=""):
    """[(sentence, bad)] for the cost panel: how much text a design of `params` parameters needs, against the word pieces
    `preview` (services.pretrain_preview; None while it is being counted) found, and how long a run of ROWS rows should be.
    Speed is the newest run's measured one, else studio_compute.throughput (the Pretraining page's time model)."""
    if preview is None:
        return [(T["compute_counting"], False)]
    pieces = preview.get("pieces")
    if not pieces:
        return [(T["compute_none"], False)]
    speed = preview.get("tokens_per_second") or throughput(ROWS, context, params, layers, gpu or "cpu")
    p = plan(params, pieces, tokens_per_second=speed, batch_tokens=ROWS * context)
    tail = (T["compute_under"] if p["under_trained"] else T["compute_repeats"] if p["over_repeated"]
            else T["compute_reread"].format(times=f"{p['epochs_for_optimal']:.1f}", most=MAX_EPOCHS) if params > p["data_params"]
            else T["compute_roomy"] if p["data_params"] >= 4 * params else T["compute_enough"])
    about = "about " if preview.get("estimated") else ""
    lines = [(T["compute_supports"].format(supported=about + amount(p["data_params"]), rule=TOKENS_PER_PARAM, params=amount(params)) + tail,
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


def preset_draft(spec):
    """(draft choices, anchor) that load `spec` into the Builder. The draft is written onto the preset itself, so what the picture
    has no control for (width, reading window, the training recipe) comes with it, and the old version stays the comparison."""
    s = read_spec(spec)
    return dict(s), (validate_spec(spec), s)


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
    if b["W"] > a["W"]:
        out.append(EXPECT["wider"])
    if b["W"] < a["W"]:
        out.append(EXPECT["narrower"])
    if b["V"] != a["V"]:
        out.append(EXPECT[("letters", "small", "large", "words")[b["V"]]])
    for slot in ("embedding", "position", *BLOCK_SLOTS, "head"):
        if b["M"][slot] != a["M"][slot]:
            module = registry.get(slot, b["M"][slot])
            out.append(T["part_expect"].format(part=T["part_" + slot], name=module.name, blurb=module.blurb))
    out.extend(issues)
    return out or [T["change_something"]]


# ---------------------------------------------------------------------------------------------- guess, train, check (#83)

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


# ---------------------------------------------------------------------------------------------- one change at a time (#83)

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
        take(dict(V=draft["V"]))
    if draft["L"] != s["L"]:
        take(resize(s, draft["L"]))
    for key in ("H", "W"):
        if draft[key] != s[key] and knob_applies(draft, key):
            take({key: draft[key]})
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
        step_mode = "from_scratch" if (i == 0 and mode == "from_scratch") or (prev and s["V"] != prev["V"]) else "catch_up"
        job_id = new_id()
        job = catchup_job(job_id, start + i, spec, step_mode, before, datasets)
        out = str(Path(run_dir) / f"queue_{job_id}") if run_dir else None
        job.params["output_dir"] = out
        jobs.append(job)
        before = str(Path(out) / "best.npz") if out else None
    return jobs


# ---------------------------------------------------------------------------------------------- lab notebook (#83)

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


# ---------------------------------------------------------------------------------------------- learn the parts (#83)

PART_LINKS = dict(embedding=("trace", "inside3d"), position=("trace", "inside3d"), mixer=("head_zoom", "value_map", "head_lab", "inside3d"),
                  channel=("inside3d", "trace"), norm=("trace",), final_norm=("trace",), head=("loss", "trace"))  # part -> pages that show it work
HEAD_PAGES = ("head_zoom", "value_map", "head_lab")  # about attention heads: only for a look-back that has them
LINK_LABELS = dict(trace="trace", head_zoom="inside_head", value_map="value_map", loss="loss", head_lab="head_lab", inside3d="3d")


def example_pieces(v, lists):
    """The example sentence cut by word list v ([label]), or None while the lists are read or that list is not built."""
    tok = lists["tokenizers"][v] if lists else None
    return [label for label, _known in cut.chips(tok, lists["sentence"])] if tok is not None else None


def example_text(part, s, short, lists=None):
    """What part `part` (running module `short`) does to the example sentence under the draft's choices `s`."""
    d, rows, pieces = s["d"], VS[s["V"]], example_pieces(s["V"], lists)
    upto = T["up_to"] if s["V"] in (1, 3) else ""  # built from your text: the size is a cap
    last, before = (pieces[-1].strip() or pieces[-1], len(pieces) - 1) if pieces and len(pieces) > 1 else (None, 0)
    slot = "norm" if part == "final_norm" else part
    if slot == "norm" or slot == "mixer" or slot == "channel":
        if registry.get(slot, short).skip:
            return T["example_skipped"]
    if part == "embedding":
        if pieces:
            return T["example_table_pieces"].format(sentence=lists["sentence"], count=len(pieces), cs="" if len(pieces) == 1 else "s", pieces=" · ".join(pieces), d=d,
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
    """What the side panel says about one part of the draft `s`: {title, what, module, module_text, example, links}. `part` is a
    PART_LINKS key; `block` is the block a sub-layer was clicked in (None: a model-wide part or the all-blocks choice).
    `lists` is services.word_lists() (None while it is read). links are [(route, label, block)] to the pages that show it work."""
    slot = "norm" if part == "final_norm" else part
    short = block_parts(s, block)[slot] if block is not None and slot in BLOCK_SLOTS else s["M"][slot]
    title = T["final_norm"] if part == "final_norm" else T["part_" + slot]
    if block is not None:
        title = T["explain_in_block"].format(part=title, n=block + 1)
    what = f"{T['final_norm_tip']}\n\n{part_tip('norm')}" if part == "final_norm" else part_tip(slot)
    links = [(route, T[LINK_LABELS[route]], block or 0) for route in PART_LINKS[part]
             if route not in HEAD_PAGES or "heads" in fields_of(slot, short)]
    return dict(title=title, what=what, module=T["explain_now"].format(name=module_name(slot, short)), module_text=module_tip(slot, short),
                example=example_text(part, s, short, lists), links=links)


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


UNLOCK = 3  # trained versions before module menus and per-block choices show by themselves
ADVANCED = "builder_advanced"  # the remembered Advanced toggle (Services.setting)


def trained_count(versions):
    return sum(trained(v) for v in versions)


def advanced_view(versions, setting=None):
    """Module menus and per-block choices show: as the person set the Advanced toggle, else once UNLOCK versions have trained."""
    return bool(setting) if setting is not None else trained_count(versions) >= UNLOCK


MISSING = {1: "Built from your text: add some in Your data to see these pieces.",
           2: "GPT-2’s list could not be downloaded. It is fetched the first time you train with it.",
           3: "Built from your text: add some in Your data to see these pieces."}


def pieces_for(v, lists):
    """([(label, state)], count line) for the example sentence under word list v. `lists` = services.word_lists(), or None
    while it is still being read. The count line puts every list's count side by side, in the order of the buttons."""
    if lists is None:
        lists = {"tokenizers": (cut.Tokenizer("byte"), None, None, None), "sentence": cut.EXAMPLE}
        note = "Reading your text…"
    else:
        note = MISSING.get(v)
    line = " · ".join("–" if n is None else f"{n:,}" for n in cut.counts(lists)) + " pieces"
    tok = lists["tokenizers"][v]
    if tok is None:
        return [(note, "neutral")], line
    chips = [(label, "pieces" if known else "attention") for label, known in cut.chips(tok, lists["sentence"])]
    lost = sum(state == "attention" for _, state in chips)
    if lost:
        line += f", {lost} word{'s' if lost > 1 else ''} lost to “unknown”"
    return chips, line


def chart_args(old, new, refs):
    """BpcPlot keywords: the current version's curve bold, the new version's faint and continuing from the old one's end.
    `old`/`new` are run_curve dicts (or None); refs = references() dict. Missing numbers stay missing."""
    old, new = old or {}, new or {}
    base_x = old.get("total") or (old["steps"][-1][0] if old.get("steps") else 0)
    shift = lambda pts: [(x + base_x, y) for x, y in pts or []]  # noqa: E731
    named = (("uniform guess", "uniform_bpc"), ("counting word frequencies", "unigram_bpc"), ("best so far", "best_so_far_bpc"))
    return dict(smooth=list(old.get("smooth") or []), heldout=list(old.get("heldout") or []), steps=shift(new.get("steps")),
                refs=[(name, refs[key]) for name, key in named if refs and refs.get(key) is not None],
                total=base_x + (new.get("total") or 0) or None,
                band=refs.get("noise_band") if refs else None)


def version_name(n):
    """'version 3', or 'no version yet' for 0: the number before the first design is saved."""
    return f"version {n}" if n else "no version yet"


def curve_note(n, phase):
    """The one line shown in place of the chart while there is no curve to draw."""
    if phase == "training":
        return T["no_curve_starting"]
    if n:
        return NO_CURVE.format(n=n)
    return T["no_curve_first"] if phase == "edit" else T["no_curve_waiting"]


def answer_rows(n, trained, phase):
    """[(version number, why it cannot answer or "")] for the 'when answering' card; [] while neither version has trained.
    `trained` = the numbers among n and n + 1 that have trained weights. Version 0 is not a version, so it never gets a row."""
    if not trained:
        return []
    return [(k, "" if k in trained else T["compare_later"] if phase == "edit" and k == n + 1 else T["not_trained"])
            for k in (n, n + 1) if k]


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
    cfg_ = {**config_from_spec(spec), "steps": STEPS, "batch_size": ROWS, "validation": .1}
    scratch = mode == "from_scratch"
    return Job(job_id, "pretrain", f"Version {number} {'from scratch' if scratch else 'catch-up'}",
               params={"config": cfg_, "datasets": list(datasets), "mode": mode, "init_from": None if scratch else parent_checkpoint,
                       "output_dir": None, "user_text": False, "version": number})


def text_sets(summary):
    """Ids of the plain-text datasets from services.data('text') (a dict or a DataSummary)."""
    sets = summary.get("sets", []) if isinstance(summary, dict) else getattr(summary, "sets", [])
    return [s["id"] for s in sets]


class BuilderPage(Page):
    title, lead = T["h1"], T["lead"]

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.default = default_spec()
        self.phase, self.base, self.saved, self.job_id = "edit", None, None, None
        self.cur = read_spec(self.default)
        self.draft = dict(self.cur)
        self.anchor = None  # (spec, choices) when a design from another page was reopened here
        self._staged, self.dismissed = None, None
        self.tray = self.chips = self.costs = self.expect = None
        self.pieces_box = None
        self.note = None
        self.chip_texts, self.cost_say, self.expect_texts, self.actions = [], {}, [], {}
        self.guide_text = self.guide_details = self.hover_text = ""
        self.old_curve = self.new_curve = self.refs = None
        self.lists = None  # services.word_lists(), read on the slow lane
        self.attr_cache, self.attr_fresh, self.attr_busy = {}, set(), set()  # run facts and Compare runs per pair of versions
        self.painting_expect = False
        self.plot_args, self.plot = {}, None
        self._chart_token = None
        self.answers, self.answer_q, self.answer_stop, self.answer_labels, self.workers = {}, queue.Queue(), threading.Event(), {}, []
        self.pill_bar = self.pill_label = None
        self._curve_at = 0.
        self.open, self.scope = set(), {}  # blocks shown with their parts; blocks whose picks apply to them only
        self.draft_spec = None
        self.presets = preset_specs(gpt2_words_ready())
        self.compute_preview, self._asking = None, False  # services.pretrain_preview: the size of your text, None until counted
        self.lr_stop, self.lr_result, self.lr_for = None, None, None  # the learning-rate test: running (its stop event), its answer, the design it ran on
        self.preset_key = None  # the ready-made design being looked at
        self.asking, self.guesses, self._predict_for = None, [], None  # ("one" | "split", the draft spec) while the save-time guesses show
        self.picks = {}  # {(version index, dimension): label} picked so far, kept when the page is rebuilt
        self.chain = set()  # ids of split jobs that each start from the one before: their order is not to be changed here
        self._split_key = self._split_plan = None  # the split of the draft into one-change versions, kept until the draft changes
        self.speeds, self.notebook_rows, self._notebook_token = {}, None, None  # measured step times and the lab notebook (slow lane)
        self.check_shown, self.notebook_texts = None, []
        self.explaining, self.explained = None, None  # (part, block or None) the side panel explains, and what it said
        self.experiment = None  # (question, the draft it set up) from a starting question
        self.advanced, self.advanced_set = False, None  # module menus and per-block choices shown; the toggle as clicked here

    # state ------------------------------------------------------------------
    @property
    def n(self):
        return self.base.number if self.base else 0

    @property
    def nv(self):
        return f"version {self.n + 1}"

    def version(self, number):
        return next((v for v in self.services.versions() if v.number == number), None)

    def base_spec(self):
        return validate_spec(self.base.spec if self.base and self.base.spec else self.default)

    def job_for(self, number):
        """The queued or running pretrain job of a version, with its phase."""
        q = self.services.queue()
        if q.job and q.job.params.get("version") == number:
            return q.job, "training"
        return next(((j, "queued") for j in q.up_next if j.params.get("version") == number), (None, None))

    def refresh(self):
        current = self.services.current_version()
        if self.phase == "edit":
            self._restore_saved(current)
        if self.phase == "edit":
            changed = (current.number if current else 0) != self.n
            self.base = current
            self.cur = read_spec(current.spec) if current and current.spec else read_spec(self.default)
            if changed:
                self.draft, self.anchor, self._staged = dict(self.cur), None, None
                self.services.stage_draft(None)
            staged = self.services.draft_spec()
            if staged is not None and staged != self._staged:  # a design reopened from Compare runs
                try:
                    self.draft, self.anchor = read_spec(staged), (staged, read_spec(staged))
                except ValueError:
                    staged = None
            elif staged is None and self._staged is not None:  # discarded from Quick test
                self.draft, self.anchor = dict(self.cur), None
            self._staged = staged
        self.attr_fresh.clear()  # read the run files again (they may have changed); the last result shows meanwhile
        self.load_chart()
        self.render()
        self.load_lists()
        self.load_notebook()

    def _restore_saved(self, current):
        """A saved version nobody has trained yet shows its coral tray, wherever it was saved (e.g. from Quick test)."""
        if not (current and current.status == "saved" and current.parent and current.spec and current.number != self.dismissed):
            return
        parent = self.version(current.parent)
        if not (parent and parent.spec):
            return
        job, phase = self.job_for(current.number)
        self.base, self.saved = parent, current
        self.cur, self.draft, self.anchor = read_spec(parent.spec), read_spec(current.spec), None
        self.job_id, self.phase = (job.id, phase) if job else (None, "saved")

    def load_chart(self):
        """Curves and reference lines from the real run files; nothing invented when there are none. Scoring the held-back text
        can take a while the first time: that read runs on the shell's worker and the page redraws when it arrives."""
        base = (self.version(self.base.number) or self.base) if self.base else None  # as stored now (a split step may have trained since)
        newer = self.version(self.n + 1) if self.phase != "edit" else None
        self._chart_token = token = object()  # a later load_chart supersedes one still on the worker

        def work():
            try:
                refs = self.services.references(base) if base and base.best_checkpoint else None
            except (OSError, ValueError, KeyError, StopIteration):
                refs = None
            return self.curve_of(base), self.curve_of(newer), refs

        def apply(chart, late):
            if self._chart_token is not token:
                return
            self.old_curve, self.new_curve, self.refs = chart
            self.plot_args = chart_args(self.old_curve, self.new_curve, self.refs)
            if late:
                self.render()
        if self.services.ready(work):
            apply(work(), False)
        else:
            self.quietly(work, lambda chart: apply(chart, True))

    def load_lists(self):
        """The four word lists fitted on your text, for the pieces card. Fitting reads a sample of the text (and fetches
        GPT-2's list once), so it runs on the slow lane; services keep the result until the stored data changes."""
        def work():
            try:
                return self.services.word_lists()
            except (OSError, ValueError):
                return None

        def apply(lists):
            if lists is not None and lists is not self.lists:
                self.lists = lists
                if self.pieces_box is not None and self.pieces_box.winfo_exists():
                    self._paint_pieces()
                    self._paint_explain()
        self.quietly(work, apply, lane="slow")

    def load_notebook(self):
        """The lab notebook and the measured step times behind the guess check. Reading every version's run files (metrics,
        and run facts that hash the data file once) runs on the slow lane; the card and the guess tray repaint when it arrives."""
        services = self.services
        self._notebook_token = token = object()

        def work():  # no widgets here
            versions = services.versions()
            speeds, facts = {}, {}
            for v in versions:
                try:
                    speeds[v.number] = services.run_speed(v.number)
                except Exception:  # an unreadable run is a missing number, never a stuck card
                    speeds[v.number] = None
                try:
                    facts[v.number] = services.run_facts(v.number)
                except Exception:
                    facts[v.number] = {}
            return speeds, notebook(versions, speeds, facts)

        def apply(value):
            if self._notebook_token is not token:
                return
            self.speeds, self.notebook_rows = value
            self._paint_notebook()
            if self.tray is not None and self.tray.winfo_exists():
                self._paint_tray()
        self.quietly(work, apply, lane="slow")

    def curve_of(self, version):
        runs = getattr(version, "runs", None) or ([str(Path(version.best_checkpoint).parent)] if version and version.best_checkpoint else [])
        if not runs:
            return None
        try:
            return self.services.run_curve(runs[-1])
        except (OSError, ValueError, KeyError, StopIteration):
            return None

    def on_event(self, kind, payload):
        if kind == "versions":
            return self.refresh()
        if kind == "answer":
            return
        if kind != "queue":
            return
        q = self.services.queue()
        if self.phase == "queued" and q.job and q.job.id == self.job_id:
            self.phase = "training"
            return self.render()
        if self.phase in ("queued", "training") and self.job_id and not (q.job and q.job.id == self.job_id) \
                and not any(j.id == self.job_id for j in q.up_next):
            self.load_chart()  # our job left the queue: it finished or was removed
            return self.render()
        if self.phase == "queued" and self.outlook() not in self.tray_text:  # paused, held, or the job ahead finished
            return self.render()
        if self.pill_bar is not None:  # progress only: do not rebuild the page for every step
            self._paint_pill(q)
            if self.phase == "training" and time.time() - self._curve_at > 2:
                self._curve_at = time.time()
                job = q.job
                out = job.params.get("output_dir") if job else None
                if out:
                    try:
                        self.new_curve = self.services.run_curve(out)
                    except (OSError, ValueError, KeyError, StopIteration):
                        return
                    self.plot_args = chart_args(self.old_curve, self.new_curve, self.refs)
                    if self.plot:
                        self.plot.set(**self.plot_args)
                    elif self.plot_args["steps"] or self.plot_args["smooth"]:
                        self.render()  # the first points have arrived: the note becomes a chart

    def edit(self, **patch):
        if self.phase == "edit":
            self.draft.update(patch)
            self.stage()
            self._paint_design()

    def spec_base(self):
        """(spec, its choices) the draft is written onto: a design reopened from another page, else the current version."""
        return self.anchor or (self.base_spec(), self.cur)

    def spec_for(self, s):
        base, was = self.spec_base()
        return write_spec(base, s, was)

    def changes(self):
        try:
            return change_chips(self.base_spec(), self.spec_for(self.draft), self.cur, self.draft)
        except ValueError:
            return []

    def word_changed(self):
        return self.draft["V"] != self.cur["V"]

    def needs_scratch(self):
        """The word list changed, or there is no trained version to start from."""
        return self.word_changed() or not (self.base and self.base.best_checkpoint)

    def start_note(self):
        if not self.word_changed() and self.needs_scratch():
            return NO_PARENT.format(n=self.n)
        return NOTES[warm_kind(self.cur, self.draft)]

    def expectations(self):
        issues = new_issues(self.base_spec(), self.draft_spec) if self.draft_spec else ()
        versions = self.services.versions()
        facts, compares = self.attribution_inputs(versions) if self.draft["L"] != self.cur["L"] else (None, None)
        return expectations(self.cur, self.draft, versions, issues, facts, compares)

    def attribution_inputs(self, versions):
        """(facts, compares) for "last time blocks changed", from what a worker already read, else (None, None). The run
        files (the first read hashes the whole data file) and the Compare runs are read on the slow lane, once per pair of
        versions until something changes, and the line is painted again when they arrive."""
        pair = blocks_change_pair(versions)
        if pair is None:
            return None, None
        key = (pair[0].number, pair[1].number)
        if key not in self.attr_fresh and key not in self.attr_busy:
            self.attr_busy.add(key)
            services = self.services

            def work():  # no widgets here
                try:
                    return key, services.run_facts(key[0]), services.run_facts(key[1]), services.compare_runs()
                except Exception:  # whatever went wrong, the line says it could not check and the pair is not stuck busy
                    return key, {}, {}, []

            def apply(value):
                self.attr_busy.discard(key)
                self.attr_fresh.add(key)
                self.attr_cache[key] = value[1:]
                label = getattr(self, "guide_label", None)
                if not self.painting_expect and label is not None and label.winfo_exists():
                    self._paint_expect()
            self.quietly(work, apply, lane="slow")
        got = self.attr_cache.get(key)
        return (None, None) if got is None else ((lambda n, f=dict(zip(key, got[:2])): f[n]), lambda c=got[2]: c)

    def explain_part(self, part, block=None):
        """Open the side panel on one part (a PART_LINKS key), in block `block` or the all-blocks choice."""
        self.explaining = (part, block)
        self._paint_explain()

    def close_explain(self):
        self.explaining = None
        self._paint_explain()

    def follow(self, route, block):
        """Open a look-inside page on the part the panel explains: Inside a head and Value map open at that block's first head."""
        page = self.navigate(route)
        if route == "head_zoom" and hasattr(page, "select"):
            page.select(block, 0)
        elif route == "value_map" and hasattr(page, "pick"):
            page.pick(block, 0)

    def try_question(self, key):
        """A starting question: the draft becomes the current version with that one change, and the question goes with it."""
        draft, why = experiment_draft(self.cur, key)
        if self.phase != "edit" or draft is None:
            self.note = why or T["exp_not_now"]
            return self._paint_tray()
        self.draft, self.anchor, self.asking = draft, None, None  # an open guess tray was about the old draft
        self.experiment = (T[dict(EXPERIMENTS)[key]], copy.deepcopy(draft))
        self.stage()
        self._paint_design()

    def set_advanced(self, on):
        """The Advanced toggle: remembered by services (written on the slow lane), shown at once."""
        self.advanced_set = on
        services = self.services
        self.quietly(lambda: services.set_setting(ADVANCED, on), lambda _none: None, lane="slow")
        self.render()

    def preview_preset(self, key):
        """Show a ready-made design's cost; nothing changes until it is loaded."""
        self.preset_key = key
        self._paint_preset()

    def use_preset(self):
        """Load the design being looked at into the draft: an ordinary unsaved change against the current version."""
        spec = dict(self.presets).get(self.preset_key)
        if self.phase == "edit" and spec:
            self.draft, self.anchor = preset_draft(spec)
            self.stage()
            self._paint_design()

    def import_gpt2(self):
        """Queue the one-time GPT-2 download and conversion; the new version gets GPT-2's weights when the queue finishes it."""
        if self.phase != "edit" or not messagebox.askyesno(T["import_gpt2"], T["import_gpt2_ask"]):
            return
        try:
            self.services.import_pretrained("gpt2")
            self.note = T["import_gpt2_started"]
        except ValueError as error:
            self.note = str(error)
        self.render()

    def pick(self, slot, short, block=None):
        """Module `short` for `slot`: in every block, or in block `block` only."""
        self.edit(**choose(self.draft, slot, short, block))

    def knob(self, key, value):
        """A seg click. Heads and thinking room mean nothing for a look-back or thinking room without them."""
        if key not in ("H", "W") or knob_applies(self.draft, key):
            self.edit(**{key: value})
        else:
            self._paint_design()  # put the seg back

    def stage(self):
        """Publish the unsaved draft so Quick test and Compare runs can use it."""
        spec = None
        if self.phase == "edit" and self.changes():
            try:
                spec = self.spec_for(self.draft)
            except ValueError:
                spec = None
        if spec != self._staged:
            self._staged = spec
            self.services.stage_draft(spec)

    # actions ----------------------------------------------------------------
    def discard(self):
        self.draft, self.anchor, self.experiment = dict(self.cur), None, None
        self.stage()
        self._paint_design()

    def ask_save(self):
        """"Save as version N": first a quick guess at what the change will do (skippable); version 1 has nothing to compare."""
        if self.base is None or self.draft_spec is None:
            return self.save()
        self.asking, self.picks = ("one", self.draft_spec), {}
        self._paint_tray()

    def ask_split(self):
        if self.split_plan():
            self.asking, self.picks = ("split", self.draft_spec), {}
            self._paint_tray()

    def cancel_guesses(self):
        self.asking = None
        self._paint_tray()

    def save_guessed(self, skip=False):
        """Save what the guesses were asked about, with the guesses picked (none when skipped)."""
        kind = self.asking[0] if self.asking else "one"
        picks = [{} if skip else guesses_from({key: chip.value for key, chip in chips.items()}) for chips in self.guesses]
        self.asking = None
        if kind == "split":
            self._paint_tray()  # the guess buttons go before the slow read of the data list, so a second click cannot save again
            return self.split(picks)
        self.save(picks[0] if picks else None)

    def split_plan(self):
        """split_plan for the draft, or [] (a design reopened from another page, or nothing saved yet, is saved whole)."""
        if self.anchor is not None or self.base is None or self.phase != "edit":
            return []
        key = copy.deepcopy((self.cur, self.draft))
        if key != self._split_key:
            self._split_key, self._split_plan = key, split_plan(self.base_spec(), self.cur, self.cur, self.draft)
        return self._split_plan

    def guess_steps(self):
        """[(title, old spec, new spec)] for each version the guesses are about."""
        if not self.asking:
            return []
        if self.asking[0] == "one":
            return [("", self.base_spec(), self.draft_spec)]
        out, old = [], self.base_spec()
        for i, (_s, spec, chips) in enumerate(self.split_plan()):
            out.append((T["predict_step_title"].format(n=self.n + 1 + i, change=", ".join(chips)), old, spec))
            old = spec
        return out

    def save(self, prediction=None):
        try:
            spec = self.spec_for(self.draft)
        except ValueError as error:
            self.note = str(error)
            return self.render()
        if problems := errors(spec):
            self.note = problems[0]
            return self.render()
        question = held_question(self.experiment, self.draft)
        prediction = dict(prediction or {}, **({"question": question} if question else {}))
        self.saved = self.services.save_version(spec, ", ".join(self.changes()) or ("first design" if self.base is None else "no change"),
                                                prediction or None)
        self.phase, self._staged, self.anchor, self.experiment = "saved", None, None, None
        self.services.stage_draft(None)
        self.load_chart()
        self.render()

    def queue_it(self, from_scratch):
        self.slow(lambda: text_sets(self.services.data("text")), lambda sets: self._queue(sets, from_scratch))

    def split(self, predictions=()):
        """Save the draft as one version per change and queue them one after another, each starting from the one before."""
        plan = self.split_plan()
        if not plan:
            self.note = T["one_change_cannot"]
            return self.render()
        self.slow(lambda: text_sets(self.services.data("text")), lambda sets: self._split(sets, plan, list(predictions)))

    def _split(self, sets, plan, predictions):
        if not sets:
            self.note = NO_TEXT
            return self.render()
        parent = self.base.best_checkpoint if self.base else None
        mode = "from_scratch" if plan[0][0]["V"] != self.cur["V"] or not parent else "catch_up"
        saved = [self.services.save_version(spec, ", ".join(chips), predictions[i] if i < len(predictions) else None)
                 for i, (_s, spec, chips) in enumerate(plan)]
        run_dir = getattr(getattr(self.services, "workspace", None), "run_dir", None)
        jobs = split_jobs(saved[0].number, plan, mode, parent, sets, run_dir)
        for job in jobs:
            self.services.enqueue(job)
        self.chain |= {job.id for job in jobs}
        self.base, self.saved = saved[-2], saved[-1]  # the tray follows the last one: "version N, from version N - 1"
        self.cur, self.draft, self.anchor = read_spec(saved[-2].spec), read_spec(saved[-1].spec), None
        self.job_id, self.phase, self._staged = jobs[-1].id, "queued", None
        self.note = T["split_note"].format(first=saved[0].number, last=saved[-1].number)
        self.services.stage_draft(None)
        self.load_chart()
        self.render()

    def _queue(self, sets, from_scratch):
        if not sets:
            self.note = NO_TEXT
            return self.render()
        parent = self.base.best_checkpoint if self.base else None
        mode = "from_scratch" if from_scratch or self.needs_scratch() else "catch_up"
        spec = self.saved.spec if self.saved and self.saved.spec else self.spec_for(self.draft)
        self.job_id = uuid.uuid4().hex[:8]
        self.services.enqueue(catchup_job(self.job_id, self.n + 1, spec, mode, parent, sets))
        self.phase = "queued"
        self.render()

    def to_front(self):
        """Move the job to the head of the waiting list (the running job is not interrupted)."""
        for _ in range(len(self.services.queue().up_next)):
            if self.position() == 0:
                break
            self.services.move_up(self.job_id)
        self.render()

    def outlook(self):
        return queued_outlook(self.services, self.job_id)

    def position(self):
        return next((i for i, j in enumerate(self.services.queue().up_next) if j.id == self.job_id), None)

    def new_changes(self):
        self.base = self.services.current_version() or self.saved
        self.dismissed = self.base.number if self.base else None
        self.cur = read_spec(self.base.spec) if self.base and self.base.spec else dict(self.draft)
        self.draft, self.anchor, self.phase, self.job_id = dict(self.cur), None, "edit", None
        self.load_chart()
        self.render()

    def ask_both(self):
        """Same question to both versions (those with trained weights); replies stream in."""
        self.answer_stop.set()
        self.answer_stop = stop = threading.Event()
        self.answers, self.workers = {}, []
        for number in (self.n, self.n + 1):
            v = self.version(number)
            if v and v.best_checkpoint:
                self.answers[number] = ""
                self.workers.append(threads.start(self._reply, v, stop))
        self.render()
        self._pump()

    def _reply(self, v, stop):
        put = lambda text: self.answer_q.put((v.number, text))  # noqa: E731
        try:
            final = self.services.chat(v, [{"role": "user", "content": PROMPT}], put, stop)
            if final:
                put(final)
        except Exception as error:  # noqa: BLE001 - shown in place of the reply
            put(plain_error(error))

    def _pump(self):
        try:
            while True:
                number, text = self.answer_q.get_nowait()
                if number in self.answers:
                    self.answers[number] = text
                    if number in self.answer_labels:
                        self.answer_labels[number].configure(text=text)
        except queue.Empty:
            pass
        if any(t.is_alive() for t in self.workers) or not self.answer_q.empty():
            self.after(200, self._pump)

    # view -------------------------------------------------------------------
    def render(self):
        for child in self.body.winfo_children():
            child.destroy()
        self.stage()
        self.answer_labels, self.segs = {}, {}
        versions = self.services.versions()
        setting = self.advanced_set if self.advanced_set is not None else self.services.setting(ADVANCED)
        self.advanced = advanced_view(versions, setting)
        self._pill()
        top = tk.Frame(self.body, bg=C["bg"])
        top.pack(fill="x", pady=(0, GUTTER))
        left = tk.Frame(top, bg=C["bg"])
        left.pack(side="left", fill="both", expand=True, padx=(0, GUTTER))
        right = tk.Frame(top, bg=C["bg"], width=340)
        right.pack(side="left", fill="y")
        self._disclosure(left, versions)
        self._tower(left)
        self._explain(right)
        self._start_from(right)
        self._experiments(right)
        self._controls(right)
        self._costs()
        self._expect()
        self._tray()
        self._charts()
        self._notebook()
        self._look_inside()
        self._paint_design()

    def _paint_design(self):
        """Everything the draft decides, repainted in place: rebuilding the page on each setting flashed it."""
        try:
            self.draft_spec = self.spec_for(self.draft)
        except ValueError:
            self.draft_spec = None
        self._paint_tower()
        for key in self.segs:
            self._paint_seg(key)
        for key, note in self.knob_notes.items():
            note.configure(text="" if knob_applies(self.draft, key) else T["no_heads" if key == "H" else "no_room"])
        self._paint_costs()
        self._paint_expect()
        self._paint_tray()
        self._paint_explain()
        self._paint_experiments()

    def _disclosure(self, parent, versions):
        """The Advanced toggle, and until it is on how many trained versions are left before the menus show by themselves."""
        row = tk.Frame(parent, bg=C["bg"])
        row.pack(fill="x", pady=(0, 8))
        Toggle(row, T["advanced"], self.advanced, self.set_advanced, bg="bg").pack(side="left")
        info_icon(row, T["advanced_tip"].format(n=UNLOCK), bg="bg").pack(side="left", padx=(4, 0))
        left = UNLOCK - trained_count(versions)
        self.basic_note = T["basic_note"].format(left=left, s="" if left == 1 else "s") if not self.advanced and left > 0 else ""
        if self.basic_note:
            tk_label(row, self.basic_note, 12, "muted", "bg", wraplength=420, justify="left").pack(side="left", padx=(12, 0))

    def _explain(self, parent):
        """The side panel a click on a part opens: what it does, its module, its effect on the example sentence, where to see it."""
        self.explain_box = Card(parent)
        self.explain_box.pack(fill="x", pady=(0, 8))
        self.explained = None

    def _paint_explain(self):
        box = self.explain_box
        if box is None or not box.winfo_exists():
            return
        for child in box.winfo_children():
            child.destroy()
        head = tk.Frame(box, bg=C["panel"])
        head.pack(fill="x")
        Kicker(head, T["explain"], bg="panel").pack(side="left")
        if self.explaining is None:
            self.explained = None
            tk_label(box, T["explain_hint"], 12, "muted", "panel", wraplength=300, justify="left").pack(anchor="w", pady=(4, 0))
            return
        close = tk_label(head, T["explain_close"], 12, "muted", "panel", cursor="hand2", takefocus=1)
        close.pack(side="right")
        _activate(close, self.close_explain)
        part, block = self.explaining
        if block is not None and block >= self.draft["L"]:
            block = self.draft["L"] - 1
        self.explained = e = explain(part, self.draft, block, self.lists)
        tk_label(box, e["title"], 15, "text", "panel", font=font(15, bold=True)).pack(anchor="w", pady=(4, 2))
        tk_label(box, e["what"], 12, "text2", "panel", wraplength=300, justify="left").pack(anchor="w")
        tip(tk_label(box, e["module"], 13, "text", "panel", font=font(13, bold=True)), e["module_text"]).pack(anchor="w", pady=(6, 0))
        tk_label(box, e["module_text"], 12, "muted", "panel", wraplength=300, justify="left").pack(anchor="w")
        Kicker(box, T["explain_example"], bg="panel").pack(anchor="w", pady=(8, 0))
        tk_label(box, e["example"], 12, "text2", "panel", wraplength=300, justify="left").pack(anchor="w")
        row = tk.Frame(box, bg=C["panel"])
        row.pack(anchor="w", pady=(8, 0))
        tk_label(row, T["explain_see"], 12, "muted", "panel").pack(anchor="w")
        self.part_links = {}
        for route, label, at in e["links"]:
            link = tk_label(row, label + " ›", 13, "violet", "panel", cursor="hand2", takefocus=1)
            link.pack(anchor="w")
            self.part_links[route] = link
            _activate(link, lambda r=route, at=at: self.follow(r, at))

    def _clickable(self, widget, part, block=None):
        """A tower widget that opens the side panel on `part`."""
        widget.configure(cursor="hand2", takefocus=1)
        _activate(widget, lambda: self.explain_part(part, block))
        return widget

    def _experiments(self, parent):
        """Starting questions: each sets up a one-change draft from the current version, with the question saved alongside."""
        card = Card(parent)
        card.pack(fill="x", pady=(0, 8))
        head = tk.Frame(card, bg=C["panel"])
        head.pack(anchor="w")
        Kicker(head, T["experiments"], bg="panel").pack(side="left")
        info_icon(head, T["experiments_tip"]).pack(side="left", padx=(4, 0))
        self.question_links = {}
        for question, tries in experiment_rows(self.cur):
            row = tk.Frame(card, bg=C["panel"])
            row.pack(anchor="w", fill="x", pady=(4, 0))
            tk_label(row, question, 13, "text", "panel", wraplength=300, justify="left").pack(anchor="w")
            line = tk.Frame(row, bg=C["panel"])
            line.pack(anchor="w")
            for key, label, why in tries:
                link = tk_label(line, label + " ›", 12, "violet", "panel", cursor="hand2", takefocus=1)
                link.pack(side="left", padx=(0, 12))
                link.why = why
                if why:
                    tip(link, why)
                _activate(link, lambda key=key: self.try_question(key))
                self.question_links[key] = link
        self.question_label = tk_label(card, "", 12, "amber", "panel", wraplength=300, justify="left")

    def _paint_experiments(self):
        for link in self.question_links.values():
            link.configure(fg=C["violet"] if self.phase == "edit" and not link.why else C["muted"])
        question = held_question(self.experiment, self.draft) if self.phase == "edit" else None
        self.question_label.configure(text=T["exp_asking"].format(question=question) if question else "")
        if question:
            self.question_label.pack(anchor="w", pady=(8, 0))
        else:
            self.question_label.pack_forget()

    def _pill(self):
        q = self.services.queue()
        row = tk.Frame(self.body, bg=C["panel"], highlightthickness=1, highlightbackground=C["border"], padx=14, pady=8)
        row.pack(anchor="w", pady=(0, GUTTER))
        self.pill_label = tk_label(row, "", 14, "text", "panel")
        self.pill_label.pack(side="left")
        self.pill_bar = Progress(row)
        self.pill_bar.configure(width=80)
        self.pill_bar.pack(side="left", padx=(10, 0))
        self._paint_pill(q)

    def _paint_pill(self, q):
        self.pill_label.configure(text=pill_text(q.job, self.base), fg=C["violet"] if q.job else C["text2"])
        self.pill_bar.set(q.step / q.total if q.job and q.total else None)

    def _tower(self, parent):
        """The model bottom to top, drawn top-down: next word, word scores, even out, the blocks, position, word table, your text."""
        box = tk.Frame(parent, bg=C["panel2"], highlightthickness=1, highlightbackground=C["border"])
        box.pack(fill="x")
        tip(tk_label(box, T["next_word"], 15, "text", "panel2", font=font(15, bold=True)), T["next_word_desc"]).pack(anchor="w", padx=12, pady=8)
        tk_label(parent, "↑", 14, "violet").pack(anchor="w", padx=40)
        self.pickers = []  # (menu button, was label, slot, block index or None) of the model-wide parts
        self._part_row(parent, "head")
        row = self._row(parent)
        self._clickable(tip(tk_label(row, T["final_norm"], 14, "text", "panel2", font=font(14, bold=True), width=14),
                            f"{T['final_norm_tip']}\n\n{part_tip('norm')}"), "final_norm").pack(side="left", padx=12, pady=6)
        self.final_norm = tk_label(row, "", 13, "text2", "panel2")
        self.final_norm.pack(side="left")
        tip(self.final_norm, lambda: module_tip("norm", self.draft["M"]["norm"]))
        tk_label(row, f"  {T['same_as_blocks']}", 12, "muted", "panel2").pack(side="left")
        tk_label(parent, "↑", 14, "violet").pack(anchor="w", padx=40)
        self.block_box = tk.Frame(parent, bg=C["bg"])
        self.block_box.pack(fill="x")
        self.block_rows = []  # (row, name, heads, width, tag, header, toggle, parts, detail) per block, block 1 first; drawn top-down
        step = tk.Frame(parent, bg=C["bg"])
        step.pack(anchor="w", pady=8)
        tip(tk_label(step, f"{T['blocks']}  ", 14, "text", "bg"), f"{T['blocks_desc']}\n\n{glossary_text('block')}").pack(side="left")
        self.minus = button(step, "−", lambda: self.edit(**resize(self.draft, max(1, self.draft["L"] - 1))))
        self.plus = button(step, "+", lambda: self.edit(**resize(self.draft, min(12, self.draft["L"] + 1))))
        self.minus.pack(side="left")
        self.blocks_label = tk_label(step, "", 18, "text", "bg", width=3, anchor="center")
        self.blocks_label.pack(side="left")
        self.plus.pack(side="left")
        self.blocks_was = tk_label(step, "", 13, "muted", "bg")
        self.blocks_was.pack(side="left")
        for b in (self.minus, self.plus):
            b.state(["!disabled"] if self.phase == "edit" else ["disabled"])
        tk_label(parent, "↑", 14, "violet").pack(anchor="w", padx=40)
        self._part_row(parent, "position")
        self._part_row(parent, "embedding")
        tk_label(parent, "↑", 14, "violet").pack(anchor="w", padx=40)
        text = tk.Frame(parent, bg=C["panel2"], highlightthickness=1, highlightbackground=C["border"])
        text.pack(fill="x")
        head = tk.Frame(text, bg=C["panel2"])
        head.pack(fill="x", padx=12, pady=(8, 0))
        tip(tk_label(head, T["your_text_title"], 14, "text", "panel2", font=font(14, bold=True)), glossary_text("tokenizer")).pack(side="left")
        self.pieces_count = tk_label(head, "", 12, "muted", "panel2")
        self.pieces_count.pack(side="left")
        self._seg(text, list(WORDS), "V", (0, 1, 2, 3), "panel2")
        for option, term in zip(WORDS, WORD_TERMS):
            tip(self.segs["V"][0].buttons[option], glossary_text(term))
        self.pieces_box = tk.Frame(text, bg=C["panel2"])
        self.pieces_box.pack(anchor="w", fill="x", padx=12, pady=(0, 8))
        self.pieces_shown = None

    def _row(self, parent):
        row = tk.Frame(parent, bg=C["panel2"], highlightthickness=1, highlightbackground=C["border"])
        row.pack(fill="x", pady=2)
        return row

    def _part_row(self, parent, slot):
        """One model-wide part: its name and its module menu."""
        row = self._row(parent)
        self._clickable(tip(tk_label(row, T["part_" + slot], 14, "text", "panel2", font=font(14, bold=True), width=14), part_tip(slot)),
                        slot).pack(side="left", padx=12, pady=6)
        self.pickers.append((*self._picker(row, slot, lambda: None, "panel2"), slot, None))

    def _picker(self, parent, slot, block, bg):
        """(menu button, was label) choosing `slot`'s module; block() says where a pick goes: None = every block, or a block index.
        Modules that cannot work in this design are listed greyed out with the reason. Until Advanced is on it is a plain label."""
        if not self.advanced:
            mb = tk.Label(parent, font=font(13), bg=C[bg], fg=C["text"], disabledforeground=C["muted"], padx=8, pady=3)
            tip(mb, lambda: module_tip(slot, mb.module))
            mb.pack(side="left")
            was = tk_label(parent, "", 12, "muted", bg)
            was.pack(side="left", padx=(6, 0))
            return mb, was
        mb = tk.Menubutton(parent, font=font(13), relief="flat", bg=C[bg], fg=C["text"], activebackground=C["panel"], activeforeground=C["text"],
                           disabledforeground=C["muted"], takefocus=1, cursor="hand2", padx=8, pady=3, indicatoron=True,
                           highlightthickness=1, highlightbackground=C["border"], highlightcolor=C["violet"])
        menu = tk.Menu(mb, tearoff=0, font=font(13), bg=C["panel"], fg=C["text"], activebackground=C["violet"], activeforeground=C["bg"],
                       disabledforeground=C["dim"])

        def fill():
            menu.delete(0, "end")
            base, was = self.spec_base()
            for short, name, _blurb, reason in part_options(base, self.draft, was, slot, block()):
                menu.add_command(label=f"{name} · {T['wont_fit']}: {reason}" if reason else name, state="disabled" if reason else "normal",
                                 command=lambda short=short: self.pick(slot, short, block()))
        menu.configure(postcommand=fill)
        mb.configure(menu=menu)
        for seq in ("<space>", "<Return>"):
            mb.bind(seq, lambda _e: menu.tk_popup(mb.winfo_rootx(), mb.winfo_rooty() + mb.winfo_height()) if self.phase == "edit" else None)
        tip(mb, lambda: module_tip(slot, mb.module))
        mb.pack(side="left")
        was = tk_label(parent, "", 12, "muted", bg)
        was.pack(side="left", padx=(6, 0))
        return mb, was

    def _paint_picker(self, mb, was, slot, now, before, bg):
        mb.module = now
        changed = before is not None and now != before
        mb.configure(text=module_name(slot, now), fg=C["amber"] if changed else C["text"], bg=C[bg],
                     state="normal" if self.phase == "edit" else "disabled")
        was.configure(text=f"{T['was'].format(name=module_name(slot, before))}  ·  changed" if changed else "", bg=C[bg])

    def _paint_tower(self):
        cur, dr, rows = self.cur, self.draft, self.block_rows
        for mb, was, slot, _ in self.pickers:
            self._paint_picker(mb, was, slot, dr["M"][slot], cur["M"][slot], "panel2")
        self.final_norm.configure(text=module_name("norm", dr["M"]["norm"]), fg=C["amber"] if dr["M"]["norm"] != cur["M"]["norm"] else C["text2"])
        while len(rows) < max(cur["L"], dr["L"]):  # a new block goes on top
            i = len(rows)
            row = tk.Frame(self.block_box, highlightthickness=1)
            row.pack(fill="x", pady=2, **({"before": rows[-1][0]} if rows else {}))
            header = tk.Frame(row)
            header.pack(fill="x")
            toggle = tk_label(header, "▸", 13, "muted", cursor="hand2", takefocus=1)
            toggle.pack(side="left", padx=(8, 0))
            _activate(toggle, lambda i=i: self.toggle(i))
            tip(toggle, T["block_parts_tip"])
            name = tk_label(header, f"Block {i + 1}", 13, width=8)
            name.pack(side="left", padx=4, pady=8)
            heads = self._clickable(tip(tk_label(header, "", 13, "pink"), glossary_text("heads")), "mixer", i)
            heads.pack(side="left")
            width = self._clickable(tip(tk.Frame(header, bg=C["blue"], height=8), glossary_text("expansion")), "channel", i)
            width.pack(side="left", padx=10)
            parts = tk_label(header, "", 12, "muted")
            parts.pack(side="left")
            tip(parts, lambda i=i: "\n\n".join(module_tip(slot, block_parts(self.draft, i)[slot]) for slot in ("mixer", "channel")))
            tag = tk_label(header, "", 12, "amber")
            tag.pack(side="right", padx=8)
            detail = tk.Frame(row)
            rows.append((row, name, heads, width, tag, header, toggle, parts, detail))
        while len(rows) > max(cur["L"], dr["L"]):
            rows.pop()[0].destroy()
        knobs = dr["H"] != cur["H"] or dr["W"] != cur["W"]
        for i, (row, name, heads, width, tag, header, toggle, parts, detail) in enumerate(rows):
            added, removed = i >= cur["L"], i >= dr["L"]
            shown = cur if removed else dr
            mine, own = block_parts(shown, i), own_slots(shown, i)
            changed = not added and not removed and (knobs or block_parts(cur, i) != mine)
            bg = "amber_bg" if added else "violet_bg" if own else "panel2"
            row.configure(bg=C[bg], highlightbackground=C["amber"] if added else C["violet"] if own else C["border"])
            for w in (header, toggle, name, heads, parts, tag, detail):
                w.configure(bg=C[bg])
            toggle.configure(text="▾" if i in self.open and not removed else "▸", highlightbackground=C[bg])
            name.configure(fg=C["dim"] if removed else C["text"])
            heads.configure(text="●" * shown["H"] if "heads" in fields_of("mixer", mine["mixer"]) else "")
            if "d_ff" in fields_of("channel", mine["channel"]):
                width.configure(width=[36, 72, 144][shown["W"]])
                width.pack(side="left", padx=10, after=heads)
            else:
                width.pack_forget()
            parts.configure(text=f"{module_name('mixer', mine['mixer'])} · {module_name('channel', mine['channel'])}")
            text = "new" if added else "removed" if removed else "changed" if changed else T["own_parts"] if own else ""
            tag.configure(text=text, fg=C["violet"] if text == T["own_parts"] else C["amber"])
            self._paint_detail(i, detail, bg, removed)
        self.blocks_label.configure(text=str(dr["L"]), fg=C["amber"] if dr["L"] != cur["L"] else C["text"])
        self.blocks_was.configure(text=f"  was {cur['L']}" if dr["L"] != cur["L"] else "")
        self._paint_pieces()

    def _paint_pieces(self):
        v = self.draft["V"]
        chips, count = pieces_for(v, self.lists)
        self.pieces_count.configure(text=f"  {count}")
        if (v, id(self.lists)) != self.pieces_shown:
            self.pieces_shown = v, id(self.lists)
            for child in self.pieces_box.winfo_children():
                child.destroy()
            for k in range(0, len(chips), 14):  # wrap long piece lists
                line = tk.Frame(self.pieces_box, bg=C["panel2"])
                line.pack(anchor="w")
                for label, state in chips[k:k + 14]:
                    Chip(line, label, state).pack(side="left", padx=1, pady=1)

    def toggle(self, i):
        self.open ^= {i}
        self._paint_tower()

    def _paint_detail(self, i, detail, bg, removed):
        """Block i's sub-layers top-down (what a word passes through bottom-up), each with its module menu."""
        for child in detail.winfo_children():
            child.destroy()
        if i not in self.open or removed:
            detail.pack_forget()
            return
        detail.pack(fill="x", padx=(32, 12), pady=(0, 8))
        if self.advanced:  # per-block choices
            one = self.scope.get(i, False)
            options = [T["scope_all"], T["scope_one"].format(n=i + 1)]
            scope = Segmented(detail, options, options[one], lambda opt, i=i: self.set_scope(i, opt != options[0]), bg=bg)
            scope.pack(anchor="w", pady=(0, 4))
        cur, dr = self.cur, self.draft
        mine, before = block_parts(dr, i), block_parts(cur, i) if i < cur["L"] else None
        for slot in (None, "channel", "norm", None, "mixer", "norm"):  # None: the add-back
            line = tk.Frame(detail, bg=C[bg])
            line.pack(fill="x", pady=1)
            if slot is None:
                tip(tk_label(line, f"⊕ {T['add_back']}", 12, "muted", bg), glossary_text("residual")).pack(side="left")
                continue
            skipped = registry.get(slot, mine[slot]).skip
            self._clickable(tip(tk_label(line, T["part_" + slot], 13, "dim" if skipped else "text", bg, width=14), part_tip(slot)),
                            slot, i).pack(side="left")
            mb, was = self._picker(line, slot, lambda i=i: i if self.advanced and self.scope.get(i) else None, bg)
            self._paint_picker(mb, was, slot, mine[slot], before and before[slot], bg)
            if (i, slot) in dr["O"]:
                tk_label(line, T["only_here"], 12, "violet", bg).pack(side="left", padx=(8, 0))

    def set_scope(self, i, one):
        self.scope[i] = one
        self._paint_tower()

    def _seg(self, parent, options, key, values, bg="panel"):
        label, value = dict(zip(values, options)), dict(zip(options, values))
        seg = Segmented(parent, options, None, lambda opt: self.knob(key, value[opt]), bg=bg)
        seg.pack(anchor="w", padx=12 if bg == "panel2" else 0, pady=4)
        self.segs[key] = seg, label

    def _paint_seg(self, key):
        seg, label = self.segs[key]
        seg.set(label.get(self.draft[key]))
        # ponytail: Tk borders cannot be dashed, so the saved choice gets a violet outline and a "saved" in its label.
        was = label.get(self.cur[key]) if self.cur[key] != self.draft[key] else None
        for opt, b in seg.buttons.items():
            b.configure(highlightbackground=C["violet"] if opt == was else C["border"], text=f"{opt} · saved" if opt == was else opt)

    def _start_from(self, parent):
        """Pick a ready-made design, read what it costs on this computer and its recipe, then load it into the draft."""
        card = Card(parent)
        card.pack(fill="x", pady=(0, 8))
        head = tk.Frame(card, bg=C["panel"])
        head.pack(anchor="w")
        Kicker(head, T["start_from"], bg="panel").pack(side="left")
        info_icon(head, T["start_from_tip"]).pack(side="left", padx=(4, 0))
        mb = tk.Menubutton(card, font=font(13), relief="flat", bg=C["panel"], fg=C["text"], activebackground=C["panel2"], activeforeground=C["text"],
                           disabledforeground=C["muted"], takefocus=1, cursor="hand2", padx=8, pady=3, indicatoron=True,
                           highlightthickness=1, highlightbackground=C["border"], highlightcolor=C["violet"])
        menu = tk.Menu(mb, tearoff=0, font=font(13), bg=C["panel"], fg=C["text"], activebackground=C["violet"], activeforeground=C["bg"])
        for key, spec in self.presets:
            menu.add_command(label=spec["name"], command=lambda key=key: self.preview_preset(key))
        mb.configure(menu=menu)
        for seq in ("<space>", "<Return>"):
            mb.bind(seq, lambda _e: menu.tk_popup(mb.winfo_rootx(), mb.winfo_rooty() + mb.winfo_height()) if self.phase == "edit" else None)
        tip(mb, lambda: glossary_text(preset_term(self.preset_key)) if self.preset_key else T["start_from_tip"])
        mb.pack(anchor="w", pady=(4, 0))
        self.preset_mb = mb
        self.preset_about = tk_label(card, "", 12, "text2", "panel", wraplength=300)
        tip(self.preset_about, lambda: glossary_text(preset_term(self.preset_key)) if self.preset_key else "")
        self.preset_lines = [tk_label(card, "", 12, "muted", "panel", wraplength=300) for _ in range(4)]  # size, fit, speed, recipe
        self.preset_btn = button(card, T["preset_load"], self.use_preset)
        pretrained = Card(parent)
        pretrained.pack(fill="x", pady=(0, 8))
        self.import_btn = button(pretrained, T["import_gpt2"], self.import_gpt2)
        self.import_btn.pack(anchor="w")
        tip(self.import_btn, lambda: T["import_gpt2_tip"])
        self._paint_preset()

    def _paint_preset(self):
        spec = dict(self.presets).get(self.preset_key)
        self.preset_mb.configure(text=spec["name"] if spec else T["start_from_pick"], state="normal" if self.phase == "edit" else "disabled")
        widgets = (self.preset_about, *self.preset_lines, self.preset_btn)
        for w in widgets:
            w.pack_forget()
        if not spec:
            return
        cost = self.preset_cost = preset_cost(spec, gpu=gpu_kind())
        self.preset_about.configure(text=T["preset_" + self.preset_key])
        texts = [*cost["lines"], T["preset_recipe"].format(recipe=recipe_text(spec))]
        for label, text, bad in zip(self.preset_lines, texts, (False, cost["fits"] is False, cost["slow"], False)):
            label.configure(text=text, fg=C["amber"] if bad else C["muted"])
        self.preset_btn.state(["!disabled"] if self.phase == "edit" else ["disabled"])
        self.preset_about.pack(anchor="w", pady=(6, 2))
        for label in self.preset_lines:
            label.pack(anchor="w", pady=1)
        self.preset_btn.pack(anchor="w", pady=(6, 0))

    def _controls(self, parent):
        self.knob_notes = {}
        for name, hint, color, opts, key, values in (
                ("Look-back heads", f"{T['look_back_heads_tip']}\n\n{glossary_text('heads')}", "pink", [str(h) for h in HEADS], "H", HEADS),
                ("Thinking room", f"{T['thinking_room_tip']}\n\n{glossary_text('expansion')}", "blue", list(WIDTHS), "W", (0, 1, 2))):
            card = Card(parent)
            card.pack(fill="x", pady=(0, 8))
            head = tk.Frame(card, bg=C["panel"])
            head.pack(anchor="w")
            tk_label(head, name, 15, color, "panel", font=font(15, bold=True)).pack(side="left")
            info_icon(head, hint).pack(side="left", padx=(4, 0))
            self._seg(card, opts, key, values)
            self.knob_notes[key] = tk_label(card, "", 12, "muted", "panel", wraplength=300)
            self.knob_notes[key].pack(anchor="w")

    def _costs(self):
        card = Card(self.body)
        card.pack(fill="x", pady=(0, GUTTER))
        Kicker(card, T["what_costs"], bg="panel").pack(anchor="w")
        self.cost_rows = []
        for name, f, more, less in ((T["size_label"], "params", T["cost_bigger"], T["cost_smaller"]),
                                    (T["time_per_step"], "time", T["cost_slower"], T["cost_faster"]),
                                    (T["memory_while_training"], "memory", T["cost_more"], T["cost_less"])):
            row = tk.Frame(card, bg=C["panel"])
            row.pack(fill="x", pady=4)
            label = tk_label(row, name, 14, "text", "panel", width=24)
            label.pack(side="left")
            if term := {"params": "parameters", "time": "step"}.get(f):
                tip(label, glossary_text(term))
            bars = tk.Frame(row, bg=C["panel"])
            bars.pack(side="left", fill="x", expand=True)
            drawn = []
            for label in ("now", "new"):
                line = tk.Frame(bars, bg=C["panel"])
                line.pack(fill="x")
                tk_label(line, label, 11, "dim", "panel", width=4).pack(side="left")
                drawn.append(tk.Frame(line, bg=C["dim"], height=7))
                drawn[-1].pack(side="left", pady=1)
            said = tk_label(row, "", 13, "muted", "panel", width=22)
            said.pack(side="left")
            self.cost_rows.append((name, f, more, less, *drawn, said))
        self._param_bar(card)
        row = tk.Frame(card, bg=C["panel"])
        row.pack(fill="x", pady=(8, 2))
        tk_label(row, "Can the old weights carry over?", 14, "text", "panel", width=24).pack(side="left")
        self.warm_chip = Chip(row)
        self.warm_none = tk_label(row, "nothing to carry over: no change yet", 13, "muted", "panel")
        self.warm_text = tk_label(card, "", 12, "muted", "panel", wraplength=780)
        self.fits_row = tk.Frame(card, bg=C["panel"])
        self.fits_row.pack(anchor="w", pady=(6, 0))
        self.fits_label = tk_label(self.fits_row, "", 12, "muted", "panel")
        self.fits_label.pack(side="left")
        info_icon(self.fits_row, lambda: self.fits_text).pack(side="left", padx=(4, 0))
        self.compute_labels = [tk_label(card, "", 12, "muted", "panel", wraplength=780) for _ in range(3)]
        for label in self.compute_labels:
            label.pack(anchor="w", pady=(6, 0))
        self.compute_tip = T["compute_tip"].format(rule=TOKENS_PER_PARAM, most=MAX_EPOCHS)
        tip(self.compute_labels[0], lambda: self.compute_tip)
        row = tk.Frame(card, bg=C["panel"])
        row.pack(anchor="w", pady=(8, 0))
        self.lr_button = button(row, T["lr_find"], self.find_lr)
        self.lr_button.pack(side="left")
        info_icon(row, T["lr_tip"]).pack(side="left", padx=(4, 0))
        self.lr_label = tk_label(card, "", 12, "muted", "panel", wraplength=780)
        self.lr_label.pack(anchor="w", pady=(4, 0))
        self._paint_lr()
        self._ask_compute()

    def _param_bar(self, card):
        """Where the size goes: the parameters stacked by part, the current version above the draft, to one scale."""
        row = tk.Frame(card, bg=C["panel"])
        row.pack(fill="x", pady=4)
        label = tk_label(row, T["params_where"], 14, "text", "panel", width=24)
        label.pack(side="left")
        tip(label, T["params_where_tip"])
        bars = tk.Frame(row, bg=C["panel"])
        bars.pack(side="left")
        self.param_segs = []
        for name in ("now", "new"):
            line = tk.Frame(bars, bg=C["panel"])
            line.pack(fill="x")
            tk_label(line, name, 11, "dim", "panel", width=4).pack(side="left")
            segs = {key: tk.Frame(line, bg=C[colour], height=7) for key, colour in PARAM_PARTS}
            for seg in segs.values():
                seg.pack(side="left", pady=1)
            self.param_segs.append(segs)
        legend = tk.Frame(card, bg=C["panel"])
        legend.pack(anchor="w", padx=(0, 0))
        self.param_legend = {key: tk_label(legend, "", 12, colour, "panel") for key, colour in PARAM_PARTS}
        for key, colour in PARAM_PARTS:
            tk_label(legend, "■", 12, colour, "panel").pack(side="left")
            self.param_legend[key].pack(side="left", padx=(2, 12))
        self.param_label = tk_label(card, "", 12, "amber", "panel", wraplength=780, justify="left")
        self.param_label.pack(anchor="w")

    def _paint_params(self, a, b):
        for segs, widths in zip(self.param_segs, param_bars(a["parts"], b["parts"])):
            for key, px in widths:
                segs[key].configure(width=max(1, px))
        self.param_texts = param_shares(b["parts"])
        for key, text in self.param_texts:
            self.param_legend[key].configure(text=text)
        self.param_label.configure(text=param_note(b["parts"]))

    def _ask_compute(self):
        """The size of your text, counted once per page: known at once when it was counted before, else in the background."""
        known = self.services.cached_preview([])
        if known is not None or self.compute_preview is not None:
            self.compute_preview = known if known is not None else self.compute_preview
        elif not self._asking:
            self._asking = True
            self.quietly(lambda: self.services.pretrain_preview([]), self._got_compute, lane="slow")

    def _got_compute(self, preview):
        self._asking, self.compute_preview = False, preview or {}
        if self.winfo_exists() and self.compute_labels[0].winfo_exists():
            self._paint_compute()

    def _paint_compute(self):
        spec = self.draft_spec or self.base_spec()
        if self.word_changed():
            lines = [(T["compute_recount"], False)]
        else:
            lines = compute_lines(sizes(spec, self.draft["V"])["params"], spec["dims"]["n_ctx"], spec["dims"]["n_layer"],
                                  self.compute_preview, gpu_kind())
        for label, (text, bad) in zip(self.compute_labels, lines + [("", False)] * len(self.compute_labels)):
            label.configure(text=text, fg=C["amber"] if bad else C["muted"])

    def find_lr(self):
        """Run the learning-rate range test on the draft in the background; pressed again while it runs, it stops."""
        if self.lr_stop is not None:
            self.lr_stop.set()
            return
        spec, stop = self.draft_spec or self.base_spec(), threading.Event()
        self.lr_stop, self.lr_result, self.lr_for = stop, None, spec

        def work():
            try:
                return self.services.lr_range_test(spec, stop)
            except Exception as error:  # say it under the button; the button comes back
                return {"error": plain_error(error)}

        def done(result):
            self.lr_stop, self.lr_result = None, result
            if self.winfo_exists() and self.lr_label.winfo_exists():
                self._paint_lr()
        self._paint_lr()
        self.quietly(work, done, lane="slow")

    def _paint_lr(self):
        running = self.lr_stop is not None
        current = ((self.lr_for or {}).get("train", {}).get("optimizer", {}).get("config", {}) or {}).get("lr")
        self.lr_button.configure(text=T["lr_stop"] if running else T["lr_find"])
        stale = self.lr_for != (self.draft_spec or self.base_spec())  # the draft changed since the test ran
        self.lr_label.configure(text=T["lr_running"] if running else "" if stale else lr_text(self.lr_result, current))

    def _paint_costs(self):
        a = sizes(self.base_spec(), self.cur["V"])
        b = sizes(self.draft_spec, self.draft["V"]) if self.draft_spec else a
        self.cost_say = {}
        for name, f, more, less, now, new, said in self.cost_rows:
            x, y = a[f], b[f]
            s = self.cost_say[name] = say(x, y, more, less)
            same = s == T["about_same"]
            old_pct, new_pct = bar(x, y)
            now.configure(width=max(2, old_pct * 3))
            new.configure(width=max(2, new_pct * 3), bg=C["violet"] if same else C["amber"])
            said.configure(text=s, fg=C["muted"] if same else C["amber"])
        self._paint_params(a, b)
        self.warm = warm = warm_outlook(self.cur, self.draft)
        if warm:
            self.warm_none.pack_forget()
            self.warm_chip.set(warm["chip"], warm["state"])
            self.warm_chip.pack(side="left")
            self.warm_text.configure(text=warm["text"])
            self.warm_text.pack(anchor="w", before=self.fits_row)
        else:
            self.warm_chip.pack_forget()
            self.warm_text.pack_forget()
            self.warm_text.configure(text="")
            self.warm_none.pack(side="left")
        ok = fits(b)
        verdict = T["training_memory_fits"] if ok is None else ("fits" if ok else "doesn't fit")
        self.fits_text = f"{T['training_memory']} {verdict} {T['training_memory_desc']}"
        self.fits_label.configure(text=f"{T['training_memory']} {verdict}")
        self._paint_compute()

    def _expect(self):
        card = Card(self.body)
        card.pack(fill="x", pady=(0, GUTTER))
        Kicker(card, T["what_expect"], bg="panel").pack(anchor="w")
        row = tk.Frame(card, bg=C["panel"])
        row.pack(anchor="w", pady=(2, 0))
        self.guide_label = tk_label(row, "", 14, "text", "panel")
        self.guide_label.pack(side="left")
        info_icon(row, lambda: self.hover_text).pack(side="left", padx=(6, 0))

    def _paint_expect(self):
        self.guide_text, self.guide_details = design_expectation(self.draft, self.draft_spec or self.base_spec())
        self.painting_expect = True
        try:
            self.expect_texts = self.expectations()
        finally:
            self.painting_expect = False
        self.hover_text = "\n\n".join([self.guide_details, *self.expect_texts])  # the card itself is one line
        self.guide_label.configure(text=self.guide_text)

    def _tray(self):
        self.tray_box = tk.Frame(self.body, bg=C["bg"])
        self.tray_box.pack(fill="x", pady=(0, GUTTER))
        self.tray = StateTray(self.tray_box)
        self.tray.pack(fill="x", pady=(0, 8))
        self.chips = tk.Frame(self.tray_box, bg=C["bg"])
        self.predict_slot = tk.Frame(self.tray_box, bg=C["bg"])  # the save-time guesses
        self.predict_slot.pack(fill="x")
        self.button_slot = tk.Frame(self.tray_box, bg=C["bg"])
        self.button_slot.pack(fill="x")
        self.check_tray = StateTray(self.tray_box)  # a saved guess against what training did
        self.buttons, self.actions, self._predict_for = None, {}, None

    def _paint_tray(self):
        ph, n, nv = self.phase, self.n, self.nv
        Nv = nv[0].upper() + nv[1:]
        scratch = self.needs_scratch()
        self.chip_texts = []
        actions = []
        if self.asking and (ph != "edit" or self.asking[1] != self.draft_spec):  # the draft moved on: ask again on the next save
            self.asking = None
        if ph == "edit":
            self.chip_texts = self.changes()
            if self.chip_texts:
                state, kicker, text = "unsaved", T["not_saved"], T["not_saved_text"].format(version=nv)
                if question := held_question(self.experiment, self.draft):
                    text = T["exp_asking"].format(question=question) + ". " + text
                plan = self.split_plan() if len(self.chip_texts) > 1 else []
                if len(self.chip_texts) > 1 and self.base is not None:
                    text += " " + T["one_change"].format(count=len(self.chip_texts)) + " " + \
                        (T["one_change_split"].format(count=len(plan)) if plan else T["one_change_cannot"])
                if self.asking:
                    save = T["predict_split_save"].format(count=len(plan)) if self.asking[0] == "split" else T["predict_save"]
                    actions = [(save, self.save_guessed, True), (T["predict_skip"], lambda: self.save_guessed(True), False),
                               (T["predict_back"], self.cancel_guesses, False)]
                else:
                    actions = [(T["quick_test_first"], lambda: self.navigate("quicktest"), False),
                               (f"Save as {nv}", self.ask_save, True), (T["discard"], self.discard, False)]
                    if plan:
                        actions.append((T["split"].format(count=len(plan)), self.ask_split, False))
            elif self.base is None:  # nothing saved yet: the starting design itself can become version 1
                state, kicker, text = "unsaved", T["first_save"], T["first_save_text"].format(version=nv)
                actions = [(f"Save as {nv}", self.save, True), (T["quick_test_first"], lambda: self.navigate("quicktest"), False)]
            else:
                state, kicker, text = "neutral", T["no_changes"], T["no_changes_text"].format(version=n)
        elif ph == "saved":
            state = "needs_training"
            kicker = T["saved_text_needs_training" if scratch else "saved_text_needs_catchup"].format(new_version=nv)
            text = T["saved_text"].format(startNote=self.start_note(), current_version=f"version {n}", new_version=nv) if n else self.start_note()
            text += f" {if_added(self.services.queue_outlook())}."
            actions = [(T["add_training" if scratch else "add_catchup"], lambda: self.queue_it(False), True)]
            if not scratch:
                actions.append((T["start_from_scratch"], lambda: self.queue_it(True), False))
        elif ph == "queued":
            state, kicker, text = "live", T["queued"], T["queued_text"].format(new_version=Nv)
            when = self.outlook()
            text += f" {when}." if when else ""
            if self.position() and self.job_id not in self.chain:  # a split step must not run before the one it starts from
                actions = [(T["move_to_front"], self.to_front, True)]
        else:
            state = "training"
            kicker = T["training"].format(new_version=nv, status="IS TRAINING" if scratch else "IS CATCHING UP")
            text = TRAINING_TEXT.format(new=nv, n=n)
            actions = [(T["start_new_changes"], self.new_changes, True)]
        if self.note:
            text += " " + self.note
            self.note = None
        self.tray.set(state, kicker, text)
        self.tray_kicker = kicker
        self.tray_text = text
        chips = self.chips.winfo_children()
        for chip in chips[len(self.chip_texts):]:
            chip.destroy()
        for i, c in enumerate(self.chip_texts):
            if i < len(chips):
                chips[i].set(c, "unsaved")
            else:
                Chip(self.chips, c, "unsaved").pack(side="left", padx=(0, 6))
        if self.chip_texts:
            self.chips.pack(anchor="w", pady=(0, 6), after=self.tray)
        else:
            self.chips.pack_forget()  # an emptied frame keeps its last size
        if [label for label, _, _ in actions] != list(self.actions):
            if self.buttons:
                self.buttons.destroy()
            self.buttons, self.actions = tk.Frame(self.button_slot, bg=C["bg"]), {}
            self.buttons.pack(anchor="w")
            for label, cmd, primary in actions:
                self.actions[label] = btn = button(self.buttons, label, cmd, primary)
                btn.pack(side="left", padx=(0, 8))
        if self.asking != self._predict_for:
            self._predict_for = self.asking
            self._paint_predict()
        self._paint_check()

    def _paint_predict(self):
        """The save-time guesses: a row of chips per dimension worth a guess, for each version being saved."""
        for child in self.predict_slot.winfo_children():
            child.destroy()
        self.guesses = []
        steps = self.guess_steps()
        if not steps:
            return
        card = Card(self.predict_slot)
        card.pack(fill="x", pady=(0, 8))
        Kicker(card, T["predict"], bg="panel").pack(anchor="w")
        if question := held_question(self.experiment, self.draft):
            tk_label(card, T["predict_question"].format(question=question), 14, "text", "panel", wraplength=780).pack(anchor="w", pady=(2, 0))
        tk_label(card, T["predict_text"], 13, "muted", "panel", wraplength=780).pack(anchor="w", pady=(2, 4))
        for i, (title, old, new) in enumerate(steps):
            if title:
                tk_label(card, title, 13, "text", "panel", wraplength=780).pack(anchor="w", pady=(6, 0))
            chips = {}
            for key in prediction_questions(old, new):
                row = tk.Frame(card, bg=C["panel"])
                row.pack(anchor="w", pady=2)
                tk_label(row, T["predict_" + key], 13, "text", "panel", width=22, anchor="w").pack(side="left")
                chips[key] = GuessChips(row, [T["predict_" + d] for d in GUESSES[key]],
                                        lambda label, at=(i, key): self.picks.__setitem__(at, label))
                chips[key].set(self.picks.get((i, key)))
                chips[key].pack(side="left")
            self.guesses.append(chips)

    def _paint_check(self):
        """The guess made when the version was saved, against what training did: the current version while editing, else
        the one being trained. Step times come from the notebook's read (self.speeds); nothing here reads a file."""
        target = self.base if self.phase == "edit" else self.version(self.n + 1)
        shown = None
        if target is not None and getattr(target, "prediction", None):
            old = self.version(target.parent or target.number - 1)
            shown = check_text(old, target, self.speeds)
        self.check_shown = shown
        if shown:
            self.check_tray.set(*shown)
            self.check_tray.pack(fill="x", pady=(8, 0))
        else:
            self.check_tray.pack_forget()

    def _charts(self):
        card = Card(self.body)
        card.pack(fill="x", pady=(0, GUTTER))
        Kicker(card, T["while_training"], bg="panel").pack(anchor="w")
        old, new = bool(self.plot_args.get("smooth")), bool(self.plot_args.get("steps"))
        self.plot = self.ask_btn = None
        if old or new:
            self.plot = BpcPlot(card, 640, 200)
            self.plot.pack(anchor="w", pady=6)
            self.plot.set(**self.plot_args)
            legend = tk.Frame(card, bg=C["panel"])
            legend.pack(anchor="w")
            if old:
                tk_label(legend, version_name(self.n).capitalize(), 12, "violet", "panel").pack(side="left", padx=(0, 16))
            if new:
                faint = f"faint line: {self.nv}" + (f", continuing from version {self.n}'s end" if old else "")
                tk_label(legend, faint, 12, "amber", "panel").pack(side="left")
        else:
            self.no_curve = curve_note(self.n, self.phase)
            tk_label(card, self.no_curve, 13, "muted", "panel").pack(anchor="w", pady=6)
        head = tk.Frame(card, bg=C["panel"])
        head.pack(anchor="w", pady=(8, 0))
        Kicker(head, T["when_answering"], bg="panel").pack(side="left")
        info_icon(head, T["answering_note"]).pack(side="left", padx=(4, 0))
        trained = {k for k in (self.n, self.n + 1) if self.version(k) and self.version(k).best_checkpoint}
        rows = answer_rows(self.n, trained, self.phase)
        if not rows:  # nothing can answer, so no prompt, rows or button
            self.nothing_to_ask = T["nothing_to_ask"].format(n=self.n) if self.n else T["nothing_to_ask_first"]
            tk_label(card, self.nothing_to_ask, 13, "muted", "panel", wraplength=780).pack(anchor="w", pady=6)
            return
        tk_label(card, T["example_prompt"], 14, "text", "panel", font=font(14, serif=True)).pack(anchor="w")
        for number, why_not in rows:
            if number in self.answers:
                text = self.answers[number] or "…"
            else:
                text = why_not or T["its_answer_appears"]
            row = tk.Frame(card, bg=C["panel"])
            row.pack(anchor="w", fill="x")
            tk_label(row, version_name(number), 13, "violet" if number == self.n else "amber", "panel", width=14).pack(side="left")
            label = tk_label(row, text, 13, "text" if number in self.answers else "muted", "panel", font=font(13, serif=number in self.answers), wraplength=640)  # serif is for model text only
            label.pack(side="left")
            self.answer_labels[number] = label
        both = len(trained) == 2
        self.ask_btn = button(card, "Ask both versions" if both else f"Ask {version_name(next(iter(trained)))}", self.ask_both)
        self.ask_btn.pack(anchor="w", pady=(6, 0))

    def _notebook(self):
        card = Card(self.body)
        card.pack(fill="x", pady=(0, GUTTER))
        head = tk.Frame(card, bg=C["panel"])
        head.pack(anchor="w")
        Kicker(head, T["notebook"], bg="panel").pack(side="left")
        info_icon(head, T["notebook_tip"]).pack(side="left", padx=(4, 0))
        self.notebook_box = tk.Frame(card, bg=C["panel"])
        self.notebook_box.pack(anchor="w", fill="x", pady=(4, 0))
        self._paint_notebook()

    def _paint_notebook(self):
        box = getattr(self, "notebook_box", None)
        if box is None or not box.winfo_exists():
            return
        for child in box.winfo_children():
            child.destroy()
        self.notebook_texts = notebook_lines(self.notebook_rows)
        quiet = not self.notebook_rows
        for line in self.notebook_texts:
            tk_label(box, line, 13, "muted" if quiet else "text", "panel", wraplength=780, justify="left").pack(anchor="w", pady=1)

    def _look_inside(self):
        """Compare runs and Quick test; the look-inside pages are linked from the parts they explain (the side panel)."""
        Kicker(self.body, T["try_it"]).pack(anchor="w")
        row = tk.Frame(self.body, bg=C["bg"])
        row.pack(anchor="w", pady=6)
        self.links = {}
        for route, key in LINKS:
            link = tk_label(row, T[key], 14, "muted", cursor="hand2", takefocus=1)
            link.pack(side="left", padx=(0, 16))
            self.links[route] = link
            _activate(link, lambda r=route: self.navigate(r))
