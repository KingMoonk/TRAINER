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
from studio_arch import model_spec_presets
from studio_compute import MAX_EPOCHS, TOKENS_PER_PARAM, plan, step_seconds, throughput
from studio_estimate import estimate
from studio_glossary import MODULE_TERMS, glossary_text
from studio_slots import BLOCK_SLOTS
from studio_spec import block_at, config_from_spec, set_block, validate_spec
from studio_validate import errors, validate
from studio_versions import OTHER_PARTS, _facts, spec_diff
from studio_warmstart import NOTES
from studio_ia.attribution import attribute, blocks_change_text, run_changes, signed_move
from studio_ia import pieces as cut
from studio_ia.services import Job
from studio_ia.copy import BUILDER as T
from studio_ia.expectation import expectation


def say(x, y, more, less):
    """Describe a cost ratio using the same wording in Builder and Quick test."""
    r = y / x
    if abs(r - 1) < 0.05:
        return T["about_same"]
    return T["about_multiplier"].format(mult=f"{r if r > 1 else 1 / r:.1f}") + " " + (more if r > 1 else less)

FF = (2, 4, 8)  # thinking-room width as a multiple of d_model: narrow / normal / wide
TOKENIZERS = ("byte", "bpe_custom", "bpe_gpt2", "word")
VS = (256, 8192, 50257, 8192)  # word-list sizes behind letters / small / large / whole words (the cap; less on little data)
WORDS = ("Letters", "Small word list", "Large word list", "Whole words")
WIDTHS = ("Narrow", "Normal", "Wide")
HEADS = (1, 4, 8)
PART_SLOTS = ("embedding", "position", "head")  # the model-wide parts the tower shows; every block has BLOCK_SLOTS
PART_TERMS = dict(embedding="embedding", position="position", norm="norm", mixer="token_mixer", channel="mlp", head="output_head")
WORD_TERMS = ("bytes", "bpe", "gpt2_bpe", "whole_words")  # GLOSSARY entries behind WORDS
STEPS, ROWS = 1000, 16  # a catch-up job here; the Pretraining page is where length, rows and data are chosen
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
    d, rows = s["d"], VS[s["V"]]
    upto = T["up_to"] if s["V"] in (1, 3) else ""  # built from your text: the size is a cap
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
