"""Pretraining's decisions without Tk: the steps, the engine config, whether Continue is allowed, the size verdict, the
expectation and the params of the job "Add to the queue" makes. The Tk page and the API (studio_api.routers.pretraining) both use them."""
from __future__ import annotations

from studio_ia.expectation import expectation, table_params


T = {
    "h1": "Pretraining",
    "lead": "Feed it text. It learns by predicting the next word, over and over.",
    "model_label": "MODEL BEING TRAINED",
    "version_source": "Version 5 · from the Network designer",
    "data": "Data",
    "what_reads": "what it reads",
    "data_status_format": "{count}% · {mb} MB of {total} MB",
    "enough_text": "ENOUGH TEXT?",
    "text_estimate": "This model could use about {times}× more text than you have",
    "text_about_right": "You have about as much text as the rule of thumb asks for",
    "text_plenty": "You have plenty of text: about {times}× what the rule of thumb asks for",
    "text_unknown": "Choose some text to see whether it is enough for this model",
    "estimating": "Estimating how much text you have…",
    "reading_text": "Reading your text…",
    "steps_estimating": "Estimating steps and time…",
    "word_pieces_plain": "A common rule of thumb for the best use of compute is about 20 word pieces per parameter. Repeating your text helps less with each pass, and after a few passes it starts memorising; the run page marks the passes.",
    "word_pieces_rule": "{ratio} word pieces per parameter; a common rule of thumb for the best use of compute is about 20. Repeating your text helps less with each pass, and after a few passes it starts memorising; the run page marks the passes.",
    "word_pieces_plenty": "{ratio} word pieces per parameter; a common rule of thumb for the best use of compute is about 20. One pass over your text is already more than that, so repeating it is not needed.",
    "text_only": "Plain text only. Conversations such as Dolly 15K and Alpaca belong to",
    "post_training_r1": "Post-training round 1",
    "post_training_r1_end": "; problems with a checkable answer to round 2. A model learns the language from plain text first.",
    "batches": "Batches",
    "batches_desc": "how the text is fed in",
    "rebuilt_auto": "rebuilt automatically when the word list changes",
    "one_step": "One step: {rows} rows × 256 pieces. Each colour is a different example; the dark square marks where one ends.",
    "layout_note": "Layout shown with an example; real rows come from your data.",
    "rows_per_step": "Rows per step",
    "rows_desc": "more rows: steadier learning, more memory",
    "held_back": "Held back for testing",
    "held_back_desc": "split by example, exact duplicates removed, never trained on",
    "per_step_pieces": "{pieces} pieces per step · one pass over your data ≈ {passes} steps",
    "per_step_only": "{pieces} pieces per step",
    "train_label": "Train",
    "continue_start": "continue or start over",
    "version_needs": "version 5 needs catch-up training",
    "how_long": "How long",
    "steps_time": "{steps} steps · about {minutes} min on this computer",
    "steps_only": "{steps} steps · the time shows after the first run on this computer",
    "steps_none": "Choose some text to see how many steps this takes",
    "learning_rate": "Learning rate:",
    "lr_auto": "auto",
    "lr_info": "({lr}, set with width-scaled setup)",
    "schedule_label": "Schedule:",
    "schedule_value": "warm up, then steady",
    "schedule_note": "· a short cool-down makes each checkpoint",
    "schedule_desc": "A steady schedule means training can keep going for as long as you like, and runs of different lengths stay comparable at any point.",
    "add_queue": "Add to the queue",
    "queue_note": "Runs after the current job on this computer, or right away if nothing is running. Chat keeps {parent} until {version} beats it on held-back text.",
    "collection": "Recommended collection",
    "collection_desc": "plain text to download, checked, then added to your sources",
    "download_all": "Download and add all",
    "download": "Download and add",
    "import": "Add to my sources",
    "state_not downloaded": "Not downloaded",
    "state_downloading": "Downloading",
    "state_downloaded": "Downloaded",
    "state_imported": "In your sources",
    "held_out": "held-out: kept for testing, not trained on",
    "failed": "Could not finish: {error} Nothing was added; try again.",
}


ROWS = ("8", "16", "32")
HOLD = ("2%", "5%", "10%")
LENGTHS = {"Short": 1, "Medium": 2, "Long": 4}  # passes over the chosen text
RULE = 20  # word pieces per parameter, a rule of thumb only
ABOUT_RIGHT = (0.8 * RULE, 1.5 * RULE)  # ratios in this band read "about right"; below it too little, above it plenty
FEW_BELOW, MANY_ABOVE = 10, 40  # word pieces per parameter: under the first the text is short for the model, over the second the model could be bigger
CHARS_PER_PIECE = 4.0  # characters in a word piece when the tokenizer has not been measured on this text (English prose with a 12k-piece list is near it)


def estimated_pieces(characters, chars_per_piece=None):
    """Word pieces in `characters` of text, from the tokenizer's measured characters per piece, else CHARS_PER_PIECE. Never tokenizes."""
    return round(characters / (chars_per_piece or CHARS_PER_PIECE))


def size_advice(pieces, params):
    """(verdict, ratio, target_params) for `pieces` word pieces against a model of `params` parameters: "few" under FEW_BELOW pieces
    per parameter, "right" up to MANY_ABOVE, "many" above. target_params is the size at which this text is RULE pieces per parameter."""
    ratio = pieces / params
    verdict = "few" if ratio < FEW_BELOW else "right" if ratio <= MANY_ABOVE else "many"
    return verdict, ratio, round(pieces / RULE)


def word_list_changed(v, parent):
    """A different word list (tokenizer) makes a warm start invalid."""
    return parent is not None and v is not None and v.spec.get("tokenizer") != parent.spec.get("tokenizer")


def can_continue(v, parent):
    """Continue needs a parent version with a trained checkpoint and the same word list."""
    return parent is not None and bool(parent.best_checkpoint) and not word_list_changed(v, parent)


def forced_mode(v, parent, requested):
    """The mode a run may use: catch_up only when continuing is valid, otherwise from_scratch."""
    return requested if requested == "from_scratch" or can_continue(v, parent) else "from_scratch"


DESIGN_KEYS = ("tokenizer", "dims", "embedding", "position", "block", "blocks", "head")  # what sets the weights' shapes


def continue_sources(v, versions, absolute):
    """The trained versions a catch-up of `v` may start from (a best checkpoint and the same word list), `v` itself first (training
    it further on more text), then newest first, each with what its run did (run_source). `absolute` turns a stored checkpoint path
    into one on this computer."""
    if v is None:
        return []
    trained = [p for p in versions if p.best_checkpoint and not word_list_changed(v, p)]
    return [run_source(p, v, absolute) for p in sorted(trained, key=lambda p: (p.number != v.number, -p.number))]


def run_source(p, v, absolute):
    """What the run behind version `p`'s best checkpoint did, read from its run.json (and diloco.json for a run over several
    machines): how long it trained, its losses, its size and device, and whether `v` has the same design (every tensor is copied)
    or another one (only the matching ones are)."""
    from pathlib import Path

    run_dir = Path(absolute(p.best_checkpoint)).parent
    run, diloco = _json(run_dir / "run.json"), _json(run_dir / "diloco.json")
    settings = diloco.get("settings") if isinstance(diloco.get("settings"), dict) else {}
    seconds = run.get("duration_seconds") or _between(run.get("createdAt"), run.get("finishedAt"))
    return {"version": p.number, "name": p.name or "", "change": p.change or "", "status": run.get("status") or "",
            "finished_at": run.get("finishedAt"), "steps": _whole(run.get("step")), "steps_planned": _whole(run.get("steps")),
            "seconds": seconds or None, "tokens": _whole(run.get("trained_tokens")) or None, "params": _whole(run.get("parameters")),
            "best_val_loss": _finite(run.get("best_val_loss"), run.get("best_loss")), "train_loss": _finite(run.get("loss")),
            "device": run.get("device_name") or p.train_device or None, "rounds": _whole(settings.get("rounds")),
            "inner_steps": _whole(settings.get("inner_steps")), "same_design": same_design(p.spec, v.spec),
            "parent": p.number == v.parent, "this_version": p.number == v.number}


def same_design(a, b):
    """True when two specs give weights of the same shapes, so a start from one copies every tensor into the other."""
    return all((a or {}).get(key) == (b or {}).get(key) for key in DESIGN_KEYS)


def _json(path):
    from studio_core import read_json
    try:
        found = read_json(path) if path.exists() else {}
    except (OSError, ValueError):
        return {}
    return found if isinstance(found, dict) else {}


def _between(start, end):
    """Seconds from one ISO time to another; None when either is missing or unreadable."""
    from datetime import datetime
    try:
        return max(0., (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds())
    except (TypeError, ValueError):
        return None


def _whole(x):
    return int(x) if isinstance(x, (int, float)) and not isinstance(x, bool) and x == x else None


def _finite(*values):
    """The first of `values` that is a finite number; None when none is."""
    import math
    return next((float(x) for x in values if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)), None)


def steps_for(pieces, per_step, passes):
    """Steps for `passes` passes over `pieces` word pieces at `per_step` pieces a step; None when the size is unknown."""
    return max(1, -(-int(pieces) * passes // per_step)) if pieces and per_step else None


def pretrain_config(spec, steps, rows, hold):
    """Engine config for a plain-text run of `spec`: loss on every piece, `hold` of the text held back for testing."""
    from studio_spec import config_from_spec
    return {**config_from_spec(spec), "steps": steps, "batch_size": rows, "validation": hold, "objective": "all"}


def source_action(row):
    """(button label, service to call) for a collection row; (None, None) while it works or once it is in your sources."""
    return {"not downloaded": (T["download"], "download"), "downloaded": (T["import"], "import")}.get(row["state"], (None, None))


def source_status(row, note=""):
    """The plain-words line under a collection row: a failure first, then live progress, then the state."""
    if row.get("error"):
        return T["failed"].format(error=row["error"].rstrip(".") + ".")
    if row["state"] == "downloading":
        return note or T["state_downloading"]
    return T["state_" + row["state"]]


def held_out(dataset_id):
    """True for a catalogue set that is a validation or test split: trained on, it could no longer measure anything."""
    import studio_catalog as cat
    return next((e.get("split") for e in cat.ALL_CATALOG if f"catalog_{e['id']}" == dataset_id), None) in ("validation", "test")


def collection_items(rows, notes):
    """(key, data): each collection row with its latest progress line."""
    return [(r["id"], (dict(r), notes.get(r["id"], ""))) for r in rows]  # a copy: the services may change a row in place


def tile_items(sets, use):
    """(key, data): one tile per text set, ticked when it is used."""
    return [(s["id"], {"id": s["id"], "name": s["name"], "count": s["count"], "held": held_out(s["id"]), "on": bool(use.get(s["id"]))})
            for s in sets]


def enough_text(ratio):
    """(headline, key) for `ratio` word pieces per parameter: too little, about right, or plenty. Plenty means one pass
    already covers the rule of thumb, so repeating the text is not needed."""
    low, high = ABOUT_RIGHT
    if ratio < low:
        return T["text_estimate"].format(times=f"{RULE / max(ratio, 1e-9):.1f}"), "little"
    if ratio <= high:
        return T["text_about_right"], "about_right"
    return T["text_plenty"].format(times=f"{ratio / RULE:.0f}" if ratio >= 3 * RULE else f"{ratio / RULE:.1f}"), "plenty"


def ratio_text(ratio):
    """A word-pieces-per-parameter ratio in plain words: exact when small, rounded to two digits when large."""
    if ratio < 10:
        return f"{ratio:.2f}"
    if ratio < 100:
        return f"{ratio:.1f}"
    digits = len(str(int(ratio))) - 2
    return f"about {round(ratio, -digits):,.0f}"


def version_expectation(spec, preview, passes):
    """(headline, details) of what to expect from this version after `passes` passes over the chosen text; empty until the size preview is known."""
    params = (preview or {}).get("params")
    if not spec or not params:
        return "", ""
    tokenizer = spec["tokenizer"]["id"]
    pieces = preview.get("pieces")
    return expectation(params, tokenizer, pieces * passes if pieces else None, "text",
                       table_params(tokenizer, spec["dims"]["d_model"], preview.get("vocab")))


def sets_of(summary):
    return (summary["sets"] if isinstance(summary, dict) else summary.sets) or []


def user_text(sets, chosen):
    """True when a chosen set is the person's own text (not the bundled library): the queue then adds a restore job."""
    return any(not s.get("library") for s in sets if s["id"] in chosen)


def plateau_rule(steps, run_hours, setting=None):
    """The plateau rule (studio_plateau) a job of `steps` steps gets, or None for none. `setting`: True always, False never, None by
    length: on for a run expected to take over an hour (`run_hours`, from the measured step rate), and when it is not measured yet for
    one of at least LONG_STEPS steps."""
    import studio_plateau
    if setting is False or not steps:
        return None
    long = run_hours > studio_plateau.LONG_RUN_SECONDS / 3600 if run_hours is not None else steps >= LONG_STEPS
    return studio_plateau.default_rule(steps) if setting or long else None


LONG_STEPS = 20_000  # a run this long is taken to need the plateau rule when its speed is not known


def plateau_line(rule):
    """The plain sentence for a plateau rule: when the run will end itself."""
    if not rule:
        return ""
    return (f"Stops by itself, keeping its best, once the best validation loss improves by less than {rule['min_gain']:.0%} over "
            f"{rule['window_steps']:,} steps (not before step {rule['min_steps']:,}).")


def job_params(v, parent, chosen, mode, rows, hold, steps, pieces, own_text, plateau=None, source=None):
    """(label, params) of the pretrain job "Add to the queue" adds for version `v` (see Job in studio_ia.services). A catch-up starts
    from `source`'s best checkpoint (default `parent`; `v` itself trains further from its own), and `mode` is forced to from_scratch when that is not valid; `hold` is the
    held-back fraction, `pieces` the size preview's; `plateau` the rule that ends the run when its validation loss goes flat (plateau_rule)."""
    cfg = pretrain_config(v.spec, steps, rows, hold)
    start = source or parent
    mode = forced_mode(v, start, mode)
    init = start.best_checkpoint if mode == "catch_up" else None
    label = f"Version {v.number} {'catch-up' if mode == 'catch_up' else 'from scratch'}"
    if mode == "catch_up" and start is not parent and start is not v:
        label += f" from version {start.number}"
    token_limit = (v.spec.get("train", {}).get("run", {}) or {}).get("token_limit_millions", 0)
    return label, {"config": cfg, "datasets": list(chosen), "mode": mode, "init_from": init, "output_dir": None,
                   "user_text": own_text, "previous_best": parent.best_checkpoint if parent else None,
                   "version": v.number, "heldout_fraction": cfg["validation"], "pieces_per_pass": pieces,
                   **({"token_budget": token_limit * 1_000_000} if token_limit else {}),
                   **({"plateau": plateau} if plateau else {})}
