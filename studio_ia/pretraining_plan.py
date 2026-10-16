"""Pretraining's decisions without Tk: the steps, the engine config, whether Continue is allowed, the size verdict, the
expectation and the params of the job "Add to the queue" makes. The Tk page and the API (studio_api.routers.pretraining) both use them."""
from __future__ import annotations

from studio_ia.copy import PRETRAINING as T
from studio_ia.expectation import expectation, table_params

ROWS = ("8", "16", "32")
HOLD = ("2%", "5%", "10%")
LENGTHS = {"Short": 1, "Medium": 2, "Long": 4}  # passes over the chosen text
RULE = 20  # word pieces per parameter, a rule of thumb only
ABOUT_RIGHT = (0.8 * RULE, 1.5 * RULE)  # ratios in this band read "about right"; below it too little, above it plenty


def word_list_changed(v, parent):
    """A different word list (tokenizer) makes a warm start invalid."""
    return parent is not None and v is not None and v.spec.get("tokenizer") != parent.spec.get("tokenizer")


def can_continue(v, parent):
    """Continue needs a parent version with a trained checkpoint and the same word list."""
    return parent is not None and bool(parent.best_checkpoint) and not word_list_changed(v, parent)


def forced_mode(v, parent, requested):
    """The mode a run may use: catch_up only when continuing is valid, otherwise from_scratch."""
    return requested if requested == "from_scratch" or can_continue(v, parent) else "from_scratch"


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
    return next((e.get("split") for e in cat.CATALOG if f"catalog_{e['id']}" == dataset_id), None) in ("validation", "test")


def collection_items(rows, notes):
    """(key, data) for KeyedRows: each collection row with its latest progress line."""
    return [(r["id"], (dict(r), notes.get(r["id"], ""))) for r in rows]  # a copy: the services may change a row in place


def tile_items(sets, use):
    """(key, data) for KeyedRows: one tile per text set, ticked when it is used."""
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


def job_params(v, parent, chosen, mode, rows, hold, steps, pieces, own_text):
    """(label, params) of the pretrain job "Add to the queue" adds for version `v` (see Job in studio_ia.services). `mode` is
    forced to from_scratch when continuing from `parent` is not valid; `hold` is the held-back fraction, `pieces` the size preview's."""
    cfg = pretrain_config(v.spec, steps, rows, hold)
    mode = forced_mode(v, parent, mode)
    init = parent.best_checkpoint if mode == "catch_up" else None
    label = f"Version {v.number} {'catch-up' if mode == 'catch_up' else 'from scratch'}"
    return label, {"config": cfg, "datasets": list(chosen), "mode": mode, "init_from": init, "output_dir": None,
                   "user_text": own_text, "previous_best": parent.best_checkpoint if parent else None,
                   "version": v.number, "heldout_fraction": cfg["validation"], "pieces_per_pass": pieces}
