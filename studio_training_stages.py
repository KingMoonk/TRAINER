"""Mid-training from an existing checkpoint. Chat SFT is Round 1 (studio_rl.run_round), so it is not duplicated here.

The worker owns training and measurement; the app adopts the candidate only after
the returned gate has passed. Autopilot calls ``run_stage`` directly with the same
interface as a queued single stage.
"""
from __future__ import annotations

import json
from pathlib import Path

import studio_eval
import studio_gate
import studio_model
import studio_rl

STAGES = ("midtrain",)
MIX = {
    "midtrain": {"text": .4, "smol_smoltalk": .3, "mmlu_auxiliary": .15, "gsm8k": .15},
}
DEFAULT_CHARS = 2_000_000


def stage_config(stage, checkpoint, rows, overrides=None):
    """The stage's bounded default recipe; a caller can lower steps for a small run."""
    if stage not in STAGES:
        raise ValueError(f"Unknown training stage: {stage}.")
    base = studio_model.load_checkpoint(checkpoint, weights_only=True)
    old = base["config"]
    steps = 2000 if old["width"] < 512 else 5000
    cfg = {**old, "objective": "all", "learning_rate": min(3e-4, old["learning_rate"] * .3),
           "steps": steps, "warmup": max(1, round(steps * .03)), "eval_every": max(1, steps // 10),
           "validation": .05 if len(rows) >= 20 else 0.}
    cfg.update(overrides or {})
    cfg["objective"] = "all"
    if overrides and "steps" in overrides:
        if "warmup" not in overrides:
            cfg["warmup"] = max(1, round(cfg["steps"] * .03))
        if "eval_every" not in overrides:
            cfg["eval_every"] = max(1, cfg["steps"] // 10)
    cfg["warmup"] = min(cfg["warmup"], max(0, cfg["steps"] - 1))
    cfg["eval_every"] = min(cfg["eval_every"], cfg["steps"])
    return studio_model.validate_config(cfg)


def _gate(checkpoint, candidate, context, stage, stop):
    """Measure the same held-back rows on both weights and include the frozen assistant suite."""
    docs = context.post_rows("heldout_docs")
    problems = context.post_rows("heldout_problems")
    conversations = context.post_rows("heldout_conversations")
    scores = studio_rl.doc_scores(checkpoint, docs, stop)
    band = studio_eval.band_from_scores(scores) if scores else None
    before = studio_rl.measure(checkpoint, docs, problems, stop, conversations, scores)
    after = studio_rl.measure(candidate, docs, problems, stop, conversations)
    ok, checks = studio_gate.passes(before, after, band, number=None)
    card = context.gate_eval(candidate, baseline=checkpoint) if hasattr(context, "gate_eval") else None
    if card and card.get("verdict"):
        checks.extend(card["verdict"]["checks"])
        ok = ok and card["verdict"]["ok"]
    if before.get("conv_loss") is not None and after.get("conv_loss") is not None:
        changed = after["conv_loss"] <= before["conv_loss"]
        checks.append({"name": "Held-back chat: answer error did not rise", "ok": changed,
                       "detail": f"{after['conv_loss']:.3f} per answer token, was {before['conv_loss']:.3f}"})
        ok = ok and changed
    return ok, before, after, band, checks, card


def run_stage(stage, checkpoint, context, output_dir, emit, stop, pause, *, config=None, datasets=None, resume=None, candidate=None):
    """Train a stage and return its gated candidate without changing any version.

    ``datasets`` maps source ids to positive shares; absent means the installed
    starter catalogue. ``context.stage_rows`` excludes frozen held-out data.
    Result: {stage, ok, checkpoint, before, after, band, checks, card, verdict}.
    """
    if stage not in STAGES:
        raise ValueError(f"Unknown training stage: {stage}.")
    checkpoint = str(checkpoint)
    if not Path(checkpoint).is_file():
        raise ValueError("The version's checkpoint is missing. Choose a trained version.")
    if candidate is None:
        if resume:
            frozen = Path(resume).parent / "dataset.jsonl"
            if not frozen.is_file():
                raise ValueError("The interrupted stage has no dataset snapshot to resume.")
            rows = [json.loads(line) for line in frozen.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            rows = context.stage_rows(stage, datasets or MIX[stage], max_chars=DEFAULT_CHARS)
        if not rows:
            raise ValueError(f"No training rows for {stage}. Download its starter datasets in Your data, or choose source sets.")
        cfg = stage_config(stage, checkpoint, rows, config)
        out = Path(output_dir)
        trained = studio_model.train(cfg, rows, out, emit=emit, stop=stop, pause=pause,
                                     resume=resume, init_from=None if resume else checkpoint)
        candidate = str(out / "best.npz" if (out / "best.npz").exists() else out / "last.npz")
    else:
        trained = {"status": "completed"}
        candidate = str(candidate)
    if stop.is_set() or not Path(candidate).is_file() or trained.get("status") != "completed":
        return {"stage": stage, "ok": False, "checkpoint": candidate if Path(candidate).is_file() else checkpoint,
                "stopped": True, "verdict": "The stage stopped before its gate completed."}
    try:
        ok, before, after, band, checks, card = _gate(checkpoint, candidate, context, stage, stop)
    except studio_eval.Stopped:
        return {"stage": stage, "ok": False, "checkpoint": candidate, "stopped": True,
                "verdict": "The gate stopped before it could compare the weights."}
    result = {"stage": stage, "ok": bool(ok), "checkpoint": candidate, "before": before, "after": after,
              "band": band, "checks": checks, "card": card,
              "verdict": "The stage passed its held-back checks." if ok else "The stage finished, but its held-back checks failed; the version keeps its previous weights."}
    emit({"type": "gate", **result})
    return result
