"""Bounded parameter choices between autopilot phases.

The caller persists the returned choice before starting a phase. Measurements may be
attached to a transition as ``metrics``; a completed training checkpoint also has
``metrics.jsonl`` alongside it. No decision depends on a random draw or changes files.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path


LIMITS = {
    "learning_rate": (1e-7, 1.), "steps": (1, 20_000_000),
    "batch_size": (1, 128), "accumulation": (1, 64), "context": (32, 2048),
    "width": (16, 512), "layers": (1, 12),
    "beta": (1e-4, 2.), "sft_weight": (0., 10.), "kl_coefficient": (0., 10.),
}
INTEGERS = {"steps", "batch_size", "accumulation", "context", "width", "layers"}


def _range(name, bounds):
    allowed = LIMITS[name]
    value = bounds.get(name, allowed)
    if isinstance(value, dict):
        value = (value.get("min"), value.get("max"))
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} bounds must be a minimum and maximum.")
    low, high = value
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in (low, high)):
        raise ValueError(f"{name} bounds must be finite numbers.")
    if low < allowed[0] or high > allowed[1] or low > high:
        raise ValueError(f"{name} bounds must fit the supported range {allowed}.")
    if name in INTEGERS and (int(low) != low or int(high) != high):
        raise ValueError(f"{name} bounds must be whole numbers.")
    return low, high


def _bounded(name, value, bounds):
    low, high = _range(name, bounds)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    chosen = max(low, min(high, value))
    return int(chosen) if name in INTEGERS else float(chosen)


def _metrics(last):
    """Only the prior phase's own measurements; a missing or bad log is no evidence."""
    metrics = last.get("metrics") if isinstance(last.get("metrics"), dict) else {}
    if "loss_curve" in metrics or not last.get("checkpoint"):
        return metrics
    try:
        path = Path(last["checkpoint"]).parent / "metrics.jsonl"
        with path.open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
    except (OSError, ValueError, TypeError):
        return metrics
    curve = [[r["step"], r["val_loss"]] for r in rows
             if isinstance(r, dict) and isinstance(r.get("step"), (int, float))
             and isinstance(r.get("val_loss"), (int, float))]
    return {**metrics, "loss_curve": curve}


def _probe(config, metrics, bounds, evidence, reasons):
    """Choose only among finished, finite LR probes; never infer an optimum from one run."""
    points = metrics.get("lr_probe")
    if not isinstance(points, list) or len(points) < 2:
        return
    clean = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            continue
        lr, loss = point
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in (lr, loss)):
            low, high = _range("learning_rate", bounds)
            if low <= lr <= high:
                clean.append((float(lr), float(loss)))
    if len(clean) < 2:
        return
    from studio_experiments import best_lr

    before = config["learning_rate"]
    config["learning_rate"] = best_lr(clean)
    evidence["lr_probe"] = {"points": clean, "selected": config["learning_rate"]}
    if config["learning_rate"] != before:
        reasons.append("the measured LR probe favored a lower held-out loss")


def _curve(config, kind, last, metrics, bounds, evidence, reasons):
    if last.get("kind") != kind or "steps" not in config:
        return
    points = metrics.get("loss_curve")
    if not isinstance(points, list):
        return
    from studio_scaling import fit_curve, predict, FLAT_PER_DOUBLING

    try:
        fit = fit_curve(points)
    except (TypeError, ValueError):
        return
    if fit is None:
        return
    current = max(float(p[0]) for p in points if isinstance(p, (list, tuple)) and len(p) >= 2)
    now, later = predict(fit, current), predict(fit, current * 2)
    gain = now - later
    evidence["loss_fit"] = {"points": fit["points"], "rmse": fit["rmse"],
                            "predicted_gain_per_doubling": gain / now, "last_step": current}
    if gain <= max(FLAT_PER_DOUBLING * now, 2 * fit["rmse"]):
        return
    before = config["steps"]
    config["steps"] = _bounded("steps", math.ceil(before * 1.5), bounds)
    if config["steps"] > before:
        reasons.append("the same phase's held-out loss is still falling beyond fit noise")


def _proposal(config, last, metrics, bounds, evidence, reasons):
    if not (metrics.get("too_small") is True or "too small" in str(last.get("reason", "")).lower()):
        return
    proposal = {}
    for name, factor in (("width", 1.25), ("layers", 1.25)):
        if name in config and isinstance(config[name], int) and not isinstance(config[name], bool):
            larger = _bounded(name, math.ceil(config[name] * factor), bounds)
            if larger > config[name]:
                proposal[name] = larger
    evidence["design_proposal"] = proposal or None
    reasons.append("the prior plateau says this design is too small" + ("; a larger design is proposed" if proposal else "; growth is at the user limit"))


def choose(phase, prior_transitions, bounds):
    """Return ``{spec, reason, evidence}`` for the next phase.

    ``phase`` is the full phase spec, with training knobs under ``config``. Bounds
    map knob names to ``(min, max)`` or ``{min, max}``. Unknown phase keys survive.
    Architecture growth is a proposal, never an implicit weight-shape change.
    """
    if not isinstance(phase, dict) or not isinstance(phase.get("config"), dict):
        raise ValueError("A phase needs a config dictionary.")
    if not isinstance(bounds, dict):
        raise ValueError("Bounds must be a dictionary.")
    if not isinstance(prior_transitions, (list, tuple)):
        raise ValueError("Prior transitions must be a sequence.")
    spec = copy.deepcopy(phase)
    config = spec["config"]
    evidence, reasons = {}, []
    for name in LIMITS:
        if name in config:
            old = config[name]
            config[name] = _bounded(name, old, bounds)
            if config[name] != old:
                reasons.append(f"{name} was limited to the user-set range")
    for name in ("token_budget", "wall_seconds"):
        if name in spec and name in bounds:
            low, high = bounds[name]
            if not 0 < low <= high:
                raise ValueError(f"{name} bounds must be positive and ordered.")
            spec[name] = max(low, min(high, spec[name]))
    if not prior_transitions:
        return {"spec": spec, "reason": "; ".join(reasons) or "fixed phase defaults; no earlier measurements", "evidence": evidence}
    last = prior_transitions[-1]
    if not isinstance(last, dict):
        raise ValueError("Each transition must be a dictionary.")
    metrics = _metrics(last)
    evidence["prior_phase"] = {"index": last.get("index"), "kind": last.get("kind"), "accepted": last.get("accepted")}
    if "learning_rate" in config:
        _probe(config, metrics, bounds, evidence, reasons)
        if metrics.get("diverged") is True or "diverg" in str(last.get("reason", "")).lower():
            before = config["learning_rate"]
            config["learning_rate"] = _bounded("learning_rate", before / 2, bounds)
            evidence["divergence"] = {"prior_reason": last.get("reason"), "learning_rate_before": before}
            if config["learning_rate"] < before:
                reasons.append("the prior phase diverged, so the learning rate was halved")
    _curve(config, spec.get("kind"), last, metrics, bounds, evidence, reasons)
    if isinstance(metrics.get("max_batch_size"), int) and not isinstance(metrics["max_batch_size"], bool) and "batch_size" in config:
        if metrics["max_batch_size"] < _range("batch_size", bounds)[0]:
            raise ValueError("The measured memory fit cannot meet the minimum batch size.")
        before = config["batch_size"]
        config["batch_size"] = _bounded("batch_size", min(before, metrics["max_batch_size"]), bounds)
        evidence["memory_fit"] = {"max_batch_size": metrics["max_batch_size"]}
        if config["batch_size"] < before:
            reasons.append("the prior memory fit requires a smaller batch")
    _proposal(config, last, metrics, bounds, evidence, reasons)
    return {"spec": spec, "reason": "; ".join(reasons) or "fixed phase defaults; no actionable earlier measurement", "evidence": evidence}
