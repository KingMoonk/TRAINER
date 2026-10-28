"""Auto Run parameter controller (#729): from what the last phase showed, choose the next phase's learning rate, steps and DPO beta.
Pure: no queue, model or disk. The orchestrator (studio_ia.autorun) records `outcome(...)` of every phase that ends and asks `decide(...)`
before it builds the next phase's job; both results are journalled and shown on the Auto Run page.

Rules (each choice names its evidence; a rule that has nothing to act on changes nothing):
* a phase that diverged (a loss that is not finite, or a result that says so): the next phase starts at half the learning rate that
  diverged (`LR_AFTER_DIVERGENCE`);
* a phase whose fitted curve (studio_scaling.fit_curve over `result["curve"]`, [(tokens or steps, validation loss)]) still falls by at
  least the cut-off per doubling (studio_scaling.FLAT_PER_DOUBLING): the next phase runs `LONGER` times the steps; one whose curve is flat
  runs `SHORTER` times (stop paying once doubling buys less than the cut-off);
* a plateau on the model's newest text (`result["too_small"]`, set by studio_stages, or a flat fitted curve on pretrain): a bigger design
  is PROPOSED, never started by itself (it changes what the person is training);
* round 4 (DPO) that failed: beta up when the held-back text error rose (stay closer to the reference), down when preference accuracy was
  not above the gate's threshold (let it move).

Bounds: `plan["bounds"]` = {"learning_rate"|"steps"|"beta": {"min", "max"}}, over DEFAULT_BOUNDS. Every value the controller sets, and every
tuned value already in the settings, is clamped into them; a choice that the bounds cut says so."""
from __future__ import annotations

import math

import studio_scaling

DEFAULT_BOUNDS = {"learning_rate": {"min": 1e-6, "max": 1e-2}, "steps": {"min": 10, "max": 100000}, "beta": {"min": 0.01, "max": 1.0}}
INTEGER = ("steps",)
TUNED = {"midtrain": ("steps",), "sft": ("learning_rate", "steps"), "rlvr": ("learning_rate", "steps"),
         "distill": ("learning_rate", "steps"), "dpo": ("learning_rate", "steps", "beta")}   # what each phase's job reads (studio_ia.autorun._build)
LR_AFTER_DIVERGENCE = 0.5
LONGER, SHORTER = 1.5, 0.75
BETA_UP, BETA_DOWN = 1.5, 0.7
GROW_PHASES = ("pretrain", "midtrain")


def bounds_of(plan):
    """The bounds in force: DEFAULT_BOUNDS with the plan's own over them."""
    out = {k: dict(v) for k, v in DEFAULT_BOUNDS.items()}
    for key, value in ((plan or {}).get("bounds") or {}).items():
        if key in out and isinstance(value, dict):
            out[key].update({k: value[k] for k in ("min", "max") if value.get(k) is not None})
    return out


def validate_bounds(bounds):
    """Every problem with a plan's `bounds`, as sentences."""
    if bounds in (None, {}):
        return []
    if not isinstance(bounds, dict):
        return ["The bounds must be an object."]
    problems = []
    for key, value in bounds.items():
        if key not in DEFAULT_BOUNDS:
            problems.append(f"There is no bound named {key!r}; use {', '.join(DEFAULT_BOUNDS)}.")
        elif not isinstance(value, dict) or not all(isinstance(value.get(k), (int, float)) and not isinstance(value.get(k), bool)
                                                    and math.isfinite(value.get(k)) and value.get(k) > 0 for k in ("min", "max")):
            problems.append(f"The {key} bound needs a min and a max above 0.")
        elif value["min"] > value["max"]:
            problems.append(f"The {key} bound's min is above its max.")
    return problems


def _clamp(key, value, bounds):
    low, high = bounds[key]["min"], bounds[key]["max"]
    value = min(max(value, low), high)
    return int(round(value)) if key in INTEGER else float(f"{value:.6g}")


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def outcome(kind, attempt, status, result, used):
    """What a phase that ended showed, as the small record `decide` reads and the page shows: {phase, attempt, status, settings (the
    tuned ones it ran with), diverged, ok, after_bpc, failed (names of the gate checks that failed), gain_per_doubling (fitted, or None),
    too_small}."""
    result = result if isinstance(result, dict) else {}
    after = result.get("after") if isinstance(result.get("after"), dict) else {}
    text = f"{result.get('error', '')} {result.get('verdict', '')}".lower()
    losses = [after.get(k) for k in ("bpc", "conv_loss") if k in after]
    diverged = bool(result.get("diverged")) or any(v is not None and not _finite(v) for v in losses) or "diverg" in text or "nan" in text.split()
    gain = None
    fit = studio_scaling.fit_curve(result["curve"]) if result.get("curve") else None
    if fit:
        last = max(float(x) for x, _ in result["curve"])
        now = studio_scaling.predict(fit, last)
        gain = (now - studio_scaling.predict(fit, 2 * last)) / now if now > 0 else 0.0
    flat = gain is not None and gain < studio_scaling.FLAT_PER_DOUBLING
    return {"phase": kind, "attempt": attempt, "status": status, "ok": result.get("ok"), "diverged": diverged,
            "settings": {k: used[k] for k in ("learning_rate", "steps", "beta") if _finite(used.get(k))}, "after_bpc": after.get("bpc") if _finite(after.get("bpc")) else None,
            "failed": [str(c.get("name", "")) for c in result.get("checks") or [] if isinstance(c, dict) and c.get("ok") is False],
            "gain_per_doubling": gain, "too_small": bool(result.get("too_small")) or (kind in GROW_PHASES and flat)}


def decide(kind, settings, bounds, history):
    """The settings `kind` runs with, from the plan's `settings` and the `history` of outcomes (oldest first).
    Returns {"settings", "choices": [{"param", "from", "to", "why"}], "proposal": {"action", "why"} | None, "evidence": outcome | None}."""
    settings = dict(settings)
    choices, proposal = [], None
    last = history[-1] if history else None
    names = TUNED.get(kind, ())
    for key in names:        # a value already in the settings is held to the bounds too
        if _finite(settings.get(key)) and _clamp(key, settings[key], bounds) != settings[key]:
            was, settings[key] = settings[key], _clamp(key, settings[key], bounds)
            choices.append({"param": key, "from": was, "to": settings[key], "why": f"The plan's {key.replace('_', ' ')} was outside your bounds."})

    def set_(key, target, why):
        was = settings.get(key)
        clamped = _clamp(key, target, bounds)
        note = "" if clamped == (int(round(target)) if key in INTEGER else float(f"{target:.6g}")) else f" Held to your bound ({bounds[key]['min']:g} to {bounds[key]['max']:g})."
        if clamped != was:
            settings[key] = clamped
            choices.append({"param": key, "from": was, "to": clamped, "why": why + note})

    if last:
        where = f"the {last['phase']} phase" + (f" (attempt {last['attempt']})" if last["attempt"] > 1 else "")
        base_lr = last["settings"].get("learning_rate", settings.get("learning_rate"))
        if "learning_rate" in names and last["diverged"] and _finite(base_lr):
            set_("learning_rate", base_lr * LR_AFTER_DIVERGENCE, f"{where.capitalize()} diverged, so the learning rate is halved from {base_lr:g}.")
        gain = last["gain_per_doubling"]
        base_steps = last["settings"].get("steps", settings.get("steps"))
        if "steps" in names and gain is not None and _finite(base_steps) and not last["diverged"]:
            cut = studio_scaling.FLAT_PER_DOUBLING
            if gain >= cut:
                set_("steps", base_steps * LONGER, f"The loss curve of {where} is still falling ({gain:.1%} per doubling, cut-off {cut:.0%}), so the steps grow from {base_steps:g}.")
            else:
                set_("steps", base_steps * SHORTER, f"The loss curve of {where} is flat ({gain:.1%} per doubling, below the {cut:.0%} cut-off), so the steps shrink from {base_steps:g}.")
        if last["too_small"]:
            proposal = {"action": "grow", "why": f"The loss of {where} has stopped falling on the newest text: this design has learned what it can. "
                                                 "Continue with a bigger design (a warm start keeps what it learned)."}
        if kind == "dpo" and "beta" in names and last["phase"] == "dpo" and last["status"] in ("failed", "done") and last["ok"] is False:
            beta = last["settings"].get("beta", settings.get("beta", 0.1))
            if any("error did not rise" in name for name in last["failed"]):
                set_("beta", beta * BETA_UP, f"Held-back text error rose in {where}, so beta goes up from {beta:g} to stay closer to the starting model.")
            elif any("preference accuracy" in name for name in last["failed"]):
                set_("beta", beta * BETA_DOWN, f"Preference accuracy was not above the gate in {where}, so beta goes down from {beta:g} to let the model move.")
    return {"settings": settings, "choices": choices, "proposal": proposal, "evidence": last}
