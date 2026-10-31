"""The Auto Run's parameter controller: between phases it picks the next phase's settings from what the phases before it showed.

Pure: nothing here touches the queue, a model or the disk. The orchestrator (studio_ia.autorun) asks `decide` when it builds a phase's
job and applies the settings it returns; it asks `record_attempt` when a phase ends. Both return NEW state and leave their argument alone.

Rules (each needs real evidence, and does nothing without it):
* divergence: a round attempt that stopped on a non-finite loss (studio_model raises "Non-finite loss. Reduce the learning rate.") is
  tried again at half its learning rate (`LR_CUT`);
* still falling: when the pretrain run's fitted curve (studio_scaling.fit_curve over its validation points) still lowers the error by
  at least `studio_scaling.FLAT_PER_DOUBLING` per doubling of what it read, the mid-training phase gets `LENGTHEN` times its steps;
* too small: when that curve went flat well before the run ended (`TOO_SMALL_SHARE`, the same test the pretraining plan uses), a bigger
  design is PROPOSED, with studio_stages.TOO_SMALL's reason. It is shown, never applied: growing the design is the person's call;
* gate history: a round 4 attempt whose gate said no is tried again with a larger DPO beta, and a round 2 attempt whose gate said no with
  a larger KL coefficient, both pulling the model closer to the weights it started from (`PULL`).

* next cycle (#901): when the last phase, the rlhf loop, stopped because the report card stopped improving (`cycle_decision`), the run
  starts another cycle of its enabled phases if the loop kept at least one round and fewer than `max_cycles` have run, and finishes
  otherwise. Either way the decision and its evidence are kept in `state["decisions"]` and journalled ("cycle"). A bigger design is
  still only proposed (rule bigger_design), never applied.

Limits: `plan["controller"] = {"enabled": bool, "bounds": {key: {"min", "max"}}, "max_cycles": 1..MAX_CYCLES (default 1)}` for the keys in `KEYS`. No choice leaves its bounds (a value
is clamped, and the choice says so); the person sees and edits them in the plan. Without a `controller` in the plan nothing is tuned.

State: `state["attempts"]` keeps what each ended attempt showed (the evidence that survives a retry, which clears the phase's own result),
`state["choices"]` one record per decision: {phase, attempt, key, was, value, rule, why, evidence, applied, clamped}. Every choice is also
a "tuned" line in the run's journal, with the same evidence."""
from __future__ import annotations

import copy
import math

import studio_autorun as ar
import studio_scaling as scaling
import studio_stages
import studio_training_stages

KEYS = ("learning_rate", "steps", "beta", "kl")
DEFAULT_BOUNDS = {"learning_rate": {"min": 1e-6, "max": 1e-3}, "steps": {"min": 10, "max": 20000},
                  "beta": {"min": .02, "max": .5}, "kl": {"min": 0., "max": .2}}
LABEL = {"learning_rate": "Learning rate", "steps": "Steps", "beta": "DPO beta", "kl": "KL pull-back", "design": "Design"}
LR_START = {"sft": 1e-4, "rlvr": 1e-4, "distill": 1e-4}   # the rate each round starts at when the plan gives none (studio_rl: sft_learning_rate, practice_with_checks, learn_from_teacher)
KL_START, BETA_START = .02, .1                            # studio_rl.KL_DEFAULT and studio_rlhf.DPO_DEFAULTS["beta"]
LR_CUT = .5          # a diverged attempt's next learning rate is this share of its last
LENGTHEN = 1.5       # a phase whose curve is still falling gets this many times its steps
PULL = 1.5           # a failed gate's next beta (or KL coefficient) is this many times its last
TOO_SMALL_SHARE = .6  # the curve is flat before this share of the run (studio_ia.sizing uses the same figure)
MAX_ATTEMPTS = 50
MAX_CYCLES = 5        # the most a plan may allow
DEFAULT_CYCLES = 1    # one cycle: the run finishes after the first, unless the person allows more
MAX_POINTS = 200      # validation points a phase result keeps
SETTING = {"learning_rate": "learning_rate", "steps": "steps", "beta": "beta", "kl": "kl"}   # the plan setting each key writes
APPLIES = {"learning_rate": ("sft", "rlvr", "distill"), "steps": ("midtrain",), "beta": ("dpo",), "kl": ("rlvr",)}


def default_controller():
    return {"enabled": True, "bounds": copy.deepcopy(DEFAULT_BOUNDS), "max_cycles": DEFAULT_CYCLES}


def validate(controller):
    """Every problem with a plan's `controller`, as plain sentences."""
    if not isinstance(controller, dict):
        return ["The controller must be an object with enabled and bounds."]
    problems = []
    if not isinstance(controller.get("enabled"), bool):
        problems.append("The controller needs enabled set to true or false.")
    cycles = controller.get("max_cycles", DEFAULT_CYCLES)
    if isinstance(cycles, bool) or not isinstance(cycles, int) or not 1 <= cycles <= MAX_CYCLES:
        problems.append(f"The most cycles must be a whole number from 1 to {MAX_CYCLES}.")
    bounds = controller.get("bounds")
    if not isinstance(bounds, dict):
        return problems + ["The controller needs its bounds as an object."]
    for key in KEYS:
        found = bounds.get(key)
        if not isinstance(found, dict):
            problems.append(f"The controller needs bounds for {LABEL[key].lower()}.")
            continue
        low, high = found.get("min"), found.get("max")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in (low, high)):
            problems.append(f"The {LABEL[key].lower()} bounds must be numbers.")
        elif low < 0 or high < low or (key != "kl" and high <= 0):
            problems.append(f"The {LABEL[key].lower()} bounds need 0 <= min <= max (and a max above 0).")
    return problems


def enabled(plan):
    controller = plan.get("controller")
    return bool(controller) and controller.get("enabled") is True and not validate(controller)


def clamp(plan, key, value):
    """(value inside the plan's bounds for `key`, whether it had to move). Steps are whole numbers."""
    found = plan["controller"]["bounds"][key]
    moved = min(max(value, found["min"]), found["max"])
    if key == "steps":
        moved = int(round(moved))
    return moved, moved != (int(round(value)) if key == "steps" else value)


# --------------------------------------------------------------------------- the evidence

def diverged(error):
    """Whether a phase's error sentence is the engine's non-finite loss stop."""
    return "non-finite" in str(error or "").lower()


def _gate(result):
    """The small part of a round's result worth keeping: whether its gate passed, the sentence, and how far it moved from the start weights."""
    result = result if isinstance(result, dict) else {}
    out = {"ok": result.get("ok", result.get("passed")), "verdict": str(result.get("verdict") or "")}
    if isinstance(result.get("mean_kl"), (int, float)):
        out["mean_kl"] = result["mean_kl"]
    return out


def record_attempt(plan, state, phase, job_status, result, now):
    """State with what the ended attempt of `phase` showed appended to `state["attempts"]`: its status, error, whether it diverged, the
    gate it met, and the settings it ran with (`used`). A "blocked" outcome ran nothing and is not kept."""
    if job_status == "blocked" or not enabled(plan):
        return state
    state = copy.deepcopy(state)
    entry = state["phases"][ar._index(plan, phase)]
    result = result if isinstance(result, dict) else {}
    error = str(result.get("error") or "")
    row = {"phase": phase, "attempt": entry["attempt"], "status": job_status, "error": error, "diverged": diverged(error),
           "gate": _gate(result), "used": used(plan, state, phase, entry["attempt"])}
    attempts = [a for a in state.get("attempts", []) if (a["phase"], a["attempt"]) != (phase, entry["attempt"])]
    state["attempts"] = (attempts + [row])[-MAX_ATTEMPTS:]
    return state


def used(plan, state, phase, attempt):
    """The tuned settings `phase` ran with in `attempt`: what the controller chose, else what the plan says."""
    settings = plan["phases"][ar._index(plan, phase)]["settings"]
    out = {key: settings[key] for key in KEYS if isinstance(settings.get(SETTING[key]), (int, float)) and not isinstance(settings.get(SETTING[key]), bool)}
    for choice in state.get("choices", []):
        if choice["phase"] == phase and choice["attempt"] == attempt and choice["applied"]:
            out[choice["key"]] = choice["value"]
    return out


def _last_attempt(state, phase, attempt):
    earlier = [a for a in state.get("attempts", []) if a["phase"] == phase and a["attempt"] < attempt]
    return max(earlier, key=lambda a: a["attempt"]) if earlier else None


def curve_report(points):
    """What the pretrain run's validation points say: {"fit", "last", "falling", "flat_share"} or None when they cannot be fitted.
    `points` is [(x, validation loss)] with x in tokens (or steps) read. falling: doubling what it read still lowers the fitted error
    by FLAT_PER_DOUBLING of it; flat_share: where the curve went flat as a share of the run."""
    fit = scaling.fit_curve(points or [])
    if not fit:
        return None
    xs = [x for x, y in points if x and x > 0]
    first, last = min(xs), max(xs)
    now = scaling.predict(fit, last)
    gain = (now - scaling.predict(fit, 2 * last)) / now
    flat = scaling.flat_point(fit, first, last)
    return {"floor": round(float(fit["floor"]), 4), "exponent": round(float(fit["exponent"]), 3), "points": int(fit["points"]),
            "gain_per_doubling": round(float(gain), 4), "falling": bool(gain >= scaling.FLAT_PER_DOUBLING), "flat_share": round(float(flat / last), 3), "loss_end": round(float(now), 4)}


# --------------------------------------------------------------------------- the decision

def _choice(plan, phase, attempt, key, was, wanted, rule, why, evidence):
    value, clamped = clamp(plan, key, wanted)
    if clamped:
        why = f"{why} Held to your {LABEL[key].lower()} limit of {value:g}."
    return {"phase": phase, "attempt": attempt, "key": key, "was": was, "value": value, "rule": rule, "why": why, "evidence": evidence,
            "applied": True, "clamped": clamped}


def decide(plan, state, phase, curve=None, width=None):
    """The choices for `phase`'s next job, [] when the controller is off or nothing in the evidence calls for one. `curve` is the
    pretrain run's validation points [(tokens read, loss)] (Services.loss_evidence), `width` the design's width (for mid-training's default
    steps). A choice already made for this phase and attempt is returned as it was, so asking again (a restart) never changes it."""
    if not enabled(plan):
        return []
    entry = state["phases"][ar._index(plan, phase)]
    attempt = entry["attempt"]
    made = [c for c in state.get("choices", []) if c["phase"] == phase and c["attempt"] == attempt]
    if made:
        return copy.deepcopy(made)
    settings = plan["phases"][ar._index(plan, phase)]["settings"]
    out = []
    before = _last_attempt(state, phase, attempt)

    if before and before["diverged"] and phase in APPLIES["learning_rate"]:
        was = before["used"].get("learning_rate") or LR_START.get(phase)   # (what the failed attempt ran with)
        if was:
            out.append(_choice(plan, phase, attempt, "learning_rate", was, was * LR_CUT, "divergence",
                               f"Attempt {before['attempt']} stopped on a loss that was not a number, so the learning rate is cut to {LR_CUT:.0%} of {was:g}.",
                               {"attempt": before["attempt"], "error": before["error"]}))

    if before and before["gate"]["ok"] is False:
        for key, kind, start, name in (("beta", "dpo", BETA_START, "beta"), ("kl", "rlvr", KL_START, "KL coefficient")):
            if phase == kind:
                was = before["used"].get(key, start)
                evidence = {"attempt": before["attempt"], "verdict": before["gate"]["verdict"], **({"mean_kl": before["gate"]["mean_kl"]} if "mean_kl" in before["gate"] else {})}
                out.append(_choice(plan, phase, attempt, key, was, max(was, 1e-6) * PULL, "gate_history",
                                   f"Attempt {before['attempt']} did not pass its gate, so the {name} goes from {was:g} up to hold the model closer to the weights it started from.",
                                   evidence))

    pretrain = state["phases"][ar._index(plan, "pretrain")] if any(p["kind"] == "pretrain" for p in plan["phases"]) else None
    if curve is None and pretrain and isinstance(pretrain.get("result"), dict):
        curve = pretrain["result"].get("curve")      # (the points the pretrain phase kept in its own result)
    report = curve_report(curve) if pretrain and pretrain["status"] == "passed" else None
    if report:
        if phase == "midtrain" and report["falling"]:
            base = settings.get("steps") or (studio_training_stages.default_steps(width) if width else None)
            if base:
                out.append(_choice(plan, phase, attempt, "steps", base, base * LENGTHEN, "still_falling",
                                   f"The pretraining curve still falls {report['gain_per_doubling']:.1%} for every doubling of what it read (the cut-off is "
                                   f"{scaling.FLAT_PER_DOUBLING:.0%}), so mid-training runs {LENGTHEN:g} times as many steps.", report))
        if report["flat_share"] < TOO_SMALL_SHARE and not any(c["rule"] == "bigger_design" for c in state.get("choices", [])):
            share = f"{report['flat_share']:.0%}"
            out.append({"phase": phase, "attempt": attempt, "key": "design", "was": None, "value": None, "rule": "bigger_design",
                        "why": f"{studio_stages.TOO_SMALL}The pretraining curve went flat after {share} of the run. {studio_stages.SUGGESTION} "
                               "This is a proposal: nothing was changed.", "evidence": report, "applied": False, "clamped": False})
    small = next((e for e in state["phases"] if isinstance(e.get("result"), dict) and e["result"].get("too_small")), None)
    if small and not any(c["rule"] == "bigger_design" for c in [*state.get("choices", []), *out]):
        out.append({"phase": phase, "attempt": attempt, "key": "design", "was": None, "value": None, "rule": "bigger_design",
                    "why": f"{studio_stages.TOO_SMALL}The {small['kind']} stage stopped as too small. {studio_stages.SUGGESTION} This is a proposal: nothing was changed.",
                    "evidence": {"phase": small["kind"], "too_small": True}, "applied": False, "clamped": False})
    return out


def settings_with(settings, choices):
    """`settings` with the applied choices written in."""
    out = dict(settings)
    for choice in choices:
        if choice["applied"] and choice["key"] in SETTING:
            out[SETTING[choice["key"]]] = choice["value"]
    return out


def add_choices(state, choices, now):
    """State with `choices` kept in `state["choices"]` and one "tuned" journal event each, with the evidence. Choices already kept
    (same phase, attempt and key) are left alone."""
    fresh = [c for c in choices if not any((o["phase"], o["attempt"], o["key"]) == (c["phase"], c["attempt"], c["key"]) for o in state.get("choices", []))]
    if not fresh:
        return state
    state = copy.deepcopy(state)
    state["choices"] = state.get("choices", []) + fresh
    for choice in fresh:
        ar._event(state, "tuned", now, phase=choice["phase"], attempt=choice["attempt"], key=choice["key"], was=choice["was"], value=choice["value"],
                  rule=choice["rule"], applied=choice["applied"], why=choice["why"], evidence=choice["evidence"])
    return state


# --------------------------------------------------------------------------- another cycle, or finish

def max_cycles(plan):
    return int((plan.get("controller") or {}).get("max_cycles", DEFAULT_CYCLES))


def cycle_decision(plan, state):
    """After the last phase: {cycle, decision: "another" | "finish", why, evidence}, or None when there is nothing to decide (the controller
    is off, the rlhf phase did not pass, or this cycle was decided already). From the rlhf phase's result: why the loop stopped, how many
    rounds it kept and what the report card said got better in them.
    * the loop did not stop on a flat report card (it ran out of rounds or time): finish, since nothing says another cycle would help;
    * it stopped flat and kept no round: finish, another cycle would repeat what found nothing;
    * it stopped flat after keeping rounds, and the plan's cycle limit is reached: finish;
    * otherwise: another cycle (the enabled phases run again, on whatever data the workspace holds by then)."""
    if not enabled(plan) or not any(p["kind"] == "rlhf" for p in plan["phases"]):
        return None
    entry = state["phases"][ar._index(plan, "rlhf")]
    result = entry.get("result")
    cycle = state.get("cycle", 1)
    if entry["status"] != "passed" or not isinstance(result, dict) or any(d["cycle"] == cycle for d in state.get("decisions", [])):
        return None
    reason = str(result.get("stop_reason") or "")
    kept = int(result.get("kept") or 0)
    rounds = result.get("rounds") if isinstance(result.get("rounds"), list) else []
    better = sorted({name for r in rounds if r.get("status") == "kept" for name in r.get("card_better") or []})
    limit = max_cycles(plan)
    evidence = {"stop_reason": reason, "kept": kept, "rounds": len(rounds), "card_better": better, "cycle": cycle, "max_cycles": limit,
                "bigger_design_proposed": any(c["rule"] == "bigger_design" for c in state.get("choices", []))}
    if "stopped improving" not in reason.lower():
        decision, why = "finish", "The RLHF loop did not stop on a flat report card, so nothing says another cycle would help."
    elif not kept:
        decision, why = "finish", "The report card stopped improving and the loop kept no round, so another cycle would repeat what found nothing."
    elif cycle >= limit:
        decision, why = "finish", f"The report card stopped improving, and you allowed {limit} cycle{'s' if limit != 1 else ''}; this was cycle {cycle}."
    else:
        decision = "another"
        why = (f"The report card stopped improving after the loop kept {kept} round{'s' if kept != 1 else ''}"
               f"{' (better: ' + ', '.join(better) + ')' if better else ''}, and cycle {cycle} of {limit} is done, so cycle {cycle + 1} starts.")
        if evidence["bigger_design_proposed"]:
            why += " The bigger design proposed earlier is still yours to apply."
    return {"cycle": cycle, "decision": decision, "why": why, "evidence": evidence}


def add_decision(state, decision, now):
    """State with `decision` kept in `state["decisions"]` and one "cycle" journal event with its evidence."""
    state = copy.deepcopy(state)
    state["decisions"] = state.get("decisions", []) + [decision]
    ar._event(state, "cycle", now, cycle=decision["cycle"], decision=decision["decision"], why=decision["why"], evidence=decision["evidence"])
    return state
