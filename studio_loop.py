"""The automated RLHF loop: round after round from the current best version, kept or rolled back by evidence, resumable for days.

One round is: (1) collect preferences from the current best: rule-checked pairs where a problem has a rule (exact, nothing to
exploit) and judge-graded best-of-N pairs elsewhere; (2) a preference round (DPO or IPO, `studio_rlhf.dpo_train`); (3) the gate;
(4) keep the result as the new best, or roll back. Rounds repeat until a stop rule says so.

Reward hacking is what a loop that runs for days is for ever doing to itself, so the gate never trusts the judge alone:
* held-back text and checked problems the judge never sees are split in two halves: the GATE half decides, the AUDIT half is
  measured every round and can only veto, never approve. A candidate that passes the gate while the audit half regresses against
  the loop's starting point is rolled back and flagged, and repeated flags stop the loop ("suspected judge exploitation");
* the candidate is gated against the current best AND against the starting model (the anchor), with a drift budget, so a slow
  slide of small in-noise losses cannot add up;
* a round is kept only if nothing regressed AND something improved: more checked problems solved, or the judge prefers it on
  prompts that were held back from pair building, without its replies growing longer than `max_length_ratio` (length is the
  commonest thing a judge is hacked with).

Everything is on disk: `state.json` (atomic, rewritten at each phase), `journal.jsonl` (one line per finished round, with all
the evidence), `pairs_kept.jsonl` (replay) and `round_NNNN/` folders (collect.jsonl is written prompt by prompt, so a restart in
the middle of a day of judging does not pay for it twice). A restart reads the state and carries on from the last kept round.

Placement: a round's collect phase is the `collect(spec, ...)` function, whose spec and result are plain JSON. `run_loop(...,
collector=...)` takes any callable with that signature, which is where a peer-side runner plugs in (studio_remote, #442).
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
import threading
import time
from pathlib import Path

import studio_gate
import studio_rules
from studio_core import atomic_json, now, uid

DEFAULTS = {
    "max_rounds": 20,           # stop rule: rounds tried (kept or not)
    "patience": 4,              # stop rule: this many rounds in a row that were not kept
    "max_hours": None,          # stop rule: working time across restarts (None = no limit)
    "prompts_per_round": 24, "candidates": 4, "max_tokens": 120, "temperature_low": .6, "temperature_high": 1.0,
    "dpo_steps": 60, "beta": .1, "learning_rate": 1e-4, "loss_type": "dpo", "replay": 200, "device": "auto",
    "gate_prompts": 8,          # prompts kept back from pair building, for the judge's head-to-head
    "gate_samples": 2, "gate_temperature": .4, "win_margin": .05, "tie_eps": .5,
    "max_length_ratio": 1.5,    # replies may not grow longer than this share of the parent's on the gate prompts
    "drift_bpc": .02,           # held-back error may rise this share above the starting model's, in total
    "drift_pass": .02,          # checked-problem pass rate may fall this much below the starting model's, in total
    "audit_flags": 2,           # stop after this many kept rounds in a row that pass the gate and fail the audit
    "keep_checkpoints": 3,      # kept round folders on disk besides the starting model and the current best
    "judge_wait_s": 6 * 3600,   # how long to wait for an unreachable judge before stopping
    "max_failures": 8, "seed": 5,
}
STATUSES = ("running", "waiting_judge", "stopped", "finished", "failed")
KEPT, ROLLED_BACK, TOO_FEW, JUDGE_DOWN = "kept", "rolled_back", "too_few_pairs", "judge_down"


class LoopError(ValueError):
    """A loop that cannot start, with the sentence the page shows."""


class JudgeDown(RuntimeError):
    """The judge failed too many prompts in a row."""


# --------------------------------------------------------------------------- small helpers

def clean_config(config):
    cfg = {**DEFAULTS, **{k: v for k, v in (config or {}).items() if k in DEFAULTS}}
    for key, low, high in (("max_rounds", 1, 100000), ("patience", 1, 1000), ("prompts_per_round", 2, 5000), ("candidates", 2, 8),
                           ("dpo_steps", 1, 5000), ("gate_prompts", 0, 200), ("gate_samples", 1, 8), ("audit_flags", 1, 100),
                           ("keep_checkpoints", 1, 50), ("replay", 0, 100000), ("max_failures", 1, 1000)):
        cfg[key] = int(cfg[key])
        if not low <= cfg[key] <= high:
            raise LoopError(f"{key} must be between {low} and {high}.")
    for key in ("beta", "learning_rate", "win_margin", "max_length_ratio", "drift_bpc", "drift_pass"):
        cfg[key] = float(cfg[key])
    if cfg["max_hours"] is not None:
        cfg["max_hours"] = float(cfg["max_hours"])
        if cfg["max_hours"] <= 0:
            raise LoopError("max_hours must be more than 0, or left empty.")
    if cfg["loss_type"] not in ("dpo", "ipo"):
        raise LoopError("loss_type must be dpo or ipo.")
    return cfg


def _key(value):
    return hashlib.sha1(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:16]


def split_halves(rows):
    """(gate half, audit half) of held-back rows: a fixed split by content, so it is the same after a restart and does not move when
    rows are added or reordered."""
    ordered = sorted(rows, key=_key)
    return ordered[0::2], ordered[1::2]


def _doc(row):
    return row if isinstance(row, str) else row.get("text") or f"{row.get('user', '')}\n{row.get('assistant', '')}"


def rate(evidence):
    k, n = evidence.get("pass_k", 0), evidence.get("pass_n", 0)
    return k / n if n else None


def _num(value):
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


# --------------------------------------------------------------------------- the files

def paths(directory):
    d = Path(directory)
    return {"state": d / "state.json", "journal": d / "journal.jsonl", "replay": d / "pairs_kept.jsonl"}


def load_state(directory):
    try:
        return json.loads(paths(directory)["state"].read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def save_state(directory, state):
    state["updated"] = now()
    Path(directory).mkdir(parents=True, exist_ok=True)
    atomic_json(paths(directory)["state"], state)


def read_jsonl(path):
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return []
    rows = []
    for line in text.split("\n"):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass  # a line cut by a crash: that prompt is simply done again
    return rows


def append_jsonl(path, row):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        stream.flush()


def summary(state, journal_tail=None):
    """The state as the API shows it: the rounds without their bulky parts."""
    out = {k: state.get(k) for k in ("id", "version", "status", "stop_reason", "config", "anchor", "current", "kept", "failed_streak",
                                     "audit_flagged", "elapsed_s", "created", "updated", "phase", "note", "current_eval", "anchor_eval")}
    out["rounds"] = [{k: v for k, v in r.items() if k != "examples"} for r in state.get("rounds", [])]
    out["round"] = state.get("next_round", 1)
    return out


def list_loops(run_dir):
    """Every loop in a run folder (newest first): their summaries."""
    found = []
    for folder in Path(run_dir).glob("rlhf_loop_*"):
        state = load_state(folder)
        if state:
            found.append(summary(state))
    return sorted(found, key=lambda s: s.get("created") or "", reverse=True)


# --------------------------------------------------------------------------- collect (the placeable phase)

def rule_pair(model, problem, cfg, rng, stop):
    """A pair from a problem with a rule: the shortest passing answer against a failing one. (pairs, info); no pair when all of the
    answers pass or all fail. The rule is the label, so nothing here can be argued with."""
    from studio_rl import _messages, _prompt_text
    from studio_rlhf import is_cut, temperature_spread
    messages = _messages(problem)
    prompt, rule = _prompt_text(messages), problem["rule"]
    n = cfg["candidates"]
    budget = studio_rules.max_tokens(rule)
    answers = model.generate_group(messages, n, max_tokens=budget, top_k=40, stop=stop,
                                   temperature=temperature_spread(n, None, cfg["temperature_low"], cfg["temperature_high"]))
    unique = list(dict.fromkeys(a for a in answers if a.strip()))
    cut = {a for a in unique if is_cut(model.tokenizer, a, budget)}   # ran into the budget: never ended, so never a chosen answer
    passed = [a for a in unique if a not in cut and studio_rules.check(rule, prompt, a, problem.get("params"))]
    failed = [a for a in unique if a not in passed]
    info = {"skip": None, "passed": len(passed), "of": len(unique)}
    if not passed or not failed:
        info["skip"] = "every answer passed" if passed else "no answer passed"
        return [], info
    chosen = min(passed, key=len)
    rejected = rng.choice(failed)
    return [{"messages": messages, "chosen": chosen, "rejected": rejected, "rejected_cut": rejected in cut, "kind": "verified", "reference": None,
             "source_id": None, "reason": f"rule: {rule}", "scores": [1., 0.], "n_candidates": len(unique)}], info


def collect(spec, judge, emit=None, stop=None, pause=None, done=None, record=None, model=None):
    """Build the round's pairs from `spec`: {"round", "checkpoint", "items": [{"key", "problem" | "prompt"}], "config", "seed"}.

    Returns {"pairs": [...], "skipped": {reason: n}, "errors": n}. `done` maps an item's key to the result an earlier attempt
    stored (it is not asked again), `record(key, result)` stores each fresh one. Raises JudgeDown when the judge fails
    `max_failures` prompts in a row. Spec and result are JSON, so another machine can run this."""
    from studio_model import LocalModel
    from studio_rlhf import TeacherError, _skip_label, teacher_pairs
    emit, stop, pause = emit or (lambda e: None), stop or threading.Event(), pause or threading.Event()
    cfg = clean_config(spec["config"])
    rng = random.Random(spec["seed"])
    done = done or {}
    pairs, skipped, errors, streak = [], {}, 0, 0
    local = model or LocalModel(spec["checkpoint"], seed=spec["seed"], device=cfg["device"])
    teacher_cfg = {"candidates": cfg["candidates"], "max_tokens": cfg["max_tokens"], "temperatures": None, "rewrite_below": 4,
                   "min_margin": 1, "tiebreak_gap": .5, "extra_pairs": False}
    items = spec["items"]
    for index, item in enumerate(items):
        while pause.is_set() and not stop.is_set():
            time.sleep(.1)
        if stop.is_set():
            break
        if item["key"] in done:
            result = done[item["key"]]
        else:
            result = {"pairs": [], "skip": None}
            try:
                if "problem" in item:
                    made, info = rule_pair(local, item["problem"], cfg, rng, stop)
                else:
                    made, info = teacher_pairs(judge, local, item["prompt"], {**teacher_cfg, "temperatures": None}, rng, stop, judge)
                result = {"pairs": made, "skip": info.get("skip")}
                streak = 0
            except Exception as exc:  # one bad judge reply must never end a day's run
                if stop.is_set():
                    break
                errors += 1
                streak += 1
                result = {"pairs": [], "skip": _skip_label(exc), "error": str(exc)[:200]}
                if streak >= cfg["max_failures"] or (isinstance(exc, TeacherError) and exc.fatal):
                    raise JudgeDown(f"The judge failed {streak} prompt(s) in a row ({result['skip']}: {result['error']}).") from None
            if result["skip"] == "stopped":
                break
            if record:
                record(item["key"], result)
        pairs += result["pairs"]
        if result.get("skip") and not result["pairs"]:
            skipped[result["skip"]] = skipped.get(result["skip"], 0) + 1
        emit({"type": "loop_phase", "phase": "collect", "round": spec["round"], "done": index + 1, "total": len(items), "pairs": len(pairs)})
    return {"pairs": pairs, "skipped": skipped, "errors": errors}


# --------------------------------------------------------------------------- the decision

def decide(parent, candidate, anchor, band, anchor_band, judge, length_ratio, cfg):
    """(kept, checks, improved). parent/candidate/anchor: the gate half's {"bpc", "pass_k", "pass_n"}. band: the parent's noise band.
    judge: {"win_rate", "needed", ...} or None. A check's ok is None when nothing could be measured (it never fails a round).

    Kept only when every check holds AND something improved (more checked problems solved, or the judge won)."""
    ok, checks = studio_gate.passes(parent, candidate, band)
    b0, b1 = anchor.get("bpc"), candidate.get("bpc")
    if b0 is not None and b1 is not None and math.isfinite(b0) and math.isfinite(b1):
        limit = max(b0 * (1 + cfg["drift_bpc"]), anchor_band[1] if anchor_band else b0)
        checks.append({"name": "Drift from the starting model: held-back error", "ok": bool(b1 <= limit),
                       "detail": f"{b1:.3f} bits per character, the starting model had {b0:.3f}, the budget allows {limit:.3f}"})
    else:
        checks.append({"name": "Drift from the starting model: held-back error", "ok": None, "detail": "not measured: no held-back text"})
    r0, r1 = rate(anchor), rate(candidate)
    if r0 is not None and r1 is not None:
        checks.append({"name": "Drift from the starting model: checked problems", "ok": bool(r1 >= r0 - cfg["drift_pass"]),
                       "detail": f"{candidate['pass_k']} of {candidate['pass_n']} solved, the starting model solved {anchor['pass_k']} of {anchor['pass_n']}"})
    else:
        checks.append({"name": "Drift from the starting model: checked problems", "ok": None, "detail": "no held-back problems to check"})
    if length_ratio is None:
        checks.append({"name": "Replies did not grow longer", "ok": None, "detail": "not measured: no judge prompts"})
    else:
        checks.append({"name": "Replies did not grow longer", "ok": bool(length_ratio <= cfg["max_length_ratio"]),
                       "detail": f"replies are {length_ratio:.2f} times as long as the parent's, the limit is {cfg['max_length_ratio']:.2f}"})
    p0, p1 = rate(parent), rate(candidate)
    solved_more = p0 is not None and p1 is not None and candidate["pass_k"] > parent["pass_k"] and p1 > p0
    judged = bool(judge) and judge.get("win_rate") is not None and judge["win_rate"] >= judge["needed"]
    said = []
    if solved_more:
        said.append("more checked problems solved")
    if judged:
        said.append(f"the judge preferred it in {judge['win_rate'] * 100:.0f}% of held-back comparisons")
    if not said:
        said.append("no more checked problems solved")
        if judge and judge.get("win_rate") is not None:
            said.append(f"the judge's win rate {judge['win_rate'] * 100:.0f}% is under the {judge['needed'] * 100:.0f}% needed")
    checks.append({"name": "Something improved", "ok": bool(solved_more or judged), "detail": " and ".join(said)})
    kept = all(c["ok"] is not False for c in checks)
    return kept, checks, {"solved_more": bool(solved_more), "judge": bool(judged)}


def audit_check(anchor_audit, candidate_audit, band):
    """The audit half against the starting model, with the gate's own rule: (ok, checks)."""
    if not anchor_audit or not candidate_audit:
        return True, []
    return studio_gate.passes(anchor_audit, candidate_audit, band)


def stop_reason(state, cfg, elapsed):
    """Why the loop must stop now (a sentence), or None."""
    if state["next_round"] > cfg["max_rounds"]:
        return f"Reached the {cfg['max_rounds']} rounds it was asked to try."
    if state["failed_streak"] >= cfg["patience"]:
        return f"{state['failed_streak']} rounds in a row were not kept: the version has stopped improving here."
    if cfg["max_hours"] is not None and elapsed >= cfg["max_hours"] * 3600:
        return f"Used the {cfg['max_hours']:g} hours it was given."
    if state["audit_flagged"] >= cfg["audit_flags"]:
        return ("Suspected judge exploitation: the last rounds passed the gate while the audit half, which nothing trains on or "
                "decides with, got worse than the starting model. Stopped on the last kept version.")
    return None


# --------------------------------------------------------------------------- the loop

def build_items(rows, judge_prompts, judge, cfg, round_number):
    """The round's prompts: checked problems and judge prompts in proportion to how many there are, a fresh draw each round."""
    pool = [{"key": _key(["p", p.get("prompt"), p.get("rule")]), "problem": p} for p in rows.get("problems", []) if p.get("rule") in studio_rules.CHECKABLE]
    pool += [{"key": _key(["j", p["messages"]]), "prompt": p} for p in judge_prompts] if judge else []
    random.Random(cfg["seed"] * 100003 + round_number).shuffle(pool)
    return pool[:cfg["prompts_per_round"]]


def mean_length(examples, side):
    lengths = [len(e[side]) for e in examples or []]
    return sum(lengths) / len(lengths) if lengths else None


def run_loop(directory, checkpoint, rows, judge, config=None, emit=None, stop=None, pause=None, version=None, loop_id=None,
             measure=None, band=None, trainer=None, versus=None, collector=None, wait=None, clock=None, budget_token=None):
    """Run (or resume) the loop in `directory`. rows: {"conversations", "problems", "heldout_docs", "heldout_problems"} as
    studio_jobs/post_rows give them. judge: an OllamaTeacher-like object or None (checked problems only). The keyword seams are
    for tests and for placing work elsewhere (collector); each defaults to the real thing. Returns the final state's summary."""
    import studio_eval
    import studio_rl
    import studio_rlhf
    emit = emit or (lambda e: None)
    stop, pause = stop or threading.Event(), pause or threading.Event()
    measure = measure or studio_rl.measure
    band = band or (lambda ckpt, docs: studio_eval.noise_band(ckpt, [_doc(d) for d in docs]) if docs else None)
    trainer = trainer or studio_rlhf.dpo_train
    collector = collector or collect
    wait = wait or (lambda seconds: stop.wait(seconds))
    clock = clock or time.monotonic
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state = load_state(directory)
    docs_gate, docs_audit = split_halves(rows.get("heldout_docs", []))
    probs_gate, probs_audit = split_halves(rows.get("heldout_problems", []))
    if state is None:
        cfg = clean_config(config)
        checkable = [p for p in rows.get("problems", []) if p.get("rule") in studio_rules.CHECKABLE]
        if not judge and not checkable:
            raise LoopError("The loop needs something to learn from: problems a rule can check, or an Ollama judge chosen in settings.")
        if not (docs_gate or probs_gate):
            raise LoopError("The loop needs held-back text or held-back problems to gate on, so it cannot be fooled by its own judge.")
        if not checkpoint:
            raise LoopError("Train this version in Pretraining first.")
        state = {"id": loop_id or uid("loop"), "version": version, "status": "running", "stop_reason": "", "config": cfg,
                 "anchor": str(checkpoint), "current": str(checkpoint), "anchor_eval": None, "current_eval": None, "anchor_audit": None,
                 "current_audit": None, "current_band": None, "anchor_band": None, "audit_band": None, "next_round": 1, "kept": 0,
                 "failed_streak": 0, "audit_flagged": 0, "elapsed_s": 0., "created": now(), "rounds": [], "phase": "", "note": ""}
    elif budget_token and state.get("budget_token") != budget_token:
        # Continue, asked by the user: the stop rules are asked afresh (this many more rounds), the streaks start again. The token makes
        # an automatic re-run after a restart of the same job not do it twice.
        asked = clean_config(config)
        state["config"] = clean_config({**state["config"], **{k: v for k, v in (config or {}).items() if k in DEFAULTS},
                                        "max_rounds": state["next_round"] - 1 + asked["max_rounds"]})
        state.update(failed_streak=0, audit_flagged=0, budget_token=budget_token)
    cfg = clean_config(state["config"])
    judge_prompts = []
    if judge:
        every = studio_rlhf.prompts_from(rows.get("conversations", []))
        every.sort(key=lambda p: _key(p["messages"]))
        reserved = min(cfg["gate_prompts"], max(0, len(every) - 4))
        gate_prompts, judge_prompts = every[:reserved], every[reserved:]  # the gate's prompts are never used to make pairs
    else:
        gate_prompts = []
    started = clock()
    base_elapsed = state["elapsed_s"]

    def elapsed():
        return base_elapsed + clock() - started

    def save(**fields):
        state.update(fields)
        state["elapsed_s"] = elapsed()
        save_state(directory, state)
        emit({"type": "loop_state", "loop": state["id"], **summary(state)})

    def progress(phase, round_number, **extra):
        emit({"type": "loop_progress", "phase": phase, "round": round_number, "step": len(state["rounds"]), "total": cfg["max_rounds"], **extra})

    def emit_inner(event):  # the preference round's own step counter must not become the job's progress
        event = dict(event)
        if event.get("type") == "metric":
            event["type"] = "loop_metric"
            event.pop("total", None)
        emit(event)

    save(status="running", stop_reason="", note="")
    # --- the starting point: measured once, and kept in the state
    if state["anchor_eval"] is None:
        save(phase="measuring the starting model")
        progress("measure", 0)
        gate0 = measure(state["anchor"], docs_gate, probs_gate)
        audit0 = measure(state["anchor"], docs_audit, probs_audit) if (docs_audit or probs_audit) else None
        save(anchor_eval=gate0, current_eval=gate0, anchor_audit=audit0, current_audit=audit0, anchor_band=band(state["anchor"], docs_gate),
             current_band=band(state["anchor"], docs_gate), audit_band=band(state["anchor"], docs_audit) if docs_audit else None)
    waited = 0.
    reason, judge_lost = None, False
    while not stop.is_set():
        reason = stop_reason(state, cfg, elapsed())
        if reason:
            break
        number = state["next_round"]
        round_dir = directory / f"round_{number:04d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        t0 = clock()
        record = {"round": number, "started": now(), "node": "", "parent": state["current"]}
        items = build_items(rows, judge_prompts, judge, cfg, number)
        spec = {"round": number, "checkpoint": state["current"], "items": items, "config": cfg, "seed": cfg["seed"] * 1000 + number}
        save(phase=f"round {number}: collecting preferences")
        done = {r["key"]: r["result"] for r in read_jsonl(round_dir / "collect.jsonl")}
        try:
            collected = collector(spec, judge, emit, stop, pause, done=done, record=lambda key, result: append_jsonl(round_dir / "collect.jsonl", {"key": key, "result": result}))
        except JudgeDown as exc:
            if waited >= cfg["judge_wait_s"]:
                reason = f"The judge stayed unreachable for {waited / 3600:.1f} hours. Start the loop again when it is back: it carries on from round {number}."
                judge_lost = True
                break
            save(status="waiting_judge", note=str(exc), phase=f"round {number}: waiting for the judge")
            pause_for = min(600., 30. * 2 ** min(5, int(waited // 60)))
            wait(pause_for)
            waited += pause_for
            continue
        waited = 0.
        if stop.is_set():
            break
        save(status="running", note="")
        pairs = collected["pairs"]
        for p in pairs:
            p.setdefault("id", uid("pair"))
        record["pairs"] = {"total": len(pairs), "skipped": collected["skipped"], "errors": collected["errors"],
                           **{kind: sum(p.get("kind") == kind for p in pairs) for kind in ("verified", "comparison", "rewrite")}}
        replay = read_jsonl(paths(directory)["replay"])[-cfg["replay"]:] if cfg["replay"] else []
        record["replayed"] = len(replay)
        verdict, candidate_path, dpo, flagged = None, None, None, False
        if len(pairs) < 2:
            verdict, why = TOO_FEW, f"Only {len(pairs)} usable pair(s) from {len(items)} prompts: too few to train on."
            record.update(checks=[], before=state["current_eval"], after=None)
        else:
            save(phase=f"round {number}: training")
            progress("train", number)
            dpo_config = {"device": cfg["device"], "beta": cfg["beta"], "learning_rate": cfg["learning_rate"], "steps": cfg["dpo_steps"],
                          "loss_type": cfg["loss_type"], "validation": .1, "seed": spec["seed"]}
            try:
                dpo = trainer(state["current"], pairs + replay, round_dir / "dpo", dpo_config, emit_inner, stop, pause, name=f"RLHF loop round {number}")
            except Exception as exc:  # a failed preference round is a rolled-back round, not the end of a multi-day run
                verdict, why = ROLLED_BACK, f"Training failed ({str(exc)[:200]}); kept the previous version."
                record.update(checks=[], before=state["current_eval"], after=None, error=str(exc)[:300])
            if stop.is_set():
                break
            if dpo is not None:
                candidate_path = dpo["checkpoint"]
                save(phase=f"round {number}: checking the result against held-back data")
                progress("gate", number)
                after = measure(candidate_path, docs_gate, probs_gate)
                audit_after = measure(candidate_path, docs_audit, probs_audit) if state["anchor_audit"] else None
                judged, ratio = None, None
                if judge and gate_prompts:
                    try:
                        if versus:
                            result = versus(candidate_path, state["current"], gate_prompts, judge)
                        else:
                            result = studio_rlhf.compare_checkpoints(
                                candidate_path, state["current"], gate_prompts, judge, cfg["gate_samples"], cfg["max_tokens"], cfg["gate_temperature"],
                                True, cfg["tie_eps"], cfg["device"], stop, emit_inner, random.Random(spec["seed"]), len(gate_prompts) * cfg["gate_samples"])
                        judged = {"win_rate": result.get("win_rate"), "wins": result.get("wins"), "ties": result.get("ties"), "losses": result.get("losses"),
                                  "n": result.get("n"), "needed": .5 + cfg["win_margin"]}
                        a, b = mean_length(result.get("examples"), "a"), mean_length(result.get("examples"), "b")
                        ratio = a / b if a is not None and b else None
                    except Exception as exc:
                        if stop.is_set():
                            break
                        judged = {"win_rate": None, "needed": .5 + cfg["win_margin"], "error": str(exc)[:200]}
                if stop.is_set():
                    break
                kept, checks, improved = decide(state["current_eval"], after, state["anchor_eval"], state["current_band"], state["anchor_band"],
                                                judged, ratio, cfg)
                audit_ok, audit_checks = audit_check(state["anchor_audit"], audit_after, state["audit_band"])
                record.update(checks=checks, before=state["current_eval"], after=after, anchor=state["anchor_eval"], judge=judged,
                              length_ratio=ratio, improved=improved, audit={"before": state["anchor_audit"], "after": audit_after, "ok": audit_ok,
                                                                          "checks": audit_checks})
                if kept and not audit_ok:  # the audit half can only veto: it never approves
                    kept = False
                    flagged = True
                    checks.append({"name": "Audit half: no worse than the starting model", "ok": False,
                                   "detail": "; ".join(c["detail"] for c in audit_checks if c["ok"] is False)})
                verdict = KEPT if kept else ROLLED_BACK
                why = ("Kept: nothing got worse and something improved." if kept else
                       "Rolled back: " + "; ".join(c["name"].lower() for c in checks if c["ok"] is False) + ".")
                if kept:
                    record["checkpoint"] = candidate_path
        record.update(status=verdict, reason=why, finished=now(), seconds=round(clock() - t0, 1))
        if dpo:
            final = dpo.get("final_validation") or {}
            record["dpo"] = {"steps": dpo.get("step"), "pairs": dpo.get("pairs"), "val_reward_accuracy": _num(final.get("reward_accuracy")),
                             "chosen_len": _num(final.get("chosen_len")), "rejected_len": _num(final.get("rejected_len")), "loss": _num(dpo.get("loss"))}
        # --- apply the verdict and write the evidence before anything else can fail
        fields = {"next_round": number + 1}
        record["audit_flag"] = flagged
        if verdict == KEPT:
            fields.update(current=candidate_path, current_eval=record["after"], current_audit=record["audit"]["after"] or state["current_audit"],
                          kept=state["kept"] + 1, failed_streak=0, audit_flagged=0, current_band=band(candidate_path, docs_gate))
            for pair in pairs:
                append_jsonl(paths(directory)["replay"], pair)
        else:
            fields.update(failed_streak=state["failed_streak"] + 1, audit_flagged=state["audit_flagged"] + 1 if flagged else state["audit_flagged"])
        record["rolled_back_to"] = state["current"] if verdict != KEPT else None
        state["rounds"].append(record)
        append_jsonl(paths(directory)["journal"], record)
        save(phase="", **fields)
        emit({"type": "loop_round", "loop": state["id"], **{k: v for k, v in record.items()}})
        progress("round", number)
        _tidy(directory, state, cfg, number, verdict)
    stopped = stop.is_set() and not reason
    save(status="stopped" if stopped or judge_lost else "finished", stop_reason=reason or ("Stopped." if stopped else ""), phase="")
    return summary(state)


def _tidy(directory, state, cfg, number, verdict):
    """Disk: a rolled-back round's weights are not needed (the journal keeps its evidence); older kept rounds go beyond the limit."""
    round_dir = Path(directory) / f"round_{number:04d}"
    if verdict != KEPT:
        for name in ("dpo",):
            shutil.rmtree(round_dir / name, ignore_errors=True)
        return
    kept = [r for r in state["rounds"] if r.get("status") == KEPT]
    for old in kept[:-cfg["keep_checkpoints"]]:
        folder = Path(directory) / f"round_{old['round']:04d}" / "dpo"
        if Path(old.get("checkpoint") or "").resolve() != Path(state["current"]).resolve():
            shutil.rmtree(folder, ignore_errors=True)
            old["pruned"] = True


# --------------------------------------------------------------------------- the queue job

def run_job(job, context, emit, stop, pause):
    """The `rlhf_loop` job handler's body. params: output_dir, checkpoint, version, config, loop_id; optional rows (tests, remote)."""
    import studio_rlhf
    p = job.params
    rows = p.get("rows") or {kind: context.post_rows(kind) for kind in ("conversations", "problems", "heldout_docs", "heldout_problems")}
    settings = studio_rlhf.load_settings(context.root)
    judge = None
    model = p.get("judge_model") or settings.get("teacher_model")   # the loop's own choice first, the workspace's teacher otherwise
    if model and not p.get("no_judge"):
        judge = studio_rlhf.OllamaTeacher(settings["url"], model, rubric=settings["rubric"],
                                          judge_model=None if p.get("judge_model") else settings.get("judge_model") or None, stop=stop).judge_teacher()
    return run_loop(p["output_dir"], p.get("checkpoint"), rows, judge, p.get("config"), emit, stop, pause, version=p.get("version"),
                    loop_id=p.get("loop_id"), budget_token=p.get("budget_token"))


def adoption(state, rows, measure=None, band=None):
    """The gate result for making a loop's best the version's checkpoint, on ALL the held-back data (both halves, which the loop
    gated and audited on separately): the starting model against the current best, with the gate every round uses. The result has the
    shape of a post-training round's, so WorkspaceServices.record_round can keep it (round 5)."""
    import studio_eval
    import studio_rl
    measure = measure or studio_rl.measure
    band = band or (lambda ckpt, docs: studio_eval.noise_band(ckpt, [_doc(d) for d in docs]) if docs else None)
    docs, problems = rows.get("heldout_docs", []), rows.get("heldout_problems", [])
    if state["current"] == state["anchor"]:
        return {"round": 5, "ok": False, "checks": [], "verdict": "No round was kept yet, so there is nothing to adopt.", "before": None,
                "after": None, "band": None, "checkpoint": state["current"]}
    before, after = measure(state["anchor"], docs, problems), measure(state["current"], docs, problems)
    ok, checks = studio_gate.passes(before, after, band(state["anchor"], docs))
    verdict = (f"The loop's best, after {state['kept']} kept round(s), did not get worse on any held-back data." if ok
               else "The loop's best is worse than the version on the full held-back data, so the version stays as it is.")
    return {"round": 5, "ok": ok, "checks": checks, "verdict": verdict, "before": before, "after": after, "band": band(state["anchor"], docs),
            "checkpoint": state["current"]}
