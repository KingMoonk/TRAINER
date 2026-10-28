"""Auto Run: a plan of training phases and the state machine that walks it. Pure: nothing here talks to the queue, a model or the
network. The orchestrator (#832) asks `next_action`, does what it says, and feeds the result back through `apply_outcome`.

Files, in `data/auto_runs/<id>/`: `plan.json` (what to do), `state.json` (where it is; atomic, rewritten at every transition) and
`journal.jsonl` (one line per transition). Every transition function returns a NEW state and leaves its argument alone; the
events it made are kept in `state["events"]` until `save_state` writes them to the journal and drops them.

Plan: `{id, name, version, budget: {hours}, phases: [{kind, enabled, settings, on_pass, on_fail}]}`.
* kinds, in this fixed order, each at most once: data, pretrain, midtrain, sft, rlvr, distill, dpo (`KINDS`). Each maps to a queue
  job kind in `JOB_KIND` (data is not a job; pretrain, midtrain and sft are the jobs of the same name, midtrain/sft from
  studio_training_stages (#836); rlvr, distill and dpo are `post_round` jobs, the orchestrator picks the round number);
* `on_pass`: continue | pause (pause after the phase passed, so the person can look);
* `on_fail`: pause (wait for the person, who retries or skips) | stop (end the run) | skip (carry on without the phase);
* `budget.hours`: wall-clock hours from `start`, or None for no limit. No phase starts past the deadline, and a pretrain phase's
  `hours` setting is capped to what remains (`phase_settings`). The clock keeps running while the run is paused.

State: `{id, status, current, reason, started, deadline, phases: [...], events}`; times are ISO strings like `studio_core.now()`.
* status: draft | running | pausing (a pause was asked for; the running phase finishes first) | paused | waiting_user (a phase needs
  the person) | stopped | done | failed;
* `current`: the kind of the phase being worked on, or None; `reason`: the sentence the page shows for the status;
* each phase, aligned with the plan's: `{kind, status, job_id, attempt, started, ended, checkpoint, result, error}`, status
  pending | queued | running | passed | failed | skipped | stopped | blocked. A disabled phase starts as skipped.
* job_id is `{run}-{index}-{attempt}` (index = position in the plan, attempt starts at 1), so enqueueing it twice is the same job.

Actions from `next_action(plan, state, now)`; `phase` is the kind:
* ("enqueue", phase, job_id): add that job to the queue (idempotent), with `phase_settings(...)`; then call `begin`;
* ("data", phase): run the data stage in process (it is not a queue job); then call `begin`;
* ("wait",): nothing to do yet (a job is queued or running, or the run is not running);
* ("pause", reason): a pause was asked for and nothing is running; call `settle_pause`;
* ("finish", status, reason): the run is over (done | stopped | failed); call `finish`.

`apply_outcome(plan, state, phase, job_status, result, now)`: job_status is the queue's done | failed | stopped, or "missing" (the job
was deleted by the person: applies on_fail) or "blocked" (the phase needs the person: waiting_user). `result` is a dict that may hold
`checkpoint`, `error`, `reason` and `passed` (False makes a done job a failed phase, e.g. a gate that said no). A stage job's result
(`studio_training_stages.run_stage`: `ok`, `checkpoint`, `verdict`, `stopped`) is accepted as it is: `ok` stands in for `passed`,
`verdict` for the error, and `stopped` True makes the phase stopped.
A stopped job (the person stopped it) leaves the phase stopped and the run paused; `resume` runs it again as the next attempt.

`reconcile(state, queue_lookup, history_lookup)`, after a restart: returns the actions to take for the phases that were queued or
running. A job still waiting/held/running -> ("wait", phase); finished (in the queue or the history) -> ("outcome", phase,
job_status, result), to give to `apply_outcome`; missing from both -> ("enqueue", phase, same job_id). Only this path re-enqueues.
A data phase that was running is run again: ("data", phase). The lookups take a job id and return a dict (or object) with `status`,
`result` and `error`, or None.
"""
from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from studio_core import atomic_json, now as clock

KINDS = ("data", "pretrain", "midtrain", "sft", "rlvr", "distill", "dpo")
JOB_KIND = {"pretrain": "pretrain", "midtrain": "midtrain", "sft": "sft", "rlvr": "post_round", "distill": "post_round",
            "dpo": "post_round"}  # the queue job kind of each phase; data runs in process
ON_PASS = ("continue", "pause")
ON_FAIL = ("pause", "stop", "skip")
RUN_STATUSES = ("draft", "running", "pausing", "paused", "waiting_user", "stopped", "done", "failed")
PHASE_STATUSES = ("pending", "queued", "running", "passed", "failed", "skipped", "stopped", "blocked")
LIVE_JOB = ("waiting", "held", "running")      # the queue's statuses for a job that is not finished
OUTCOMES = ("done", "failed", "stopped", "missing", "blocked")
DEFAULT_HOURS = 12.0


def _autotune():
    import studio_autotune     # (it imports this module)
    return studio_autotune


class AutorunError(ValueError):
    """A plan or a transition that cannot be, with the sentence the page shows."""


# --------------------------------------------------------------------------- the plan

def DEFAULT_PLAN(version):
    """The plan a new Auto Run starts from: every phase on except mid-training, continue on a pass, pause on a failure."""
    return {
        "id": "", "name": "Auto Run", "version": version, "budget": {"hours": DEFAULT_HOURS},
        "phases": [{"kind": kind, "enabled": kind != "midtrain", "settings": {}, "on_pass": "continue", "on_fail": "pause"} for kind in KINDS],
        "controller": _autotune().default_controller(),
    }


def validate(plan):
    """Every problem with `plan`, as plain sentences; an empty list means it can run."""
    if not isinstance(plan, dict):
        return ["The plan must be an object."]
    problems = []
    run_id = plan.get("id")
    if not isinstance(run_id, str) or not run_id.strip():
        problems.append("The plan needs an id.")
    elif any(c in run_id for c in "/\\ ") or run_id in (".", ".."):
        problems.append("The plan id cannot hold slashes or spaces.")
    if not isinstance(plan.get("name"), str) or not plan["name"].strip():
        problems.append("The plan needs a name.")
    if plan.get("version") in (None, ""):
        problems.append("The plan needs a version.")
    budget = plan.get("budget")
    if not isinstance(budget, dict):
        problems.append("The budget must be an object with hours.")
    else:
        hours = budget.get("hours")
        if hours is not None and (isinstance(hours, bool) or not isinstance(hours, (int, float)) or not hours > 0):
            problems.append("The budget hours must be more than 0, or left empty for no limit.")
    phases = plan.get("phases")
    if not isinstance(phases, list) or not phases:
        problems.append("The plan needs at least one phase.")
        return problems
    last = -1
    seen = set()
    for number, phase in enumerate(phases, 1):
        if not isinstance(phase, dict):
            problems.append(f"Phase {number} must be an object.")
            continue
        kind = phase.get("kind")
        label = f"Phase {number}" if kind not in KINDS else f"The {kind} phase"
        if kind not in KINDS:
            problems.append(f"Phase {number} has the unknown kind {kind!r}; use one of {', '.join(KINDS)}.")
        elif kind in seen:
            problems.append(f"The {kind} phase appears more than once.")
        elif KINDS.index(kind) < last:
            problems.append(f"The {kind} phase is out of order; the order is {', '.join(KINDS)}.")
        if kind in KINDS:
            seen.add(kind)
            last = max(last, KINDS.index(kind))
        if not isinstance(phase.get("enabled"), bool):
            problems.append(f"{label} needs enabled set to true or false.")
        if not isinstance(phase.get("settings"), dict):
            problems.append(f"{label} needs its settings as an object.")
        if phase.get("on_pass") not in ON_PASS:
            problems.append(f"{label} has the bad on_pass {phase.get('on_pass')!r}; use {' or '.join(ON_PASS)}.")
        if phase.get("on_fail") not in ON_FAIL:
            problems.append(f"{label} has the bad on_fail {phase.get('on_fail')!r}; use {', '.join(ON_FAIL[:-1])} or {ON_FAIL[-1]}.")
    if not any(isinstance(p, dict) and p.get("enabled") is True for p in phases):
        problems.append("At least one phase must be enabled.")
    if plan.get("controller") is not None:
        problems += _autotune().validate(plan["controller"])
    return problems


def _need_valid(plan):
    problems = validate(plan)
    if problems:
        raise AutorunError(" ".join(problems))


def _index(plan, kind):
    for i, phase in enumerate(plan["phases"]):
        if phase["kind"] == kind:
            return i
    raise AutorunError(f"The plan has no {kind} phase.")


def job_id(plan_or_state, index, attempt):
    return f"{plan_or_state['id']}-{index}-{attempt}"


# --------------------------------------------------------------------------- time

def _time(value):
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, timezone.utc)
    moment = datetime.fromisoformat(value)
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def remaining_hours(state, now):
    """Hours left to the deadline (never below 0), or None when the budget has no limit."""
    if not state.get("deadline"):
        return None
    return max(0.0, (_time(state["deadline"]) - _time(now)).total_seconds() / 3600)


# --------------------------------------------------------------------------- state

def new_state(plan):
    _need_valid(plan)
    phases = [{"kind": p["kind"], "status": "pending" if p["enabled"] else "skipped", "job_id": None, "attempt": 1, "started": None,
               "ended": None, "checkpoint": None, "result": None, "error": ""} for p in plan["phases"]]
    return {"id": plan["id"], "status": "draft", "current": None, "reason": "", "started": None, "deadline": None, "phases": phases,
            "events": []}


def _copy(state):
    return copy.deepcopy(state)


def _event(state, event, now, **extra):
    state.setdefault("events", []).append({"t": _iso(now), "event": event, "status": state["status"], "current": state["current"],
                                           "reason": state["reason"], **extra})


def _iso(now):
    if isinstance(now, str):
        return now
    return (now if isinstance(now, datetime) else _time(now)).isoformat(timespec="seconds")


def _set(state, status, reason, now, event):
    state["status"], state["reason"] = status, reason
    _event(state, event, now)


def _open(phase):
    return phase["status"] in ("pending", "queued", "running", "blocked")


def _first_open(state):
    for phase in state["phases"]:
        if _open(phase):
            return phase
    return None


def _current(state):
    phase = _first_open(state)
    state["current"] = phase["kind"] if phase else None


def start(plan, state, now):
    """draft -> running; sets the start time and the deadline."""
    _need_valid(plan)
    if state["status"] != "draft":
        raise AutorunError("This run was already started.")
    state = _copy(state)
    state["started"] = _iso(now)
    hours = plan["budget"].get("hours")
    state["deadline"] = _iso(_time(now) + timedelta(hours=hours)) if hours else None
    _current(state)
    _set(state, "running", "", now, "start")
    return state


def request_pause(state, now):
    """The person asked to pause: at once when nothing is queued or running, otherwise once that phase has finished."""
    if state["status"] != "running":
        raise AutorunError("Only a running run can be paused.")
    state = _copy(state)
    busy = any(p["status"] in ("queued", "running") for p in state["phases"])
    _set(state, "pausing" if busy else "paused", "Paused by you." if not busy else "Pausing after the current phase.", now,
         "pause_requested" if busy else "paused")
    return state


def settle_pause(state, now):
    """pausing -> paused, once nothing is queued or running any more."""
    if state["status"] != "pausing":
        return state
    state = _copy(state)
    _set(state, "paused", "Paused by you.", now, "paused")
    return state


def resume(plan, state, now):
    """paused / waiting_user -> running. A failed, stopped or blocked phase is tried again as the next attempt (blocked keeps its
    attempt: nothing ran)."""
    if state["status"] not in ("paused", "waiting_user"):
        raise AutorunError("Only a paused run can be resumed.")
    state = _copy(state)
    for phase in state["phases"]:
        if phase["status"] in ("failed", "stopped"):
            phase.update(status="pending", attempt=phase["attempt"] + 1, job_id=None, started=None, ended=None, error="", result=None)
        elif phase["status"] == "blocked":
            phase.update(status="pending", error="")
    _current(state)
    _set(state, "running", "", now, "resume")
    return state


def park(plan, state, phase, now):
    """A pause was asked for and the phase's job is only waiting in the queue (the orchestrator has held it): the phase goes back to
    pending, keeping its job id and attempt, and the run is paused once nothing else is queued or running. `resume` then asks for the
    same job again (the orchestrator releases the held one)."""
    if state["status"] != "pausing":
        raise AutorunError("Only a pausing run can park a phase.")
    state = _copy(state)
    entry = state["phases"][_index(plan, phase)]
    if entry["status"] != "queued":
        raise AutorunError(f"The {phase} phase is {entry['status']}, not queued.")
    entry["status"] = "pending"
    _event(state, "parked", now, phase=phase, job_id=entry["job_id"])
    if not any(p["status"] in ("queued", "running") for p in state["phases"]):
        _set(state, "paused", "Paused by you.", now, "paused")
    return state


def skip_phase(plan, state, phase, now):
    """The person skips a failed, stopped or blocked phase instead of retrying it."""
    state = _copy(state)
    entry = state["phases"][_index(plan, phase)]
    if entry["status"] not in ("failed", "stopped", "blocked", "pending"):
        raise AutorunError(f"The {phase} phase cannot be skipped now.")
    entry.update(status="skipped", ended=_iso(now))
    _current(state)
    _event(state, "skip", now, phase=phase)
    return state


def stop(state, reason, now):
    """The person ends the run: queued and running phases become stopped, the others stay as they were."""
    if state["status"] in ("stopped", "done", "failed"):
        return state
    state = _copy(state)
    for phase in state["phases"]:
        if phase["status"] in ("queued", "running"):
            phase.update(status="stopped", ended=_iso(now))
    _current(state)
    _set(state, "stopped", reason or "Stopped by you.", now, "stop")
    return state


def finish(state, status, reason, now):
    """Apply the ("finish", status, reason) action."""
    if status not in ("done", "stopped", "failed"):
        raise AutorunError(f"A run cannot finish as {status}.")
    state = _copy(state)
    _set(state, status, reason, now, "finish")
    return state


# --------------------------------------------------------------------------- what to do next

def phase_settings(plan, state, phase, now):
    """The settings the phase runs with: the plan's, with a pretrain phase's hours capped to the budget that remains."""
    entry = plan["phases"][_index(plan, phase)]
    settings = copy.deepcopy(entry["settings"])
    left = remaining_hours(state, now)
    if phase == "pretrain" and left is not None:
        asked = settings.get("hours")
        settings["hours"] = round(min(asked, left) if isinstance(asked, (int, float)) and asked > 0 else left, 4)
    return settings


def next_action(plan, state, now):
    status = state["status"]
    if status in ("stopped", "done", "failed"):
        return ("finish", status, state["reason"])
    if status in ("draft", "paused", "waiting_user"):
        return ("wait",)
    busy = any(p["status"] in ("queued", "running") for p in state["phases"])
    if busy:
        return ("wait",)
    if status == "pausing":
        return ("pause", state["reason"] or "Paused by you.")
    phase = _first_open(state)
    if phase is None:
        passed = [p for p in state["phases"] if p["status"] == "passed"]
        if passed:
            return ("finish", "done", "Every phase has finished.")
        return ("finish", "done", "Every phase was skipped.")
    if phase["status"] == "blocked":
        return ("wait",)
    left = remaining_hours(state, now)
    if left is not None and left <= 0:
        return ("finish", "stopped", "The time budget is spent.")
    index = _index(plan, phase["kind"])
    if phase["kind"] == "data":
        return ("data", "data")
    return ("enqueue", phase["kind"], job_id(state, index, phase["attempt"]))


def begin(plan, state, action, now):
    """Record that the orchestrator acted on an ("enqueue", ...) or ("data", ...) action: the phase is queued (or running)."""
    if action[0] not in ("enqueue", "data"):
        raise AutorunError("Only an enqueue or data action can begin.")
    state = _copy(state)
    index = _index(plan, action[1])
    entry = state["phases"][index]
    if entry["status"] != "pending" and not (entry["status"] in ("queued", "running") and entry["started"]):
        raise AutorunError(f"The {action[1]} phase is {entry['status']}, not pending.")
    if entry["status"] == "pending":
        entry.update(started=_iso(now), ended=None)
    if action[0] == "enqueue":
        entry.update(status="queued", job_id=action[2])
    else:
        entry.update(status="running", job_id=None)
    state["current"] = action[1]
    _event(state, "begin", now, phase=action[1], job_id=entry["job_id"])
    return state


def mark_running(plan, state, phase, now):
    """The queue started the phase's job: queued -> running."""
    state = _copy(state)
    entry = state["phases"][_index(plan, phase)]
    if entry["status"] == "queued":
        entry["status"] = "running"
        _event(state, "running", now, phase=phase)
    return state


# --------------------------------------------------------------------------- the outcome of a phase

def apply_outcome(plan, state, phase, job_status, result, now):
    if job_status not in OUTCOMES:
        raise AutorunError(f"Unknown job outcome {job_status!r}; use {', '.join(OUTCOMES)}.")
    index = _index(plan, phase)
    if state["phases"][index]["status"] not in ("queued", "running", "pending"):
        return state  # a late report for a phase that already ended: nothing to change
    result = result if isinstance(result, dict) else {}
    state = _copy(state)
    entry, policy = state["phases"][index], plan["phases"][index]
    entry["ended"] = _iso(now)
    if result.get("checkpoint"):
        entry["checkpoint"] = result["checkpoint"]
    entry["result"] = result or None
    pausing = state["status"] == "pausing"
    error = str(result.get("error") or "")
    passed = result.get("passed", result.get("ok"))
    if passed is False:
        error = error or str(result.get("verdict") or "")
    if job_status == "done" and result.get("stopped"):
        job_status = "stopped"
    elif job_status == "done" and passed is False:
        job_status, error = "failed", error or f"The {phase} phase did not pass its check."
    if job_status == "done":
        entry.update(status="passed", error="")
        _current(state)
        _event(state, "passed", now, phase=phase)
        if policy["on_pass"] == "pause":
            _set(state, "paused", f"The {phase} phase passed; paused as planned.", now, "paused")
        elif pausing:
            _set(state, "paused", "Paused by you.", now, "paused")
    elif job_status == "blocked":
        entry.update(status="blocked", error=str(result.get("reason") or error))
        state["current"] = phase
        _set(state, "waiting_user", entry["error"] or f"The {phase} phase needs you.", now, "blocked")
    elif job_status == "stopped":
        entry.update(status="stopped", error=error)
        state["current"] = phase
        _set(state, "paused", f"The {phase} job was stopped.", now, "job_stopped")
    else:  # failed, or missing: the person deleted the job
        error = error or ("The job was removed from the queue." if job_status == "missing" else f"The {phase} job failed.")
        if policy["on_fail"] == "skip":
            entry.update(status="skipped", error=error)
            _current(state)
            _event(state, "failed_skipped", now, phase=phase)
            if pausing:
                _set(state, "paused", "Paused by you.", now, "paused")
        elif policy["on_fail"] == "stop":
            entry.update(status="failed", error=error)
            state["current"] = phase
            _set(state, "stopped", f"The {phase} phase failed: {error}", now, "failed_stop")
        else:
            entry.update(status="failed", error=error)
            state["current"] = phase
            _set(state, "paused", f"The {phase} phase failed: {error}", now, "failed_pause")
    if state["status"] in ("running", "pausing") and _first_open(state) is None:
        _current(state)
        _set(state, "done", "Every phase has finished.", now, "done")
    return state


# --------------------------------------------------------------------------- after a restart

def _field(job, name, default=None):
    return job.get(name, default) if isinstance(job, dict) else getattr(job, name, default)


def reconcile(state, queue_lookup, history_lookup):
    """What to do about each phase that was queued or running when the app went away (see the module docstring). Pure: it reads
    the lookups and returns actions; the state is unchanged."""
    if state["status"] not in ("running", "pausing"):
        return []
    actions = []
    for entry in state["phases"]:
        if entry["status"] not in ("queued", "running"):
            continue
        kind, jid = entry["kind"], entry["job_id"]
        if not jid:
            actions.append(("data", kind))
            continue
        job = queue_lookup(jid)
        if job is not None and _field(job, "status") in LIVE_JOB:
            actions.append(("wait", kind))
            continue
        if job is None:
            job = history_lookup(jid)
        if job is None:
            actions.append(("enqueue", kind, jid))
            continue
        status = _field(job, "status")
        if status in LIVE_JOB:
            actions.append(("wait", kind))
            continue
        result = dict(_field(job, "result") or {})
        if _field(job, "error"):
            result.setdefault("error", _field(job, "error"))
        actions.append(("outcome", kind, status if status in ("done", "failed", "stopped") else "failed", result))
    return actions


# --------------------------------------------------------------------------- the files

def paths(directory):
    d = Path(directory)
    return {"plan": d / "plan.json", "state": d / "state.json", "journal": d / "journal.jsonl"}


def _load(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def load_plan(directory):
    return _load(paths(directory)["plan"])


def save_plan(directory, plan):
    _need_valid(plan)
    atomic_json(paths(directory)["plan"], plan)


def load_state(directory):
    return _load(paths(directory)["state"])


def save_state(directory, state):
    """Write state.json atomically and move the state's pending events into journal.jsonl, one line each."""
    events = state.pop("events", [])
    state["updated"] = clock()
    try:
        atomic_json(paths(directory)["state"], {**state, "events": []})
        if events:
            journal = paths(directory)["journal"]
            with journal.open("a", encoding="utf-8") as stream:
                for event in events:
                    stream.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
    finally:
        state["events"] = []
    return state


def read_journal(directory):
    try:
        lines = paths(directory)["journal"].read_text(encoding="utf-8").split("\n")
    except OSError:
        return []
    rows = []
    for line in lines:
        if line.strip():
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass  # a line cut by a crash
    return rows
