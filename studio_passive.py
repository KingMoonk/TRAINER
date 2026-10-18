"""Passive training: a detached trainer that repeats train -> save -> learn from the teacher -> go again or step back.

Layout under the workspace root:
  data/passive.sqlite       one row per session: plan, state, command, heartbeat (the only channel between app, trainer, watchdog)
  runs/<id>/journal.jsonl   append-only events; eval.jsonl (fixed held-out set); snapshot.jsonl (training mixture);
                            lessons.jsonl; one run folder per training round
Run it with `python -m studio_passive run <id> --root <root>`. One trainer per session: it holds runs/<id>/trainer.lock while it runs.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import shutil
import signal
import sqlite3
import sys
import tempfile
import threading
import time
from contextlib import closing
from pathlib import Path

from studio_core import ROOT, fingerprint, module_command, now, replace_file, uid

ACTIVE = ("queued", "running", "paused")
UNITS = {"minutes": 60, "hours": 3600, "days": 86400}
PIVOTS = ("halve_lr", "next_dataset", "drop_lessons")
DEFAULT_PLAN = {
    "stop": {"duration": 8, "unit": "hours", "patience_rounds": 3, "target_error": None, "win_rate": None},
    "limits": {"memory_fraction": 0.5, "keep_checkpoints": 10, "keep_awake": True, "pause_on_battery": True},
    "plateau": {"min_improvement": 0.01, "checks": 5, "max_steps": 2000},
    "teacher": {"polish": True, "lessons": True},
    "worse": {"limit": 2, "pivots": list(PIVOTS)},
    "data": {"sources": [], "max_records": 20000, "use_all": False, "balance": "tokens", "reviewed_only": False, "weights": {}},
    "config": {},
    "base_checkpoint": None,
}
BEAT, POLL, PAUSED_POLL = 5.0, 1.0, 3.0  # seconds between heartbeats / between command checks while a round is training / while paused
MAX_TICK = 10.0  # a longer gap between two clock ticks is a sleeping computer, not training


# --------------------------------------------------------------------------- plan

def _number(value, label, low=None, high=None, integer=False, allow_none=False):
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or (integer and int(value) != value):
        raise ValueError(f"{label} must be {'a whole number' if integer else 'a number'}.")
    if (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(f"{label} must be {'at least ' + str(low) if high is None else 'between ' + str(low) + ' and ' + str(high)}.")
    return int(value) if integer else value


def merge_plan(overrides):
    """Deep-merge `overrides` over DEFAULT_PLAN and validate it. Raises ValueError with a plain-language message."""
    overrides = overrides or {}
    plan = copy.deepcopy(DEFAULT_PLAN)
    for key, value in overrides.items():
        if key not in plan:
            raise ValueError(f"Unknown setting: {key}.")
        if isinstance(plan[key], dict) and key != "config":
            if not isinstance(value, dict):
                raise ValueError(f"{key} must be a group of settings.")
            for sub in value:
                if sub not in plan[key]:
                    raise ValueError(f"Unknown setting: {key}.{sub}.")
            plan[key].update(copy.deepcopy(value))
        else:
            plan[key] = copy.deepcopy(value)
    stop, limits, plateau, worse, data = plan["stop"], plan["limits"], plan["plateau"], plan["worse"], plan["data"]
    if stop["unit"] not in UNITS:
        raise ValueError("The time limit must be in minutes, hours or days.")
    _number(stop["duration"], "The time limit", 0.0000001)
    _number(stop["patience_rounds"], "Rounds without a better checkpoint", 0, integer=True)
    _number(stop["target_error"], "The target test error", 0, allow_none=True)
    _number(stop["win_rate"], "The head-to-head share", 0.0000001, 1, allow_none=True)
    _number(limits["memory_fraction"], "The memory limit", 0.0000001, 1)
    _number(limits["keep_checkpoints"], "Checkpoints to keep", 1, integer=True)
    _number(plateau["min_improvement"], "The improvement threshold", 0.0000001)
    _number(plateau["checks"], "The number of checks", 2, integer=True)
    _number(plateau["max_steps"], "The step limit", 1, integer=True)
    _number(worse["limit"], "Checkpoints in a row that are worse", 1, integer=True)
    _number(data["max_records"], "The record limit", 1, integer=True)
    if not isinstance(worse["pivots"], list) or any(p not in PIVOTS for p in worse["pivots"]):
        raise ValueError("Each change to try when it gets worse must be one of: " + ", ".join(PIVOTS) + ".")
    if data["balance"] not in ("records", "tokens"):
        raise ValueError("Balance must be records or tokens.")
    if not isinstance(data["sources"], list) or not isinstance(plan["config"], dict):
        raise ValueError("Data sources must be a list and the training setup a group of settings.")
    if plateau["min_improvement"] >= 1:
        raise ValueError("The improvement threshold must be below 100%.")
    for name, value in (("Keep the computer awake", limits["keep_awake"]), ("Pause on battery power", limits["pause_on_battery"]), ("Polish the checkpoint", plan["teacher"]["polish"]),
                        ("Carry the lessons forward", plan["teacher"]["lessons"]), ("Use all records", data["use_all"]), ("Reviewed records only", data["reviewed_only"])):
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be on or off.")
    if plan["base_checkpoint"] is not None and not Path(str(plan["base_checkpoint"])).is_file():
        raise ValueError("The checkpoint to start from no longer exists.")
    from studio_model import validate_config
    try:
        validate_config(plan["config"])
    except (TypeError, KeyError, AttributeError) as exc:
        raise ValueError(f"The training setup is not valid: {exc}") from exc
    return plan


def patience_hides_pivots(plan):
    """True when the patience stop always comes first, so the changes to try when it gets worse never run."""
    return bool(plan["stop"]["patience_rounds"] and plan["worse"]["pivots"] and plan["stop"]["patience_rounds"] <= plan["worse"]["limit"])


def duration_seconds(plan):
    return float(plan["stop"]["duration"]) * UNITS[plan["stop"]["unit"]]


def _plural(n, word):
    return f"{n} {word}" + ("" if n == 1 else "s")


def plan_summary(plan):
    """One plain line per drill-down row."""
    stop, limits, plateau, teacher, worse = plan["stop"], plan["limits"], plan["plateau"], plan["teacher"], plan["worse"]
    duration, unit = stop["duration"], stop["unit"]
    parts = [f"{duration:g} {unit[:-1] if duration == 1 else unit}"]
    if stop["patience_rounds"]:
        parts.append(f"{_plural(stop['patience_rounds'], 'round')} without a better checkpoint")
    if stop["target_error"] is not None:
        parts.append(f"test error reaches {stop['target_error']:g}")
    if stop["win_rate"] is not None:
        parts.append(f"it wins {stop['win_rate'] * 100:.0f}% of head-to-head replies against the starting model")
    names = {"halve_lr": "halve the learning rate", "next_dataset": "switch to the next dataset", "drop_lessons": "leave out the newest lessons"}
    steps = [names[p] for p in worse["pivots"]]
    linux, off = sys.platform.startswith("linux"), " (not supported on this platform)"
    awake_ok = sys.platform in ("darwin", "win32") or (linux and shutil.which("systemd-inhibit"))
    battery_ok = sys.platform in ("darwin", "win32") or linux
    hidden = (f" Note: it stops after {_plural(stop['patience_rounds'], 'round')} without a better checkpoint, before it ever steps back, so these changes are never tried."
              f" Raise that above {worse['limit']} to use them.") if patience_hides_pivots(plan) else ""
    return {"stop": "Stops after " + ", or after ".join(parts) + ".",
            "limits": f"Up to {limits['memory_fraction'] * 100:.0f}% of memory (best effort), keeps {limits['keep_checkpoints']} checkpoints"
                      + (", keeps the computer awake" + ("" if awake_ok else off) if limits["keep_awake"] else "")
                      + (", pauses on battery" + ("" if battery_ok else off) if limits["pause_on_battery"] else "") + ".",
            "plateau": f"Saves a checkpoint when test error improves by less than {plateau['min_improvement'] * 100:g}% across {plateau['checks']} checks, or after {plateau['max_steps']:,} steps.",
            "teacher": ("Polishes each checkpoint" if teacher["polish"] else "No polishing")
                       + (" and carries the teacher's better answers forward" if teacher["lessons"] else "") + " (skipped without a teacher).",
            "worse": f"After {_plural(worse['limit'], 'checkpoint')} in a row worse than the best: go back to the best, then " + (", then ".join(steps) if steps else "stop") + ". When the list runs out it stops." + hidden}


def progress_summary(state, plan):
    """The trainer's state as the detail view shows it: phase, round, best and current error, the checkpoints with their errors, the
    learning rate and batch size, the time used against the limit. Missing keys (a session that has not set up yet) are None or empty."""
    checkpoints = [{"n": c["n"], "round": c.get("round", c["n"]), "error": c["error"], "elapsed": c.get("elapsed"), "at": c.get("at"), "parent": c.get("parent"),
                    "kept": c.get("kept", True), "discarded": c.get("discarded", False), "pivot": c.get("pivot"),
                    "polished": bool((c.get("teacher") or {}).get("polished")), "win_rate": (c.get("teacher") or {}).get("win_rate"),
                    "lessons": (c.get("teacher") or {}).get("lessons", 0), "best": c["n"] == state.get("best")} for c in state.get("checkpoints", [])]
    best, current = _checkpoint(state, state.get("best")) if state.get("checkpoints") else None, _checkpoint(state, state.get("current")) if state.get("checkpoints") else None
    return {"phase": state.get("phase") or "", "round": state.get("round", 0), "best": state.get("best"), "best_error": best and best["error"],
            "current": state.get("current"), "current_error": current and current["error"], "learning_rate": state.get("learning_rate"),
            "batch_size": state.get("batch_size"), "win_rate": state.get("win_rate"), "stop_reason": state.get("stop_reason"),
            "active_seconds": state.get("active_seconds", 0.0), "limit_seconds": duration_seconds(plan), "worse_streak": state.get("worse_streak", 0),
            "rounds_since_best": state.get("rounds_since_best", 0), "restarts": state.get("restart_count", 0), "teacher": state.get("teacher") or None,
            "checkpoints": checkpoints}


# --------------------------------------------------------------------------- pure decisions

def plateaued(losses, plan):
    """True when the last `checks` test errors improved by less than min_improvement (as a share of the first of them)."""
    checks = plan["plateau"]["checks"]
    if len(losses) < checks:
        return False
    window = losses[-checks:]
    return (window[0] - min(window)) / max(abs(window[0]), 1e-12) < plan["plateau"]["min_improvement"]


def _checkpoint(state, n):
    return next((c for c in state["checkpoints"] if c["n"] == n), None)


def _next_pivot(state, plan):
    """(index, name) of the next usable change in the plan's list, or None. next_dataset needs two or more sources."""
    for i in range(state.get("pivot_index", 0), len(plan["worse"]["pivots"])):
        name = plan["worse"]["pivots"][i]
        if name != "next_dataset" or len(plan["data"]["sources"]) >= 2:
            return i, name
    return None


def stop_reason(state, plan, now):
    """Why the session should end, or None. `now` is the clock; time since the last tick counts as active unless paused."""
    active = state.get("active_seconds", 0.0) + (max(0.0, now - state["tick"]) if state.get("tick") is not None and not state.get("paused") else 0.0)
    if active >= duration_seconds(plan):
        return "duration"
    stop = plan["stop"]
    if stop["patience_rounds"] and state.get("rounds_since_best", 0) >= stop["patience_rounds"]:
        return "patience"
    best = _checkpoint(state, state.get("best"))
    if stop["target_error"] is not None and best and best["error"] <= stop["target_error"]:
        return "target"
    if stop["win_rate"] is not None and state.get("win_rate") is not None and state["win_rate"] >= stop["win_rate"]:
        return "win_rate"
    if state.get("worse_streak", 0) >= plan["worse"]["limit"] and _next_pivot(state, plan) is None:
        return "pivots"
    return None


def next_move(state, plan):
    if state.get("worse_streak", 0) < plan["worse"]["limit"]:
        return {"action": "continue"}
    pivot = _next_pivot(state, plan)
    if pivot is None:
        return {"action": "stop", "reason": "pivots"}
    return {"action": "rollback", "pivot": pivot[1], "index": pivot[0]}


# --------------------------------------------------------------------------- the record

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY, status TEXT NOT NULL, command TEXT NOT NULL DEFAULT '', plan TEXT NOT NULL, state TEXT NOT NULL,
  trainer_pid INTEGER, watchdog_pid INTEGER, heartbeat REAL, restarts TEXT NOT NULL DEFAULT '[]', error TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL)
"""
COLUMNS = ("status", "command", "plan", "state", "trainer_pid", "watchdog_pid", "heartbeat", "restarts", "error")
COMMANDS = ("pause", "resume", "stop")


class AlreadyActive(RuntimeError, ValueError):
    """Another session is queued, running or paused. A RuntimeError first, so the API answers 409; still a ValueError for older callers."""


# Switching to WAL takes an exclusive lock that the busy timeout does not wait for, so two threads opening the
# store at once could fail with "database is locked". Opening is serialised, and WAL is only set when it is not yet on.
_OPEN_LOCK = threading.Lock()


class PassiveStore:
    def __init__(self, root):
        self.root = Path(root)
        self.path = self.root / "data" / "passive.sqlite"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _OPEN_LOCK, self.connect() as db:
            if db.execute("PRAGMA journal_mode").fetchone()[0] != "wal":
                db.execute("PRAGMA journal_mode=WAL")
            db.execute(SCHEMA)

    def connect(self):
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        return closing(db)

    @staticmethod
    def decode(row):
        if row is None:
            return None
        out = dict(row)
        for key in ("plan", "state", "restarts"):
            out[key] = json.loads(out[key])
        return out

    def create(self, plan):
        plan = merge_plan(plan)
        session_id, stamp = uid("passive"), now()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                if db.execute(f"SELECT 1 FROM sessions WHERE status IN ({','.join('?' * len(ACTIVE))})", ACTIVE).fetchone():
                    raise AlreadyActive("A passive training session is already running. Stop it before starting another.")
                db.execute("INSERT INTO sessions(id, status, plan, state, created_at, updated_at) VALUES (?, 'queued', ?, '{}', ?, ?)",
                           (session_id, json.dumps(plan), stamp, stamp))
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise
        self.directory(session_id)
        return session_id

    def requeue(self, session_id):
        """Put a failed or stopped session back in the queue so a watchdog starts its trainer again, which carries on from the last
        checkpoint it saved (the state keeps `started`). Forgets the old process ids, the restart count and the watchdog's memory of
        fruitless restarts, or a session that gave up would give up again at once. Returns the status it had; None for an unknown
        session; raises RuntimeError for one that is not failed or stopped, and AlreadyActive while another session is active."""
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                row = self.decode(db.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone())
                if row is None:
                    db.execute("ROLLBACK")
                    return None
                if row["status"] not in ("failed", "stopped"):
                    raise RuntimeError(f"Passive session {session_id} is {row['status']}, so it cannot be resumed.")
                if db.execute(f"SELECT 1 FROM sessions WHERE status IN ({','.join('?' * len(ACTIVE))})", ACTIVE).fetchone():
                    raise AlreadyActive("A passive training session is already running. Stop it before resuming another.")
                state = {**row["state"], "stop_reason": None, "paused": False, "phase": "idle"} if row["state"] else {}
                db.execute("UPDATE sessions SET status = 'queued', command = '', error = '', trainer_pid = NULL, watchdog_pid = NULL, heartbeat = NULL,"
                           " restarts = '[]', state = ?, updated_at = ? WHERE id = ?", (json.dumps(state), now(), session_id))
                db.execute("COMMIT")
            except BaseException:
                try:
                    db.execute("ROLLBACK")
                except sqlite3.OperationalError:
                    pass  # already rolled back above
                raise
        (self.directory(session_id) / "watchdog.json").unlink(missing_ok=True)
        self.append(session_id, {"type": "resume", "reason": "requeued", "from": row["status"]})
        return row["status"]

    def _one(self, sql, args=()):
        with self.connect() as db:
            return self.decode(db.execute(sql, args).fetchone())

    def get(self, session_id):
        return self._one("SELECT * FROM sessions WHERE id = ?", (session_id,))

    def active(self):
        return self._one(f"SELECT * FROM sessions WHERE status IN ({','.join('?' * len(ACTIVE))}) ORDER BY created_at DESC, rowid DESC", ACTIVE)

    def latest(self):
        return self._one("SELECT * FROM sessions ORDER BY created_at DESC, rowid DESC")

    def list(self):
        with self.connect() as db:
            return [self.decode(r) for r in db.execute("SELECT * FROM sessions ORDER BY created_at DESC, rowid DESC")]

    def update(self, session_id, **columns):
        unknown = set(columns) - set(COLUMNS)
        if unknown:
            raise ValueError("Unknown column: " + ", ".join(sorted(unknown)))
        values = [json.dumps(v) if isinstance(v, (dict, list)) else v for v in columns.values()]
        with self.connect() as db:
            db.execute(f"UPDATE sessions SET {''.join(k + ' = ?, ' for k in columns)}updated_at = ? WHERE id = ?", (*values, now(), session_id))

    def command(self, session_id, command, expect=("", "pause", "resume")):
        """Leave `command` for the trainer. Only replaces a pending command listed in `expect`; nothing but a stop replaces a stop. True when written."""
        if command not in COMMANDS:
            raise ValueError("Command must be pause, resume or stop.")
        sql, args = "UPDATE sessions SET command = ?, updated_at = ? WHERE id = ?", [command, now(), session_id]
        if command != "stop":
            sql += f" AND command IN ({','.join('?' * len(expect))})"
            args += list(expect)
        with self.connect() as db:
            return db.execute(sql, args).rowcount > 0

    def take_command(self, session_id, command):
        """Clear `command` only if it is still the pending one: a stop written in between is never lost. True when taken."""
        with self.connect() as db:
            return db.execute("UPDATE sessions SET command = '', updated_at = ? WHERE id = ? AND command = ?", (now(), session_id, command)).rowcount > 0

    def beat(self, session_id, pid):
        """Heartbeat from trainer `pid`; False (and nothing written) when the record names another trainer."""
        with self.connect() as db:
            return db.execute("UPDATE sessions SET heartbeat = ?, trainer_pid = ?, updated_at = ? WHERE id = ? AND (trainer_pid IS NULL OR trainer_pid = ?)",
                              (time.time(), pid, now(), session_id, pid)).rowcount > 0

    def directory(self, session_id):
        path = self.root / "runs" / session_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def journal(self, session_id):
        return _read_jsonl(self.directory(session_id) / "journal.jsonl")

    def journal_page(self, session_id, limit=50, before=None):
        """(events, total, first): up to `limit` journal events that come before event number `before` (the newest ones when None), oldest
        first, each with its `index`; `first` is the index of the first one returned, which is the next page's `before`."""
        rows = self.journal(session_id)
        end = len(rows) if before is None else max(0, min(before, len(rows)))
        start = max(0, end - max(1, limit))
        return [{**rows[i], "index": i} for i in range(start, end)], len(rows), start

    def append(self, session_id, event):
        with (self.directory(session_id) / "journal.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({**event, "at": now()}, ensure_ascii=False) + "\n")

    def lessons(self, session_id):
        return _read_jsonl(self.directory(session_id) / "lessons.jsonl")

    def set_lessons(self, session_id, rows):
        # ponytail: read-modify-write; the app and the trainer rarely touch it at the same moment. Add a lock file if that changes.
        _write_jsonl(self.directory(session_id) / "lessons.jsonl", rows)


def _read_jsonl(path):
    if not Path(path).exists():
        return []
    rows = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass  # a half-written last line after a crash
    return rows


def _write_jsonl(path, rows):
    path = Path(path)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        replace_file(name, path)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise


def take_lock(path):
    """An exclusive OS lock on `path`, held while the returned file stays open; None when another process holds it."""
    handle = open(path, "a+b")
    try:
        if sys.platform == "win32":
            import msvcrt
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    return handle


# --------------------------------------------------------------------------- OS probes (patched in tests)

def process_memory():
    """Bytes this process uses now. Falls back to the peak only where the current figure cannot be read."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [(n, ctypes.c_size_t) for n in (
                "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
        return int(counters.WorkingSetSize)
    if sys.platform == "darwin":
        try:
            import ctypes
            info = (ctypes.c_uint64 * 12)()  # rusage_info_v0: 16-byte uuid, user and system time, two wakeup counts, pageins, wired size, resident size, ...
            if ctypes.CDLL("/usr/lib/libproc.dylib").proc_pid_rusage(os.getpid(), 0, info) == 0:
                return int(info[8])
        except (OSError, AttributeError):
            pass
    try:
        return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak if sys.platform == "darwin" else peak * 1024  # bytes on macOS, kilobytes on Linux


def total_memory():
    if sys.platform == "win32":
        import ctypes

        class Status(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [(n, ctypes.c_ulonglong) for n in (
                "ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile", "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual")]
        status = Status()
        status.dwLength = ctypes.sizeof(status)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return int(status.ullTotalPhys)
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")


# --------------------------------------------------------------------------- real work (replaced by fakes in tests)

def build_examples(root, data):
    from studio_core import Workspace
    from studio_planning import build_training_set
    return build_training_set(Workspace(root), data["sources"], data["max_records"], 42, data["reviewed_only"], data["weights"] or None,
                              balance=data["balance"], use_all=data["use_all"])["examples"]


def default_evaluate(path, examples):
    from studio_rlhf import evaluate_checkpoint
    return evaluate_checkpoint(path, examples, limit=len(examples))["loss"]


def default_train(config, examples, output_dir, emit, stop, pause, init_from=None, resume=None):
    """Train one round. `resume` is the last round's checkpoint: the round continues it with the optimizer state and the step count
    (config["steps"] is how many steps to add), on the round's own data. A checkpoint training refuses before its first step (a polished
    one has no optimizer state, say) emits "resume_refused" and the round starts from its weights alone, as `init_from` does."""
    import studio_memory
    import studio_model
    options = dict(offload_optimizer=studio_memory.offload_wanted(studio_model.validate_config(config), 1.0))  # CUDA, and only when the state cannot fit
    if not resume:
        return studio_model.train(config, examples, output_dir, emit, stop, pause, init_from=init_from, **options)
    started = []

    def relay(event):
        started.append(True)
        emit(event)
    try:
        return studio_model.train(config, examples, output_dir, relay, stop, pause, resume=resume, rebase=True, **options)
    except (OSError, ValueError, KeyError) as exc:
        if started:
            raise
        emit({"type": "resume_refused", "reason": str(exc) or type(exc).__name__})
        return studio_model.train(config, examples, output_dir, emit, stop, pause, init_from=resume, **options)


def _teacher(settings):
    from studio_rlhf import OllamaTeacher
    return OllamaTeacher(settings["url"], settings["teacher_model"], rubric=settings["rubric"], judge_model=settings.get("judge_model") or None)


def default_teacher_round(directory, plan, settings, checkpoint, examples, stop, emit):
    """One preference round from `checkpoint`. Returns {"checkpoint", "win_rate", "lessons": [{"messages", "chosen", "rejected"}]}.

    Raises if the teacher cannot be reached; the loop skips the step for that round.
    """
    from studio_rlhf import PreferenceStore, auto_rlhf, prompts_from
    teacher = _teacher(settings)
    teacher.models()  # RuntimeError when Ollama is not reachable
    store = PreferenceStore(directory)
    before = len(store)
    result = auto_rlhf(str(checkpoint), prompts_from(examples[:500]), teacher, store, directory, {"rounds": 1},
                       {"device": plan["config"].get("device", "auto")}, emit, stop, None, teacher.judge_teacher())
    if result.get("aborted"):
        raise RuntimeError(result["aborted"])
    rows = [r for r in store.all()[before:] if r.get("kind") == "rewrite"]
    return {"checkpoint": result["checkpoint"], "win_rate": result["win_rate"], "lessons": [dict(r) for r in rows]}


def default_versus(settings, new, old, examples, stop):
    """Share of head-to-head replies where `new` beats `old`, as graded by the teacher."""
    from studio_rlhf import compare_checkpoints, prompts_from
    return compare_checkpoints(new, old, prompts_from(examples)[:12], _teacher(settings).judge_teacher(), stop=stop)["win_rate"]


# --------------------------------------------------------------------------- the loop

class _Diverged(Exception):
    """A training round whose error stopped being a number."""


class _Trainer:
    def __init__(self, root, session_id, train, teacher_round, evaluate, build, versus, clock, interrupt, awake=None):
        self.root, self.id, self.clock = Path(root), session_id, clock
        self.train, self.teacher_round, self.evaluate, self.build, self.versus = train, teacher_round, evaluate, build, versus
        self.store = PassiveStore(root)
        self.directory = self.store.directory(session_id)
        row = self.store.get(session_id)
        self.plan, self.state, self.status = row["plan"], row["state"], row["status"]
        self.interrupt = interrupt
        self.lock = threading.RLock()
        self.pause, self.stop, self.cause, self.user = threading.Event(), threading.Event(), None, False
        self.done, self.ready = threading.Event(), threading.Event()  # ready: setup is over and the state may be ticked and polled
        self.settings = {}
        self.awake_hooks, self.awake = awake, None  # (keep, release) from studio_passive_watch, and the keep-awake handle while held
        self.leftover = None  # a checkpoint the last process saved for the next round but never evaluated

    # -- bookkeeping
    def journal(self, type_, **fields):
        self.store.append(self.id, {"type": type_, **fields})

    def save(self, **columns):
        self.store.update(self.id, state=self.state, **columns)

    def tick(self):
        with self.lock:
            t = self.clock()
            if self.state.get("tick") is not None and not self.pause.is_set() and 0.0 < t - self.state["tick"] <= MAX_TICK:
                self.state["active_seconds"] += t - self.state["tick"]
            self.state["tick"] = t
            self.state["paused"] = self.pause.is_set()

    def poll(self):
        """Take the command the app left in the record, if any."""
        with self.lock:
            command = self.store.get(self.id)["command"]
            if not command:
                return
            if command == "stop":  # stays in the record until run() writes the stopped status: if this process dies first, the restarted trainer still sees it
                self.user = True
                self.pause.clear()
                self.stop.set()
                return
            if not self.store.take_command(self.id, command):
                return  # it changed since it was read (a stop, usually): the next poll takes the new one
            if command == "pause" and not self.user:
                self.pause.set()
                self.store.update(self.id, status="paused")
                self.journal("pause")
            elif command == "resume":
                self.pause.clear()
                self.store.update(self.id, status="running")
                self.journal("resume")

    def guard(self):
        """Called often while training: clock, commands, time limit, memory."""
        self.tick()
        self.poll()
        if self.interrupt.is_set() or stop_reason(self.state, self.plan, self.clock()) == "duration":
            self.stop.set()
        used, limit = process_memory(), int(total_memory() * self.plan["limits"]["memory_fraction"])
        self.state["memory"] = {"used": used, "limit": limit}
        if used > limit and self.state["batch_size"] > 1 and not self.cause:
            self.cause = "memory"
            self.stop.set()

    def pulse(self):
        # ponytail: the heartbeat proves the process is alive, not that it is progressing; the watchdog only checks liveness.
        last = 0.0
        while not self.done.wait(PAUSED_POLL if self.pause.is_set() else POLL):
            try:
                if self.interrupt.is_set():
                    self.stop.set()  # a SIGTERM while train() sits in its own pause loop: only the round's stop ends that loop
                if self.ready.is_set():
                    self.tick()
                    self.poll()
                self.stay_awake(True)
                if time.monotonic() - last >= BEAT:
                    self.store.beat(self.id, os.getpid())
                    last = time.monotonic()
            except Exception:
                pass  # a locked record, or any other hiccup, must not end the heartbeat: the watchdog would kill a healthy trainer
        self.stay_awake(False)

    def stay_awake(self, want):
        """Hold the keep-awake request while running, drop it while paused. Only the pulse thread calls this: on Windows the request belongs to a thread."""
        if not self.awake_hooks:
            return
        keep, release = self.awake_hooks
        want = want and not self.pause.is_set()
        if want and self.awake is None:
            self.awake = keep(os.getpid()) or False  # False: asked and refused; asked again after the next resume
        elif not want and self.awake is not None:
            release(self.awake)
            self.awake = None

    def wait_if_paused(self):
        while self.pause.is_set() and not self.user and not self.interrupt.is_set():
            time.sleep(PAUSED_POLL)
            self.poll()
            self.tick()

    # -- data
    def setup(self):
        from studio_rlhf import load_settings
        if self.state.get("started"):
            self.journal("restart", round=self.state.get("round"), best=self.state.get("best"))
            self.state["phase"] = "idle"
            self.state["restart_count"] = self.state.get("restart_count", 0) + 1
            if self.state.pop("training_batch", None) is not None and self.state["batch_size"] > 1:
                # ponytail: best effort. The last process died mid-round without running any cleanup (SIGKILL: the OOM killer, or a kill
                # after a hang), so assume memory and halve the batch. A plain crash or a signal clears the mark and keeps the batch.
                self.state["batch_size"] //= 2
                self.journal("memory", round=self.state.get("round"), restart=True, batch_size=self.state["batch_size"])
            n = len(self.state["checkpoints"]) + 1
            folders = sorted(self.directory.glob(f"round{n:03d}_*"), key=lambda p: p.stat().st_mtime, reverse=True)
            self.leftover = next((f / name for f in folders for name in ("best.npz", "last.npz") if (f / name).exists()), None)
        else:
            lr = self.plan["config"].get("learning_rate", 0.001)
            bs = self.plan["config"].get("batch_size", 4)
            self.state.update(round=0, phase="idle", started=self.clock(), active_seconds=0.0, tick=None, best=None, current=None, worse_streak=0,
                              rounds_since_best=0, pivot_index=0, learning_rate=lr, batch_size=bs, source_index=None, stop_reason=None,
                              teacher={"model": "", "reachable": None, "done": 0, "total": 0}, live=[], memory={"used": 0, "limit": 0},
                              checkpoints=[], win_rate=None, pending_pivot=None, paused=False)
            if self.plan["stop"]["win_rate"] is not None and not load_settings(self.root).get("teacher_model"):
                self.journal("error", message="The head-to-head stop condition was ignored because no teacher is set up. The session still ends by its other limits.")
            if patience_hides_pivots(self.plan):
                self.journal("error", message=f"It stops after {self.plan['stop']['patience_rounds']} rounds without a better checkpoint, before it would step back, "
                                              "so the changes to try when it gets worse will not run.")
        if not (self.directory / "snapshot.jsonl").exists():
            examples = self.build(self.root, self.plan["data"])
            if len(examples) < 2:
                raise ValueError("Passive training needs at least two examples to keep some aside for testing.")
            examples = list(examples)
            random.Random(42).shuffle(examples)
            held = examples[:min(200, max(1, len(examples) // 10))]
            _write_jsonl(self.directory / "eval.jsonl", held)  # the snapshot is written last: it marks the split as complete
            _write_jsonl(self.directory / "snapshot.jsonl", examples[len(held):])
            self.journal("start", train=len(examples) - len(held), held_out=len(held), sources=self.plan["data"]["sources"])
        self.eval_examples = _read_jsonl(self.directory / "eval.jsonl")
        self.snapshot = _read_jsonl(self.directory / "snapshot.jsonl")
        self.sweep()
        self.state["tick"] = self.clock()
        self.pause.clear()
        if self.status == "paused":
            self.pause.set()
        else:
            self.status = "running"
        self.save(status=self.status, error="")

    def training_examples(self):
        index = self.state["source_index"]
        if index is None:
            base = self.snapshot
        else:
            path = self.directory / f"source_{index}.jsonl"
            if not path.exists():
                held = {fingerprint(e) for e in self.eval_examples}
                rows = self.build(self.root, {**self.plan["data"], "sources": [self.plan["data"]["sources"][index]]})
                _write_jsonl(path, [e for e in rows if fingerprint(e) not in held])
            base = _read_jsonl(path)
        return base + [{"messages": row["messages"] + [{"role": "assistant", "content": row["answer"]}]}
                       for row in self.store.lessons(self.id) if row.get("status") not in ("dropped", "left_out")]

    # -- one round
    def continues(self):
        """True when the next round carries the optimizer over: it starts from the checkpoint the last round saved. A first round, and a
        round after a step back to an earlier best, start with a fresh optimizer."""
        state = self.state
        last = state["checkpoints"][-1] if state["checkpoints"] else None
        return bool(last and state["current"] == last["n"] and state["pending_pivot"] is None)

    def on_event(self, event):
        self.guard()
        if event.get("type") == "resume_refused":
            self.journal("optimizer", round=self.state["round"], mode="fresh", reason=str(event.get("reason", ""))[:300])
        if event.get("type") == "metric" and event.get("val_loss") is not None:
            error = float(event["val_loss"])
            self.state["live"].append([round(self.state["active_seconds"], 1), error])
            self.journal("check", round=self.state["round"], step=event.get("step"), error=error)
            if not self.stop.is_set() and plateaued([e for _, e in self.state["live"]], self.plan):
                self.stop.set()
            self.save()

    def train_round(self, n):
        state = self.state
        init = state["current"]
        init_from = self.plan["base_checkpoint"] if init is None else _checkpoint(state, init)["path"]
        examples = self.training_examples()
        # Each round (and each restart) draws its batches in a different order. A round that carries on from the last checkpoint resumes
        # it (weights, Adam's moments and the step count, so the schedule goes on); the first round and a step back start from weights only.
        resume = _checkpoint(state, init)["path"] if init is not None and self.continues() else None
        seed = (int(self.plan["config"].get("seed", 42)) + n + state.get("restart_count", 0)) % 2147483648
        try:
            while True:
                state["live"], self.cause, self.stop = [], None, threading.Event()
                if self.user:
                    self.stop.set()
                config = {**self.plan["config"], "steps": self.plan["plateau"]["max_steps"], "learning_rate": state["learning_rate"], "batch_size": state["batch_size"], "seed": seed}
                run_dir = self.directory / f"round{n:03d}_{uid('run')[-6:]}"
                state["training_batch"] = state["batch_size"]  # still here after a restart only if this process was killed outright: see setup()
                self.save()
                try:
                    if resume:
                        self.train(config, examples, run_dir, self.on_event, self.stop, self.pause, resume=resume)
                    else:
                        self.train(config, examples, run_dir, self.on_event, self.stop, self.pause, init_from=init_from)
                except InterruptedError:
                    if not (self.user or self.interrupt.is_set()):
                        raise
                    return None  # stopped while the data was still being prepared: there is no checkpoint to keep
                except ValueError as exc:
                    if "Non-finite" not in str(exc):  # the engine says "Non-finite loss" or "Non-finite gradient"
                        raise
                    shutil.rmtree(run_dir, ignore_errors=True)
                    raise _Diverged(str(exc)) from exc
                (run_dir / "dataset.jsonl").unlink(missing_ok=True)  # a copy of the whole training set per round would fill the disk; snapshot.jsonl and lessons.jsonl rebuild it
                if self.cause != "memory":
                    break
                state["batch_size"] = max(1, state["batch_size"] // 2)
                self.journal("memory", round=n, used=state["memory"]["used"], limit=state["memory"]["limit"], batch_size=state["batch_size"])
                self.sweep()  # the aborted run's folder
                self.save()
        finally:
            state.pop("training_batch", None)
        for name in ("best.npz", "last.npz"):
            if (run_dir / name).exists():
                return run_dir / name
        raise RuntimeError("Training did not save a checkpoint.")

    def teach(self, entry):
        state, plan = self.state, self.plan
        from studio_rlhf import load_settings
        self.settings = load_settings(self.root)
        info = state["teacher"]
        info["model"] = self.settings.get("teacher_model", "")
        self.stop = threading.Event()  # the round's own stop (plateau, time) has done its job
        if self.user or self.interrupt.is_set() or stop_reason(state, plan, self.clock()) == "duration":
            self.stop.set()
            return
        if not (plan["teacher"]["polish"] or plan["teacher"]["lessons"]) or not info["model"]:
            return
        state["phase"], info["done"], info["total"] = "teach", 0, 0

        def emit(event):
            if event.get("type") == "rlhf_progress":
                info["done"], info["total"] = event["done"], event["total"]
                self.guard()
                self.save()
        try:
            start = (entry["round"] - 1) * 500 % max(len(self.snapshot), 1)  # the teacher round reads the first 500 prompts: a new window each round
            result = self.teacher_round(self.directory, plan, self.settings, entry["path"], self.snapshot[start:] + self.snapshot[:start], self.stop, emit)
        except Exception as exc:  # the teacher is optional: skip this round's lesson and keep going
            info["reachable"] = False
            self.journal("teacher", round=entry["round"], skipped=str(exc)[:300])
            return
        info["reachable"] = True
        polished = False
        new = result.get("checkpoint")
        if plan["teacher"]["polish"] and new and str(new) != str(entry["path"]):
            # Rule: the polished checkpoint replaces this round's one only if it already won head to head (auto_rlhf's gate)
            # and its held-out error is not worse; otherwise the original stays and the polish is thrown away.
            error = self.evaluate(str(new), self.eval_examples)
            if error <= entry["error"]:
                entry.update(original_path=entry["path"], path=str(new), error=error)
                polished = True
        rows = []
        if plan["teacher"]["lessons"]:
            rows = [{"id": uid("lesson"), "round": entry["round"], "checkpoint": entry["n"], "prompt": (next((m["content"] for m in reversed(r["messages"]) if m["role"] == "user"), "")),
                     "reply": r.get("rejected", ""), "answer": r["chosen"], "messages": r["messages"], "status": "pending", "at": now()}
                    for r in result.get("lessons", [])]
            if rows:
                self.store.set_lessons(self.id, self.store.lessons(self.id) + rows)
                self.journal("lesson", round=entry["round"], checkpoint=entry["n"], count=len(rows))
        entry["teacher"] = {"polished": polished, "win_rate": result.get("win_rate"), "lessons": len(rows)}
        self.journal("teacher", round=entry["round"], polished=polished, win_rate=result.get("win_rate"), lessons=len(rows))

    def prune(self):
        cps, limit = self.state["checkpoints"], self.plan["limits"]["keep_checkpoints"]
        protected = {self.state["best"], self.state["current"]}
        if self.plan["stop"]["win_rate"] is not None and not self.plan["base_checkpoint"]:
            protected.add(1)  # the starting model for the head-to-head comparison
        for c in sorted(cps, key=lambda c: not c.get("discarded")):  # checkpoints on a branch that was stepped back from go first, then the oldest
            if sum(x["kept"] for x in cps) <= limit:
                break
            if c["kept"] and c["n"] not in protected:
                for path in {Path(p) for p in (c["path"], c.get("original_path")) if p}:  # a polished checkpoint keeps the one it replaced on disk too
                    if path.resolve().is_relative_to(self.directory.resolve()):
                        for file in (path, path.with_name("last.npz")):
                            file.unlink(missing_ok=True)
                        top = self.directory / path.resolve().relative_to(self.directory.resolve()).parts[0]
                        if top != path and top.name.startswith(("round", "rlhf_")):
                            shutil.rmtree(top, ignore_errors=True)  # the run folder holds nothing else worth keeping
                c["kept"] = False

    def sweep(self):
        """Delete round*/rlhf_* folders no checkpoint points at: memory reruns, crashed rounds, rejected polishes."""
        used = [Path(p).resolve() for c in self.state["checkpoints"] for p in (c["path"], c.get("original_path")) if p]
        used += [self.leftover.resolve()] if self.leftover else []
        for folder in self.directory.iterdir():
            if folder.is_dir() and folder.name.startswith(("round", "rlhf_")) and not any(p.is_relative_to(folder.resolve()) for p in used):
                shutil.rmtree(folder, ignore_errors=True)

    def one_round(self):
        state, plan = self.state, self.plan
        n = len(state["checkpoints"]) + 1
        state.update(round=n, phase="train", live=[])
        path, error, self.leftover = self.leftover, None, None
        if path is not None:  # the last process trained this round and died before it was evaluated: keep that checkpoint
            try:
                error = float(self.evaluate(str(path), self.eval_examples))
            except Exception as exc:  # killed while writing it, say
                self.journal("error", message=f"The checkpoint the last run left for round {n} could not be read ({str(exc)[:200]}), so the round trains again.", round=n)
                path = None
        self.journal("round", round=n, parent=state["current"], learning_rate=state["learning_rate"], batch_size=state["batch_size"],
                     optimizer="resumed" if self.continues() else "fresh", **({"recovered": True} if path else {}))
        self.save()
        if path is None:
            try:
                path = self.train_round(n)
            except _Diverged as exc:
                return self.diverged(n, exc)
            if path is None or self.interrupt.is_set():
                return
        state["phase"] = "save"
        if error is None:
            error = float(self.evaluate(str(path), self.eval_examples))
        if not math.isfinite(error):
            if Path(path).resolve().parent.parent == self.directory.resolve():
                shutil.rmtree(Path(path).parent, ignore_errors=True)
            return self.diverged(n, _Diverged(f"its test error is {error}"))
        entry = {"n": n, "round": n, "path": str(path), "error": error, "elapsed": round(state["active_seconds"], 1),
                 "at": now(), "parent": state["current"], "kept": True, "discarded": False, "pivot": state["pending_pivot"], "teacher": None}
        state["pending_pivot"] = None
        before = {k: state[k] for k in ("best", "current", "worse_streak", "rounds_since_best")}
        state["checkpoints"].append(entry)
        improved = self.judge(entry)  # before the teacher: a crash while it teaches must not lose a new best
        self.save()
        self.teach(entry)
        self.sweep()  # a polish that was not kept
        if (entry["teacher"] or {}).get("polished"):
            state.update(before)  # the polish lowered the error: judge it again from where it stood
            improved = self.judge(entry)
        self.journal("checkpoint", n=n, round=n, error=entry["error"], path=entry["path"], best=improved, teacher=entry["teacher"], elapsed=entry["elapsed"])
        if improved and plan["stop"]["win_rate"] is not None:
            self.win_rate(entry)
        self.prune()
        state["phase"] = "idle"
        move = next_move(state, plan)
        if move["action"] == "rollback" and not self.user:
            self.rollback(move, entry)
        self.save()

    def judge(self, entry):
        """Best / current / streak bookkeeping for a new checkpoint. True when it is the new best."""
        state = self.state
        best = _checkpoint(state, state["best"])
        improved = best is None or entry["error"] < best["error"]
        if improved:
            state.update(best=entry["n"], worse_streak=0, rounds_since_best=0)
        else:
            state["worse_streak"] += 1
            state["rounds_since_best"] += 1
        state["current"] = entry["n"]
        return improved

    def diverged(self, n, exc):
        """The round blew up: treat it as a checkpoint that is worse than the best. Step back and make the next change; with no best yet, just halve the learning rate."""
        state = self.state
        state["phase"] = "idle"
        best, limit = state["best"], self.plan["worse"]["limit"]
        if best is None:
            if state["learning_rate"] <= 1e-7:
                raise exc
            state["learning_rate"] = max(1e-7, state["learning_rate"] / 2)
            self.journal("error", message=f"Training in round {n} went wrong ({exc}), so it tried again with a smaller learning rate.", round=n)
        else:
            state["worse_streak"] = max(state["worse_streak"], limit)
            state["rounds_since_best"] += 1
            move = next_move(state, self.plan)
            self.journal("error", message=f"Training in round {n} went wrong ({exc}), so it " + (f"went back to checkpoint {best} and tried a change." if move["action"] == "rollback" else "has no more changes to try."), round=n)
            if move["action"] == "rollback":
                self.rollback(move, {"n": state["current"], "error": None})
        self.save()

    def win_rate(self, entry):
        start = self.plan["base_checkpoint"] or (_checkpoint(self.state, 1) or {}).get("path")
        if not start or str(start) == entry["path"] or not self.settings.get("teacher_model"):
            return
        try:
            self.state["win_rate"] = self.versus(self.settings, entry["path"], str(start), self.eval_examples, self.stop)
        except Exception as exc:
            self.journal("teacher", round=entry["round"], skipped="comparison with the starting model failed: " + str(exc)[:200])

    def rollback(self, move, entry):
        state = self.state
        best, name = state["best"], move["pivot"]
        discarded = [c["n"] for c in state["checkpoints"] if c["n"] > best and not c["discarded"]]
        for c in state["checkpoints"]:
            c["discarded"] = c["discarded"] or c["n"] > best
        # a step back gives the change its own patience window, or the default patience would end the session before the second change is tried
        state.update(current=best, worse_streak=0, rounds_since_best=0, pivot_index=move["index"] + 1, pending_pivot=name)
        if name == "halve_lr":
            state["learning_rate"] = max(1e-7, state["learning_rate"] / 2)
        elif name == "next_dataset":
            count = len(self.plan["data"]["sources"])
            state["source_index"] = 0 if state["source_index"] is None else (state["source_index"] + 1) % count
        else:
            rows = self.store.lessons(self.id)
            for row in rows:
                if row["checkpoint"] > best and row.get("status") == "pending":
                    row["status"] = "left_out"
            self.store.set_lessons(self.id, rows)
        self.journal("rollback", **{"from": entry["n"], "to": best, "pivot": name, "discarded": discarded, "error": entry["error"]})

    # -- the session
    def run(self):
        pulse = threading.Thread(target=self.pulse, daemon=True)
        try:
            self.store.beat(self.id, os.getpid())
            pulse.start()  # before the slow setup: building the data can outlast the watchdog's silence limit
            try:
                self.setup()  # inside the try so a setup failure is recorded with its reason, not only in trainer.log
                self.ready.set()
                while True:
                    self.wait_if_paused()
                    self.tick()
                    self.poll()
                    reason = "user" if self.user else stop_reason(self.state, self.plan, self.clock())
                    if reason or self.interrupt.is_set():
                        break
                    self.one_round()
                    if self.interrupt.is_set():
                        break
            finally:
                self.done.set()  # the pulse thread polls and writes the status: it must be gone before the last save
                if pulse.is_alive():
                    pulse.join(timeout=5)
            if reason:
                self.state.update(stop_reason=reason, phase="idle", tick=None)
                self.journal("stop", reason=reason, best=self.state["best"])
                self.save(status="stopped" if reason == "user" else "finished", command="")
            else:
                self.save()  # interrupted by a signal: stay "running" so the watchdog starts us again
        except BaseException as exc:
            self.journal("error", message=str(exc)[:500], round=self.state.get("round"))
            self.save(error=f"{type(exc).__name__}: {exc}"[:500])
            raise
        finally:
            self.done.set()
            if pulse.is_alive():
                pulse.join(timeout=5)
        return self.state


def run_session(root, session_id, *, train=default_train, teacher_round=default_teacher_round, evaluate=default_evaluate,
                build=build_examples, versus=default_versus, clock=time.monotonic, interrupt=None, awake=None):
    """The loop. Resumable: a restarted trainer reads the record and carries on; an interrupted round keeps the checkpoint it saved, or is redone.

    `awake`: (keep, release) to hold the computer awake while running and not paused, or None.
    """
    row = PassiveStore(root).get(session_id)
    if row is None:
        raise KeyError(f"No passive session {session_id}.")
    if row["status"] not in ACTIVE:
        return row["state"]
    return _Trainer(root, session_id, train, teacher_round, evaluate, build, versus, clock, interrupt or threading.Event(), awake).run()


def trainer_command(root, session_id):
    return module_command("studio_passive", "run", session_id, "--root", str(root))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="studio_passive")
    sub = parser.add_subparsers(dest="action", required=True)
    run = sub.add_parser("run", help="run (or resume) a passive training session")
    run.add_argument("session_id")
    run.add_argument("--root", default=str(ROOT))
    args = parser.parse_args(argv)
    store = PassiveStore(args.root)
    row = store.get(args.session_id)
    lock = row and take_lock(store.directory(args.session_id) / "trainer.lock")
    if row and lock is None:
        sys.exit(f"Another trainer is already running passive session {args.session_id}; this one exits.")
    interrupt = threading.Event()
    for name in ("SIGINT", "SIGTERM"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), lambda *_: interrupt.set())  # the pulse thread passes it on to the round's stop within a second
    awake = None
    if row and row["plan"]["limits"]["keep_awake"]:
        import studio_passive_watch  # here, not at the top: it imports this module
        awake = (studio_passive_watch.keep_awake, studio_passive_watch.release_awake)
    try:
        run_session(args.root, args.session_id, interrupt=interrupt, awake=awake)
    finally:
        if lock:
            lock.close()


if __name__ == "__main__":
    main()
