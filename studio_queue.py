"""One machine, one queue: a single runner executes one job at a time and persists data/queue.json."""
import dataclasses
import os
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import studio_passive_watch as platform_helpers  # battery + keep-awake live there; never duplicated
from studio_core import atomic_json

KINDS = ("pretrain", "quicktest", "compare", "post_round", "restore_answers", "checkpoint_now")
MEMORY = {"half": .5, "most": .75, "all": 1.}
SAMPLE_MB = 64  # ponytail: flat per-sample memory guess for the batch cap; jobs may override with params["sample_mb"]


@dataclass
class Plan:
    hours: str = "any"             # any | nights | set; "any" so a first Train starts at once
    start_hour: int = 22           # window for "nights" (22-7) and "set"
    end_hour: int = 7
    memory: str = "most"           # half | most | all
    idle_only: bool = True
    idle_after_s: float = 60.      # no keyboard or mouse for this long counts as idle
    pause_on_battery: bool = True
    keep_awake: bool = False
    chat_follows_best: bool = True


@dataclass
class Job:
    kind: str
    label: str = ""
    note: str = ""
    params: dict = field(default_factory=dict)
    status: str = "waiting"         # waiting | running | done | failed | stopped
    eta_s: float | None = None
    created_at: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    error: str = ""


@dataclass
class QueueState:
    job: Job | None = None         # the running job, None when idle
    step: int = 0
    total: int = 0
    phase: str = ""
    lr: float | None = None
    eta_s: float | None = None
    paused: bool = False           # by the user
    blocked: str | None = None     # plan reason the machine may not train right now (hours / idle / battery)
    stopping: bool = False         # stop_after_cooldown in progress
    up_next: list = field(default_factory=list)
    plan: dict = field(default_factory=dict)  # asdict(Plan)


def idle_seconds():
    """Seconds since last keyboard/mouse input, or None when unknown. macOS only (ioreg HIDIdleTime)."""
    if sys.platform != "darwin":
        return None
    try:
        out = subprocess.run(["ioreg", "-c", "IOHIDSystem"], capture_output=True, text=True, timeout=5).stdout
        return int(next(l for l in out.splitlines() if "HIDIdleTime" in l).split("=")[-1]) / 1e9
    except (OSError, StopIteration, ValueError, subprocess.SubprocessError):
        return None


def physical_memory():
    from studio_passive import total_memory  # has the Windows path; os.sysconf is POSIX-only
    return total_memory()


def batch_cap(plan, sample_mb=SAMPLE_MB, ram=None):
    return max(1, int((ram or physical_memory()) * MEMORY[plan.memory] / (sample_mb * 2 ** 20)))


def in_window(plan, hour):
    if plan.hours == "any":
        return True
    start, end = (22, 7) if plan.hours == "nights" else (plan.start_hour, plan.end_hour)
    return start <= hour < end if start < end else hour >= start or hour < end


class Runner:
    def __init__(self, path, handlers, services=None, tick=.05, clock=datetime.now):
        self.path, self.handlers, self.services, self.tick, self.clock = Path(path), handlers, services, tick, clock
        self.lock = threading.RLock()
        self.subscribers = []
        self.jobs, self.plan = [], Plan()
        self._load()
        self.running = None            # Job on the runner thread
        self.progress = {}
        self.user_paused = False
        self.stopping = False
        self.blocked = None
        self.pause_event, self.stop_event = threading.Event(), threading.Event()
        self.closed = threading.Event()
        self.thread = None

    # ---- persistence
    def _load(self):
        if not self.path.exists():
            return
        import json
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.plan = Plan(**{k: v for k, v in data.get("plan", {}).items() if k in Plan.__dataclass_fields__})
        self.jobs = [Job(**j) for j in data.get("jobs", [])]
        for job in self.jobs:          # a run cut short by a restart goes back to the front
            if job.status == "running":
                job.status = "waiting"

    def _save(self):
        done = [j for j in self.jobs if j.status not in ("waiting", "running")][-50:]
        live = [j for j in self.jobs if j.status in ("waiting", "running")]
        self.jobs = done + live
        atomic_json(self.path, {"plan": dataclasses.asdict(self.plan), "jobs": [dataclasses.asdict(j) for j in self.jobs]})

    # ---- events
    def subscribe(self, callback):
        self.subscribers.append(callback)

    def _emit(self, kind, payload=None):
        for callback in list(self.subscribers):
            try:
                callback(kind, payload or {})
            except Exception:  # noqa: BLE001 - a broken listener must never stop training
                pass

    # ---- public API
    def enqueue(self, job):
        if isinstance(job, str):
            job = Job(kind=job)
        if job.kind not in KINDS:
            raise ValueError(f"Unknown job kind: {job.kind}")
        with self.lock:
            self.jobs.append(job)
            self._save()
        self._emit("queue", {"job": job.id})
        return job

    def move_up(self, job_id):
        with self.lock:
            queued = [j for j in self.jobs if j.status == "waiting"]
            i = next((i for i, j in enumerate(queued) if j.id == job_id), 0)
            if i:
                a, b = self.jobs.index(queued[i - 1]), self.jobs.index(queued[i])
                self.jobs[a], self.jobs[b] = self.jobs[b], self.jobs[a]
                self._save()
        self._emit("queue", {"job": job_id})

    def remove(self, job_id):
        with self.lock:
            job = next((j for j in self.jobs if j.id == job_id), None)
            if job is self.running:
                self.stop_event.set()  # the running job is stopped; the runner marks it
            elif job:
                self.jobs.remove(job)
                self._save()
        self._emit("queue", {"job": job_id})

    def pause(self):
        self.user_paused = True
        self._sync_pause()

    def resume(self):
        self.user_paused = False
        self._sync_pause()

    def set_plan(self, plan):
        with self.lock:
            self.plan = Plan(**plan) if isinstance(plan, dict) else plan
            self._save()
        self._emit("plan", dataclasses.asdict(self.plan))

    def checkpoint_now(self, then_stop=False):
        """Branch a short cool-down on a copy of the running job's last checkpoint; the main run keeps going."""
        with self.lock:
            main = self.running
        handler = self.handlers.get("checkpoint_now")
        if not main or not handler:
            return False
        child = Job("checkpoint_now", f"Checkpoint of {main.label or main.kind}", params={**main.params, "parent": main.id})

        def work():
            self._emit("checkpoint", {"job": main.id, "phase": "start"})
            try:
                handler(child, self.services, lambda e: self._emit("train", {"job": child.id, **e}), threading.Event(), threading.Event())
            except Exception as exc:  # noqa: BLE001
                self._emit("checkpoint", {"job": main.id, "phase": "failed", "error": str(exc)})
                return
            self._emit("checkpoint", {"job": main.id, "phase": "done"})
            if then_stop:
                self.stop_event.set()
        threading.Thread(target=work, daemon=True).start()
        return True

    def stop_after_cooldown(self):
        self.stopping = True
        if not self.checkpoint_now(then_stop=True):
            self.stop_event.set()

    def state(self):
        with self.lock:
            p = self.progress
            return QueueState(
                job=self.running, step=p.get("step", 0), total=p.get("total", 0), phase=p.get("phase", ""), lr=p.get("lr"),
                eta_s=p.get("eta_s"), paused=self.user_paused, blocked=self.blocked, stopping=self.stopping,
                up_next=[j for j in self.jobs if j.status == "waiting"], plan=dataclasses.asdict(self.plan))

    def start(self):
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def shutdown(self, timeout=10):
        self.closed.set()
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout)

    # ---- runner thread
    def _sync_pause(self):
        (self.pause_event.set if self.user_paused or self.blocked else self.pause_event.clear)()

    def _gate(self):
        """Why the plan forbids training right now, or None."""
        plan = self.plan
        if not in_window(plan, self.clock().hour):
            return "outside hours"
        if plan.pause_on_battery and platform_helpers.on_battery():
            return "on battery"
        if plan.idle_only:
            idle = idle_seconds()
            if idle is not None and idle < plan.idle_after_s:
                return "computer in use"
        return None

    def _watch(self):
        reason = self._gate()
        if reason != self.blocked:
            self.blocked = reason
            self._sync_pause()
            self._emit("blocked", {"reason": reason})

    def _loop(self):
        while not self.closed.is_set():
            self._watch()
            with self.lock:
                job = next((j for j in self.jobs if j.status == "waiting"), None)
            if not job or self.blocked or self.user_paused:
                time.sleep(self.tick)
                continue
            self._run(job)

    def _run(self, job):
        handler = self.handlers.get(job.kind)
        with self.lock:
            job.status, self.running, self.progress, self.stopping = "running", job, {}, False
            job.params = {**job.params, "max_batch_size": batch_cap(self.plan, job.params.get("sample_mb", SAMPLE_MB))}
            self.stop_event.clear()
            self._save()
        started = time.monotonic()

        def emit(event):
            if event.get("total"):
                step, total = event.get("step", 0), event["total"]
                self.progress.update(step=step, total=total, lr=event.get("lr", self.progress.get("lr")),
                                     phase=event.get("phase", self.progress.get("phase", "")))
                if step:
                    self.progress["eta_s"] = (time.monotonic() - started) / step * (total - step)
            self._emit("train", {"job": job.id, **event})

        awake = platform_helpers.keep_awake(os.getpid()) if self.plan.keep_awake else None
        outcome = {}

        def work():
            try:
                handler(job, self.services, emit, self.stop_event, self.pause_event)
            except Exception as exc:  # noqa: BLE001
                outcome["error"] = str(exc) or type(exc).__name__
        worker = threading.Thread(target=work, daemon=True)
        self._emit("job", {"job": job.id, "status": "running"})
        worker.start()
        while worker.is_alive():
            self._watch()
            worker.join(self.tick)
        platform_helpers.release_awake(awake)
        with self.lock:
            job.status = "failed" if "error" in outcome else "stopped" if self.stop_event.is_set() else "done"
            job.error = outcome.get("error", "")
            self.running = None
            self.stopping = False
            if job.status == "done" and job.kind == "pretrain" and job.params.get("user_text"):
                self.jobs.append(Job("restore_answers", "Restore how it answers", params={"round": 3, "teacher": job.params.get("previous_best")},
                                     note="added automatically after new text; a short lesson from the previous version restores how it answers"))
            if self.closed.is_set() and job.status == "stopped":
                job.status = "waiting"  # shut down mid-run: try again next start
            self._save()
        self._emit("job", {"job": job.id, "status": job.status, "error": job.error})
