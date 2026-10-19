"""One machine, one queue: a single runner executes one job at a time and persists data/queue.json."""
import dataclasses
import logging
import os
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import studio_passive_watch as platform_helpers  # battery + keep-awake live there; never duplicated
from studio_core import atomic_json, read_json
from studio_passive import PassiveStore
from studio_timing import timed

KINDS = ("pretrain", "quicktest", "compare", "post_round", "restore_answers", "checkpoint_now", "import_pretrained",
         "pretrain_distributed", "diloco_member")
MEMORY = {"half": .5, "most": .75, "all": 1.}
QUEUED = ("waiting", "held")  # jobs listed under Up next, in order; only "waiting" ones are picked to run
MIN_FREE = 2 * 2 ** 30  # a job needs at least this much free disk to start
PASSIVE = "passive training is running"  # the gate's reason while a passive session holds the machine; "(session <id>)" follows it


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
    status: str = "waiting"         # waiting | held | running | done | failed | stopped
    eta_s: float | None = None
    created_at: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    error: str = ""
    node: str = ""                  # the peer it runs (or ran) on; "" for a job that runs here
    epoch: int = 0                  # the lease epoch it was last placed at (studio_scheduler); a write carrying another epoch is refused
    owner: str = ""                 # for a job another node owns and this one runs: how to reach that node


@dataclass(frozen=True)
class Gate:
    """Whether the plan lets the machine train right now. Frozen, so two readings compare with == (the runner announces a change)."""
    reason: str | None = None            # PASSIVE + " (session <id>)" | "outside hours" | "on battery" | "computer in use" | None (may train)
    resumes_at: datetime | None = None   # for "outside hours": when the next window opens; None for every other reason
    supported: bool = True               # False when idle_only is on but this computer cannot report idle time (the plan then ignores it)


@dataclass
class QueueState:
    job: Job | None = None         # the running job, None when idle
    step: int = 0
    total: int = 0
    phase: str = ""
    lr: float | None = None
    eta_s: float | None = None
    paused: bool = False           # by the user; persisted, so true when idle too ("Paused by you")
    blocked: str | None = None     # == gate.reason, kept for callers that only need the words
    gate: Gate = field(default_factory=Gate)
    stopping: bool = False         # stop_after_cooldown in progress
    up_next: list = field(default_factory=list)
    plan: dict = field(default_factory=dict)  # asdict(Plan)


def _idle_macos():
    try:
        out = subprocess.run(["ioreg", "-c", "IOHIDSystem"], capture_output=True, text=True, timeout=5).stdout
        return int(next(l for l in out.splitlines() if "HIDIdleTime" in l).split("=")[-1]) / 1e9
    except (OSError, StopIteration, ValueError, subprocess.SubprocessError):
        return None


def _idle_windows():
    import ctypes
    from ctypes import wintypes

    class LastInput(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD)]

    info = LastInput(ctypes.sizeof(LastInput), 0)
    try:
        user32, kernel32 = ctypes.windll.user32, ctypes.windll.kernel32
        kernel32.GetTickCount.restype = wintypes.DWORD
        if not user32.GetLastInputInfo(ctypes.byref(info)):
            return None
        return ((kernel32.GetTickCount() - info.dwTime) & 0xFFFFFFFF) / 1000  # both are 32-bit millisecond ticks that wrap
    except (OSError, AttributeError):
        return None


def idle_seconds():
    """Seconds since last keyboard/mouse input, or None when this computer cannot say (Linux, or a failed lookup).
    macOS reads ioreg's HIDIdleTime; Windows reads GetLastInputInfo."""
    if sys.platform == "darwin":
        return _idle_macos()
    if sys.platform == "win32":
        return _idle_windows()
    return None


def size_text(n):
    return f"{n / 2 ** 30:.1f} GB" if n >= 2 ** 30 else f"{n / 2 ** 20:.0f} MB"


def disk_need(root):
    """Free space a run needs: 3 times the largest checkpoint already saved under `root`, never less than MIN_FREE (the last
    checkpoint is the best guide to the next one)."""
    largest = max((p.stat().st_size for p in Path(root).glob("runs/*/*.npz")), default=0)
    return max(MIN_FREE, 3 * largest)


def passive_session(root):
    """The id of the passive training session (studio_passive) under `root` that is queued, running or paused, else None. It trains
    in its own process and a paused one still holds its model in memory, so the queue waits for it rather than compete for the machine."""
    if not (Path(root) / "data" / "passive.sqlite").exists():  # never used here: opening the store would create it
        return None
    try:
        session = PassiveStore(root).active()
    except sqlite3.Error:
        return None  # an unreadable record cannot hold the queue forever; Resources › Passive training shows the session
    return session and session["id"]


def passive_running(root):
    """True while a passive training session is queued, running or paused (see passive_session)."""
    return passive_session(root) is not None


def run_folders(root, job):
    """Where a job wrote (or would have): its `output_dir` when the app had picked one, else the folder the app names after it."""
    runs = Path(root) / "runs"
    found = [Path(job.params["output_dir"])] if job.params.get("output_dir") else []
    return found + [runs / f"{name}_{job.id}" for name in ("queue", "compare")]


def reconcile_run(folder):
    """A run whose process has died must not say it is training: mark its run.json "stopped" (and `interrupted`, so a page can say the
    app stopped under it). True when it changed one. Only a record still saying "training" is touched."""
    record = Path(folder) / "run.json"
    try:
        data = read_json(record)
    except (OSError, ValueError):
        return False
    if not isinstance(data, dict) or data.get("status") != "training":
        return False
    atomic_json(record, {**data, "status": "stopped", "interrupted": True})
    return True


def in_window(plan, hour):
    if plan.hours == "any":
        return True
    start, end = (22, 7) if plan.hours == "nights" else (plan.start_hour, plan.end_hour)
    return start <= hour < end if start < end else hour >= start or hour < end


def next_start(plan, now):
    """When the plan next lets training start, as a whole hour: `now` when already inside the window, the next
    opening otherwise, None when the plan has no hours limit ("any"). The single source for "Starts tonight at 22:00"."""
    if plan.hours == "any":
        return None
    if in_window(plan, now.hour):
        return now
    start = (22 if plan.hours == "nights" else plan.start_hour) % 24
    opens = now.replace(hour=start, minute=0, second=0, microsecond=0)
    return opens if opens > now else opens + timedelta(days=1)


class Runner:
    """Runs one job at a time, each in a worker process of its own (studio_worker). `prepare(job)` runs in this process just before a
    job starts: it fills in what the handler would otherwise ask the app for (output folder, checkpoints) and returns the context
    the handler gets, {"root", "run_dir"}. Without it the context is the folder above data/ and its runs/.

    `ledger` (studio_memory.Ledger) is what the app holds outside the job, chat's models: its total is taken off the share a job
    gets. `make_room()` asks chat to evict its oldest model and says whether it did. `on_start(job)` is called as a job is about to
    start, before its share is worked out, and `on_end(job)` once it has ended: chat leaves the GPU for the one and may return for
    the other. All three run on the runner thread, never the Tk one, and a failing hook never stops the queue."""

    def __init__(self, path, handlers, prepare=None, tick=3., clock=datetime.now, ledger=None, make_room=None, on_start=None, on_end=None,
                 cluster=None):
        self.path, self.handlers, self.prepare, self.tick, self.clock = Path(path), handlers, prepare, tick, clock
        self.ledger, self.make_room, self.on_start, self.on_end = ledger, make_room, on_start, on_end
        self.root = self.path.parent.parent
        self.handles = set()           # live workers: the running job's and any cool-down branched off it
        self.lock = threading.RLock()
        self.subscribers = []
        self.jobs, self.plan = [], Plan()
        self.user_paused = False       # a user intent: restored from queue.json, so a pause survives a restart
        self.stopping_job = ""         # the id of a running job the person stopped: persisted, so a restart does not run it again
        self.reap_grace = 35.          # seconds a worker an earlier app left gets to leave by itself (studio_worker.EXIT_GRACE and a margin)
        self.dead_runs = []            # jobs whose process an earlier app left running: their run.json is reconciled once it is stopped
        self._load()
        self.running = None            # Job on the runner thread
        self.progress = {}
        self.stopping = False
        self.gate = Gate()
        self.blocked = None
        self.wake = threading.Event()  # set by anything the runner thread should act on now: a new job, a resume, a plan change, shutdown
        self._gate_lock = threading.Lock()
        self._gate_at, self._gate_stale, self._gate_busy = -float("inf"), False, False
        self.pause_event, self.stop_event = threading.Event(), threading.Event()
        self._sync_pause()
        self.closed = threading.Event()
        self.thread = None
        import studio_remote
        self.cluster = (cluster or studio_remote.Cluster)(self)  # inert until peers are listed in the workspace config; `cluster(runner)` for tests

    # ---- persistence
    def _load(self):
        if not self.path.exists():
            return
        import json
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.plan = Plan(**{k: v for k, v in data.get("plan", {}).items() if k in Plan.__dataclass_fields__})
        self.jobs = [Job(**j) for j in data.get("jobs", [])]
        self.user_paused = bool(data.get("user_paused", False))
        self.stopping_job = str(data.get("stopping") or "")
        for job in self.jobs:          # a run cut short by a restart goes back to the front
            if job.status == "running" and not job.owner and not job.node:
                self.dead_runs.append(job)
            if job.status == "running" and job.id == self.stopping_job and not job.owner and not job.node:
                job.status, job.error = "stopped", ""  # the person stopped it; the app went before it ended
            elif job.owner and job.status in ("running", "waiting", "held"):  # another node's job cannot renew its lease now: its owner places it again
                job.status, job.error = "stopped", "This machine restarted while it ran the job for another machine, which will place it again."
            elif job.status == "running" and job.node:  # running on a peer: placed again, at the next epoch, so the peer's copy is fenced
                import studio_scheduler
                studio_scheduler.interrupt(job)  # from its last checkpoint, with the steps left and the offset that go with it
            elif job.status == "running":
                job.status = "waiting"
        self.stopping_job = ""

    def _save(self):
        done = [j for j in self.jobs if j.status not in ("waiting", "running", "held")][-50:]
        live = [j for j in self.jobs if j.status in ("waiting", "running", "held")]
        dropped = [j for j in self.jobs if j.status not in ("waiting", "running", "held") and not any(j is d for d in done)]
        if dropped and getattr(self, "cluster", None) is not None:
            self.cluster.trimmed(dropped)  # (no I/O: it only remembers their result keys, which a sweep must not delete)
        self.jobs = done + live
        atomic_json(self.path, {"plan": dataclasses.asdict(self.plan), "user_paused": self.user_paused, "stopping": self.stopping_job,
                                "jobs": [dataclasses.asdict(j) for j in self.jobs]})

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
        self.wake.set()
        self._emit("queue", {"job": job.id})
        return job

    def _queued(self):
        return [j for j in self.jobs if j.status in QUEUED]

    def move(self, job_id, index):
        """Put a waiting (or held) job at `index` among the queued jobs; 0 runs it next."""
        with self.lock:
            queued = self._queued()
            job = next((j for j in queued if j.id == job_id), None)
            if job is not None:
                others = [j for j in queued if j is not job]
                index = max(0, min(int(index), len(others)))
                order = others[:index] + [job] + others[index:]
                for slot, j in zip([i for i, j in enumerate(self.jobs) if j.status in QUEUED], order):
                    self.jobs[slot] = j
                self._save()
        self.wake.set()
        self._emit("queue", {"job": job_id})

    def move_up(self, job_id):
        with self.lock:
            i = next((i for i, j in enumerate(self._queued()) if j.id == job_id), 0)
        self.move(job_id, max(0, i - 1))

    def hold(self, job_id, held=True):
        """A held job keeps its place but is skipped until released; the jobs after it run meanwhile."""
        with self.lock:
            job = next((j for j in self._queued() if j.id == job_id), None)
            if job is not None:
                job.status = "held" if held else "waiting"
                self._save()
        self.wake.set()
        self._emit("queue", {"job": job_id})

    def pin(self, job_id, pinned=None):
        """Keep a waiting job on one machine (its address or hostname): it runs there or waits, and is never moved. None lets it
        go to any machine again. The placement pass that follows reads the new pin."""
        with self.lock:
            job = next((j for j in self._queued() if j.id == job_id), None)
            if job is not None:
                job.params = {**{k: v for k, v in job.params.items() if k != "pinned"}, **({"pinned": pinned} if pinned else {})}
                self._save()
        self.wake.set()
        self._emit("queue", {"job": job_id})

    def remove(self, job_id):
        remote = None
        with self.lock:
            job = next((j for j in self.jobs if j.id == job_id), None)
            if job is self.running:
                self.stopping_job = job.id  # saved first: an app that goes before the job ends must not run it again
                self._save()
                self.stop_event.set()  # the running job is stopped (the runner thread tells its worker); the runner marks it
            elif job and job.node and job.status == "running":
                remote = job           # running on a peer: it is told to stop, and the record is kept as stopped
            elif job:
                self.jobs.remove(job)
                self._save()
                self.cluster.removed(job)  # (under the lock, so a sweep never sees it gone before its result key is remembered; no I/O)
        if remote:
            self.cluster.cancel(remote)
        self.wake.set()
        self._emit("queue", {"job": job_id})

    def pause(self):
        """Stop starting jobs (and pause the running one) until resume(); saved, and announced as a "queue" event."""
        self._set_paused(True)

    def resume(self):
        self._set_paused(False)

    def _set_paused(self, paused):
        with self.lock:
            self.user_paused = paused
            self._sync_pause()
            self._save()
        self.wake.set()
        self._emit("queue", {"paused": paused})

    def set_plan(self, plan):
        with self.lock:
            self.plan = Plan(**plan) if isinstance(plan, dict) else plan
            self._gate_stale = True  # the next _watch reads the gate afresh, whatever its age
            self._save()
        self.wake.set()
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
                handle = self._start(child, handler, lambda e: self._emit("train", {"job": child.id, **e}))
                handle.done.wait()
                error = handle.error
            except Exception as exc:  # noqa: BLE001
                error = str(exc) or type(exc).__name__
            if error:
                self._emit("checkpoint", {"job": main.id, "phase": "failed", "error": error})
                return
            self._emit("checkpoint", {"job": main.id, "phase": "done"})
            if then_stop:
                self.stop_event.set()
                self.wake.set()
        threading.Thread(target=work, daemon=True).start()
        return True

    def stop_after_cooldown(self):
        self.stopping = True
        if not self.checkpoint_now(then_stop=True):
            self.stop_event.set()
            self.wake.set()

    def state(self):
        gate = self.gate  # the last reading of the runner thread; reading it here would spawn ioreg/pmset on the Tk thread
        if not self.thread:
            self._watch_soon()
        with self.lock:
            p = self.progress
            return QueueState(
                job=self.running, step=p.get("step", 0), total=p.get("total", 0), phase=p.get("phase", ""), lr=p.get("lr"),
                eta_s=p.get("eta_s"), paused=self.user_paused, blocked=gate.reason, gate=gate, stopping=self.stopping,
                up_next=self._queued(), plan=dataclasses.asdict(self.plan))

    def start(self):
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        self.cluster.start()

    def shutdown(self, timeout=10):
        """Stop the running job and wait up to `timeout` for it to save. A worker still saving then is left to finish: it exits by
        itself once this process has gone (its socket closes), and the job runs again at the next start."""
        self.closed.set()
        self.stop_event.set()
        self.wake.set()
        for handle in list(self.handles):
            handle.stop()
        if self.thread:
            self.thread.join(timeout)
        self.cluster.close()

    # ---- runner thread
    def _sync_pause(self):
        (self.pause_event.set if self.user_paused or self.blocked else self.pause_event.clear)()

    @timed("queue._gate")
    def _gate(self):
        """The plan's verdict right now, after an active passive session, which holds the queue (and pauses a running job) until
        it ends. Idle detection that this computer cannot do is reported (supported=False) and the
        idle_only setting is then ignored, rather than silently passing as if the computer were idle."""
        plan = self.plan
        idle = idle_seconds() if plan.idle_only else None
        supported = not (plan.idle_only and idle is None)
        now = self.clock()
        session = passive_session(self.path.parent.parent)
        if session is not None:  # first: it is why nothing trains, whatever the plan says
            return Gate(f"{PASSIVE} (session {session})", None, supported)
        if not in_window(plan, now.hour):
            return Gate("outside hours", next_start(plan, now), supported)
        if plan.pause_on_battery and platform_helpers.on_battery():
            return Gate("on battery", None, supported)
        if idle is not None and idle < plan.idle_after_s:
            return Gate("computer in use", None, supported)
        return Gate(None, None, supported)

    def _watch(self):
        """Read the gate at most once per `tick` (it can spawn ioreg and pmset), or at once after the plan changed."""
        with self._gate_lock:
            if not self._gate_stale and time.monotonic() - self._gate_at < self.tick:
                return
            self._gate_stale = False
            gate = self._gate()
            self._gate_at = time.monotonic()
            if gate == self.gate:
                return
            self.gate, self.blocked = gate, gate.reason
            self._sync_pause()
        self._emit("blocked", {"reason": gate.reason, "gate": gate})

    def _watch_soon(self):
        """Before the runner thread exists, state() asks for a reading on a short-lived thread and shows the last one meanwhile."""
        if self._gate_busy or (time.monotonic() - self._gate_at < self.tick and not self._gate_stale):
            return
        self._gate_busy = True

        def work():
            try:
                self._watch()
            finally:
                self._gate_busy = False
        threading.Thread(target=work, daemon=True).start()

    def _reap(self):
        """Workers an app that crashed left behind: their jobs went back to waiting in _load. A worker whose app went away is already
        saving its checkpoint and leaving by itself, so each gets up to the worker's own grace to finish before it is stopped; then the
        runs they left saying "training" are marked stopped."""
        import studio_worker
        try:
            studio_worker.reap(self.root / "data" / "workers", grace=self.reap_grace)
        except OSError:
            pass
        dead, self.dead_runs = self.dead_runs, []
        for job in dead:
            for folder in run_folders(self.root, job):
                reconcile_run(folder)

    def _start(self, job, handler, emit, on_end=None):
        """Start `job` in a worker process; the handle is tracked until the process ends."""
        import studio_worker
        context = self.prepare(job) if self.prepare else {"root": str(self.root), "run_dir": str(self.root / "runs")}

        def ended():
            self.handles.difference_update([h for h in list(self.handles) if h.done.is_set()])
            if on_end:
                on_end()
        handle = studio_worker.Handle(job, handler, context, emit, self.root / "data" / "workers", self.root / "logs" / "worker.log", ended)
        self.handles.add(handle)
        if handle.done.is_set():  # it ended before it was added
            self.handles.discard(handle)
        return handle

    def _loop(self):
        self._reap()
        while not self.closed.is_set():
            self.wake.clear()  # before looking: whatever sets it from here on is seen by the next pass
            self._watch()
            self._place_waiting()
            job = self._next_local()
            if not job or self.blocked or self.user_paused:
                self.wake.wait(self.tick)
                continue
            self._run(job)

    def _next_local(self):
        """The first waiting job that belongs on this machine: a job pinned to, or better placed on, a peer is skipped, not waited for."""
        with self.lock:
            waiting = [j for j in self.jobs if j.status == "waiting"]
        return next((j for j in waiting if self.cluster.place(j).where == "local"), None)

    def _place_waiting(self):
        """Hand each waiting job that belongs on a peer to it. Runs between jobs and while one runs, so a free peer is used at once. The
        plan's hours still apply (they say when to train) and so does a pause; this machine's battery and keyboard do not, because they
        say nothing about the peer."""
        if self.user_paused or self.blocked == "outside hours":
            return
        with self.lock:
            waiting = [j for j in self.jobs if j.status == "waiting"]
        for job in waiting:
            placement = self.cluster.place(job)
            if placement.where == "remote":
                self.cluster.dispatch(job, placement.node)

    def _disk_problem(self):
        """A sentence when the disk holding the queue and the checkpoints cannot fit a run, else ''. A failed job names the
        cause; a gate would hold the queue forever with no job to blame."""
        try:
            free, need = shutil.disk_usage(self.path.parent).free, disk_need(self.path.parent.parent)
        except OSError:
            return ""  # cannot tell; the job runs and its own failure is explained
        if free >= need:
            return ""
        return (f"Not enough disk space for this run: {size_text(free)} free, about {size_text(need)} needed. "
                "Free space in Resources › Storage.")

    def _hook(self, hook, job):
        if hook is None:
            return
        try:
            hook(job)
        except Exception:  # noqa: BLE001 - chat moving between devices must never stop training
            logging.getLogger("groundwork.queue").exception("a queue hook failed")

    def _share(self, job):
        """The share of the device's memory the job may take: the plan's, less what chat's models hold there. A job with less than
        one chat model's bytes to spare first has chat evict its oldest. Only a pretrain job sizes anything by the share."""
        share = MEMORY[self.plan.memory]
        if self.ledger is None or job.kind != "pretrain":
            return share
        import studio_memory
        from studio_model import validate_config
        try:
            device = studio_memory.device_of(validate_config(job.params["config"]))
            total = studio_memory.budget(device, 1.)
        except Exception:  # noqa: BLE001 - an invalid config or a missing GPU fails the job in its worker, with the reason
            return share
        held = self.ledger.total(device)
        while self.make_room and share * total - held < self.ledger.largest(device) and self.make_room():
            held, before = self.ledger.total(device), held
            if held == before:
                break
        return max(0., share - held / total)

    def follow_up(self, job):
        """Queue what a job that ended leaves to do (the caller holds the lock and saves): after a pretrain on new text, the
        restore_answers round that teaches it its old manner again. The cluster calls it for a pretrain that ran on a peer, once its
        version is here."""
        if job.status == "done" and job.kind == "pretrain" and job.params.get("user_text"):
            self.jobs.append(Job("restore_answers", "Restore how it answers", params={"round": 3, "teacher": job.params.get("previous_best")},
                                 note="added automatically after new text; a short lesson from the previous version restores how it answers"))

    def _resume_note(self, job, event):
        """Say on the job which way a restart went: on from its own checkpoint, or from step 0 because training refused the file."""
        if event["type"] == "resumed":
            text = (f"The app stopped while this ran; it continued from its own checkpoint at step "
                    f"{event.get('base', 0) + event.get('step', 0)}, with the optimizer state it had.")
        else:
            text = (f"The app stopped while this ran; it could not continue from its own checkpoint ({event.get('reason') or 'unusable'}), "
                    "so it started again from step 0.")
        with self.lock:
            job.note = f"{job.note} {text}".strip()
            self._save()

    def _run(self, job):
        handler = self.handlers.get(job.kind)
        short = self._disk_problem()
        self._hook(self.on_start, job)  # chat leaves the GPU before the job is given a share of it
        share = self._share(job)  # before the lock: evicting waits on chat's own lock
        with self.lock:
            job.status, self.running, self.progress, self.stopping = "running", job, {}, False
            job.params = {**job.params, "memory_share": share}  # studio_memory.batch_cap fits the job's batch to it
            self.stop_event.clear()
            self._save()
        started = time.monotonic()

        resumed_at = [0]  # the step this run began at (a resume), so the time per step ignores the steps an earlier run did

        def emit(event):
            if event.get("type") in ("resumed", "resume_refused"):
                self._resume_note(job, event)
            if event.get("total"):
                step, total = event.get("step", 0), event["total"]
                if event.get("type") == "resumed":  # progress starts from the checkpoint's step, not 0
                    step = resumed_at[0] = event.get("base", 0) + step
                self.progress.update(step=step, total=total, lr=event.get("lr", self.progress.get("lr")),
                                     phase=event.get("phase", self.progress.get("phase", "")))
                if step > resumed_at[0]:
                    self.progress["eta_s"] = (time.monotonic() - started) / (step - resumed_at[0]) * (total - step)
            self._emit("train", {"job": job.id, **event})

        awake = platform_helpers.keep_awake(os.getpid()) if self.plan.keep_awake else None
        error, handle = short, None
        if not short:
            try:
                handle = self._start(job, handler, emit, self.wake.set)
                with self.lock:
                    self._save()  # prepare filled in the job's output folder and resume point: a restart now finds them
            except Exception as exc:  # noqa: BLE001 - prepare or the spawn failed: the job fails with the reason
                error = str(exc) or type(exc).__name__
        self._emit("job", {"job": job.id, "status": "running"})
        while handle and not handle.done.is_set():
            self.wake.clear()
            self._watch()
            self._place_waiting()
            handle.sync(self.stop_event, self.pause_event)  # whatever set them also set wake
            if not handle.done.is_set():
                self.wake.wait(self.tick)
        platform_helpers.release_awake(awake)
        with self.lock:
            if handle:
                job.params.update(handle.params)
                error = handle.error
            job.status = "failed" if error else "stopped" if self.stop_event.is_set() else "done"
            job.error = error
            self.running = None
            self.stopping = False
            self.stopping_job = ""
            self.follow_up(job)
            if self.closed.is_set() and job.status == "stopped":
                job.status = "waiting"  # shut down mid-run: try again next start
            self._save()
        if not self.closed.is_set():
            self._hook(self.on_end, job)
        self._emit("job", {"job": job.id, "status": job.status, "error": job.error})
