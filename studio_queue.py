"""One machine, one queue: a single runner executes one job at a time and persists data/queue.json."""
import dataclasses
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import studio_platform as platform_helpers  # battery + keep-awake live there; never duplicated
from studio_core import atomic_json, read_json
from studio_failures import explain
import studio_run_watch
from studio_timing import timed

KINDS = ("pretrain", "quicktest", "compare", "post_round", "restore_answers", "checkpoint_now", "import_pretrained",
         "pretrain_distributed", "diloco_member", "rlhf_loop", "mlperf")
MEMORY = {"half": .5, "most": .75, "all": 1.}
QUEUED = ("waiting", "held")  # jobs listed under Up next, in order; only "waiting" ones are picked to run
SIZING_PERCENT = "sizing_percent"  # with SIZING: how far the size of the text is worked out, in percent (set in memory; not saved on its own)
SCALED_NOTE = "Scaled to this computer:"  # starts the part of a pretrain job's note the queue writes from studio_fit; rewritten at each start
SIZING = "sizing"  # a param of a job added before the size of its text was known: it keeps its place but is not run until fill() gives its steps
FAILED_SHOWN = 5
MIN_FREE = 2 * 2 ** 30  # a job needs at least this much free disk to start


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
    error: str = ""                 # a failed job's reason as one sentence (studio_failures)
    details: str = ""               # the raw text behind it, "" when the sentence is the error's own text
    node: str = ""                  # the peer it runs (or ran) on; "" for a job that runs here
    epoch: int = 0                  # the lease epoch it was last placed at (studio_scheduler); a write carrying another epoch is refused
    owner: str = ""                 # for a job another node owns and this one runs: how to reach that node


@dataclass(frozen=True)
class Gate:
    """Whether the plan lets the machine train right now. Frozen, so two readings compare with == (the runner announces a change)."""
    reason: str | None = None            # "outside hours" | "on battery" | "computer in use" | None (may train)
    resumes_at: datetime | None = None   # for "outside hours": when the next window opens; None for every other reason
    supported: bool = True               # False when idle_only is on but this computer cannot report idle time (the plan then ignores it)


@dataclass
class QueueState:
    job: Job | None = None         # the running job, None when idle
    step: int = 0
    total: int = 0
    phase: str = ""                # warm-up | steady | cool-down, or "preparing" while the text is tokenized
    prepare: int | None = None     # percent of the text prepared, while phase == "preparing"
    prepare_phase: str | None = None  # what the preparation is doing: reading | tokenizer | tokenizing
    lr: float | None = None
    eta_s: float | None = None
    paused: bool = False           # by the user; persisted, so true when idle too ("Paused by you")
    blocked: str | None = None     # == gate.reason, kept for callers that only need the words
    gate: Gate = field(default_factory=Gate)
    stopping: bool = False         # stop_after_cooldown in progress
    up_next: list = field(default_factory=list)
    plan: dict = field(default_factory=dict)  # asdict(Plan)
    failed: list = field(default_factory=list)  # jobs that ended failed (the last FAILED_SHOWN), with their sentence in .error; removing one dismisses it
    notices: list = field(default_factory=list)  # what the running job reports while it goes (studio_run_watch): {kind, text, at}


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


def ignores_plan(job):
    """True when `job` carries `ignore_plan` in its params: the queue's plan (hours, battery, idle) does not hold it. Only the cluster
    self-test sets it, on its own run and the member jobs it invites, so a person at the keyboard can prove the cluster."""
    return bool(job is not None and job.params.get("ignore_plan"))


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
                 cluster=None, profile=None):
        self.path, self.handlers, self.prepare, self.tick, self.clock = Path(path), handlers, prepare, tick, clock
        self.ledger, self.make_room, self.on_start, self.on_end = ledger, make_room, on_start, on_end
        self.profile = profile  # () -> studio_fit.Profile of this machine as it is now; a pretrain job is fitted to it just before it starts. None: not fitted
        self.root = self.path.parent.parent
        self.handles = set()           # live workers: the running job's and any cool-down branched off it
        self.lock = threading.RLock()
        self.subscribers = []
        self.jobs, self.plan = [], Plan()
        self.ended_on_load = []        # jobs the restart ended (a distributed run that cannot resume), for whoever keeps records of them
        self.user_paused = False       # a user intent: restored from queue.json, so a pause survives a restart
        self.stopping_job = ""         # the id of a running job the person stopped: persisted, so a restart does not run it again
        self.reap_grace = 35.          # seconds a worker an earlier app left gets to leave by itself (studio_worker.EXIT_GRACE and a margin)
        self.kill_grace = 8.           # seconds a worker gets to stop after "Stop now" (a step and a save) before its process is ended
        self._kill_at = None           # monotonic time "Stop now" ends the running job's process; None when no immediate stop is pending
        self.dead_runs = []            # jobs whose process an earlier app left running: their run.json is reconciled once it is stopped
        self._load()
        self.running = None            # Job on the runner thread
        self.yielding = None           # (id of the running job being stepped aside, the status it goes back to) until it has stopped
        self.progress = {}
        self.notices = []              # the running job's live notices (studio_run_watch), cleared when a job starts and ends
        self._noted, self._read_at, self._pace_at = set(), 0., 0.  # kinds of notice already written into the job's note; when the machine and the pace were last read
        self.watch_every = studio_run_watch.WATCH_EVERY  # seconds between two hardware readings of the running job
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
        # Once start() runs, queue.json is written on a thread of its own, off self.lock: on Windows a save can take seconds (fsync, and
        # replace_file waiting out a scanner's lock), and every reader of the queue (GET /queue, /queue/jobs) waited on the lock meanwhile.
        self._written = threading.Condition()
        self._unsaved, self._writing, self._writer = None, False, None
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
            if job.status in QUEUED and job.params.get(SIZING):  # what was working out its size went with the app
                job.status, job.error = "failed", "The app stopped while the size of this run's text was worked out. Add the run again."
                continue
            if job.status == "running" and not job.owner and not job.node:
                self.dead_runs.append(job)
            if job.status == "running" and job.id == self.stopping_job and not job.owner and not job.node:
                job.status, job.error = "stopped", ""  # the person stopped it; the app went before it ended
            elif job.owner and job.status in ("running", "waiting", "held"):  # another node's job cannot renew its lease now: its owner places it again
                job.status, job.error = "stopped", "This machine restarted while it ran the job for another machine, which will place it again."
            elif job.status == "running" and job.kind == "pretrain_distributed":  # its coordinator went with this process
                import studio_distributed
                if getattr(studio_distributed.RUNS.get(job.id), "stopped", False):  # the person stopped the run; the app went before its job ended
                    job.status = "stopped"
                    continue
                job.error = studio_distributed.recover(job)
                job.status = "failed" if job.error else "waiting"  # resumes from its last completed round, or ends saying why
                if job.error:
                    self.ended_on_load.append(job)
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
        state = {"plan": dataclasses.asdict(self.plan), "user_paused": self.user_paused, "stopping": self.stopping_job,
                 "jobs": [dataclasses.asdict(j) for j in self.jobs]}
        if self._writer is None:
            atomic_json(self.path, state)
            return
        with self._written:
            self._unsaved = state  # a later save replaces one not yet written: only the latest state reaches the file
            self._written.notify_all()

    def _write_loop(self):
        while True:
            with self._written:
                self._written.wait_for(lambda: self._unsaved is not None)
                state, self._unsaved, self._writing = self._unsaved, None, True
            try:
                atomic_json(self.path, state)
            except OSError:
                logging.getLogger("groundwork.queue").exception("could not save the queue to %s", self.path)
            finally:
                with self._written:
                    self._writing = False
                    self._written.notify_all()

    def flush(self, timeout=None):
        """Wait until every save so far is in queue.json; True when it is."""
        with self._written:
            return self._written.wait_for(lambda: self._unsaved is None and not self._writing, timeout)

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
        """Add `job` to the end of the queue and return it; a job whose id is already in the queue (any status) is not added again, and
        the one there is returned: a caller that chose the id can ask again after a timeout without making two."""
        if isinstance(job, str):
            job = Job(kind=job)
        if job.kind not in KINDS:
            raise ValueError(f"Unknown job kind: {job.kind}")
        with self.lock:
            there = next((j for j in self.jobs if j.id == job.id), None)
            if there is not None:
                return there
            self.jobs.append(job)
            self._save()
        self.wake.set()
        self._emit("queue", {"job": job.id})
        return job

    def fill(self, job_id, label, params):
        """Give a job added while the size of its text was worked out (params[SIZING]) its label and params, so it may run. False when
        it is no longer waiting for them: removed meanwhile. A pin set on it in the meantime is kept."""
        with self.lock:
            job = next((j for j in self._queued() if j.id == job_id and j.params.get(SIZING)), None)
            if job is not None:
                pinned = {"pinned": job.params["pinned"]} if job.params.get("pinned") else {}
                job.label, job.params = label, {**{k: v for k, v in params.items() if k != SIZING}, **pinned}
                self._save()
        if job is None:
            return False
        self.wake.set()
        self._emit("queue", {"job": job_id})
        return True

    def size_progress(self, job_id, percent):
        """Note how far the size of the text is worked out (0-100) for a job still waiting for it (params[SIZING]), and announce it when
        the percent moved. Set on the job without a save, since a restart fails a sizing job anyway. False when it is gone."""
        with self.lock:
            job = next((j for j in self._queued() if j.id == job_id and j.params.get(SIZING)), None)
            if job is None:
                return False
            percent = max(0, min(100, int(percent)))
            if job.params.get(SIZING_PERCENT) == percent:
                return True
            job.params[SIZING_PERCENT] = percent
        self._emit("queue", {"job": job_id})
        return True

    def fail(self, job_id, error):
        """End a job still waiting for its size (params[SIZING]) as failed, with `error` as its reason."""
        with self.lock:
            job = next((j for j in self._queued() if j.id == job_id and j.params.get(SIZING)), None)
            if job is not None:
                job.status, job.error = "failed", error
                self._save()
        if job is not None:
            self._emit("queue", {"job": job_id})
            self._emit("job", {"job": job_id, "status": "failed", "error": error})

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
                self._kill_at = time.monotonic() + self.kill_grace  # only this stop is immediate: a worker that has not left by then is ended
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

    def yield_running(self, to=None):
        """Step the running job aside so the waiting job `to` runs next: the running job is stopped and goes back to waiting behind it,
        and continues from its own checkpoint when its turn comes (studio_jobs.continue_here). With no `to` it is held instead, until the
        person releases it. False when nothing runs, or `to` is not waiting."""
        with self.lock:
            running, ahead = self.running, next((j for j in self._queued() if j.id == to), None)
            if running is None or (to is not None and ahead is None):
                return False
            if ahead is not None:
                self.jobs.remove(ahead)
                self.jobs.insert(self.jobs.index(running), ahead)
                ahead.status = "waiting"
            self.yielding = (running.id, "waiting" if ahead is not None else "held")
            self.stop_event.set()
            self._save()
        self.wake.set()
        self._emit("queue", {"job": running.id})
        return True

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
        run = main.params.get("output_dir")
        if run and not (Path(run) / "last.npz").is_file():  # nothing saved yet to cool down from
            if not then_stop:
                self._emit("checkpoint", {"job": main.id, "phase": "failed", "error": "The run has not saved a checkpoint yet."})
            return False
        child = Job("checkpoint_now", f"Checkpoint of {main.label or main.kind}", params={**main.params, "parent": main.id})

        def work():
            self._emit("checkpoint", {"job": main.id, "phase": "start"})
            try:
                handle = self._start(child, handler, lambda e: self._emit("train", {"job": child.id, **e}))
                handle.done.wait()
                error = handle.error
            except Exception as exc:  # noqa: BLE001
                error = explain(exc)[0]
            self._emit("checkpoint", {"job": main.id, **({"phase": "failed", "error": error} if error else {"phase": "done"})})
            if then_stop:  # a failed cool-down still stops: the person asked for the run to end
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
                job=self.running, step=p.get("step", 0), total=p.get("total", 0), phase=p.get("phase", ""), prepare=p.get("prepare"), prepare_phase=p.get("prepare_phase"), lr=p.get("lr"),
                eta_s=p.get("eta_s"), paused=self.user_paused, blocked=gate.reason, gate=gate, stopping=self.stopping,
                up_next=self._queued(), plan=dataclasses.asdict(self.plan), notices=[dict(n) for n in self.notices],
                failed=[j for j in self.jobs if j.status == "failed"][-FAILED_SHOWN:])

    def start(self):
        self._writer = threading.Thread(target=self._write_loop, daemon=True, name="queue-save")
        self._writer.start()
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
        self.flush(timeout)

    # ---- runner thread
    def _sync_pause(self):
        held = self.blocked and not ignores_plan(self.running)
        (self.pause_event.set if self.user_paused or held else self.pause_event.clear)()

    @timed("queue._gate")
    def _gate(self):
        """The plan's verdict right now. Idle detection that this computer cannot do is reported (supported=False) and the
        idle_only setting is then ignored, rather than silently passing as if the computer were idle."""
        plan = self.plan
        idle = idle_seconds() if plan.idle_only else None
        supported = not (plan.idle_only and idle is None)
        now = self.clock()
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
            if not job or (self.blocked and not ignores_plan(job)) or self.user_paused:
                self.wake.wait(self.tick)
                continue
            self._run(job)

    def _next_local(self):
        """The first waiting job that belongs on this machine: a job pinned to, or better placed on, a peer is skipped, not waited for, and
        so is one still waiting for the size of its text."""
        with self.lock:
            waiting = [j for j in self.jobs if j.status == "waiting" and not j.params.get(SIZING)]
        return next((j for j in waiting if self.cluster.place(j).where == "local"), None)

    def _place_waiting(self):
        """Hand each waiting job that belongs on a peer to it. Runs between jobs and while one runs, so a free peer is used at once. The
        plan's hours still apply (they say when to train) and so does a pause; this machine's battery and keyboard do not, because they
        say nothing about the peer."""
        if self.user_paused or self.blocked == "outside hours":
            return
        with self.lock:
            waiting = [j for j in self.jobs if j.status == "waiting" and not j.params.get(SIZING)]
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

    def _fitted(self, job, share):
        """(params, note, problem): a pretrain job's params fitted to this machine as it is now (studio_fit.refit: the free GPU memory, RAM and
        disk are read again, since another app or an earlier run may hold more than when the plan was made), the job's note with what
        that scaled and why, and a sentence when even the smallest scaled run cannot fit (the job then fails with it before a worker
        starts). A job that is not a fresh pretrain run, or a queue without a profile, is left as it is."""
        params, note = job.params, job.note.split(SCALED_NOTE)[0].strip()
        if self.profile is None or job.kind != "pretrain":
            return params, job.note, ""
        import studio_fit
        try:
            found = studio_fit.refit(params, self.profile(), share, remote=bool(job.owner))
        except Exception:  # noqa: BLE001 - a machine or a config that cannot be read is run as planned; the worker's own checks still apply
            logging.getLogger("groundwork.queue").exception("fitting a job to this machine failed")
            return params, job.note, ""
        if found is None:
            return params, job.note, ""
        if found.refusal:
            return params, note, found.refusal
        params = studio_fit.applied(params, found)
        said = " ".join(n.text for n in found.notes)
        return params, f"{note} {SCALED_NOTE} {said}".strip() if said else note, ""

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

    def _notify(self, job, found, accumulate=False):
        """Show `found` (a studio_run_watch notice) on the running job: in the state's notices, and once per kind in the job's note, which
        outlives the run. A notice of the same kind replaces the last one, unless `accumulate` (each halving is its own event)."""
        with self.lock:
            if self.running is not job:
                return
            kept = [n for n in self.notices if accumulate or n["kind"] != found["kind"]]
            self.notices = [*kept, found][-studio_run_watch.OOM_SHOWN:]
            if accumulate or found["kind"] not in self._noted:
                self._noted.add(found["kind"])
                job.note = f"{job.note} {found['text']}".strip()
                self._save()
        self._emit("queue", {"job": job.id})

    def _clear(self, job, kind):
        """Drop the notice of `kind` once what it said no longer holds (the job's note keeps what was said)."""
        with self.lock:
            if self.running is not job or not any(n["kind"] == kind for n in self.notices):
                return
            self.notices = [n for n in self.notices if n["kind"] != kind]
        self._emit("queue", {"job": job.id})

    def _watch_event(self, job, event, first_step):
        """What a running pretrain job's progress events say: a batch halved after running out of memory, and where its pace puts its end."""
        if job.kind != "pretrain":
            return
        found = studio_run_watch.oom_notice(event)
        if found:
            self._notify(job, found, accumulate=True)
        if event.get("type") != "metric" or not event.get("elapsed_seconds") or time.monotonic() - self._pace_at < self.watch_every / 3:
            return
        self._pace_at = time.monotonic()
        found = studio_run_watch.projection(
            step=event.get("step", 0), first_step=first_step, total=event.get("total", 0), trained_tokens=event.get("trained_tokens", 0),
            elapsed=event["elapsed_seconds"], token_budget=job.params.get("token_budget"), window_seconds=studio_run_watch.window_seconds(job.params))
        if found:
            self._notify(job, found)
        else:
            self._clear(job, "overrun")

    def _watch_machine(self, job):
        """Read the machine again (every `watch_every` seconds) and say on the running job when its free disk or RAM has fallen below what
        the run needs. Nothing is stopped: the run goes on and the person decides."""
        if self.profile is None or job.kind != "pretrain" or time.monotonic() - self._read_at < self.watch_every:
            return
        self._read_at = time.monotonic()
        try:
            need = studio_run_watch.needs(job.params)
            found = studio_run_watch.resources(self.profile(), need) if need else []
        except Exception:  # noqa: BLE001 - a machine that cannot be read is no news
            logging.getLogger("groundwork.queue").exception("re-reading the hardware of a running job failed")
            return
        for kind in ("disk", "ram"):
            mine = next((n for n in found if n["kind"] == kind), None)
            if mine:
                self._notify(job, mine)
            else:
                self._clear(job, kind)

    def _run(self, job):
        handler = self.handlers.get(job.kind)
        short = self._disk_problem()
        self._hook(self.on_start, job)  # chat leaves the GPU before the job is given a share of it
        share = self._share(job)  # before the lock: evicting waits on chat's own lock
        params, note, refused = self._fitted(job, share)  # after chat made room: the free memory it reads is what the job really gets
        short = short or refused
        with self.lock:
            job.status, self.running, self.progress, self.stopping = "running", job, {}, False
            self._sync_pause()  # a job that ignores the plan is not held by the gate that stood when it started
            job.params = {**params, "memory_share": share}  # studio_memory.batch_cap fits the job's batch to it
            job.note = note
            self.notices, self._noted, self._read_at, self._pace_at = [], set(), time.monotonic(), 0.
            self.stop_event.clear()
            self._kill_at = None
            self._save()
        started = time.monotonic()

        resumed_at = [0]  # the step this run began at (a resume), so the time per step ignores the steps an earlier run did

        def emit(event):
            if event.get("type") in ("resumed", "resume_refused"):
                self._resume_note(job, event)
            self._watch_event(job, event, resumed_at[0])
            if event.get("type") == "prepare":  # the text is being tokenized: no step yet, only how far along it is
                self.progress.update(phase="preparing", prepare=event.get("percent", 0), prepare_phase=event.get("phase"))
            elif event.get("total"):
                step, total = event.get("step", 0), event["total"]
                if event.get("type") == "resumed":  # progress starts from the checkpoint's step, not 0
                    step = resumed_at[0] = event.get("base", 0) + step
                before = self.progress.get("phase", "")
                self.progress.pop("prepare", None)  # the first step ends the preparing phase
                self.progress.pop("prepare_phase", None)
                self.progress.update(step=step, total=total, lr=event.get("lr", self.progress.get("lr")),
                                     phase=event.get("phase", "" if before == "preparing" else before))
                if step > resumed_at[0]:
                    self.progress["eta_s"] = (time.monotonic() - started) / (step - resumed_at[0]) * (total - step)
            self._emit("train", {"job": job.id, **event})

        awake = platform_helpers.keep_awake(os.getpid()) if self.plan.keep_awake else None
        error, details, handle = short, "", None
        if not short:
            try:
                handle = self._start(job, handler, emit, self.wake.set)
                with self.lock:
                    self._save()  # prepare filled in the job's output folder and resume point: a restart now finds them
            except Exception as exc:  # noqa: BLE001 - prepare or the spawn failed: the job fails with the reason
                error, details = explain(exc)
        self._emit("job", {"job": job.id, "status": "running"})
        while handle and not handle.done.is_set():
            self.wake.clear()
            self._watch()
            self._place_waiting()
            self._watch_machine(job)
            handle.sync(self.stop_event, self.pause_event)  # whatever set them also set wake
            wait = self.tick
            if self._kill_at is not None and not handle.done.is_set():
                left = self._kill_at - time.monotonic()
                if left <= 0:  # "Stop now" was not answered within the grace: end the process (the job still ends as stopped)
                    self._kill_at = None
                    handle.kill()
                else:
                    wait = min(wait, left)
            if not handle.done.is_set():
                self.wake.wait(wait)
        platform_helpers.release_awake(awake)
        with self.lock:
            if handle:
                job.params.update(handle.params)
                error, details = handle.error, handle.details
            error = error or job.params.pop("ended_reason", "")  # a watchdog ended the job (studio_cluster_watch): its sentence is the error
            status = "failed" if error else "stopped" if self.stop_event.is_set() else "done"
            if self.yielding and self.yielding[0] == job.id and status == "stopped":
                status = self.yielding[1]  # stepped aside for another job: back in the queue, not in the history
            self.yielding = None
            job.error, job.details = error, details if error else ""
            job.status = status  # last: a reader that waits for the status without the lock then finds the error and details in place
            self.running = None
            self.notices = []
            self._sync_pause()
            self.stopping = False
            self.stopping_job = ""
            self._kill_at = None
            self.follow_up(job)
            if self.closed.is_set() and job.status == "stopped":
                job.status = "waiting"  # shut down mid-run: try again next start
            self._save()
        if not self.closed.is_set():
            self._hook(self.on_end, job)
        self._emit("job", {"job": job.id, "status": job.status, "error": job.error})
