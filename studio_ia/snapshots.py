"""Snapshots, commands and the render scheduler: how a page gets its state without doing slow work on the Tk thread (#82).

* A snapshot is the latest state of one topic ("queue", "versions", ...), computed on a worker by the topic's producer and
  published with a version number that grows by one per publish. The Tk thread reads it with get(): a dict lookup, no lock,
  no I/O. Computing one on the Tk thread raises under tests (studio_ui_guard).
* The events a topic depends on (TOPICS) make its worker compute it again. Requests that arrive while one is waiting are
  merged into it. Each publish emits ("snapshot", topic).
* A command is a UI action run on the command worker. When it is done, the topics it changes are computed again and
  ("command", {name, ok, error}) is emitted. The UI never waits for it: it sees the outcome in the new snapshots.
* Render collects the topics the "snapshot" events name and calls each subscriber at most once per frame, at most
  about 30 frames a second.
"""
from __future__ import annotations

import collections
import copy
import dataclasses
import logging
import threading
import time
from typing import Any, Callable

import studio_ui_guard as ui_guard
from studio_ia import threads

_log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class Snapshot:
    topic: str
    version: int  # 1 for the first publish of a topic, then one more per publish
    value: Any  # treat as read-only: the next publish replaces it, never changes it


@dataclasses.dataclass(frozen=True)
class Versions:
    """The "versions" topic: every version, the newest, and the one Chat follows (services.best_version)."""
    items: tuple = ()
    current: Any = None
    best: Any = None

    @property
    def chat(self):
        return self.best or self.current


def queue_snapshot(services):
    """A copy of services.queue(), so a snapshot does not change under its reader: FakeServices changes its QueueState in
    place, and the runner thread changes its Jobs. Shallow per Job: their params (which can hold a large result) are shared."""
    q = services.queue()
    return dataclasses.replace(q, job=copy.copy(q.job), up_next=[copy.copy(j) for j in q.up_next], plan=dict(q.plan))


def versions_snapshot(services):
    """best_version() can score held-back text, which takes seconds the first time: one reason this runs on a worker."""
    try:
        return Versions(tuple(services.versions()), services.current_version(), services.best_version())
    except NotImplementedError:  # a Services with no versions behind it
        return Versions()


@dataclasses.dataclass(frozen=True)
class Topic:
    make: Callable  # make(services) -> the snapshot's value; runs on a worker
    events: tuple  # the event kinds after which it is computed again
    often: tuple = ()  # kinds that arrive several times a second: computed again at most once per `every` seconds for them
    every: float = 1.0  # each worker that finishes costs the Tk thread a full collection (threads.release), about 25 ms


TOPICS = {
    "queue": Topic(queue_snapshot, ("queue", "job", "blocked", "plan"), often=("train",)),
    "versions": Topic(versions_snapshot, ("versions", "job")),
}
OWN_EVENTS = ("snapshot", "command")  # what this module emits; the shell keeps them from the pages' on_event


class Worker:
    """Runs jobs in order on a thread started when the first job arrives, which ends once there are none left. submit() is safe
    from any thread. The thread holds the collector while it runs (threads.start) and the Tk thread's settle() gives it back."""

    def __init__(self):
        self._jobs, self._lock, self._running = collections.deque(), threading.Lock(), False
        self._idle, self.stopped = threading.Event(), False
        self._idle.set()

    def submit(self, job):
        with self._lock:
            if self.stopped:
                return
            self._jobs.append(job)
            self._idle.clear()
            if self._running:
                return
            self._running = True
        threads.start(self._drain)

    def _drain(self):
        while True:
            with self._lock:
                if not self._jobs or self.stopped:
                    self._jobs.clear()
                    self._running = False
                    self._idle.set()
                    return
                job = self._jobs.popleft()
            try:
                job()
            except Exception:
                _log.exception("a snapshot worker job failed")
            del job

    def stop(self):
        with self._lock:
            self.stopped = True
            if not self._running:
                self._jobs.clear()

    def wait(self, timeout=None):
        """Tests: True once no job is waiting or running."""
        return self._idle.wait(timeout)


class Snapshots:
    """Per-topic snapshots of one Services, and its command channel. Inert until start(): no threads, no subscription."""

    def __init__(self, services, topics=None, clock=time.monotonic):
        self.services, self.topics, self.clock = services, dict(TOPICS if topics is None else topics), clock
        self._latest: dict[str, Snapshot] = {}  # replaced one key at a time, so a reader on another thread needs no lock
        self._workers = {name: Worker() for name in self.topics}
        self._commands = Worker()
        self._locks = {name: threading.Lock() for name in self.topics}  # one compute of a topic at a time, and in order
        self._asked, self._asked_lock = set(), threading.Lock()  # topics with a compute waiting to start
        self._unsubscribe = None
        self._often_at, self._held_back = {}, set()  # Tk thread: when an `often` event last asked each topic; topics it held back

    # reads: any thread, no I/O -----------------------------------------------------------------------------------------------
    def get(self, topic, default=None):
        """The latest value of `topic`, or `default` before its first publish."""
        snap = self._latest.get(topic)
        return default if snap is None else snap.value

    def snapshot(self, topic):
        """The latest Snapshot of `topic` (with its version), or None."""
        return self._latest.get(topic)

    # writes: workers -------------------------------------------------------------------------------------------------------
    def start(self):
        """Tk thread, once the app runs: follow the services' events and compute every topic."""
        for worker in (*self._workers.values(), self._commands):
            worker.stopped = False  # a later app on the same services
        with self._asked_lock:
            self._asked.clear()  # what a stopped worker dropped
        if self._unsubscribe is None:
            self._unsubscribe = self.services.subscribe(self.on_event)
        self.refresh(*self.topics)

    def stop(self):
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        for worker in (*self._workers.values(), self._commands):
            worker.stop()

    def on_event(self, kind, payload):
        """Tk thread (the services' subscribers run there)."""
        if kind in OWN_EVENTS:
            return
        self.refresh(*(name for name, topic in self.topics.items() if kind in topic.events))
        for name, topic in self.topics.items():
            if kind in topic.often:
                self._held_back.add(name)
        self.tick()

    def tick(self):
        """Tk thread, every frame: ask the topics `often` events held back, once their `every` has passed."""
        now = self.clock()
        for name in list(self._held_back):
            last = self._often_at.get(name)
            if last is None or now - last >= self.topics[name].every:
                self._often_at[name] = now
                self._held_back.discard(name)
                self.refresh(name)

    def refresh(self, *topics):
        """Any thread: compute these topics again on their workers. A topic already waiting for its worker is not asked twice."""
        for name in topics:
            with self._asked_lock:
                if name in self._asked:
                    continue
                self._asked.add(name)
            self._workers[name].submit(lambda name=name: self._run(name))

    def _run(self, name):
        with self._asked_lock:
            self._asked.discard(name)  # a request from now on is for state newer than what this compute reads
        self.compute(name)

    def compute(self, name):
        """Worker only: run the topic's producer, publish what it returns and emit ("snapshot", name). On the Tk thread it raises
        under tests and is logged otherwise (studio_ui_guard). A producer that fails keeps the last snapshot."""
        ui_guard.check(f"computing the {name} snapshot")
        with self._locks[name]:
            try:
                value = self.topics[name].make(self.services)
            except Exception:
                _log.exception("computing the %s snapshot failed", name)
                return None
            return self.publish(name, value)

    def publish(self, name, value):
        last = self._latest.get(name)
        snap = Snapshot(name, last.version + 1 if last else 1, value)
        self._latest[name] = snap
        self.services.emit("snapshot", name)
        return snap

    # commands --------------------------------------------------------------------------------------------------------------
    def send(self, work, *args, topics=(), name=""):
        """Any thread, never waits: run work(*args) on the command worker, then compute `topics` again and emit
        ("command", {"name", "ok", "error"}). work runs off the Tk thread, so it must not touch or hold widgets."""
        def run():
            try:
                work(*args)
                error = None
            except Exception as e:
                _log.exception("command %s failed", name or getattr(work, "__name__", work))
                error = e
            for topic in topics:
                self.compute(topic)
            self.services.emit("command", {"name": name, "ok": error is None, "error": error})
        self._commands.submit(run)

    def wait(self, timeout=10):
        """Tests: True once every worker is idle."""
        deadline = time.monotonic() + timeout
        workers = (self._commands, *self._workers.values())
        while time.monotonic() < deadline:
            busy = [w for w in workers if not w.wait(0)]
            if not busy:
                return True
            busy[0].wait(min(.05, max(0.0, deadline - time.monotonic())))
        return all(w.wait(0) for w in workers)


class Render:
    """Tk thread. mark() the topics that changed; frame() calls each subscriber whose topics changed, once, with them, at most
    once per `interval` seconds. A frame that comes too soon keeps the topics for the next one."""

    def __init__(self, interval=1 / 30, clock=time.monotonic):
        self.interval, self.clock = interval, clock
        self._dirty, self._subscribers, self._last = set(), [], None

    def subscribe(self, topics, redraw):
        """redraw(changed topics) after a frame in which any of `topics` changed. Returns an unsubscribe function."""
        entry = (frozenset(topics), redraw)
        self._subscribers.append(entry)
        return lambda: self._subscribers.remove(entry) if entry in self._subscribers else None

    def mark(self, *topics):
        self._dirty.update(topics)

    def frame(self):
        """Redraw what changed; returns how many subscribers were called."""
        if not self._dirty:
            return 0
        now = self.clock()
        if self._last is not None and now - self._last < .9 * self.interval:  # after() ticks land a little early or late
            return 0
        self._last, dirty, self._dirty = now, self._dirty, set()
        called = 0
        for topics, redraw in list(self._subscribers):
            changed = topics & dirty
            if changed:
                called += 1
                try:
                    redraw(changed)
                except Exception:  # one page's error must not stop the others redrawing
                    _log.exception("redrawing %s failed", sorted(changed))
        return called
