"""The busy registry: heavy operations announce themselves so the person sees something is happening.

    with services.busy("Loading model tiny-v3", total=2) as task:
        read()
        task.step()
        build()
        task.step()

`total=None` means the work has no measurable steps (an indeterminate bar). Every change is emitted as a "busy" event whose
payload is the whole list, {"items": [{"id", "label", "done", "total"}]}, so a client that missed one still ends up right.
Safe from any thread.
"""

import itertools
import threading
import time

PROGRESS_EVERY = 0.1  # seconds between "busy" events for steps alone; a start and an end are always sent


class Task:
    def __init__(self, registry, id, label, total):
        self._registry, self.id, self.label, self.total, self.done = registry, id, label, total, 0
        self._sent = 0.

    def step(self, count=1):
        """One more unit of the work is finished."""
        self.set(self.done + count)

    def set(self, done, total=None, label=None):
        """Report `done` units finished; `total` and `label` replace the earlier ones when given."""
        self._registry._update(self, done, total, label)

    def view(self):
        return {"id": self.id, "label": self.label, "done": self.done, "total": self.total}


class Busy:
    def __init__(self, emit):
        self._emit, self._lock, self._ids = emit, threading.Lock(), itertools.count(1)
        self._tasks = {}

    def __call__(self, label, total=None):
        return _Scope(self, label, total)

    def items(self):
        with self._lock:
            return [task.view() for task in self._tasks.values()]

    def _publish(self):
        self._emit("busy", {"items": self.items()})

    def _start(self, label, total):
        with self._lock:
            task = Task(self, f"busy-{next(self._ids)}", label, total)
            self._tasks[task.id] = task
        self._publish()
        return task

    def _update(self, task, done, total, label):
        with self._lock:
            task.done = done
            if total is not None:
                task.total = total
            if label is not None:
                task.label = label
            now = time.monotonic()
            final = task.total is not None and task.done >= task.total
            if not final and now - task._sent < PROGRESS_EVERY:
                return
            task._sent = now
        self._publish()

    def _end(self, task):
        with self._lock:
            self._tasks.pop(task.id, None)
        self._publish()


class _Scope:
    def __init__(self, registry, label, total):
        self._registry, self._label, self._total, self._task = registry, label, total, None

    def __enter__(self):
        self._task = self._registry._start(self._label, self._total)
        return self._task

    def __exit__(self, *exc):
        self._registry._end(self._task)
        return False
