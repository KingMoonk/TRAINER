"""Long reads as jobs: POST returns an id, progress and the result arrive on /events, GET /jobs/{id} for late joiners."""

import dataclasses
import json
import queue
import threading
import time
import uuid

from fastapi import APIRouter, HTTPException, Request

from studio_api.events import encode

PROGRESS_EVERY = 0.25


@dataclasses.dataclass
class JobRecord:
    id: str
    kind: str
    status: str = "queued"  # queued, running, done, failed, cancelled
    progress: str = ""
    fraction: float | None = None
    result: object = None
    error: str = ""
    created_at: float = dataclasses.field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None


class Jobs:
    """One worker thread runs submitted callables in order; every state change is emitted as a "job" event."""

    def __init__(self, services):
        self.services = services
        self._records = {}
        self._stops = {}
        self._lock = threading.Lock()
        self._queue = queue.SimpleQueue()
        self._reads = queue.SimpleQueue()  # slow read-only scans (the data page's summaries) wait here, so they never hold the main worker
        self._keys = {}
        self._worker = threading.Thread(target=self._run, args=(self._queue,), name="api-jobs", daemon=True)
        self._worker.start()
        threading.Thread(target=self._run, args=(self._reads,), name="api-reads", daemon=True).start()

    def submit(self, kind, work, own_thread=False, lane="main", key=None):
        """`work(stop)` runs on the worker thread with a threading.Event it may poll; its return is the result. `own_thread` starts it
        at once on a thread of its own instead, for work that must not wait behind a slow read (the size of the chosen text). `lane`
        "reads" queues it on the reads worker, apart from the main one. `key` shares one unfinished job between identical requests:
        asking again while it is queued or running returns it instead of starting a second scan."""
        with self._lock:
            held = self._records.get(self._keys.get(key)) if key is not None else None
            if held is not None and held.status in ("queued", "running"):
                return held
            record = JobRecord(id=uuid.uuid4().hex[:8], kind=kind)
            self._records[record.id] = record
            self._stops[record.id] = threading.Event()
            if key is not None:
                self._keys[key] = record.id
        self._announce(record)
        if own_thread:
            threading.Thread(target=self._execute, args=(record.id, work, False), name=f"api-job-{kind}", daemon=True).start()
        else:
            (self._reads if lane == "reads" else self._queue).put((record.id, work))
        return record

    def get(self, job_id):
        return self._records.get(job_id)

    def all(self):
        return sorted(self._records.values(), key=lambda r: r.created_at)

    def cancel(self, job_id):
        record = self._records.get(job_id)
        if record is None:
            return None
        self._stops[job_id].set()
        if record.status == "queued":
            self._finish(record, "cancelled")
        return record

    def close(self):
        for stop in self._stops.values():
            stop.set()
        self._queue.put(None)
        self._reads.put(None)

    def _announce(self, record):
        self.services.emit("job", dataclasses.asdict(record))

    def _finish(self, record, status, result=None, error=""):
        record.status, record.result, record.error = status, result, error
        record.finished_at = time.time()
        self._announce(record)

    def _run(self, lane):
        while True:
            item = lane.get()
            if item is None:
                return
            self._execute(*item, shared=True)

    def _execute(self, job_id, work, shared):
        record, stop = self._records[job_id], self._stops[job_id]
        if record.status != "queued":
            return
        record.status, record.started_at = "running", time.time()
        self._announce(record)
        done = threading.Event()
        if shared:
            self.services.loading = ("", None)  # a note from an earlier read is not this job's
            threading.Thread(target=self._progress, args=(record, done), daemon=True).start()
        try:  # an exception with plain = True is already a sentence for people: the job's error is that sentence
            result = work(stop)
        except Exception as exc:  # noqa: BLE001 - the job's failure is its result
            done.set()
            self._finish(record, "failed", error=str(exc) if getattr(exc, "plain", False) else f"{type(exc).__name__}: {exc}")
        else:
            done.set()
            text, fraction = self.services.loading if shared else ("", None)
            if text:  # the poll may not have seen the last note (a short job): the finished record keeps it
                record.progress, record.fraction = text, fraction
            self._finish(record, "cancelled" if stop.is_set() else "done", result=result)

    def _progress(self, record, done):
        last = None
        while not done.wait(PROGRESS_EVERY):
            text, fraction = self.services.loading
            if (text, fraction) != last and record.status == "running":
                last = (text, fraction)
                record.progress, record.fraction = text, fraction
                self._announce(record)


router = APIRouter(prefix="/jobs", tags=["jobs"])


def _view(record):
    return json.loads(encode(dataclasses.asdict(record)))


@router.get("")
def list_jobs(request: Request):
    return [_view(r) for r in request.app.state.jobs.all()]


@router.get("/{job_id}")
def get_job(job_id: str, request: Request):
    record = request.app.state.jobs.get(job_id)
    if record is None:
        raise HTTPException(404, f"No job {job_id}")
    return _view(record)


@router.delete("/{job_id}")
def cancel_job(job_id: str, request: Request):
    record = request.app.state.jobs.cancel(job_id)
    if record is None:
        raise HTTPException(404, f"No job {job_id}")
    return _view(record)
