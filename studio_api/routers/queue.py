"""The queue: what is running and waiting, the plan that gates it, and the controls (hold, move, remove, pause, checkpoint)."""

import time
import uuid
from time import sleep
from datetime import datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, computed_field

from studio_api.app import get_services
from studio_api.models_queue_data import Model
from studio_ia.services import Job, queued_outlook

router = APIRouter(prefix="/queue", tags=["queue"])
Services = Annotated[Any, Depends(get_services)]


class JobModel(Model):
    """studio_ia.services.Job (studio_queue.Job when the runner is real)."""

    id: str
    kind: str  # pretrain | quicktest | compare | post_round | restore_answers | checkpoint_now | import_pretrained
    label: str = ""
    note: str = ""
    params: dict = {}
    status: str = "waiting"  # waiting | held | running | done | failed | stopped
    eta_s: float | None = None
    created_at: float = 0.0
    error: str = ""  # a failed job's reason as one sentence saying what to do next
    details: str = ""  # the raw text behind it, shown on request; "" when the sentence is all there is
    node: str = ""  # the peer it runs or ran on; "" when it runs here
    epoch: int = 0  # the lease epoch it was last placed at
    owner: str = ""  # a job another node queued and this one runs: how to reach that node

    @computed_field  # percent of the text's size worked out while a queued job still waits for it (params.sizing); null for any other job
    @property
    def sizing(self) -> int | None:
        if self.status not in ("waiting", "held") or not self.params.get("sizing"):
            return None
        return int(self.params.get("sizing_percent") or 0)


class NewJob(BaseModel):
    """`params.pinned` (a machine's address or hostname) keeps the job on that machine: it runs there or waits, and is never moved.
    `id` is the caller's own id for the job, so a POST sent again after it timed out is added once: the job already queued under that id
    is answered instead (whatever its status). Absent, the id is made here."""

    id: str | None = Field(None, pattern=r"^[A-Za-z0-9_-]{1,64}$")
    kind: str
    label: str = ""
    note: str = ""
    params: dict = {}
    eta_s: float | None = None


class GateModel(Model):
    """studio_queue.Gate: whether the plan lets the machine train right now."""

    reason: str | None = None  # "outside hours" | "on battery" | "computer in use" | None
    resumes_at: datetime | None = None
    supported: bool = True


class PlanModel(BaseModel):
    """studio_queue.Plan, the dict set_plan takes."""

    hours: Literal["any", "nights", "set"] = "any"
    start_hour: int = Field(22, ge=0, le=23)
    end_hour: int = Field(7, ge=0, le=23)
    memory: Literal["half", "most", "all"] = "most"
    idle_only: bool = True
    idle_after_s: float = Field(60.0, ge=0)
    pause_on_battery: bool = True
    keep_awake: bool = False
    chat_follows_best: bool = True


class NoticeModel(Model):
    """One thing the running job reports while it goes (studio_run_watch): its batch halved after running out of memory, free disk or RAM
    below what it needs, or a projected end past the time window."""

    kind: Literal["oom", "disk", "ram", "overrun"]
    text: str
    at: float  # epoch seconds


class QueueStateModel(Model):
    """studio_ia.services.QueueState."""

    job: JobModel | None = None
    step: int = 0
    total: int = 0
    phase: str = ""
    prepare: int | None = None  # percent of the text prepared while phase is "preparing"
    prepare_phase: Literal["reading", "tokenizer", "tokenizing"] | None = None  # what the preparation is doing now
    saving: bool = False  # the running job is writing a checkpoint right now
    lr: float | None = None
    eta_s: float | None = None
    paused: bool = False
    blocked: str | None = None
    gate: GateModel = GateModel()
    stopping: bool = False
    up_next: list[JobModel] = []
    plan: PlanModel = PlanModel()
    failed: list[JobModel] = []  # jobs that ended failed (the last few), until removed
    notices: list[NoticeModel] = []  # what the running job reports while it goes


class Outlook(BaseModel):
    outlook: str


class MoveTo(BaseModel):
    index: int = Field(ge=0)


class Hold(BaseModel):
    held: bool = True


class Pin(BaseModel):
    pinned: str | None = Field(None, max_length=200)  # a machine's address or hostname; null or "" lets the job go anywhere again


class Started(BaseModel):
    started: bool


class NextStepModel(Model):
    route: str  # the page that does it
    title: str
    line: str


def _state(services):
    return QueueStateModel.model_validate(services.queue())


def _waiting(services, job_id):
    if job_id not in {j.id for j in services.queue().up_next}:
        raise HTTPException(404, f"No waiting job {job_id}")


@router.get("")
def get_queue(services: Services) -> QueueStateModel:
    """Services.queue"""
    return _state(services)


def local_jobs(services):
    """This machine's jobs: the running one first, then the jobs running on a peer, then Up next in order."""
    state = services.queue()
    away = [j for j in getattr(getattr(services, "runner", None), "jobs", ()) if j.node and j.status == "running" and not j.owner]
    return ([state.job] if state.job is not None else []) + away + list(state.up_next)


@router.get("/jobs")
def list_jobs(services: Services) -> list[JobModel]:
    """The running job first, then the jobs running on a peer, then Up next in order."""
    return [JobModel.model_validate(j) for j in local_jobs(services)]


@router.get("/done")
def list_done(services: Services) -> list[JobModel]:
    """Services.done_jobs: the jobs that ended done and the queue still remembers (the last few), newest first. The queue's own state
    lists only failed ones, so this is how Runs finds a finished job to list and open."""
    return [JobModel.model_validate(j) for j in reversed(services.done_jobs())]


@router.post("/jobs", status_code=201)
def enqueue(job: NewJob, services: Services) -> JobModel:
    """Services.enqueue: add a job to the end of Up next, or answer the job already queued under the caller's `id`. An unknown kind is a
    400."""
    made = Job(id=job.id or uuid.uuid4().hex[:8], created_at=time.time(), **job.model_dump(exclude={"id"}))
    return JobModel.model_validate(services.enqueue(made) or made)


SETTLE_S = 2.5  # how long Start waits to see whether the job it starts fails at once
SETTLE_STEP_S = .05


def _failed_ids(services):
    return {j.id for j in getattr(services.queue(), "failed", [])}


def _settled(state, seen_failed, waited):
    """True once Start can say what really happened: a job failed since the call began, the plan or a pause holds the queue, nothing
    is left to start, or a job has been running a moment without failing."""
    if {j.id for j in state.failed} - seen_failed or state.paused or state.blocked:
        return True
    return (state.job is None and not state.up_next) or (state.job is not None and waited >= .8)


@router.post("/start")
def start_queue(services: Services) -> QueueStateModel:
    """Services.start_queue: let jobs saved as waiting by an earlier session run. Answers with the queue's state after the start: it
    waits up to a few seconds for a job that fails at once (no room on the disk, a refused checkpoint) and reports it under `failed`
    with its sentence, instead of a bare 200 followed by a job that has quietly gone."""
    return _start_and_settle(services)


def _start_and_settle(services):
    before = _failed_ids(services)
    services.start_queue()
    waited = 0.
    while waited < SETTLE_S and not _settled(services.queue(), before, waited):
        sleep(SETTLE_STEP_S)
        waited += SETTLE_STEP_S
    return _state(services)


@router.post("/jobs/{job_id}/start_now")
def start_now(job_id: str, services: Services) -> QueueStateModel:
    """Run a waiting job now: release it if held, put it first in Up next, lift a pause and start the queue, answering as Start
    does. A job already running is not stopped: the chosen one goes next. The plan (hours, battery, idle) still holds it."""
    _waiting(services, job_id)
    services.hold(job_id, False)
    services.move(job_id, 0)
    if services.queue().paused:
        services.resume()
    return _start_and_settle(services)


@router.post("/jobs/{job_id}/move_up")
def move_up(job_id: str, services: Services) -> QueueStateModel:
    """Services.move_up: one place earlier in Up next."""
    _waiting(services, job_id)
    services.move_up(job_id)
    return _state(services)


@router.post("/jobs/{job_id}/move")
def move(job_id: str, to: MoveTo, services: Services) -> QueueStateModel:
    """Services.move: put a waiting job at `index` in Up next (0 runs next)."""
    _waiting(services, job_id)
    services.move(job_id, to.index)
    return _state(services)


@router.post("/jobs/{job_id}/hold")
def hold(job_id: str, services: Services, body: Hold = Hold()) -> QueueStateModel:
    """Services.hold: a held job keeps its place and is skipped; held=false releases it."""
    _waiting(services, job_id)
    services.hold(job_id, body.held)
    return _state(services)


@router.post("/jobs/{job_id}/pin")
def pin(job_id: str, body: Pin, services: Services) -> QueueStateModel:
    """Keep a waiting job on one machine (`pinned`: its address or hostname), or let it go anywhere again (null). A pin is never
    overridden: the job runs there or waits, and its waiting reason says why. Services has no pin, so a node with a job queue
    pins through the Runner, and the stand-in services edit the job in place."""
    _waiting(services, job_id)
    pinned = (body.pinned or "").strip() or None
    runner = getattr(services, "runner", None)
    if runner is not None:
        runner.pin(job_id, pinned)
    else:
        job = next(j for j in services.queue().up_next if j.id == job_id)
        job.params = {**{k: v for k, v in job.params.items() if k != "pinned"}, **({"pinned": pinned} if pinned else {})}
        services.emit("queue", services.queue())
    return _state(services)


@router.delete("/jobs/{job_id}")
def remove(job_id: str, services: Services) -> QueueStateModel:
    """Services.remove: drop a waiting job, or stop the running one, including one that runs on a peer (it is told to stop). On a
    failed job it dismisses it from the list."""
    if job_id not in {j.id for j in [*local_jobs(services), *services.queue().failed]}:
        raise HTTPException(404, f"No job {job_id}")
    services.remove(job_id)
    return _state(services)


@router.post("/jobs/{job_id}/restart")
def restart(job_id: str, services: Services) -> QueueStateModel:
    """Run a failed job again: a copy with the same kind, label, note and params goes first in Up next, and the failed entry is
    dismissed. Only a failed job restarts; any other id is a 404."""
    failed = next((j for j in services.queue().failed if j.id == job_id), None)
    if failed is None:
        raise HTTPException(404, f"No failed job {job_id}")
    again = Job(id=uuid.uuid4().hex[:8], created_at=time.time(), kind=failed.kind, label=failed.label, note=failed.note, params=dict(failed.params))
    services.enqueue(again)
    services.move(again.id, 0)
    services.remove(job_id)
    return _state(services)


@router.get("/outlook")
def outlook(services: Services) -> Outlook:
    """Services.queue_outlook for a job added to the end of the queue now."""
    return Outlook(outlook=services.queue_outlook())


@router.get("/jobs/{job_id}/outlook")
def job_outlook(job_id: str, services: Services) -> Outlook:
    """Services.queue_outlook for a queued job: when it starts. Empty once it has left the queue."""
    return Outlook(outlook=queued_outlook(services, job_id))


@router.post("/pause")
def pause(services: Services) -> QueueStateModel:
    """Services.pause"""
    services.pause()
    return _state(services)


@router.post("/resume")
def resume(services: Services) -> QueueStateModel:
    """Services.resume"""
    services.resume()
    return _state(services)


@router.post("/checkpoint_now")
def checkpoint_now(services: Services) -> Started:
    """Services.checkpoint_now: branch a short cool-down from the running job's last checkpoint. started=false when nothing is running."""
    return Started(started=bool(services.checkpoint_now()))


@router.post("/stop_after_cooldown")
def stop_after_cooldown(services: Services) -> QueueStateModel:
    """Services.stop_after_cooldown: finish the running job's cool-down, then stop it."""
    services.stop_after_cooldown()
    return _state(services)


@router.get("/plan")
def get_plan(services: Services) -> PlanModel:
    """The plan inside Services.queue()."""
    return _state(services).plan


@router.put("/plan")
def set_plan(plan: PlanModel, services: Services) -> PlanModel:
    """Services.set_plan"""
    services.set_plan(plan.model_dump())
    return _state(services).plan


@router.get("/next_step")
def next_step(services: Services) -> NextStepModel:
    """Services.next_step: the single "what next" answer."""
    return NextStepModel.model_validate(services.next_step())
