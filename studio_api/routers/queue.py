"""The queue: what is running and waiting, the plan that gates it, and the controls (hold, move, remove, pause, checkpoint)."""

import time
import uuid
from datetime import datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

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
    error: str = ""


class NewJob(BaseModel):
    kind: str
    label: str = ""
    note: str = ""
    params: dict = {}
    eta_s: float | None = None


class GateModel(Model):
    """studio_queue.Gate: whether the plan lets the machine train right now."""

    reason: str | None = None  # "passive training is running" | "outside hours" | "on battery" | "computer in use" | None
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


class QueueStateModel(Model):
    """studio_ia.services.QueueState."""

    job: JobModel | None = None
    step: int = 0
    total: int = 0
    phase: str = ""
    lr: float | None = None
    eta_s: float | None = None
    paused: bool = False
    blocked: str | None = None
    gate: GateModel = GateModel()
    stopping: bool = False
    up_next: list[JobModel] = []
    plan: PlanModel = PlanModel()


class Outlook(BaseModel):
    outlook: str


class MoveTo(BaseModel):
    index: int = Field(ge=0)


class Hold(BaseModel):
    held: bool = True


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


@router.get("/jobs")
def list_jobs(services: Services) -> list[JobModel]:
    """The running job first, then Up next in order."""
    state = services.queue()
    return [JobModel.model_validate(j) for j in ([state.job] if state.job is not None else []) + list(state.up_next)]


@router.post("/jobs", status_code=201)
def enqueue(job: NewJob, services: Services) -> JobModel:
    """Services.enqueue: add a job to the end of Up next. An unknown kind is a 400."""
    made = Job(id=uuid.uuid4().hex[:8], created_at=time.time(), **job.model_dump())
    services.enqueue(made)
    return JobModel.model_validate(made)


@router.post("/start")
def start_queue(services: Services) -> QueueStateModel:
    """Services.start_queue: let jobs saved as waiting by an earlier session run."""
    services.start_queue()
    return _state(services)


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


@router.delete("/jobs/{job_id}")
def remove(job_id: str, services: Services) -> QueueStateModel:
    """Services.remove: drop a waiting job, or stop the running one."""
    state = services.queue()
    if job_id not in {j.id for j in [state.job, *state.up_next] if j is not None}:
        raise HTTPException(404, f"No job {job_id}")
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
