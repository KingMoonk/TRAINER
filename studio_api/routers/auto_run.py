"""Auto Run: a plan of training phases (pretrain, then rounds 1-4) the orchestrator (studio_ia.autorun) walks over the job queue.

Reads: the runs, the default plan, and one run with its plan, state, last 50 journal lines and the live progress of the job it waits
for. Writes: make a run (a draft), change the plan of a draft or paused run, start, pause, resume, stop, skip a phase, and
continue with a choice when it waits for the person. Changes of state arrive on GET /events as `auto_run` {id, status, current, reason}.
A plan that cannot be is 422 with the sentences that say why; a step the run's state does not allow is 409; an unknown run is 404."""

from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict

import studio_autorun as ar
from studio_api.app import get_services
from studio_ia.autorun import PlanLocked

router = APIRouter(prefix="/auto-runs", tags=["auto-runs"])
Services = Annotated[Any, Depends(get_services)]

RunStatus = Literal["draft", "running", "pausing", "paused", "waiting_user", "stopped", "done", "failed"]
PhaseStatus = Literal["pending", "queued", "running", "passed", "failed", "skipped", "stopped", "blocked"]


class AutoRunPhasePlan(BaseModel):
    """One phase of a plan. kind: data | pretrain | midtrain | sft | rlvr | distill | dpo, each at most once, in that order.
    on_pass: continue | pause. on_fail: pause | stop | skip. `settings` are the phase's own (the pretraining or round settings)."""
    kind: str
    enabled: bool
    settings: dict = {}
    on_pass: str = "continue"
    on_fail: str = "pause"


class AutoRunBudget(BaseModel):
    hours: float | None = None  # wall-clock hours for the whole run; null: no limit


class AutoRunBound(BaseModel):
    min: float
    max: float


class AutoRunPlanModel(BaseModel):
    id: str = ""  # empty on a new run: the server makes one
    name: str
    version: int
    budget: AutoRunBudget
    bounds: dict[str, AutoRunBound] = {}  # learning_rate | steps | beta: the limits the parameter controller stays within (over the defaults)
    phases: list[AutoRunPhasePlan]


class AutoRunNewRun(BaseModel):
    plan: AutoRunPlanModel


class AutoRunPlanBody(BaseModel):
    plan: AutoRunPlanModel


class AutoRunChoice(BaseModel):
    choice: Literal["next", "retry", "stop", "adopt"]  # skip the phase and go on | run it again | end the run | keep the failed round's checkpoint, then go on


class AutoRunPhaseState(BaseModel):
    model_config = ConfigDict(extra="allow")
    kind: str
    status: PhaseStatus
    job_id: str | None = None  # the queue job's id once the phase has been asked for
    attempt: int = 1
    started: str | None = None
    ended: str | None = None
    checkpoint: str | None = None
    result: dict | None = None
    error: str = ""


class AutoRunTuning(BaseModel):
    """The parameters the controller chose for a phase and why. choices: what it changed from the plan; proposal: a bigger design, to
    be taken by the person; evidence: what the last phase showed."""
    model_config = ConfigDict(extra="allow")
    phase: str
    attempt: int = 1
    settings: dict = {}
    choices: list[dict] = []
    proposal: dict | None = None
    evidence: dict | None = None


class AutoRunRunState(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    status: RunStatus
    current: str | None = None  # the phase being done or next
    reason: str = ""  # a sentence saying why the run waits, paused, stopped or failed
    started: str | None = None
    deadline: str | None = None  # null: no budget limit
    updated: str | None = None
    phases: list[AutoRunPhaseState]
    tuning: list[AutoRunTuning] = []  # one entry per phase job built, oldest first


class AutoRunJournalLine(BaseModel):
    model_config = ConfigDict(extra="allow")
    t: str
    event: str
    status: str = ""
    current: str | None = None
    reason: str = ""


class AutoRunJobProgress(BaseModel):
    """The queue job of the phase that is queued or running. phase, step, total and fraction are set only while it is the job training."""
    id: str
    kind: str
    label: str = ""
    status: str  # waiting | held | running | done | failed | stopped
    phase: str = ""  # warm-up | steady | cool-down | preparing
    step: int | None = None
    total: int | None = None
    fraction: float | None = None
    eta_s: float | None = None


class AutoRunRunView(BaseModel):
    id: str
    plan: AutoRunPlanModel
    state: AutoRunRunState
    journal: list[AutoRunJournalLine]  # the last 50 lines, oldest first
    progress: AutoRunJobProgress | None = None  # null when no phase has a job in the queue


class AutoRunRunSummary(BaseModel):
    id: str
    name: str
    version: int
    status: RunStatus
    current: str | None = None
    reason: str = ""
    started: str | None = None
    updated: str | None = None
    phases: list[AutoRunPhaseState]


def auto(services):
    return services.autorun


def known(services, run_id):
    if run_id not in auto(services).ids():
        raise HTTPException(404, f"There is no Auto Run {run_id}.")


def progress(services, state):
    entry = next((p for p in state["phases"] if p["status"] in ("running", "queued") and p["job_id"]), None)
    job = services.find_job(entry["job_id"]) if entry else None
    if job is None:
        return None
    out = {"id": job.id, "kind": job.kind, "label": job.label, "status": job.status, "phase": "", "eta_s": job.eta_s}
    queue = services.queue()
    if queue.job is not None and queue.job.id == job.id:
        out.update(phase=queue.phase, step=queue.step, total=queue.total, eta_s=queue.eta_s or job.eta_s,
                   fraction=(queue.step / queue.total) if queue.total else None)
    return out


def view(services, body):
    return {"id": body["plan"]["id"], **body, "progress": progress(services, body["state"])}


def act(services, run_id, call, *args):
    """Run an orchestrator control. A state that does not allow it (or a checkpoint that is not there to adopt) is 409 with the sentence."""
    known(services, run_id)
    try:
        return view(services, call(run_id, *args))
    except ValueError as exc:  # AutorunError is a ValueError
        raise HTTPException(409, str(exc)) from exc


@router.get("", response_model=list[AutoRunRunSummary])
def list_runs(services: Services):
    runs = []
    for run_id in auto(services).ids():
        body = auto(services).view(run_id)
        plan, state = body["plan"], body["state"]
        runs.append({"id": run_id, "name": plan["name"], "version": plan["version"], "status": state["status"], "current": state["current"],
                     "reason": state["reason"], "started": state["started"], "updated": state.get("updated"), "phases": state["phases"]})
    return sorted(runs, key=lambda r: r["updated"] or "", reverse=True)


@router.get("/defaults", response_model=AutoRunPlanModel)
def defaults(version: int):
    """The plan a new Auto Run starts from for `version`: every phase on except mid-training, continue on a pass, pause on a failure."""
    return ar.DEFAULT_PLAN(version)


@router.post("", response_model=AutoRunRunView, status_code=201)
def create(body: AutoRunNewRun, services: Services):
    try:
        run_id = auto(services).create(body.plan.model_dump())
    except ar.AutorunError as exc:
        raise HTTPException(422, str(exc)) from exc
    return view(services, auto(services).view(run_id))


@router.get("/{run_id}", response_model=AutoRunRunView)
def get_run(run_id: str, services: Services):
    known(services, run_id)
    return view(services, auto(services).view(run_id))


@router.put("/{run_id}/plan", response_model=AutoRunRunView)
def change_plan(run_id: str, body: AutoRunPlanBody, services: Services):
    """Only a draft or paused run (else 409), and for a paused one only the current and later phases that have no job yet (else 422)."""
    known(services, run_id)
    try:
        return view(services, auto(services).replace_plan(run_id, body.plan.model_dump()))
    except PlanLocked as exc:
        raise HTTPException(409, str(exc)) from exc
    except ar.AutorunError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/{run_id}/start", response_model=AutoRunRunView)
def start(run_id: str, services: Services):
    return act(services, run_id, auto(services).start)


@router.post("/{run_id}/pause", response_model=AutoRunRunView)
def pause(run_id: str, services: Services):
    return act(services, run_id, auto(services).pause)


@router.post("/{run_id}/resume", response_model=AutoRunRunView)
def resume(run_id: str, services: Services):
    return act(services, run_id, auto(services).resume)


@router.post("/{run_id}/stop", response_model=AutoRunRunView)
def stop(run_id: str, services: Services):
    return act(services, run_id, auto(services).stop)


@router.post("/{run_id}/skip", response_model=AutoRunRunView)
def skip(run_id: str, services: Services):
    """Skip the current phase (a failed, stopped, blocked or not yet started one). The run stays as it is; resume it."""
    return act(services, run_id, auto(services).skip)


@router.post("/{run_id}/continue", response_model=AutoRunRunView)
def continue_run(run_id: str, body: AutoRunChoice, services: Services):
    return act(services, run_id, auto(services).continue_run, body.choice)
