"""Passive training sessions and the watchdog that starts them when you sign in."""

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict

from studio_api.app import get_services

router = APIRouter(prefix="/passive", tags=["passive"])


class PassiveSession(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    status: str
    created: str | None = None
    stop: Any = None
    trainer: str = ""
    watchdog: str = ""
    restarts: int = 0
    heard: float | None = None  # seconds since the trainer's last heartbeat
    error: Any = None


class Passive(BaseModel):
    sessions: list[PassiveSession]
    signin: bool  # the watchdog is set to start when you sign in


class PassivePlan(BaseModel):
    """studio_passive.plan_summary: one plain sentence per group of settings."""
    stop: str
    limits: str
    plateau: str
    teacher: str
    worse: str


class PassiveCheckpoint(BaseModel):
    n: int
    round: int
    error: float  # the held-out test error
    elapsed: float | None = None  # active seconds when it was saved
    at: str | None = None
    parent: int | None = None
    kept: bool = True
    discarded: bool = False  # a rollback threw it away
    pivot: str | None = None  # the change tried before this round
    polished: bool = False  # the teacher's polish replaced it
    win_rate: float | None = None
    lessons: int = 0
    best: bool = False


class PassiveTeacher(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str = ""
    reachable: bool | None = None
    done: int = 0
    total: int = 0


class PassiveProgress(BaseModel):
    phase: str  # idle | train | save | teach, "" before the trainer set up
    round: int = 0
    best: int | None = None  # the best checkpoint's number
    best_error: float | None = None
    current: int | None = None
    current_error: float | None = None
    learning_rate: float | None = None
    batch_size: int | None = None
    win_rate: float | None = None
    stop_reason: str | None = None
    active_seconds: float = 0
    limit_seconds: float
    worse_streak: int = 0
    rounds_since_best: int = 0
    restarts: int = 0
    teacher: PassiveTeacher | None = None
    checkpoints: list[PassiveCheckpoint]


class PassiveEvent(BaseModel):
    """One line of the trainer's journal. `type` is start, round, check, checkpoint, rollback, teacher, lesson, restart, power, pause,
    resume, stop, memory or error; the other fields depend on it (message, reason, error, from, to, ...)."""
    model_config = ConfigDict(extra="allow")
    index: int
    type: str
    at: str | None = None


class PassiveJournal(BaseModel):
    events: list[PassiveEvent]  # oldest first
    total: int
    first: int  # index of events[0]; ask for `before=first` to get the events before them


class PassiveDetail(BaseModel):
    session: PassiveSession
    plan: PassivePlan
    progress: PassiveProgress
    journal: PassiveJournal


class PassiveLesson(BaseModel):
    """One thing the teacher showed the model: the prompt, the model's own reply and the better answer it should learn."""
    id: str
    round: int
    checkpoint: int
    prompt: str = ""
    reply: str = ""
    answer: str = ""
    status: Literal["pending", "dropped", "left_out"]  # pending: trained on; dropped: you left it out; left_out: a step back left it out
    at: str | None = None


class LessonChange(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["pending", "dropped"]  # "dropped" leaves it out of the next round's training, "pending" keeps it


class PassiveStatus(BaseModel):
    status: str


class StartPassive(BaseModel):
    """The New session form. Every field has the plan's default (studio_passive.DEFAULT_PLAN); the plan is validated by merge_plan."""
    model_config = ConfigDict(extra="forbid")
    version: int | None = None  # the version to train; null: the current one
    sources: list[str] | None = None  # the text sets to train on; null or empty: all of them
    duration: float = 8
    unit: Literal["minutes", "hours", "days"] = "hours"
    patience_rounds: int = 3  # stop after this many rounds without a better checkpoint; 0 turns it off
    target_error: float | None = None  # stop when the test error reaches this
    teacher: bool = True  # polish each checkpoint with the teacher model and carry its lessons forward
    memory_fraction: float = 0.5
    keep_awake: bool = True
    pause_on_battery: bool = True


class PassiveStarted(BaseModel):
    id: str
    status: str


@router.get("", response_model=Passive)
def passive(services=Depends(get_services)):
    return services.passive()


@router.get("/{session_id}", response_model=PassiveDetail)
def passive_detail(session_id: str, limit: int = Query(50, ge=1, le=200), before: int | None = Query(None, ge=0), services=Depends(get_services)):
    """One session: its row, plan, progress (phase, round, best error, the error of each checkpoint) and a page of the trainer's journal,
    the newest `limit` events or those before event number `before`. 404 for an unknown session."""
    found = services.passive_detail(session_id, limit, before)
    if found is None:
        raise HTTPException(404, f"No passive session {session_id}")
    return found


@router.get("/{session_id}/lessons", response_model=list[PassiveLesson])
def passive_lessons(session_id: str, services=Depends(get_services)):
    """The lessons the teacher gave this session, oldest first. 404 for an unknown session."""
    found = services.passive_lessons(session_id)
    if found is None:
        raise HTTPException(404, f"No passive session {session_id}")
    return found


@router.patch("/{session_id}/lessons/{lesson_id}", response_model=PassiveLesson)
def change_passive_lesson(session_id: str, lesson_id: str, body: LessonChange, services=Depends(get_services)):
    """Drop a lesson (the next round leaves it out of the training) or keep it. 404 for an unknown session or lesson."""
    found = services.set_passive_lesson(session_id, lesson_id, body.status)
    if found is None:
        raise HTTPException(404, f"No lesson {lesson_id} in passive session {session_id}")
    return found


@router.post("", response_model=PassiveStarted, status_code=201)
def start_passive(body: StartPassive, services=Depends(get_services)):
    """Create a session from the form and start its watchdog. 400 with a sentence for a bad plan or version, 409 while another session is active."""
    plan = {"stop": {"duration": body.duration, "unit": body.unit, "patience_rounds": body.patience_rounds, "target_error": body.target_error},
            "limits": {"memory_fraction": body.memory_fraction, "keep_awake": body.keep_awake, "pause_on_battery": body.pause_on_battery},
            "teacher": {"polish": body.teacher, "lessons": body.teacher},
            "data": {"sources": body.sources or []}}
    return PassiveStarted(id=services.start_passive(plan, body.version), status="queued")


@router.post("/{session_id}/stop", response_model=PassiveStatus)
def stop_passive(session_id: str, services=Depends(get_services)):
    status = services.stop_passive(session_id)
    if status is None:
        raise HTTPException(404, f"No passive session {session_id}")
    return PassiveStatus(status=status)


def command(session_id, command, services):
    status = services.command_passive(session_id, command)
    if status is None:
        raise HTTPException(404, f"No passive session {session_id}")
    return PassiveStatus(status=status)


@router.post("/{session_id}/pause", response_model=PassiveStatus)
def pause_passive(session_id: str, services=Depends(get_services)):
    """Ask a session to pause at the trainer's next check. 409 when it has ended or is being stopped."""
    return command(session_id, "pause", services)


@router.post("/{session_id}/resume", response_model=PassiveStatus)
def resume_passive(session_id: str, services=Depends(get_services)):
    """Ask a paused session to carry on; a stopped or failed one goes back in the queue (status "queued"), its watchdog starts and the
    trainer continues from its last checkpoint. 409 when it has finished, is being stopped, or another session is active."""
    return command(session_id, "resume", services)


@router.post("/{session_id}/keep-going", response_model=PassiveStatus)
def keep_passive_going(session_id: str, services=Depends(get_services)):
    """Start the watchdog again for an active session whose watchdog and trainer are gone. 409 when the session has ended."""
    status = services.keep_passive_going(session_id)
    if status is None:
        raise HTTPException(404, f"No passive session {session_id}")
    return PassiveStatus(status=status)


@router.delete("/signin", status_code=204)
def remove_passive_signin(services=Depends(get_services)):
    services.remove_passive_signin()
