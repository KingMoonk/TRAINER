"""Passive training sessions and the watchdog that starts them when you sign in."""

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
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
