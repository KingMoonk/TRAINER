"""Passive training sessions and the watchdog that starts them when you sign in."""

from typing import Any

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
    checkpoints: int = 0  # checkpoints the session has saved
    version: int | None = None  # the version made from its best checkpoint, once saved


class Passive(BaseModel):
    sessions: list[PassiveSession]
    signin: bool  # the watchdog is set to start when you sign in


class PassiveStatus(BaseModel):
    status: str


@router.get("", response_model=Passive)
def passive(services=Depends(get_services)):
    return services.passive()


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
    """Ask a paused session to carry on. 409 when it has ended or is being stopped."""
    return command(session_id, "resume", services)


@router.post("/{session_id}/keep-going", response_model=PassiveStatus)
def keep_passive_going(session_id: str, services=Depends(get_services)):
    """Start the watchdog again for an active session whose watchdog and trainer are gone. 409 when the session has ended."""
    status = services.keep_passive_going(session_id)
    if status is None:
        raise HTTPException(404, f"No passive session {session_id}")
    return PassiveStatus(status=status)


class PassiveSaved(BaseModel):
    version: int
    created: bool  # False when that checkpoint was already saved as this version


@router.post("/{session_id}/save", response_model=PassiveSaved)
def save_passive(session_id: str, services=Depends(get_services)):
    """Save the session's best checkpoint as a trained version. Saving the same checkpoint again answers with the version it made.
    409 when the session has no checkpoint yet."""
    saved = services.save_passive(session_id)
    if saved is None:
        raise HTTPException(404, f"No passive session {session_id}")
    return PassiveSaved(**saved)


@router.delete("/signin", status_code=204)
def remove_passive_signin(services=Depends(get_services)):
    services.remove_passive_signin()
