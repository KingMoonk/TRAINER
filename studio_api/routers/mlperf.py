"""Training speed (studio_mlperf): the frozen reference tasks, their leaderboard, history and compare views, and queueing a run."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from studio_api.app import get_services
from studio_api.models_work import plain

router = APIRouter(prefix="/mlperf", tags=["mlperf"])


class Loose(BaseModel):
    model_config = ConfigDict(extra="allow")


class TaskOut(Loose):
    id: str
    version: int
    title: str
    target_bpc: float
    checksum: str
    attempts: int
    step_cap: int


class BoardOut(Loose):
    tasks: list[TaskOut]
    board: dict[str, Any] = Field(description="studio_mlperf.leaderboard: results per division, fastest first, and the other task versions held apart")
    history: dict[str, Any] = Field(description="the single-device division's converged times in time order, with the app version of each")
    devices: list[str] = Field(description="the devices this machine can time a task on")


class RunRequest(BaseModel):
    task: str
    device: str = "cpu"
    attempts: int | None = Field(None, ge=1, le=5)


class Queued(BaseModel):
    id: str
    label: str


class CompareRequest(BaseModel):
    ids: list[str] = Field(min_length=2)


@router.get("/board", response_model=BoardOut)
def board(task: str = "small", services=Depends(get_services)):
    try:
        return BoardOut.model_validate(plain(services.mlperf_board(task)))
    except ValueError as error:
        raise HTTPException(404, str(error)) from error


@router.post("/compare")
def compare(body: CompareRequest, services=Depends(get_services)) -> dict[str, Any]:
    """Summaries of the chosen results side by side. 422 when they are of different tasks or task versions."""
    try:
        return plain(services.mlperf_compare(body.ids))
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


@router.post("/run", response_model=Queued, status_code=201)
def run(body: RunRequest, services=Depends(get_services)):
    """Queue a run of the task on this machine; it starts when the queue is free."""
    try:
        return Queued(**services.mlperf_enqueue(body.task, body.device, body.attempts))
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
