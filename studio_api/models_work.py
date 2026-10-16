"""What the post-training, chat, inspect, benchmark and viewer routers share: the answer to starting a job, the plain-data
conversion for job results, and the response model for a Version."""

import dataclasses
import math
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import Request
from pydantic import BaseModel, Field


class JobOut(BaseModel):
    """A studio_api.jobs.JobRecord as it stands when the job is submitted. Its progress and result arrive on GET /events as "job"
    events; GET /jobs/{id} serves late joiners."""
    id: str
    kind: str
    status: str
    progress: str = ""
    fraction: float | None = None
    result: Any = None
    error: str = ""
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None


class VersionOut(BaseModel):
    number: int
    name: str = ""
    change: str = ""
    parent: int | None = None
    spec: dict[str, Any] = Field(default_factory=dict)
    status: str = "draft"
    best_checkpoint: str | None = None
    scorecard: dict[str, Any] = Field(default_factory=dict)
    verdict: str = ""
    prediction: dict[str, Any] = Field(default_factory=dict)


def plain(value):
    """Data a browser can parse: numpy become lists and numbers, dataclasses dicts, keys strings, NaN and infinity null."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return plain(dataclasses.asdict(value))
    if hasattr(value, "model_dump"):  # a pydantic model
        return plain(value.model_dump())
    if isinstance(value, dict):
        return {key if isinstance(key, str) else str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    return str(value)


def version_out(version) -> VersionOut:
    return VersionOut(**{field.name: getattr(version, field.name) for field in dataclasses.fields(version)})


def start_job(request: Request, kind: str, read, model: type[BaseModel] | None = None) -> JobOut:
    """Run `read(stop)` on the jobs worker and answer with its id. The result is `plain(read(...))`, shaped by `model` when given."""
    def work(stop):
        result = plain(read(stop))
        return model.model_validate(result).model_dump(mode="json") if model else result
    record = request.app.state.jobs.submit(kind, work)
    return JobOut(**{name: getattr(record, name) for name in JobOut.model_fields})
