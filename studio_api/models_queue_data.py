"""Models and helpers the queue, data and pretraining routers share: the job record a long read answers with."""

import dataclasses
import json
from typing import Any

from fastapi import Request
from pydantic import BaseModel, ConfigDict

from studio_api.events import encode


class Model(BaseModel):
    """Base for models read off the dataclasses and dicts Services returns."""

    model_config = ConfigDict(from_attributes=True, extra="ignore")


class JobRecordModel(Model):
    """studio_api.jobs.JobRecord: what POST of a long read returns, what GET /jobs/{id} answers, what /events carries as "job"."""

    id: str
    kind: str
    status: str  # queued | running | done | failed | cancelled
    progress: str = ""
    fraction: float | None = None
    result: Any = None
    error: str = ""
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None


def job_view(record, model=JobRecordModel):
    return model.model_validate(json.loads(encode(dataclasses.asdict(record))))


def submit(request: Request, kind, work, model=JobRecordModel, own_thread=False, lane="main", key=None):
    """Run `work(stop)` on the API's job worker; the result is the job's `result`. `model` is a JobRecordModel whose
    `result` is typed, so the OpenAPI schema says what the finished job holds. `own_thread` runs it at once instead of in turn."""
    return job_view(request.app.state.jobs.submit(kind, work, own_thread, lane, key), model)
