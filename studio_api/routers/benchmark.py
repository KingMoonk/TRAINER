"""Benchmark: the fixed suite scored on every trained version, the few-shot check, how much training one version saved over
another, and the Compare runs experiments. Reads that score a checkpoint or parse the held-back data are jobs: POST answers with
its id and the result arrives on GET /events (and GET /jobs/{id})."""

from typing import Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, ConfigDict, Field

from studio_api.app import get_services
from studio_api.models_work import JobStarted, plain, start_job

router = APIRouter(prefix="/benchmark", tags=["benchmark"])


class Loose(BaseModel):
    model_config = ConfigDict(extra="allow")  # the engine may add keys; a client reads what it knows


class BenchmarkRow(Loose):
    number: int
    name: str = ""
    lineage: str = ""
    tokens: int | None = None
    result: dict[str, Any] | None = None  # studio_bench.run_suite's dict; None until the suite has run on this version


class BenchmarkOut(Loose):
    suite: str
    sizes: dict[str, int] = Field(default_factory=dict)
    dropped: dict[str, int] = Field(default_factory=dict)
    rows: list[BenchmarkRow] = Field(default_factory=list)


class FewShotRow(Loose):
    shots: int
    used: float
    correct: int
    accuracy: float | None = None


class FewShotOut(Loose):
    items: int
    tasks: int = 0
    held_back: bool = True  # False: no held-back choice problem exists, so training ones were asked
    chance: float | None = None
    rows: list[FewShotRow] = Field(default_factory=list)


class CompareRunOut(Loose):
    base: dict[str, Any]
    variants: list[Any] = Field(default_factory=list)
    mode: str | None = None
    results: dict[str, Any] = Field(default_factory=dict)
    finished: Any = None


class VersionRequest(BaseModel):
    version: int | None = None  # None = the best trained version


class RunRequest(BaseModel):
    version: int


class Verdict(Loose):
    percent: float | None = None  # share of the older run's training the newer one saved
    comparable: bool
    reason: str = ""


@router.post("/read", response_model=JobStarted)
def read_benchmark(request: Request):
    """The benchmark track as stored, one row per trained version; never scores anything. A job: it reads the held-back data.
    The result is a BenchmarkOut."""
    services = request.app.state.services
    return start_job(request, "benchmark.read", lambda stop: services.benchmark(), BenchmarkOut)


@router.post("/run", response_model=JobStarted)
def run_benchmark(request: Request, body: RunRequest):
    """Score (or read back) the suite for one trained version. Slow. The result is the suite's dict for that version."""
    services = request.app.state.services
    return start_job(request, "benchmark.run", lambda stop: services.run_benchmark(body.version))


@router.post("/few-shot", response_model=JobStarted)
def few_shot(request: Request, body: VersionRequest):
    """Lettered problems answered with 0, 1 and a few worked examples in the prompt, and no training. Slow.
    The result is a FewShotOut."""
    services = request.app.state.services
    return start_job(request, "benchmark.few_shot", lambda stop: services.few_shot(body.version), FewShotOut)


@router.post("/compare-runs", response_model=JobStarted)
def compare_runs(request: Request):
    """The finished Compare runs experiments, oldest first. The result is a list of CompareRunOut."""
    services = request.app.state.services

    def read(stop):
        return [CompareRunOut.model_validate(plain(row)).model_dump(mode="json") for row in services.compare_runs()]
    return start_job(request, "benchmark.compare_runs", read)


@router.get("/verdict", response_model=Verdict)
def verdict(a: int, b: int, services=Depends(get_services)):
    """How much training the newer of versions `a` and `b` saved over the older; no verdict unless both finished the same schedule."""
    return Verdict.model_validate(plain(services.verdict(a, b)))
