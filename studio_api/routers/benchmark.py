"""Benchmark: the fixed suite scored on every trained version, the few-shot check, how much training one version saved over
another, and the Compare runs experiments. Reads that score a checkpoint or parse the held-back data are jobs: POST answers with
its id and the result arrives on GET /events (and GET /jobs/{id})."""

import time
import uuid
from dataclasses import asdict
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from studio_api.app import get_services
from studio_api.models_work import JobOut, plain, start_job
from studio_ia.services import Job

STEPS, WALL_SECONDS = 300, 120.  # fixed length per run, as the Tk page: a choice when the board has one
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
    names: list[str] = Field(default_factory=list)  # base first, then the variants; may be missing in an older run
    question: str = ""
    guess: str = ""
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


@router.post("/read", response_model=JobOut, status_code=202)
def read_benchmark(request: Request):
    """The benchmark track as stored, one row per trained version; never scores anything. A job: it reads the held-back data.
    The result is a BenchmarkOut."""
    services = request.app.state.services
    return start_job(request, "benchmark.read", lambda stop: services.benchmark(), BenchmarkOut)


@router.post("/run", response_model=JobOut, status_code=202)
def run_benchmark(request: Request, body: RunRequest):
    """Score (or read back) the suite for one trained version. Slow. The result is the suite's dict for that version."""
    services = request.app.state.services
    return start_job(request, "benchmark.run", lambda stop: services.run_benchmark(body.version))


@router.post("/few-shot", response_model=JobOut, status_code=202)
def few_shot(request: Request, body: VersionRequest):
    """Lettered problems answered with 0, 1 and a few worked examples in the prompt, and no training. Slow.
    The result is a FewShotOut."""
    services = request.app.state.services
    return start_job(request, "benchmark.few_shot", lambda stop: services.few_shot(body.version), FewShotOut)


@router.post("/compare-runs", response_model=JobOut, status_code=202)
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


class CompareSetup(BaseModel):
    """What Compare runs asks: the base version, the variants (other saved versions), and how to hold them equal."""

    base: int | None = None  # None = the current version
    variants: list[int] = Field(default_factory=list)
    mode: Literal["params", "compute", "tokens", "wall-clock"] = "params"
    seeds: Literal[1, 3, 5] = 3
    retune_lr: bool = True
    question: str = ""
    guess: str = ""  # better | about the same | worse; needed to queue, not to plan


class PlanRow(BaseModel):
    variant: int
    name: str
    matched: bool
    reason: str = ""
    adjustments: list[Any] = Field(default_factory=list)
    steps: int | None = None
    parameters: int
    relative_error: float | None = None
    changes: list[str] = Field(default_factory=list)  # the raw knob changes from the base


class ComparePlan(BaseModel):
    variants: list[PlanRow]
    n_runs: int
    n_sweeps: int
    wall_seconds: float
    base: int | None = None


class Queued(BaseModel):
    id: str
    label: str
    outlook: str


def _experiment(services, body: CompareSetup):
    import studio_experiments as ex
    from studio_spec import default_spec, validate_spec
    from studio_versions import change_text

    versions = {v.number: v for v in services.versions()}
    base_version = versions.get(body.base) if body.base is not None else services.current_version()
    if body.base is not None and base_version is None:
        raise HTTPException(404, f"No version {body.base}")
    for number in body.variants:
        if number not in versions:
            raise HTTPException(404, f"No version {number}")
    try:
        base = validate_spec(base_version.spec) if base_version and base_version.spec else validate_spec(default_spec())
        specs = [validate_spec(versions[n].spec) for n in body.variants]
        names = ["Base"] + [change_text(base, spec) or f"Variant {i + 1}" for i, spec in enumerate(specs)]
        experiment = ex.ExperimentSpec(base, specs, seeds=body.seeds, mode=body.mode, steps=STEPS, wall_seconds=WALL_SECONDS,
                                       question=body.question.strip(), lr_retune=body.retune_lr, guess=body.guess, names=names)
        return experiment, ex.plan(experiment), base, base_version
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


@router.post("/compare-plan", response_model=ComparePlan)
def compare_plan(body: CompareSetup, services=Depends(get_services)):
    """The runs a Compare setup would make, and how each variant is matched to the base. Queues nothing. 422 says why a setup cannot run."""
    import studio_experiments as ex
    experiment, plan, base, base_version = _experiment(services, body)
    rows = [PlanRow(**{k: row[k] for k in PlanRow.model_fields if k in row and k != "changes"},
                    changes=[] if not row["variant"] else ex.spec_changes(base, experiment.variants[row["variant"] - 1]))
            for row in plan["variants"]]
    return ComparePlan(variants=rows, n_runs=plan["n_runs"], n_sweeps=plan["n_sweeps"], wall_seconds=WALL_SECONDS,
                       base=base_version.number if base_version else None)


@router.post("/compare-queue", response_model=Queued, status_code=201)
def compare_queue(body: CompareSetup, services=Depends(get_services)):
    """Queue the runs of a Compare setup. It needs a variant, a guess for the first one, and plain text to train on."""
    experiment, plan, _base, _version = _experiment(services, body)
    if not body.variants:
        raise HTTPException(422, "Add a variant first.")
    if not body.guess:
        raise HTTPException(422, "Pick your guess first.")
    summary = services.data("text")
    sets = [s["id"] for s in (summary.get("sets", []) if isinstance(summary, dict) else getattr(summary, "sets", []))]
    if not sets:
        raise HTTPException(422, "Your data has no plain text yet. Add some in Your data, then queue the runs.")
    label = f"Compare runs: {plan['n_runs']} runs" + (f" + {plan['n_sweeps']} short sweeps" if plan["n_sweeps"] else "")
    job = Job(id=uuid.uuid4().hex[:8], kind="compare", label=label, note=experiment.question, created_at=time.time(),
              params={"experiment": asdict(experiment), "dataset_kind": "text", "datasets": sets, "guess": body.guess})
    services.enqueue(job)
    return Queued(id=job.id, label=label, outlook=services.queue_outlook(job))
