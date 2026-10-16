"""Quick test: the staged design change, the throwaway micro run of it against the current version, and what to do with it.

The run itself is a queue job (kind "quicktest"); its progress and result arrive on GET /events as "train" events
(`quicktest`, `quicktest_done`) and the newest finished result is kept for GET /quicktest."""

import time
import uuid
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict

import studio_quicktest
from studio_api.app import get_services
from studio_ia.services import Job, Version
from studio_spec import default_spec, validate_spec
from studio_versions import spec_diff

router = APIRouter(prefix="/quicktest", tags=["quicktest"])

Dataset = Literal["stories", "slice", "problems"]
Guess = Literal["Lower", "Same", "Higher"]
MINUTES = (1, 3, 10)


class QuickTestDesign(BaseModel):
    title: str
    description: str  # "4 blocks · 4 look-back heads · normal thinking room"


class QuickTestDraft(BaseModel):
    changes: list[str]  # spec_diff words, empty when the staged design equals the current one
    description: str
    cost: str  # "same size · same work per step", or how much bigger/slower


class QuickTestRun(BaseModel):
    steps: int
    bpc: float | None = None
    curve: list[list[float]] = []
    sample: str = ""
    model_config = ConfigDict(extra="allow")


class QuickTestCheck(BaseModel):
    id: str
    ok: bool | None = None  # None: not run
    detail: str = ""


class QuickTestReferences(BaseModel):
    uniform: float
    unigram: float


class QuickTestResult(BaseModel):
    """studio_quicktest.run's result, as stored."""
    model_config = ConfigDict(extra="allow")
    minutes: float
    shrink: bool = False
    dataset: str = ""
    stopped: bool = False
    verdict: Literal["lower", "about the same", "higher"] | None = None
    delta: float | None = None
    threshold: float | None = None
    references: QuickTestReferences | None = None
    checks: list[QuickTestCheck] = []
    broken: str | None = None  # id of the failed check to look at
    learned_context: bool = False
    current: QuickTestRun | None = None
    draft: QuickTestRun | None = None
    finished_at: float | None = None


class QuickTestState(BaseModel):
    current: QuickTestDesign
    draft: QuickTestDraft | None  # None: no change staged in the designer
    next_number: int  # the number Save idea would give
    last: QuickTestResult | None  # the newest finished result, if it finished after `since`


class QuickTestStageDraft(BaseModel):
    spec: dict


class QuickTestRunRequest(BaseModel):
    dataset: Dataset = "stories"
    minutes: Literal[1, 3, 10] = 3
    shrink: bool = False
    guess: Guess


class QuickTestStarted(BaseModel):
    job_id: str
    outlook: str  # when it starts, e.g. "Starts now"
    started_at: float  # the server clock when it was queued: GET /quicktest?since= shows a result that finished after it


def _specs(services):
    v = services.current_version()
    current = validate_spec(v.spec) if v and v.spec else validate_spec(default_spec())
    draft = services.draft_spec()
    return v, current, (validate_spec(draft) if draft else None)


def _describe(spec):
    from studio_ia.pages.quicktest import describe  # the Tk page owns the wording until it is retired
    return describe(spec)


def _cost(current, draft):
    from studio_ia.pages.quicktest import cost_sentence
    return cost_sentence(current, draft)


def _has_text(services):
    summary = services.data("text")
    sets = summary.get("sets", []) if isinstance(summary, dict) else getattr(summary, "sets", [])
    return bool(sets)


@router.get("", response_model=QuickTestState)
def quicktest_state(since: float = 0, services=Depends(get_services)):
    """The current version, the staged change (if any) and the newest finished result that finished after `since` (epoch seconds)."""
    v, current, draft = _specs(services)
    return QuickTestState(
        current=QuickTestDesign(title=f"Version {v.number}" if v else "Current version", description=_describe(current)),
        draft=QuickTestDraft(changes=spec_diff(current, draft), description=_describe(draft), cost=_cost(current, draft)) if draft else None,
        next_number=(v.number if v else 0) + 1,
        last=studio_quicktest.load_last(services, since))


@router.put("/draft", status_code=204)
def stage_draft(body: QuickTestStageDraft, services=Depends(get_services)):
    """Stage an unsaved design change for Quick test and Compare (the designer's draft)."""
    services.stage_draft(validate_spec(body.spec))


@router.delete("/draft", status_code=204)
def discard_draft(services=Depends(get_services)):
    services.stage_draft(None)


@router.post("/run", response_model=QuickTestStarted, status_code=202)
def run(body: QuickTestRunRequest, services=Depends(get_services)):
    """Queue a quick test of the staged change against the current version. 400 without a staged change or, for "slice", without text."""
    _v, current, draft = _specs(services)
    if draft is None:
        raise HTTPException(400, "Stage a change in the designer first.")
    if body.dataset == "slice" and not _has_text(services):
        raise HTTPException(400, "Your data has no plain text yet. Add some in Your data, or pick another tiny dataset.")
    job = Job(id=uuid.uuid4().hex[:8], kind="quicktest", label=f"Quick test · {body.minutes} min each", created_at=time.time(),
              params={"current_spec": current, "draft_spec": draft, "dataset": body.dataset, "minutes": body.minutes,
                      "shrink": body.shrink, "guess": body.guess})
    services.enqueue(job)
    return QuickTestStarted(job_id=job.id, outlook=services.queue_outlook(job), started_at=job.created_at)


@router.post("/save", response_model=Version, status_code=201)
def save_idea(services=Depends(get_services)):
    """Save the staged change as the next version and clear the stage; the designer then shows it as saved and needing training."""
    _v, current, draft = _specs(services)
    if draft is None:
        raise HTTPException(409, "No change is staged.")
    version = services.save_version(draft, ", ".join(spec_diff(current, draft)) or "no change")
    services.stage_draft(None)
    return version
