"""Pretraining: the page's steps and config (POST /pretraining/plan), the size preview, the word lists, the learning-rate range
test and the training-run page's reads (curve, phases, references, gate result). The slow ones are jobs (studio_api.jobs).

The draft design (Services.stage_draft / draft_spec) is not served: the browser holds it and sends the spec it means.
"""

from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from studio_api.app import get_services
from studio_api.models_queue_data import JobRecordModel, Model, submit
from studio_ia import pretraining_plan as plan
from studio_ia.services import if_added

router = APIRouter(prefix="/pretraining", tags=["pretraining"])
Services = Annotated[Any, Depends(get_services)]

LISTS = ("letters", "small", "gpt2", "words")  # the order of Services.word_lists()["tokenizers"]


class PreviewModel(Model):
    """Services.pretrain_preview. Every field is absent when there is no design yet."""

    pieces: int | None = None  # tokenised size of the chosen text (estimated for large text)
    estimated: bool = False
    params: int | None = None
    vocab: int | None = None
    context: int | None = None
    first_rows: list[list[int]] = []  # each cell holds an example number, -1 where an example ends
    tokens_per_second: float | None = None


class PreviewRequest(BaseModel):
    dataset_ids: list[str] = []  # empty: every set
    rows: int = Field(16, ge=1, le=256)


class Chip(BaseModel):
    label: str
    known: bool  # False: the whole-word list does not hold it, the model sees only "unknown"


class WordList(BaseModel):
    id: str  # letters | small | gpt2 | words
    vocab_size: int
    count: int  # pieces the example sentence is cut into
    chips: list[Chip]


class WordListsModel(BaseModel):
    """Services.word_lists, with each tokenizer cut down to what the page draws. A list that is not available is null."""

    sentence: str
    lists: list[WordList | None]


class RangeRequest(BaseModel):
    spec: dict  # the design to test (the browser's draft, or a saved version's spec)


class RangeResult(Model):
    """studio_compute.lr_range_test"""

    rates: list[float] = []
    losses: list[float] = []
    smoothed: list[float] = []
    suggestion: float | None = None
    steepest: int | None = None
    diverged: bool = False
    failed_at: float | None = None
    cancelled: bool = False
    lr_min: float | None = None
    lr_max: float | None = None


class PreviewJob(JobRecordModel):
    result: PreviewModel | None = None


class WordListsJob(JobRecordModel):
    result: WordListsModel | None = None


class RangeJob(JobRecordModel):
    result: RangeResult | None = None


class TextSet(Model):
    """One set of text the run may read. `on` is the page's starting tick: every set but a held-out (validation or test) one."""

    id: str
    name: str
    count: int
    held: bool  # a validation or test split: trained on, it could no longer measure anything
    library: bool = False  # bundled with the app, not the person's own
    on: bool


class PlanRequest(BaseModel):
    """The answers on the Pretraining page. The browser keeps them (the draft choices); every call sends them whole."""

    version: int | None = None  # None: the current version
    dataset_ids: list[str] = []  # the chosen text sets
    mode: Literal["catch_up", "from_scratch"] = "catch_up"
    rows: int = Field(16, ge=1, le=256)  # rows per step
    hold: float = Field(0.05, ge=0.0, lt=0.5)  # fraction of the text held back for testing
    passes: int = Field(2, ge=1, le=100)  # passes over the chosen text: Short 1, Medium 2, Long 4
    preview_rows: int = Field(6, ge=1, le=256)  # rows of the size preview the page draws; the cached preview is read with this


class EnoughText(BaseModel):
    ratio: float  # word pieces per parameter
    ratio_text: str
    headline: str
    verdict: Literal["little", "about_right", "plenty"]


class Expectation(BaseModel):
    headline: str
    details: str


class NewPretrainJob(BaseModel):
    """The body for POST /queue/jobs."""

    kind: Literal["pretrain"] = "pretrain"
    label: str
    params: dict


class PretrainPlan(BaseModel):
    version: int | None = None  # null: no design saved yet
    parent: int | None = None
    mode: Literal["catch_up", "from_scratch"]  # the mode a run may use: catch_up only when continuing is valid
    can_continue: bool
    continue_blocked: Literal["word_list_changed", "no_trained_parent"] | None = None
    estimating: bool  # the size preview is not known yet: POST /pretraining/preview, then ask again
    pieces: int | None = None
    estimated: bool = False
    params: int | None = None
    context: int = 256
    pieces_per_step: int
    passes_over_text: int | None = None  # steps in one pass over the chosen text
    steps: int | None = None
    minutes: int | None = None  # null until a run has shown this computer's speed
    lr: float | None = None
    enough: EnoughText | None = None
    expectation: Expectation | None = None
    config: dict | None = None  # the engine config the run will use
    can_add: bool  # "Add to the queue" may be pressed
    reason: str = ""  # why it may not
    job: NewPretrainJob | None = None  # post this to /queue/jobs
    outlook: str = ""  # when a job added now would start


def _sets(services):
    return services.text_sets()


def _parent(services, v):
    if v is None or v.parent is None:
        return None
    return next((p for p in services.versions() if p.number == v.parent), None)


@router.get("/text_sets")
def text_sets(services: Services) -> list[TextSet]:
    """The sets of text to choose from, as the Pretraining page's tiles."""
    out = []
    for s in _sets(services):
        held = plan.held_out(s["id"])
        out.append(TextSet(id=s["id"], name=s["name"], count=s["count"], held=held, library=bool(s.get("library")), on=not held))
    return out


@router.post("/plan")
def pretrain_plan(body: PlanRequest, services: Services) -> PretrainPlan:
    """What the Pretraining page shows for these answers: whether Continue is allowed, the steps, the size verdict, the
    expectation, the engine config and the job to post to /queue/jobs. Reads the size preview only if it is already known
    (`estimating` otherwise); it never computes it."""
    v = services.current_version() if body.version is None else _version(services, body.version)
    parent = _parent(services, v)
    ok = plan.can_continue(v, parent)
    blocked = None if ok else "word_list_changed" if plan.word_list_changed(v, parent) else "no_trained_parent"
    mode = plan.forced_mode(v, parent, body.mode)
    ids = list(dict.fromkeys(body.dataset_ids))
    try:
        known = services.cached_preview(ids, body.preview_rows) if ids else {}
    except NotImplementedError:
        known = {}
    estimating = known is None
    preview = known or {}
    pieces, params, context = preview.get("pieces"), preview.get("params"), preview.get("context", 256)
    per_step = body.rows * context
    steps = None if estimating else plan.steps_for(pieces, per_step, body.passes)
    tps = preview.get("tokens_per_second")
    enough = None
    if not estimating and pieces and params:
        ratio = pieces / params
        headline, verdict = plan.enough_text(ratio)
        enough = EnoughText(ratio=ratio, ratio_text=plan.ratio_text(ratio), headline=headline, verdict=verdict)
    expect = None
    if v is not None and not estimating:
        headline, details = plan.version_expectation(v.spec, preview, body.passes)
        expect = Expectation(headline=headline, details=details) if headline else None
    reason = ("Save a design in the Network designer first." if v is None else "Choose some text first." if not ids
              else "The size of the text is still being worked out." if estimating
              else "The size of the chosen text is not known, so the steps cannot be worked out." if not steps else "")
    config = job = None
    if v is not None and steps:
        config = plan.pretrain_config(v.spec, steps, body.rows, body.hold)
    if not reason:
        label, job_params = plan.job_params(v, parent, ids, mode, body.rows, body.hold, steps, pieces,
                                            plan.user_text(_sets(services), ids))
        job = NewPretrainJob(label=label, params=job_params)
    lr = (((v.spec if v else None) or {}).get("train", {}).get("optimizer", {}).get("config", {}) or {}).get("lr")
    return PretrainPlan(
        version=v.number if v else None, parent=parent.number if parent else None, mode=mode, can_continue=ok, continue_blocked=blocked,
        estimating=estimating, pieces=pieces, estimated=bool(preview.get("estimated")), params=params, context=context,
        pieces_per_step=per_step, passes_over_text=max(1, round(pieces / per_step)) if pieces else None, steps=steps,
        minutes=max(1, round(steps * per_step / tps / 60)) if steps and tps else None, lr=lr, enough=enough, expectation=expect,
        config=config, can_add=not reason, reason=reason, job=job, outlook=if_added(services.queue_outlook()))


def word_lists_plain(lists):
    """word_lists() with its tokenizer objects replaced by their name, size and the example sentence cut by each."""
    from studio_ia import pieces

    sentence, out = lists["sentence"], []
    for name, tok in zip(LISTS, lists["tokenizers"]):
        if tok is None:
            out.append(None)
            continue
        chips = [{"label": label, "known": known} for label, known in pieces.chips(tok, sentence)]
        out.append({"id": name, "vocab_size": tok.vocab_size, "count": len(chips), "chips": chips})
    return {"sentence": sentence, "lists": out}


@router.get("/preview")
def cached_preview(services: Services, dataset_ids: Annotated[list[str], Query()] = [],
                   rows: Annotated[int, Query(ge=1, le=256)] = 16) -> PreviewModel | None:
    """Services.cached_preview: the preview when it is already known, else null. Never computes; POST /pretraining/preview does."""
    found = services.cached_preview(list(dataset_ids), rows)
    return None if found is None else PreviewModel.model_validate(found)


@router.post("/preview", status_code=202)
def pretrain_preview(body: PreviewRequest, request: Request, services: Services) -> PreviewJob:
    """Services.pretrain_preview as a job."""
    return submit(request, "pretrain_preview", lambda stop: services.pretrain_preview(body.dataset_ids, body.rows), PreviewJob)


@router.post("/word_lists", status_code=202)
def word_lists(request: Request, services: Services) -> WordListsJob:
    """Services.word_lists as a job."""
    return submit(request, "word_lists", lambda stop: word_lists_plain(services.word_lists()), WordListsJob)


@router.post("/lr_range_test", status_code=202)
def lr_range_test(body: RangeRequest, request: Request, services: Services) -> RangeJob:
    """Services.lr_range_test as a job; DELETE /jobs/{id} ends it early.
    A design with no plain text to try it on fails the job with a plain sentence."""
    return submit(request, "lr_range_test", lambda stop: services.lr_range_test(body.spec, stop=stop), RangeJob)


# the training-run page: the run is found from a version number or a queue job, never from a path the client sends ------------


class Sample(Model):
    step: int
    text: str


class CurveModel(Model):
    """Services.run_curve. Series are [step, value] pairs; error is in bits per character."""

    steps: list[tuple[float, float]] = []
    smooth: list[tuple[float, float]] = []
    heldout: list[tuple[float, float]] = []
    lr: list[tuple[float, float]] = []
    phases: list[tuple[float, float, str]] = []  # (from_step, to_step, name)
    total: int = 0
    samples: list[Sample] = []
    sample_every: int | None = None
    prompt: str | None = None


class PhaseModel(BaseModel):
    from_step: float
    to_step: float
    name: str  # warm-up | steady | cool-down


class ReferencesModel(Model):
    """Services.references: reference lines in bits per character; null where unknown."""

    uniform_bpc: float | None = None
    unigram_bpc: float | None = None
    best_so_far_bpc: float | None = None


class GateModel(Model):
    """Services.gate_result: how Chat weighed a version against the best one before it. diff and upper are bits per character."""

    better: bool
    diff: float
    upper: float
    vs: int  # the version it was weighed against


def _version(services, number):
    found = next((v for v in services.versions() if v.number == number), None)
    if found is None:
        raise HTTPException(404, f"No version {number}")
    return found


def _run_of_version(services, number):
    version = _version(services, number)
    runs = getattr(version, "runs", None) or ([str(Path(version.best_checkpoint).parent)] if version.best_checkpoint else [])
    if not runs:
        raise HTTPException(404, f"Version {number} has no training run yet")
    return _absolute(services, runs[-1])


def _run_of_job(services, job_id):
    state = services.queue()
    job = next((j for j in ([state.job] if state.job is not None else []) + list(state.up_next) if j.id == job_id), None)
    if job is None:
        raise HTTPException(404, f"No job {job_id}")
    if not job.params.get("output_dir"):
        raise HTTPException(404, f"Job {job_id} has no training run yet")
    return _absolute(services, job.params["output_dir"])


def _absolute(services, run):
    root = getattr(services, "root", None)
    return str(Path(root) / run) if root and not Path(run).is_absolute() else str(run)


def _phases(found):
    return [PhaseModel(from_step=a, to_step=b, name=name) for a, b, name in found]


@router.get("/versions/{number}/curve")
def version_curve(number: int, services: Services) -> CurveModel:
    """Services.run_curve for the version's latest run."""
    return CurveModel.model_validate(services.run_curve(_run_of_version(services, number)))


@router.get("/versions/{number}/phases")
def version_phases(number: int, services: Services) -> list[PhaseModel]:
    """Services.run_phases for the version's latest run."""
    return _phases(services.run_phases(_run_of_version(services, number)))


@router.get("/jobs/{job_id}/curve")
def job_curve(job_id: str, services: Services) -> CurveModel:
    """Services.run_curve for the run a queue job writes to (the running job's live curve)."""
    return CurveModel.model_validate(services.run_curve(_run_of_job(services, job_id)))


@router.get("/jobs/{job_id}/phases")
def job_phases(job_id: str, services: Services) -> list[PhaseModel]:
    """Services.run_phases for the run a queue job writes to."""
    return _phases(services.run_phases(_run_of_job(services, job_id)))


@router.get("/versions/{number}/references")
def references(number: int, services: Services) -> ReferencesModel:
    """Services.references: the lines the error curve is read against."""
    return ReferencesModel.model_validate(services.references(_version(services, number)))


@router.get("/versions/{number}/gate_result")
def gate_result(number: int, services: Services) -> GateModel | None:
    """Services.gate_result: null when Chat did not weigh the version."""
    found = services.gate_result(_version(services, number))
    return None if found is None else GateModel.model_validate(found)
