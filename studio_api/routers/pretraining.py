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
from studio_ia import sizing
import studio_data_kinds
import studio_fit
from studio_ia.services import if_added
from studio_spec import RUN_INTS

CHAT_SHARE_MAX = studio_data_kinds.CHAT_SHARE_MAX
MAX_STEPS = RUN_INTS["steps"][1]  # the most steps a run may have (validate_spec refuses more)

router = APIRouter(prefix="/pretraining", tags=["pretraining"])
Services = Annotated[Any, Depends(get_services)]

LISTS = ("letters", "small", "gpt2")  # the order of Services.word_lists()["tokenizers"]


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
    known: bool


class WordList(BaseModel):
    id: str  # letters | small | gpt2
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


class Started(BaseModel):
    """The queue job POST /pretraining/start added: its label is a provisional one while the size of its text is worked out."""

    id: str
    label: str


class StartJob(JobRecordModel):
    result: Started | None = None


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
    hours: float = Field(sizing.DEFAULT_HOURS, gt=0, le=24 * 60)  # the time the person can give; the suggested design and steps fit it
    steps: int | None = Field(None, ge=1, le=100_000_000)  # a step count chosen instead of `passes` (the suggestion's "use these steps")
    plateau: bool | None = None  # end the run when its validation loss goes flat: true always, false never, null for runs over an hour
    from_version: int | None = None  # the version whose best checkpoint a catch-up starts from (one of the plan's `sources`); null: the default
    window_hours: float = Field(studio_fit.DEFAULT_WINDOW_HOURS, gt=0, le=24 * 365)  # the longest the run may take; a token budget that needs longer is cut to fit
    chat_share: float = Field(0.0, ge=0.0, le=CHAT_SHARE_MAX)  # conversations mixed into the text as this share of its characters; 0 (off) is the default
    keep_my_settings: bool = False  # "use my settings anyway": keep the batch and token budget when the run only fits tightly or is only slow


class ScaleNote(BaseModel):
    """One thing the run was scaled for (studio_fit.Note). `advisory`: it only says the run is tight or slow, and "use my settings anyway" keeps the figure."""

    kind: Literal["batch", "offload", "context", "tokens", "model", "slow", "tight"]
    reason: Literal["vram", "ram", "disk", "time"]
    text: str
    advisory: bool = False


class Blocker(BaseModel):
    """The resource behind a refusal, in bytes."""

    resource: Literal["vram", "ram", "disk"]
    needed: int
    available: int


class Hardware(BaseModel):
    """What this computer has, measured now (studio_fit.hardware_profile). A figure that could not be read is null."""

    device: str  # cuda | mps | cpu
    gpu_name: str = ""
    vram_total: int = 0
    vram_free: int | None = None
    ram_total: int | None = None
    ram_free: int | None = None
    disk_free: int | None = None
    cpu_cores: int = 1
    tokens_per_second: float | None = None  # the newest device probe on this machine; null before any
    measured_at: float | None = None  # when it was made (epoch seconds)


class Scaling(BaseModel):
    """How the run was fitted to this computer (studio_fit.fit_solo): every change with its reason, and what the person may do about it."""

    notes: list[ScaleNote] = []
    scaled: bool  # the config or token budget differs from the one asked for
    can_keep_mine: bool = False  # "use my settings anyway" keeps the person's figure
    kept_mine: bool = False
    window_hours: float
    hours: float | None = None  # the run's time at the measured speed; null without one
    token_budget: int | None = None
    offload: bool = False  # the optimizer state lives in RAM
    refusal: str = ""  # a plain sentence when even the smallest scaled run cannot fit; set whether or not a blocker names a resource
    blocker: Blocker | None = None  # the resource behind a refusal, when one is to blame
    hardware: Hardware


class PlateauRule(BaseModel):
    """studio_plateau: the run ends (keeping its best) once the best validation loss improves by under `min_gain` over the window."""

    window_steps: int | None = None
    window_seconds: float | None = None
    min_gain: float
    min_steps: int = 0


class StepRate(Model):
    """studio_scaling.measure_step_rate: what a step of the design costs on this computer, on the device a run would use."""

    seconds_per_step: float
    tokens_per_second: float
    device: str  # cpu | cuda | mps
    device_name: str
    note: str = ""  # why the backend chose it, when it chose
    rows: int
    context: int


class StepRateRequest(BaseModel):
    version: int | None = None  # None: the current version
    dataset_ids: list[str] = []  # the chosen sets; their cached size preview gives the word list size
    rows: int = Field(16, ge=1, le=256)
    preview_rows: int = Field(6, ge=1, le=256)


class StepRateJob(JobRecordModel):
    result: StepRate | None = None


class Suggestion(BaseModel):
    """The design and step count that fit the hours given (studio_ia.sizing.suggest_design)."""

    width: int
    layers: int
    params: int
    steps: int
    hours: float  # what those steps take here
    tokens_per_param: float
    same_design: bool  # the current design already is the best fit: only the steps change
    text: str


class Expected(BaseModel):
    """What the plan can expect of the validation error. `fit`: drawn from this design's own earlier run; `rule`: no run to read."""

    basis: Literal["fit", "rule"]
    loss_planned: float | None = None  # at the last planned step
    loss_flat: float | None = None  # where the curve goes flat
    flat_steps: int | None = None
    curve: list[list[float]] = []  # [step, fitted validation loss] points


class Sizing(BaseModel):
    """Does the model fit the text and the time? (studio_ia.sizing.sizing)"""

    verdict: Literal["model_too_small", "data_too_small", "balanced"]
    warning: str
    tokens_per_param: float  # word pieces the planned steps read, per parameter
    measured: bool  # False: this computer has not timed the design yet (POST /pretraining/step_rate)
    seconds_per_step: float | None = None
    tokens_per_second: float | None = None
    device: str | None = None  # what the run will really use
    run_hours: float | None = None  # the planned steps, at the measured rate
    hours: float  # the time the person gave
    expected: Expected | None = None
    suggestion: Suggestion | None = None
    gpu_note: str = ""


class EnoughText(BaseModel):
    ratio: float  # word pieces per parameter
    ratio_text: str
    headline: str
    verdict: Literal["little", "about_right", "plenty"]


class SizeReadout(BaseModel):
    """The text against the design, from stored character counts: ready at once, before the slow size preview is known."""

    characters: int
    pieces: int  # word pieces: counted once the size preview is known, else characters / chars_per_piece
    counted: bool  # True: `pieces` is the size preview's count; False: an estimate that tokenized nothing
    chars_per_piece: float  # the figure the estimate used: the tokenizer's measured one when known, else 4
    params: int
    ratio: float  # word pieces per parameter
    ratio_text: str
    verdict: Literal["few", "right", "many"]  # under about 10 per parameter, about 10 to 40, above about 40 (the rule of thumb is 20)
    target_params: int  # the parameter count at which this text is about 20 word pieces per parameter


class Expectation(BaseModel):
    headline: str
    details: str


class NewPretrainJob(BaseModel):
    """The body for POST /queue/jobs."""

    kind: Literal["pretrain"] = "pretrain"
    label: str
    params: dict


class ContinueSource(BaseModel):
    """A trained version a catch-up may start from, and what its run did (studio_ia.pretraining_plan.run_source). Null: the run did not record it."""

    version: int
    name: str = ""
    change: str = ""
    status: str = ""  # the run's own: completed | stopped | ...
    finished_at: str | None = None  # ISO time
    steps: int | None = None  # steps trained
    steps_planned: int | None = None
    seconds: float | None = None  # wall time of the run
    tokens: int | None = None  # word pieces trained on
    params: int | None = None
    best_val_loss: float | None = None  # the best loss on the held-back text: the checkpoint's
    train_loss: float | None = None  # the last training loss
    device: str | None = None
    rounds: int | None = None  # set for a run over several machines: rounds of `inner_steps` steps each
    inner_steps: int | None = None
    same_design: bool  # every tensor is copied; otherwise only the ones whose shapes match
    parent: bool  # the version this one was saved from
    this_version: bool = False  # the version itself: a catch-up from it trains it further


class PretrainPlan(BaseModel):
    version: int | None = None  # null: no design saved yet
    parent: int | None = None
    mode: Literal["catch_up", "from_scratch"]  # the mode a run may use: catch_up only when continuing is valid
    can_continue: bool
    sources: list[ContinueSource] = []  # what a catch-up may start from, newest first
    continue_from: int | None = None  # the one a catch-up starts from: `from_version`, else the version itself when trained, else the parent, else the newest
    continue_blocked: Literal["word_list_changed", "no_trained_parent"] | None = None
    estimating: bool  # the size preview is not known yet: POST /pretraining/preview, then ask again
    pieces: int | None = None
    estimated: bool = False
    params: int | None = None
    context: int = 256
    pieces_per_step: int
    passes_over_text: int | None = None  # steps in one pass over the chosen text
    steps: int | None = None
    minutes: int | None = None  # null until this computer has timed the design (POST /pretraining/step_rate)
    lr: float | None = None
    sizing: Sizing | None = None  # null until the size of the text is known
    scaling: Scaling | None = None  # null until the run is known, or when this computer's hardware cannot be read
    enough: EnoughText | None = None
    readout: SizeReadout | None = None  # null without chosen text or a valid design
    expectation: Expectation | None = None
    config: dict | None = None  # the engine config the run will use
    can_add: bool  # "Add to the queue" may be pressed
    reason: str = ""  # why it may not
    job: NewPretrainJob | None = None  # post this to /queue/jobs
    outlook: str = ""  # when a job added now would start
    plateau: PlateauRule | None = None  # the rule that ends the run when its validation loss goes flat; null: none
    plateau_line: str = ""


def _sets(services):
    return services.text_sets()


def _parent(services, v):
    if v is None or v.parent is None:
        return None
    return next((p for p in services.versions() if p.number == v.parent), None)


def _readout(services, spec, ids, preview):
    """The fast readout: characters from stored counts, pieces estimated (or the preview's count once known), the design's parameters."""
    if not ids or not spec:
        return None
    try:
        found = services.text_size(ids, spec)
    except NotImplementedError:
        return None
    characters, params = found.get("characters") or 0, preview.get("params") or found.get("params")
    if not characters or not params:
        return None
    counted = bool(preview.get("pieces"))
    pieces = preview["pieces"] if counted else plan.estimated_pieces(characters, found.get("chars_per_piece"))
    verdict, ratio, target = plan.size_advice(pieces, params)
    return SizeReadout(characters=characters, pieces=pieces, counted=counted, chars_per_piece=characters / pieces if counted else found.get("chars_per_piece") or plan.CHARS_PER_PIECE,
                       params=params, ratio=ratio, ratio_text=plan.ratio_text(ratio), verdict=verdict, target_params=target)


def _sizing(services, v, mode, preview, steps, body, measured):
    """The plan's size section: the earlier run's curve is read only when this run starts from step 0 (a catch-up starts lower)."""
    from studio_spec import validate_spec
    try:
        spec = validate_spec(v.spec)
    except (ValueError, KeyError, TypeError):
        return None
    evidence = services.loss_evidence(v) if mode == "from_scratch" or v.parent is None else None
    gpu = services.device_facts() if measured else None
    found = sizing.sizing(spec, preview.get("vocab") or 256, body.rows, 1, preview["pieces"], preview["params"], steps, body.hours,
                          measured, evidence, gpu)
    found["measured"] = measured is not None
    return Sizing.model_validate(found)


def _fitted(services, v, config, body, mode, pieces, readout, measured, source=None):
    """The run fitted to this computer (studio_fit.fit_solo), with the figures the queue needs to fit it again at start; None when the
    hardware cannot be read or the design cannot be fitted. A solo run is scaled; a run over several machines is sized per member elsewhere."""
    profile = services.hardware_profile()
    if profile is None:
        return None
    limit = ((v.spec.get("train", {}).get("run", {}) or {}).get("token_limit_millions") or 0) * 1_000_000 or None
    speed = measured["tokens_per_second"] if measured else profile.tokens_per_second
    chars = readout.chars_per_piece if readout else plan.CHARS_PER_PIECE
    init = _absolute(services, source.best_checkpoint) if mode == "catch_up" and source is not None and source.best_checkpoint else None
    try:
        cached = services.cached_tokens(list(dict.fromkeys(body.dataset_ids)), config, init)  # tokens already in the token cache: no disk charged for them
    except (NotImplementedError, OSError, ValueError, KeyError, TypeError):
        cached = 0
    try:
        fit = studio_fit.fit_solo(config, profile, share=services.memory_share(), pieces=pieces, chars_per_piece=chars, token_budget=limit,
                                  window_hours=body.window_hours, speed=speed, keep_mine=body.keep_my_settings, fixed_architecture=mode == "catch_up",
                                  cached_pieces=cached)
    except (ValueError, KeyError, TypeError):
        return None
    shown = fit.to_dict()
    scaling = Scaling(notes=shown["notes"], scaled=shown["scaled"], can_keep_mine=shown["can_keep_mine"], kept_mine=shown["kept_mine"],
                      window_hours=body.window_hours, hours=shown["hours"], token_budget=shown["token_budget"], offload=shown["offload"],
                      refusal=shown["refusal"], blocker=shown["blocker"], hardware=Hardware.model_validate(profile.to_dict()))
    record = studio_fit.record(fit, config, limit, window_hours=body.window_hours, keep_mine=body.keep_my_settings, speed=speed,
                               pieces=pieces, chars_per_piece=chars, cached_pieces=cached)
    return {"fit": fit, "scaling": scaling, "record": record}


def _unusable_reason(unusable):
    """Why a run cannot start when the chosen text has nothing a run may use: generated text is a draft until it is marked Good."""
    names = ", ".join(f"{name} ({rows:,} rows waiting for review)" if rows else name for name, rows in unusable)
    return f"There is no text to train on yet in {names}. Generated text is used for nothing until it is marked Good: review it in Your data."


@router.get("/hardware")
def hardware(services: Services) -> Hardware | None:
    """What this computer has right now: GPU memory, free RAM and disk, and the speed the newest device probe measured. Null when it
    cannot be read. The plan scales a run to it; the figures are read fresh on every call."""
    found = services.hardware_profile()
    return None if found is None else Hardware.model_validate(found.to_dict())


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
    versions = services.versions()
    sources = plan.continue_sources(v, versions, lambda path: _absolute(services, path))
    numbers = [s["version"] for s in sources]
    if body.from_version is not None and body.from_version not in numbers:
        raise HTTPException(400, f"Version {body.from_version} has no trained weights with this version's word list to start from.")
    default = v.number if v is not None and v.number in numbers else parent.number if parent and parent.number in numbers else next(iter(numbers), None)
    chosen = body.from_version if body.from_version is not None else default
    source = next((p for p in versions if p.number == chosen), None)
    ok = source is not None
    blocked = (None if ok else "word_list_changed" if parent is not None and parent.best_checkpoint and plan.word_list_changed(v, parent)
               else "no_trained_parent")
    mode = plan.forced_mode(v, source, body.mode)
    ids = list(dict.fromkeys(body.dataset_ids))
    try:
        known = services.cached_preview(ids, body.preview_rows) if ids else {}
    except NotImplementedError:
        known = {}
    unusable = services.no_usable_text(ids) if ids and known is not None else []
    estimating = bool(ids) and not unusable and (known is None or not known.get("pieces"))
    preview = known or {}
    pieces, params, context = preview.get("pieces"), preview.get("params"), preview.get("context", 256)
    per_step = body.rows * context
    steps = None if estimating or not ids else body.steps or plan.steps_for(pieces, per_step, body.passes)
    if steps:
        steps = min(steps, MAX_STEPS)  # the engine refuses more; text this large is not passed over in full (passes_over_text shows how much is)
    measured = services.cached_step_rate(v.spec, body.rows) if v is not None and v.spec else None
    readout = _readout(services, v.spec if v is not None else None, ids, preview)
    config = fitted = scaling = None
    if v is not None and steps and not unusable:
        config = plan.pretrain_config(v.spec, steps, body.rows, body.hold)
        fitted = _fitted(services, v, config, body, mode, pieces, readout, measured, source)
        if fitted is not None:
            scaling = fitted["scaling"]
            if not fitted["fit"].refusal:
                steps = fitted["fit"].steps
    redesigned = fitted is not None and fitted["fit"].config.get("architecture") != config.get("architecture")  # the sizing below is of the design asked for
    size = (_sizing(services, v, mode, preview, steps, body, measured)
            if v is not None and steps and pieces and params and not redesigned else None)
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
              else _unusable_reason(unusable) if unusable
              else "The size of the text is still being worked out." if estimating
              else "The size of the chosen text is not known, so the steps cannot be worked out." if not steps else "")
    if fitted is not None and fitted["fit"].refusal and not reason:
        reason = fitted["fit"].refusal
    chat_sets = [s["id"] for s in services.conversation_sets()] if body.chat_share > 0 else []
    if body.chat_share > 0 and not chat_sets and not reason:
        reason = "A chat share needs conversations to mix in. Add some in Your data, or turn the chat share off."
    if fitted is not None:
        config = fitted["fit"].config
    job = None
    rule = plan.plateau_rule(steps, steps * measured["seconds_per_step"] / 3600 if steps and measured else None, body.plateau)
    if not reason:
        label, job_params = plan.job_params(v, parent, ids, mode, body.rows, body.hold, steps, pieces,
                                            plan.user_text(_sets(services), ids), rule, source, body.chat_share, chat_sets)
        if fitted is not None:
            job_params.update(fitted["fit"].job_params())
            job_params[studio_fit.RECORD] = fitted["record"]
        job = NewPretrainJob(label=label, params=job_params)
    lr = (((v.spec if v else None) or {}).get("train", {}).get("optimizer", {}).get("config", {}) or {}).get("lr")
    return PretrainPlan(
        version=v.number if v else None, parent=parent.number if parent else None, mode=mode, can_continue=ok, continue_blocked=blocked,
        sources=[ContinueSource.model_validate(s) for s in sources], continue_from=chosen,
        estimating=estimating, pieces=pieces, estimated=bool(preview.get("estimated")), params=params, context=context,
        pieces_per_step=per_step, passes_over_text=max(1, round(pieces / per_step)) if pieces else None, steps=steps,
        minutes=max(1, round(steps * measured["seconds_per_step"] / 60)) if steps and measured else None, lr=lr, sizing=size,
        scaling=scaling, enough=enough, readout=readout, expectation=expect, plateau=PlateauRule.model_validate(rule) if rule else None, plateau_line=plan.plateau_line(rule),
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


@router.post("/start", status_code=202)
def pretrain_start(body: PlanRequest, request: Request, services: Services) -> Started:
    """Add the run the plan builds for these answers to the queue now, even while the size of the chosen text is still being worked
    out: the run then waits in the queue, marked `params.sizing` and never started, while a job works the size out
    (Services.pretrain_preview) and gives the run its steps, config and label, or fails it with the plan's reason. 409 with the reason
    when the run cannot be added for anything but its size."""
    import time
    import uuid

    from studio_ia.services import Job

    made = pretrain_plan(body, services)
    if made.can_add:
        job = Job(id=uuid.uuid4().hex[:8], created_at=time.time(), note="", **made.job.model_dump())
        added = services.enqueue(job) or job
        return Started(id=added.id, label=added.label)
    if not made.estimating or made.version is None:
        raise HTTPException(409, made.reason)
    job = Job(id=uuid.uuid4().hex[:8], kind="pretrain", label=f"Version {made.version}, once the size of its text is known",
              created_at=time.time(), params={"sizing": True})
    added = services.enqueue(job) or job

    def work(stop):
        try:
            ids = list(dict.fromkeys(body.dataset_ids))
            cached = services.cached_preview(ids, body.preview_rows)
            if cached is None or not cached.get("pieces"):
                services.pretrain_preview(ids, body.preview_rows, progress=lambda percent: services.size_progress(added.id, percent))
            if stop.is_set():
                raise RuntimeError("Working out the size of the text was stopped. Add the run again.")
            ready = pretrain_plan(body, services)
            if not ready.can_add:
                raise ValueError(ready.reason)
        except Exception as exc:
            services.fail_job(added.id, str(exc) or type(exc).__name__)
            raise
        services.fill_job(added.id, ready.job.label, ready.job.params)
        return {"id": added.id, "label": ready.job.label}
    submit(request, "pretrain_start", work, StartJob, own_thread=True)  # not behind a slow read: the run waits in the queue for this
    return Started(id=added.id, label=added.label)


@router.post("/step_rate", status_code=202)
def step_rate(body: StepRateRequest, request: Request, services: Services) -> StepRateJob:
    """Time a few steps of the version's design on the device a run of it would use, as a job. The plan reads the answer
    (`sizing.measured`) for the time a run takes and the design that fits the hours given. 409 while the size of the chosen text is
    still unknown (the word list size comes from it)."""
    v = services.current_version() if body.version is None else _version(services, body.version)
    if v is None or not v.spec:
        raise HTTPException(409, "Save a design in the Network designer first.")
    preview = services.cached_preview(list(dict.fromkeys(body.dataset_ids)), body.preview_rows) if body.dataset_ids else {}
    if preview is None:
        raise HTTPException(409, "The size of the text is still being worked out.")
    vocab = preview.get("vocab") or 256
    return submit(request, "step_rate", lambda stop: services.measure_step_rate(v.spec, body.rows, vocab, stop=stop), StepRateJob)


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
    """The run folder a job writes to. A job that left the queue is found where the queue remembers it (`find_job`: failed and ended
    jobs keep their params), else through the version whose run folder is the job's own (`queue_<id>`, where a job writes when its
    params name no folder)."""
    job = services.find_job(job_id)
    if job is not None and job.params.get("output_dir"):
        return _absolute(services, job.params["output_dir"])
    for version in services.versions():
        for run in getattr(version, "runs", None) or []:
            if Path(run).name == f"queue_{job_id}":
                return _absolute(services, run)
    if job is None:
        raise HTTPException(404, f"No job {job_id}")
    raise HTTPException(404, f"Job {job_id} has no training run yet")


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


def auto_run_pretrain(services, settings, version):
    """The pretrain Job an Auto Run's pretrain phase enqueues: the plan POST /pretraining/start builds for the same answers (sizing
    and fitting included). `settings` holds PlanRequest fields (hours, passes, steps, mode, ...; dataset_ids defaults to every text set
    that is not held back). When the size of the text is not known yet it is worked out here, on the caller's thread. ValueError with
    the plan's own sentence when the run cannot be added."""
    from studio_ia.services import Job

    fields = {k: v for k, v in settings.items() if k in PlanRequest.model_fields and k != "version"}
    if not fields.get("dataset_ids"):
        fields["dataset_ids"] = [s["id"] for s in _sets(services) if not plan.held_out(s["id"])]
    body = PlanRequest(version=version, **fields)
    made = pretrain_plan(body, services)
    if not made.can_add and made.estimating:
        services.pretrain_preview(list(dict.fromkeys(body.dataset_ids)), body.preview_rows)
        made = pretrain_plan(body, services)
    if not made.can_add:
        raise ValueError(made.reason)
    return Job(id="", note="", **made.job.model_dump())
