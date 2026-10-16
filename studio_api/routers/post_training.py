"""Post-training: what each round has done, the choices a round offers, queueing a round, and starting from a pretrained model."""

import hashlib
import uuid
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.exceptions import HTTPException
from pydantic import BaseModel, ConfigDict, Field

import studio_registry as registry
from studio_api.app import get_services
from studio_api.models_work import JobOut, VersionOut, start_job, version_out
from studio_ia.copy import FEEDBACK as T
from studio_ia.services import Job

router = APIRouter(prefix="/post-training", tags=["post-training"])

ROUND_NAMES = ("Practice answers", "Practice with checks", "Learn from a teacher", "Your votes")
LEASH = (("Close", .5, T["close_note"]), ("Medium", .1, T["medium_note"]), ("Far", .02, T["far_note"]))  # label, beta, note
LORA_RANKS = (4, 8, 16, 32)  # adapter sizes offered; studio_lora's own default (8) is the middle one
CHUNK = 1 << 20


class RoundState(BaseModel):
    status: str = "idle"  # done | running | queued | idle
    note: str = ""


class OverlapOut(BaseModel):
    model_config = ConfigDict(extra="allow")
    checked: int = 0
    left_out: int = 0
    note: str | None = None


class PostTrainingOut(BaseModel):
    """Services.post_training: missing keys are fine, a client reads them as unknown."""
    rounds: dict[int, RoundState] = Field(default_factory=dict)
    counts: dict[str, int] = Field(default_factory=dict)
    gates: dict[int, dict[str, Any]] = Field(default_factory=dict)
    overlap: dict[str, OverlapOut] = Field(default_factory=dict)
    example: Any = None


class PostTrainingJobOut(JobOut):
    result: PostTrainingOut | None = None


class ImportModelOut(BaseModel):
    id: str
    name: str
    size: int


@router.get("/import-models", response_model=list[ImportModelOut])
def import_models():
    """Models this server can import; size is the expected download in bytes."""
    from studio_import import MODELS
    return [ImportModelOut(id=key, name=model["title"], size=model["size"]) for key, model in MODELS.items()]


class LeashOut(BaseModel):
    name: str
    beta: float
    note: str


class TeacherOut(BaseModel):
    number: int
    name: str


class OptionsOut(BaseModel):
    leash: list[LeashOut]
    lora_ranks: list[int]
    lora_blocker: str | None  # why LoRA cannot be chosen for round 1 from this version, in plain words
    lora_default: bool  # whether the version already trains adapters
    teachers: list[TeacherOut]  # trained versions that could teach this one in round 3
    default_teacher: int | None


class RoundRequest(BaseModel):
    round: int = Field(ge=1, le=4)
    version: int | None = None  # None = the current version
    lora: bool | None = None  # round 1: adapters (True) or every weight (False); None = what the version already does
    lora_rank: int = 8
    teacher: Literal["version", "correction"] = "version"  # round 3: another trained version, or this model shown a correction
    teacher_version: int | None = None  # None = the previous best
    leash: Literal["Close", "Medium", "Far"] = "Medium"  # round 4


class RoundQueued(BaseModel):
    id: str
    label: str
    round: int
    params: dict[str, Any]


class RowsRequest(BaseModel):
    kind: Literal["conversations", "problems", "pairs", "corrections", "heldout_docs", "heldout_problems"]


class StatusRequest(BaseModel):
    version: int | None = None


def pick_version(services, number):
    """The version asked for, the current one when `number` is None; 404 for a number that is not a version."""
    if number is None:
        return services.current_version()
    found = next((v for v in services.versions() if v.number == number), None)
    if found is None:
        raise HTTPException(404, f"There is no version {number}.")
    return found


def is_mixed_precision(spec):
    """True when the design's precision module runs matrix multiplies in 16 bits (bf16, fp16); LoRA refuses these."""
    entry = ((spec or {}).get("train") or {}).get("precision")
    entry = {"id": entry} if isinstance(entry, str) else entry
    if not isinstance(entry, dict) or not entry.get("id"):
        return False
    try:
        return registry.get("precision", entry["id"]).compute is not None
    except ValueError:
        return False


def lora_blocker(version):
    """Why LoRA cannot be chosen for a round that starts from `version` (plain words), or None; mirrors what studio_model.train
    refuses: LoRA without a checkpoint to start from, and LoRA under mixed precision."""
    if not (version and version.best_checkpoint):
        return T["lora_no_checkpoint"]
    return T["lora_mixed"] if is_mixed_precision(version.spec) else None


def lora_default(version):
    return bool((((version.spec if version else None) or {}).get("train") or {}).get("run", {}).get("lora"))


def teacher_options(versions, version):
    """Trained versions that could teach `version`: any other one with a checkpoint, the same word list and the same reading window."""
    def shape(v):
        spec = v.spec or {}
        return spec.get("tokenizer"), (spec.get("dims") or {}).get("n_ctx")
    return [v for v in versions if version and v.number != version.number and v.best_checkpoint and shape(v) == shape(version)]


def default_teacher(options, version):
    """The previous best (the latest option before `version`), else the latest option; None when there is none."""
    earlier = [v for v in options if version and v.number < version.number]
    return (earlier or options or [None])[-1]


def round_blocker(round_number, counts, teacher_is_version, has_teacher, checkpoint):
    """Why a round cannot be queued yet (plain words), or None. counts = post_training()["counts"]."""
    need = {1: ("conversations", 1, "Add conversations in Your data first."),
            2: ("problems", 1, "Make problems a rule can check first."),
            4: ("pairs", 2, "Rate at least 2 pairs in Chat first.")}
    if not checkpoint:
        return "Train this version in Pretraining first."
    if round_number == 3:
        if teacher_is_version:
            return None if has_teacher else "There is no other trained version to learn from yet."
        return None if counts.get("corrections") else "Teach a correction in Chat first."
    key, least, why = need[round_number]
    return None if (counts.get(key) or 0) >= least else why


@router.post("/status", response_model=PostTrainingJobOut, status_code=202)
def read_status(request: Request, body: StatusRequest):
    """What each round has done for a version, and what is stored for the rounds to use. A job: it counts the stored data.
    The result is a PostTrainingOut."""
    services = request.app.state.services
    version = pick_version(services, body.version)
    return start_job(request, "post_training.status", lambda stop: services.post_training(version), PostTrainingOut)


@router.post("/rows", response_model=JobOut, status_code=202)
def read_rows(request: Request, body: RowsRequest):
    """The rows behind a round (conversations, problems, rated pairs, corrections) or the held-back ones. A job: the held-back
    text is parsed once and its overlap with training is checked."""
    services = request.app.state.services
    return start_job(request, f"post_training.rows.{body.kind}", lambda stop: services.post_rows(body.kind))


@router.get("/options", response_model=OptionsOut)
def options(version: int | None = None, services=Depends(get_services)):
    """The choices a round offers for a version: LoRA for round 1, the teachers for round 3, the leash for round 4."""
    chosen = pick_version(services, version)
    teachers = teacher_options(services.versions(), chosen)
    teacher = default_teacher(teachers, chosen)
    return OptionsOut(leash=[LeashOut(name=name, beta=beta, note=note) for name, beta, note in LEASH], lora_ranks=list(LORA_RANKS),
                      lora_blocker=lora_blocker(chosen), lora_default=lora_default(chosen),
                      teachers=[TeacherOut(number=v.number, name=v.name) for v in teachers],
                      default_teacher=teacher.number if teacher else None)


@router.post("/rounds", response_model=RoundQueued)
def queue_round(body: RoundRequest, services=Depends(get_services)):
    """Add a post-training round to the queue, with the choices the page offers. Refused with the page's own sentence when the
    round has nothing to learn from yet."""
    version = pick_version(services, body.version)
    if version is None:
        raise ValueError("There is no version yet. Build one first.")
    counts = (services.post_training(version) or {}).get("counts", {})
    teachers = teacher_options(services.versions(), version)
    why = round_blocker(body.round, counts, body.teacher == "version", bool(teachers), bool(version.best_checkpoint))
    if why:
        raise ValueError(why)
    name = ROUND_NAMES[body.round - 1]
    params = {"round": body.round, "version": version.number, "checkpoint": version.best_checkpoint}
    if body.round == 1:
        if body.lora_rank not in LORA_RANKS:
            raise ValueError(f"LoRA rank is one of {', '.join(map(str, LORA_RANKS))}.")
        wanted = lora_default(version) if body.lora is None else body.lora
        blocked = lora_blocker(version)
        if wanted and blocked and body.lora:
            raise ValueError(blocked)
        if not blocked:
            params["config"] = {"lora": {"rank": body.lora_rank} if wanted else None}
            if wanted:
                name += " · " + T["lora_job"].format(rank=body.lora_rank)
    if body.round == 3:
        teacher = None
        if body.teacher == "version":
            teacher = next((v for v in teachers if v.number == body.teacher_version), None) or default_teacher(teachers, version)
            if body.teacher_version is not None and (teacher is None or teacher.number != body.teacher_version):
                raise ValueError(f"Version {body.teacher_version} cannot teach version {version.number}: "
                                 "it needs a checkpoint, the same word list and the same reading window.")
        params["teacher"] = teacher.best_checkpoint if teacher else None
        params["teacher_choice"] = "previous best" if body.teacher == "version" else "correction"
    if body.round == 4:
        params["beta"] = next(beta for label, beta, _note in LEASH if label == body.leash)
    job = Job(uuid.uuid4().hex[:8], "post_round", f"Round {body.round} · {name}", f"post-training round {body.round}", params=params)
    services.enqueue(job)
    if body.round == 3:  # a round 3 goes to the front of the queue, as the page does
        for _ in range(len(services.queue().up_next)):
            services.move_up(job.id)
    return RoundQueued(id=job.id, label=job.label, round=body.round, params=params)


def keep_weights(services, name, upload):
    """Copy an uploaded model file into the workspace's pretrained folder, so the import job finds it instead of downloading it.
    The file must be the published one: its size and sha256 are checked as it is copied, and it is kept only when both match."""
    import studio_import
    root = getattr(services, "root", None)
    if root is None:
        raise ValueError("This server has no workspace to keep the file in.")
    info = studio_import.MODELS[name]
    dest = Path(root) / "data" / "pretrained" / name / "model.safetensors"
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    digest, size = hashlib.sha256(), 0
    try:
        with part.open("wb") as out:
            while chunk := upload.file.read(CHUNK):
                size += len(chunk)
                if size > info["size"]:
                    break
                digest.update(chunk)
                out.write(chunk)
        if size != info["size"] or digest.hexdigest() != studio_import.GPT2_SHA256:
            raise ValueError(f"That file is not the published {info['title']} weights. Check you chose "
                             f"{info['source']}'s model.safetensors, or leave the file out and the server downloads it.")
        part.replace(dest)
    finally:
        part.unlink(missing_ok=True)


@router.post("/imports", response_model=VersionOut)
def import_pretrained(name: str = Form("gpt2"), weights: UploadFile | None = File(None), lora: bool = Form(False),
                      lora_rank: int = Form(8), services=Depends(get_services)):
    """Start from a pretrained open model: saves its design as a new version and queues the conversion; the version trains
    (gets a checkpoint) when the job ends. `weights` is the model's published model.safetensors, for a server that cannot
    download it. `lora` fine-tunes small adapters instead of every weight."""
    import studio_import
    if name not in studio_import.MODELS:
        raise ValueError(f"Unknown pretrained model {name!r}.")
    if lora_rank not in LORA_RANKS:
        raise ValueError(f"LoRA rank is one of {', '.join(map(str, LORA_RANKS))}.")
    if weights is not None and weights.filename:
        keep_weights(services, name, weights)
    return version_out(services.import_pretrained(name, lora={"rank": lora_rank} if lora else None))
