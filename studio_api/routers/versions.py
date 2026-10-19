"""Versions: the design history. Fast reads are GETs; anything that scores a checkpoint or reads a run's files is a job."""

import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from studio_api.app import get_services
from studio_ia.services import Version
from studio_versions import NoCheckpoint

router = APIRouter(prefix="/versions", tags=["versions"])


class JobOut(BaseModel):
    """A studio_api.jobs.JobRecord; the result arrives on /events and GET /jobs/{id}."""
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


class NewVersion(BaseModel):
    spec: dict
    change: str = ""
    prediction: dict[str, str] | None = Field(None, description='the guess made on saving, e.g. {"error": "lower"}')


def _job(request, kind, work):
    record = request.app.state.jobs.submit(kind, work)
    return JobOut(**{k: getattr(record, k) for k in JobOut.model_fields})


class PublishedVersion(BaseModel):
    """One version in the version store: what its machine published, and whether this workspace has it."""
    key: str
    number: int
    name: str | None = None
    change: str | None = None
    origin: str = Field(description="the machine that published it")
    device: str | None = Field(None, description="the device that trained it (cuda, mps, cpu), when the checkpoint recorded one")
    train_device: str | None = Field(None, description="the same, under the name the version row uses")
    score_device: str | None = Field(None, description="the device that scored it on the held-back text, when known")
    step: int | None = None
    published_at: str | None = None
    size: int | None = Field(None, description="bytes of the weights file in the store")
    scorecard: dict = Field(default_factory=dict, description="as measured on the machine that trained it")
    mine: bool = Field(description="this machine published it")
    local: bool = Field(description="this workspace already has it as a version")
    local_number: int | None = Field(None, description="the version number it has here")


class SharedVersion(BaseModel):
    """What a peer needs to add this version without the store: the metadata `publish` uploads next to the weights."""
    key: str
    number: int
    name: str | None = None
    change: str | None = None
    parent: int | None = None
    spec: dict
    scorecard: dict = Field(default_factory=dict, description="as measured on this machine")
    origin: str = Field(description="this machine's name in the version store")
    device: str | None = None
    train_device: str | None = None
    score_device: str | None = None
    step: int | None = None
    published_at: str = Field(description="when this copy of the weights was made")
    weights_dtype: str = "float16"
    size: int = Field(description="bytes of the weights file")
    sha256: str = Field(description="of the weights file")
    format: int = 1


def _known(services, number):
    if not any(v.number == number for v in services.versions()):
        raise HTTPException(404, f"No version {number}")


@router.get("", response_model=list[Version])
def list_versions(services=Depends(get_services)):
    return services.versions()


@router.post("", response_model=Version, status_code=201)
def save_version(body: NewVersion, services=Depends(get_services)):
    return services.save_version(body.spec, body.change, body.prediction)


@router.get("/current", response_model=Version | None)
def current_version(services=Depends(get_services)):
    return services.current_version()


@router.post("/best", response_model=JobOut, status_code=202)
def best_version(request: Request, services=Depends(get_services)):
    """The version Chat follows. Weighing versions scores their checkpoints on the held-back text, so it is a job."""
    return _job(request, "versions.best", lambda stop: services.best_version())


@router.post("/{number}/checkpoint-step", response_model=JobOut, status_code=202)
def checkpoint_step(number: int, request: Request, services=Depends(get_services)):
    _known(services, number)
    return _job(request, "versions.checkpoint_step", lambda stop: services.checkpoint_step(number))


@router.post("/{number}/run-facts", response_model=JobOut, status_code=202)
def run_facts(number: int, request: Request, services=Depends(get_services)):
    """What the version's run did ({data, steps, tokens, start, lr, extra}). The first read hashes the data file."""
    _known(services, number)
    return _job(request, "versions.run_facts", lambda stop: services.run_facts(number))


@router.post("/{number}/run-speed", response_model=JobOut, status_code=202)
def run_speed(number: int, request: Request, services=Depends(get_services)):
    _known(services, number)
    return _job(request, "versions.run_speed", lambda stop: services.run_speed(number))


class ContinueBody(BaseModel):
    kind: Literal["same", "new_data", "grow"] = Field(description="same: more steps on the same data, with the optimizer, sampler and random "
                                                                  "state it had; new_data: more steps on other text, same weights and step; "
                                                                  "grow: a bigger design started from these weights")
    steps: int = Field(ge=1, le=10_000_000, description="steps to add (same, new_data), or to train the grown design for (grow)")
    dataset_ids: list[str] = Field([], description="the text to continue on (new_data; grow, default the run's own)")
    width: int | None = Field(None, ge=8, le=8192, description="grow: the design's width (with layers), when no spec is given")
    layers: int | None = Field(None, ge=1, le=96)
    spec: dict | None = Field(None, description="grow: the bigger design itself")
    rows: int = Field(16, ge=1, le=256, description="grow: rows per step")
    hold: float = Field(0.05, ge=0.0, lt=0.5, description="grow: fraction of the text held back")


@router.post("/{number}/continue", response_model=JobOut, status_code=202)
def continue_version(number: int, body: ContinueBody, request: Request, services=Depends(get_services)):
    """Queue a pretrain job that continues the version's newest run. The job's result is {job_id, label, version, kind, from_step}; the
    queued job's label says what it continues and from which step. A run that cannot be continued that way (no full checkpoint, a
    design its weights cannot go into) fails the job with a sentence."""
    _known(services, number)
    if body.kind == "new_data" and not body.dataset_ids:
        raise HTTPException(422, "Choose the text to continue on.")
    if body.kind == "grow" and not (body.spec or (body.width and body.layers)):
        raise HTTPException(422, "Say the bigger design: a spec, or a width and layers.")
    return _job(request, "versions.continue", lambda stop: services.continue_run(number, **body.model_dump()))


@router.get("/published", response_model=list[PublishedVersion])
def published_versions(services=Depends(get_services)):
    """The versions in the version store, newest first. 409 with a sentence on what to set up when no store is configured, or on what
    to do when it cannot be reached."""
    return services.published_versions()


@router.post("/{number}/publish", response_model=JobOut, status_code=202)
def publish_version(number: int, request: Request, services=Depends(get_services)):
    """Upload the version's weights (half precision, no optimizer state) and metadata to the version store. The job's result is
    {key, number, origin}. 409 with a sentence when no store is configured or the version has not trained."""
    version = next((v for v in services.versions() if v.number == number), None)
    if version is None:
        raise HTTPException(404, f"No version {number}")
    services.version_store()
    if not version.best_checkpoint:
        raise NoCheckpoint(number)
    return _job(request, "versions.publish", lambda stop: services.publish_version(number))


@router.post("/{number}/pull", response_model=JobOut, status_code=202)
def pull_version(number: int, request: Request, origin: str | None = None, services=Depends(get_services)):
    """Download a published version and add it here as a trained version (`number` is its number on the machine that published it;
    `origin` says which machine when several published one). The job's result is the new version, and its progress says where the
    weights came from. The machine that published it is asked first (`/weights`), when the workspace lists it among its nodes and it
    answers; with `origin` given the store is not read then. Otherwise, or when it cannot send it, they come from the store. 409 when
    no store is configured."""
    services.version_store()
    return _job(request, "versions.pull", lambda stop: services.pull_version(number, origin))


@router.get("/{number}/weights/meta", response_model=SharedVersion)
def version_weights_meta(number: int, services=Depends(get_services)):
    """The description of version `number`'s weights file (`GET /versions/{number}/weights`), for a peer that pulls it from this
    machine. The first call makes the half-precision copy and may take a while. 409 when the version has not trained."""
    _known(services, number)
    return services.share_version(number)[1]


@router.get("/{number}/weights", response_class=FileResponse)
def version_weights(number: int, services=Depends(get_services)):
    """Stream version `number`'s weights to a peer: half precision, no optimizer state, as `publish` would upload them. Describe
    them first with `/weights/meta`; the sha256 there is what the download must hash to."""
    _known(services, number)
    return FileResponse(services.share_version(number)[0], media_type="application/octet-stream", filename=f"version-{number}-weights.npz")


def _file_name(services, number, kind):
    name = next((v.name for v in services.versions() if v.number == number), "") or f"version-{number}"
    return f"{re.sub(r'[^A-Za-z0-9._-]+', '-', name).strip('-') or 'version'}-{kind}" + (".gguf" if kind == "gguf" else ".zip")


@router.get("/{number}/export", response_class=FileResponse)
def export_version(number: int, kind: Literal["hf", "gguf"], services=Depends(get_services)):
    """Download the trained version's best checkpoint: "hf" is a zip of the Hugging Face folder, "gguf" the .gguf file."""
    _known(services, number)
    tmp = Path(tempfile.mkdtemp(prefix="groundwork-export-"))
    cleanup = BackgroundTask(shutil.rmtree, tmp, ignore_errors=True)
    try:
        target = tmp / ("model" if kind == "hf" else "model.gguf")
        services.export_version(number, kind, str(target))
        if not target.exists():
            raise HTTPException(404, "The export wrote no file")
        if target.is_dir():
            archive = tmp / "model.zip"
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as zf:  # weights do not compress: skip the time
                for path in sorted(target.rglob("*")):
                    if path.is_file():
                        zf.write(path, path.relative_to(target).as_posix())
            target = archive
        return FileResponse(target, filename=_file_name(services, number, kind), background=cleanup)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
