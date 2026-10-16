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
    step: int | None = None
    published_at: str | None = None
    size: int | None = Field(None, description="bytes of the weights file in the store")
    scorecard: dict = Field(default_factory=dict, description="as measured on the machine that trained it")
    mine: bool = Field(description="this machine published it")
    local: bool = Field(description="this workspace already has it as a version")
    local_number: int | None = Field(None, description="the version number it has here")


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
    `origin` says which machine when several published one). The job's result is the new version. 409 when no store is configured."""
    services.version_store()
    return _job(request, "versions.pull", lambda stop: services.pull_version(number, origin))


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
