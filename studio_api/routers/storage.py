"""Storage: what a clean-up would delete, doing it, and the free space where checkpoints are written."""

from typing import Literal

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from studio_api.app import get_services

router = APIRouter(prefix="/storage", tags=["storage"])


class StorageUse(BaseModel):
    items: int
    bytes: int


class CleanRequest(BaseModel):
    days: int | None = Field(None, ge=0, description="delete runs older than this many days; null = all history")
    data: bool = Field(False, description="also delete your data (only with days null)")


class FreeSpace(BaseModel):
    bytes: int


class Leftover(BaseModel):
    id: str = Field(description="what to send back to remove it; a path under the workspace")
    kind: Literal["interrupted", "failed", "snapshot"]
    name: str
    belongs_to: str = Field(description="the job it came from, or why nothing needs it")
    bytes: int = Field(description="what removing it frees")
    shared: int = Field(description="bytes it holds that other runs still link to (not freed)")


class RemoveLeftovers(BaseModel):
    ids: list[str] = Field(min_length=1, max_length=1000)


class ExportRow(BaseModel):
    number: int
    name: str
    gguf_problem: str = Field("", description='why this design cannot be written as GGUF, in plain words; "" when it can')


@router.get("", response_model=StorageUse)
def storage(days: int | None = Query(None, ge=0), data: bool = False, services=Depends(get_services)):
    """What a clean-up with the same arguments would delete."""
    items, size = services.storage(days, data)
    return StorageUse(items=items, bytes=size)


@router.post("/clean", response_model=StorageUse)
def clean_storage(body: CleanRequest, services=Depends(get_services)):
    """Delete it. Returns what was deleted. 409 while a job is running."""
    items, size = services.clean_storage(body.days, body.data)
    return StorageUse(items=items, bytes=size)


@router.get("/leftovers", response_model=list[Leftover])
def leftovers(services=Depends(get_services)):
    """What training left in runs/ that no result needs, largest first: attempts moved aside by a restart, failed runs, and dataset
    snapshots no run uses. A running job's folder, the checkpoint a waiting job continues from and a version's runs are never listed."""
    return [Leftover(**row) for row in services.leftovers()]


@router.post("/leftovers/remove", response_model=StorageUse)
def remove_leftovers(body: RemoveLeftovers, services=Depends(get_services)):
    """Delete the listed leftovers. An id that is not listed any more (already gone, or protected since) is ignored."""
    items, size = services.remove_leftovers(body.ids)
    return StorageUse(items=items, bytes=size)


@router.get("/free", response_model=FreeSpace)
def free_space(services=Depends(get_services)):
    return FreeSpace(bytes=services.free_space())


@router.get("/exports", response_model=list[ExportRow])
def exports(services=Depends(get_services)):
    """The versions that have trained, so can be exported, each with the reason GGUF is unavailable for it (if it is)."""
    import studio_export
    from studio_model import effective_spec, validate_config
    return [ExportRow(number=v.number, name=v.name, gguf_problem=studio_export.gguf_problem(effective_spec(validate_config(v.spec))))
            for v in services.versions() if v.best_checkpoint]
