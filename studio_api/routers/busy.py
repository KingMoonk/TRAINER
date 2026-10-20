"""GET /busy: the heavy operations running right now (studio_ia.busy). The same list arrives on /events as a "busy" event."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from studio_api.app import get_services

router = APIRouter(tags=["busy"])


class BusyItem(BaseModel):
    id: str
    label: str = Field(description="what is happening, as a sentence fragment for the person")
    done: int = Field(ge=0, description="steps finished")
    total: int | None = Field(description="steps in all; null when the work has no measurable steps (an indeterminate bar)")


class BusyList(BaseModel):
    items: list[BusyItem]


@router.get("/busy", response_model=BusyList)
def busy(services=Depends(get_services)):
    """Everything busy now, oldest first. Empty when idle."""
    return BusyList(items=services.busy.items())
