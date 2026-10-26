"""GET /node: this machine as a cluster peer sees it (studio_cluster.NodeInfo.payload)."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel

import studio_cluster
from studio_api.app import get_services
from studio_api.routers.pretraining import Hardware
from studio_api.version import app_version

router = APIRouter(tags=["node"])


class Speed(BaseModel):
    device: str  # the device this node's own auto picks for the design: cpu | cuda | mps
    seconds: dict[str, float]  # measured seconds per step on each device that could be timed
    tokens: int  # tokens one step read
    age_s: float  # how long ago it was measured


class Node(BaseModel):
    hostname: str
    version: str
    device: str  # cuda | mps | cpu
    device_name: str
    memory_bytes: int
    busy: bool
    dtypes: list[str]
    speeds: dict[str, Speed] = {}  # design shape (a hash) -> this machine's measured step time for it; only probes valid for this machine as it is now
    profile: Hardware | None = None  # what this machine has free now (studio_fit.hardware_profile): the coordinator of a distributed run sizes this member from it; null where it cannot be read
    memory_share: float | None = None  # the part of its memory this machine's queue plan lets a job take: the coordinator sizes this member's batch against that share


@router.get("/node", response_model=Node)
def node(services=Depends(get_services)):
    """Sync, so the GPU probe (cached after the first call) runs off the event loop."""
    profile = services.hardware_profile()
    return studio_cluster.own_info(busy=services.queue().job is not None, version=app_version(), profile=profile,
                                   memory_share=services.memory_share()).payload()
