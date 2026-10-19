"""GET /node: this machine as a cluster peer sees it (studio_cluster.NodeInfo.payload)."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel

import studio_cluster
from studio_api.app import get_services
from studio_api.version import app_version

router = APIRouter(tags=["node"])


class Node(BaseModel):
    hostname: str
    version: str
    device: str  # cuda | mps | cpu
    device_name: str
    memory_bytes: int
    busy: bool
    dtypes: list[str]


@router.get("/node", response_model=Node)
def node(services=Depends(get_services)):
    """Sync, so the GPU probe (cached after the first call) runs off the event loop."""
    return studio_cluster.own_info(busy=services.queue().job is not None, version=app_version()).payload()
