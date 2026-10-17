"""The observation store's node-to-node call (docs/distributed.md, "The observation store"): `POST /state/sync` takes the caller's
version vector and the entries it pushes, and answers with every entry newer than that vector and this node's own vector. It sits
behind the same node-to-node auth as the other peer routes (studio_api.auth). Nothing in the store is written by a route: facts
come from their origin, and arrive here by sync."""

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from studio_api.routers.cluster import Cluster

router = APIRouter(prefix="/state", tags=["state"])


class StateEntry(BaseModel):
    origin: str
    kind: str
    key: str
    incarnation: int = Field(ge=0)
    seq: int = Field(ge=1)
    age: float = Field(ge=0)  # seconds since the origin wrote it, as the sender counts them
    value: Any


class SyncBody(BaseModel):
    vector: dict[str, tuple[int, int]] = {}  # origin -> (incarnation, seq) of the newest entry the caller holds
    entries: list[StateEntry] = []  # what the caller pushes: entries it holds that the peer lacks


class SyncReply(BaseModel):
    entries: list[StateEntry]
    vector: dict[str, tuple[int, int]]


@router.post("/sync")
def sync(body: SyncBody, cluster: Cluster) -> SyncReply:
    """Push-pull anti-entropy. An empty vector gets the whole state."""
    return SyncReply(**cluster.state.handle_sync(body.model_dump()))
