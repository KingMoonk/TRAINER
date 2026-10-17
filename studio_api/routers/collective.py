"""The DiLoCo wire protocol (docs/distributed.md, "Wire protocol"): what members of a distributed run call on the coordinator's node.

The coordinator for a run (studio_diloco.Coordinator) lives in this process, registered by the owner's pretrain_distributed job.
Weights, shards and pseudo-gradients are binary .npz bodies; a request from an earlier round, generation or lease epoch is 409
and changes nothing. Node-to-node authentication is the app's (studio_api.auth), not this router's.
"""

from typing import Any

from fastapi import APIRouter, HTTPException, Path, Request, Response
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

import studio_diloco

router = APIRouter(prefix="/collective/{run}", tags=["collective"])
Run = Path(pattern=studio_diloco.RUN_ID, description="the run id: the pretrain_distributed job's id")
BINARY = {200: {"content": {"application/octet-stream": {"schema": {"type": "string", "format": "binary"}}}}}


class JoinRequest(BaseModel):
    node: str
    epoch: int = 0  # the member's scheduler lease epoch


class LeaveRequest(BaseModel):
    node: str
    generation: int
    epoch: int


class JoinInfo(BaseModel):
    """What a member needs to start: the config and tokenizer, and the run's shape. `round` is the round to ask GET /rounds/{r} about."""

    node: str
    generation: int
    round: int
    config: dict[str, Any]
    tokenizer: dict[str, Any]
    inner_steps: int
    rounds: int
    total_steps: int
    delta_dtype: str
    shard_count: int
    seed: int
    vocab_size: int


class RoundView(BaseModel):
    """state: open (this member's assignment is filled in), pending (not open yet, or this member joins at a later boundary),
    closed (`current` is the round to ask about) or finished. member false means the member must POST /join again."""

    round: int
    state: str
    member: bool
    current: int
    generation: int | None = None
    rank: int | None = None
    deadline_s: float | None = None
    shards: list[int] | None = None
    members: int | None = None


class Accepted(BaseModel):
    accepted: bool = True


def coordinator(run: str) -> "studio_diloco.Coordinator":
    found = studio_diloco.COORDINATORS.get(run)
    if found is None:
        raise HTTPException(404, f"No distributed run {run!r} on this node.")
    return found


@router.post("/join", response_model=JoinInfo)
def join(run: str = Run, body: JoinRequest = ...):
    """Join (or re-join with a newer epoch). Takes effect at the next round boundary, at once when no round is open."""
    return coordinator(run).join(body.node, body.epoch)


@router.get("/params", response_class=Response, responses=BINARY)
def params(round: int, run: str = Run):
    """The open round's global weights, fp32, as an .npz."""
    return Response(coordinator(run).params_bytes(round), media_type="application/octet-stream")


@router.get("/shards/{shard}", response_class=Response, responses=BINARY)
def shard(shard: int, run: str = Run):
    """One training shard (tokens, target flags and offsets) as an .npz."""
    try:
        return Response(coordinator(run).shard_bytes(shard), media_type="application/octet-stream")
    except KeyError:
        raise HTTPException(404, f"No shard {shard}.") from None


@router.post("/rounds/{round}/delta", status_code=202, response_model=Accepted,
             openapi_extra={"requestBody": {"required": True, "content": {"application/octet-stream": {"schema": {"type": "string", "format": "binary"}}}}})
async def delta(request: Request, round: int, node: str, generation: int, epoch: int, tokens: int, steps: int,
                seconds: float = 0., loss: float | None = None, run: str = Run):
    """A member's pseudo-gradient (theta - theta_k) as an .npz, with its target-token count and steps. 409 when stale."""
    found = coordinator(run)
    body = await request.body()

    def apply():
        found.submit(round, node, generation, epoch, tokens, steps, studio_diloco.decode_arrays(body), seconds, loss, len(body))
    await run_in_threadpool(apply)
    return Accepted()


@router.get("/rounds/{round}", response_model=RoundView)
def round_state(round: int, node: str = "", wait: float = 0., run: str = Run):
    """Where a round stands for a member. With `wait` it long-polls (at most 30 s) until that changes."""
    return coordinator(run).round_view(round, node, wait)


@router.post("/leave", status_code=204)
def leave(run: str = Run, body: LeaveRequest = ...):
    """Leave at the next round boundary; the open round stops waiting for this member."""
    coordinator(run).leave(body.node, body.generation, body.epoch)
    return Response(status_code=204)
