"""The DiLoCo wire protocol (docs/distributed.md, "Wire protocol"): what members of a distributed run call on the coordinator's node.

The coordinator for a run (studio_diloco.Coordinator) lives in this process, registered by the owner's pretrain_distributed job.
Weights, shards and pseudo-gradients are binary .npz bodies; a request from an earlier round, generation, lease epoch or term is
409 and changes nothing. Every call carries the run's `term` in its query (0 on the first join, before a member knows it); every
answer carries the coordinator's term, in the X-Run-Term header and in its JSON body. A call with a higher term than the
coordinator's tells it a later coordinator exists: it steps down (Coordinator.step_down) and refuses the call. Node-to-node authentication is the app's (studio_api.auth), not this router's.
"""

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request, Response
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

import studio_diloco

router = APIRouter(prefix="/collective/{run}", tags=["collective"])
Run = Path(pattern=studio_diloco.RUN_ID, description="the run id: the pretrain_distributed job's id")
Term = Query(description="the run's term as the caller knows it; lower than the coordinator's is 409, higher makes it step down. 0 only on /join", ge=0)
BINARY = {200: {"content": {"application/octet-stream": {"schema": {"type": "string", "format": "binary"}}}}}


class Versioned(BaseModel):
    term: int = 1  # the coordinator's term, on every JSON answer


class JoinRequest(BaseModel):
    node: str
    epoch: int = 0  # the member's scheduler lease epoch


class LeaveRequest(BaseModel):
    node: str
    generation: int
    epoch: int


class ProgressRequest(BaseModel):
    """Where a member is in its round: no weights. `round` is the round it is working on or about to work on."""

    node: str
    generation: int
    epoch: int
    round: int
    phase: Literal["downloading", "training", "uploading", "waiting"]
    step: int = 0  # inner steps done in the round
    loss: float | None = None
    tokens_per_second: float | None = None


class JoinInfo(Versioned):
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


class RoundView(Versioned):
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
    replica: bool = False  # this member is one of the run's replicas: it also takes GET /state each round and GET /validation once
    lr_scale: float = 1.  # the factor the run's pivots (halve_lr) put on the learning rate from this round on
    steps: int = 0  # the inner steps dealt to this member for the round (0: the run's H, as an older coordinator says)
    paused: bool = False  # the owner paused the run: a pending round stays pending until it is resumed
    polishing: bool = False  # the teacher is polishing the weights: a pending round stays pending until it is done


class Accepted(Versioned):
    accepted: bool = True


def coordinator(run: str, request: Request) -> "studio_diloco.Coordinator":
    """The run's coordinator in this app's registry: the app's own `state.coordinators` when it has one (an in-process cluster of several
    nodes in the tests gives each its own), else the process's (studio_diloco.COORDINATORS)."""
    registry = getattr(request.app.state, "coordinators", None)
    found = (studio_diloco.COORDINATORS if registry is None else registry).get(run)
    if found is None:
        raise HTTPException(404, f"No distributed run {run!r} on this node.")
    return found


def admit(run: str, term: int, response: Response, request: Request, learning: bool = False) -> "studio_diloco.Coordinator":
    """The run's coordinator, once the call's term is admitted; its term goes on the answer. A refused term is 409 with the term too."""
    found = coordinator(run, request)
    try:
        found.check_term(term, learning)
    except studio_diloco.StaleTerm as exc:
        raise HTTPException(409, str(exc), headers={studio_diloco.TERM_HEADER: str(found.term)}) from None
    response.headers[studio_diloco.TERM_HEADER] = str(found.term)
    return found


def member_call(request: Request, response: Response, run: str = Run, term: int = Term) -> "studio_diloco.Coordinator":
    return admit(run, term, response, request)


def binary(found: "studio_diloco.Coordinator", content: bytes) -> Response:
    return Response(content, media_type="application/octet-stream", headers={studio_diloco.TERM_HEADER: str(found.term)})


@router.post("/join", response_model=JoinInfo)
def join(request: Request, response: Response, run: str = Run, term: int = Term, body: JoinRequest = ...):
    """Join (or re-join with a newer epoch). Takes effect at the next round boundary, at once when no round is open.
    A member that does not know the term yet sends 0 and learns it from the answer."""
    return admit(run, term, response, request, learning=True).join(body.node, body.epoch, term or None)


@router.get("/params", response_class=Response, responses=BINARY)
def params(round: int, node: str = "", found: "studio_diloco.Coordinator" = Depends(member_call)):
    """The open round's global weights, fp32, as an .npz. `node`, who is asking, is noted on the round's timeline."""
    return binary(found, found.params_bytes(round, node))


@router.get("/shards/{shard}", response_class=Response, responses=BINARY)
def shard(shard: int, found: "studio_diloco.Coordinator" = Depends(member_call)):
    """One training shard (tokens, target flags and offsets) as an .npz."""
    try:
        return binary(found, found.shard_bytes(shard))
    except KeyError:
        raise HTTPException(404, f"No shard {shard}.") from None


@router.get("/state", response_class=Response, responses=BINARY)
def state(round: int, found: "studio_diloco.Coordinator" = Depends(member_call)):
    """A replica's copy of what goes with theta of the latest round: the outer momentum (fp32) and diloco.json, in one .npz.
    409 when `round` is not the latest, 404 before any round has closed."""
    try:
        return binary(found, found.state_bytes(round))
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from None


@router.get("/validation", response_class=Response, responses=BINARY)
def validation(found: "studio_diloco.Coordinator" = Depends(member_call)):
    """The validation windows as an .npz (the shard format), for a replica once per run. Replicas never train on them."""
    return binary(found, found.validation_bytes())


@router.get("/weights", response_class=Response, responses=BINARY)
def weights(found: "studio_diloco.Coordinator" = Depends(member_call)):
    """A finished run's final weights (its last.npz), for the owner when the run finished here after a failover. 409 while rounds are left."""
    return binary(found, found.final_bytes())


@router.post("/rounds/{round}/delta", status_code=202, response_model=Accepted,
             openapi_extra={"requestBody": {"required": True, "content": {"application/octet-stream": {"schema": {"type": "string", "format": "binary"}}}}})
async def delta(request: Request, round: int, node: str, generation: int, epoch: int, tokens: int, steps: int,
                seconds: float = 0., loss: float | None = None, found: "studio_diloco.Coordinator" = Depends(member_call)):
    """A member's pseudo-gradient (theta - theta_k) as an .npz, with its target-token count and steps. 409 when stale."""
    body = await request.body()

    def apply():
        found.submit(round, node, generation, epoch, tokens, steps, studio_diloco.decode_arrays(body), seconds, loss, len(body), found.term)
    await run_in_threadpool(apply)
    return Accepted(term=found.term)


@router.post("/progress", response_model=Accepted)
def progress(found: "studio_diloco.Coordinator" = Depends(member_call), body: ProgressRequest = ...):
    """A member's report of its phase, step in the round, inner loss and speed, about every 2 s. 409 when stale, as a delta is.
    `accepted` is false when the coordinator was busy closing a round and took nothing: the next report says it all again."""
    return Accepted(term=found.term, accepted=found.progress(body.round, body.node, body.generation, body.epoch, body.phase, body.step,
                                                             body.loss, body.tokens_per_second, found.term))


@router.get("/rounds/{round}", response_model=RoundView)
def round_state(round: int, node: str = "", wait: float = 0., found: "studio_diloco.Coordinator" = Depends(member_call)):
    """Where a round stands for a member. With `wait` it long-polls (at most 30 s) until that changes."""
    return found.round_view(round, node, wait)


@router.post("/leave", status_code=204)
def leave(found: "studio_diloco.Coordinator" = Depends(member_call), body: LeaveRequest = ...):
    """Leave at the next round boundary; the open round stops waiting for this member."""
    found.leave(body.node, body.generation, body.epoch, found.term)
    return Response(status_code=204, headers={studio_diloco.TERM_HEADER: str(found.term)})
