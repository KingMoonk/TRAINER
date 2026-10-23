"""The DiLoCo wire protocol (docs/distributed.md, "Wire protocol"): what members of a distributed run call on the coordinator's node.

The coordinator for a run (studio_diloco.Coordinator) lives in this process, registered by the owner's pretrain_distributed job.
Weights, shards and pseudo-gradients are binary .npz bodies; a request from an earlier round, generation, lease epoch or term is
409 and changes nothing. Every call carries the run's `term` in its query (0 on the first join, before a member knows it); every
answer carries the coordinator's term, in the X-Run-Term header and in its JSON body. A call with a higher term than the
coordinator's tells it a later coordinator exists: it steps down (Coordinator.step_down) and refuses the call. Node-to-node authentication is the app's (studio_api.auth), not this router's.
"""

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel

import studio_diloco
import studio_parts

router = APIRouter(prefix="/collective/{run}", tags=["collective"])
# A long poll holds its thread for up to 30 s per member: its own pool keeps them from filling the threads every other plain `def` route shares.
LONG_POLLS = ThreadPoolExecutor(max_workers=64, thread_name_prefix="api-long-poll")
Run = Path(pattern=studio_diloco.RUN_ID, description="the run id: the pretrain_distributed job's id")
Term = Query(description="the run's term as the caller knows it; lower than the coordinator's is 409, higher makes it step down. 0 only on /join", ge=0)
BINARY = {200: {"content": {"application/octet-stream": {"schema": {"type": "string", "format": "binary"}}}}}


class StreamedFile(FileResponse):
    chunk_size = 1 << 20  # Starlette's 64 KiB makes a thread hop per chunk: slow for a 20 MB shard over a long link


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
    phase: Literal["downloading", "loading", "training", "uploading", "waiting"]
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
    plan: list[dict[str, Any]] | None = None  # the run's changes of length and H (studio_diloco.plan_h); none from an older coordinator
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
    rounds: int | None = None  # the run's rounds as it stands (the owner can extend it); the plan below says how the rounds are laid out
    plan: list[dict[str, Any]] | None = None  # the run's changes of length and H (studio_diloco.plan_h); a member takes them from every round it opens
    paused: bool = False  # the owner paused the run: a pending round stays pending until it is resumed
    polishing: bool = False  # the teacher is polishing the weights: a pending round stays pending until it is done


class Accepted(Versioned):
    accepted: bool = True


class Progressed(Accepted):
    plan: list[dict]  # the run's plan (studio_diloco.plan_h): a member training a round learns of an extension from it, for its replica


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


def binary(found: "studio_diloco.Coordinator", content: bytes, request: Request | None = None) -> Response:
    """A binary body. With the request's `Range: bytes=a-b` it answers that part (206, Content-Range: the body's size is how a member
    learns to fetch the parts at the same time, studio_peer.PeerClient); without one, the whole body."""
    headers = {studio_diloco.TERM_HEADER: str(found.term), "Accept-Ranges": "bytes"}
    try:
        span = studio_parts.byte_range(request.headers.get("range") if request is not None else None, len(content))
    except studio_parts.BadPart as exc:
        raise HTTPException(416, str(exc), headers={"Content-Range": f"bytes */{len(content)}"}) from None
    if span is None:
        return Response(content, media_type="application/octet-stream", headers=headers)
    first, last = span
    return Response(content[first:last + 1], status_code=206, media_type="application/octet-stream",
                    headers={**headers, "Content-Range": f"bytes {first}-{last}/{len(content)}"})


PARTS = threading.Lock()
parts_clock = time.monotonic  # studio_parts.Assembler's clock


def assembler(found: "studio_diloco.Coordinator") -> "studio_parts.Assembler":
    """The run's parts of changes being posted (kept on its coordinator, so a coordinator that goes takes them with it)."""
    with PARTS:
        if getattr(found, "upload_parts", None) is None:
            found.upload_parts = studio_parts.Assembler()
        return found.upload_parts


@router.post("/join", response_model=JoinInfo)
def join(request: Request, response: Response, run: str = Run, term: int = Term, body: JoinRequest = ...):
    """Join (or re-join with a newer epoch). Takes effect at the next round boundary, at once when no round is open.
    A member that does not know the term yet sends 0 and learns it from the answer."""
    return admit(run, term, response, request, learning=True).join(body.node, body.epoch, term or None)


@router.get("/params", response_class=Response, responses=BINARY)
def params(request: Request, round: int, node: str = "", since: int | None = None, found: "studio_diloco.Coordinator" = Depends(member_call)):
    """The open round's global weights, fp32, as an .npz (a `Range` gets that part). `node`, who is asking, is noted on the round's
    timeline. A member that holds the weights of round `since` (the round before) gets the outer update that led to this round's
    instead, quantized to the run's delta_dtype, as an .npz marked with a `__update_from__` entry; with fp32 runs, or after a pivot or
    polish, it gets the weights."""
    try:
        span = request.headers.get("range")
        later = bool(span) and not span.replace(" ", "").lower().startswith("bytes=0-")  # a later part of a ranged download: counted once, with the first
        return binary(found, found.params_bytes(round, node, since, count=not later), request)
    except studio_diloco.Gone as exc:
        raise HTTPException(410, str(exc)) from None


@router.get("/shards/{shard}", response_class=Response, responses=BINARY)
def shard(shard: int, found: "studio_diloco.Coordinator" = Depends(member_call)):
    """One training shard (tokens, target flags and offsets) as an .npz; a `Range` gets that part (a file response answers it)."""
    try:
        found.ensure_serving()
        path = found.shard_path(shard)
    except studio_diloco.Gone as exc:
        raise HTTPException(410, str(exc)) from None
    except KeyError:
        raise HTTPException(404, f"No shard {shard}.") from None
    return StreamedFile(path, media_type="application/octet-stream", headers={studio_diloco.TERM_HEADER: str(found.term)})


@router.get("/shards/{shard}/hash")
def shard_hash(shard: int, found: "studio_diloco.Coordinator" = Depends(member_call)) -> dict[str, str]:
    """The sha256 of one shard's .npz body, so a member that holds it from an earlier run skips the download."""
    try:
        found.ensure_serving()
        return {"hash": found.shard_hash(shard)}
    except studio_diloco.Gone as exc:
        raise HTTPException(410, str(exc)) from None
    except KeyError:
        raise HTTPException(404, f"No shard {shard}.") from None


@router.get("/state", response_class=Response, responses=BINARY)
def state(request: Request, round: int, found: "studio_diloco.Coordinator" = Depends(member_call)):
    """A replica's copy of what goes with theta of the latest round: the outer momentum (fp32) and diloco.json, in one .npz.
    409 when `round` is not the latest, 404 before any round has closed."""
    try:
        return binary(found, found.state_bytes(round), request)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from None


@router.get("/validation", response_class=Response, responses=BINARY)
def validation(request: Request, found: "studio_diloco.Coordinator" = Depends(member_call)):
    """The validation windows as an .npz (the shard format), for a replica once per run. Replicas never train on them."""
    return binary(found, found.validation_bytes(), request)


@router.get("/weights", response_class=Response, responses=BINARY)
def weights(request: Request, found: "studio_diloco.Coordinator" = Depends(member_call)):
    """A finished run's final weights (its last.npz), for the owner when the run finished here after a failover. 409 while rounds are left."""
    return binary(found, found.final_bytes(), request)


OCTET_BODY = {"requestBody": {"required": True, "content": {"application/octet-stream": {"schema": {"type": "string", "format": "binary"}}}}}


def rates(size: int, seconds: float, down_bps: float) -> dict[str, float]:
    """What a member's transfers of the round achieved, bytes per second: `down` as it reported (its weights download), `up` as seen here."""
    return {"down": max(0., down_bps), "up": size / seconds if seconds > 1e-6 else 0.}


@router.post("/rounds/{round}/delta", status_code=202, response_model=Accepted, openapi_extra=OCTET_BODY)
async def delta(request: Request, round: int, node: str, generation: int, epoch: int, tokens: int, steps: int,
                seconds: float = 0., loss: float | None = None, down_bps: float = 0.,
                found: "studio_diloco.Coordinator" = Depends(member_call)):
    """A member's pseudo-gradient (theta - theta_k) as an .npz, with its target-token count and steps. 409 when stale.
    `down_bps` is how fast the member's weights came down this round, for the round's health."""
    began = time.monotonic()
    body = await request.body()
    speed = rates(len(body), time.monotonic() - began, down_bps)

    def apply():
        found.submit(round, node, generation, epoch, tokens, steps, studio_diloco.decode_arrays(body), seconds, loss, len(body), found.term, speed)
    await run_in_threadpool(apply)
    return Accepted(term=found.term)


@router.post("/rounds/{round}/delta/parts", status_code=202, response_model=Accepted, openapi_extra=OCTET_BODY)
async def delta_part(request: Request, round: int, node: str, generation: int, epoch: int, tokens: int, steps: int, count: int,
                     total: int, sha256: str, part: int = 0, part_sha256: str = "", seconds: float = 0., loss: float | None = None,
                     down_bps: float = 0., found: "studio_diloco.Coordinator" = Depends(member_call)):
    """One part of a member's pseudo-gradient, posted with the others at the same time (a long link holds one stream to a fraction of
    its speed). `part` of `count` parts, the whole body `total` bytes with hash `sha256`; the part carries its own `part_sha256`.
    The last part to arrive completes the body, which is checked against `sha256` and then taken as `delta` takes one (409 when stale,
    400 when a part or the whole fails its hash: the member sends it again). `count=0` takes nothing: it asks whether this route exists."""
    if count == 0:
        return Accepted(term=found.term)
    began = parts_clock()
    body = await request.body()
    key = (round, node, generation, epoch)
    try:
        whole, took = assembler(found).add(key, part, count, total, sha256, body, part_sha256, began)
    except studio_parts.BadPart as exc:
        raise HTTPException(400, str(exc)) from None
    if whole is None:
        return Accepted(term=found.term)
    speed = rates(total, took, down_bps)

    def apply():
        found.submit(round, node, generation, epoch, tokens, steps, studio_diloco.decode_arrays(whole), seconds, loss, len(whole), found.term, speed)
    await run_in_threadpool(apply)
    return Accepted(term=found.term)


@router.post("/progress", response_model=Progressed)
def progress(found: "studio_diloco.Coordinator" = Depends(member_call), body: ProgressRequest = ...):
    """A member's report of its phase, step in the round, inner loss and speed, about every 2 s. 409 when stale, as a delta is.
    `accepted` is false when the coordinator was busy closing a round and took nothing: the next report says it all again."""
    taken = found.progress(body.round, body.node, body.generation, body.epoch, body.phase, body.step, body.loss, body.tokens_per_second, found.term)
    return Progressed(term=found.term, accepted=taken, plan=[dict(entry) for entry in found.plan])


@router.get("/rounds/{round}", response_model=RoundView)
async def round_state(round: int, node: str = "", wait: float = 0., found: "studio_diloco.Coordinator" = Depends(member_call)):
    """Where a round stands for a member. With `wait` it long-polls (at most 30 s) until that changes."""
    return await asyncio.get_running_loop().run_in_executor(LONG_POLLS, found.round_view, round, node, wait)


@router.post("/leave", status_code=204)
def leave(found: "studio_diloco.Coordinator" = Depends(member_call), body: LeaveRequest = ...):
    """Leave at the next round boundary; the open round stops waiting for this member."""
    found.leave(body.node, body.generation, body.epoch, found.term)
    return Response(status_code=204, headers={studio_diloco.TERM_HEADER: str(found.term)})
