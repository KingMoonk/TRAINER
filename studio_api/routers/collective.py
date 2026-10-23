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

import anyio

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel

import studio_diloco
import studio_parts
import studio_download_admission
import studio_upload_admission

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
    delta_sparsity: float = 0.  # the share of each tensor's entries a member's change carries (top-k by magnitude); 0: all of them
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


class DeltaAnnouncement(BaseModel):
    protocol: Literal[1] = 1
    node: str
    generation: int
    epoch: int
    tokens: int
    steps: int
    total: int
    sha256: str
    count: int
    part_size: int
    seconds: float = 0.
    loss: float | None = None
    down_bps: float = 0.


def admission_identity(found, run, number, body):
    validate_upload_member(found, number, body.node, body.generation, body.epoch)
    return transfer_identity(run, number, found.term, body.node, body.generation, body.epoch, body.tokens, body.steps,
                             body.seconds, body.loss, body.down_bps)


def validate_upload_member(found, number, node, generation, epoch):
    with found.cond:
        found._advance()
        try:
            found._admit(found.current, number, node, generation, epoch, found.term)
        except studio_diloco.Stale as exc:
            if not found._carries(found.current, number, node, generation, epoch, found.term):
                raise HTTPException(409, str(exc), headers={studio_diloco.TERM_HEADER: str(found.term)}) from None


def upload_admission(request: Request):
    return request.app.state.upload_admission


def admission_error(exc):
    status = 429 if "budget is full" in str(exc) or "admission is full" in str(exc) else 409
    raise HTTPException(status, str(exc)) from None


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
    if request is not None and request.headers.get("x-transfer-admission") == "1":
        return studio_download_admission.response(content, request, headers)
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


async def prepared_binary(request, found, produce):
    # Bulk serialization/hash work waits outside the default control worker pool.
    return await anyio.to_thread.run_sync(lambda: binary(found, produce(), request),
                                         limiter=request.app.state.download_build_limiter if request.headers.get("x-transfer-admission") == "1" else None)


@router.get("/params", response_class=Response, responses=BINARY)
async def params(request: Request, round: int, node: str = "", since: int | None = None, found: "studio_diloco.Coordinator" = Depends(member_call)):
    """The open round's global weights, fp32, as an .npz (a `Range` gets that part). `node`, who is asking, is noted on the round's
    timeline. A member that holds the weights of round `since` (the round before) gets the outer update that led to this round's
    instead, quantized to the run's delta_dtype, as an .npz marked with a `__update_from__` entry; with fp32 runs, or after a pivot or
    polish, it gets the weights."""
    try:
        span = request.headers.get("range")
        later = bool(span) and not span.replace(" ", "").lower().startswith("bytes=0-")  # a later part of a ranged download: counted once, with the first
        return await prepared_binary(request, found, lambda: found.params_bytes(round, node, since, count=not later))
    except studio_diloco.Gone as exc:
        raise HTTPException(410, str(exc)) from None


@router.get("/shards/{shard}", response_class=Response, responses=BINARY)
async def shard(shard: int, request: Request, found: "studio_diloco.Coordinator" = Depends(member_call)):
    """One shard; admitted readers share an immutable version and the process egress budget."""
    admitted = request.headers.get("x-transfer-admission") == "1"

    def prepare():
        try:
            found.ensure_serving()
            path = found.shard_path(shard)
        except studio_diloco.Gone as exc:
            raise HTTPException(410, str(exc)) from None
        except KeyError:
            raise HTTPException(404, f"No shard {shard}.") from None
        if admitted:
            return studio_download_admission.response(studio_download_admission.snapshot(path), request,
                                                      {studio_diloco.TERM_HEADER: str(found.term)})
        return StreamedFile(path, media_type="application/octet-stream", headers={studio_diloco.TERM_HEADER: str(found.term)})

    return await anyio.to_thread.run_sync(prepare, limiter=request.app.state.download_build_limiter if request.headers.get("x-transfer-admission") == "1" else None)


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
async def state(request: Request, round: int, found: "studio_diloco.Coordinator" = Depends(member_call)):
    """A replica's copy of what goes with theta of the latest round: the outer momentum (fp32) and diloco.json, in one .npz.
    409 when `round` is not the latest, 404 before any round has closed."""
    try:
        return await prepared_binary(request, found, lambda: found.state_bytes(round))
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from None


@router.get("/validation", response_class=Response, responses=BINARY)
async def validation(request: Request, found: "studio_diloco.Coordinator" = Depends(member_call)):
    """The validation windows as an .npz (the shard format), for a replica once per run. Replicas never train on them."""
    return await prepared_binary(request, found, found.validation_bytes)


@router.get("/weights", response_class=Response, responses=BINARY)
async def weights(request: Request, found: "studio_diloco.Coordinator" = Depends(member_call)):
    """A finished run's final weights (its last.npz), for the owner when the run finished here after a failover. 409 while rounds are left."""
    return await prepared_binary(request, found, found.final_bytes)


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
    upload_admission(request)._emit("bypass", run=request.path_params["run"], round=round, node=node, bytes=len(body))
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
    upload_admission(request)._emit("bypass", run=request.path_params["run"], round=round, node=node, bytes=total)
    speed = rates(total, took, down_bps)

    def apply():
        found.submit(round, node, generation, epoch, tokens, steps, studio_diloco.decode_arrays(whole), seconds, loss, len(whole), found.term, speed)
    await run_in_threadpool(apply)
    return Accepted(term=found.term)


@router.post("/rounds/{round}/delta/transfers")
def announce_delta(request: Request, round: int, body: DeltaAnnouncement,
                   run: str = Run,
                   found: "studio_diloco.Coordinator" = Depends(member_call)):
    """Opt in to bounded receiver grants for one immutable encoded delta."""
    if body.steps < 1 or body.tokens < 0 or body.seconds < 0:
        raise HTTPException(422, "Delta counters must be non-negative and steps must be positive.")
    manager = upload_admission(request)
    identity = transfer_identity(run, round, found.term, body.node, body.generation, body.epoch, body.tokens, body.steps,
                                 body.seconds, body.loss, body.down_bps)
    try:
        existing = manager.resume(identity, body.total, body.sha256, body.count, body.part_size)
    except studio_upload_admission.AdmissionError as exc:
        admission_error(exc)
    if existing is not None:
        if existing["complete"]:
            return existing
        try:
            admission_identity(found, run, round, body)
        except HTTPException:
            manager.cancel_transfer(existing["transfer_id"], existing["incarnation"])
            raise
        manager.fence_member(run, body.node, body.generation, body.epoch, found.term)
        return existing
    identity = admission_identity(found, run, round, body)
    manager.fence_member(run, body.node, body.generation, body.epoch, found.term)
    try:
        return manager.announce(identity, body.total, body.sha256, body.count, body.part_size)
    except studio_upload_admission.AdmissionError as exc:
        admission_error(exc)


def transfer_identity(run, round, term, node, generation, epoch, tokens, steps, seconds, loss, down_bps):
    return (run, term, round, node, generation, epoch, tokens, steps, seconds, loss, down_bps)


async def apply_granted_delta(request, found, manager, transfer_id, incarnation, identity, number, node, generation, epoch,
                              tokens, steps, seconds, loss, down_bps):
    def apply():
        # Claim inside the worker, never before an await which can be cancelled.
        # Concurrent POST retries/status polls observe the same application owner.
        pending = manager.pending(transfer_id, incarnation, identity)
        if pending is None:
            return
        claimed_body, claimed_elapsed = pending
        pending = None
        failed = True
        try:
            delta = studio_diloco.decode_arrays(claimed_body)
            speed = rates(len(claimed_body), claimed_elapsed, down_bps)
            with found.cond:
                found.submit(number, node, generation, epoch, tokens, steps, delta, seconds,
                             loss, len(claimed_body), found.term, speed)
                manager.mark_complete(transfer_id, incarnation, identity)
            failed = False
        finally:
            claimed_body = None
            manager.end_apply(transfer_id, failed=failed)

    try:
        await anyio.to_thread.run_sync(apply, limiter=request.app.state.upload_apply_limiter)
    except studio_diloco.Stale as exc:
        raise HTTPException(409, str(exc), headers={studio_diloco.TERM_HEADER: str(found.term)}) from None


@router.get("/rounds/{round}/delta/transfers/capabilities")
def delta_transfer_capabilities(request: Request, round: int, term: int = Term,
                                found: "studio_diloco.Coordinator" = Depends(member_call)):
    return upload_admission(request).capabilities()


@router.get("/rounds/{round}/delta/transfers/{transfer_id}/grants")
async def delta_grants(request: Request, transfer_id: str, round: int, incarnation: str, node: str, generation: int,
                       epoch: int, tokens: int, steps: int, term: int = Term, seconds: float = 0., loss: float | None = None,
                       down_bps: float = 0., run: str = Run,
                       found: "studio_diloco.Coordinator" = Depends(member_call)):
    identity = transfer_identity(run, round, found.term, node, generation, epoch, tokens, steps, seconds, loss, down_bps)
    manager = upload_admission(request)
    try:
        if manager.ready_to_apply(transfer_id, incarnation, identity):
            await apply_granted_delta(request, found, manager, transfer_id, incarnation, identity, round, node,
                                      generation, epoch, tokens, steps, seconds, loss, down_bps)
        if not manager.completed(transfer_id, incarnation, identity):
            try:
                await run_in_threadpool(validate_upload_member, found, round, node, generation, epoch)
            except HTTPException:
                if not manager.completed(transfer_id, incarnation, identity):
                    manager.cancel_transfer(transfer_id, incarnation, reason="stale")
                    raise
            manager.fence_member(run, node, generation, epoch, found.term)
        return manager.grant(transfer_id, incarnation, identity, priority=request.headers.get("x-transfer-priority") == "required")
    except studio_upload_admission.AdmissionError as exc:
        admission_error(exc)


@router.delete("/rounds/{round}/delta/transfers/{transfer_id}", status_code=204)
def cancel_delta(request: Request, transfer_id: str, round: int, incarnation: str, node: str, generation: int,
                 epoch: int, tokens: int, steps: int, term: int = Term, seconds: float = 0., loss: float | None = None,
                 down_bps: float = 0., run: str = Run,
                 found: "studio_diloco.Coordinator" = Depends(member_call)):
    identity = transfer_identity(run, round, found.term, node, generation, epoch, tokens, steps, seconds, loss, down_bps)
    manager = upload_admission(request)
    try:
        manager.completed(transfer_id, incarnation, identity)  # bind cancellation to the immutable identity
        manager.cancel_transfer(transfer_id, incarnation)
    except studio_upload_admission.AdmissionError as exc:
        admission_error(exc)
    return Response(status_code=204)


@router.post("/rounds/{round}/delta/transfers/{transfer_id}/parts", status_code=202, response_model=Accepted,
             openapi_extra=OCTET_BODY)
async def granted_delta_part(request: Request, transfer_id: str, round: int, incarnation: str, grant_id: str,
                             part: int, size: int, part_sha256: str, node: str, generation: int, epoch: int,
                             tokens: int, steps: int, term: int = Term, seconds: float = 0., loss: float | None = None,
                             down_bps: float = 0., run: str = Run,
                             found: "studio_diloco.Coordinator" = Depends(member_call)):
    identity = transfer_identity(run, round, found.term, node, generation, epoch, tokens, steps, seconds, loss, down_bps)
    manager = upload_admission(request)
    receiving = False
    body_buffer = body = finished = None
    try:
        disposition = manager.begin_part(transfer_id, incarnation, identity, grant_id, part, size)
        if disposition == "duplicate":
            return Accepted(term=found.term)
        receiving = True
        try:
            await run_in_threadpool(validate_upload_member, found, round, node, generation, epoch)
            manager.fence_member(run, node, generation, epoch, found.term)
        except HTTPException:
            manager.cancel_transfer(transfer_id, incarnation, reason="stale")
            raise
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                declared = int(content_length)
            except ValueError:
                raise studio_upload_admission.AdmissionError("Content-Length is invalid.") from None
            if declared != size:
                raise studio_upload_admission.AdmissionError("Content-Length does not match the granted part size.")
        body_buffer = bytearray()
        async with asyncio.timeout(studio_upload_admission.RECEIVE_TIMEOUT):
            async for chunk in request.stream():
                if len(body_buffer) + len(chunk) > size:
                    raise studio_upload_admission.AdmissionError("Request body exceeds the granted part size.")
                body_buffer.extend(chunk)
        if len(body_buffer) != size:
            raise studio_upload_admission.AdmissionError("Request body does not match the granted part size.")
        body = bytes(body_buffer)
        body_buffer = None
        finished = await run_in_threadpool(manager.finish_part, transfer_id, incarnation, identity, grant_id, part, body, part_sha256)
        body = None
        if finished is not None:
            finished = None
            await apply_granted_delta(request, found, manager, transfer_id, incarnation, identity, round, node,
                                      generation, epoch, tokens, steps, seconds, loss, down_bps)
        return Accepted(term=found.term)
    except TimeoutError:
        raise HTTPException(408, "Granted part receive deadline exceeded.") from None
    except studio_upload_admission.AdmissionError as exc:
        admission_error(exc)
    finally:
        body_buffer = body = finished = None
        if receiving:
            manager.end_part(grant_id)


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
