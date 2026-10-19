"""GET /events: every Services.emit(kind, payload) as a server-sent event, to every subscriber."""

import asyncio
import dataclasses
import datetime
import json

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

KEEPALIVE = 15.0


def _plain(value):
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, (datetime.datetime, datetime.date)):
        return value.isoformat()
    if isinstance(value, tuple) and hasattr(value, "_asdict"):
        return value._asdict()
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def encode(payload):
    return json.dumps(payload, default=_plain)


def sse(kind, payload):
    """One unnamed SSE message whose data is {"kind", "payload"}: EventSource.onmessage in the browser sees every kind."""
    return f"data: {encode({'kind': kind, 'payload': payload})}\n\n"


def status_snapshot(services):
    """The shell's current state, including a truthful snapshot for a newly connected browser."""
    version, queue = services.current_version(), services.queue()
    try:
        disk = services.free_space() / 1_000_000_000
    except (OSError, NotImplementedError):
        disk = None
    return {
        "version": f"{version.name or f'Version {version.number}'} · {version.status}" if version else None,
        "queue": {"state": "paused" if queue.paused else "running" if queue.job else "idle",
                  "running": int(queue.job is not None), "waiting": sum(job.status == "waiting" for job in queue.up_next)},
        "resources": {"device": None, "diskFreeGb": disk},
    }


class Broadcaster:
    """Fans Services events out to asyncio queues, one per open /events response.

    `publish` runs on whichever thread emits, so it only hands the text to each client's loop.
    """

    def __init__(self, services):
        self.services = services
        self._clients = {}
        self._off = services.watch(self.publish)

    def publish(self, kind, payload):
        text = sse(kind, payload)
        for queue, loop in list(self._clients.values()):
            try:
                loop.call_soon_threadsafe(queue.put_nowait, (kind, text))
            except RuntimeError:
                pass  # loop closed: the client is gone

    def open(self):
        queue = asyncio.Queue()
        self._clients[id(queue)] = (queue, asyncio.get_running_loop())
        return queue

    def drop(self, queue):
        self._clients.pop(id(queue), None)

    def close(self):
        self._off()
        self._clients.clear()


router = APIRouter(tags=["events"])


@router.get("/events")
async def events(request: Request):
    broadcaster = request.app.state.broadcaster

    async def stream():
        queue = broadcaster.open()
        try:
            yield ": connected\n\n"
            yield sse("status", await asyncio.to_thread(status_snapshot, broadcaster.services))
            while not await request.is_disconnected():
                try:
                    kind, text = await asyncio.wait_for(queue.get(), KEEPALIVE)
                    yield text
                    if kind in {"queue", "versions", "blocked", "plan", "storage"}:
                        yield sse("status", await asyncio.to_thread(status_snapshot, broadcaster.services))
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
        finally:
            broadcaster.drop(queue)

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return StreamingResponse(stream(), media_type="text/event-stream", headers=headers)
