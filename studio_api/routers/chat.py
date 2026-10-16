"""Chat: stream one reply from a trained version, and save what the user teaches or rates.

A reply streams on its own response (POST /chat/reply answers with text/event-stream), not through GET /events. The tokens
belong to one request, so only the tab that asked sees them; /events fans every event out to every client. A client that
closes the response also stops the reply: the route sets the `stop` event the model polls.
"""

import asyncio
import threading
from typing import Any, Literal

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from studio_api.app import get_services
from studio_api.events import KEEPALIVE, sse
from studio_api.models_work import plain
from studio_ia.services import plain_error

router = APIRouter(prefix="/chat", tags=["chat"])


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ReplyRequest(BaseModel):
    version: int | None = None  # None = the best trained version
    messages: list[Message] = Field(min_length=1)
    ablate: list[tuple[int, int]] | None = None  # (block, head) pairs mean-ablated for this reply only


class CorrectionRequest(BaseModel):
    version: int | None = None
    messages: list[Message] = Field(min_length=1)
    correction: str = Field(min_length=1)
    whole_answer: bool = False  # True: the correction is a whole new answer, kept as a conversation for round 1


class Reply(BaseModel):
    version: int | None = None
    reply: str


class PairRequest(BaseModel):
    messages: list[Message] = Field(min_length=1)
    chosen: Reply
    rejected: Reply


class PairSaved(BaseModel):
    record: dict[str, Any] | None = None  # the stored pair, as the preference store wrote it


def run_reply(services, body, put, stop):
    """Run one reply on the calling thread. put(kind, payload) gets "text" (the reply so far, cumulative), then "done" with the
    whole reply; a failure becomes an "error" with a sentence. The same contract as the Chat page's stream_reply."""
    seen = []

    def on_text(text):
        seen.append(text)
        put("text", {"text": text})
    try:
        final = services.chat(body.version, [m.model_dump() for m in body.messages], on_text, stop, ablate=body.ablate or None)
        if final and not seen:
            put("text", {"text": final})
        put("done", {"text": final or (seen[-1] if seen else "")})
    except Exception as error:  # noqa: BLE001 - a failed reply is a sentence in the stream, not a dropped connection
        put("error", {"message": plain_error(error)})


@router.post("/reply")
async def reply(body: ReplyRequest, request: Request):
    """Stream the reply as server-sent events: `text` ({text}: the reply so far), then `done` ({text}) or `error` ({message})."""
    services = request.app.state.services
    loop, events, stop = asyncio.get_running_loop(), asyncio.Queue(), threading.Event()

    def put(kind, payload):
        try:
            loop.call_soon_threadsafe(events.put_nowait, (kind, payload))
        except RuntimeError:
            stop.set()  # the loop is closed: nobody is listening, so the model should stop

    worker = threading.Thread(target=run_reply, args=(services, body, put, stop), name="api-chat", daemon=True)

    async def stream():
        worker.start()
        try:
            yield ": connected\n\n"
            while True:
                try:
                    kind, payload = await asyncio.wait_for(events.get(), KEEPALIVE)
                except asyncio.TimeoutError:
                    yield ": ping\n\n"  # a model still loading says nothing for a while
                    continue
                yield sse(kind, payload)  # the same framing as GET /events
                if kind in ("done", "error"):
                    return
        finally:
            stop.set()  # the client left or the reply is over: either way the model should stop

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return StreamingResponse(stream(), media_type="text/event-stream", headers=headers)


@router.post("/corrections", status_code=204)
def teach_correction(body: CorrectionRequest, services=Depends(get_services)):
    """Save a correction: for round 3 (the teacher sees it, the student does not), or with `whole_answer` as a round 1 conversation."""
    services.teach_correction(body.version, [m.model_dump() for m in body.messages], body.correction, body.whole_answer)
    return Response(status_code=204)


@router.post("/pairs", response_model=PairSaved)
def rate_pair(body: PairRequest, services=Depends(get_services)):
    """Save a rated pair for round 4: the answer the user preferred and the one they did not."""
    saved = services.rate_pair([m.model_dump() for m in body.messages], body.chosen.model_dump(), body.rejected.model_dump())
    return PairSaved(record=plain(saved) if saved else None)
