"""Inspect: the Look-inside reads of a trained version. Each runs the model, so each is a job: POST answers with its id and the
result arrives on GET /events (and GET /jobs/{id}). Arrays are lists, ready for JSON."""

from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from studio_api.models_work import JobStarted, start_job

router = APIRouter(prefix="/inspect", tags=["inspect"])

Mode = Literal["mean", "zero"]  # how a switched-off head is replaced: by its average output, or by zeros
Head = tuple[int, int]  # (block, head)


class Inspect(BaseModel):
    version: int | None = None  # None = the best trained version


class OnText(Inspect):
    text: str = Field(min_length=1)


class ForwardRequest(OnText):
    hooks: bool = True  # False: the logits only
    names: list[str] | None = None  # keep only these hooks (e.g. "block0.attn.pattern"); a whole model's hooks are large


class TraceRequest(OnText):
    pos: int = -1


class LossRequest(OnText):
    pos: int = -1
    trained: bool = True  # False: the same design with fresh weights


class ValueMapRequest(Inspect):
    heads: int = Field(ge=1)
    block: int = Field(default=0, ge=0)
    head: int = Field(default=0, ge=0)


class RankRequest(Inspect):
    mode: Mode = "mean"


class HeadsOffRequest(RankRequest):
    off: list[Head]


class ReplyRequest(HeadsOffRequest):
    prompt: str = Field(min_length=1)


@router.post("/forward", response_model=JobStarted)
def forward(request: Request, body: ForwardRequest):
    """{ids, tokens, logits, hooks}: one forward pass over `text`."""
    services = request.app.state.services

    def read(stop):
        out = services.inspect_forward(body.version, body.text, hooks=body.hooks)
        if body.names is not None:
            out = {**out, "hooks": {name: value for name, value in out["hooks"].items() if name in body.names}}
        return out
    return start_job(request, "inspect.forward", read)


@router.post("/trace", response_model=JobStarted)
def trace(request: Request, body: TraceRequest):
    """{tokens, pos, rows}: what each stage of the model does to the token at `pos`."""
    services = request.app.state.services
    return start_job(request, "inspect.trace", lambda stop: services.inspect_trace(body.version, body.text, body.pos))


@router.post("/prepared", response_model=JobStarted)
def prepared(request: Request, body: Inspect):
    """{ids, tokens}: the text the version's pieces are read from."""
    services = request.app.state.services
    return start_job(request, "inspect.prepared", lambda stop: services.inspect_prepared(body.version))


@router.post("/loss", response_model=JobStarted)
def loss(request: Request, body: LossRequest):
    """{tokens, pos, probabilities, target, pieces}: how likely each next piece was at `pos`."""
    services = request.app.state.services
    return start_job(request, "inspect.loss", lambda stop: services.inspect_loss(body.version, body.text, body.pos, body.trained))


@router.post("/value-map", response_model=JobStarted)
def value_map(request: Request, body: ValueMapRequest):
    """{d_model, heads, blocks, singular}: the singular values of one head's value and output weights."""
    services = request.app.state.services
    return start_job(request, "inspect.value_map", lambda stop: services.inspect_value_map(body.version, body.heads, body.block, body.head))


@router.post("/rank-heads", response_model=JobStarted)
def rank_heads(request: Request, body: RankRequest):
    """[{block, head, rise}]: how much worse the held-back text gets with each head switched off. Slow."""
    services = request.app.state.services
    return start_job(request, "inspect.rank_heads", lambda stop: services.inspect_rank_heads(body.version, body.mode))


@router.post("/heads-off", response_model=JobStarted)
def heads_off(request: Request, body: HeadsOffRequest):
    """{before, after}: held-back bits per character with the (block, head) pairs in `off` switched off."""
    services = request.app.state.services
    return start_job(request, "inspect.heads_off", lambda stop: services.inspect_heads_off(body.version, [tuple(h) for h in body.off], body.mode))


@router.post("/reply", response_model=JobStarted)
def reply(request: Request, body: ReplyRequest):
    """The greedy answer to `prompt` with the (block, head) pairs in `off` switched off for this reply only. The result is the text."""
    services = request.app.state.services
    return start_job(request, "inspect.reply", lambda stop: services.inspect_reply(body.version, body.prompt, [tuple(h) for h in body.off], body.mode))
