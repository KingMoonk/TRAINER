"""Inspect: the Look-inside reads of a trained version. Each runs the model, so each is a job: POST answers with its id and the
result arrives on GET /events (and GET /jobs/{id}). Arrays are lists, ready for JSON."""

from typing import Literal

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, model_validator

from studio_api.models_work import JobOut, plain, start_job

router = APIRouter(prefix="/inspect", tags=["inspect"])

Mode = Literal["mean", "zero"]  # how a switched-off head is replaced: by its average output, or by zeros
Head = tuple[int, int]  # (block, head)


class Inspect(BaseModel):
    version: int | None = None  # None = the best trained version


class OnText(Inspect):
    text: str = Field(min_length=1)


class ForwardRequest(Inspect):
    text: str | None = Field(default=None, min_length=1)
    ids: list[int] | None = Field(default=None, min_length=1)  # or the piece ids themselves, e.g. from /inspect/prepared; exactly one of text and ids
    hooks: bool = True  # False: the logits only
    names: list[str] | None = None  # keep only these hooks (e.g. "block0.attn.pattern"); a whole model's hooks are large

    @model_validator(mode="after")
    def one_input(self):
        if (self.text is None) == (self.ids is None):
            raise ValueError("Send either text or ids.")
        return self


class TraceRequest(OnText):
    pos: int = -1


class LossRequest(OnText):
    pos: int = -1
    trained: bool = True  # False: the same design with fresh weights


class LogitLensRequest(OnText):
    top_k: int = Field(default=5, ge=1, le=100)  # guesses kept per position and stage


class AblateRequest(OnText):
    mode: Mode = "zero"
    heads: list[Head] | None = None  # (block, head) pairs to ablate, one at a time; None = every head


class HeadStatsRequest(BaseModel):
    pattern: list = Field(min_length=1)  # attention [batch][head][query][key], e.g. a "blockN.attn.pattern" hook from /inspect/forward


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


@router.post("/forward", response_model=JobOut, status_code=202)
def forward(request: Request, body: ForwardRequest):
    """{ids, tokens, logits, hooks}: one forward pass over `text`, or over the piece `ids`."""
    services = request.app.state.services

    def read(stop):
        out = services.inspect_forward(body.version, body.text if body.ids is None else body.ids, hooks=body.hooks)
        if body.names is not None:
            out = {**out, "hooks": {name: value for name, value in out["hooks"].items() if name in body.names}}
        return out
    return start_job(request, "inspect.forward", read)


@router.post("/trace", response_model=JobOut, status_code=202)
def trace(request: Request, body: TraceRequest):
    """{tokens, pos, rows}: what each stage of the model does to the token at `pos`."""
    services = request.app.state.services
    return start_job(request, "inspect.trace", lambda stop: services.inspect_trace(body.version, body.text, body.pos))


@router.post("/logit-lens", response_model=JobOut, status_code=202)
def logit_lens(request: Request, body: LogitLensRequest):
    """[{hook, probabilities, top_ids, top_probabilities}]: what the model would answer if it stopped after each stage."""
    services = request.app.state.services
    return start_job(request, "inspect.logit_lens", lambda stop: services.inspect_logit_lens(body.version, body.text, body.top_k))


@router.post("/ablate", response_model=JobOut, status_code=202)
def ablate(request: Request, body: AblateRequest):
    """{baseline, mode, heads: [{block, head, loss, delta, entropy, mean_distance, first_token_mass, previous_token_mass}]}: the loss with each head switched off alone."""
    services = request.app.state.services
    heads = None if body.heads is None else [tuple(h) for h in body.heads]
    return start_job(request, "inspect.ablate", lambda stop: services.inspect_ablate(body.version, body.text, mode=body.mode, heads=heads))


@router.post("/head-stats")
def head_stats(request: Request, body: HeadStatsRequest):
    """{entropy, mean_distance, first_token_mass, previous_token_mass}, each one number per head: a plain computation on the posted
    pattern, so it answers at once instead of as a job."""
    try:
        pattern = np.asarray(body.pattern, dtype=np.float64)
    except (TypeError, ValueError):
        raise HTTPException(422, "The pattern must be a rectangular array of numbers.") from None
    try:
        return plain(request.app.state.services.inspect_head_stats(pattern))
    except ValueError as error:
        raise HTTPException(422, str(error)) from None


@router.post("/prepared", response_model=JobOut, status_code=202)
def prepared(request: Request, body: Inspect):
    """{ids, tokens}: the text the version's pieces are read from."""
    services = request.app.state.services
    return start_job(request, "inspect.prepared", lambda stop: services.inspect_prepared(body.version))


@router.post("/loss", response_model=JobOut, status_code=202)
def loss(request: Request, body: LossRequest):
    """{tokens, pos, probabilities, target, pieces}: how likely each next piece was at `pos`."""
    services = request.app.state.services
    return start_job(request, "inspect.loss", lambda stop: services.inspect_loss(body.version, body.text, body.pos, body.trained))


@router.post("/value-map", response_model=JobOut, status_code=202)
def value_map(request: Request, body: ValueMapRequest):
    """{d_model, heads, blocks, singular}: the singular values of one head's value and output weights."""
    services = request.app.state.services
    return start_job(request, "inspect.value_map", lambda stop: services.inspect_value_map(body.version, body.heads, body.block, body.head))


@router.post("/rank-heads", response_model=JobOut, status_code=202)
def rank_heads(request: Request, body: RankRequest):
    """[{block, head, rise}]: how much worse the held-back text gets with each head switched off. Slow."""
    services = request.app.state.services
    return start_job(request, "inspect.rank_heads", lambda stop: services.inspect_rank_heads(body.version, body.mode))


@router.post("/heads-off", response_model=JobOut, status_code=202)
def heads_off(request: Request, body: HeadsOffRequest):
    """{before, after}: held-back bits per character with the (block, head) pairs in `off` switched off."""
    services = request.app.state.services
    return start_job(request, "inspect.heads_off", lambda stop: services.inspect_heads_off(body.version, [tuple(h) for h in body.off], body.mode))


@router.post("/reply", response_model=JobOut, status_code=202)
def reply(request: Request, body: ReplyRequest):
    """The greedy answer to `prompt` with the (block, head) pairs in `off` switched off for this reply only. The result is the text."""
    services = request.app.state.services
    return start_job(request, "inspect.reply", lambda stop: services.inspect_reply(body.version, body.prompt, [tuple(h) for h in body.off], body.mode))
