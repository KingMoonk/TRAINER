"""Pretraining: the size preview, the word lists and the learning-rate range test. The slow ones are jobs (studio_api.jobs).

The draft design (Services.stage_draft / draft_spec) is not served: the browser holds it and sends the spec it means.
"""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, Field

from studio_api.app import get_services
from studio_api.models_queue_data import JobRecordModel, Model, submit

router = APIRouter(prefix="/pretraining", tags=["pretraining"])
Services = Annotated[Any, Depends(get_services)]

LISTS = ("letters", "small", "gpt2", "words")  # the order of Services.word_lists()["tokenizers"]


class PreviewModel(Model):
    """Services.pretrain_preview. Every field is absent when there is no design yet."""

    pieces: int | None = None  # tokenised size of the chosen text (estimated for large text)
    estimated: bool = False
    params: int | None = None
    vocab: int | None = None
    context: int | None = None
    first_rows: list[list[int]] = []  # each cell holds an example number, -1 where an example ends
    tokens_per_second: float | None = None


class PreviewRequest(BaseModel):
    dataset_ids: list[str] = []  # empty: every set
    rows: int = Field(16, ge=1, le=256)


class Chip(BaseModel):
    label: str
    known: bool  # False: the whole-word list does not hold it, the model sees only "unknown"


class WordList(BaseModel):
    id: str  # letters | small | gpt2 | words
    vocab_size: int
    count: int  # pieces the example sentence is cut into
    chips: list[Chip]


class WordListsModel(BaseModel):
    """Services.word_lists, with each tokenizer cut down to what the page draws. A list that is not available is null."""

    sentence: str
    lists: list[WordList | None]


class RangeRequest(BaseModel):
    spec: dict  # the design to test (the browser's draft, or a saved version's spec)


class RangeResult(Model):
    """studio_compute.lr_range_test"""

    rates: list[float] = []
    losses: list[float] = []
    smoothed: list[float] = []
    suggestion: float | None = None
    steepest: int | None = None
    diverged: bool = False
    failed_at: float | None = None
    cancelled: bool = False
    lr_min: float | None = None
    lr_max: float | None = None


class PreviewJob(JobRecordModel):
    result: PreviewModel | None = None


class WordListsJob(JobRecordModel):
    result: WordListsModel | None = None


class RangeJob(JobRecordModel):
    result: RangeResult | None = None


def word_lists_plain(lists):
    """word_lists() with its tokenizer objects replaced by their name, size and the example sentence cut by each."""
    from studio_ia import pieces

    sentence, out = lists["sentence"], []
    for name, tok in zip(LISTS, lists["tokenizers"]):
        if tok is None:
            out.append(None)
            continue
        chips = [{"label": label, "known": known} for label, known in pieces.chips(tok, sentence)]
        out.append({"id": name, "vocab_size": tok.vocab_size, "count": len(chips), "chips": chips})
    return {"sentence": sentence, "lists": out}


@router.get("/preview")
def cached_preview(services: Services, dataset_ids: Annotated[list[str], Query()] = [],
                   rows: Annotated[int, Query(ge=1, le=256)] = 16) -> PreviewModel | None:
    """Services.cached_preview: the preview when it is already known, else null. Never computes; POST /pretraining/preview does."""
    found = services.cached_preview(list(dataset_ids), rows)
    return None if found is None else PreviewModel.model_validate(found)


@router.post("/preview", status_code=202)
def pretrain_preview(body: PreviewRequest, request: Request, services: Services) -> PreviewJob:
    """Services.pretrain_preview as a job."""
    return submit(request, "pretrain_preview", lambda stop: services.pretrain_preview(body.dataset_ids, body.rows), PreviewJob)


@router.post("/word_lists", status_code=202)
def word_lists(request: Request, services: Services) -> WordListsJob:
    """Services.word_lists as a job."""
    return submit(request, "word_lists", lambda stop: word_lists_plain(services.word_lists()), WordListsJob)


@router.post("/lr_range_test", status_code=202)
def lr_range_test(body: RangeRequest, request: Request, services: Services) -> RangeJob:
    """Services.lr_range_test as a job; DELETE /jobs/{id} ends it early.
    A design with no plain text to try it on fails the job with a plain sentence."""
    return submit(request, "lr_range_test", lambda stop: services.lr_range_test(body.spec, stop=stop), RangeJob)
