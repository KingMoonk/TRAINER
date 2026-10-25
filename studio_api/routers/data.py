"""Your data: the three kinds of rows (text, conversations, problems), imports, the recommended collections and drafts.

Files cross the wire, not paths: `import_data` takes a multipart upload, saved to a temporary folder the service then
imports from. A corpus already on the machine is picked by path from /data/browse, which only lists under the workspace
and the home folder. Reads that can scan a whole store (the summary, drafts from a local model, rebuilding downloads) are
jobs (studio_api.jobs): POST returns a job id, progress and the result arrive on /events, GET /jobs/{id} for late joiners.
"""

import base64
import json
import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel, Field

from studio_api.app import get_services
from studio_api.models_queue_data import JobRecordModel, Model, submit
import studio_rules

router = APIRouter(prefix="/data", tags=["data"])
Services = Annotated[Any, Depends(get_services)]
Kind = Literal["text", "conversations", "problems"]
DraftKind = Literal["text", "conversations"]

FIELDS = {"text": ("text",), "conversations": ("user", "assistant"), "problems": ("prompt", "rule")}  # what add_data takes
UPLOADS = Path(tempfile.gettempdir()) / "groundwork-uploads"
UPLOAD_KEPT_S = 24 * 3600  # a folder older than this belongs to an upload whose job never ran
LISTED = 5000  # entries one folder listing returns


class DataRow(Model):
    id: str
    a: str  # text: the set or title; conversations: the prompt; problems: the prompt
    b: str  # text: the first line; conversations: the answer; problems: the rule
    status: str
    dataset: str


class DataSet(Model):
    id: str
    name: str
    count: int


class SummaryModel(Model):
    """studio_data_kinds.DataSummary"""

    kind: str
    count: int = 0
    characters: int = 0
    used_by: str = ""
    count_line: str = ""
    sets: list[DataSet] = []
    rules: dict[str, int] = {}
    near_duplicates: int = 0
    held_back: int = 0
    scanned: int = 0
    rows: list[DataRow] = []


class SummaryRequest(BaseModel):
    query: str = ""


class SummaryJob(JobRecordModel):
    result: SummaryModel | None = None


class RowsPage(BaseModel):
    rows: list[DataRow]
    next_cursor: str | None = None  # opaque; pass it back as ?cursor=. null: nothing follows


class AddRequest(BaseModel):
    fields: dict[str, str]  # text: {text}; conversations: {user, assistant}; problems: {prompt, rule}


class Added(BaseModel):
    added: bool


class FieldsModel(BaseModel):
    fields: dict[str, str]


class Review(BaseModel):
    keep: bool  # true: mark Good; false: delete the example


class Count(BaseModel):
    count: int


class HasText(BaseModel):
    has_text: bool


class GenerateRequest(BaseModel):
    rule: str | None = None  # None: spread over every rule
    n: int = Field(ge=1, le=5000)
    store: bool = True  # false: preview only


class GenerateReport(BaseModel):
    """studio_ia.services.problem_report"""

    items: list[dict]
    added: int
    duplicates: int
    checked: int
    passed: int


class ProblemRule(BaseModel):
    id: str
    label: str


class ProblemRules(BaseModel):
    rules: list[ProblemRule]
    max_problems: int = 5000
    max_drafts: int = 200


class DraftRequest(BaseModel):
    n: int = Field(ge=1, le=200)


class Source(Model):
    """One recommended download."""

    id: str
    name: str
    size: str
    records: str
    state: str  # not downloaded | downloading | downloaded | imported
    completed: int = 0
    total: int = 0
    error: str = ""


class DownloadRequest(BaseModel):
    ids: list[str] | None = None  # None: the kind's whole recommended collection
    install: bool = False  # also add each to your sets


class Started(BaseModel):
    started: bool = True  # progress arrives as events: download, download_done / import_done, download_failed, downloads_finished


class PathsRequest(BaseModel):
    paths: list[str]  # files on the server, from /data/browse


class Entry(BaseModel):
    name: str
    path: str
    is_dir: bool
    size: int | None = None


class Listing(BaseModel):
    path: str | None = None  # null: the list of places to start from
    parent: str | None = None  # null at a place to start from
    entries: list[Entry]


class ImportJob(JobRecordModel):
    result: list[str] | None = None  # one report line per file


class DraftsJob(JobRecordModel):
    result: Count | None = None


class RebuiltJob(JobRecordModel):
    result: list[str] | None = None  # ids rebuilt


# server-side folder picker -----------------------------------------------------------------------------------------
def roots(services):
    """Where the picker may look: the workspace folder, then the home folder."""
    found = []
    root = getattr(getattr(services, "workspace", None), "root", None)
    for place in ([Path(root)] if root else []) + [Path.home()]:
        place = place.resolve()
        if place not in found:
            found.append(place)
    return found


def inside(path: Path, places):
    return any(path == place or place in path.parents for place in places)


def allowed(services, text) -> Path:
    """The resolved path, if it is under a root (symlinks are followed first, so a link cannot lead out)."""
    path = Path(text).expanduser()
    if not path.is_absolute():
        raise HTTPException(400, "Give an absolute path from /data/browse.")
    path = path.resolve()
    if not inside(path, roots(services)):
        raise HTTPException(403, "That folder is outside the workspace and the home folder.")
    return path


@router.get("/browse")
def browse(services: Services, path: str | None = None) -> Listing:
    """A folder on the server, to pick files from. No path: the places to start from (workspace, home)."""
    if path is None:
        return Listing(entries=[Entry(name=str(p), path=str(p), is_dir=True) for p in roots(services)])
    folder = allowed(services, path)
    if not folder.is_dir():
        raise HTTPException(404, "That is not a folder.")
    entries = []
    try:
        children = sorted(folder.iterdir(), key=lambda p: (not p.is_dir(), p.name.casefold()))
    except PermissionError:
        raise HTTPException(403, "This computer does not let the app look in that folder.") from None
    for child in children:
        if child.name.startswith("."):
            continue
        try:
            is_dir = child.is_dir()
            entries.append(Entry(name=child.name, path=str(child), is_dir=is_dir, size=None if is_dir else child.stat().st_size))
        except OSError:
            continue
        if len(entries) >= LISTED:
            break
    parent = folder.parent
    return Listing(path=str(folder), parent=str(parent) if inside(parent, roots(services)) else None, entries=entries)


# uploads ------------------------------------------------------------------------------------------------------------
def _sweep():
    for old in UPLOADS.glob("*") if UPLOADS.is_dir() else ():
        try:
            if time.time() - old.stat().st_mtime > UPLOAD_KEPT_S:
                shutil.rmtree(old, ignore_errors=True)
        except OSError:
            pass


def save_uploads(files):
    """Write the uploads into a new folder under the temp directory, each under its own file name. Returns (folder, paths)."""
    _sweep()
    UPLOADS.mkdir(parents=True, exist_ok=True)
    folder, paths = UPLOADS / uuid.uuid4().hex[:12], []
    folder.mkdir()
    try:
        for n, upload in enumerate(files):
            name = re.split(r"[\\/]", upload.filename or "")[-1].strip()
            name = name if name not in ("", ".", "..") else "upload"
            target = folder / (name if not (folder / name).exists() else f"{n}-{name}")
            with target.open("wb") as out:
                shutil.copyfileobj(upload.file, out, 1 << 20)
            paths.append(str(target))
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    return folder, paths


def _import(request, services, kind, paths, cleanup=None):
    def work(stop):
        try:
            return services.import_data(kind, paths)
        finally:
            if cleanup is not None:
                shutil.rmtree(cleanup, ignore_errors=True)
    return submit(request, "import_data", work, ImportJob)


def _cursor(value):
    return None if value is None else base64.urlsafe_b64encode(json.dumps(value).encode()).decode()


def _uncursor(text):
    if not text:
        return None
    try:
        value = json.loads(base64.urlsafe_b64decode(text.encode()))
    except ValueError:
        raise ValueError("That cursor is not one this server gave.") from None
    return tuple(value) if isinstance(value, list) else value


def _example_fields(services, example_id):
    try:
        return services.data_fields(example_id)
    except (KeyError, LookupError, StopIteration, TypeError):  # no such example
        raise HTTPException(404, f"No example {example_id}") from None


# small reads and writes ---------------------------------------------------------------------------------------------
@router.get("/has_text")
def has_text(services: Services) -> HasText:
    """Services.has_text"""
    return HasText(has_text=bool(services.has_text()))


@router.get("/drafts")
def draft_count(services: Services) -> Count:
    """Services.draft_count: drafts still waiting for review."""
    return Count(count=services.draft_count())


@router.get("/problems/rules")
def problem_rules() -> ProblemRules:
    """Generatable rules in desktop picker order; imported answer keys cannot generate examples."""
    return ProblemRules(rules=[ProblemRule(id=rule, label=studio_rules.LABELS[rule]) for rule in studio_rules.RULES])


@router.post("/refresh_downloads", status_code=202)
def refresh_downloads(request: Request, services: Services) -> RebuiltJob:
    """Services.refresh_downloads as a job: rebuild downloads converted by an older app version. Result: the ids rebuilt."""
    return submit(request, "refresh_downloads", lambda stop: list(services.refresh_downloads()), RebuiltJob)


@router.get("/collections/{kind}")
def collection(kind: Kind, services: Services) -> list[Source]:
    """Services.collection: the recommended sources of one kind and where each stands."""
    return [Source.model_validate(row) for row in services.collection(kind)]


@router.post("/collections/{kind}/download")
def download_sources(kind: Kind, body: DownloadRequest, services: Services) -> Started:
    """Services.download_sources: starts on a background thread; 409 when a download is already running."""
    if not services.download_sources(body.ids, kind, body.install):
        raise HTTPException(409, "A download is already running.")
    return Started()


@router.post("/sources/{source_id}/import")
def import_source(source_id: str, services: Services) -> Started:
    """Services.import_source: add one downloaded source to your sets (background; events import_done / download_failed).
    409 when it is not downloaded or other work is running. To bring your own file, use POST /data/{kind}/import."""
    if not services.import_source(source_id):
        raise HTTPException(409, "That source is not downloaded, or another download or import is running.")
    return Started()


@router.get("/examples/{example_id}/fields")
def data_fields(example_id: str, services: Services) -> FieldsModel:
    """Services.data_fields: the fields of one stored example, to pre-fill an edit."""
    return FieldsModel(fields=_example_fields(services, example_id))


@router.put("/examples/{example_id}")
def edit_data(example_id: str, body: AddRequest, services: Services) -> FieldsModel:
    """Services.edit_data. A 400 says in plain words what is wrong (empty, a duplicate, a problem that no longer passes)."""
    wanted = _example_fields(services, example_id)
    missing = [name for name in wanted if name not in body.fields]
    if missing:
        raise HTTPException(400, f"Missing: {', '.join(missing)}.")
    services.edit_data(example_id, body.fields)
    return FieldsModel(fields=_example_fields(services, example_id))


@router.post("/examples/{example_id}/review", status_code=204)
def review_data(example_id: str, body: Review, services: Services) -> None:
    """Services.review_data: keep marks a draft Good; otherwise the example is deleted."""
    _example_fields(services, example_id)
    services.review_data(example_id, body.keep)
    services.emit("data", {})


@router.post("/problems/generate")
def generate_problems(body: GenerateRequest, services: Services) -> GenerateReport:
    """Services.generate_problems: rule-made problems (store=false previews)."""
    report = GenerateReport.model_validate(services.generate_problems(body.rule, body.n, body.store))
    if report.added:
        services.emit("data", {"kind": "problems"})
    return report


# per-kind routes ----------------------------------------------------------------------------------------------------
@router.post("/{kind}/summary", status_code=202)
def data(kind: Kind, body: SummaryRequest, request: Request, services: Services) -> SummaryJob:
    """Services.data as a job (a cold store is scanned): the sets, counts and the first sample rows; `query` filters the rows."""
    return submit(request, "data", lambda stop: SummaryModel.model_validate(services.data(kind, body.query)).model_dump(mode="json"), SummaryJob,
                  lane="reads", key=("data", kind, body.query))  # on the reads worker; the same request while one is unfinished shares it


@router.get("/{kind}/rows")
def data_page(kind: Kind, services: Services, query: str = "", cursor: str | None = None,
              limit: Annotated[int, Query(ge=1, le=1000)] = 200, review_only: bool = False) -> RowsPage:
    """Services.data_page: one page of the kind's rows whose text holds `query`, read from storage so every row can be reached."""
    rows, following = services.data_page(kind, query, _uncursor(cursor), limit, review_only=review_only)
    return RowsPage(rows=[DataRow.model_validate(r) for r in rows], next_cursor=_cursor(following))


@router.post("/{kind}/rows")
def add_data(kind: Kind, body: AddRequest, services: Services) -> Added:
    """Services.add_data: added=false when it is empty or already there."""
    missing = [name for name in FIELDS[kind] if name not in body.fields]
    if missing:
        raise HTTPException(400, f"Missing: {', '.join(missing)}.")
    return Added(added=bool(services.add_data(kind, body.fields)))


@router.post("/{kind}/import", status_code=202)
def import_data(kind: Kind, request: Request, services: Services, files: Annotated[list[UploadFile], File()]) -> ImportJob:
    """Services.import_data from a multipart upload (`files`, one or more): saved to a temporary folder and imported as a
    job. Result: one report line per file, named as uploaded."""
    folder, paths = save_uploads(files)
    return _import(request, services, kind, paths, cleanup=folder)


@router.post("/{kind}/import_paths", status_code=202)
def import_paths(kind: Kind, body: PathsRequest, request: Request, services: Services) -> ImportJob:
    """Services.import_data from files already on the server, picked in /data/browse. Only under the workspace and the home folder."""
    if not body.paths:
        raise HTTPException(400, "Pick at least one file.")
    paths = [allowed(services, p) for p in body.paths]
    for path in paths:
        if not path.is_file():
            raise HTTPException(400, f"{path.name} is not a file.")
    return _import(request, services, kind, [str(p) for p in paths])


@router.post("/{kind}/drafts", status_code=202)
def make_drafts(kind: DraftKind, body: DraftRequest, request: Request, services: Services) -> DraftsJob:
    """Services.make_drafts as a job (a local model writes them): the result is how many were stored for review. A missing
    local model fails the job with a plain sentence."""
    def write(stop):
        count = services.make_drafts(kind, body.n)
        if count:
            services.emit("data", {"kind": kind})
        return {"count": count}
    return submit(request, "make_drafts", write, DraftsJob)
