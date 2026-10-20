"""Passive training-data collection: choose topics and sources, set limits, watch the background collector, pause and resume."""

from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field

from studio_api.app import get_services


@asynccontextmanager
async def lifespan(app):
    """Keep the collector running and its shards registered while the API is up (a daemon thread; the collector process itself outlives the API)."""
    services = app.state.services
    services.collect_start()
    try:
        yield
    finally:
        services.collect_stop()


router = APIRouter(prefix="/collect", tags=["collect"], lifespan=lifespan)


class Limits(BaseModel):
    disk_mb: float
    bandwidth_mb_per_day: float
    requests_per_host_per_day: float
    seconds_between_requests: float
    max_dataset_mb: float
    max_file_mb: float
    auto_approve_mb: float


class Idle(BaseModel):
    only_when_idle: bool
    only_on_mains: bool
    unmetered_confirmed: bool
    cpu_percent: float


class Licences(BaseModel):
    non_commercial: bool
    unknown: bool
    copyleft: bool


class Sources(BaseModel):
    hub: bool
    stackexchange: bool
    github: bool
    web: bool


class StackExchange(BaseModel):
    sites: list[str]
    min_score: int
    dump_url: str


class GitHub(BaseModel):
    queries: list[str]
    max_repos: int


class Site(BaseModel):
    host: str
    seeds: list[str] = Field(default_factory=list)
    licence: str
    pages: int = 200


class Web(BaseModel):
    sites: list[Site]


class CollectConfig(BaseModel):
    enabled: bool
    paused: bool
    topics: list[str]
    custom_topics: list[str]
    sources: Sources
    licences: Licences
    limits: Limits
    idle: Idle
    approved: list[str]
    blocked: list[str]
    stackexchange: StackExchange
    github: GitHub
    web: Web


class LimitsUpdate(BaseModel):
    disk_mb: float | None = None
    bandwidth_mb_per_day: float | None = None
    requests_per_host_per_day: float | None = None
    seconds_between_requests: float | None = None
    max_dataset_mb: float | None = None
    max_file_mb: float | None = None
    auto_approve_mb: float | None = None


class IdleUpdate(BaseModel):
    only_when_idle: bool | None = None
    only_on_mains: bool | None = None
    unmetered_confirmed: bool | None = None
    cpu_percent: float | None = None


class LicencesUpdate(BaseModel):
    non_commercial: bool | None = None
    unknown: bool | None = None
    copyleft: bool | None = None


class SourcesUpdate(BaseModel):
    hub: bool | None = None
    stackexchange: bool | None = None
    github: bool | None = None
    web: bool | None = None


class StackExchangeUpdate(BaseModel):
    sites: list[str] | None = None
    min_score: int | None = None
    dump_url: str | None = None


class GitHubUpdate(BaseModel):
    queries: list[str] | None = None
    max_repos: int | None = None


class ConfigUpdate(BaseModel):
    """Any of the settings; what is left out is kept. A list or a website list replaces the whole list."""
    model_config = ConfigDict(extra="forbid")
    enabled: bool | None = None
    paused: bool | None = None
    topics: list[str] | None = None
    custom_topics: list[str] | None = None
    sources: SourcesUpdate | None = None
    licences: LicencesUpdate | None = None
    limits: LimitsUpdate | None = None
    idle: IdleUpdate | None = None
    approved: list[str] | None = None
    blocked: list[str] | None = None
    stackexchange: StackExchangeUpdate | None = None
    github: GitHubUpdate | None = None
    web: Web | None = None


class Topic(BaseModel):
    id: str
    label: str


class SourceOption(BaseModel):
    id: str
    label: str
    what: str
    terms: str = Field(description="the licence and terms note for this source")
    note: str = Field("", description="a caution, such as a tool this computer lacks; empty when none")


class Excluded(BaseModel):
    host: str
    reason: str


class LicenceOption(BaseModel):
    id: str
    label: str
    default: bool = Field(description="kept without asking")
    opt_in: str = Field("", description="the licences setting that allows it; empty for the default classes")


class CollectOptions(BaseModel):
    topics: list[Topic]
    sources: list[SourceOption]
    excluded: list[Excluded]
    licences: list[LicenceOption]
    user_agent: str


class Dropped(BaseModel):
    licence: int = 0
    quality: int = 0
    pii: int = 0
    duplicate: int = 0
    near_duplicate: int = 0
    decontam: int = 0
    invalid: int = 0


class Stats(BaseModel):
    seen: int = 0
    kept: int = 0
    dropped: Dropped = Field(default_factory=Dropped)
    redactions: int = 0
    bytes_kept: int = 0


class CollectedSet(BaseModel):
    id: str
    group: str
    name: str
    kept: int = Field(description="records the collector kept for this set")
    records: int = Field(description="records in the set now (what the trainer reads)")
    bytes: int
    seen: int
    redactions: int
    dropped: Dropped


class PlanRow(BaseModel):
    id: str
    source: str
    topic: str
    group: str
    licence: str
    licence_class: str
    bytes: int
    status: str = Field(description="planned | needs_approval | skipped")
    reason: str
    score: float
    url: str
    files: int
    approved: bool
    blocked: bool


class SourceProgress(BaseModel):
    key: str
    kind: str
    name: str
    topic: str
    status: str
    detail: str
    downloaded: int
    done: bool
    error: str


class JournalRow(BaseModel):
    at: str
    type: str
    text: str


class CollectState(BaseModel):
    enabled: bool
    paused: bool
    running: bool = Field(description="the background collector process is alive")
    status: str = Field(description="off | running | waiting | paused | idle | error")
    reason: str
    heartbeat_age: float | None = None
    stats: Stats
    sets: list[CollectedSet]
    plan: list[PlanRow]
    sources: list[SourceProgress]
    disk_used: int
    disk_cap: int
    bandwidth_today: int
    bandwidth_cap: int
    bandwidth_total: int
    requests_today: dict[str, int]
    waiting_shards: int = Field(description="finished shards not yet added to your data")
    journal: list[JournalRow]
    file: str


@router.get("", response_model=CollectState)
def collect_state(services=Depends(get_services)):
    """What the collector is doing now."""
    return services.collect_state()


@router.get("/options", response_model=CollectOptions)
def collect_options(services=Depends(get_services)):
    """Topics, sources with their licence and terms notes, the sites that are never collected."""
    return services.collect_options()


@router.get("/config", response_model=CollectConfig)
def get_config(services=Depends(get_services)):
    return services.collect_config()


@router.put("/config", response_model=CollectConfig)
def put_config(body: ConfigUpdate, services=Depends(get_services)):
    """Save settings. 400 with a sentence for a bad value (a site that is excluded, a cap out of range). Turning collection on starts it."""
    return services.save_collect_config(body.model_dump(exclude_none=True))


@router.post("/pause", response_model=CollectConfig)
def pause(services=Depends(get_services)):
    """Pause now; the collector notices within a second."""
    return services.collect_pause()


@router.post("/resume", response_model=CollectConfig)
def resume(services=Depends(get_services)):
    """Resume, turning collection on if it was off."""
    return services.collect_resume()
