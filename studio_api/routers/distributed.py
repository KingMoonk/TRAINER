"""Distributed pretraining runs this computer owns, for the Cluster page's Training tab (docs/distributed.md): start one on chosen
machines, read every run with its members and rounds, stop a run or one member.

This is the owner's side. The members' wire protocol is `/collective` (studio_api.routers.collective), the node-to-node job calls are
`/cluster`; nothing here is called by another machine.
"""

import os
import time
import uuid
from dataclasses import asdict
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

import studio_cluster
import studio_distributed as runs
import studio_diloco
import studio_facts
import studio_membership
import studio_peer
from studio_api.app import get_services
from studio_api.routers.pretraining import _version
from studio_api.version import app_version
from studio_ia import pretraining_plan as plan
from studio_ia.services import Job

router = APIRouter(prefix="/distributed", tags=["distributed"])
Services = Annotated[Any, Depends(get_services)]

ROWS = 16  # rows per step and the share of text held back for testing: the Pretraining page's answers
HOLD = 0.05
DEFAULTS = studio_diloco.Settings()


class Defaults(BaseModel):
    rounds: int
    inner_steps: int
    outer_lr: float
    outer_momentum: float
    delta_dtype: str


class StartRun(BaseModel):
    """What the Training tab's form sends. The design is a version's; this computer takes part, and starts the run as its coordinator."""

    version: int | None = None  # None: the current version
    dataset_ids: list[str] = Field(min_length=1)
    machines: list[str] = []  # the other machines to invite: addresses from the Machines list
    all_machines: bool | None = None  # true: invite every listed machine that answers on this app version instead; the rest are left out with a reason
    replaces: str | None = None  # a waiting pretrain job this run takes the place of: its version, text and place in the queue win
    rounds: int = Field(DEFAULTS.rounds, ge=1, le=100_000)
    inner_steps: int = Field(DEFAULTS.inner_steps, ge=1, le=1_000_000)
    outer_lr: float = Field(DEFAULTS.outer_lr, gt=0)
    outer_momentum: float = Field(DEFAULTS.outer_momentum, ge=0, lt=1)
    min_members: int | None = Field(None, ge=1, le=64)  # the run waits in the queue until this many machines (this one included) are free at once
    max_members: int | None = Field(None, ge=1, le=64)  # the most machines it takes; machines that free up later join it up to this many
    coordinator: str | None = None  # a machine from the Machines list to coordinate the run (it must be one of the machines invited and answer); "" (the default): this computer
    reviewed_only: bool = False  # train only on the examples marked Good (the validation text is cut from those too); 400 when the sets hold none
    overlap: bool = False  # every member (this one and the invited) overlaps its communication with its compute: it trains on while the exchange runs
    teacher: str | None = Field(None, max_length=200)  # an Ollama model that polishes the weights and judges the win-rate; needed by polish_every and target_win_rate
    polish_every: int | None = Field(None, ge=0, le=100_000)  # rounds between teacher polishes (below rounds); None or 0: no polish
    target_win_rate: float | None = Field(None, gt=0, le=1)  # stop once the best weights win this share of the teacher's head-to-head against the starting weights


class Sample(BaseModel):
    step: int
    t: float  # seconds since the round opened
    loss: float | None
    tokens_per_second: float | None


class BlockingJob(BaseModel):
    """The job on a machine that a member waits behind."""

    label: str
    eta_s: float | None  # seconds it has left; null when it cannot say


class Member(BaseModel):
    node: str  # the name it joins under
    address: str  # "local" for this computer
    local: bool
    device: str  # cuda | mps | cpu, as the machine reported when the run started; "" when unknown
    device_name: str
    state: Literal["invited", "joining", "training", "behind", "dropped", "left", "stopped", "unreachable", "waiting", "never_joined", "finished"]
    note: str  # one sentence: why it is behind, dropped or unreachable
    reason: str = ""  # the same sentence, under the shared member vocabulary: why it is waiting or unreachable, and what to do about it
    blocking_job: BlockingJob | None = None  # state waiting: the job on that machine it joins after
    missed: int  # rounds in a row it has missed
    sent: bool  # its change for the open round has arrived
    rounds_done: int
    steps: int
    tokens_per_second: float | None  # in its last round
    last_delta_bytes: int | None
    last_delta_dtype: str
    can_stop: bool
    # what it last reported (about every 2 s while it runs); empty until the coordinator has heard from it
    phase: Literal["downloading", "training", "uploading", "waiting"] | None
    step_in_round: int  # inner steps done in the open round
    round_steps: int  # inner steps dealt to this member for the open round (H unless the coordinator deals by speed)
    loss: float | None  # its live inner loss in the open round
    live_tokens_per_second: float | None
    reported_ago: float | None  # seconds since the report, on the coordinator's clock
    shards: int  # training shards it holds
    samples: list[Sample]  # its reports in the open round, for the live charts


class Round(BaseModel):
    round: int
    step: int
    loss: float | None  # the members' training loss over the round, weighted by tokens
    val_loss: float | None  # the shared weights on the held-back text
    bytes_up: int  # the members' changes, as sent
    bytes_down: int  # the weights the members took at the start of the round
    seconds: float
    members: int
    contributed: int
    dropped: list[str]
    best: bool = False  # the lowest validation loss so far: best.npz holds this round's weights while it stays the best
    pivot: str = ""  # the pivot the stop rules applied after this round ("halve_lr", "next_dataset", "drop_lessons"), else ""
    polish: Literal["", "accepted", "rejected", "skipped"] = ""  # the teacher's polish after this round: kept ("accepted": val_loss is then the polished weights' loss), tried but not kept, or failed; "" when none ran
    val_loss_before_polish: float | None = None  # the loss the round measured before an accepted polish replaced it
    polish_val_loss: float | None = None  # the loss of polished weights that were not kept
    lessons: int = 0  # the teacher's better answers written after this round, to train on from the next


class Lane(BaseModel):
    """One member's round on the coordinator's clock: seconds since the round opened. A moment is null when it never reported it."""

    node: str
    theta_taken: float | None  # it started downloading θ (the coordinator served it)
    first_step: float | None  # its first training report
    delta_sent: float | None  # it began sending its change
    delta_received: float | None  # the change arrived
    counted: bool | None  # null while the round is open and the change has not come; false: counted for nothing this round
    tokens: int
    steps: int
    delta_bytes: int


class RoundTimeline(BaseModel):
    round: int
    open: bool
    deadline: float  # seconds after the round opened
    window: float  # the deadline's length
    closed: float | None
    lanes: list[Lane]


class RunEvent(BaseModel):
    seq: int
    t: float  # seconds since the coordinator started, on its own clock (compare with Run.now_s)
    kind: Literal["joined", "left", "late", "dropped", "stale", "outer_step", "checkpoint", "finished", "pivot", "paused", "resumed", "took_over",
                  "stepped_down", "polish", "polish_off", "lessons", "win_rate", "win_rate_off", "judging", "lesson_dropped", "lesson_kept"]
    node: str  # "" for the run
    round: int
    text: str


class LessonItem(BaseModel):
    prompt: str  # what the teacher asked the model
    reply: str  # what the model said
    answer: str  # the teacher's better answer, which the run trains on


class Lesson(BaseModel):
    id: int  # the lesson's shard id
    round: int  # the round the polish that wrote it followed
    rows: int  # how many answers it holds
    status: Literal["live", "dropped", "left_out"]  # dealt to the members, dropped by you, or left out by a step back or a new coordinator
    keepable: bool  # Keep can deal it again: it is not dealt now and this computer still holds its answers
    items: list[LessonItem]  # the first few of its answers
    more: int  # ... and how many are not shown


class MemberHealth(BaseModel):
    """One member's share of one round, measured (studio_run_health.record). compute + wait + sync is the round's seconds."""

    loss: float | None  # its inner loss
    steps: int
    tokens: int
    compute_s: float  # seconds it spent training
    tokens_per_s: float | None  # tokens / compute_s: its speed alone
    wait_s: float | None  # seconds its change was in before the slowest member's; null when the arrival was not recorded
    sync_s: float  # the rest: download, upload, outer step, validation


class RoundHealth(BaseModel):
    round: int
    seconds: float
    tokens: int
    members: dict[str, MemberHealth]
    slowest: str | None  # the member whose change came in last (null with one member)
    val_loss: float | None  # outer loss: the shared weights on the held-back text after the outer step
    loss: float | None
    lr: float | None  # the outer learning rate
    update_norm: float | None  # size of the outer step
    bytes_up: int
    bytes_down: int
    dtype: str
    compression: float | None  # full-precision size of the weights over the size of one change as sent
    sync_s: float  # seconds of the round the slowest member was not computing


class HealthVerdict(BaseModel):
    state: Literal["working", "not_helping", "diverging", "not_enough_data"]
    text: str  # one sentence with its reason


class RunHealth(BaseModel):
    """Is the cluster helping (studio_run_health.summarize)? The numbers are over the latest rounds that had two or more members."""

    verdict: HealthVerdict
    basis: str  # what the speed-up compares, in a sentence
    rounds: list[RoundHealth]
    window: int  # rounds the numbers below cover
    aggregate_tokens_per_s: float | None
    baseline_node: str  # the fastest member alone
    baseline_tokens_per_s: float | None  # its measured compute speed
    speedup: float | None  # aggregate over baseline; null until enough rounds
    comm_share: float | None  # share of wall time the slowest member was not computing
    wait_node: str  # the member that waits longest for the others
    wait_share: float | None  # its share of a round spent waiting
    straggler: str  # the member whose change most often comes in last


class Run(BaseModel):
    id: str
    label: str
    version: int | None
    state: Literal["queued", "starting", "waiting", "running", "finished", "stopped", "failed"]
    note: str
    started_at: float
    owner: str  # the computer that started the run (its queue record is here)
    handoff: str = ""  # the machine chosen to coordinate: this computer hands the run to it once it holds a finished round; "" for this computer
    saved_version: int | None  # the version whose weights this run saved when it finished; None while it runs, and for a stopped or failed run
    term: int = 1  # the coordinator's term
    coordinator: str = ""  # the origin whose account of the run this is
    origin: str = ""  # whose fact this is: the node that wrote it into the observation store
    age_s: float = 0.  # seconds since the origin wrote it (0 for a run this computer coordinates: it is read live)
    origin_down: bool = False  # the origin is not answering: this is its last account of the run
    round: int  # rounds finished
    rounds: int
    inner_steps: int
    outer_lr: float
    outer_momentum: float
    delta_dtype: str
    members: list[Member]
    history: list[Round]
    health: RunHealth | None = None  # null for an account from a computer on an older version
    reviewed_only: bool = False  # the run trains only on examples marked Good
    can_stop: bool
    paused: bool = False  # the owner paused it: no new round opens until it is resumed (a round in flight finishes)
    can_pause: bool = False
    can_resume: bool = False
    can_restart: bool = False  # it ended and saved weights to start a new run from (best.npz, else last.npz)
    best_round: int | None = None  # the round whose weights are the best so far
    best_val_loss: float | None = None
    win_rate: float | None = None  # the best weights' share of head-to-head wins against the weights the run started from (target_win_rate), as last judged
    stop_rule: str = ""  # which stop rule ended the run, "" when none did
    stop_reason: str = ""  # why the run ended, in a sentence; "" while it goes
    lessons: int = 0  # the teacher's answers the run trains on as extra shards now
    lessons_left_out: int = 0  # ... and those you dropped, a drop_lessons pivot, or a successor coordinator, left out
    phase: Literal["waiting", "paused", "next_round", "training", "collecting", "outer_step", "validating", "polishing", "judging", "finished", "stopped"]
    deadline_s: float | None  # seconds left to the open round's deadline
    grace_s: float  # the wait after the deadline for a member's download and upload
    window_s: float | None  # the open round's deadline length
    contributed: int  # changes in for the open round ...
    expected: int  # ... of the members it waits for
    theta_bytes: int  # the weights, as a member downloads them
    now_s: float  # the coordinator's clock now, to place events and timelines
    timeline: list[RoundTimeline]  # the last few rounds, then the open one
    events: list[RunEvent]  # oldest first; the page shows them newest first


@router.get("/defaults")
def defaults() -> Defaults:
    """The DiLoCo settings the form starts from."""
    return Defaults(rounds=DEFAULTS.rounds, inner_steps=DEFAULTS.inner_steps, outer_lr=DEFAULTS.outer_lr,
                    outer_momentum=DEFAULTS.outer_momentum, delta_dtype=DEFAULTS.delta_dtype)


class ChatSet(BaseModel):
    """One set that holds conversations, which a run's teacher may take its prompts from."""

    id: str
    name: str
    count: int


@router.get("/chat_sets")
def chat_sets(services: Services) -> list[ChatSet]:
    """The sets that hold conversations: the form's prompt sources for a run with a teacher (they are never trained on)."""
    return [ChatSet.model_validate(row) for row in services.conversation_sets()]


def _jobs(services):
    runner = getattr(services, "runner", None)
    if runner is not None:
        return list(runner.jobs)
    state = services.queue()
    return [*([state.job] if state.job is not None else []), *state.up_next]


def _owned(services, create=False):
    """(run, job) for every run this node knows: its remembered ones, and an owner job in the queue that has no record yet."""
    jobs = {job.id: job for job in _jobs(services) if job.kind == "pretrain_distributed"}
    if create:
        for job in jobs.values():
            runs.run_of(job)
    return [(run, jobs.get(run.id)) for run in runs.RUNS.values()]


def _read(run, job):
    coordinator = studio_diloco.COORDINATORS.get(run.id)
    if coordinator is not None and job is not None and job.status in ("failed", "stopped", "done"):
        coordinator.stop()  # its own member is gone: the run cannot go on
    return runs.view(run, coordinator, job), coordinator


def _find(services, run_id):
    found = next(((run, job) for run, job in _owned(services, create=True) if run.id == run_id), None)
    if found is None:
        raise HTTPException(404, f"No run {run_id}.")
    return found


def _cluster(services):
    return getattr(getattr(services, "runner", None), "cluster", None)


def _live_rows(services):
    """This computer's runs, read live (age 0)."""
    cluster = _cluster(services)
    if cluster is None:
        return [_read(run, job)[0] for run, job in _owned(services, create=True)]
    out = []
    for run, coordinator, job in cluster.local_runs():
        if coordinator is not None and job is not None and job.status in ("failed", "stopped", "done"):
            coordinator.stop()  # its own member is gone: the run cannot go on
        out.append(runs.view(run, coordinator, job, runs.peer_facts(run, cluster)))
    return out


def _rows(services):
    """Every run in the cluster: this computer's live, the others' from the local replica with origin and age. Without a cluster
    (no queue that can take part in one) only this computer's, labelled with the run's owner (this computer's hostname)."""
    cluster = _cluster(services)
    rows = _live_rows(services)
    if cluster is None:
        rows = [{**row, "coordinator": row["owner"], "origin": row["owner"], "age_s": 0., "origin_down": False} for row in rows]
        return sorted(rows, key=lambda row: row["started_at"], reverse=True)
    return studio_facts.cluster_runs(cluster, rows)


@router.get("/runs")
def list_runs(services: Services) -> list[Run]:
    """Every run in the cluster, newest first, each with its members and rounds and the `origin` that wrote it and its `age_s`. Runs
    this computer coordinates are read live; the others come from this computer's copy of the observation store, so a node that is
    down keeps its last runs here with a growing age and `origin_down`. A machine that is offline or refuses is a member row with
    a reason; the rest of the run still reads."""
    return [Run.model_validate(row) for row in _rows(services)]


def _found(services, run_id):
    row = next((row for row in _rows(services) if row["id"] == run_id), None)
    if row is None:
        raise HTTPException(404, f"No run {run_id}.")
    return row


@router.get("/runs/{run_id}")
def get_run(run_id: str, services: Services) -> Run:
    return Run.model_validate(_found(services, run_id))


def _ask(services, row, path, read=False):
    """What the node that owns the run (it is not this one) answers to a command, or to a read when `read`."""
    cluster = _cluster(services)
    address = cluster.address_of(row["origin"]) if cluster is not None else None
    if address is None:
        raise HTTPException(409, f"{row['origin']} coordinates this run and is not one of this computer's listed machines.")
    client = cluster.client_for(address, retries=0, timeout=10.)
    try:
        return client.get_json(path) if read else client.post_json(path, {})
    except studio_peer.PeerError as exc:
        if exc.status is not None and 400 <= exc.status < 500:  # the owner refused: say what it said (404 no lesson, 409 run over)
            raise HTTPException(exc.status, exc.detail or str(exc)) from None
        raise HTTPException(502, str(exc)) from None  # not reached, or the owner failed


def _forward(services, row, path):
    """Send a command to the node that owns the run (it is not this one) and return the run as that node answers. The replica is not
    written: the owner writes the outcome and it arrives by sync."""
    answer = _ask(services, row, path)
    return Run.model_validate({**answer, "coordinator": row["origin"], "origin": row["origin"], "age_s": 0., "origin_down": False})


def _remote(services, run_id):
    """The row of a run another computer coordinates, else None."""
    cluster = _cluster(services)
    if cluster is None or any(run.id == run_id for run, _, _ in cluster.local_runs()):
        return None
    row = next((row for row in studio_facts.cluster_runs(cluster, []) if row["id"] == run_id), None)
    return row if row is not None and row["origin"] != cluster.hostname else None


def _address_for_peers(request):
    """How the other machines reach this one: TRAINER_NODE_ADDRESS, else https://<its tailnet name>, where `tailscale serve`
    listens in both `make run` (the API at /) and `make dev` (the API under /api/, which studio_peer finds), else the address this
    request came to (which only a machine on the same computer can use). Never the API's own port: it listens on loopback."""
    named = (os.environ.get("TRAINER_NODE_ADDRESS") or "").strip()
    if named:
        return named
    tailnet = getattr(getattr(request.app.state, "hosts", None), "tailnet_name", "")
    if tailnet:
        return f"https://{tailnet}"
    return str(request.base_url).rstrip("/")


def _replaced(services, job_id):
    """The waiting pretrain job a run takes the place of, and its place in Up next. 404 when it is not waiting, 409 for another kind."""
    waiting = list(services.queue().up_next)
    index = next((i for i, job in enumerate(waiting) if job.id == job_id), None)
    if index is None:
        raise HTTPException(404, f"No waiting job {job_id}.")
    if waiting[index].kind != "pretrain":
        raise HTTPException(409, "Only a pretraining job can run on all machines.")
    return waiting[index], index


def _answering(peers, own):
    """(peers that join, machines left out with why) for "all machines": a machine joins when it answers and runs this app's version."""
    joining, left_out = [], []
    for peer in peers:
        name = peer.hostname or peer.address
        if not peer.online:
            left_out.append({"node": name, "address": peer.address, "reason": peer.error or f"{name} did not answer."})
        elif peer.version and own.version and peer.version != own.version:
            left_out.append({"node": name, "address": peer.address,
                             "reason": f"{name} runs app version {peer.version}; this computer runs {own.version}."})
        else:
            joining.append(peer)
    if not joining:
        raise HTTPException(409, " ".join(["No other machine can join, so there is nothing to train with.",
                                           *(row["reason"] for row in left_out)]))
    return joining, left_out


def _teaches(body):
    """Whether the start asks for a teacher polish or a win-rate target."""
    return bool(body.polish_every) or body.target_win_rate is not None


def _teacher_options(body, services, dataset_ids):
    """(Settings fields, job params) for the teacher options of a start: nothing when none is set. 400 with a sentence when a teacher is
    named without a use for it, or a polish or win-rate target is set without a teacher or without conversations in the chosen sets (the
    coordinator takes its prompts from them, so the run would end at its start). The job then carries the conversations with the text."""
    teacher = (body.teacher or "").strip()
    if not _teaches(body):
        if teacher:
            raise HTTPException(400, "A teacher does nothing on its own: set how often it polishes, a win-rate target, or both, or leave the teacher out.")
        return {}, {}
    if not teacher:
        raise HTTPException(400, "Teacher polish and the win-rate target need a teacher: choose one of the models Ollama has.")
    polish, versus = bool(body.polish_every), body.target_win_rate is not None
    need = studio_diloco.prompts_needed(polish, versus)
    if not services.has_conversations(dataset_ids, body.reviewed_only, need):
        have = "none of them is marked Good" if body.reviewed_only else "it holds none"
        if services.has_conversations(dataset_ids, body.reviewed_only):
            why = (f"The win-rate target compares on at least {studio_diloco.MIN_EVAL_PROMPTS} held-out prompts" +
                   (f", and polishing needs {studio_diloco.MIN_EVAL_PROMPTS} more of its own" if polish else "")) if versus else "Polishing needs a prompt"
            raise HTTPException(400, f"{why}, so the chosen data needs at least {need} different questions asked in its conversations (a question asked twice "
                                     "counts once). Add conversations, or turn the teacher off" + ("" if not (polish and versus) else ", or use only one of the two") + ".")
        raise HTTPException(400, f"The teacher answers conversations from the chosen data, and {have}. Choose data with conversations in it, or leave the teacher out.")
    fields = {"teacher": teacher, "polish_every": body.polish_every or 0, **({"target_win_rate": body.target_win_rate} if body.target_win_rate is not None else {})}
    return fields, {**fields, "conversations": True}


def _check_coordinator(address, chosen, peers, own):
    """The machine chosen to coordinate a run must be one the run invites that answers on this app version; anything else is a 400 (not
    chosen) or 409 (not usable) with the reason, so a run never starts on a coordinator it cannot hand the run to. "" is this computer."""
    if not address:
        return
    if address not in chosen:
        raise HTTPException(400, f"{address} cannot coordinate the run: it is not one of the machines chosen for it.")
    peer = next((p for p in peers if p.address == address), None)
    name = (peer.hostname or address) if peer is not None else address
    if peer is None or not peer.online:
        raise HTTPException(409, f"{name} cannot coordinate the run: " + ((peer.error if peer is not None else "") or f"{name} did not answer."))
    if peer.version and own.version and peer.version != own.version:
        raise HTTPException(409, f"{name} cannot coordinate the run: it runs app version {peer.version}; this computer runs {own.version}.")


@router.post("/runs", status_code=201)
def start_run(body: StartRun, request: Request, services: Services) -> Run:
    """Queue a distributed pretraining run of a design on the chosen text, with this computer as coordinator and member. The chosen
    machines are invited when the run starts, and the form's settings are checked here (400 with a sentence). With `all_machines`
    every listed machine that answers on this app version is invited and the rest are left out (`left_out` in the job's params,
    each with a reason); with `replaces` the run takes a waiting pretrain job's version, text and place in the queue, and the
    pretrain job leaves the queue. With `coordinator` (one of the invited machines, which must answer on this app version: else 400
    or 409 with the reason) this computer starts the run and hands the coordinator role to that machine once it holds a finished
    round (studio_failover, "handoff"); without it this computer coordinates throughout."""
    replaced, index = _replaced(services, body.replaces) if body.replaces else (None, None)
    number = replaced.params.get("version") if replaced is not None else body.version
    dataset_ids = list(replaced.params.get("datasets") or []) if replaced is not None else body.dataset_ids
    version = services.current_version() if number is None else _version(services, number)
    if version is None:
        raise HTTPException(409, "Save a design in the Network designer first.")
    known = {s["id"] for s in plan.sets_of(services.data("text"))}
    chats = {s["id"] for s in services.conversation_sets()} if _teaches(body) else set()  # only the teacher's prompts come from these
    unknown = [d for d in dataset_ids if d not in known and d not in chats]
    if unknown or not dataset_ids or not any(d in known for d in dataset_ids):
        raise HTTPException(400, f"No text set {', '.join(unknown)}." if unknown else "Choose some text first.")
    if body.reviewed_only and not services.has_reviewed_text(dataset_ids):
        raise HTTPException(400, "No example in the chosen text is marked Good (and free to train on), so there is nothing to train on. Review some first, or turn off reviewed examples only.")
    shape = {k: getattr(body, k) for k in ("min_members", "max_members") if getattr(body, k) is not None}
    teaching, teaching_params = _teacher_options(body, services, dataset_ids)
    settings = studio_diloco.Settings(rounds=body.rounds, inner_steps=body.inner_steps, outer_lr=body.outer_lr,
                                      outer_momentum=body.outer_momentum, **shape, **teaching)  # ValueError -> 400
    own = studio_cluster.own_info(busy=services.queue().job is not None, version=app_version())
    listed = studio_cluster.configured_nodes({"studioConfig": {"nodes": services.setting("nodes", [])}})
    for address in body.machines:
        if address not in listed:
            raise HTTPException(400, f"{address} is not on the Machines tab.")
    chosen = listed if body.all_machines else [a for a in listed if a in body.machines]
    peers = studio_cluster.survey({"studioConfig": {"nodes": chosen}}, own)[1:]
    _check_coordinator(body.coordinator, chosen, peers, own)
    if teaching and body.coordinator:  # after the coordinator's own checks, so an offline or unknown one is told so
        raise HTTPException(400, "The teacher runs only on the computer that starts the run, so another computer cannot coordinate a run with a teacher. "
                                 "Leave the coordinator as this computer, or leave the teacher out.")
    left_out = []
    if body.all_machines:
        peers, left_out = _answering(peers, own)
    machines = [{"node": own.hostname, "address": studio_cluster.LOCAL, "device": own.device, "device_name": own.device_name},
                *({"node": peer.hostname or peer.address, "address": peer.address, "device": peer.device if peer.online else "",
                   "device_name": peer.device_name, "error": "" if peer.online else peer.error} for peer in peers)]
    if settings.min_members > len(machines):
        raise HTTPException(400, f"The run needs {settings.min_members} machines at once, but only {len(machines)} "
                                 f"{'is' if len(machines) == 1 else 'are'} chosen, this computer included.")
    config = plan.pretrain_config(version.spec, body.rounds * body.inner_steps, ROWS, HOLD)
    label = runs.across_label(version.number, len(machines))
    job = Job(id=uuid.uuid4().hex[:8], created_at=time.time(), kind="pretrain_distributed", label=label, params={
        "config": config, "datasets": list(dict.fromkeys(dataset_ids)), "mode": "from_scratch", "version": version.number,
        "heldout_fraction": HOLD, **{k: getattr(body, k) for k in ("rounds", "inner_steps", "outer_lr", "outer_momentum")},
        "delta_dtype": settings.delta_dtype, "settings": asdict(settings), **shape, **teaching_params, **({"reviewed_only": True} if body.reviewed_only else {}), **({"member_options": {"overlap": True}} if body.overlap else {}), "machines": machines, "owner": own.hostname,
        **({"coordinator": body.coordinator} if body.coordinator else {}), "peer_url": _address_for_peers(request), "api_port": request.url.port or studio_cluster.DEFAULT_PORT,
        **({"left_out": left_out} if body.all_machines else {})})
    runs.open_run(job)
    if replaced is not None:
        services.remove(replaced.id)
    services.enqueue(job)
    if index is not None:
        services.move(job.id, index)
    services.start_queue()
    return Run.model_validate(_found(services, job.id))


@router.post("/runs/{run_id}/stop")
def stop_run(run_id: str, services: Services) -> Run:
    """End a run. Every member is told it is finished at its next call, and the member jobs leave the other machines' queues. The
    weights of the last finished round stay in the run's folder. A run another computer coordinates is stopped by asking that computer."""
    remote = _remote(services, run_id)
    if remote is not None:
        return _forward(services, remote, f"distributed/runs/{run_id}/stop")
    run, job = _find(services, run_id)
    runs.stop_run(run, studio_diloco.COORDINATORS.get(run.id))
    if job is not None and job.status in ("waiting", "held", "running"):
        services.remove(run.id)
    return Run.model_validate(_read(run, job)[0])


def _hold(services, run_id, verb):
    """Pause or resume a run: this computer's coordinator does it, else the computer that coordinates the run is asked."""
    remote = _remote(services, run_id)
    if remote is not None:
        return _forward(services, remote, f"distributed/runs/{run_id}/{verb}")
    run, job = _find(services, run_id)
    coordinator = studio_diloco.COORDINATORS.get(run.id)
    if coordinator is None:
        raise HTTPException(409, "The run is not going.")
    try:
        getattr(coordinator, verb)()
    except studio_diloco.Stale as exc:
        raise HTTPException(409, str(exc)) from None
    return Run.model_validate(_read(run, job)[0])


@router.post("/runs/{run_id}/pause")
def pause_run(run_id: str, services: Services) -> Run:
    """Hold a run: no new round opens until it is resumed. A round in flight finishes, the members wait, and the paused time does not
    count toward the run's time limit. It stays paused over a restart of the app. 409 when the run is over or not going. A run
    another computer coordinates is paused by asking that computer."""
    return _hold(services, run_id, "pause")


@router.post("/runs/{run_id}/resume")
def resume_run(run_id: str, services: Services) -> Run:
    """Undo a pause: the next round opens as soon as the members are in. 409 when the run is over or not going."""
    return _hold(services, run_id, "resume")


def _lessons(services, run_id, verb=None, lesson=None):
    """The run's lessons, after `verb` ("drop" or "keep") on one: this computer's coordinator answers, else the computer that
    coordinates the run is asked."""
    suffix = f"/{lesson}/{verb}" if verb else ""
    remote = _remote(services, run_id)
    if remote is not None:
        answer = _ask(services, remote, f"distributed/runs/{run_id}/lessons{suffix}", read=verb is None)
        return [Lesson.model_validate(row) for row in answer]
    _find(services, run_id)
    coordinator = studio_diloco.COORDINATORS.get(run_id)
    if coordinator is None:
        raise HTTPException(409, "The run is not going.")
    try:
        if verb:
            getattr(coordinator, f"{verb}_lesson")(lesson)
    except KeyError:
        raise HTTPException(404, f"Run {run_id} has no lesson {lesson}.") from None
    except studio_diloco.Stale as exc:
        raise HTTPException(409, str(exc)) from None
    return [Lesson.model_validate(row) for row in coordinator.lesson_list()]


@router.get("/runs/{run_id}/lessons")
def run_lessons(run_id: str, services: Services) -> list[Lesson]:
    """What the teacher gave the run, one lesson per polish: the questions, what the model said and the better answers. 409 when the
    run is not going. A run another computer coordinates is read by asking that computer."""
    return _lessons(services, run_id)


@router.post("/runs/{run_id}/lessons/{lesson}/drop")
def drop_run_lesson(run_id: str, lesson: int, services: Services) -> list[Lesson]:
    """Leave a lesson out of the rounds that open from now on; the round in flight keeps it. 404 for an id that is no lesson, 409 when
    the run is over or not going. Answers the lessons as they stand."""
    return _lessons(services, run_id, "drop", lesson)


@router.post("/runs/{run_id}/lessons/{lesson}/keep")
def keep_run_lesson(run_id: str, lesson: int, services: Services) -> list[Lesson]:
    """Put a dropped (or stepped-back) lesson back: the next rounds deal it again. Same errors as drop."""
    return _lessons(services, run_id, "keep", lesson)


@router.post("/runs/{run_id}/restart", status_code=201)
def restart_run(run_id: str, services: Services) -> Run:
    """Start a new run (a new id) from an ended one: the same design, text, settings and machines, with the weights of the old run's
    best round (else its last) as the starting point. The outer momentum starts at zero. 409 for a run that is still going, one
    that saved no weights, or one this computer did not start. A run another computer coordinates is restarted by that computer."""
    remote = _remote(services, run_id)
    if remote is not None:
        return _forward(services, remote, f"distributed/runs/{run_id}/restart")
    run, job = _find(services, run_id)
    row = _read(run, job)[0]
    if row["state"] in runs.ACTIVE:
        raise HTTPException(409, "The run is still going. Stop it first.")
    try:
        params = runs.restart_params(run, row["term"])
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from None
    fresh = Job(id=uuid.uuid4().hex[:8], created_at=time.time(), kind="pretrain_distributed", label=f"{run.label}, restarted", params=params)
    runs.open_run(fresh)
    services.enqueue(fresh)
    run.restarted_as = fresh.id
    runs.persist()
    services.start_queue()
    return Run.model_validate(_found(services, fresh.id))


@router.post("/runs/{run_id}/members/{node}/stop")
def stop_member(run_id: str, node: str, services: Services) -> Run:
    """Take one member out of a run and keep it out. The others carry on from the next round. 409 for this computer (stop the run)
    and for a node that is not a member. A run another computer coordinates: that computer is asked."""
    remote = _remote(services, run_id)
    if remote is not None:
        return _forward(services, remote, f"distributed/runs/{run_id}/members/{node}/stop")
    run, job = _find(services, run_id)
    coordinator = studio_diloco.COORDINATORS.get(run.id)
    machine = next((m for m in run.machines if m.node == node), None)
    if machine is not None and machine.local:
        raise HTTPException(409, "This computer coordinates the run. Stop the run instead.")
    if coordinator is None:
        raise HTTPException(409, "The run is not active.")
    try:
        coordinator.drop(node)
    except studio_diloco.Stale as exc:
        raise HTTPException(409, str(exc)) from None
    runs.release_member(run, node)
    return Run.model_validate(_read(run, job)[0])


# ---- this computer as a member of other computers' runs ---------------------------------------------------------------

class Invite(BaseModel):
    """A run another computer added this one to: its member job in this computer's queue, in the shared member vocabulary."""

    job_id: str  # the member job in this computer's queue
    run_id: str
    run_label: str
    owner: str  # the computer that added this one
    node: str  # the name this computer joins under
    state: Literal["invited", "joining", "training", "behind", "dropped", "left", "stopped", "unreachable", "waiting", "never_joined", "finished"]
    reason: str  # one sentence on the state
    blocking_job: BlockingJob | None = None  # state waiting: the local job it joins after
    round: int | None = None  # rounds the run has finished, once the owner's account has arrived
    rounds: int | None = None
    tokens_per_second: float | None = None
    can_pause_and_join: bool  # the local job running now can step aside for it


class PauseRequest(BaseModel):
    """Another computer asked this one to pause the job it is running. Only a person here acts on it."""

    id: str
    from_node: str
    run_label: str
    blocking_job: BlockingJob | None = None  # the job that would pause; null once nothing runs here
    at: float


class Invites(BaseModel):
    invites: list[Invite]
    pause_requests: list[PauseRequest]


class AskPause(BaseModel):
    from_node: str = Field(min_length=1)
    run_label: str = ""


class AskPeer(BaseModel):
    address: str = Field(min_length=1)  # the machine to ask: its address from the Machines list
    run_label: str = ""  # the run this computer is about to start, for the request's text


def _running_here(services):
    """(the job running now, seconds it has left) on this computer."""
    state = services.queue()
    return state.job, state.eta_s


def _own_name(services):
    cluster = _cluster(services)
    return cluster.hostname if cluster is not None else os.uname().nodename


def _invites(services):
    """This computer's invites and pause requests, read live."""
    state = services.queue()
    running, eta = state.job, state.eta_s
    cluster = _cluster(services)
    known = {row["id"]: row for row in studio_facts.cluster_runs(cluster, [])} if cluster is not None else {}
    jobs = [job for job in [state.job, *state.up_next] if job is not None]
    rows = [studio_membership.invite_row(job, running, eta, known.get(job.params.get("run"))) for job in studio_membership.member_jobs(jobs)]
    return Invites(invites=[Invite.model_validate(row) for row in rows],
                   pause_requests=[PauseRequest.model_validate(row) for row in studio_membership.pending_requests(running, eta)])


def _member_job(services, job_id):
    state = services.queue()
    job = next((j for j in [state.job, *state.up_next] if j is not None and j.id == job_id and j.kind == "diloco_member"), None)
    if job is None:
        raise HTTPException(404, f"No invite {job_id}.")
    return job


def _yield(services, to=None):
    runner = getattr(services, "runner", None)
    if runner is None or not hasattr(runner, "yield_running"):
        raise HTTPException(409, "This computer's queue cannot step a job aside.")
    if not runner.yield_running(to):
        raise HTTPException(409, "Nothing is running here that could step aside.")


@router.get("/invites")
def list_invites(services: Services) -> Invites:
    """The runs other computers added this one to (no accept step: they join by themselves when this computer is free) and the pause
    requests other computers sent it. The same words the owner's run page uses for this machine."""
    return _invites(services)


@router.post("/invites/{job_id}/pause-and-join")
def pause_and_join(job_id: str, services: Services) -> Invites:
    """Pause the job running here and put the member job in its place. The paused job goes back to waiting and continues from its own
    checkpoint when the run ends. 409 when the member job already runs or nothing else runs."""
    job = _member_job(services, job_id)
    if job.status == "running":
        raise HTTPException(409, "It is already running.")
    _yield(services, job_id)
    return _invites(services)


@router.post("/invites/{job_id}/leave")
def leave_run(job_id: str, services: Services) -> Invites:
    """Leave the run: the owner is told to drop this member and keep it out (the same call as Stop on the owner's run page), and the
    member job leaves this queue, a running one stopping. The owner not answering does not stop this computer from leaving."""
    job = _member_job(services, job_id)
    studio_membership.tell_owner_left(job)
    state = services.queue()
    if any(j is not None and j.id == job_id for j in [state.job, *state.up_next]):
        services.remove(job_id)
    return _invites(services)


@router.post("/pause-requests", status_code=201)
def receive_pause_request(body: AskPause, services: Services) -> PauseRequest:
    """Another computer asks this one to pause its running job. It is shown here as an alert, and nothing happens until a person
    here accepts it. Asking again from the same computer for the same run refreshes the request."""
    request = studio_membership.ask_here(body.from_node, body.run_label)
    state = services.queue()
    return PauseRequest.model_validate(studio_membership.requests_row(request, state.job, state.eta_s))


@router.post("/pause-requests/{request_id}/accept")
def accept_pause_request(request_id: str, services: Services) -> Invites:
    """Pause the running job and hold it, so a waiting member job runs. The held job is released from the Queue page. 404 for an
    unknown request, 409 when nothing runs here any more."""
    state = services.queue()
    if request_id not in {r["id"] for r in studio_membership.pending_requests(state.job, state.eta_s)}:
        raise HTTPException(404, f"No pause request {request_id}.")
    _yield(services)
    studio_membership.dismiss(request_id)
    return _invites(services)


@router.delete("/pause-requests/{request_id}")
def dismiss_pause_request(request_id: str, services: Services) -> Invites:
    """Decline a pause request; the running job carries on."""
    if not studio_membership.dismiss(request_id):
        raise HTTPException(404, f"No pause request {request_id}.")
    return _invites(services)


@router.post("/peers/pause-request", status_code=201)
def ask_peer_to_pause(body: AskPeer, services: Services) -> PauseRequest:
    """Ask another machine to pause the job it is running, so a run can use it. The request shows there as an alert; nothing is
    paused until a person there accepts. 502 with a sentence when the machine does not answer."""
    cluster = _cluster(services)
    client_for = cluster.client_for if cluster is not None else None
    try:
        answer = studio_membership.ask_peer(body.address, _own_name(services), body.run_label, client_for)
    except studio_peer.PeerError as exc:
        raise HTTPException(502, str(exc)) from None
    return PauseRequest.model_validate(answer)
