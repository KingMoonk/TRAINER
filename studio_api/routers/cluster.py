"""The cluster: the machines in it, the jobs placed on or taken from other nodes, and the node-to-node calls of the scheduler
(docs/distributed.md, "Scheduler"). The rules are studio_scheduler's, the I/O studio_remote's; these routes only carry them.

The machines: `GET /cluster/nodes` surveys this node and the peers listed in the workspace config (`studioConfig.nodes`), and
`PUT /cluster/nodes` saves that list (docs/cluster.md). Neither needs a job queue.

Calls between nodes: `POST /cluster/jobs` (owner to runner: here is a job and its lease), `POST /cluster/jobs/{id}/stop`, and, to the
owner, `POST /cluster/jobs/{id}/renew` and `.../finished`. Every write to the owner carries the lease's epoch; any epoch but the
current one is answered 409, and the runner then stops its copy.
"""

import dataclasses
import time
from dataclasses import asdict
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

import studio_cluster
import studio_cluster_queue
import studio_facts
import studio_peer
import studio_scheduler as sched
from studio_api.app import get_services
from studio_api.routers.queue import PlanModel, local_jobs
from studio_api.version import app_version

router = APIRouter(prefix="/cluster", tags=["cluster"])


def get_cluster(services: Annotated[Any, Depends(get_services)]):
    cluster = getattr(getattr(services, "runner", None), "cluster", None)
    if cluster is None:
        raise HTTPException(409, "This node has no job queue that can take part in a cluster.")
    return cluster


Cluster = Annotated[Any, Depends(get_cluster)]


class ClusterNode(BaseModel):
    address: str  # the entry in the list, or "local" for this machine
    local: bool
    hostname: str  # "" for a machine that did not answer
    version: str
    device: str  # cuda | mps | cpu
    device_name: str
    memory_bytes: int  # video memory on CUDA, the computer's memory otherwise; 0 when unknown
    busy: bool
    dtypes: list[str]
    online: bool
    error: str = ""  # why a machine is offline, in plain words
    tailnet_name: str = ""  # this machine's MagicDNS name, to give to another machine's list; "" for a peer, or when Tailscale is unknown
    origin: str = ""  # the node whose facts these are: its name in the observation store; "" when this node holds none of its facts
    age_s: float | None = None  # seconds since the node last wrote its heartbeat into this computer's copy of the store; 0 for this machine
    origin_down: bool = False  # the store says the node is not answering
    last_heard_s: float | None = None  # seconds since this computer last heard from the node (an answer, or its heartbeat); None when never, or for this machine
    next_check_s: float | None = None  # seconds until this computer's poll asks the node again; None for this machine and before the first poll


class NodeList(BaseModel):
    addresses: list[str]  # the other machines: names ("windows-box", "box:9000") or URLs; this machine is always a node


class ClusterJob(BaseModel):
    id: str
    kind: str
    label: str = ""
    role: Literal["owner", "runner"]  # owner: queued here, running (or waiting) away; runner: queued elsewhere, running here
    node: str = ""  # owner: where it runs; runner: who owns it
    epoch: int = 0
    status: str
    lease: str = ""  # held | lapsed | expired
    waiting_reason: str = ""
    checkpoint_step: int | None = None


class QueueLimits(BaseModel):
    """What a machine's queue plan allows its jobs, so the run-start form can say it before a machine is chosen."""

    plan: PlanModel
    blocked: str | None = None  # why its plan holds its jobs right now ("on battery", "outside hours", "computer in use")
    paused: bool = False  # its person paused its queue


class QueueNode(BaseModel):
    address: str  # the entry in the list, or "local" for this machine
    hostname: str  # "" for a machine that never answered
    pin: str  # what to send as a job's `pinned` to keep it on this machine: its address for a listed machine (two can share a hostname), else its hostname (its address, "local", when a listed machine shares that hostname)
    local: bool
    online: bool
    error: str = ""
    seen_at: float = 0.0  # when its queue was last read (epoch seconds); 0 for this machine and for one that never answered
    origin: str = ""  # the node's name in the observation store ("" when this node holds none of its facts)
    age_s: float | None = None  # seconds since the node last wrote its queue into this computer's copy of the store; 0 for this machine
    origin_down: bool = False
    listed: bool = True  # False for a node this computer knows only from the store (another node's list names it)
    limits: QueueLimits | None = None  # its queue plan and what holds it now; None when it could not be read


class QueueRow(BaseModel):
    id: str
    kind: str
    label: str = ""
    status: Literal["running", "waiting", "held"]
    node: str  # address of the machine it runs on; for a waiting job the machine it is pinned to; "" when it may go anywhere
    hostname: str  # that machine's name; the pin as typed when no machine answers to it; "" when it may go anywhere
    queued_on: str  # the name of the machine whose queue holds it
    queued_here: bool  # False for a job another machine queued: it cannot be changed from here
    pinned: str = ""
    waiting_reason: str = ""
    stale: bool = False  # the machine is not answering: this is its last known queue
    note: str = ""  # the job's note, such as a checkpoint a peer could not store because its bucket is full
    queued_address: str = ""  # the address of the machine whose queue holds it ("local" for this one): POST /cluster/remote/{id}/stop needs it
    step: int = 0  # a running job's progress as its machine reports it; 0 when unknown
    total: int = 0
    eta_s: float | None = None


class Access(BaseModel):
    """Who may call this node's API straight over the tailnet (studioConfig.cluster, docs/cluster.md, "ACL")."""

    allow_users: list[str]  # Tailscale logins of people whose own (untagged) machines are let in
    allow_tags: list[str]  # tags of machines that are let in; tag:trainer unless changed
    login: str = ""  # the Tailscale login of this node's owner: what another machine adds to its allow_users; "" when tagged or unknown


class AccessBody(BaseModel):
    allow_users: list[str]
    allow_tags: list[str]


class RemoteStop(BaseModel):
    address: str  # the peer whose queue holds the job: a row's queued_address


class ClusterQueue(BaseModel):
    nodes: list[QueueNode]  # this machine first, then the listed peers; empty for a cluster of one (nothing to show about machines)
    jobs: list[QueueRow]  # every machine's running and waiting jobs; empty for a cluster of one, unless a job still runs on a removed peer


class LeaseModel(BaseModel):
    job_id: str
    node: str
    epoch: int = Field(ge=1)
    duration_s: float = Field(gt=0)


class JobSpec(BaseModel):
    id: str
    kind: str
    label: str = ""
    note: str = ""
    params: dict = {}
    created_at: float = 0.0
    eta_s: float | None = None


class Dispatch(BaseModel):
    job: JobSpec
    lease: LeaseModel
    owner_address: str | None = None  # how the runner reaches the owner; the address the request came from when absent
    fetch: dict[str, str] = {}  # job param name -> version store key to pull into it before the job starts


class Accepted(BaseModel):
    job_id: str
    epoch: int


class Checkpoint(BaseModel):
    key: str
    step: int = 0


class Renewal(BaseModel):
    epoch: int
    state: str = "running"  # fetching | queued | running
    step: int = 0
    total: int = 0
    events: list[dict] = []
    checkpoint: Checkpoint | None = None
    waiting: bool = False
    problem: str = ""  # why the runner could not share its latest checkpoint (the version store refused it); the job keeps training


class Renewed(BaseModel):
    epoch: int
    duration_s: float


class Finished(BaseModel):
    epoch: int
    status: Literal["done", "failed", "stopped"]
    error: str = ""
    key: str | None = None  # the finished version, published to the store
    step: int = 0


class Stop(BaseModel):
    epoch: int


@router.get("/nodes")
def nodes(request: Request, services=Depends(get_services)) -> list[ClusterNode]:
    """This node first, then every listed peer, in list order. A peer that does not answer within two seconds is listed offline with
    the reason, never dropped. Sync, so the GPU probe and the peers (asked at the same time) run off the event loop. This node's
    tailnet name comes from the host-name refresh (studio_api.hosts), never from a call to tailscale while the request waits."""
    own = studio_cluster.own_info(busy=services.queue().job is not None, version=app_version())
    state = {"studioConfig": {"nodes": services.setting("nodes", [])}}
    tailnet = getattr(getattr(request.app.state, "hosts", None), "tailnet_name", "")
    cluster = getattr(getattr(services, "runner", None), "cluster", None)
    rows = []
    for info in studio_cluster.survey(state, own, timeout=studio_cluster.POLL_TIMEOUT):
        local = info.address == studio_cluster.LOCAL
        name = cluster.hostname if cluster is not None and local else (info.hostname or (cluster._names.get(info.address, "") if cluster is not None else ""))
        facts = studio_facts.node_facts(cluster, name) if cluster is not None else None
        heard, check = (None, None) if local or cluster is None else cluster.contact(info.address)
        rows.append(ClusterNode(**asdict(info), local=local, tailnet_name=tailnet if local else "",
                                last_heard_s=0. if info.online and not local else heard, next_check_s=check,  # this very survey just heard it
                                **({"origin": facts[0], "age_s": facts[1], "origin_down": facts[2]} if facts else {})))
    return rows


@router.put("/nodes")
def save_nodes(body: NodeList, services=Depends(get_services)) -> NodeList:
    """Save the list of other machines in the workspace config. Entries are stripped, blank and repeated ones dropped; the saved list
    comes back. 400 with a sentence for an entry that is not a machine address."""
    addresses = studio_cluster.clean_nodes(body.addresses)
    services.set_setting("nodes", addresses)
    return NodeList(addresses=addresses)


@router.get("/merged")
def merged_queue(request: Request, services=Depends(get_services)) -> ClusterQueue:
    """The jobs of every machine, each with the machine it runs on or is pinned to. The peers are asked at the same time, for their
    node and their queue, and answer within two seconds or not at all; a peer that does not keeps its last known rows, marked stale.
    Sync, so the peers are asked off the event loop. Empty when no other machine is listed, except that a job still running on a machine
    that was removed from the list keeps its row, so its note (for example, a refused checkpoint) is still said."""
    addresses = studio_cluster.configured_nodes({"studioConfig": {"nodes": services.setting("nodes", [])}})
    jobs = local_jobs(services)
    if not addresses and not any(getattr(j, "node", "") and j.status == "running" for j in jobs):
        return ClusterQueue(nodes=[], jobs=[])
    state = request.app.state
    if not hasattr(state, "peer_queues"):
        state.peer_queues = studio_cluster_queue.Peers()
    cluster = getattr(getattr(services, "runner", None), "cluster", None)
    own = studio_cluster_queue.own_node(cluster)
    peers = state.peer_queues.survey(addresses)
    running = services.queue()
    progress = ({"job_id": running.job.id, "step": running.step, "total": running.total, "eta_s": running.eta_s}
                if running.job is not None else None)
    peers = _with_store_queues(cluster, peers)
    shared = any(p.info.hostname.lower() == own.hostname.lower() for p in peers)  # a hostname pin would name two machines
    own_pin = own.address if shared or not own.hostname else own.hostname
    own_limits = studio_cluster_queue.limits_of({"plan": running.plan, "blocked": running.blocked, "paused": running.paused})
    nodes = [QueueNode(address=own.address, hostname=own.hostname, pin=own_pin, local=True, online=True, limits=own_limits,
                       **_queue_stamp(cluster, own.hostname)),
             *(QueueNode(address=p.info.address, hostname=p.info.hostname, pin=p.info.address, local=False,
                         online=p.info.online, error=p.info.error, seen_at=p.seen_at, listed=p.info.address in addresses,
                         limits=p.limits or _stored_limits(cluster, p.info.hostname), **_queue_stamp(cluster, p.info.hostname))
               for p in peers)]
    rows = studio_cluster_queue.merge(own, peers, jobs, getattr(cluster, "waiting", None), progress)
    return ClusterQueue(nodes=nodes, jobs=[QueueRow(**asdict(r)) for r in rows])


def _stored_limits(cluster, hostname):
    """The limits a node last wrote into this computer's copy of the store: what a machine that does not answer is shown with."""
    if cluster is None or not hostname:
        return None
    return studio_facts.limits_facts(cluster, hostname)


def _queue_stamp(cluster, hostname):
    """origin, age_s and origin_down of a machine's queue as this computer's copy of the observation store holds it."""
    if cluster is None or not hostname:
        return {}
    jobs, age = studio_facts.queue_facts(cluster, hostname)
    if age is None and hostname != cluster.hostname:
        return {}
    return {"origin": hostname, "age_s": 0. if hostname == cluster.hostname else age, "origin_down": cluster.is_down(hostname)}


def _with_store_queues(cluster, peers):
    """The survey's peers, where a peer whose queue could not be read keeps the queue it last wrote into the store (marked stale), and
    a node the store knows that no listed machine answers to is added as a stale peer: the cluster as any node's store has it."""
    if cluster is None:
        return peers
    out, named = [], {cluster.hostname.lower()}
    for peer in peers:
        name = peer.info.hostname or cluster._names.get(peer.info.address, "")  # an offline peer forgets its name; the poll remembers it
        if name and not peer.info.hostname:
            peer = dataclasses.replace(peer, info=dataclasses.replace(peer.info, hostname=name))
        named.add(name.lower())
        if peer.stale and not peer.jobs and name:
            jobs, age = studio_facts.queue_facts(cluster, name)
            if age is not None:
                peer = dataclasses.replace(peer, jobs=tuple(jobs), seen_at=time.time() - age)
        out.append(peer)
    for origin in studio_facts.origins(cluster):
        if origin.lower() in named:
            continue
        found = studio_facts.node_facts(cluster, origin)
        jobs, age = studio_facts.queue_facts(cluster, origin)
        record = found[3] if found else {}
        info = studio_cluster.NodeInfo(address=origin, hostname=origin, version=record.get("version", ""), device=record.get("device", "cpu"),
                                       device_name=record.get("device_name", ""), memory_bytes=record.get("memory_bytes", 0),
                                       busy=bool(record.get("busy")), online=not cluster.is_down(origin),
                                       error=f"{origin} is not on this computer's list; its facts arrive through another node.")
        out.append(studio_cluster_queue.Peer(info, tuple(jobs), stale=True, seen_at=time.time() - (age or 0.)))
    return out


@router.get("/access")
def get_access(request: Request, services=Depends(get_services)) -> Access:
    """The allow lists of this node and its owner's Tailscale login. The login is what a machine that answers 401 to this one must
    add to its own allow_users. The funnel token and any other key of studioConfig.cluster are never returned."""
    try:
        tags, users = studio_cluster.allowed_callers({"studioConfig": {"cluster": services.setting("cluster", {})}})
    except ValueError as exc:  # edited by hand since the API started
        raise HTTPException(409, str(exc)) from None
    return Access(allow_users=list(users), allow_tags=list(tags), login=getattr(getattr(request.app.state, "hosts", None), "login", ""))


@router.put("/access")
def save_access(body: AccessBody, request: Request, services=Depends(get_services)) -> Access:
    """Save the allow lists in the workspace config, keeping every other key of studioConfig.cluster. Entries are stripped and repeats
    dropped. 400 with a sentence for a bad entry. The API reads the lists when it starts: a change counts after it is restarted."""
    stored = services.setting("cluster", {})
    wanted = {"allowUsers": [u.strip() for u in body.allow_users], "allowTags": [t.strip() for t in body.allow_tags]}
    try:
        studio_cluster.allowed_callers({"studioConfig": {"cluster": wanted}})
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    kept = {k: list(dict.fromkeys(v)) for k, v in wanted.items()}
    services.set_setting("cluster", {**(stored if isinstance(stored, dict) else {}), **kept})
    return get_access(request, services)


@router.post("/remote/{job_id}/stop", status_code=204)
def stop_remote(job_id: str, body: RemoteStop, services=Depends(get_services)) -> None:
    """Stop, or drop, a job that another machine queued, by asking that machine (DELETE /queue/jobs/{id} there). The machine must be
    one of the listed peers. A job queued here is stopped with DELETE /queue/jobs/{id} on this node. The peer's own refusal (401,
    404, a timeout) comes back as 502 with its sentence."""
    addresses = studio_cluster.configured_nodes({"studioConfig": {"nodes": services.setting("nodes", [])}})
    if body.address not in addresses:
        raise HTTPException(404, f"{body.address} is not one of this cluster's machines.")
    try:
        studio_peer.PeerClient(body.address, retries=0).delete_json(f"queue/jobs/{job_id}")
    except studio_peer.PeerError as exc:
        raise HTTPException(502, str(exc)) from None


@router.get("/jobs")
def jobs(cluster: Cluster) -> list[ClusterJob]:
    """Jobs queued here that run or wait to run on a peer (with the reason for a wait), and jobs this node runs for another."""
    return [ClusterJob.model_validate(row) for row in cluster.jobs_view()]


@router.post("/jobs")
def dispatch(body: Dispatch, request: Request, cluster: Cluster) -> Accepted:
    """Take a job another node owns. 409 when this node cannot run it (no version store, a kind that does not leave its owner) or the
    lease is older than one it holds."""
    lease = sched.Lease.from_dict(body.lease.model_dump())
    if body.job.id != lease.job_id:
        raise HTTPException(400, "The lease is for another job.")
    host = request.client.host if request.client else ""
    owner = body.owner_address or (f"[{host}]" if ":" in host else host)
    cluster.accept(body.job.model_dump(), lease, owner, body.fetch)
    return Accepted(job_id=lease.job_id, epoch=lease.epoch)


@router.post("/jobs/{job_id}/stop", status_code=204)
def stop(job_id: str, body: Stop, cluster: Cluster) -> None:
    """The owner no longer wants this copy of the job. 409 when the epoch is not the one this node holds."""
    cluster.stop_guest(job_id, body.epoch)


@router.post("/jobs/{job_id}/renew")
def renew(job_id: str, body: Renewal, cluster: Cluster) -> Renewed:
    """A runner extends its lease and reports progress; its events are re-emitted here as if the job ran here. 409 for any epoch but
    the current one: the job has been placed again, and the runner must stop its copy."""
    lease = cluster.renewed(job_id, body.model_dump())
    return Renewed(epoch=lease.epoch, duration_s=lease.duration_s)


@router.post("/jobs/{job_id}/finished", status_code=204)
def finished(job_id: str, body: Finished, cluster: Cluster) -> None:
    """A runner's job ended; the finished version is in the store. 409 for any epoch but the current one."""
    cluster.finished(job_id, body.model_dump())
