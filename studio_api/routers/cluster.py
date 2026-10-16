"""The cluster: the machines in it, the jobs placed on or taken from other nodes, and the node-to-node calls of the scheduler
(docs/distributed.md, "Scheduler"). The rules are studio_scheduler's, the I/O studio_remote's; these routes only carry them.

The machines: `GET /cluster/nodes` surveys this node and the peers listed in the workspace config (`studioConfig.nodes`), and
`PUT /cluster/nodes` saves that list (docs/cluster.md). Neither needs a job queue.

Calls between nodes: `POST /cluster/jobs` (owner to runner: here is a job and its lease), `POST /cluster/jobs/{id}/stop`, and, to the
owner, `POST /cluster/jobs/{id}/renew` and `.../finished`. Every write to the owner carries the lease's epoch; any epoch but the
current one is answered 409, and the runner then stops its copy.
"""

from dataclasses import asdict
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

import studio_cluster
import studio_scheduler as sched
from studio_api.app import get_services
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
def nodes(services=Depends(get_services)) -> list[ClusterNode]:
    """This node first, then every listed peer, in list order. A peer that does not answer within two seconds is listed offline with
    the reason, never dropped. Sync, so the GPU probe and the peers (asked at the same time) run off the event loop."""
    own = studio_cluster.own_info(busy=services.queue().job is not None, version=app_version())
    state = {"studioConfig": {"nodes": services.setting("nodes", [])}}
    return [ClusterNode(**asdict(info), local=info.address == studio_cluster.LOCAL)
            for info in studio_cluster.survey(state, own, timeout=studio_cluster.POLL_TIMEOUT)]


@router.put("/nodes")
def save_nodes(body: NodeList, services=Depends(get_services)) -> NodeList:
    """Save the list of other machines in the workspace config. Entries are stripped, blank and repeated ones dropped; the saved list
    comes back. 400 with a sentence for an entry that is not a machine address."""
    addresses = studio_cluster.clean_nodes(body.addresses)
    services.set_setting("nodes", addresses)
    return NodeList(addresses=addresses)


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
