"""This computer as a member of other computers' distributed runs: what it was invited to, in the words the run's owner shows for it
(studio_distributed.member_rows), and the three things a person here can do about it. Joining needs no accept step: the owner put a
`diloco_member` job on this computer's queue (studio_distributed.invite) and it runs by itself once the queue is free.

* an invite is a `diloco_member` job in the local queue, read with the owner's account of the run when the observation store has it;
* a pause request is another computer asking this one to step its running job aside; it is only ever shown, never acted on;
* leaving tells the owner to drop this member and takes the member job out of the queue; pausing the running job and joining puts the
  member job ahead of it (studio_queue.Runner.yield_running).
"""
from __future__ import annotations

import threading
import time
import uuid
from urllib.parse import urlparse

from studio_peer import PeerClient

LIVE = ("waiting", "held", "running")
ASK_TIMEOUT = 5.  # seconds the owner has to answer a leave, or a peer a pause request
REQUESTS = {}  # pause request id -> {"id", "from_node", "run_label", "at"}
_LOCK = threading.Lock()


def member_jobs(jobs):
    """The member jobs of `jobs` (anything with kind, status, params) that are still to run or running."""
    return [job for job in jobs if job.kind == "diloco_member" and job.status in LIVE]


def blocking_of(running, eta_s):
    """The local job a member job waits behind, as {"label", "eta_s"}, or None."""
    return None if running is None else {"label": running.label or running.kind, "eta_s": eta_s}


def owner_of(url):
    """The owner's name as far as its address tells: the first label of the host in `url` ("" when there is none)."""
    return (urlparse(url or "").hostname or "").split(".")[0]


def invite_row(job, running, eta_s, run):
    """One invite as the banner shows it. `running` is the local job that runs now (None when idle) and `eta_s` its time left; `run` is
    the owner's row of the run from the observation store, or None when this computer has not heard of it yet. The state is the
    owner's own word for this machine, except that this computer knows first what its own queue is doing."""
    params = job.params
    member = next((m for m in (run or {}).get("members", []) if m["node"] == params.get("node")), None)
    waiting = job.status != "running"
    blocking = blocking_of(running, eta_s) if waiting and running is not None and running.id != job.id else None
    if blocking is not None:
        state = "waiting"
        left = "" if eta_s is None else f", about {max(1, round(eta_s / 60))} min left"
        reason = f"Busy with '{blocking['label']}'{left}. It joins when that finishes."
    elif member is not None and not (member["state"] in ("invited", "waiting") and not waiting):
        state, reason, blocking = member["state"], member.get("reason") or member.get("note", ""), member.get("blocking_job")
    else:
        state, reason = ("invited", "Added to the queue. It joins by itself once this computer is free.") if waiting else ("joining", "Starting up.")
    label = (run or {}).get("label") or (job.label or "").removesuffix(" (member)")
    return {"job_id": job.id, "run_id": params.get("run", ""), "run_label": label, "owner": (run or {}).get("owner") or owner_of(params.get("coordinator_url", "")),
            "node": params.get("node", ""), "state": state, "reason": reason, "blocking_job": blocking,
            "round": (run or {}).get("round"), "rounds": (run or {}).get("rounds"),
            "tokens_per_second": (member or {}).get("live_tokens_per_second") or (member or {}).get("tokens_per_second"),
            "can_pause_and_join": state == "waiting" and blocking is not None and waiting}


# ---- pause requests ---------------------------------------------------------------------------------------------------

def ask_here(from_node, run_label):
    """Remember that `from_node` asked this computer to pause its running job. One request per computer and run; asking again
    refreshes it."""
    with _LOCK:
        found = next((r for r in REQUESTS.values() if r["from_node"] == from_node and r["run_label"] == run_label), None)
        if found is None:
            found = REQUESTS[uuid.uuid4().hex[:8]] = {"from_node": from_node, "run_label": run_label}
            found["id"] = next(k for k, v in REQUESTS.items() if v is found)
        found["at"] = time.time()
        return dict(found)


def requests_row(request, running, eta_s):
    return {**request, "blocking_job": blocking_of(running, eta_s)}


def pending_requests(running, eta_s):
    """The pause requests that still mean something: they name the job running now. With nothing running they are answered, and go."""
    with _LOCK:
        if running is None:
            REQUESTS.clear()
        return [requests_row(r, running, eta_s) for r in sorted(REQUESTS.values(), key=lambda r: r["at"])]


def dismiss(request_id):
    """Drop a pause request. False when there is none by that id."""
    with _LOCK:
        return REQUESTS.pop(request_id, None) is not None


def ask_peer(address, from_node, run_label, client_for=None):
    """Ask the computer at `address` to pause its running job (POST /distributed/pause-requests there). Raises studio_peer.PeerError."""
    return (client_for or PeerClient)(address, timeout=ASK_TIMEOUT, retries=0).post_json("/distributed/pause-requests", {"from_node": from_node, "run_label": run_label})


# ---- leaving ----------------------------------------------------------------------------------------------------------

def tell_owner_left(job, client_for=None):
    """Ask the owner of the run to drop this member and keep it out (POST /distributed/runs/{run}/members/{node}/stop on the owner, the
    call that also takes the member job off this queue). True when the owner answered; False when it could not be reached or
    refused, in which case the member job is removed here anyway."""
    params = job.params
    url = params.get("coordinator_url")
    if not url or not params.get("run") or not params.get("node"):
        return False
    try:
        (client_for or PeerClient)(url, timeout=ASK_TIMEOUT, retries=0).post_json(f"/distributed/runs/{params['run']}/members/{params['node']}/stop", {})
        return True
    except Exception:  # noqa: BLE001 - the owner may be down or may have forgotten the run; leaving here still works
        return False
