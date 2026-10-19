"""The watchdog of a distributed run this node coordinates (#479): the run's driver is this computer's own member (the owner job's
worker), and every call it makes to the coordinator is its heartbeat (studio_diloco.Coordinator.heard_ago). When the heartbeat stops, or
the owner job ended while its coordinator still goes, the run is ended here with a reason and the machines' member jobs are released, so
the members and the Training tab do not wait on a frozen round. A member whose coordinator stops answering ends itself
(studio_diloco.Member, `patience`). The clock is the coordinator's; the only I/O is the peer calls `studio_distributed.stop_run` makes."""
from __future__ import annotations

from studio_peer import PeerClient

# A member calls the coordinator at least every PROGRESS_SECONDS while it trains and every LONG_POLL while it waits for a round, so a
# silence this long is a dead driver, not a slow round: it is meant to outlast a round's validation and a big upload on a slow link
# (BEAST, Windows CUDA), and the worker's start (loading the model, fetching shards) before its first call.
DRIVER_TIMEOUT = 900.  # seconds the owner's own member may go without a call before its run is ended
SILENT = "Ended: the driver stopped responding at round {round}, so the run was stopped."
GONE = "Ended: the driver stopped at round {round}{why}, so the run was stopped."
OVER = ("failed", "stopped")  # the owner job statuses that mean its worker is gone


def driver_of(run):
    """The node name of this computer's own member of `run` (studio_distributed.Machine.local), or None."""
    return next((m.node for m in run.machines if m.local), None)


def reason_for(coordinator, job, driver, timeout=DRIVER_TIMEOUT):
    """Why the run should end now, or "" when its driver is fine: the owner job ended while the coordinator still goes, or the
    driver has not called it for `timeout` seconds while the job still says it runs."""
    if coordinator.finished:
        return ""
    if job.status in OVER:
        return GONE.format(round=coordinator.next_round, why=f" ({job.error})" if job.error else "")
    if job.status == "running" and driver and coordinator.heard_ago(driver) > timeout:
        return SILENT.format(round=coordinator.next_round)
    return ""


def end(runner, run, coordinator, job, reason, client_for=PeerClient):
    """End `run` with `reason`: the coordinator finishes (every member leaves at its next call), a worker still holding the owner job is
    stopped and the job ends failed with the reason, and the machines' member jobs are taken off their queues."""
    import studio_distributed
    coordinator.stop(reason)
    if job.status == "running":
        with runner.lock:
            job.params = {**job.params, "ended_reason": reason}  # the runner makes it the job's error when the worker has gone
        runner.remove(job.id)
    studio_distributed.stop_run(run, coordinator, client_for)


def check(runner, coordinators, client_for=PeerClient, timeout=DRIVER_TIMEOUT):
    """One pass over the coordinators of the runs this node owns; the ids of the runs it ended."""
    import studio_distributed
    with runner.lock:
        jobs = {j.id: j for j in runner.jobs if j.kind == "pretrain_distributed" and not j.owner}
    ended = []
    for run_id, coordinator in list(coordinators.items()):
        job, run = jobs.get(run_id), studio_distributed.RUNS.get(run_id)
        if job is None or run is None:
            continue
        reason = reason_for(coordinator, job, driver_of(run), timeout)
        if reason:
            end(runner, run, coordinator, job, reason, client_for)
            ended.append(run_id)
    return ended
