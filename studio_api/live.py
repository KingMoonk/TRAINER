"""Whether a run is live on this API: the one decision both ways of restarting it share. A queue job that runs, or a distributed run
this computer coordinates, lives in the API's own process, so a restart ends it. The `git pull` hook (.claude/hooks/sync-base.sh,
`python -m studio_api.live`) and `--reload` (the marker file below) both hold back while this says live.

Standard library only: the hook runs it with whatever python3 the machine has."""

import json
import os
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

MARKER_ENV = "STUDIO_API_LIVE_MARKER"  # --reload: the file the API process keeps fresh while a run is live; the reloader reads it
ACTIVE_STATES = ("starting", "waiting", "running")  # a distributed run's `state` while its coordinator is up in this process
STALE_S = 15.  # a marker older than this is a crashed API's leftover, not a live run
EVERY_S = 3.  # how often the API process refreshes the marker
DEFAULT_URL = "http://127.0.0.1:8787"  # studio_cluster.DEFAULT_PORT, written out to stay free of imports
FETCH_TIMEOUT_S = 2.


def live_reason(queue, runs):
    """Why a restart would end a run, or None when idle. `queue` is GET /queue (its `job` is the running job, else null) and `runs` is
    GET /distributed/runs. Only a run this computer coordinates counts (`age_s` 0 and its origin up): the others are replicas of
    another node's account, which a restart here does not touch."""
    job = (queue or {}).get("job")
    if job and job.get("status", "running") == "running":
        return f"queue job {job.get('label') or job.get('id') or '?'} is running"
    for run in runs or ():
        if run.get("state") in ACTIVE_STATES and not run.get("age_s") and not run.get("origin_down"):
            return f"distributed run {run.get('label') or run.get('id') or '?'} is {run['state']} (round {run.get('round', '?')})"
    return None


def marker_path(pid=None):
    """The marker of the reloader process `pid` (default this one) in the temp folder."""
    return Path(tempfile.gettempdir()) / f"groundwork-live-{os.getpid() if pid is None else pid}"


def write_marker(path, reason):
    """Refresh the marker while live (`reason` text), remove it when idle (None)."""
    try:
        if reason:
            Path(path).write_text(reason, encoding="utf-8")
        else:
            Path(path).unlink(missing_ok=True)
    except OSError:
        pass  # the next tick tries again


def marker_reason(path, now=None, stale_s=STALE_S):
    """The reason in a fresh marker, or None when there is none or it has not been refreshed for `stale_s` (a crashed process)."""
    try:
        path = Path(path)
        if (time.time() if now is None else now) - path.stat().st_mtime > stale_s:
            return None
        return path.read_text(encoding="utf-8") or "a run is live"
    except OSError:
        return None


def fetch(url, timeout=FETCH_TIMEOUT_S):
    with urllib.request.urlopen(url, timeout=timeout) as response:  # a fixed http://127.0.0.1 URL from this module's callers
        return json.load(response)


def probe(base=DEFAULT_URL, get=fetch):
    """(live reason or None) from the API at `base`; raises OSError/ValueError when it is not up or answers nonsense."""
    return live_reason(get(f"{base}/queue"), get(f"{base}/distributed/runs"))


def main(argv=None):
    """`python -m studio_api.live [URL]`: prints the reason and exits 0 when a run is live, 1 when idle, 2 when the API does not answer."""
    argv = sys.argv[1:] if argv is None else argv
    try:
        reason = probe(argv[0].rstrip("/") if argv else DEFAULT_URL)
    except (OSError, ValueError):
        return 2
    if reason:
        print(reason)
    return 0 if reason else 1


if __name__ == "__main__":
    sys.exit(main())
