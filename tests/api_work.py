"""Helpers for the router tests that start jobs: wait for a job's "job" events on /events and read its result."""
import copy

from studio_spec import default_spec, validate_spec


def small_spec(layers=2):
    spec = default_spec(preset="classic")   # the classic recipe: the tests built on this helper expect LayerNorm, GELU and a cosine schedule
    spec["dims"].update(d_model=16, n_layer=layers, n_ctx=64, dropout=0.)
    return validate_spec(spec)


def trained(services, change="trained"):
    """Save a version on FakeServices and give it a checkpoint, as a finished run does."""
    version = services.save_version(copy.deepcopy(small_spec()), change)
    version.best_checkpoint = "best.npz"
    version.status = "trained"
    return version


def finished(tap, response, timeout=10.):
    """The "job" event that ends the job `response` started (done, failed or cancelled), and the job events it sent, in order."""
    assert response.status_code == 202, response.text
    job_id = response.json()["id"]
    ends = lambda: any(p["id"] == job_id and p["status"] in ("done", "failed", "cancelled") for p in tap.of("job"))  # noqa: E731
    tap.wait_for(ends, f"job {job_id} to end", timeout)
    mine = [p for p in tap.of("job") if p["id"] == job_id]
    return mine[-1], mine
