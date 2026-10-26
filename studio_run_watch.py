"""Watching a pretraining run while it trains: what the machine has now against what the run needs, and where the pace puts its end.

studio_fit sizes a run once, before it starts. This module is what the queue asks every `WATCH_EVERY` seconds while the run goes (and on
each progress event), and its answers are notices, `{"kind", "text", "at"}`, shown on the run page and kept in the job's note:

- `resources`: the free disk or RAM has dropped below what the run still needs for its checkpoints (`needs`). A reading the machine
  could not give is skipped, never guessed.
- `overrun`: at the pace measured so far the run ends after the time window the person allowed (`projection`).
- `oom`: training halved its batch after running out of memory (`oom_notice`, from the event studio_model.train emits).

Nothing here stops or restarts a run."""
from __future__ import annotations

import time

import studio_fit
from studio_fit import CHECKPOINTS, hours_text, size_text

WATCH_EVERY = 30.  # seconds between two hardware readings
OOM_SHOWN = 5  # the newest halvings kept (a batch halves at most log2 times)
MIN_STEPS_HERE = 3  # steps this process has trained before its pace is trusted for a projection
MIN_ELAPSED = 30.  # ... and seconds


def notice(kind, text, now=None):
    return {"kind": kind, "text": text, "at": time.time() if now is None else now}


def needs(params):
    """What the run still needs from the machine that the machine may stop having: {"disk": bytes, "ram": bytes}, or {} when its config
    cannot be read. Disk: room for the checkpoint it writes (the last, the best and the copy being written, each with the optimizer's
    state); RAM: the copy of the weights a checkpoint makes, and the optimizer's state when it lives in host memory. The text it reads is
    not counted: most of it is already on disk and in memory by now, so a shortfall of it would not show in the free figures."""
    import studio_memory
    from studio_model import validate_config, vocabulary_guess
    try:
        cfg = validate_config(params["config"])
        weights = studio_memory.weights_of(cfg, vocabulary_guess(cfg))
        host = studio_memory.host_bytes(cfg, vocabulary_guess(cfg), True) if params.get("offload_optimizer") else 0
    except Exception:  # noqa: BLE001 - a config that cannot be sized has nothing to compare
        return {}
    return {"disk": CHECKPOINTS * weights * 3, "ram": 2 * weights + host}


def resources(profile, need, now=None):
    """Notices for each of disk and RAM whose free figure in `profile` is below `need` (see `needs`)."""
    out = []
    if need.get("disk") and profile.disk_free is not None and profile.disk_free < need["disk"]:
        out.append(notice("disk", f"Free disk is down to {size_text(profile.disk_free)} while the run goes; its checkpoints need about "
                                  f"{size_text(need['disk'])}. Free space in Resources > Storage or the next checkpoint may fail.", now))
    if need.get("ram") and profile.ram_free is not None and profile.ram_free < need["ram"]:
        out.append(notice("ram", f"Free RAM is down to {size_text(profile.ram_free)} while the run goes; it needs about {size_text(need['ram'])} "
                                 "to save a checkpoint. Close other apps or the run may be stopped by the system.", now))
    return out


def oom_notice(event, now=None):
    """The notice for studio_model.train's out-of-memory event ({"kind": "oom", "step", "batch", "accumulation"}), or None for any other."""
    if event.get("kind") != "oom":
        return None
    batch, accumulation = event.get("batch"), event.get("accumulation")
    return notice("oom", f"Ran out of memory at step {event.get('step')}: the batch was halved to {batch} with accumulation {accumulation} "
                         f"({batch * accumulation} rows a step) and training went on.", now)


def projection(*, step, first_step, total, trained_tokens, elapsed, token_budget, window_seconds, earlier_tokens=0, earlier_elapsed=0., now=None):
    """The overrun notice when the run, at the pace measured since it began, ends after `window_seconds`; None when it will end in time,
    when there is no window, or when the pace is not measured yet. `step`/`total`: where the run is and ends (steps); `first_step`: the
    step this process began at; `trained_tokens`/`elapsed`: what the run did in all, counting the attempts before a resume
    (`earlier_tokens`/`earlier_elapsed`, which the pace leaves out: it is measured on this attempt alone, while the window and the token
    budget are for the whole run). `token_budget`: the run's cap on tokens, if any."""
    steps_here, tokens_here, elapsed_here = step - first_step, trained_tokens - earlier_tokens, elapsed - earlier_elapsed
    if not window_seconds or not elapsed_here or steps_here < MIN_STEPS_HERE or elapsed_here < MIN_ELAPSED or tokens_here <= 0 or step >= total:
        return None
    rate = tokens_here / elapsed_here
    left = (total - step) * tokens_here / steps_here
    if token_budget:
        left = min(left, max(0, token_budget - trained_tokens))
    ends = elapsed + left / rate
    if ends <= window_seconds:
        return None
    return notice("overrun", f"At {rate:,.0f} tokens/s the run ends in about {hours_text(left / rate / 3600)}, "
                             f"{hours_text(ends / 3600)} in all: {hours_text((ends - window_seconds) / 3600)} past the "
                             f"{hours_text(window_seconds / 3600)} window you allowed.", now)


def window_seconds(params):
    """The time window of a pretrain job, in seconds, or None when none was set or the run adapts to it itself: a job with `wall_seconds`
    re-plans its schedule from its measured pace and ends inside the window, so it cannot overrun."""
    if params.get("wall_seconds"):
        return None
    hours = (params.get(studio_fit.RECORD) or {}).get("window_hours")
    return hours * 3600 if hours else None
