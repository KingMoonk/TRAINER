"""The background collector: a detached process (`python -m studio_collect --root <workspace>`) that survives app restarts.

It reads the settings and the state from data/collect/, so everything it does is resumable: a plan is kept in state.json, each
source keeps a cursor (file and row), the dedup index is committed with every shard, and the journal says what happened.
The app starts it (`ensure_running`) and talks to it only through those files: pause and resume are a settings write the service
notices within a second.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import studio_core
import studio_platform as watch
from studio_collect import config as cfg
from studio_collect.config import collect_dir
from studio_collect.dedup import Seen
from studio_collect.idle import Probes, reason_not_to_run
from studio_collect.netio import Http
from studio_collect.pipeline import Pipeline, eval_index, eval_texts
from studio_collect.store import Journal, Ledger, LimitReached, State, Stopped, disk_used

POLL = 2.0
WAIT_POLL = 10.0
FLUSH_EVERY = 60.0
IDLE_RECHECK = 3600.0
REPLAN = 24 * 3600.0
ADAPTERS = {"hub": "studio_collect.hub", "stackexchange": "studio_collect.stackexchange", "github": "studio_collect.github", "web": "studio_collect.web"}


def plan_key(inputs):
    """A hash of the settings that change what a source plans; changing one makes its plan stale (approvals and blocks only re-filter it)."""
    return hashlib.sha1(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[:12]


def wake_key(config):
    """Everything that should end a long wait: any source's plan inputs."""
    import importlib
    import importlib.util
    return plan_key({name: importlib.import_module(module).Adapter.plan_inputs(config)
                     for name, module in ADAPTERS.items() if importlib.util.find_spec(module) is not None})
def take_lock(path):
    """An exclusive OS lock on `path`, held while the returned file stays open; None when another process holds it."""
    handle = open(path, "a+b")
    try:
        if sys.platform == "win32":
            import msvcrt
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    return handle


class Context:
    """What an adapter works with. `stop()` is called often: it raises Stopped the moment collection must not go on."""

    def __init__(self, root, config, state, journal, ledger, http, pipeline, probes, clock):
        self.root, self.config, self.state, self.journal, self.ledger = Path(root), config, state, journal, ledger
        self.http, self.pipeline, self.probes, self.clock = http, pipeline, probes, clock
        self._checked, self._beat = 0.0, 0.0
        self.adapters, self.flushed = {}, clock()

    def stop(self):
        now = self.clock()
        if now - self._checked < 0.5:
            return
        self._checked = now
        self.config = cfg.load(self.root)
        self.http.limits = self.config["limits"]
        self.pipeline.opt_in = self.config["licences"]
        reason = reason_not_to_run(self.config, self.root, self.probes)
        if reason:
            raise Stopped(reason)
        if now - self._beat > 5:
            self._beat = now
            self.state.save(heartbeat=now)

    def plan_items(self, source):
        return [i for i in self.state.data["plan"] if i.get("source") == source]

    def set_plan(self, source, items):
        """Replace one source's items in the plan (the other sources' items stay)."""
        self.state.save(plan=[i for i in self.state.data["plan"] if i.get("source") != source] + list(items))

    def reserve(self, nbytes=0):
        """LimitReached when `nbytes` more would pass the disk cap."""
        cap = self.config["limits"]["disk_mb"] * 1_000_000
        if disk_used(self.root) + nbytes > cap:
            raise LimitReached(f"The disk cap of {self.config['limits']['disk_mb']:g} MB is reached. Records move to your data when the app registers them.")

    def flush(self):
        self.flushed = self.clock()
        out = self.pipeline.flush()
        self.state.add_stats(out["stats"])
        for group, stats in out["groups"].items():
            self.state.add_stats(stats, group)
        self.ledger.flush()
        self.state.save()
        return out


def adapters(ctx):
    import importlib
    found = []
    for name, module in ADAPTERS.items():
        if ctx.config["sources"].get(name):
            try:
                if name not in ctx.adapters:      # kept for the whole run: the crawler's robots.txt cache and frontier live in it
                    ctx.adapters[name] = importlib.import_module(module).Adapter(ctx)
                found.append(ctx.adapters[name])
            except ModuleNotFoundError as error:
                if error.name != module:
                    raise
    return found


def _status(state, status, reason="", clock=time.time):
    if (state.data["status"], state.data["reason"]) != (status, reason) or clock() - (state.data["heartbeat"] or 0) > 5:
        state.save(status=status, reason=reason, heartbeat=clock())


def run(root, *, transport=None, probes=None, sleep=time.sleep, clock=time.time, once=False, lock=True):
    """Collect until the settings say off. `once`: return after the first pass that did work or found none (tests)."""
    root = Path(root)
    collect_dir(root).mkdir(parents=True, exist_ok=True)
    handle = None
    if lock:
        handle = take_lock(collect_dir(root) / "service.lock")
        if handle is None:
            return "another"          # a collector already runs for this workspace
    state, journal = State(root), Journal(root)
    probes = probes or Probes()
    state.save(pid=__import__("os").getpid())
    journal.append("start", pid=state.data["pid"])
    seen = pipeline = ctx = None
    try:
        while True:
            config = cfg.load(root)
            if not config["enabled"]:
                _status(state, "off", "Collection is off.", clock)
                return "off"
            reason = reason_not_to_run(config, root, probes)
            if reason:
                _status(state, "paused" if config["paused"] else "waiting", reason, clock)
                if once:
                    return "waiting"
                sleep(POLL if config["paused"] else WAIT_POLL)
                continue
            if ctx is None:
                ledger = Ledger(root, clock)
                seen = Seen(root)
                workspace = studio_core.Workspace(root)
                added = seen.absorb_workspace(workspace)
                pipeline = Pipeline(root, seen, eval_index(eval_texts(root)), config["licences"], clock)
                transport_http = Http(ledger, config["limits"], **({"transport": transport} if transport else {}), sleep=sleep)
                ctx = Context(root, config, state, journal, ledger, transport_http, pipeline, probes, clock)
                transport_http.check = ctx.stop
                journal.append("ready", known_records=added)
            ctx.config = config
            _status(state, "running", "", clock)
            try:
                worked = work(ctx, clock)
                if not worked or clock() - ctx.flushed > FLUSH_EVERY:
                    ctx.flush()          # a source that reads small pieces (a crawl) fills a shard slowly: write what it has now and then
                state.save()
                if not worked:
                    _status(state, "idle", "Nothing left to collect with these settings. It looks again later.", clock)
                    if once:
                        return "idle"
                    wait(ctx, IDLE_RECHECK, sleep)
                elif once:
                    return "worked"
            except Stopped as stop:
                ctx.flush()
                _status(state, "paused" if config["paused"] else "waiting", str(stop), clock)
                if once:
                    return "stopped"
            except LimitReached as full:
                ctx.flush()
                journal.append("limit", message=str(full))
                _status(state, "waiting", str(full), clock)
                if once:
                    return "limit"
                wait(ctx, 300, sleep)
            except Exception as error:   # noqa: BLE001 - one bad step must not end the service
                ctx.flush()
                journal.append("error", message=f"{type(error).__name__}: {error}", trace=traceback.format_exc()[-1500:])
                _status(state, "error", f"{type(error).__name__}: {error}", clock)
                if once:
                    return "error"
                wait(ctx, 60, sleep)
    finally:
        if ctx is not None:
            try:
                ctx.flush()
            except Exception:    # noqa: BLE001 - leaving must not fail on a full disk; the records stay in the next run's re-read
                pass
        for adapter in (ctx.adapters.values() if ctx is not None else ()):
            getattr(adapter, "close", lambda: None)()
        if seen is not None:
            seen.close()
        journal.append("exit")
        if handle is not None:
            handle.close()


def wait(ctx, seconds, sleep):
    """Sleep in short steps so a settings change (pause, off, a new topic) is noticed within seconds."""
    end, waking = ctx.clock() + seconds, wake_key(cfg.load(ctx.root))
    while ctx.clock() < end:
        sleep(min(POLL, max(0.0, end - ctx.clock())))
        config = cfg.load(ctx.root)
        if not config["enabled"] or config["paused"] or wake_key(config) != waking:
            return


def work(ctx, clock):
    """One step of the first source with something to do. False when no source has anything left."""
    for adapter in adapters(ctx):
        made = ctx.state.data["plans"].get(adapter.key, {})
        key = plan_key(adapter.plan_inputs(ctx.config))
        if made.get("key") != key or clock() - (made.get("at") or 0) > REPLAN:
            adapter.plan()
            ctx.state.data["plans"][adapter.key] = {"key": key, "at": clock()}
            ctx.state.save()
        if adapter.step():
            return True
    return False


# ---- the app's side -------------------------------------------------------------------------------------

def running(root, *, alive=watch.alive, ours=watch.ours):
    pid = State(root).data.get("pid")
    return bool(pid) and alive(pid) and ours(pid, "studio_collect")


def ensure_running(root, *, spawn=watch.spawn, alive=watch.alive, ours=watch.ours):
    """Start the collector if the settings want it and it is not running. Returns its pid, or None when it is not wanted."""
    if not cfg.load(root)["enabled"]:
        return None
    state = State(root)
    if running(root, alive=alive, ours=ours):
        return state.data["pid"]
    pid = spawn(studio_core.module_command("studio_collect", "--root", str(Path(root).resolve())), collect_dir(root) / "service.log", Path(__file__).resolve().parent.parent)
    state.save(pid=pid)
    return pid


def main(argv=None):
    parser = argparse.ArgumentParser(prog="studio_collect", description="Passive training-data collector.")
    parser.add_argument("--root", default=str(studio_core.ROOT))
    parser.add_argument("--once", action="store_true", help="do one pass and exit")
    args = parser.parse_args(argv)
    print(run(args.root, once=args.once))
    return 0


if __name__ == "__main__":
    sys.exit(main())
