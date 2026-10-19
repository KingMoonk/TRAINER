"""python -m studio_api [--fake | --workspace DIR] [--bind HOST] [--port N] [--allow-origin ORIGIN] [--allow-host NAME] [--allow-cpu] [--reload] [--dump-openapi FILE]"""

import argparse
import ipaddress
import json
import logging
import math
import os
import re
import sys
import threading
from pathlib import Path

import studio_cluster
from studio_api import accelerator, live
from studio_api.hosts import allow_host

GRACEFUL_SHUTDOWN = 2  # seconds uvicorn waits for open connections (/events streams never end) before closing them on stop or reload
LOCAL = ("127.0.0.1", "localhost", "::1")
DEFAULT_PORTS = {"http": 80, "https": 443}
ORIGINS = "STUDIO_API_ALLOW_ORIGINS"  # comma-separated web origins, added to every --allow-origin
ARGV = "STUDIO_API_ARGV"  # how --reload hands the command line to the process uvicorn restarts
ROOT = Path(__file__).resolve().parent.parent


def step_duration(value):
    seconds = float(value)
    if not math.isfinite(seconds) or seconds < 0:
        raise argparse.ArgumentTypeError("fake step duration must be finite and nonnegative")
    return seconds


def schema_json():
    """The text of web/openapi.json, from the routes over FakeServices. tests/test_api_schema.py compares against it, so one
    function writes it. info.version is pinned: app_version() comes from the checkout's tags, so it differs per checkout and
    would make the file stale on every commit."""
    from studio_api.app import create_app
    from studio_api.hosts import ANY
    from studio_ia.services import FakeServices

    schema = create_app(FakeServices(), open=True, hosts=ANY).openapi()
    schema["info"]["version"] = "0"
    return json.dumps(schema, indent=2, sort_keys=True) + "\n"


LABEL = r"[A-Za-z0-9]([A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
ORIGIN = re.compile(rf"(?P<scheme>[A-Za-z]+)://(?P<host>\[[0-9A-Fa-f:.]+\]|{LABEL}(\.{LABEL})*)(:(?P<port>[0-9]{{1,5}}))?")


def origin(value):
    """`https://app.vercel.app` from a typed origin: a scheme, a host and an optional port, nothing else (no path, no `*`, no
    space). The scheme and host are lowercased; a port outside 1-65535 is refused, not rewritten to the default."""
    text = value.strip()
    found = ORIGIN.fullmatch(text[:-1] if text.endswith("/") else text)  # one trailing slash only
    scheme = found and found["scheme"].lower()
    port = found and int(found["port"]) if found and found["port"] else None
    try:
        host = f"[{ipaddress.IPv6Address(found['host'][1:-1]).compressed}]" if found and found["host"].startswith("[") else None
        if host and ipaddress.IPv6Address(found["host"][1:-1]).ipv4_mapped:
            found = None  # browsers send the dotted form unpredictably; name the IPv4 address instead
    except ValueError:
        found = None
    if not found or scheme not in DEFAULT_PORTS or (found["port"] and not 0 < port < 65536):
        raise argparse.ArgumentTypeError(f"{value!r} is not a web origin like https://app.example.com (scheme, host, optional port 1-65535).")
    return f"{scheme}://{host or found['host'].lower()}" + (f":{port}" if port and port != DEFAULT_PORTS[scheme] else "")


def parse(argv=None, prog="python -m studio_api"):
    parser = argparse.ArgumentParser(prog=prog)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--fake", action="store_true", help="serve FakeServices (no workspace)")
    source.add_argument("--workspace", type=Path, help="existing folder containing data/ and runs/ (useful from a worktree)")
    parser.add_argument("--fake-step-seconds", type=step_duration, default=.05,
                        help="seconds per simulated training step with --fake (default: 0.05)")
    parser.add_argument("--bind", default="127.0.0.1", help="address to listen on (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=studio_cluster.DEFAULT_PORT, help="port (default: the port other nodes expect)")
    parser.add_argument("--allow-origin", action="append", default=[], type=origin, metavar="ORIGIN",
                        help=f"a web origin (the Vercel site) allowed to call the API from a browser once authenticated; repeat for "
                             f"several, or list them comma-separated in ${ORIGINS} (default: none)")
    parser.add_argument("--allow-host", action="append", default=[], type=allow_host, metavar="NAME",
                        help="a host name besides localhost, an IP address, this machine's name and its Tailscale names that requests "
                             "may carry in Host (a custom DNS name or proxy; a * stands for one label); repeat for several")
    parser.add_argument("--allow-cpu", action="store_true",
                        help="run on the CPU even when this machine has a GPU; without it the API refuses to start when torch reports cpu "
                             "on a machine with one (--fake is never checked)")
    parser.add_argument("--reload", action="store_true", help="restart when a .py file changes (development)")
    parser.add_argument("--dump-openapi", metavar="FILE", help="write the OpenAPI schema to FILE and exit (pnpm gen:api reads web/openapi.json)")
    args = parser.parse_args(argv)
    try:
        from_env = [origin(v) for v in os.environ.get(ORIGINS, "").split(",") if v.strip()]
    except argparse.ArgumentTypeError as exc:
        parser.error(f"{ORIGINS}: {exc}")
    args.allow_origin = list(dict.fromkeys(args.allow_origin + from_env))
    if args.workspace is not None:
        args.workspace = args.workspace.expanduser().resolve()
        if not args.workspace.is_dir():
            parser.error(f"Workspace folder does not exist: {args.workspace}")
    return args


def build(args):
    from studio_api.app import create_app
    from studio_api.hosts import own_names

    hosts = own_names(args.bind, args.allow_host)
    if not args.fake:
        print("studio_api", accelerator.check(args.allow_cpu), flush=True)  # SystemExit before anything is served
    hosts.start_refresh()
    if args.fake:
        from studio_ia.services import FakeServices

        services = FakeServices(step_seconds=args.fake_step_seconds, snapshot_every=2)
        services.demo_run()  # the Training tab opens on a live run
    else:
        from studio_core import Workspace
        from studio_ia.services import WorkspaceServices

        workspace = Workspace(args.workspace) if args.workspace is not None else Workspace()
        services = WorkspaceServices(workspace)
        services.runner.cluster.serve_at(args.port, lambda: hosts.tailnet_name)  # where a failover's where-file sends the members
        services.resume_queue_at_start()  # a queue the last run of the app left going carries on by itself
        tags, users = studio_cluster.allowed_callers(workspace.state)
        return create_app(services, open=args.bind in LOCAL, origins=args.allow_origin, hosts=hosts, allow_tags=tags, allow_users=users,
                          token=studio_cluster.funnel_token(workspace.state))
    return create_app(services, open=True, origins=args.allow_origin, hosts=hosts, fake_traffic=True)


def local_reason(services):
    """live.live_reason over this process's own services, without the HTTP round trip: the running queue job, then the runs this
    computer coordinates (the rows GET /distributed/runs lists first)."""
    from studio_api.routers import distributed

    job = services.queue().job
    return live.live_reason({"job": job and {"status": job.status, "label": job.label}}, distributed._live_rows(services))


def keep_marker(services, path, every=live.EVERY_S, reason=local_reason, stop=None):
    """Under --reload: refresh `path` while a run is live and remove it when idle, so the reloader process (which cannot see this
    process's runs) holds a restart. A failed read leaves the marker alone: it goes stale on its own and a reload is never held for good."""
    stop = stop or threading.Event()
    while not stop.is_set():
        try:
            live.write_marker(path, reason(services))
        except Exception as exc:  # noqa: BLE001 - a bad read must not end the thread
            print(f"studio_api: live-run check failed: {exc!r}", flush=True)
        stop.wait(every)


class DeferredReload:
    """Mixin for uvicorn's reloader: a source change found while `reason()` says a run is live is held, logged once, and applied on the
    first check after the run ends. uvicorn's StatReload keeps reporting a changed file until it restarts and watchfiles reports it
    once, so the held files are remembered here rather than left to the base class."""

    def hold(self, changes, reason):
        held = getattr(self, "held", [])
        held = held + [c for c in (changes or []) if c not in held]
        why = reason()
        if held and why:
            if not getattr(self, "told", False):
                self.told = True
                logging.getLogger("uvicorn.error").warning("Reload waiting: %s. The API restarts when it ends (%d changed file(s) held).", why, len(held))
            self.held = held
            return None
        self.held, self.told = [], False
        return held or None


def deferred_reloader():
    """uvicorn's own reloader class (StatReload, or WatchFilesReload with watchfiles installed) that waits out a live run."""
    from uvicorn.supervisors import ChangeReload

    class Deferring(DeferredReload, ChangeReload):
        def should_restart(self):
            return self.hold(super().should_restart(), lambda: live.marker_reason(os.environ[live.MARKER_ENV]))

    return Deferring


def run_reloading(argv, args):
    """`uvicorn.run(..., reload=True)` with the reloader swapped for deferred_reloader(); everything else is uvicorn's own."""
    import uvicorn

    os.environ[ARGV] = json.dumps(argv)
    os.environ[live.MARKER_ENV] = str(marker := live.marker_path())
    config = uvicorn.Config("studio_api.__main__:reloaded", factory=True, reload=True, reload_dirs=[str(ROOT)], host=args.bind,
                            port=args.port, proxy_headers=False, timeout_graceful_shutdown=GRACEFUL_SHUTDOWN)
    server = uvicorn.Server(config)
    try:
        deferred_reloader()(config, target=server.run, sockets=[config.bind_socket()]).run()
    finally:
        marker.unlink(missing_ok=True)


def make(argv=None):
    args = parse(argv)
    return build(args), args.bind, args.port


def reloaded():
    """The app factory uvicorn calls in each process it starts under --reload."""
    app = build(parse(json.loads(os.environ[ARGV])))
    if live.MARKER_ENV in os.environ:
        threading.Thread(target=keep_marker, args=(app.state.services, os.environ[live.MARKER_ENV]), name="live-marker", daemon=True).start()
    return app


def main(argv=None):
    import uvicorn

    argv = sys.argv[1:] if argv is None else list(argv)
    args = parse(argv)
    if args.dump_openapi:
        with open(args.dump_openapi, "w", encoding="utf-8", newline="\n") as f:
            f.write(schema_json())
        return
    if not args.fake:
        accelerator.check(args.allow_cpu)  # once here, so a refusal is not repeated by every process --reload starts
    if args.reload:
        run_reloading(argv, args)
    else:
        uvicorn.run(build(args), host=args.bind, port=args.port, proxy_headers=False, timeout_graceful_shutdown=GRACEFUL_SHUTDOWN)


if __name__ == "__main__":
    main()
