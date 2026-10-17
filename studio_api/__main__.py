"""python -m studio_api [--fake | --workspace DIR] [--bind HOST] [--port N] [--allow-origin ORIGIN] [--reload] [--dump-openapi FILE]"""

import argparse
import ipaddress
import json
import math
import os
import re
import sys
from pathlib import Path

import studio_cluster

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
    from studio_ia.services import FakeServices

    schema = create_app(FakeServices(), open=True).openapi()
    schema["info"]["version"] = "0"
    return json.dumps(schema, indent=2, sort_keys=True) + "\n"


LABEL = r"[A-Za-z0-9]([A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
ORIGIN = re.compile(rf"(?P<scheme>[A-Za-z]+)://(?P<host>\[[0-9A-Fa-f:.]+\]|{LABEL}(\.{LABEL})*)(:(?P<port>[0-9]{{1,5}}))?")


def origin(value):
    """`https://app.vercel.app` from a typed origin: a scheme, a host and an optional port, nothing else (no path, no `*`, no
    space). The scheme and host are lowercased; a port outside 1-65535 is refused, not rewritten to the default."""
    found = ORIGIN.fullmatch(value.strip().rstrip("/"))
    scheme = found and found["scheme"].lower()
    port = found and int(found["port"]) if found and found["port"] else None
    try:
        if found and found["host"].startswith("["):
            ipaddress.IPv6Address(found["host"][1:-1])
    except ValueError:
        found = None
    if not found or scheme not in DEFAULT_PORTS or (found["port"] and not 0 < port < 65536):
        raise argparse.ArgumentTypeError(f"{value!r} is not a web origin like https://app.example.com (scheme, host, optional port 1-65535).")
    return f"{scheme}://{found['host'].lower()}" + (f":{port}" if port and port != DEFAULT_PORTS[scheme] else "")


def parse(argv=None):
    parser = argparse.ArgumentParser(prog="python -m studio_api")
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

    if args.fake:
        from studio_ia.services import FakeServices

        services = FakeServices(step_seconds=args.fake_step_seconds)
    else:
        from studio_core import Workspace
        from studio_ia.services import WorkspaceServices

        workspace = Workspace(args.workspace) if args.workspace is not None else Workspace()
        services = WorkspaceServices(workspace)
        tags, users = studio_cluster.allowed_callers(workspace.state)
        return create_app(services, open=args.bind in LOCAL, origins=args.allow_origin, allow_tags=tags, allow_users=users,
                          token=studio_cluster.funnel_token(workspace.state))
    return create_app(services, open=True, origins=args.allow_origin)


def make(argv=None):
    args = parse(argv)
    return build(args), args.bind, args.port


def reloaded():
    """The app factory uvicorn calls in each process it starts under --reload."""
    return build(parse(json.loads(os.environ[ARGV])))


def main(argv=None):
    import uvicorn

    argv = sys.argv[1:] if argv is None else list(argv)
    args = parse(argv)
    if args.dump_openapi:
        with open(args.dump_openapi, "w", encoding="utf-8", newline="\n") as f:
            f.write(schema_json())
        return
    if args.reload:
        os.environ[ARGV] = json.dumps(argv)
        uvicorn.run("studio_api.__main__:reloaded", factory=True, reload=True, reload_dirs=[str(ROOT)],
                    host=args.bind, port=args.port, proxy_headers=False)
    else:
        uvicorn.run(build(args), host=args.bind, port=args.port, proxy_headers=False)


if __name__ == "__main__":
    main()
