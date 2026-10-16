"""python -m studio_api [--fake | --workspace DIR] [--bind HOST] [--port N] [--reload] [--dump-openapi FILE]"""

import argparse
import json
import os
import sys
from pathlib import Path

import studio_cluster

LOCAL = ("127.0.0.1", "localhost", "::1")
ARGV = "STUDIO_API_ARGV"  # how --reload hands the command line to the process uvicorn restarts
ROOT = Path(__file__).resolve().parent.parent


def schema_json():
    """The text of web/openapi.json, from the routes over FakeServices. tests/test_api_schema.py compares against it, so one
    function writes it. info.version is pinned: app_version() comes from the checkout's tags, so it differs per checkout and
    would make the file stale on every commit."""
    from studio_api.app import create_app
    from studio_ia.services import FakeServices

    schema = create_app(FakeServices(), open=True).openapi()
    schema["info"]["version"] = "0"
    return json.dumps(schema, indent=2, sort_keys=True) + "\n"


def parse(argv=None):
    parser = argparse.ArgumentParser(prog="python -m studio_api")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--fake", action="store_true", help="serve FakeServices (no workspace)")
    source.add_argument("--workspace", type=Path, help="existing folder containing data/ and runs/ (useful from a worktree)")
    parser.add_argument("--bind", default="127.0.0.1", help="address to listen on (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=studio_cluster.DEFAULT_PORT, help="port (default: the port other nodes expect)")
    parser.add_argument("--reload", action="store_true", help="restart when a .py file changes (development)")
    parser.add_argument("--dump-openapi", metavar="FILE", help="write the OpenAPI schema to FILE and exit (pnpm gen:api reads web/openapi.json)")
    args = parser.parse_args(argv)
    if args.workspace is not None:
        args.workspace = args.workspace.expanduser().resolve()
        if not args.workspace.is_dir():
            parser.error(f"Workspace folder does not exist: {args.workspace}")
    return args


def build(args):
    from studio_api.app import create_app

    if args.fake:
        from studio_ia.services import FakeServices

        services = FakeServices()
    else:
        from studio_core import Workspace
        from studio_ia.services import WorkspaceServices

        workspace = Workspace(args.workspace) if args.workspace is not None else Workspace()
        services = WorkspaceServices(workspace)
    return create_app(services, open=args.fake or args.bind in LOCAL)


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
                    host=args.bind, port=args.port)
    else:
        uvicorn.run(build(args), host=args.bind, port=args.port)


if __name__ == "__main__":
    main()
