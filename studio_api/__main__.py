"""python -m studio_api [--fake] [--bind HOST] [--port N] [--reload]"""

import argparse
import json
import os
import sys
from pathlib import Path

import studio_cluster

LOCAL = ("127.0.0.1", "localhost", "::1")
ARGV = "STUDIO_API_ARGV"  # how --reload hands the command line to the process uvicorn restarts
ROOT = Path(__file__).resolve().parent.parent


def parse(argv=None):
    parser = argparse.ArgumentParser(prog="python -m studio_api")
    parser.add_argument("--fake", action="store_true", help="serve FakeServices (no workspace)")
    parser.add_argument("--bind", default="127.0.0.1", help="address to listen on (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=studio_cluster.DEFAULT_PORT, help="port (default: the port other nodes expect)")
    parser.add_argument("--reload", action="store_true", help="restart when a .py file changes (development)")
    return parser.parse_args(argv)


def build(args):
    from studio_api.app import create_app

    if args.fake:
        from studio_ia.services import FakeServices

        services = FakeServices()
    else:
        from studio_core import Workspace
        from studio_ia.services import WorkspaceServices

        services = WorkspaceServices(Workspace())
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
    if args.reload:
        os.environ[ARGV] = json.dumps(argv)
        uvicorn.run("studio_api.__main__:reloaded", factory=True, reload=True, reload_dirs=[str(ROOT)],
                    host=args.bind, port=args.port)
    else:
        uvicorn.run(build(args), host=args.bind, port=args.port)


if __name__ == "__main__":
    main()
