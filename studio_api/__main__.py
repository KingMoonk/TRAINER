"""python -m studio_api [--fake] [--bind HOST] [--port N]"""

import argparse

LOCAL = ("127.0.0.1", "localhost", "::1")


def make(argv=None):
    parser = argparse.ArgumentParser(prog="python -m studio_api")
    parser.add_argument("--fake", action="store_true", help="serve FakeServices (no workspace)")
    parser.add_argument("--bind", default="127.0.0.1", help="address to listen on (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    from studio_api.app import create_app

    if args.fake:
        from studio_ia.services import FakeServices

        services = FakeServices()
    else:
        from studio_core import Workspace
        from studio_ia.services import WorkspaceServices

        services = WorkspaceServices(Workspace())
    return create_app(services, open=args.fake or args.bind in LOCAL), args.bind, args.port


def main(argv=None):
    import uvicorn

    app, host, port = make(argv)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
