"""python -m studio_api [--fake] [--bind HOST] [--port N] [--dump-openapi FILE]"""

import argparse
import json

LOCAL = ("127.0.0.1", "localhost", "::1")


def schema_json():
    """The text of web/openapi.json, from the routes over FakeServices. tests/test_api_schema.py compares against it, so one
    function writes it. info.version is pinned: app_version() comes from the checkout's tags, so it differs per checkout and
    would make the file stale on every commit."""
    from studio_api.app import create_app
    from studio_ia.services import FakeServices

    schema = create_app(FakeServices(), open=True).openapi()
    schema["info"]["version"] = "0"
    return json.dumps(schema, indent=2, sort_keys=True) + "\n"


def make(argv=None):
    parser = argparse.ArgumentParser(prog="python -m studio_api")
    parser.add_argument("--fake", action="store_true", help="serve FakeServices (no workspace)")
    parser.add_argument("--bind", default="127.0.0.1", help="address to listen on (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--dump-openapi", metavar="FILE", help="write the OpenAPI schema to FILE and exit (pnpm gen:api reads web/openapi.json)")
    args = parser.parse_args(argv)
    from studio_api.app import create_app

    if args.dump_openapi:
        with open(args.dump_openapi, "w", encoding="utf-8", newline="\n") as f:
            f.write(schema_json())
        raise SystemExit(0)

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
