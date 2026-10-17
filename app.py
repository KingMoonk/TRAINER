"""python app.py [--fake | --workspace DIR] [--port N] ...: start the Groundwork API on this machine and open the web UI in the default browser.

Takes the flags of `python -m studio_api` (which starts the API alone, for development and for a node without a screen)."""
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser

OPEN_WITHIN = 60.  # seconds to wait for the server to answer before giving up on opening the browser
BUILD_HINT = "The web UI is not built yet: run .venv/bin/python tasks.py web:build, then start the app again."


def url_of(args):
    host = "127.0.0.1" if args.bind in ("0.0.0.0", "::") else args.bind
    return f"http://[{host}]:{args.port}/" if ":" in host else f"http://{host}:{args.port}/"


def open_when_up(url, done, after_open=None):
    """Open the browser at `url` once /node answers; stop waiting when `done` is set or after OPEN_WITHIN seconds."""
    deadline = time.monotonic() + OPEN_WITHIN
    while not done.is_set() and time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url + "node", timeout=2):
                pass
        except (urllib.error.URLError, OSError):
            done.wait(.1)
            continue
        webbrowser.open(url)
        if after_open:
            after_open()
        return


def main(argv=None, after_open=None):
    """Returns the exit code. after_open(server) is called once the browser was asked to open (tests use it to stop the server)."""
    import uvicorn

    from studio_api import app as api_app
    from studio_api.__main__ import build, parse

    args = parse(sys.argv[1:] if argv is None else list(argv), prog="python app.py")
    if args.reload or args.dump_openapi:
        print("--reload and --dump-openapi belong to python -m studio_api.")
        return 2
    if not (api_app.WEB_OUT / "index.html").is_file():
        print(BUILD_HINT)
        return 1
    server = uvicorn.Server(uvicorn.Config(build(args), host=args.bind, port=args.port, proxy_headers=False))
    done = threading.Event()
    opener = threading.Thread(target=open_when_up, args=(url_of(args), done, (lambda: after_open(server)) if after_open else None),
                              name="open-browser", daemon=True)
    opener.start()
    try:
        server.run()
    finally:
        done.set()
    return 0 if server.started else 1


if __name__ == "__main__":
    raise SystemExit(main())
