"""python app.py [--fake | --workspace DIR] [--port N] ...: start the Groundwork API on this machine and open the web UI in the default browser.

Takes the flags of `python -m studio_api` (which starts the API alone, for development and for a node without a screen).

python app.py --smoke REPORT: the release check on the frozen build. Imports the modules PyInstaller cannot see, serves FakeServices on a free port,
waits for /node to answer, stops, and writes what it found to REPORT; the exit code is 0 only when everything came up, the web build included. No browser opens."""
import importlib
import io
import logging
import pkgutil
import socket
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
import webbrowser

OPEN_WITHIN = 60.  # seconds to wait for the server to answer before giving up on opening the browser
BUILD_HINT = "The web UI is not built yet: run .venv/bin/python tasks.py web:build, then start the app again."


HIDDEN = ("studio_brain3d", "studio_worker")  # the --hidden-import list in .github/workflows/release.yml: launched by name, so the import scan misses them


def split_smoke(argv):
    """(report path or None, the other arguments): --smoke REPORT or --smoke=REPORT is app.py's own, so it is taken out before studio_api parses the rest."""
    argv = list(argv)
    for at, arg in enumerate(argv):
        if arg == "--smoke" or arg.startswith("--smoke="):
            break
    else:
        return None, argv
    if arg == "--smoke":
        report, used = (argv[at + 1] if at + 1 < len(argv) else ""), 2
    else:
        report, used = arg[len("--smoke="):], 1
    if not report or report.startswith("--"):
        raise SystemExit("--smoke needs a report file")
    return report, argv[:at] + argv[at + used:]


def missing_modules():
    """Lines naming each module that fails to import in this build: the hidden imports and every studio_modules submodule."""
    import studio_modules

    names = list(HIDDEN) + [m.name for m in pkgutil.walk_packages(studio_modules.__path__, "studio_modules.")]
    lines = []
    for name in names:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - any failure to load is the finding
            lines.append(f"import {name}: {type(exc).__name__}: {exc}")
    return lines


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def smoke(argv, report):
    """Returns the exit code: 0 when every module imported, the web build is there and /node answered. The report holds one line per
    finding, then OK or FAIL; any exception (a bind failure included) is written to it too."""
    try:
        lines = smoke_findings(argv)
    except BaseException:  # noqa: BLE001 - SystemExit from a bind failure must reach the report as well
        lines = [traceback.format_exc().rstrip()]
    failed = bool(lines)
    text = "\n".join(lines + ["FAIL" if failed else "OK"]) + "\n"
    with open(report, "w", encoding="utf-8") as f:  # a windowed build has no console, so the report file is the only output
        f.write(text)
    if sys.stdout:
        print(text, end="")
    return 1 if failed else 0


def smoke_findings(argv):
    import uvicorn

    from studio_api import app as api_app
    from studio_api.__main__ import build, parse

    lines = missing_modules()
    if not (api_app.WEB_OUT / "index.html").is_file():
        lines.append(f"web build missing: {api_app.WEB_OUT / 'index.html'}")
    if lines:
        return lines
    args = parse(argv + ["--fake", "--port", str(free_port())], prog="python app.py --smoke")
    # log_config=None: a windowed Windows build has no stdout or stderr, and uvicorn's default formatter calls isatty() on them
    server = uvicorn.Server(uvicorn.Config(build(args), host=args.bind, port=args.port, proxy_headers=False, log_config=None))
    done, answered, last_error = threading.Event(), [], []
    log = io.StringIO()  # uvicorn's and the app's own log lines, so a failed start says why
    handler = logging.StreamHandler(log)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(name).addHandler(handler)
        logging.getLogger(name).setLevel(logging.INFO)

    def stop():
        answered.append(True)
        server.should_exit = True

    def watch():
        open_when_up(url_of(args), done, stop, browser=False, on_error=lambda exc: last_error.append(exc))
        server.should_exit = True  # also when /node never answered, so the run ends

    poller = threading.Thread(target=watch, name="smoke", daemon=True)
    poller.start()
    try:
        server.run()
    finally:
        done.set()
    if not (answered and server.started):
        why = last_error[-1] if last_error else None
        detail = f"last poll error: {type(why).__name__}: {why}" if why else "no poll error"
        if isinstance(why, urllib.error.HTTPError):
            detail += f" body: {why.read(500)!r}"
        return [f"/node did not answer on port {args.port} within {OPEN_WITHIN:.0f}s",
                f"server.started={server.started} poller alive={poller.is_alive()} {detail}",
                "log:\n" + log.getvalue()[-3000:]]
    return []


def url_of(args):
    host = "127.0.0.1" if args.bind in ("0.0.0.0", "::") else args.bind
    return f"http://[{host}]:{args.port}/" if ":" in host else f"http://{host}:{args.port}/"


def open_when_up(url, done, after_open=None, browser=True, on_error=None):
    """Open the browser at `url` once /node answers (browser=False: only wait); stop waiting when `done` is set or after OPEN_WITHIN seconds."""
    deadline = time.monotonic() + OPEN_WITHIN
    while not done.is_set() and time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url + "node", timeout=2):
                pass
        except (urllib.error.URLError, OSError) as exc:
            if on_error:
                on_error(exc)
            done.wait(.1)
            continue
        if browser:
            webbrowser.open(url)
        if after_open:
            after_open()
        return


def main(argv=None, after_open=None):
    """Returns the exit code. after_open(server) is called once the browser was asked to open (tests use it to stop the server)."""
    import uvicorn

    from studio_api import app as api_app
    from studio_api.__main__ import build, parse

    report, argv = split_smoke(sys.argv[1:] if argv is None else argv)
    if report is not None:
        return smoke(argv, report)
    args = parse(argv, prog="python app.py")
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
