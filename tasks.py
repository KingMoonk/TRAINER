"""One entry point for tests, so nobody has to remember the flags.

    python tasks.py test [tests.test_x ...]   ruff + unit tests (all, or just the modules named). Run this locally.
    python tasks.py test:ci                   test + compileall. Every pull request runs it, on Linux.
    python tasks.py test:ci:e2e               test:ci + the real training run (tests.e2e_train). Nightly, on macOS and Windows.
    python tasks.py profile                   record the app with py-spy into profile.svg, timings on. Close the app to write it.
    python tasks.py web:lint                  eslint + tsc on the web UI (web/). Needs pnpm.
    python tasks.py web:test                  Vitest: unit, component and contract tests (web/src). No browser, safe to run locally.
    python tasks.py web:e2e                   Playwright e2e flows against web/out. CI only: run web:build first, and `pnpm exec playwright install chromium` once.
    python tasks.py web:fixtures              re-record web/src/contract/fixtures from FakeServices (tests/web_fixtures.py); then run web:test.
    python tasks.py web:build                 next build: the static export in web/out.
    python tasks.py dev                       the API (--reload) and the Next dev server on this machine, so training keeps the GPU (MPS on a Mac). docs/dev-compose.md.

Each level includes the one before it. Use `.venv/bin/python` locally, `.venv/Scripts/python.exe` on Windows.
"""
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

PY = sys.executable
VERBOSE = ["-v"] if os.environ.get("CI") else []  # CI logs list every test so a skip is visible
APP = Path(__file__).with_name("app.py")
WEB = Path(__file__).with_name("web")
PROFILE = Path(__file__).with_name("profile.svg")


UNIT = {"GROUNDWORK_STRICT_UI": "1", "GROUNDWORK_STALL_LOG": "0"}  # a slow primitive on a bound UI thread raises; no stall watcher in tests


def run(*cmd, env=None, cwd=None):
    print("+", " ".join(cmd), flush=True)
    code = subprocess.call(cmd, env={**os.environ, **(env or {})}, cwd=cwd)
    if code:
        # the CI step's own exit code does not tell a failed test (1) from a native crash (0xc0000005, say); this does
        print(f"+ exited with {code} ({code & 0xFFFFFFFF:#x})", flush=True)
        sys.exit(code)


def test(modules=()):
    run(PY, "-m", "ruff", "check", ".")
    if modules:
        e2e = [m for m in modules if "e2e" in m]
        if e2e:
            sys.exit(f"{e2e} train for real: they run in test:ci:e2e only.")
        run(PY, "-X", "faulthandler", "-m", "unittest", *VERBOSE, *modules, env=UNIT)
    else:
        # test_*.py only. A native crash prints every thread's Python stack instead of ending silently.
        run(PY, "-X", "faulthandler", "-m", "unittest", "discover", "-s", "tests", *VERBOSE, env=UNIT)


def test_ci():
    run(PY, "-m", "compileall", "-q", ".")  # catches syntax errors in modules no test imports
    # runners differ in CPU; a crash in a CPU kernel (bf16, say) shows up on some of them only
    run(PY, "-c", "import platform, torch; print('+ cpu:', platform.processor(), '| torch', torch.__version__, torch.backends.cpu.get_cpu_capability())")
    test()


def test_ci_e2e():
    test_ci()
    run(PY, "-m", "unittest", "tests.e2e_train", *VERBOSE, env=UNIT)  # a real training run on a committed sample, about 1 s of compute


def profile():
    """Launch the app under py-spy rather than attaching to it: launching needs no sudo on macOS. The timed() logs go to
    the terminal too, and profile.svg is written when the app exits. py-spy comes from this environment's Scripts or bin folder."""
    spy = shutil.which("py-spy", path=os.pathsep.join([os.path.dirname(PY), os.environ.get("PATH", "")]))
    if not spy:
        sys.exit(f'py-spy is not installed. Install it into this environment with:\n  "{PY}" -m pip install py-spy\n'
                 "It is a dev tool, not a runtime dependency.")
    os.environ["GROUNDWORK_TIMINGS"] = "1"
    run(spy, "record", "-o", str(PROFILE), "--subprocesses", "--", PY, str(APP))
    print(f"wrote {PROFILE}; open it in a browser", flush=True)


def pnpm(*args):
    """Run a pnpm script in web/. pnpm.cmd on Windows, since subprocess does not search for .cmd without a shell."""
    exe = shutil.which("pnpm")
    if not exe:
        sys.exit("pnpm is not installed. See web/README.md.")
    run(exe, *args, cwd=WEB)


def web_lint():
    pnpm("lint")
    pnpm("typecheck")


def web_test():
    pnpm("test")


def web_e2e():
    pnpm("test:e2e")


def web_fixtures():
    run(PY, "-m", "tests.web_fixtures")


def web_build():
    pnpm("build")


API_PORT, WEB_PORT = 8787, 3000
SERVE = (f"tailscale serve --bg http://127.0.0.1:{WEB_PORT}", f"tailscale serve --bg --set-path=/api/ http://127.0.0.1:{API_PORT}")  # `/` to the page, `/api/` to the API


def dev_commands(pnpm_exe="pnpm"):
    """(API command, web command, web env) for `dev`. Both bind 127.0.0.1 whatever happens: the API trusts the
    Tailscale-User-Login header of a loopback peer (studio_api/auth.py), so only `tailscale serve`, which runs on this machine
    and writes that header itself, may reach it. `dev` takes no arguments, so neither bind can be widened from here."""
    api = [PY, "-m", "studio_api", "--reload", "--bind", "127.0.0.1", "--port", str(API_PORT)]
    web = [pnpm_exe, "dev", "--hostname", "127.0.0.1", "--port", str(WEB_PORT)]
    return api, web, {"NEXT_PUBLIC_API_URL": "/api"}  # same origin: `tailscale serve` sends /api/* to the API, prefix stripped


def node_device(fetch, sleep=time.sleep, clock=time.monotonic, timeout=120., every=1.):
    """The `device` the API reports, polled until it answers; None after `timeout` seconds. `fetch()` returns the /node payload
    and raises while the API is not up."""
    end = clock() + timeout
    while True:
        try:
            return str(fetch().get("device") or "cpu")
        except Exception:
            if clock() >= end:
                return None
            sleep(every)


def fetch_node(port=API_PORT):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/node", timeout=2) as answer:
        return json.load(answer)


def announce(fetch=fetch_node, say=print, **poll):
    """Print the device GET /node reports. An API that finds a GPU unused by torch refuses to start (studio_api/accelerator.py)."""
    reported = node_device(fetch, **poll)
    say(f"GET /node reports device {reported}" if reported else "WARNING: the API did not answer GET /node; check its output above.")


def port_busy(port):
    with socket.socket() as probe:
        probe.settimeout(.5)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def dev():
    """The live dev loop on this machine: the API with --reload and the Next dev server, both on 127.0.0.1. Run on the host, not
    in a container, so torch sees Metal (a Docker VM on a Mac has none) or CUDA. `tailscale serve` puts them on the tailnet."""
    for name, port in (("API", API_PORT), ("web", WEB_PORT)):
        if port_busy(port):
            sys.exit(f"Port {port} ({name}) is in use. Stop what listens there first; a second API on the same data/ and runs/ is unsafe.")
    exe = shutil.which("pnpm")
    if not exe:
        sys.exit("pnpm is not installed. See web/README.md.")
    api, web, env = dev_commands(exe)
    if not (WEB / "node_modules").is_dir():
        run(exe, "install", "--frozen-lockfile", cwd=WEB)
    print("Tailscale, once (this replaces what `tailscale serve` shows at / now; `tailscale serve reset` takes it down):", flush=True)
    for line in SERVE:
        print("  " + line, flush=True)
    procs = [subprocess.Popen(api), subprocess.Popen(web, cwd=WEB, env={**os.environ, **env})]
    threading.Thread(target=announce, name="device-notice", daemon=True).start()
    code = 0
    try:
        while all(p.poll() is None for p in procs):
            time.sleep(1)
        code = next(p.returncode for p in procs if p.returncode is not None) or 1  # one stopped by itself: say so
        print(f"+ a dev process exited with {code}; stopping the other", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            p.wait()
    sys.exit(code)


TASKS = {"test": test, "test:ci": test_ci, "test:ci:e2e": test_ci_e2e, "profile": profile,
         "web:lint": web_lint, "web:test": web_test, "web:e2e": web_e2e, "web:fixtures": web_fixtures,
         "web:build": web_build, "dev": dev}

if __name__ == "__main__":
    name, args = (sys.argv[1], sys.argv[2:]) if len(sys.argv) > 1 else ("", [])
    if name not in TASKS or (args and name != "test"):
        sys.exit(__doc__)
    TASKS[name](args) if name == "test" else TASKS[name]()
