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
    python tasks.py setup                     once per machine: .venv, the right torch (MPS on a Mac, CUDA with nvidia-smi), requirements, pnpm install; prints the device. Any Python 3.12+.
    python tasks.py dev                       the live loop: the API (restarts on a .py change) and `pnpm dev` on 127.0.0.1, one terminal, Ctrl+C stops both; points `tailscale serve` at them. docs/node.md, "Developing".
    python tasks.py run                       the built app (app.py on 127.0.0.1:8787); points `tailscale serve` at it.

Each level includes the one before it. Use `.venv/bin/python` locally, `.venv/Scripts/python.exe` on Windows.
"""
import json
import os
import shutil
import signal
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


VENV = Path(__file__).with_name(".venv")
VENV_PY = VENV / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
TORCH = "torch>=2.14.1,<3"  # as in requirements-training.txt
CUDA_INDEX = "https://download.pytorch.org/whl/cu126"  # requirements-gpu.txt
MIN_PYTHON = (3, 12)
API_PORT, WEB_PORT = 8787, 3000
DEV_SERVE = {"/": f"http://127.0.0.1:{WEB_PORT}", "/api/": f"http://127.0.0.1:{API_PORT}"}  # `dev`: the page, and the API under /api/
APP_SERVE = {"/": f"http://127.0.0.1:{API_PORT}"}  # `run`: the API, which serves the built page itself


def find_python(current=sys.executable, version=sys.version_info, which=shutil.which, ask=None):
    """An interpreter of Python 3.12 or newer (the pinned NumPy and pandas need it): the one running this, else the first on PATH."""
    if tuple(version[:2]) >= MIN_PYTHON:
        return current
    ask = ask or (lambda exe: subprocess.run([exe, "-c", "import sys; print(sys.version_info >= (3, 12))"], capture_output=True,
                                              text=True, check=False).stdout.strip() == "True")
    for name in ("python3.14", "python3.13", "python3.12", "python3", "python"):
        found = which(name)
        if found and ask(found):
            return found
    sys.exit("Python 3.12 or newer is needed. On macOS install one from python.org or `brew install python@3.13`, on Windows from python.org.")


def torch_install(nvidia):
    """The pip arguments that install torch for this machine: the CUDA 12.6 build when nvidia-smi lists a GPU (Windows, Linux), the
    default wheel otherwise (on macOS it includes Metal/MPS)."""
    return ["install", TORCH, "--index-url", CUDA_INDEX] if nvidia and sys.platform != "darwin" else ["install", TORCH]


def setup():
    """One command per machine: .venv, the right torch, the requirements, pnpm install; then prints the device torch sees and fails
    when the machine has a GPU torch cannot use (studio_api/accelerator.py). Run it with any Python 3.12+; safe to run again."""
    from studio_api.accelerator import machine_accelerator

    if not shutil.which("pnpm"):
        sys.exit("pnpm is not installed. Install Node 24, then `npm install -g pnpm`.")
    if not VENV_PY.exists():
        run(find_python(), "-m", "venv", str(VENV))
    pip = [str(VENV_PY), "-m", "pip"]
    run(*pip, "install", "--upgrade", "pip")
    run(*pip, *torch_install(machine_accelerator() == "cuda"))
    run(*pip, "install", "-r", "requirements-dev.txt")
    pnpm("install", "--frozen-lockfile")
    run(str(VENV_PY), "-c", "from studio_api import accelerator; print('+', accelerator.check())")


def reload_by_uvicorn():
    """uvicorn's --reload everywhere but Windows: there it restarts its worker with CTRL_C_EVENT, which Windows delivers to every
    process on the console, so one edited .py file would stop `dev` and Next too. `dev` watches the sources itself instead."""
    return sys.platform != "win32"


def dev_commands(pnpm_exe="pnpm", reload=None):
    """(API command, web command, web env) for `dev`. Both bind 127.0.0.1 whatever happens: the API trusts the
    Tailscale-User-Login header of a loopback peer (studio_api/auth.py), so only `tailscale serve`, which runs on this machine
    and writes that header itself, may reach it. `dev` takes no arguments, so neither bind can be widened from here."""
    api = [PY, "-m", "studio_api", *(["--reload"] if (reload_by_uvicorn() if reload is None else reload) else []),
           "--bind", "127.0.0.1", "--port", str(API_PORT)]
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


def tailnet_url(names=None):
    """The https URL `tailscale serve` gives this machine, from `tailscale status` (read-only); "" when Tailscale is not running."""
    if names is None:
        from studio_api.hosts import tailscale_names
        names = tailscale_names()
    return next((f"https://{n}/" for n in names if n.endswith(".ts.net")), "")


def serve_handlers(status):
    """{path: proxy target} of the HTTPS :443 handlers in `tailscale serve status --json`."""
    found = {}
    for host, conf in ((status or {}).get("Web") or {}).items():
        if host.endswith(":443"):
            found.update({path: (h or {}).get("Proxy", "") for path, h in ((conf or {}).get("Handlers") or {}).items()})
    return found


def serve_commands(have, want):
    """The `tailscale serve` arguments that turn the handlers `have` into `want`: none when they already match."""
    off = [["serve", "--https=443", f"--set-path={path}", "off"] for path in have if path not in want]
    on = [["serve", "--bg", *([] if path == "/" else [f"--set-path={path}"]), target]
          for path, target in want.items() if have.get(path) != target]
    return off + on


def ensure_serve(want, binary, url, run=subprocess.run, say=print):
    """Point this machine's tailnet HTTPS name at `want` ({path: loopback URL}) with `tailscale serve`, changing only the paths that
    differ, so running it again changes nothing. Does nothing when Tailscale is missing or not running (`url` empty)."""
    if not binary or not url:
        say("Tailnet: Tailscale is not running here; the page is on 127.0.0.1 only.")
        return

    def handlers():
        done = run([binary, "serve", "status", "--json"], capture_output=True, text=True, timeout=10, check=False)
        try:
            return serve_handlers(json.loads(done.stdout or "{}")) if done.returncode == 0 else {}
        except ValueError:
            return {}

    commands = serve_commands(handlers(), want)
    for args in commands:
        say("+ tailscale " + " ".join(args))
        run([binary, *args], capture_output=True, text=True, timeout=30, check=False)
    if commands and handlers() != want:
        say("WARNING: `tailscale serve status` does not show " + ", ".join(f"{p} -> {t}" for p, t in want.items())
            + "; check it, or `tailscale serve reset` and start again.")
    say(f"Tailnet: {url}" + ("" if commands else " (serve already pointed here)"))


def tailnet(want):
    from studio_api.hosts import tailscale_binary
    ensure_serve(want, tailscale_binary(), tailnet_url())


def port_busy(port):
    with socket.socket() as probe:
        probe.settimeout(.5)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def spawn(cmd, **kwargs):
    """Start a dev process in its own process group (POSIX) or tree (Windows), so `stop` ends pnpm's node child as well."""
    flags = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if sys.platform == "win32" else {"start_new_session": True}
    return subprocess.Popen(cmd, **flags, **kwargs)


def stop(proc):
    if proc.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.call(["taskkill", "/PID", str(proc.pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        os.killpg(proc.pid, signal.SIGTERM)


WATCH_SKIP = {"web", "data", "runs", "node_modules", "__pycache__"}  # besides every directory starting with "." (.venv, .git, .claude/worktrees)


def scan_sources(root=Path(__file__).parent, walk=os.walk):
    """{path: mtime} of the *.py files under `root` that uvicorn --reload would have watched: not .venv, web, data, runs, .git or worktrees."""
    found = {}
    for here, dirs, files in walk(root):
        dirs[:] = [d for d in dirs if d not in WATCH_SKIP and not d.startswith(".")]
        for name in files:
            if name.endswith(".py"):
                path = os.path.join(here, name)
                try:
                    found[path] = os.stat(path).st_mtime_ns
                except OSError:
                    pass  # deleted between the listing and the stat
    return found


class SourceWatcher:
    """True once when the Python sources changed and then stopped changing (a `git pull` writes many files over a moment).
    `scan()` returns {path: mtime}; call `poll()` every second or so."""

    def __init__(self, scan=scan_sources):
        self.scan, self.seen, self.pending = scan, scan(), None

    def poll(self):
        now = self.scan()
        if now == self.seen:
            self.pending = None
            return False
        if now != self.pending:
            self.pending = now
            return False
        self.seen, self.pending = now, None
        return True


def training_busy(folder=None, alive=None, command_line=None):
    """True while a queue job's worker process runs: restarting the API now would kill it (on Windows `stop` ends the whole tree) and
    cost the steps since its last checkpoint. The workers are the specs in <workspace>/data/workers, each naming a live studio_worker."""
    import studio_passive_watch as procs
    import studio_worker
    if folder is None:
        from studio_core import ROOT
        folder = Path(ROOT) / "data" / "workers"
    alive, command_line = alive or procs.alive, command_line or procs.command_line
    for spec, pid in studio_worker.strays(folder):
        if pid and alive(pid):
            line = command_line(pid) or ""
            if "studio_worker" in line and str(spec) in line:
                return True
    return False


def dev_step(api_alive, web_alive, changed, watching, busy=False):
    """What the dev loop does next: "stop" (the web server ended, or the API did and nothing watches), "restart" (sources changed
    and the API is ours to restart), "defer" (they changed but a training job runs under the API: wait for it to end),
    "wait" (the API ended and stays down until the next change) or "keep". `changed` stays true until acted on."""
    if not web_alive or (not api_alive and not watching):
        return "stop"
    if watching and changed:
        return "defer" if busy and api_alive else "restart"
    return "wait" if watching and not api_alive else "keep"


def dev():
    """The live dev loop on this machine: the API (restarting on a .py change) and the Next dev server, both on 127.0.0.1, with both outputs in
    this terminal. Run it with the .venv interpreter (`python tasks.py setup` makes it), on the host so torch sees the GPU (Metal, CUDA).
    When Tailscale is running it points `tailscale serve` at them first (`/` the page, `/api/` the API); `run` points it back."""
    for name, port in (("API", API_PORT), ("web", WEB_PORT)):
        if port_busy(port):
            sys.exit(f"Port {port} ({name}) is in use. Stop what listens there first; a second API on the same data/ and runs/ is unsafe.")
    exe = shutil.which("pnpm")
    if not exe:
        sys.exit("pnpm is not installed. See web/README.md.")
    api, web, env = dev_commands(exe)
    watcher = None if reload_by_uvicorn() else SourceWatcher()
    if not (WEB / "node_modules").is_dir():
        run(exe, "install", "--frozen-lockfile", cwd=WEB)
    tailnet(DEV_SERVE)
    procs = [spawn(api), spawn(web, cwd=WEB, env={**os.environ, **env})]
    threading.Thread(target=announce, name="device-notice", daemon=True).start()
    code, told, pending, deferred = 0, False, False, False
    try:
        while True:
            pending = pending or bool(watcher and watcher.poll())
            api_up = procs[0].poll() is None
            step = dev_step(api_up, procs[1].poll() is None, pending, watcher is not None, bool(pending and api_up and training_busy()))
            if step == "stop":
                code = next(p.returncode for p in procs if p.returncode is not None) or 1  # one stopped by itself: say so
                print(f"+ a dev process exited with {code}; stopping the other", flush=True)
                break
            if step == "defer" and not deferred:
                deferred = True
                print("+ a .py file changed, but a training job is running: the API restarts when it ends (Ctrl+C stops everything)", flush=True)
            if step == "restart":
                print("+ a .py file changed; restarting the API", flush=True)
                stop(procs[0])
                procs[0].wait()
                procs[0], told, pending, deferred = spawn(api), False, False, False
                threading.Thread(target=announce, name="device-notice", daemon=True).start()
            elif step == "wait" and not told:
                told = True
                print(f"+ the API exited with {procs[0].returncode}; it starts again on the next .py change", flush=True)
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            stop(p)
        for p in procs:
            p.wait()
    sys.exit(code)


def run_app():
    """The built app (app.py: the API serving web/out), with `tailscale serve` pointed at it first when Tailscale is running."""
    if port_busy(API_PORT):
        sys.exit(f"Port {API_PORT} is in use. Stop what listens there first (`make dev`, another `make run`).")
    tailnet(APP_SERVE)
    try:
        run(PY, str(APP))
    except KeyboardInterrupt:
        pass


TASKS = {"test": test, "test:ci": test_ci, "test:ci:e2e": test_ci_e2e, "profile": profile,
         "web:lint": web_lint, "web:test": web_test, "web:e2e": web_e2e, "web:fixtures": web_fixtures,
         "web:build": web_build, "setup": setup, "dev": dev, "run": run_app}

if __name__ == "__main__":
    name, args = (sys.argv[1], sys.argv[2:]) if len(sys.argv) > 1 else ("", [])
    if name not in TASKS or (args and name != "test"):
        sys.exit(__doc__)
    TASKS[name](args) if name == "test" else TASKS[name]()
