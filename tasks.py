"""One entry point for tests, so nobody has to remember the flags.

    python tasks.py test [tests.test_x ...]   ruff + window-free unit tests (all, or just the modules named). Run this locally.
    python tasks.py test:ci                   test + compileall + the window tests (tests/gui_*.py). CI only.
    python tasks.py test:ci:e2e               test:ci + the real training run (tests.e2e_train). CI only.
    python tasks.py profile                   record the app with py-spy into profile.svg, timings on. Close the app to write it.
    python tasks.py web:lint                  eslint + tsc on the web UI (web/). Needs pnpm.
    python tasks.py web:test                  the web UI's tests (none yet).
    python tasks.py web:build                 next build: the static export in web/out.

Each level includes the one before it. The window tests open real Tk and Panda3D windows, so test:ci and test:ci:e2e refuse
to run unless CI is set (GitHub Actions sets it). Use `.venv/bin/python` locally, `.venv/Scripts/python.exe` on Windows.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

PY = sys.executable
VERBOSE = ["-v"] if os.environ.get("CI") else []  # CI logs list every test so a skip is visible
APP = Path(__file__).with_name("app.py")
WEB = Path(__file__).with_name("web")
PROFILE = Path(__file__).with_name("profile.svg")


UNIT = {"GROUNDWORK_STRICT_UI": "1", "GROUNDWORK_STALL_LOG": "0"}  # a slow primitive on a bound UI thread raises; no stall watcher in tests
WINDOW = {"GROUNDWORK_STRICT_UI": "0", "GROUNDWORK_STALL_LOG": "0"}  # window tests drive pages on the Tk thread directly: they log it (studio_ui_guard)


def run(*cmd, env=None, cwd=None):
    print("+", " ".join(cmd), flush=True)
    code = subprocess.call(cmd, env={**os.environ, **(env or {})}, cwd=cwd)
    if code:
        sys.exit(code)


def test(modules=()):
    run(PY, "-m", "ruff", "check", ".")
    if modules:
        window = [m for m in modules if "gui_" in m or "e2e" in m]
        if window:
            sys.exit(f"{window} open windows or train for real: they run in CI only (test:ci, test:ci:e2e).")
        run(PY, "-m", "unittest", *VERBOSE, *modules, env=UNIT)
    else:
        run(PY, "-m", "unittest", "discover", "-s", "tests", *VERBOSE, env=UNIT)  # test_*.py only, which never open a window


def test_ci():
    if not os.environ.get("CI"):
        sys.exit("test:ci opens real windows and runs in CI only. Run `python tasks.py test` locally.")
    run(PY, "-m", "compileall", "-q", ".")  # catches syntax errors in modules no test imports
    test()
    # One process per module, as in the app: Panda3D runs in its own process there, never beside Tk. A crash prints every thread's stack.
    for path in sorted(Path(__file__).with_name("tests").glob("gui_*.py")):
        run(PY, "-X", "faulthandler", "-m", "unittest", "discover", "-s", "tests", "-p", path.name, *VERBOSE, env=WINDOW)


def test_ci_e2e():
    test_ci()
    run(PY, "-m", "unittest", "tests.e2e_train", *VERBOSE, env=WINDOW)  # a real training run on a committed sample, about 1 s of compute


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


def web_build():
    pnpm("build")


TASKS = {"test": test, "test:ci": test_ci, "test:ci:e2e": test_ci_e2e, "profile": profile,
         "web:lint": web_lint, "web:test": web_test, "web:build": web_build}

if __name__ == "__main__":
    name, args = (sys.argv[1], sys.argv[2:]) if len(sys.argv) > 1 else ("", [])
    if name not in TASKS or (args and name != "test"):
        sys.exit(__doc__)
    TASKS[name](args) if name == "test" else TASKS[name]()
