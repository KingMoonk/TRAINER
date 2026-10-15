"""One entry point for tests, so nobody has to remember the flags.

    python tasks.py test [tests.test_x ...]   ruff + window-free unit tests (all, or just the modules named). Run this locally.
    python tasks.py test:ci                   test + compileall + the window tests (tests/gui_*.py). CI only.
    python tasks.py test:ci:e2e               test:ci + the real training run (tests.e2e_train). CI only.

Each level includes the one before it. The window tests open real Tk and Panda3D windows, so test:ci and test:ci:e2e refuse
to run unless CI is set (GitHub Actions sets it). Use `.venv/bin/python` locally, `.venv/Scripts/python.exe` on Windows.
"""
import os
import subprocess
import sys

PY = sys.executable
VERBOSE = ["-v"] if os.environ.get("CI") else []  # CI logs list every test so a skip is visible


def run(*cmd):
    print("+", " ".join(cmd), flush=True)
    code = subprocess.call(cmd)
    if code:
        sys.exit(code)


def test(modules=()):
    run(PY, "-m", "ruff", "check", ".")
    if modules:
        window = [m for m in modules if "gui_" in m or "e2e" in m]
        if window:
            sys.exit(f"{window} open windows or train for real: they run in CI only (test:ci, test:ci:e2e).")
        run(PY, "-m", "unittest", *VERBOSE, *modules)
    else:
        run(PY, "-m", "unittest", "discover", "-s", "tests", *VERBOSE)  # test_*.py only, which never open a window


def test_ci():
    if not os.environ.get("CI"):
        sys.exit("test:ci opens real windows and runs in CI only. Run `python tasks.py test` locally.")
    run(PY, "-m", "compileall", "-q", ".")  # catches syntax errors in modules no test imports
    test()
    run(PY, "-m", "unittest", "discover", "-s", "tests", "-p", "gui_*.py", *VERBOSE)


def test_ci_e2e():
    test_ci()
    run(PY, "-m", "unittest", "tests.e2e_train", *VERBOSE)  # a real training run on a committed sample, about 1 s of compute


TASKS = {"test": test, "test:ci": test_ci, "test:ci:e2e": test_ci_e2e}

if __name__ == "__main__":
    name, args = (sys.argv[1], sys.argv[2:]) if len(sys.argv) > 1 else ("", [])
    if name not in TASKS or (args and name != "test"):
        sys.exit(__doc__)
    TASKS[name](args) if name == "test" else TASKS[name]()
