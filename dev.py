"""Restart the studio whenever a Python file in this folder changes."""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SKIP = {"tests", "__pycache__", "runs", "data", "exports", "artifacts"}


def snapshot():
    files = {}
    for folder, dirs, names in os.walk(ROOT):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in SKIP]
        for name in names:
            if name.endswith(".py"):
                path = Path(folder, name)
                try:
                    files[path] = path.stat().st_mtime_ns
                except FileNotFoundError:
                    pass
    return files


def launch():
    return subprocess.Popen([sys.executable, str(ROOT / "app.py")], cwd=ROOT)


def main():
    seen = snapshot()
    proc = launch()
    try:
        while True:
            time.sleep(1)
            current = snapshot()
            if current != seen:
                seen = current
                print("change detected, restarting", flush=True)
                proc.terminate()
                proc.wait()
                proc = launch()
    except KeyboardInterrupt:
        proc.terminate()


if __name__ == "__main__":
    main()
