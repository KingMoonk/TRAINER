import functools
import subprocess
from pathlib import Path


@functools.lru_cache(maxsize=1)
def app_version():
    try:
        out = subprocess.run(["git", "describe", "--tags", "--always"], cwd=Path(__file__).resolve().parent.parent,
                             capture_output=True, text=True, timeout=5, check=False)
        return out.stdout.strip() or "dev"
    except (OSError, subprocess.SubprocessError):
        return "dev"
