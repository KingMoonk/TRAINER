"""One-shot cleanup of the sign-in entries the removed passive training watchdog installed (#486).

Existing installs may still have a macOS LaunchAgent, a Windows ONLOGON task or a Linux autostart file that runs
`python -m studio_passive_watch`, a module that no longer exists (launchd's KeepAlive would respawn it forever). This module is the
single documented exception to "nothing mentions passive": it must name the old entries to remove them. It is idempotent, never
raises and is cheap; the API calls it once at start-up.
"""
from __future__ import annotations

import csv
import io
import logging
import os
import subprocess
import sys
import threading
from pathlib import Path

_log = logging.getLogger(__name__)

AGENT_GLOB = "com.groundwork.passive.*.plist"
DESKTOP_GLOB = "groundwork-passive-*.desktop"
TASK_PREFIX = "GroundworkPassive-"


def _attempt(what, fn):
    try:
        return fn()
    except Exception as exc:  # a stranded entry must never stop the app starting
        _log.warning("sign-in cleanup: could not %s: %s", what, exc)
        return None


def _macos(home, run, removed):
    for path in sorted((home / "Library" / "LaunchAgents").glob(AGENT_GLOB)):
        uid = os.getuid() if hasattr(os, "getuid") else 0
        _attempt(f"unload {path.name}", lambda: run(["launchctl", "bootout", f"gui/{uid}/{path.stem}"], capture_output=True, timeout=20))
        if _attempt(f"delete {path}", lambda: path.unlink(missing_ok=True) or True):
            removed.append(str(path))


def _linux(home, removed):
    for path in sorted((home / ".config" / "autostart").glob(DESKTOP_GLOB)):
        if _attempt(f"delete {path}", lambda: path.unlink(missing_ok=True) or True):
            removed.append(str(path))


def _windows(run, removed):
    listing = _attempt("list scheduled tasks", lambda: run(["schtasks", "/Query", "/FO", "CSV", "/NH"], capture_output=True, text=True, timeout=30))
    for row in csv.reader(io.StringIO(listing.stdout if listing else "")):
        name = row[0].lstrip("\\") if row else ""
        if name.startswith(TASK_PREFIX):
            done = _attempt(f"delete task {name}", lambda: run(["schtasks", "/Delete", "/TN", name, "/F"], capture_output=True, timeout=30).returncode == 0)
            if done:
                removed.append(name)


def remove_stale_entries(root=None, *, home=None, run=subprocess.run, platform=sys.platform):
    """Remove the old sign-in entries that exist; returns what it removed. Safe to call every start: it finds nothing the second time."""
    removed = []
    home = Path(home or Path.home())
    if platform == "darwin":
        _macos(home, run, removed)
    elif platform.startswith("linux"):
        _linux(home, removed)
    elif platform == "win32":
        _windows(run, removed)
        if root is not None:
            for path in (Path(root) / "data").glob("passive-watchdog-*.cmd"):
                _attempt(f"delete {path}", lambda: path.unlink(missing_ok=True))
    if removed:
        _log.info("sign-in cleanup: removed %s", ", ".join(removed))
    return removed


def start(root=None):
    """Run remove_stale_entries on a daemon thread, so start-up never waits for launchctl or schtasks."""
    threading.Thread(target=lambda: _attempt("clean up", lambda: remove_stale_entries(root)), name="legacy-signin-cleanup", daemon=True).start()
