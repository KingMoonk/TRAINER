"""When the collector may run: enabled and not paused, on mains power, on an unmetered network, and the machine idle.

Battery uses passive training's probe (studio_passive_watch.on_battery). Passive training has no idle test, so idle here means:
no queued job running, and CPU use below the person's limit. Network cost is read from the OS where it can be (Windows, NetworkManager);
elsewhere the person confirms the connection is unmetered. Every probe is injectable for tests.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import studio_passive_watch as watch


def _windows_times():
    import ctypes
    from ctypes import wintypes
    idle, kernel, user = wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME()
    if not ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
        return None
    value = lambda t: t.dwHighDateTime << 32 | t.dwLowDateTime  # noqa: E731
    return value(idle), value(kernel) + value(user)   # kernel time includes idle time


class CpuMeter:
    """CPU use since the previous call, in percent. Windows: system times; elsewhere the one-minute load average per core."""

    def __init__(self):
        self.last = None

    def __call__(self):
        try:
            if sys.platform == "win32":
                now = _windows_times()
                before, self.last = self.last, now
                if now is None or before is None:
                    return 0.0
                total = now[1] - before[1]
                return 0.0 if total <= 0 else 100.0 * (1 - (now[0] - before[0]) / total)
            return 100.0 * os.getloadavg()[0] / (os.cpu_count() or 1)
        except (OSError, AttributeError):
            return 0.0


def queue_busy(root):
    """True while a job from the queue is running (a training run the person started)."""
    try:
        data = json.loads((Path(root) / "data" / "queue.json").read_text(encoding="utf-8"))
        return any(job.get("status") == "running" for job in data.get("jobs", []))
    except (OSError, ValueError, AttributeError):
        return False


def network_metered(*, platform=sys.platform, run=subprocess.run):
    """True / False when the OS says whether the connection is metered, None when it cannot say."""
    try:
        if platform == "win32":
            script = ("[void][Windows.Networking.Connectivity.NetworkInformation,Windows.Networking.Connectivity,ContentType=WindowsRuntime];"
                      "$p=[Windows.Networking.Connectivity.NetworkInformation]::GetInternetConnectionProfile();"
                      "if($p){[int]$p.GetConnectionCost().NetworkCostType}")
            out = run(["powershell", "-NoProfile", "-Command", script], capture_output=True, text=True, timeout=15,
                      creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout.strip()
            return None if not out.isdigit() else int(out) in (2, 3)   # 1 unrestricted, 2 fixed, 3 variable
        if platform.startswith("linux"):
            out = run(["nmcli", "-t", "-f", "GENERAL.METERED", "device", "show"], capture_output=True, text=True, timeout=10).stdout.lower()
            values = [line.split(":", 1)[1].strip() for line in out.splitlines() if ":" in line]
            if any(v in ("yes", "guess-yes") for v in values):
                return True
            return False if any(v in ("no", "guess-no") for v in values) else None
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return None


@dataclass
class Probes:
    on_battery: object = watch.on_battery
    cpu: object = field(default_factory=CpuMeter)
    queue_busy: object = queue_busy
    metered: object = network_metered
    clock: object = time.monotonic
    _metered: tuple = (None, -1e9)

    def network(self):
        value, at = self._metered
        if self.clock() - at > 60:       # asking Windows costs a PowerShell start: once a minute
            self._metered = (self.metered(), self.clock())
        return self._metered[0]


def reason_not_to_run(config, root, probes):
    """"" when collection may run now, else why not (shown on the page)."""
    if not config["enabled"]:
        return "Collection is off."
    if config["paused"]:
        return "Paused."
    idle = config["idle"]
    if idle["only_on_mains"] and probes.on_battery():
        return "Waiting for mains power."
    metered = probes.network()
    if metered is True:
        return "Waiting: this network is metered."
    if metered is None and not idle["unmetered_confirmed"]:
        return "Waiting: confirm your network is unmetered in the limits."
    if idle["only_when_idle"]:
        if probes.queue_busy(root):
            return "Waiting: a training job is running."
        cpu = probes.cpu()
        if cpu > idle["cpu_percent"]:
            return f"Waiting: the computer is busy ({cpu:.0f}% CPU)."
    return ""
