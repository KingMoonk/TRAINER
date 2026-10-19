"""Process and power helpers shared by every module that runs or watches a trainer: is a pid alive, is it ours, start one detached,
stop one, keep the computer awake while it runs, tell whether it is on battery. Neutral ground: the queue, the worker, the cluster
and passive training's watchdog all use these, so none of them has to import another's module for them."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

LOG_LIMIT = 5 * 1024 * 1024  # a log past this size starts afresh at the next spawn
_children: dict[int, subprocess.Popen] = {}  # children we spawned: reaped here so a dead trainer is not a zombie that still "exists"
_lines: dict[tuple, str] = {}  # Windows: command line by (pid, creation time); reading one starts PowerShell


# --------------------------------------------------------------------------- processes

def alive(pid):
    if not pid or pid <= 0:
        return False
    child = _children.get(pid)
    if child is not None and child.poll() is not None:
        return False
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.restype = wintypes.HANDLE
        handle = kernel.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return ctypes.get_last_error() == 5  # access denied: it exists
        code = wintypes.DWORD()
        ok = kernel.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel.CloseHandle(handle)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _created(pid):
    """Windows: the creation time of process `pid`, or None. A pid and its creation time name one process for good."""
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.restype = wintypes.HANDLE
    handle = kernel.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return None
    times = [wintypes.FILETIME() for _ in range(4)]
    ok = kernel.GetProcessTimes(handle, *[ctypes.byref(t) for t in times])
    kernel.CloseHandle(handle)
    return (times[0].dwHighDateTime << 32 | times[0].dwLowDateTime) if ok else None


def command_line(pid):
    """The command line of process `pid`: "" when there is no such process, None when it could not be read."""
    try:
        if sys.platform == "win32":
            key = (int(pid), _created(pid))  # PowerShell takes about half a second and the page asks every 2 seconds: once per process
            if key[1] is not None and key in _lines:
                return _lines[key]
            command = ["powershell", "-NoProfile", "-Command", f"(Get-CimInstance Win32_Process -Filter 'ProcessId={int(pid)}').CommandLine"]
            line = subprocess.run(command, capture_output=True, text=True, timeout=30, creationflags=subprocess.CREATE_NO_WINDOW).stdout.strip()
            if key[1] is not None:
                _lines[key] = line
            return line
        return subprocess.run(["ps", "-o", "command=", "-p", str(int(pid))], capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def ours(pid, *needles, read=command_line):
    """True when `pid` is still the process we started: its command line holds every needle. After a reboot or a long sleep a pid can belong to anything."""
    line = read(pid)
    return line is None or (bool(line) and all(n in line for n in needles))  # unreadable: trust the pid rather than start a second trainer


def spawn(command, log_path, cwd):
    """Start `command` fully detached, output appended to `log_path` (started afresh past LOG_LIMIT); returns its pid."""
    for pid, child in list(_children.items()):
        if child.poll() is not None:  # reaped: no zombie left behind for each finished trainer or watchdog
            del _children[pid]
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    options = {"creationflags": subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW, "close_fds": True} if sys.platform == "win32" else {"start_new_session": True}
    with open(log_path, "wb" if log_path.exists() and log_path.stat().st_size > LOG_LIMIT else "ab") as log:
        child = subprocess.Popen(command, cwd=str(cwd), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, **options)
    _children[child.pid] = child
    return child.pid


def terminate(pid, *, run=subprocess.run, platform=sys.platform, is_alive=alive, sleep=time.sleep, grace=5.0):
    """Stop a silent trainer: SIGTERM, then SIGKILL after `grace` seconds (Windows: taskkill /T /F). True once it is gone."""
    try:
        if platform == "win32":
            run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
        else:
            os.kill(pid, signal.SIGTERM)
            waited = 0.0
            while is_alive(pid) and waited < grace:
                sleep(0.1)
                waited += 0.1
            if is_alive(pid):
                os.kill(pid, signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        pass
    for _ in range(20):  # a kill is not instant: up to two seconds for it to land
        if not is_alive(pid):
            return True
        sleep(0.1)
    return False


# --------------------------------------------------------------------------- keep awake, battery

def keep_awake(pid, *, platform=sys.platform, popen=subprocess.Popen):
    """Ask the computer not to sleep while process `pid` runs; returns a handle for release_awake, or None.
    Windows: call it from inside that process, from the thread that will release it (the request is per thread)."""
    quiet = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    try:
        if platform == "darwin":
            return popen(["caffeinate", "-i", "-w", str(pid)], **quiet)
        if platform.startswith("linux"):  # systemd machines; without systemd-inhibit this raises FileNotFoundError and nothing happens
            return popen(["systemd-inhibit", "--what=sleep:idle", "--who=Groundwork", "--why=Passive training", "--mode=block", "tail", f"--pid={pid}", "-f", "/dev/null"], **quiet)
        if platform == "win32":
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
            return True
    except (OSError, AttributeError):
        pass
    return None


def release_awake(handle, *, platform=sys.platform):
    """Undo keep_awake (while paused). Never raises."""
    try:
        if hasattr(handle, "terminate"):
            handle.terminate()
            handle.wait(timeout=5)
        elif handle is True and platform == "win32":
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)  # ES_CONTINUOUS alone clears this thread's request
    except (OSError, AttributeError, subprocess.SubprocessError):
        pass


def on_battery(*, platform=sys.platform, run=subprocess.run, supplies=Path("/sys/class/power_supply")):
    try:
        if platform.startswith("linux"):
            read = lambda p, name: (p / name).read_text().strip() if (p / name).exists() else ""  # noqa: E731
            found = [(read(p, "type"), read(p, "online")) for p in supplies.iterdir()] if supplies.is_dir() else []
            adapters = [online for kind, online in found if kind != "Battery" and online]
            return any(kind == "Battery" for kind, _ in found) and bool(adapters) and "1" not in adapters
        if platform == "darwin":
            out = run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=5).stdout
            return "Battery Power" in out.splitlines()[0]
        if platform == "win32":
            import ctypes

            class Status(ctypes.Structure):
                _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte), ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                            ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]
            status = Status()
            return bool(ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status))) and status.ACLineStatus == 0
    except (OSError, IndexError, subprocess.SubprocessError, AttributeError):
        pass
    return False
