"""Passive training watchdog: keeps the detached trainer alive, starts at sign-in, keeps the computer awake, pauses on battery.

`python -m studio_passive_watch --root <workspace>` runs the loop; the app calls `start(root)` after creating a session.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import plistlib
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

from studio_core import module_command
from studio_passive import ACTIVE, PassiveStore, trainer_command

HEARTBEAT_SECONDS, MAX_RESTARTS, RESTART_WINDOW, POLL_SECONDS = 60, 5, 600, 5
REPO = Path(__file__).resolve().parent
_children: dict[int, subprocess.Popen] = {}  # children we spawned: reaped here so a dead trainer is not a zombie that still "exists"


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


def spawn(command, log_path, cwd):
    """Start `command` fully detached, output appended to `log_path`; returns its pid."""
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    options = {"creationflags": subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW, "close_fds": True} if sys.platform == "win32" else {"start_new_session": True}
    with open(log_path, "ab") as log:
        child = subprocess.Popen(command, cwd=str(cwd), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, **options)
    _children[child.pid] = child
    return child.pid


def terminate(pid, *, run=subprocess.run, platform=sys.platform, is_alive=alive, sleep=time.sleep, grace=5.0):
    """Stop a silent trainer: SIGTERM, then SIGKILL after `grace` seconds (Windows: taskkill /T /F)."""
    try:
        if platform == "win32":
            run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
            return
        os.kill(pid, signal.SIGTERM)
        waited = 0.0
        while is_alive(pid) and waited < grace:
            sleep(0.1)
            waited += 0.1
        if is_alive(pid):
            os.kill(pid, signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        pass


# --------------------------------------------------------------------------- keep awake, battery

def keep_awake(pid, *, platform=sys.platform, popen=subprocess.Popen):
    """Ask the computer not to sleep while process `pid` runs. Windows: call it from inside that process (the request is per thread)."""
    try:
        if platform == "darwin":
            return popen(["caffeinate", "-i", "-w", str(pid)], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if platform == "win32":
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
            return True
    except (OSError, AttributeError):
        pass
    return None


def on_battery(*, platform=sys.platform, run=subprocess.run):
    try:
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


# --------------------------------------------------------------------------- start at sign-in

def _label(root):
    return hashlib.sha1(str(Path(root).resolve()).encode()).hexdigest()[:8]


def _watch_command(root):
    if getattr(sys, "frozen", False):
        return module_command("studio_passive_watch", "--root", str(Path(root).resolve()))
    python = Path(sys.executable)
    if sys.platform == "win32" and python.with_name("pythonw.exe").exists():
        python = python.with_name("pythonw.exe")  # no console window
    return [str(python), "-m", "studio_passive_watch", "--root", str(Path(root).resolve())]


def install(root, *, home=None, run=subprocess.run, platform=sys.platform):
    """Start the watchdog at sign-in. Returns the plist path / task name, or None (never raises: training must go on)."""
    root = Path(root).resolve()
    try:
        if platform == "darwin":
            path = Path(home or Path.home()) / "Library" / "LaunchAgents" / f"com.groundwork.passive.{_label(root)}.plist"
            path.parent.mkdir(parents=True, exist_ok=True)
            log = str(root / "data" / "passive-watchdog.log")
            path.write_bytes(plistlib.dumps({"Label": path.stem, "ProgramArguments": module_command("studio_passive_watch", "--root", str(root)), "WorkingDirectory": str(REPO),
                                             "RunAtLoad": True, "KeepAlive": False, "StandardOutPath": log, "StandardErrorPath": log}))
            return str(path)
        if platform == "win32":
            # ponytail: runs the script by path so the repo is on sys.path without a working directory; a moved repo needs the next session to reinstall.
            python = Path(sys.executable)
            python = python.with_name("pythonw.exe") if python.with_name("pythonw.exe").exists() else python
            name = f"GroundworkPassive-{_label(root)}"
            command = module_command("studio_passive_watch", "--root", str(root)) if getattr(sys, "frozen", False) else [str(python), str(REPO / "studio_passive_watch.py"), "--root", str(root)]
            run(["schtasks", "/Create", "/TN", name, "/TR", subprocess.list2cmdline(command), "/SC", "ONLOGON", "/F"],
                capture_output=True, check=True)
            return name
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def uninstall(root, *, home=None, run=subprocess.run, platform=sys.platform):
    root = Path(root).resolve()
    try:
        if platform == "darwin":
            (Path(home or Path.home()) / "Library" / "LaunchAgents" / f"com.groundwork.passive.{_label(root)}.plist").unlink(missing_ok=True)
        elif platform == "win32":
            run(["schtasks", "/Delete", "/TN", f"GroundworkPassive-{_label(root)}", "/F"], capture_output=True)
    except (OSError, subprocess.SubprocessError):
        pass


# --------------------------------------------------------------------------- one pass

def _memo(store, session_id):
    try:
        return json.loads((store.directory(session_id) / "watchdog.json").read_text())
    except (OSError, ValueError):
        return {}


def _remember(store, session_id, memo):
    (store.directory(session_id) / "watchdog.json").write_text(json.dumps(memo))


def _power(store, session, memo, battery):
    """Pause on battery, resume on mains: only ever undoes its own pause. The engine journals pause/resume itself."""
    sid = session["id"]
    if not session["plan"]["limits"]["pause_on_battery"] and not memo.get("paused"):
        return None
    if session["plan"]["limits"]["pause_on_battery"] and battery():
        if session["status"] == "running" and not session["command"] and not memo.get("paused"):
            store.command(sid, "pause")
            _remember(store, sid, {**memo, "paused": True})
            return "battery_paused"
        return None
    if not memo.get("paused"):
        return None
    word = None
    if session["status"] == "paused" or session["command"] == "pause":
        store.command(sid, "resume")
        word = "battery_resumed"
    _remember(store, sid, {**memo, "paused": False})
    return word


def check(store, *, now=time.time, spawn=spawn, alive=alive, kill=terminate, on_battery=on_battery):
    """One watchdog pass. Returns none | ok | paused | started | restarted | gave_up | battery_paused | battery_resumed."""
    session = store.active()
    if session is None:
        return "none"
    sid, pid, t, memo = session["id"], session["trainer_pid"], now(), _memo(store, session["id"])
    running = alive(pid)
    # a trainer spawned moments ago has not beaten yet, and the heartbeat column still holds the old value
    if running and t - max(session["heartbeat"] or 0, memo.get("spawned", 0)) < HEARTBEAT_SECONDS:
        word = _power(store, session, memo, on_battery)
        if word:
            store.append(sid, {"type": "power", "reason": word})  # the engine's own pause/resume lines say when; this says why
        return word or ("paused" if session["status"] == "paused" else "ok")
    first = not pid and not memo.get("spawned")
    if not first and sum(t - r < RESTART_WINDOW for r in session["restarts"]) >= MAX_RESTARTS:
        if running:
            kill(pid)
        message = f"The trainer kept stopping, so passive training gave up after {MAX_RESTARTS} restarts in {RESTART_WINDOW // 60} minutes." + (f" The last problem: {session['error']}" if session["error"] else " See trainer.log.")
        store.update(sid, status="failed", error=message)
        store.append(sid, {"type": "error", "message": message})
        return "gave_up"
    if running:
        kill(pid)
    new = spawn(trainer_command(store.root, sid), store.directory(sid) / "trainer.log", REPO)
    _remember(store, sid, {**memo, "spawned": t})
    if first:
        store.update(sid, trainer_pid=new)
        return "started"
    store.update(sid, trainer_pid=new, restarts=[*session["restarts"], t][-50:])
    store.append(sid, {"type": "restart", "reason": "silent" if running else "stopped", "pid": new})
    return "restarted"


# --------------------------------------------------------------------------- the daemon

def watch(root, *, poll=POLL_SECONDS, sleep=time.sleep, me=None, **hooks):
    """Check every `poll` seconds until no session is active; then remove the sign-in entry and return."""
    store, me = PassiveStore(root), me or os.getpid()
    while True:
        session = store.active()
        if session is None:
            uninstall(root)
            return
        other = session["watchdog_pid"]
        # ponytail: a recycled pid could look like a live watchdog; it only delays a restart of the watchdog until that process exits.
        if other and other != me and hooks.get("alive", alive)(other):
            return
        if other != me:
            store.update(session["id"], watchdog_pid=me)
        try:
            check(store, **hooks)
        except Exception:
            traceback.print_exc()  # a locked record or a failed spawn must not end the watchdog
        sleep(poll)


def start(root, session_id=None, *, spawn=spawn, install=install, alive=alive):
    """Called by the app after `PassiveStore.create`: spawn the detached watchdog, register it for sign-in, return its pid."""
    store = PassiveStore(root)
    session = store.get(session_id) if session_id else store.active()
    if session is None or session["status"] not in ACTIVE:
        raise ValueError("There is no passive training session to keep going.")
    if alive(session["watchdog_pid"]):
        return session["watchdog_pid"]
    pid = spawn(_watch_command(root), Path(root) / "data" / "passive-watchdog.log", REPO)
    store.update(session["id"], watchdog_pid=pid)
    install(root)
    return pid


def status(store, session, *, now=time.time, alive=alive):
    """The plain facts for the 'Keeping it going' table."""
    t, beat = now(), session["heartbeat"]
    fresh = t - max(beat or 0, _memo(store, session["id"]).get("spawned", 0)) < HEARTBEAT_SECONDS
    trainer = "Stopped" if not alive(session["trainer_pid"]) else "Running" if fresh else "Restarting"
    return {"trainer": trainer, "heard": None if beat is None else max(0.0, t - beat), "watchdog": "Running" if alive(session["watchdog_pid"]) else "Stopped", "restarts": len(session["restarts"])}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="studio_passive_watch")
    parser.add_argument("--root", default=str(REPO))
    watch(parser.parse_args(argv).root)


if __name__ == "__main__":
    main()
