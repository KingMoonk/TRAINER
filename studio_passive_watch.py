"""Passive training watchdog: keeps the detached trainer alive, starts at sign-in, keeps the computer awake, pauses on battery.

`python -m studio_passive_watch --root <workspace>` runs the loop; the app calls `start(root)` after creating a session.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import plistlib
import subprocess
import sys
import time
import traceback
from pathlib import Path

from studio_core import atomic_json, module_command
from studio_passive import ACTIVE, PassiveStore, trainer_command
from studio_platform import LOG_LIMIT, _children, _lines, alive, command_line, keep_awake, on_battery, ours, release_awake, spawn, terminate  # noqa: F401 - re-exported: #486 removes this module

HEARTBEAT_SECONDS, MAX_RESTARTS, RESTART_WINDOW, POLL_SECONDS = 60, 5, 600, 5
REPO = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- start at sign-in

def _label(root):
    return hashlib.sha1(str(Path(root).resolve()).encode()).hexdigest()[:8]


def _watch_command(root):
    python = Path(sys.executable)
    if sys.platform == "win32" and python.with_name("pythonw.exe").exists():
        python = python.with_name("pythonw.exe")  # no console window
    return [str(python), "-m", "studio_passive_watch", "--root", str(Path(root).resolve())]


def install(root, *, home=None, run=subprocess.run, platform=sys.platform):
    """Start the watchdog at sign-in. Returns the plist / .desktop path or the task name, or None (never raises: training must go on)."""
    root = Path(root).resolve()
    try:
        if platform == "darwin":
            path = Path(home or Path.home()) / "Library" / "LaunchAgents" / f"com.groundwork.passive.{_label(root)}.plist"
            path.parent.mkdir(parents=True, exist_ok=True)
            log = str(root / "data" / "passive-watchdog.log")
            # KeepAlive: launchd starts it again only if it crashed; a watchdog that finished (no session left) stays finished
            path.write_bytes(plistlib.dumps({"Label": path.stem, "ProgramArguments": module_command("studio_passive_watch", "--root", str(root)), "WorkingDirectory": str(REPO),
                                             "RunAtLoad": True, "KeepAlive": {"SuccessfulExit": False}, "StandardOutPath": log, "StandardErrorPath": log}))
            return str(path)
        if platform.startswith("linux"):
            path = _desktop(root, home)
            path.parent.mkdir(parents=True, exist_ok=True)
            quote = lambda a: '"' + "".join("\\" + c if c in '"`$\\' else c for c in a).replace("\\", "\\\\").replace("%", "%%") + '"'  # noqa: E731 (Desktop Entry spec)
            path.write_text("[Desktop Entry]\nType=Application\nName=Groundwork passive training\nNoDisplay=true\nX-GNOME-Autostart-enabled=true\n"
                            f"Path={REPO}\nExec={' '.join(map(quote, module_command('studio_passive_watch', '--root', str(root))))}\n")
            return str(path)
        if platform == "win32":
            # ponytail: runs the script by path so the repo is on sys.path without a working directory; a moved repo needs the next session to reinstall.
            python = Path(sys.executable)
            python = python.with_name("pythonw.exe") if python.with_name("pythonw.exe").exists() else python
            name = f"GroundworkPassive-{_label(root)}"
            command = [str(python), str(REPO / "studio_passive_watch.py"), "--root", str(root)]
            script = _script(root)  # schtasks /TR holds at most 261 characters; the command line itself can be longer
            script.parent.mkdir(parents=True, exist_ok=True)
            script.write_text('@echo off\r\nstart "" ' + subprocess.list2cmdline(command).replace("%", "%%") + "\r\n", encoding="utf-8")
            try:
                run(["schtasks", "/Create", "/TN", name, "/TR", f'"{script}"', "/SC", "ONLOGON", "/F"],
                    capture_output=True, check=True)
            except (OSError, subprocess.SubprocessError):
                script.unlink(missing_ok=True)  # no task, so no script left claiming there is one
                raise
            return name
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _desktop(root, home=None):
    return Path(home or Path.home()) / ".config" / "autostart" / f"groundwork-passive-{_label(root)}.desktop"


def _script(root):
    return Path(root) / "data" / f"passive-watchdog-{_label(root)}.cmd"


def _task(root):
    return f"GroundworkPassive-{_label(root)}"


def _task_exists(root, run):
    try:
        return run(["schtasks", "/Query", "/TN", _task(root)], capture_output=True).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def uninstall(root, *, home=None, run=subprocess.run, platform=sys.platform):
    """Remove the sign-in entry. Returns False (and says so on stderr) when a Windows task could not be deleted and is still there; never raises."""
    root = Path(root).resolve()
    try:
        if platform == "darwin":
            (Path(home or Path.home()) / "Library" / "LaunchAgents" / f"com.groundwork.passive.{_label(root)}.plist").unlink(missing_ok=True)
        elif platform.startswith("linux"):
            _desktop(root, home).unlink(missing_ok=True)
        elif platform == "win32":
            try:
                deleted = run(["schtasks", "/Delete", "/TN", _task(root), "/F"], capture_output=True).returncode == 0
            except (OSError, subprocess.SubprocessError):
                deleted = False
            if not deleted and _task_exists(root, run):
                print(f"Could not delete the sign-in task {_task(root)}; remove it in Task Scheduler.", file=sys.stderr, flush=True)
                return False  # keep the script: the task still points at it
            _script(root).unlink(missing_ok=True)
    except (OSError, subprocess.SubprocessError):
        return False
    return True


# --------------------------------------------------------------------------- one pass

def _memo(store, session_id):
    try:
        return json.loads((store.directory(session_id) / "watchdog.json").read_text())
    except (OSError, ValueError):
        return {}


def _remember(store, session_id, memo):
    atomic_json(store.directory(session_id) / "watchdog.json", memo)


def _power(store, session, memo, battery):
    """Pause on battery, resume on mains: only ever undoes its own pause. The engine journals pause/resume itself."""
    sid = session["id"]
    if not session["plan"]["limits"]["pause_on_battery"] and not memo.get("paused"):
        return None
    if session["plan"]["limits"]["pause_on_battery"] and battery():
        # only ever replaces no command or a pause: a stop or the user's resume written since `session` was read wins
        if session["status"] == "running" and not session["command"] and not memo.get("paused") and store.command(sid, "pause", expect=("", "pause")):
            _remember(store, sid, {**memo, "paused": True})
            return "battery_paused"
        return None
    if not memo.get("paused"):
        return None
    word = None
    if (session["status"] == "paused" or session["command"] == "pause") and store.command(sid, "resume", expect=("", "pause")):
        word = "battery_resumed"
    _remember(store, sid, {**memo, "paused": False})
    return word


def check(store, *, now=time.time, spawn=spawn, alive=alive, kill=terminate, on_battery=on_battery, ours=ours, woke=False):
    """One watchdog pass. Returns none | ok | paused | started | restarted | waiting | gave_up | battery_paused | battery_resumed.

    `woke`: this watchdog was not running for a while (it just started, or the computer slept); a restart now is not held against the trainer.
    """
    session = store.active()
    if session is None:
        return "none"
    sid, pid, t, memo = session["id"], session["trainer_pid"], now(), _memo(store, session["id"])
    running = alive(pid) and ours(pid, "studio_passive", sid)  # a live pid that is not our trainer counts as a dead trainer and is never killed
    # a trainer spawned moments ago has not beaten yet, and the heartbeat column still holds the old value
    if running and t - max(session["heartbeat"] or 0, memo.get("spawned", 0)) < HEARTBEAT_SECONDS:
        word = _power(store, session, memo, on_battery)
        if word:
            store.append(sid, {"type": "power", "reason": word})  # the engine's own pause/resume lines say when; this says why
        return word or ("paused" if session["status"] == "paused" else "ok")
    first = not pid and not memo.get("spawned")
    saved = len(session["state"].get("checkpoints", []))
    stuck = "checkpoints" in memo and memo["checkpoints"] == saved  # no new checkpoint since the last (re)start
    in_window = sum(t - r < RESTART_WINDOW for r in session["restarts"]) >= MAX_RESTARTS
    if not first and not woke and (in_window or (stuck and memo.get("fruitless", 0) >= MAX_RESTARTS)):
        if running:
            kill(pid)
        why = f"{MAX_RESTARTS} restarts in {RESTART_WINDOW // 60} minutes" if in_window else f"{MAX_RESTARTS} restarts in a row without a new checkpoint"
        message = f"The trainer kept stopping, so passive training gave up after {why}." + (f" The last problem: {session['error']}" if session["error"] else " See trainer.log.")
        store.update(sid, status="failed", error=message)
        store.append(sid, {"type": "error", "message": message})
        return "gave_up"
    if running:
        kill(pid)
        if alive(pid):
            return "waiting"  # never two trainers: spawn on a later pass, once the old one is gone
    new = spawn(trainer_command(store.root, sid), store.directory(sid) / "trainer.log", REPO)
    fruitless = memo.get("fruitless", 0) if first or woke else memo.get("fruitless", 0) + 1 if stuck else 0
    _remember(store, sid, {**memo, "spawned": t, "checkpoints": saved, "fruitless": fruitless})
    if first:
        store.update(sid, trainer_pid=new)
        return "started"
    store.update(sid, trainer_pid=new, **({} if woke else {"restarts": [*session["restarts"], t][-50:]}))
    store.append(sid, {"type": "restart", "reason": "silent" if running else "stopped", "pid": new})
    return "restarted"


# --------------------------------------------------------------------------- the daemon

def watch(root, *, poll=POLL_SECONDS, sleep=time.sleep, me=None, **hooks):
    """Check every `poll` seconds until no session is active; then remove the sign-in entry and return."""
    store, me, clock, last = PassiveStore(root), me or os.getpid(), hooks.get("now", time.time), None
    error, logged = None, 0.0
    while True:
        try:  # a locked record, a full disk or a failed spawn must not end the watchdog
            session = store.active()
            if session is None:
                uninstall(root)
                return
            other = session["watchdog_pid"]
            if other and other != me and hooks.get("alive", alive)(other) and hooks.get("ours", ours)(other, "studio_passive_watch"):
                return
            if other != me:
                store.update(session["id"], watchdog_pid=me)
            t = clock()
            gap, last = None if last is None else t - last, t
            slept = gap is not None and gap > 6 * poll
            if slept and session["trainer_pid"]:
                # this loop was asleep (the computer slept): the heartbeat looks old only because nobody was awake to hear it, so the trainer gets a fresh grace period
                _remember(store, session["id"], {**_memo(store, session["id"]), "spawned": t})
            check(store, woke=gap is None or slept, **hooks)  # first pass (sign-in after a reboot, say) or after a sleep: a restart is not the trainer's fault
        except Exception:
            text = traceback.format_exc()
            if text != error or time.monotonic() - logged >= 60:  # the same failure every 5 seconds is said once a minute
                print(text, file=sys.stderr, flush=True)
                error, logged = text, time.monotonic()
        sleep(poll)


def start(root, session_id=None, *, spawn=spawn, install=install, alive=alive, ours=ours):
    """Called by the app after `PassiveStore.create`: spawn the detached watchdog, register it for sign-in, return its pid."""
    store = PassiveStore(root)
    session = store.get(session_id) if session_id else store.active()
    if session is None or session["status"] not in ACTIVE:
        raise ValueError("There is no passive training session to keep going.")
    if alive(session["watchdog_pid"]) and ours(session["watchdog_pid"], "studio_passive_watch"):
        return session["watchdog_pid"]
    pid = spawn(_watch_command(root), Path(root) / "data" / "passive-watchdog.log", REPO)
    store.update(session["id"], watchdog_pid=pid)
    install(root)
    return pid


def installed(root, *, home=None, run=subprocess.run, platform=sys.platform):
    """True when the watchdog is registered to start at sign-in (Windows: Task Scheduler has the task, asked with `schtasks /Query`)."""
    root = Path(root).resolve()
    if platform == "darwin":
        return (Path(home or Path.home()) / "Library" / "LaunchAgents" / f"com.groundwork.passive.{_label(root)}.plist").exists()
    if platform.startswith("linux"):
        return _desktop(root, home).exists()
    return platform == "win32" and _task_exists(root, run)


def stop(store, session_id, *, alive=alive, ours=ours):
    """The app's Stop: leave "stop" for the trainer. With neither the trainer nor the watchdog running nobody would take it,
    so the record is closed here instead. Returns the session's status afterwards."""
    session = store.get(session_id)
    if session is None or session["status"] not in ACTIVE:
        return session and session["status"]
    store.command(session_id, "stop")
    trainer = alive(session["trainer_pid"]) and ours(session["trainer_pid"], "studio_passive", session_id)
    watchdog = alive(session["watchdog_pid"]) and ours(session["watchdog_pid"], "studio_passive_watch")
    if trainer or watchdog:
        return session["status"]
    store.update(session_id, status="stopped", command="")
    store.append(session_id, {"type": "stop", "reason": "user"})
    return "stopped"


def status(store, session, *, now=time.time, alive=alive, ours=ours):
    """The plain facts for the 'Keeping it going' table."""
    t, beat = now(), session["heartbeat"]
    fresh = t - max(beat or 0, _memo(store, session["id"]).get("spawned", 0)) < HEARTBEAT_SECONDS
    trainer = "Stopped" if not (alive(session["trainer_pid"]) and ours(session["trainer_pid"], "studio_passive", session["id"])) else "Running" if fresh else "Restarting"
    return {"trainer": trainer, "heard": None if beat is None else max(0.0, t - beat), "watchdog": "Running" if alive(session["watchdog_pid"]) and ours(session["watchdog_pid"], "studio_passive_watch") else "Stopped", "restarts": len(session["restarts"])}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="studio_passive_watch")
    parser.add_argument("--root", default=str(REPO))
    watch(parser.parse_args(argv).root)


if __name__ == "__main__":
    main()
