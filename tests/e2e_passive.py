"""Passive training end to end with real processes: session -> watchdog -> trainer, a hard kill, pause/resume, and a finish on the time limit.

Not named test_*.py so `unittest discover` skips it; the release workflow runs `python -m unittest tests.e2e_passive -v`.
Every wait polls with a generous timeout and fails with the session row, the journal tail and trainer.log, because on a Windows runner that log is all anyone gets.
The real sign-in entry (LaunchAgent / schtasks) is only created when CI is set; otherwise macOS gets a throwaway HOME (the watchdog inherits it, so it removes the entry from there)
and Windows gets a no-op install.
"""
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import studio_passive_watch as watch
from studio_core import Workspace
from studio_passive import PassiveStore, merge_plan

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "sample.jsonl"
DURATION_MINUTES = 1.25  # of active (unpaused) time
PLAN = {"stop": {"duration": DURATION_MINUTES, "unit": "minutes", "patience_rounds": 0},
        "limits": {"pause_on_battery": False},  # a laptop test machine on battery must not pause itself
        "teacher": {"polish": False, "lessons": False},
        "plateau": {"checks": 3, "max_steps": 200},
        "worse": {"limit": 1000000, "pivots": []},  # only the time limit may end this session
        "config": {"width": 32, "layers": 1, "heads": 2, "context": 128, "batch_size": 8, "learning_rate": 0.01, "eval_every": 10, "device": "cpu"}}
REAL_ENTRY = bool(os.environ.get("CI"))


def kill_hard(pid):
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
        else:
            os.kill(pid, signal.SIGKILL)
    except OSError:
        pass


@unittest.skipUnless(sys.platform in ("darwin", "win32"), "macOS and Windows only")
class PassiveEndToEnd(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        if not REAL_ENTRY and sys.platform == "darwin":
            home = patch.dict(os.environ, {"HOME": str(self.root / "home")})
            home.start()
            self.addCleanup(home.stop)
        self.pids, self.id = set(), None
        self.addCleanup(self.cleanup)
        workspace = Workspace(self.root)
        report = workspace.import_files([FIXTURE])
        self.assertIn("200 added", report[0])
        self.dataset = workspace.state["datasets"][0]["id"]
        self.store = PassiveStore(self.root)

    # -- helpers
    def cleanup(self):
        row = self.store.get(self.id) if self.id else None
        for pid in {*self.pids, *((row["trainer_pid"], row["watchdog_pid"]) if row else ())}:
            if pid:
                kill_hard(pid)
        if sys.platform != "win32":
            subprocess.run(["pkill", "-9", "-f", str(self.root)], capture_output=True)  # anything still holding this temp workspace
        if self.entry_checkable:
            watch.uninstall(self.root)

    entry_checkable = property(lambda self: REAL_ENTRY or sys.platform == "darwin")

    def entry_exists(self):
        label = watch._label(self.root)
        if sys.platform == "darwin":
            return (Path.home() / "Library" / "LaunchAgents" / f"com.groundwork.passive.{label}.plist").exists()
        return subprocess.run(["schtasks", "/Query", "/TN", f"GroundworkPassive-{label}"], capture_output=True).returncode == 0

    def dump(self):
        directory = self.store.directory(self.id)
        tail = lambda path, n: "".join(path.read_text(errors="replace").splitlines(True)[-n:]) if path.exists() else "(missing)"  # noqa: E731
        row = self.store.get(self.id)
        row["plan"], row["state"] = "(omitted)", {k: v for k, v in row["state"].items() if k != "live"}
        return (f"\n--- session row\n{json.dumps(row, indent=1, default=str)}\n--- journal tail\n{tail(directory / 'journal.jsonl', 15)}"
                f"--- trainer.log tail\n{tail(directory / 'trainer.log', 60)}\n--- passive-watchdog.log tail\n{tail(self.root / 'data' / 'passive-watchdog.log', 30)}")

    def fail_with_dump(self, message):
        self.fail(message + self.dump())

    def wait(self, what, predicate, timeout=60):
        deadline = time.time() + timeout
        while time.time() < deadline:
            value = predicate()
            if value:
                return value
            time.sleep(0.5)
        self.fail_with_dump(f"Timed out after {timeout}s waiting for: {what}")

    def row(self):
        row = self.store.get(self.id)
        self.pids.update(p for p in (row["trainer_pid"], row["watchdog_pid"]) if p)
        return row

    def events(self, *types):
        return [e for e in self.store.journal(self.id) if e["type"] in types]

    def alive(self, pid):
        return watch.alive(pid)

    # -- the test
    def test_session_survives_a_kill_pauses_resumes_and_finishes(self):
        plan = merge_plan({**PLAN, "data": {"sources": [self.dataset], "max_records": 500}})
        self.id = self.store.create(plan)
        install = None if REAL_ENTRY or sys.platform == "darwin" else (lambda root: None)
        watchdog = watch.start(self.root, self.id, **({"install": install} if install else {}))
        self.pids.add(watchdog)
        if self.entry_checkable:
            self.assertTrue(self.entry_exists(), "start() did not install the sign-in entry" + self.dump())

        # trainer up, beating, and has written a check and a checkpoint
        self.wait("the trainer to beat", lambda: (r := self.row())["trainer_pid"] and r["heartbeat"] and r["status"] == "running" and self.alive(r["trainer_pid"]), 90)
        self.wait("a check and a checkpoint in the journal", lambda: self.events("check") and self.events("checkpoint"), 90)
        before = self.row()
        old_pid, old_checkpoints, old_active = before["trainer_pid"], len(before["state"]["checkpoints"]), before["state"]["active_seconds"]
        self.assertGreaterEqual(old_checkpoints, 1, self.dump())

        # kill the trainer hard: the watchdog starts another one that carries on
        kill_hard(old_pid)
        self.wait("the old trainer to die", lambda: not self.alive(old_pid), 15)
        after = self.wait("the watchdog to start a new trainer that beats", lambda: (r := self.row())["trainer_pid"] not in (old_pid, None) and self.alive(r["trainer_pid"])
                          and r["heartbeat"] and r["heartbeat"] > time.time() - 30 and r["restarts"] and r, 60)
        self.assertNotEqual(after["trainer_pid"], old_pid)
        self.assertEqual(len(after["restarts"]), 1, self.dump())
        self.assertTrue(any(e.get("pid") == after["trainer_pid"] and e.get("reason") == "stopped" for e in self.events("restart")), "the watchdog journalled no restart" + self.dump())
        self.wait("the new trainer to resume from the saved state (its own restart line)", lambda: any("round" in e for e in self.events("restart")), 60)
        self.wait("a checkpoint after the restart", lambda: len(self.row()["state"]["checkpoints"]) > old_checkpoints, 90)
        resumed = self.row()["state"]
        self.assertGreaterEqual(resumed["active_seconds"], old_active, "the clock started over" + self.dump())
        rounds = self.events("round")
        self.assertTrue(rounds[-1]["parent"] is not None, "the round after the restart started from scratch, not from a checkpoint" + self.dump())
        self.assertEqual(resumed["checkpoints"][0]["n"], 1)
        self.assertGreater(len(resumed["checkpoints"]), old_checkpoints)

        # pause: status flips and nothing moves
        self.store.command(self.id, "pause")
        self.wait("status paused", lambda: self.row()["status"] == "paused", 30)
        time.sleep(3)  # let a step that was already running finish
        progress = lambda: (len(self.events("check", "checkpoint", "round")), round(self.row()["state"]["active_seconds"], 1))  # noqa: E731
        frozen = progress()
        time.sleep(8)
        self.assertEqual(progress(), frozen, "training moved while paused" + self.dump())
        self.assertEqual(self.row()["status"], "paused")

        # resume: progress again
        self.store.command(self.id, "resume")
        self.wait("status running", lambda: self.row()["status"] == "running", 30)
        self.wait("progress after resume", lambda: progress()[0] > frozen[0], 90)

        # ends on its own at the time limit
        final = self.wait("the session to finish", lambda: (r := self.row())["status"] in ("finished", "failed", "stopped") and r, 240)
        self.assertEqual(final["status"], "finished", self.dump())
        self.assertEqual(final["state"]["stop_reason"], "duration", self.dump())
        self.assertGreaterEqual(final["state"]["active_seconds"], DURATION_MINUTES * 60 - 5, self.dump())
        kept = [c for c in final["state"]["checkpoints"] if c["kept"]]
        self.assertTrue(kept and all(Path(c["path"]).exists() for c in kept), "a kept checkpoint has no file" + self.dump())
        self.assertTrue(self.events("stop")[-1]["reason"] == "duration")
        self.wait("both processes to exit", lambda: not self.alive(final["watchdog_pid"]) and not self.alive(final["trainer_pid"]), 60)
        if self.entry_checkable:
            self.assertFalse(self.entry_exists(), "the sign-in entry is still installed" + self.dump())


if __name__ == "__main__":
    unittest.main()
