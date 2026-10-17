"""Passive training under Resources: the watchdog's sign-in check and Stop, the services view, and the page's logic (window-free)."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import studio_passive_watch as watch
from studio_ia.services import WorkspaceServices
from studio_passive import PassiveStore, merge_plan


def row(**kw):
    return {"id": "p1", "status": "running", "created": "2026-10-15T08:30:00+00:00", "stop": "Stops after 8 hours.",
            "trainer": "Running", "watchdog": "Running", "restarts": 0, "heard": 30.0, "error": "", **kw}


class SignIn(unittest.TestCase):
    def test_installed_follows_install_and_uninstall_on_each_platform(self):
        run = MagicMock()
        for platform in ("darwin", "linux", "win32"):
            with self.subTest(platform), tempfile.TemporaryDirectory() as home, tempfile.TemporaryDirectory() as root:
                self.assertFalse(watch.installed(root, home=home, platform=platform))
                watch.install(root, home=home, run=run, platform=platform)
                self.assertTrue(watch.installed(root, home=home, platform=platform))
                watch.uninstall(root, home=home, run=run, platform=platform)
                self.assertFalse(watch.installed(root, home=home, platform=platform))

    def test_other_platforms_never_start_at_sign_in(self):
        self.assertFalse(watch.installed("x", platform="freebsd"))


class Stop(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = PassiveStore(tmp.name)
        self.id = self.store.create(merge_plan({}))
        self.store.update(self.id, status="running", trainer_pid=4001, watchdog_pid=4002)

    def test_a_running_trainer_is_asked_and_stops_itself(self):
        status = watch.stop(self.store, self.id, alive=lambda pid: pid == 4001, ours=lambda *a: True)
        session = self.store.get(self.id)
        self.assertEqual((status, session["status"], session["command"]), ("running", "running", "stop"))

    def test_a_watchdog_alone_still_takes_the_stop_to_a_restarted_trainer(self):
        watch.stop(self.store, self.id, alive=lambda pid: pid == 4002, ours=lambda *a: True)
        self.assertEqual((self.store.get(self.id)["status"], self.store.get(self.id)["command"]), ("running", "stop"))

    def test_with_nobody_running_the_record_is_closed_here(self):
        status = watch.stop(self.store, self.id, alive=lambda pid: False)
        session = self.store.get(self.id)
        self.assertEqual((status, session["status"], session["command"]), ("stopped", "stopped", ""))
        self.assertEqual(self.store.journal(self.id)[-1]["type"], "stop")
        self.assertIsNone(self.store.active())

    def test_a_live_pid_that_is_not_ours_counts_as_nobody(self):
        self.assertEqual(watch.stop(self.store, self.id, alive=lambda pid: True, ours=lambda *a: False), "stopped")

    def test_a_finished_session_is_left_alone(self):
        self.store.update(self.id, status="finished")
        self.assertEqual(watch.stop(self.store, self.id, alive=lambda pid: False), "finished")
        self.assertEqual(self.store.get(self.id)["command"], "")
        self.assertIsNone(watch.stop(self.store, "nope"))


class WorkspaceView(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.services = SimpleNamespace(root=self.root)  # passive() reads nothing but the workspace root

    def view(self):
        with patch.object(watch, "installed", return_value=True):
            return WorkspaceServices.passive(self.services)

    def test_a_workspace_that_never_used_it_reads_empty_and_creates_nothing(self):
        self.assertEqual(self.view(), {"sessions": [], "signin": True})
        self.assertFalse((self.root / "data" / "passive.sqlite").exists())

    def test_sessions_newest_first_with_status_and_plan(self):
        store = PassiveStore(self.root)
        old = store.create(merge_plan({}))
        store.update(old, status="finished", restarts=[1.0, 2.0])
        new = store.create(merge_plan({}))
        store.update(new, status="running")
        sessions = self.view()["sessions"]
        self.assertEqual([s["id"] for s in sessions], [new, old])
        self.assertEqual((sessions[0]["status"], sessions[0]["trainer"]), ("running", "Stopped"))  # no live trainer pid
        self.assertEqual((sessions[1]["trainer"], sessions[1]["watchdog"], sessions[1]["restarts"]), ("Stopped", "Stopped", 2))
        self.assertTrue(sessions[0]["stop"].startswith("Stops after"))

    def test_stop_with_nothing_running_closes_the_session(self):
        store = PassiveStore(self.root)
        sid = store.create(merge_plan({}))
        self.assertEqual(WorkspaceServices.stop_passive(self.services, sid), "stopped")
        self.assertEqual(self.view()["sessions"][0]["status"], "stopped")


class KeepGoing(unittest.TestCase):
    """Bringing back a session whose watchdog and trainer died: a fake spawn, never a real process."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.services = SimpleNamespace(root=self.root)
        self.spawned, self.installed = [], []
        self.hooks = dict(spawn=lambda command, log, cwd: self.spawned.append(command) or 777, install=self.installed.append,
                          alive=lambda pid: False, ours=lambda *a: False)

    def keep(self, session_id=None):
        return WorkspaceServices.keep_passive_going(self.services, session_id, **self.hooks)

    def test_a_dead_watchdog_is_started_again_and_recorded(self):
        store = PassiveStore(self.root)
        sid = store.create(merge_plan({}))
        store.update(sid, status="running", trainer_pid=4001, watchdog_pid=4002)
        self.assertEqual(self.keep(sid), "running")
        self.assertEqual((len(self.spawned), store.get(sid)["watchdog_pid"]), (1, 777))
        self.assertEqual(self.installed, [self.root])

    def test_without_an_id_the_active_session_is_used_and_a_live_watchdog_is_left_alone(self):
        store = PassiveStore(self.root)
        sid = store.create(merge_plan({}))
        store.update(sid, status="paused", watchdog_pid=4002)
        self.hooks.update(alive=lambda pid: True, ours=lambda *a: True)
        self.assertEqual(self.keep(), "paused")
        self.assertEqual(self.spawned, [])

    def test_nothing_to_keep_going(self):
        self.assertIsNone(self.keep())  # never used: no record is created
        self.assertFalse((self.root / "data" / "passive.sqlite").exists())
        store = PassiveStore(self.root)
        sid = store.create(merge_plan({}))
        store.update(sid, status="stopped")
        self.assertIsNone(self.keep())
        self.assertIsNone(self.keep("nope"))
        with self.assertRaises(RuntimeError):
            self.keep(sid)
        self.assertEqual(self.spawned, [])

    def test_the_api_start_restarts_the_active_session_and_never_raises(self):
        store = PassiveStore(self.root)
        sid = store.create(merge_plan({}))
        store.update(sid, status="running")
        services = SimpleNamespace(root=self.root)
        services.keep_passive_going = lambda *a, **kw: WorkspaceServices.keep_passive_going(services, *a, **kw)
        WorkspaceServices.resume_passive_at_start(services, **self.hooks)
        self.assertEqual(len(self.spawned), 1)
        self.hooks["spawn"] = lambda *a: (_ for _ in ()).throw(OSError("no python"))
        WorkspaceServices.resume_passive_at_start(services, **self.hooks)  # prints, does not raise


class PauseResume(unittest.TestCase):
    """The page's Pause and Resume write the command the trainer takes and journals (tests.test_passive.test_pause_and_resume)."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.services = SimpleNamespace(root=self.root)
        self.store = PassiveStore(self.root)
        self.id = self.store.create(merge_plan({}))
        self.store.update(self.id, status="running")

    def send(self, command, session_id=None):
        return WorkspaceServices.command_passive(self.services, session_id or self.id, command)

    def test_the_command_is_left_for_the_trainer(self):
        self.assertEqual(self.send("pause"), "running")  # the trainer has not taken it yet
        self.assertEqual(self.store.get(self.id)["command"], "pause")
        self.store.update(self.id, status="paused", command="")
        self.assertEqual(self.send("resume"), "paused")
        self.assertEqual(self.store.get(self.id)["command"], "resume")

    def test_unknown_and_never_used_are_none(self):
        self.assertIsNone(self.send("pause", "nope"))
        with tempfile.TemporaryDirectory() as empty:
            self.assertIsNone(WorkspaceServices.command_passive(SimpleNamespace(root=Path(empty)), "x", "pause"))
            self.assertFalse((Path(empty) / "data" / "passive.sqlite").exists())

    def test_an_ended_or_stopping_session_is_refused(self):
        self.store.command(self.id, "stop")
        with self.assertRaises(RuntimeError):
            self.send("pause")
        self.assertEqual(self.store.get(self.id)["command"], "stop")  # the stop is not replaced
        for status in ("stopped", "finished", "failed"):
            self.store.update(self.id, status=status)
            with self.assertRaises(RuntimeError):
                self.send("pause")
        with self.assertRaises(ValueError):
            self.send("stop")


if __name__ == "__main__":
    unittest.main()
