"""Passive training under Resources: the watchdog's sign-in check and Stop, the services view, and the page's logic (window-free)."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import studio_passive_watch as watch
from studio_ia.pages.passive import competes, session_view, when
from studio_ia.services import FakeServices, Job, QueueState, WorkspaceServices
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


class PageLogic(unittest.TestCase):
    def test_running_session_reads_plainly_and_can_stop(self):
        headline, facts, can_stop = session_view(row())
        self.assertTrue(headline.startswith("Running · started "), headline)
        self.assertEqual(facts, "Trainer: Running · watchdog: Running · restarts: 0 · last heard under a minute ago")
        self.assertTrue(can_stop)

    def test_finished_session_has_no_stop_and_no_heartbeat(self):
        headline, facts, can_stop = session_view(row(status="finished", trainer="Stopped", watchdog="Stopped", restarts=3, heard=4000.0))
        self.assertTrue(headline.startswith("Finished"))
        self.assertNotIn("last heard", facts)
        self.assertFalse(can_stop)

    def test_heard_long_ago_drops_the_about(self):
        self.assertIn("last heard 1.7 hours ago", session_view(row(heard=6000.0))[1])

    def test_when_shows_local_time_or_the_raw_stamp(self):
        self.assertRegex(when("2026-10-15T08:30:00+00:00"), r"^\d\d [A-Z][a-z]{2} \d\d:\d\d$")
        self.assertEqual((when("garbage"), when(None)), ("garbage", ""))

    def test_warns_only_when_a_passive_session_and_a_queue_job_both_train(self):
        busy, idle = QueueState(job=Job("1", "pretrain")), QueueState()
        view = {"sessions": [row()], "signin": True}
        self.assertIn("share this computer", competes(view, busy))
        self.assertEqual(competes(view, idle), "")
        self.assertEqual(competes(view, QueueState(job=Job("1", "pretrain"), paused=True)), "")
        self.assertEqual(competes({"sessions": [row(status="paused")], "signin": False}, busy), "")

    def test_fake_services_stop_and_remove_sign_in(self):
        f = FakeServices()
        self.assertEqual(f.passive(), {"sessions": [], "signin": False})
        f.passive_sessions, f.passive_signin = [row()], True
        self.assertEqual(f.stop_passive("p1"), "stopped")
        f.remove_passive_signin()
        view = f.passive()
        self.assertEqual((view["sessions"][0]["status"], view["signin"]), ("stopped", False))


if __name__ == "__main__":
    unittest.main()
