"""When the collector may run, and starting the background service from the app (no process is started)."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from studio_collect import config, idle, service
from studio_collect.store import State
from tests.collect_fakes import idle_probes


class Gate(unittest.TestCase):
    def reason(self, overrides, **probe):
        base = {"enabled": True, "idle": {"unmetered_confirmed": True}}
        with tempfile.TemporaryDirectory() as root:
            return idle.reason_not_to_run(config.merge({**base, **overrides}), root, idle_probes(**probe))

    def test_runs_when_idle_on_mains_and_unmetered(self):
        self.assertEqual(self.reason({}), "")

    def test_off_and_paused(self):
        self.assertEqual(self.reason({"enabled": False}), "Collection is off.")
        self.assertEqual(self.reason({"paused": True}), "Paused.")

    def test_each_condition_can_be_turned_off_by_the_person(self):
        self.assertIn("mains", self.reason({}, on_battery=lambda: True))
        self.assertEqual(self.reason({"idle": {"only_on_mains": False, "unmetered_confirmed": True}}, on_battery=lambda: True), "")
        self.assertIn("busy", self.reason({}, cpu=lambda: 90.0))
        self.assertEqual(self.reason({"idle": {"only_when_idle": False, "unmetered_confirmed": True}}, cpu=lambda: 90.0), "")

    def test_an_unknown_network_needs_the_person_to_confirm_it_is_unmetered(self):
        self.assertIn("confirm", self.reason({"idle": {"unmetered_confirmed": False}}, metered=lambda: None))
        self.assertEqual(self.reason({}, metered=lambda: None), "")
        self.assertIn("metered", self.reason({}, metered=lambda: True))     # the OS saying metered wins over the person's confirmation


class Probing(unittest.TestCase):
    def test_networkmanager_and_windows_answers(self):
        run = lambda out: (lambda *a, **k: SimpleNamespace(stdout=out))  # noqa: E731
        self.assertTrue(idle.network_metered(platform="linux", run=run("GENERAL.METERED:yes\n")))
        self.assertFalse(idle.network_metered(platform="linux", run=run("GENERAL.METERED:guess-no\nGENERAL.METERED:unknown\n")))
        self.assertIsNone(idle.network_metered(platform="linux", run=run("GENERAL.METERED:unknown\n")))
        self.assertTrue(idle.network_metered(platform="win32", run=run("2\n")))
        self.assertFalse(idle.network_metered(platform="win32", run=run("1\n")))
        self.assertIsNone(idle.network_metered(platform="darwin"))

    def test_a_running_queue_job_makes_the_machine_busy(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertFalse(idle.queue_busy(root))
            (Path(root) / "data").mkdir()
            (Path(root) / "data" / "queue.json").write_text(json.dumps({"jobs": [{"status": "done"}, {"status": "running"}]}))
            self.assertTrue(idle.queue_busy(root))


class Starting(unittest.TestCase):
    def test_it_starts_only_when_wanted_and_not_twice(self):
        started = []
        spawn = lambda command, log, cwd: started.append(command) or 4242  # noqa: E731
        with tempfile.TemporaryDirectory() as root:
            self.assertIsNone(service.ensure_running(root, spawn=spawn))
            config.save(root, {"enabled": True})
            self.assertEqual(service.ensure_running(root, spawn=spawn, alive=lambda p: False), 4242)
            self.assertIn("studio_collect", started[0])
            self.assertEqual(State(root).data["pid"], 4242)
            self.assertEqual(service.ensure_running(root, spawn=spawn, alive=lambda p: True, ours=lambda *a: True), 4242)
            self.assertEqual(len(started), 1)


if __name__ == "__main__":
    unittest.main()
