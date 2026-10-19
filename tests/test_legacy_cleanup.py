import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import studio_legacy_cleanup as cleanup


class Run:
    def __init__(self, stdout=""):
        self.calls, self.stdout = [], stdout

    def __call__(self, cmd, **kw):
        self.calls.append(cmd)
        return SimpleNamespace(stdout=self.stdout, returncode=0)


class StaleSigninTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_macos_agents_are_unloaded_and_deleted_once(self):
        agents = self.home / "Library" / "LaunchAgents"
        agents.mkdir(parents=True)
        (agents / "com.groundwork.passive.ab12cd34.plist").write_text("x")
        other = agents / "com.example.other.plist"
        other.write_text("x")
        run = Run()
        self.assertEqual(len(cleanup.remove_stale_entries(home=self.home, run=run, platform="darwin")), 1)
        self.assertTrue(other.exists())
        self.assertEqual(run.calls[0][:2], ["launchctl", "bootout"])
        self.assertEqual(cleanup.remove_stale_entries(home=self.home, run=run, platform="darwin"), [])

    def test_linux_autostart_files_are_deleted(self):
        auto = self.home / ".config" / "autostart"
        auto.mkdir(parents=True)
        (auto / "groundwork-passive-ab12cd34.desktop").write_text("x")
        keep = auto / "other.desktop"
        keep.write_text("x")
        self.assertEqual(len(cleanup.remove_stale_entries(home=self.home, run=Run(), platform="linux")), 1)
        self.assertTrue(keep.exists())

    def test_windows_tasks_and_scripts_are_deleted(self):
        data = self.home / "ws" / "data"
        data.mkdir(parents=True)
        (data / "passive-watchdog-ab12cd34.cmd").write_text("x")
        run = Run('"\\GroundworkPassive-ab12cd34","N/A","Ready"\n"\\Other","N/A","Ready"\n')
        self.assertEqual(cleanup.remove_stale_entries(self.home / "ws", home=self.home, run=run, platform="win32"), ["GroundworkPassive-ab12cd34"])
        self.assertIn(["schtasks", "/Delete", "/TN", "GroundworkPassive-ab12cd34", "/F"], run.calls)
        self.assertEqual(list(data.iterdir()), [])

    def test_errors_are_swallowed(self):
        def boom(*a, **k):
            raise OSError("no launchctl")
        agents = self.home / "Library" / "LaunchAgents"
        agents.mkdir(parents=True)
        (agents / "com.groundwork.passive.ab12cd34.plist").write_text("x")
        with self.assertLogs("studio_legacy_cleanup", "WARNING"):
            cleanup.remove_stale_entries(home=self.home, run=boom, platform="darwin")
        self.assertEqual(cleanup.remove_stale_entries(home=self.home, run=boom, platform="win32"), [])


if __name__ == "__main__":
    unittest.main()
