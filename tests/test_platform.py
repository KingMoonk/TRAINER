"""studio_platform: the process and power helpers the queue, the worker, the cluster and passive training's watchdog share."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import studio_platform as watch

REPO = Path(__file__).resolve().parent.parent


AC = "Now drawing from 'AC Power'\n -InternalBattery-0 (id=1)\t100%; charged; 0:00 remaining present: true\n"
BATTERY = "Now drawing from 'Battery Power'\n -InternalBattery-0 (id=1)\t80%; discharging; 3:10 remaining present: true\n"


class PowerTests(unittest.TestCase):
    def test_on_battery_reads_pmset(self):
        run = lambda out: MagicMock(return_value=SimpleNamespace(stdout=out))  # noqa: E731
        self.assertTrue(watch.on_battery(platform="darwin", run=run(BATTERY)))
        self.assertFalse(watch.on_battery(platform="darwin", run=run(AC)))
        self.assertFalse(watch.on_battery(platform="darwin", run=run("")))
        self.assertFalse(watch.on_battery(platform="darwin", run=MagicMock(side_effect=FileNotFoundError)))
        self.assertFalse(watch.on_battery(platform="freebsd", run=MagicMock(side_effect=AssertionError)))

    def test_on_battery_linux_reads_the_power_supplies(self):
        with tempfile.TemporaryDirectory() as tmp:
            supplies = Path(tmp)
            self.assertFalse(watch.on_battery(platform="linux", supplies=supplies))  # a desktop: nothing listed
            for name, values in (("AC", {"type": "Mains", "online": "1"}), ("BAT0", {"type": "Battery"})):
                (supplies / name).mkdir()
                for key, value in values.items():
                    (supplies / name / key).write_text(value + "\n")
            self.assertFalse(watch.on_battery(platform="linux", supplies=supplies))
            (supplies / "AC" / "online").write_text("0\n")
            self.assertTrue(watch.on_battery(platform="linux", supplies=supplies))

    def test_on_battery_windows_reads_the_line_status(self):
        import ctypes
        for line, expected in ((0, True), (1, False)):
            def fill(ref, line=line):
                ref._obj.ACLineStatus = line
                return 1
            fake = SimpleNamespace(kernel32=SimpleNamespace(GetSystemPowerStatus=fill))
            with patch.object(ctypes, "windll", fake, create=True):
                self.assertEqual(watch.on_battery(platform="win32"), expected)

    def test_keep_awake_macos_watches_the_pid(self):
        popen = MagicMock()
        self.assertIs(watch.keep_awake(123, platform="darwin", popen=popen), popen.return_value)
        self.assertEqual(popen.call_args[0][0], ["caffeinate", "-i", "-w", "123"])
        self.assertIsNone(watch.keep_awake(123, platform="darwin", popen=MagicMock(side_effect=FileNotFoundError)))

    def test_keep_awake_windows_sets_the_execution_state(self):
        import ctypes
        call = MagicMock()
        with patch.object(ctypes, "windll", SimpleNamespace(kernel32=SimpleNamespace(SetThreadExecutionState=call)), create=True):
            self.assertTrue(watch.keep_awake(123, platform="win32"))
        call.assert_called_once_with(0x80000001)

    def test_keep_awake_linux_uses_systemd_inhibit_and_release_undoes_it(self):
        popen = MagicMock()
        handle = watch.keep_awake(123, platform="linux", popen=popen)
        self.assertEqual(popen.call_args[0][0][0], "systemd-inhibit")
        self.assertIn("--pid=123", popen.call_args[0][0])
        watch.release_awake(handle, platform="linux")
        handle.terminate.assert_called_once()
        self.assertIsNone(watch.keep_awake(123, platform="linux", popen=MagicMock(side_effect=FileNotFoundError)))

    def test_release_awake_windows_clears_the_execution_state(self):
        import ctypes
        call = MagicMock()
        with patch.object(ctypes, "windll", SimpleNamespace(kernel32=SimpleNamespace(SetThreadExecutionState=call)), create=True):
            watch.release_awake(True, platform="win32")
        call.assert_called_once_with(0x80000000)

    def test_keep_awake_elsewhere_does_nothing(self):
        popen = MagicMock()
        self.assertIsNone(watch.keep_awake(1, platform="freebsd", popen=popen))
        popen.assert_not_called()


class ProcessTests(unittest.TestCase):
    def test_ours_compares_the_command_line(self):
        line = "/usr/bin/python -m studio_passive run passive_1 --root /tmp/w"
        self.assertTrue(watch.ours(5, "studio_passive", "passive_1", read=lambda pid: line))
        self.assertFalse(watch.ours(5, "studio_passive", "passive_2", read=lambda pid: line))
        self.assertFalse(watch.ours(5, "studio_passive", read=lambda pid: ""))  # no such process
        self.assertTrue(watch.ours(5, "studio_passive", read=lambda pid: None))  # unreadable: trust it
        self.assertTrue(watch.command_line(os.getpid()))

    def test_alive(self):
        self.assertTrue(watch.alive(os.getpid()))
        for pid in (None, 0, -1):
            self.assertFalse(watch.alive(pid))
        child = subprocess.Popen([sys.executable, "-c", "pass"])
        child.wait()
        self.assertFalse(watch.alive(child.pid))

    def test_terminate_escalates_to_kill_after_the_grace(self):
        if sys.platform == "win32":
            self.skipTest("POSIX signals")
        with patch("os.kill") as kill:
            watch.terminate(77, is_alive=lambda pid: True, sleep=lambda s: None, grace=0.3)
        self.assertEqual([c[0][1] for c in kill.call_args_list], [15, 9])
        with patch("os.kill") as kill:
            watch.terminate(77, is_alive=lambda pid: False, sleep=lambda s: None)
        self.assertEqual(kill.call_count, 1)

    def test_terminate_windows_uses_taskkill(self):
        run = MagicMock()
        self.assertTrue(watch.terminate(77, run=run, platform="win32", is_alive=lambda pid: False))
        self.assertEqual(run.call_args[0][0], ["taskkill", "/PID", "77", "/T", "/F"])

    def test_spawn_reaps_finished_children_and_restarts_a_huge_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "trainer.log"
            log.write_bytes(b"x" * (watch.LOG_LIMIT + 1))
            first = watch.spawn([sys.executable, "-c", "print('hello')"], log, REPO)
            watch._children[first].wait(20)
            self.assertEqual(log.read_text().strip(), "hello")
            second = watch.spawn([sys.executable, "-c", "pass"], log, REPO)
            watch._children[second].wait(20)
            self.assertNotIn(first, watch._children)

    def test_windows_reads_a_command_line_once_per_process(self):
        run = MagicMock(return_value=SimpleNamespace(stdout="python -m studio_passive_watch\n"))
        with patch.object(sys, "platform", "win32"), patch.object(watch, "_created", lambda pid: 1234), patch.object(watch.subprocess, "run", run), \
                patch.object(watch.subprocess, "CREATE_NO_WINDOW", 0, create=True), patch.dict(watch._lines, clear=True):
            self.assertEqual([watch.command_line(9) for _ in range(3)], ["python -m studio_passive_watch"] * 3)
        run.assert_called_once()



class HomeTests(unittest.TestCase):
    def test_the_helpers_import_from_their_new_modules(self):
        import studio_memory
        import studio_passive_watch
        for name in ("on_battery", "keep_awake", "release_awake", "alive", "terminate", "spawn", "ours", "command_line"):
            self.assertIs(getattr(studio_passive_watch, name), getattr(watch, name), name)  # the watchdog re-exports them; #486 removes it
        import studio_passive
        for name in ("process_memory", "total_memory"):
            self.assertTrue(callable(getattr(studio_memory, name)))
            self.assertTrue(callable(getattr(studio_passive, name)))
        self.assertGreater(studio_memory.total_memory(), studio_memory.process_memory())


if __name__ == "__main__":
    unittest.main()
