"""Passive training watchdog: one pass of check() with fake processes and clock, sign-in install, keep-awake, battery, and a real detached spawn."""
import json
import os
import plistlib
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import studio_passive_watch as watch
from studio_passive import PassiveStore, merge_plan

NOW = 100000.0


class Rig:
    """A session in a temp workspace plus fake spawn / alive / kill / battery."""

    def __init__(self, test, **limits):
        tmp = tempfile.TemporaryDirectory()
        test.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.store = PassiveStore(self.root)
        self.id = self.store.create(merge_plan({"limits": limits}))
        self.t, self.living, self.spawned, self.killed, self.battery = NOW, set(), [], [], False
        self.strangers = set()  # live pids that belong to some other program

    def spawn(self, command, log, cwd):
        self.spawned.append((command, Path(log), cwd))
        self.living.add(5000 + len(self.spawned))
        return 5000 + len(self.spawned)

    def kill(self, pid):
        self.killed.append(pid)
        self.living.discard(pid)

    def check(self, **options):
        return watch.check(self.store, now=lambda: self.t, spawn=self.spawn, alive=lambda pid: pid in self.living, ours=lambda pid, *needles: pid not in self.strangers,
                           kill=options.pop("kill", self.kill), on_battery=lambda: self.battery, **options)

    def row(self):
        return self.store.get(self.id)

    def set(self, **columns):
        self.store.update(self.id, **columns)

    def memo(self):
        return json.loads((self.store.directory(self.id) / "watchdog.json").read_text())

    def types(self):
        return [e["type"] for e in self.store.journal(self.id)]


class CheckTests(unittest.TestCase):
    def test_no_active_session(self):
        rig = Rig(self)
        rig.set(status="finished")
        self.assertEqual(rig.check(), "none")
        self.assertEqual(rig.spawned, [])

    def test_healthy_trainer_is_left_alone(self):
        rig = Rig(self)
        rig.living.add(7)
        rig.set(status="running", trainer_pid=7, heartbeat=NOW - 59)
        self.assertEqual(rig.check(), "ok")
        self.assertEqual((rig.spawned, rig.killed), ([], []))

    def test_first_start_of_a_queued_session_is_not_a_restart(self):
        rig = Rig(self)
        self.assertEqual(rig.check(), "started")
        command, log, cwd = rig.spawned[0]
        self.assertEqual(command[1:], ["-m", "studio_passive", "run", rig.id, "--root", str(rig.root)])
        self.assertEqual((log, cwd), (rig.store.directory(rig.id) / "trainer.log", watch.REPO))
        self.assertTrue((cwd / "studio_passive.py").exists())
        self.assertEqual((rig.row()["trainer_pid"], rig.row()["restarts"], rig.types()), (5001, [], []))

    def test_dead_trainer_is_restarted_and_recorded(self):
        rig = Rig(self)
        rig.set(status="running", trainer_pid=7, heartbeat=NOW - 5)
        self.assertEqual(rig.check(), "restarted")
        row = rig.row()
        self.assertEqual((row["trainer_pid"], row["restarts"], row["status"], rig.killed), (5001, [NOW], "running", []))
        event = rig.store.journal(rig.id)[0]
        self.assertEqual((event["type"], event["reason"], event["pid"]), ("restart", "stopped", 5001))

    def test_silent_live_trainer_is_terminated_then_restarted(self):
        rig = Rig(self)
        rig.living.add(7)
        rig.set(status="running", trainer_pid=7, heartbeat=NOW - 61)
        self.assertEqual(rig.check(), "restarted")
        self.assertEqual((rig.killed, rig.row()["trainer_pid"], rig.store.journal(rig.id)[0]["reason"]), ([7], 5001, "silent"))

    def test_a_just_spawned_trainer_gets_time_to_beat(self):
        rig = Rig(self)
        rig.set(status="running", trainer_pid=7, heartbeat=NOW - 500)
        rig.living.add(7)
        self.assertEqual(rig.check(), "restarted")
        for later in (1, 30, 59):  # the heartbeat column still holds the old value
            rig.t = NOW + later
            self.assertEqual(rig.check(), "ok", later)
        self.assertEqual((len(rig.spawned), rig.killed, len(rig.row()["restarts"])), (1, [7], 1))
        rig.t = NOW + 61  # it never beat: now it is silent
        self.assertEqual(rig.check(), "restarted")
        self.assertEqual((len(rig.spawned), rig.killed[-1]), (2, 5001))

    def test_a_first_start_that_never_beats_gets_the_same_grace(self):
        rig = Rig(self)
        self.assertEqual(rig.check(), "started")
        rig.t = NOW + 10
        self.assertEqual(rig.check(), "ok")
        self.assertEqual(len(rig.spawned), 1)

    def test_gives_up_after_five_restarts_in_ten_minutes(self):
        rig = Rig(self)
        rig.set(status="running", trainer_pid=7, heartbeat=NOW - 5, restarts=[NOW - 500, NOW - 400, NOW - 300, NOW - 200, NOW - 100])
        self.assertEqual(rig.check(), "gave_up")
        row = rig.row()
        self.assertEqual((row["status"], rig.spawned, rig.store.active()), ("failed", [], None))
        self.assertIn("gave up after 5 restarts in 10 minutes", row["error"])
        self.assertIn("See trainer.log", row["error"])
        self.assertEqual(rig.store.journal(rig.id)[-1]["type"], "error")

    def test_giving_up_keeps_the_trainers_last_error(self):
        rig = Rig(self)
        rig.set(status="running", trainer_pid=7, heartbeat=NOW - 5, error="ValueError: needs two examples", restarts=[NOW - 500, NOW - 400, NOW - 300, NOW - 200, NOW - 100])
        self.assertEqual(rig.check(), "gave_up")
        self.assertIn("The last problem: ValueError: needs two examples", rig.row()["error"])
        self.assertNotIn("See trainer.log", rig.row()["error"])

    def test_gives_up_after_five_slow_restarts_without_a_new_checkpoint(self):
        rig = Rig(self)
        rig.set(status="running", trainer_pid=7, heartbeat=NOW - 5)
        watch._remember(rig.store, rig.id, {"spawned": NOW - 3600, "checkpoints": 0})  # as the first start left it
        words = []
        for i in range(6):  # one crash every 20 minutes: never 5 in 10 minutes
            rig.t = NOW + i * 1200
            rig.living.clear()
            words.append(rig.check())
        self.assertEqual(words, ["restarted"] * 5 + ["gave_up"])
        self.assertIn("5 restarts in a row without a new checkpoint", rig.row()["error"])

    def test_a_new_checkpoint_resets_the_count(self):
        rig = Rig(self)
        rig.set(status="running", trainer_pid=7, heartbeat=NOW - 5)
        for i in range(8):
            rig.t = NOW + i * 1200
            rig.living.clear()
            rig.set(state={"checkpoints": [{"n": n} for n in range(i)]})  # each run saved one more
            self.assertEqual(rig.check(), "restarted")

    def test_a_restart_after_a_reboot_or_sleep_is_not_counted(self):
        rig = Rig(self)
        rig.set(status="running", trainer_pid=7, heartbeat=NOW - 5, restarts=[NOW - 500, NOW - 400, NOW - 300, NOW - 200, NOW - 100])
        self.assertEqual(rig.check(woke=True), "restarted")
        self.assertEqual(len(rig.row()["restarts"]), 5)

    def test_a_trainer_that_outlives_its_kill_is_not_joined_by_a_second(self):
        rig = Rig(self)
        rig.living.add(7)
        rig.set(status="running", trainer_pid=7, heartbeat=NOW - 61)
        self.assertEqual(rig.check(kill=rig.killed.append), "waiting")  # the kill has not landed yet
        self.assertEqual((rig.killed, rig.spawned), ([7], []))

    def test_old_restarts_do_not_count(self):
        rig = Rig(self)
        rig.set(status="running", trainer_pid=7, heartbeat=NOW - 5, restarts=[NOW - 2000, NOW - 1500, NOW - 1000, NOW - 700, NOW - 601, NOW - 100])
        self.assertEqual(rig.check(), "restarted")
        self.assertEqual(len(rig.row()["restarts"]), 7)

    def test_paused_session_whose_trainer_died_is_restarted_still_paused(self):
        rig = Rig(self)
        rig.set(status="paused", trainer_pid=7, heartbeat=NOW - 90)
        self.assertEqual(rig.check(), "restarted")
        self.assertEqual((rig.row()["status"], rig.row()["command"]), ("paused", ""))  # the engine reads status paused and starts paused
        rig.living.add(5001)
        rig.t = NOW + 1
        self.assertEqual(rig.check(), "paused")

    def test_pauses_on_battery_and_resumes_on_mains(self):
        rig = Rig(self)
        rig.living.add(7)
        rig.set(status="running", trainer_pid=7, heartbeat=NOW - 1)
        rig.battery = True
        self.assertEqual(rig.check(), "battery_paused")
        self.assertEqual((rig.row()["command"], rig.memo()["paused"]), ("pause", True))
        rig.set(command="", status="paused")  # the trainer took the command
        self.assertEqual(rig.check(), "paused")
        self.assertEqual(rig.row()["command"], "")
        rig.battery = False
        self.assertEqual(rig.check(), "battery_resumed")
        self.assertEqual((rig.row()["command"], rig.memo()["paused"]), ("resume", False))
        rig.set(command="", status="running")
        self.assertEqual(rig.check(), "ok")
        self.assertEqual(rig.row()["command"], "")

    def test_battery_pause_and_resume_are_journalled_with_their_reason(self):
        rig = Rig(self)
        rig.living.add(7)
        rig.set(status="running", trainer_pid=7, heartbeat=NOW - 1)
        rig.battery = True
        rig.check()
        rig.check()  # nothing new to say while it stays paused on battery
        rig.battery = False
        rig.set(status="paused", command="")
        rig.check()
        self.assertEqual([e["reason"] for e in rig.store.journal(rig.id) if e["type"] == "power"], ["battery_paused", "battery_resumed"])

    def test_mains_returns_before_the_trainer_took_the_pause(self):
        rig = Rig(self)
        rig.living.add(7)
        rig.set(status="running", trainer_pid=7, heartbeat=NOW - 1)
        rig.battery = True
        rig.check()
        rig.battery = False
        self.assertEqual(rig.check(), "battery_resumed")
        self.assertEqual(rig.row()["command"], "resume")

    def test_never_resumes_a_pause_the_user_made(self):
        rig = Rig(self)
        rig.living.add(7)
        rig.set(status="paused", trainer_pid=7, heartbeat=NOW - 1)
        for battery in (True, False, True, False):
            rig.battery = battery
            self.assertEqual(rig.check(), "paused")
            self.assertEqual(rig.row()["command"], "")

    def test_power_never_overwrites_a_stop(self):
        rig = Rig(self)
        rig.living.add(7)
        rig.set(status="running", trainer_pid=7, heartbeat=NOW - 1)
        rig.battery = True
        rig.check()
        rig.set(status="paused", command="stop")  # the user pressed Stop while it was paused on battery
        rig.battery = False
        rig.check()
        self.assertEqual(rig.row()["command"], "stop")

    def test_a_user_resume_on_battery_is_not_fought(self):
        rig = Rig(self)
        rig.living.add(7)
        rig.set(status="running", trainer_pid=7, heartbeat=NOW - 1)
        rig.battery = True
        rig.check()
        rig.set(command="", status="running")  # the user resumed anyway
        self.assertEqual(rig.check(), "ok")
        self.assertEqual(rig.row()["command"], "")

    def test_setting_off_means_no_pause(self):
        rig = Rig(self, pause_on_battery=False)
        rig.living.add(7)
        rig.set(status="running", trainer_pid=7, heartbeat=NOW - 1)
        rig.battery = True
        self.assertEqual(rig.check(), "ok")

    def test_a_recycled_pid_is_a_dead_trainer_and_is_never_killed(self):
        rig = Rig(self)
        rig.living.add(7)
        rig.strangers.add(7)
        rig.set(status="running", trainer_pid=7, heartbeat=NOW - 5)
        self.assertEqual(rig.check(), "restarted")
        self.assertEqual((rig.killed, rig.row()["trainer_pid"], rig.store.journal(rig.id)[0]["reason"]), ([], 5001, "stopped"))

    def test_giving_up_does_not_kill_a_recycled_pid(self):
        rig = Rig(self)
        rig.living.add(7)
        rig.strangers.add(7)
        rig.set(status="running", trainer_pid=7, heartbeat=NOW - 500, restarts=[NOW - i for i in range(1, 6)])
        self.assertEqual(rig.check(), "gave_up")
        self.assertEqual(rig.killed, [])

    def test_ours_compares_the_command_line(self):
        line = "/usr/bin/python -m studio_passive run passive_1 --root /tmp/w"
        self.assertTrue(watch.ours(5, "studio_passive", "passive_1", read=lambda pid: line))
        self.assertFalse(watch.ours(5, "studio_passive", "passive_2", read=lambda pid: line))
        self.assertFalse(watch.ours(5, "studio_passive", read=lambda pid: ""))  # no such process
        self.assertTrue(watch.ours(5, "studio_passive", read=lambda pid: None))  # unreadable: trust it
        self.assertTrue(watch.command_line(os.getpid()))

    def test_status_for_the_table(self):
        rig = Rig(self)
        rig.living.update({7, 8})
        rig.set(status="running", trainer_pid=7, watchdog_pid=8, heartbeat=NOW - 12, restarts=[1.0, 2.0])
        table = lambda: watch.status(rig.store, rig.row(), now=lambda: rig.t, alive=lambda pid: pid in rig.living, ours=lambda *a: True)  # noqa: E731
        self.assertEqual(table(), {"trainer": "Running", "heard": 12, "watchdog": "Running", "restarts": 2})
        rig.t = NOW + 100
        self.assertEqual(table()["trainer"], "Restarting")
        rig.living.clear()
        self.assertEqual((table()["trainer"], table()["watchdog"]), ("Stopped", "Stopped"))
        rig.set(heartbeat=None)
        self.assertIsNone(table()["heard"])


class WatchTests(unittest.TestCase):
    def test_exits_and_uninstalls_when_nothing_is_active(self):
        rig = Rig(self)
        rig.set(status="stopped")
        with patch.object(watch, "uninstall") as uninstall:
            watch.watch(rig.root, sleep=lambda s: self.fail("slept"), me=1)
        uninstall.assert_called_once_with(rig.root)

    def test_loops_until_the_session_ends_and_claims_the_record(self):
        rig = Rig(self)
        passes = []

        def sleep(seconds):
            passes.append(seconds)
            if len(passes) == 2:
                rig.set(status="finished")
        with patch.object(watch, "uninstall"):
            watch.watch(rig.root, poll=3, sleep=sleep, me=4242, spawn=rig.spawn, alive=lambda pid: pid in rig.living, ours=lambda *a: True)
        self.assertEqual((passes, len(rig.spawned)), ([3, 3], 1))
        self.assertEqual(rig.row()["watchdog_pid"], 4242)

    def test_waking_from_sleep_gives_the_trainer_a_fresh_grace_period(self):
        rig = Rig(self)
        rig.living.add(7)
        rig.set(status="running", trainer_pid=7, heartbeat=NOW - 5)
        passes = []

        def sleep(seconds):
            passes.append(seconds)
            rig.t += 3 if len(passes) == 1 else 8 * 3600 if len(passes) == 2 else 0  # the second sleep is the computer sleeping
            if len(passes) == 4:
                rig.set(status="finished")
        with patch.object(watch, "uninstall"):
            watch.watch(rig.root, poll=3, sleep=sleep, me=4242, now=lambda: rig.t, spawn=rig.spawn, alive=lambda pid: pid in rig.living, ours=lambda *a: True, kill=rig.killed.append)
        self.assertEqual((rig.killed, rig.spawned), ([], []))

    def test_a_silent_trainer_is_still_killed_without_a_sleep(self):
        rig = Rig(self)
        rig.living.add(7)
        rig.set(status="running", trainer_pid=7, heartbeat=NOW - 500)
        with patch.object(watch, "uninstall"):
            watch.watch(rig.root, poll=3, sleep=lambda s: rig.set(status="finished"), me=4242, now=lambda: rig.t, spawn=rig.spawn, alive=lambda pid: pid in rig.living, ours=lambda *a: True, kill=rig.killed.append)
        self.assertEqual(rig.killed, [7])

    def test_second_watchdog_exits_at_once(self):
        rig = Rig(self)
        rig.set(watchdog_pid=99)
        rig.living.add(99)
        with patch.object(watch, "uninstall") as uninstall:
            watch.watch(rig.root, sleep=lambda s: self.fail("slept"), me=4242, spawn=rig.spawn, alive=lambda pid: pid in rig.living, ours=lambda *a: True)
        self.assertEqual((rig.row()["watchdog_pid"], rig.spawned), (99, []))
        uninstall.assert_not_called()

    def test_a_recycled_watchdog_pid_is_taken_over(self):
        rig = Rig(self)
        rig.set(watchdog_pid=99)
        rig.living.add(99)
        with patch.object(watch, "uninstall"):
            watch.watch(rig.root, sleep=lambda s: rig.set(status="finished"), me=4242, spawn=rig.spawn, alive=lambda pid: pid in rig.living, ours=lambda *a: False)
        self.assertEqual(rig.row()["watchdog_pid"], 4242)

    def test_a_dead_watchdog_pid_is_taken_over(self):
        rig = Rig(self)
        rig.set(watchdog_pid=99)
        with patch.object(watch, "uninstall"):
            watch.watch(rig.root, sleep=lambda s: rig.set(status="finished"), me=4242, spawn=rig.spawn, alive=lambda pid: pid in rig.living, ours=lambda *a: True)
        self.assertEqual(rig.row()["watchdog_pid"], 4242)

    def test_start_spawns_once(self):
        rig = Rig(self)
        installed = []
        pid = watch.start(rig.root, rig.id, spawn=rig.spawn, install=installed.append, alive=lambda pid: pid in rig.living, ours=lambda *a: True)
        command, log, cwd = rig.spawned[0]
        self.assertEqual((command[1:], log, cwd), (["-m", "studio_passive_watch", "--root", str(rig.root.resolve())], rig.root / "data" / "passive-watchdog.log", watch.REPO))
        self.assertEqual((pid, rig.row()["watchdog_pid"], installed), (5001, 5001, [rig.root]))
        self.assertEqual(watch.start(rig.root, spawn=rig.spawn, install=installed.append, alive=lambda pid: pid in rig.living, ours=lambda *a: True), 5001)
        self.assertEqual(len(rig.spawned), 1)
        rig.living.clear()
        self.assertEqual(watch.start(rig.root, spawn=rig.spawn, install=installed.append, alive=lambda pid: pid in rig.living, ours=lambda *a: True), 5002)

    def test_a_failing_record_does_not_end_the_watchdog(self):
        rig = Rig(self)
        rig.set(status="finished")
        passes = []
        real = PassiveStore.active

        def active(store):
            if not passes:
                raise OSError("disk I/O error")
            return real(store)
        with patch.object(PassiveStore, "active", active), patch.object(watch, "uninstall"), patch("sys.stderr"):
            watch.watch(rig.root, sleep=passes.append, me=1)
        self.assertEqual(len(passes), 1)

    def test_start_without_a_session_says_so(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            watch.start(tmp, spawn=lambda *a: 1, install=lambda r: None)


class InstallTests(unittest.TestCase):
    def test_macos_writes_a_launch_agent_under_the_given_home(self):
        with tempfile.TemporaryDirectory() as home, tempfile.TemporaryDirectory() as root:
            run = MagicMock()
            real = Path.home() / "Library" / "LaunchAgents"
            before = sorted(real.glob("com.groundwork.passive.*")) if real.exists() else []
            path = watch.install(root, home=home, run=run, platform="darwin")
            self.assertTrue(path.startswith(home))
            plist = plistlib.loads(Path(path).read_bytes())
            self.assertEqual(plist["ProgramArguments"], [sys.executable, "-m", "studio_passive_watch", "--root", str(Path(root).resolve())])
            self.assertEqual((plist["WorkingDirectory"], plist["RunAtLoad"], plist["KeepAlive"]), (str(watch.REPO), True, {"SuccessfulExit": False}))
            self.assertTrue(plist["Label"].startswith("com.groundwork.passive.") and Path(path).name == plist["Label"] + ".plist")
            self.assertTrue(plist["StandardOutPath"].startswith(str(Path(root).resolve())))
            run.assert_not_called()
            self.assertEqual(sorted(real.glob("com.groundwork.passive.*")) if real.exists() else [], before)
            watch.uninstall(root, home=home, run=run, platform="darwin")
            self.assertFalse(Path(path).exists())
            watch.uninstall(root, home=home, run=run, platform="darwin")  # already gone is fine

    def test_frozen_build_points_at_its_own_executable(self):
        run = MagicMock()
        with tempfile.TemporaryDirectory() as home, tempfile.TemporaryDirectory() as root, patch.object(sys, "frozen", True, create=True):
            expected = [sys.executable, "--run-module", "studio_passive_watch", "--root", str(Path(root).resolve())]
            self.assertEqual(watch._watch_command(root), expected)
            path = watch.install(root, home=home, run=run, platform="darwin")
            self.assertEqual(plistlib.loads(Path(path).read_bytes())["ProgramArguments"], expected)
            watch.install(root, run=run, platform="win32")
            script = run.call_args[0][0][run.call_args[0][0].index("/TR") + 1].strip('"')
            self.assertIn(subprocess.list2cmdline(expected), Path(script).read_text())

    def test_windows_creates_and_deletes_a_scheduled_task(self):
        run = MagicMock()
        with tempfile.TemporaryDirectory() as root:
            name = watch.install(root, run=run, platform="win32")
            args = run.call_args[0][0]
            self.assertEqual(args[:4], ["schtasks", "/Create", "/TN", name])
            self.assertEqual(args[args.index("/SC") + 1], "ONLOGON")
            self.assertIn("/F", args)
            task = args[args.index("/TR") + 1]
            self.assertTrue(task.endswith('.cmd"') and len(task) < 261)  # schtasks' limit; the long command line lives in the script
            self.assertIn("studio_passive_watch.py", Path(task.strip('"')).read_text())
            self.assertIn(str(Path(root).resolve()), Path(task.strip('"')).read_text())
            watch.uninstall(root, run=run, platform="win32")
            self.assertEqual(run.call_args[0][0], ["schtasks", "/Delete", "/TN", name, "/F"])
            self.assertFalse(Path(task.strip('"')).exists())

    def test_failures_never_raise(self):
        with tempfile.TemporaryDirectory() as root:
            boom = MagicMock(side_effect=subprocess.CalledProcessError(1, "schtasks"))
            self.assertIsNone(watch.install(root, run=boom, platform="win32"))
            watch.uninstall(root, run=boom, platform="win32")
            blocker = Path(root) / "file"
            blocker.write_text("x")
            self.assertIsNone(watch.install(root, home=blocker, platform="darwin"))

    def test_linux_writes_an_autostart_entry_under_the_given_home(self):
        with tempfile.TemporaryDirectory() as home, tempfile.TemporaryDirectory() as root:
            path = Path(watch.install(root + ' "$x', home=home, platform="linux"))
            self.assertEqual(path.parent, Path(home) / ".config" / "autostart")
            text = path.read_text()
            self.assertIn('" "-m" "studio_passive_watch" "--root" "', text)  # the interpreter path is escaped differently on Windows
            self.assertIn('\\\\"\\\\$x"', text)  # a quote and a dollar sign, escaped as the Desktop Entry spec says
            watch.uninstall(root + ' "$x', home=home, platform="linux")
            self.assertFalse(path.exists())

    def test_other_platforms_do_nothing(self):
        run = MagicMock()
        self.assertIsNone(watch.install("x", run=run, platform="freebsd"))
        watch.uninstall("x", run=run, platform="freebsd")
        run.assert_not_called()


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

    def test_real_detached_spawn_runs_logs_and_exits(self):
        from studio_passive import merge_plan as merge
        # Windows CI: something still holds child.log after the child has exited and every assertion passed; not yet explained
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            store = PassiveStore(tmp)
            sid = store.create(merge({"stop": {"duration": 0.0001, "unit": "minutes", "patience_rounds": 0}, "teacher": {"polish": False, "lessons": False}, "plateau": {"checks": 2, "max_steps": 5}}))
            code = ("import sys, studio_passive as p\n"
                    "def train(config, examples, out, emit, stop, pause, init_from=None):\n"
                    "    out.mkdir(parents=True); (out / 'best.npz').write_text('x')\n"
                    "p.run_session(sys.argv[1], sys.argv[2], train=train, evaluate=lambda path, ex: 1.0, build=lambda r, d: [{'n': i} for i in range(5)])\n"
                    "print('child done')\n")
            log = Path(tmp) / "child.log"
            pid = watch.spawn([sys.executable, "-c", code, tmp, sid], log, watch.REPO)
            self.addCleanup(lambda: watch.alive(pid) and os.kill(pid, 9))
            self.assertGreater(pid, 0)
            deadline = time.time() + 120  # a cold torch import on a Windows runner takes most of a minute
            while watch.alive(pid) and time.time() < deadline:
                time.sleep(0.05)
            late = watch.alive(pid)
            if late:
                watch.terminate(pid)  # or Windows cannot delete the log it still holds, which hides this failure
            self.assertFalse(late, log.read_text())
            self.assertIn("child done", log.read_text())
            self.assertEqual(store.get(sid)["status"], "finished")

    def test_spawn_reaps_finished_children_and_restarts_a_huge_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "trainer.log"
            log.write_bytes(b"x" * (watch.LOG_LIMIT + 1))
            first = watch.spawn([sys.executable, "-c", "print('hello')"], log, watch.REPO)
            watch._children[first].wait(20)
            self.assertEqual(log.read_text().strip(), "hello")
            second = watch.spawn([sys.executable, "-c", "pass"], log, watch.REPO)
            watch._children[second].wait(20)
            self.assertNotIn(first, watch._children)

    def test_windows_reads_a_command_line_once_per_process(self):
        run = MagicMock(return_value=SimpleNamespace(stdout="python -m studio_passive_watch\n"))
        with patch.object(sys, "platform", "win32"), patch.object(watch, "_created", lambda pid: 1234), patch.object(watch.subprocess, "run", run), \
                patch.object(watch.subprocess, "CREATE_NO_WINDOW", 0, create=True), patch.dict(watch._lines, clear=True):
            self.assertEqual([watch.command_line(9) for _ in range(3)], ["python -m studio_passive_watch"] * 3)
        run.assert_called_once()

    def test_cli_exits_at_once_without_a_session(self):
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as home:
            env = {**os.environ, "HOME": home, "USERPROFILE": home}
            done = subprocess.run([sys.executable, "-m", "studio_passive_watch", "--root", root], cwd=watch.REPO, env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual((done.returncode, done.stderr), (0, ""))


if __name__ == "__main__":
    unittest.main()
