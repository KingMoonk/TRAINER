import tempfile
import threading
import time
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

import studio_queue
from studio_queue import Job, Plan, Runner

ANY = Plan(hours="any", idle_only=False, pause_on_battery=False)


def wait(condition, timeout=5):
    end = time.time() + timeout
    while time.time() < end:
        if condition():
            return True
        time.sleep(.01)
    return False


class Fake:
    """Handlers that log what ran; a 'gate' event holds a job open until the test releases it."""
    def __init__(self):
        self.ran, self.gate, self.started = [], threading.Event(), threading.Event()
        self.steps = 0

    def job(self, job, services, emit, stop, pause):
        self.ran.append(job.label)
        self.started.set()
        while not (self.gate.is_set() or stop.is_set()):
            while pause.is_set() and not stop.is_set():
                time.sleep(.005)
            self.steps += 1
            emit({"type": "metric", "step": self.steps, "total": 1000, "lr": .001})
            time.sleep(.005)

    def quick(self, job, services, emit, stop, pause):
        self.ran.append(job.label)

    def cooldown(self, job, services, emit, stop, pause):
        self.ran.append("cooldown of " + job.params["parent"])


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "data" / "queue.json"
        self.fake = Fake()
        self.handlers = {"pretrain": self.fake.job, "quicktest": self.fake.quick, "restore_answers": self.fake.quick,
                         "checkpoint_now": self.fake.cooldown}

    def runner(self, plan=ANY, **kw):
        r = Runner(self.path, self.handlers, **{"tick": .01, **kw})
        r.set_plan(plan)
        self.addCleanup(r.shutdown)
        return r

    def job(self, label, kind="quicktest", **params):
        return Job(kind, label, params=params)

    def test_sequential_and_persisted_done(self):
        r = self.runner()
        for name in "abc":
            r.enqueue(self.job(name))
        r.start()
        self.assertTrue(wait(lambda: self.fake.ran == ["a", "b", "c"] and r.state().job is None and not r.state().up_next))
        again = Runner(self.path, self.handlers)
        self.assertEqual([j.status for j in again.jobs], ["done"] * 3)

    def test_persistence_across_restart_requeues_interrupted_job(self):
        r = self.runner()
        r.enqueue(self.job("long", "pretrain"))
        r.enqueue(self.job("next"))
        r.start()
        self.assertTrue(self.fake.started.wait(5))
        r.shutdown()
        again = Runner(self.path, self.handlers, tick=.01)
        self.assertEqual([j.label for j in again.state().up_next], ["long", "next"])
        self.assertEqual(again.plan, ANY)
        self.fake.gate.set()
        again.start()
        self.addCleanup(again.shutdown)
        self.assertTrue(wait(lambda: not again.state().up_next and again.state().job is None))
        self.assertEqual(self.fake.ran[-2:], ["long", "next"])

    def test_pause_resume(self):
        r = self.runner()
        r.enqueue(self.job("long", "pretrain"))
        r.start()
        self.assertTrue(wait(lambda: self.fake.steps > 3))
        r.pause()
        time.sleep(.05)
        frozen = self.fake.steps
        time.sleep(.1)
        self.assertEqual(self.fake.steps, frozen)
        self.assertTrue(r.state().paused)
        self.assertEqual(r.state().total, 1000)
        r.resume()
        self.assertTrue(wait(lambda: self.fake.steps > frozen + 3))
        self.fake.gate.set()

    def test_pause_persists_and_is_reported_when_idle(self):
        r = self.runner()
        events = []
        r.subscribe(lambda kind, payload: events.append((kind, payload)))
        r.pause()  # nothing is running
        self.assertIsNone(r.state().job)
        self.assertTrue(r.state().paused)
        self.assertIn(("queue", {"paused": True}), events)
        r.enqueue(self.job("a"))
        r.start()
        time.sleep(.15)
        self.assertEqual(self.fake.ran, [])  # a paused queue starts nothing
        again = Runner(self.path, self.handlers, tick=.01)  # an app restart
        self.assertTrue(again.state().paused)
        self.assertTrue(again.pause_event.is_set())
        events.clear()
        r.resume()
        self.assertIn(("queue", {"paused": False}), events)
        self.assertTrue(wait(lambda: self.fake.ran == ["a"]))
        self.assertFalse(Runner(self.path, self.handlers).state().paused)

    def test_move_up_and_remove(self):
        r = self.runner()
        a, b, c = (r.enqueue(self.job(n)) for n in "abc")
        r.move_up(c.id)
        self.assertEqual([j.label for j in r.state().up_next], ["a", "c", "b"])
        r.move_up(a.id)  # already first
        r.remove(c.id)
        self.assertEqual([j.label for j in r.state().up_next], ["a", "b"])
        self.assertEqual([j.label for j in Runner(self.path, self.handlers).jobs], ["a", "b"])

    def test_move_to_any_place_and_hold_skips_until_released(self):
        r = self.runner()
        a, b, c = (r.enqueue(self.job(n)) for n in "abc")
        r.move(c.id, 0)
        self.assertEqual([j.label for j in r.state().up_next], ["c", "a", "b"])
        r.move(c.id, 9)  # past the end: last
        self.assertEqual([j.label for j in r.state().up_next], ["a", "b", "c"])
        r.hold(a.id)
        self.assertEqual([(j.label, j.status) for j in Runner(self.path, self.handlers).jobs][0], ("a", "held"))  # kept across restart
        r.start()
        self.assertTrue(wait(lambda: self.fake.ran == ["b", "c"]))
        self.assertEqual([j.label for j in r.state().up_next], ["a"])  # still listed, still in place
        r.hold(a.id, False)
        self.assertTrue(wait(lambda: self.fake.ran == ["b", "c", "a"]))

    def test_hour_window_blocks_start(self):
        plan = Plan(hours="set", start_hour=22, end_hour=7, idle_only=False, pause_on_battery=False)
        noon = lambda: datetime(2026, 1, 1, 12)  # noqa: E731
        r = self.runner(plan, clock=noon)
        r.enqueue(self.job("a"))
        r.start()
        time.sleep(.2)
        self.assertEqual(self.fake.ran, [])
        self.assertEqual(r.state().blocked, "outside hours")
        r.clock = lambda: datetime(2026, 1, 1, 23)
        self.assertTrue(wait(lambda: self.fake.ran == ["a"]))

    def test_gate_carries_the_next_start(self):
        plan = Plan(hours="nights", idle_only=False, pause_on_battery=False)
        r = self.runner(plan, clock=lambda: datetime(2026, 1, 1, 12))
        r.enqueue(self.job("a"))
        r.start()
        self.assertTrue(wait(lambda: r.state().gate.reason == "outside hours"))
        gate = r.state().gate
        self.assertEqual((gate.resumes_at, gate.supported, r.state().blocked), (datetime(2026, 1, 1, 22), True, "outside hours"))
        r.clock = lambda: datetime(2026, 1, 1, 23)
        self.assertTrue(wait(lambda: self.fake.ran == ["a"]))
        self.assertEqual(r.state().gate, studio_queue.Gate())

    def test_unsupported_idle_detection_is_reported_and_does_not_block(self):
        with mock.patch.object(studio_queue, "idle_seconds", lambda: None):
            r = self.runner(Plan(hours="any", idle_only=True, pause_on_battery=False))
            r.enqueue(self.job("a"))
            r.start()
            self.assertTrue(wait(lambda: self.fake.ran == ["a"]))  # the setting is ignored, not silently "idle"
            self.assertEqual(r.state().gate, studio_queue.Gate(None, None, False))
            r.set_plan(Plan(hours="any", idle_only=False, pause_on_battery=False))
            self.assertTrue(wait(lambda: r.state().gate.supported))

    def test_next_start_across_the_midnight_wrap(self):
        nights, evening = Plan(hours="nights"), Plan(hours="set", start_hour=22, end_hour=7)
        for plan in (nights, evening):
            self.assertIsNone(studio_queue.next_start(Plan(hours="any"), datetime(2026, 1, 1, 12)))
            self.assertEqual(studio_queue.next_start(plan, datetime(2026, 1, 1, 12, 34, 56)), datetime(2026, 1, 1, 22))
            self.assertEqual(studio_queue.next_start(plan, datetime(2026, 1, 1, 7)), datetime(2026, 1, 1, 22))  # window ends at 7:00 sharp
            inside = datetime(2026, 1, 1, 23, 30)
            self.assertEqual(studio_queue.next_start(plan, inside), inside)  # already open
            inside = datetime(2026, 1, 2, 3)  # after midnight, still the same window
            self.assertEqual(studio_queue.next_start(plan, inside), inside)
        day = Plan(hours="set", start_hour=9, end_hour=17)
        self.assertEqual(studio_queue.next_start(day, datetime(2026, 1, 1, 23)), datetime(2026, 1, 2, 9))  # opens tomorrow
        self.assertEqual(studio_queue.next_start(day, datetime(2026, 1, 1, 6)), datetime(2026, 1, 1, 9))

    def test_idle_seconds_by_platform(self):
        with mock.patch.object(studio_queue.sys, "platform", "linux"):
            self.assertIsNone(studio_queue.idle_seconds())
        windll = mock.Mock()
        windll.user32.GetLastInputInfo.side_effect = lambda info: setattr(info._obj, "dwTime", 1000) or 1
        windll.kernel32.GetTickCount.return_value = 4500
        with mock.patch.object(studio_queue.sys, "platform", "win32"), mock.patch("ctypes.windll", windll, create=True):
            self.assertEqual(studio_queue.idle_seconds(), 3.5)
            windll.kernel32.GetTickCount.return_value = 500  # the 32-bit tick counter wrapped past the last input
            self.assertEqual(studio_queue.idle_seconds(), (500 - 1000 + 2 ** 32) / 1000)
            windll.user32.GetLastInputInfo.side_effect = None
            windll.user32.GetLastInputInfo.return_value = 0
            self.assertIsNone(studio_queue.idle_seconds())

    def test_battery_and_idle_pause_then_resume(self):
        plan = Plan(hours="any", idle_only=True, idle_after_s=30, pause_on_battery=True)
        battery, idle = {"on": False}, {"s": 120.}
        with mock.patch.object(studio_queue.platform_helpers, "on_battery", lambda: battery["on"]), \
                mock.patch.object(studio_queue, "idle_seconds", lambda: idle["s"]):
            r = self.runner(plan)
            r.enqueue(self.job("long", "pretrain"))
            r.start()
            self.assertTrue(wait(lambda: self.fake.steps > 3))
            battery["on"] = True
            self.assertTrue(wait(lambda: r.state().blocked == "on battery"))
            time.sleep(.05)
            frozen = self.fake.steps
            time.sleep(.1)
            self.assertEqual(self.fake.steps, frozen)
            battery["on"] = False
            idle["s"] = 1.
            self.assertTrue(wait(lambda: r.state().blocked == "computer in use"))
            idle["s"] = 120.
            self.assertTrue(wait(lambda: self.fake.steps > frozen + 3))
            self.fake.gate.set()

    def test_gate_is_read_once_per_tick_not_per_pass(self):
        reads = []

        def idle():
            reads.append(1)
            return 120.
        with mock.patch.object(studio_queue, "idle_seconds", idle):
            r = Runner(self.path, self.handlers, tick=.3)
            r.set_plan(Plan(hours="any", idle_only=True, pause_on_battery=False))
            self.addCleanup(r.shutdown)
            r.enqueue(self.job("long", "pretrain"))
            r.start()
            self.assertTrue(wait(lambda: self.fake.steps > 3))
            time.sleep(.5)
            self.assertLessEqual(len(reads), 4)  # the old 50 ms poll made this about 20 in the same time
            self.fake.gate.set()

    def test_enqueue_and_resume_wake_the_runner_at_once(self):
        r = self.runner(tick=30)  # no polling to fall back on within the test
        r.start()
        time.sleep(.05)
        r.enqueue(self.job("a"))
        self.assertTrue(wait(lambda: self.fake.ran == ["a"], 2))
        r.pause()
        r.enqueue(self.job("b"))
        time.sleep(.1)
        self.assertEqual(self.fake.ran, ["a"])
        r.resume()
        self.assertTrue(wait(lambda: self.fake.ran == ["a", "b"], 2))

    def test_plan_change_rereads_the_gate_at_once(self):
        r = self.runner(Plan(hours="set", start_hour=22, end_hour=7, idle_only=False, pause_on_battery=False),
                        clock=lambda: datetime(2026, 1, 1, 12), tick=30)
        r.enqueue(self.job("a"))
        r.start()
        self.assertTrue(wait(lambda: r.state().blocked == "outside hours", 2))
        r.set_plan(ANY)
        self.assertTrue(wait(lambda: self.fake.ran == ["a"], 2))

    def test_state_never_reads_the_gate_on_the_calling_thread(self):
        caller, seen = threading.get_ident(), []
        with mock.patch.object(studio_queue, "idle_seconds", lambda: seen.append(threading.get_ident()) or 120.):
            r = Runner(self.path, self.handlers)
            r.set_plan(Plan(hours="any", idle_only=True, pause_on_battery=False))
            r.state()  # not started: a short-lived thread reads it
            self.assertTrue(wait(lambda: seen))
            r.start()
            self.addCleanup(r.shutdown)
            r.state()
        self.assertNotIn(caller, seen)

    def test_checkpoint_now_keeps_main_running(self):
        r = self.runner()
        main = r.enqueue(self.job("long", "pretrain"))
        events = []
        r.subscribe(lambda kind, payload: events.append((kind, payload)))
        r.start()
        self.assertTrue(wait(lambda: self.fake.steps > 2))
        self.assertTrue(r.checkpoint_now())
        self.assertTrue(wait(lambda: ("checkpoint", {"job": main.id, "phase": "done"}) in events))
        self.assertIn("cooldown of " + main.id, self.fake.ran)
        before = self.fake.steps
        self.assertTrue(wait(lambda: self.fake.steps > before + 3))
        self.assertEqual(r.state().job.id, main.id)
        self.assertEqual(main.status, "running")
        self.fake.gate.set()

    def test_stop_after_cooldown_then_stops(self):
        r = self.runner()
        main = r.enqueue(self.job("long", "pretrain"))
        r.start()
        self.assertTrue(wait(lambda: self.fake.steps > 2))
        r.stop_after_cooldown()
        self.assertTrue(wait(lambda: main.status == "stopped"))
        self.assertIn("cooldown of " + main.id, self.fake.ran)

    def test_user_text_pretrain_appends_restore_answers(self):
        r = self.runner()
        r.enqueue(Job("pretrain", "night", params={"user_text": True, "previous_best": "best.npz"}))
        self.fake.gate.set()
        r.start()
        self.assertTrue(wait(lambda: "Restore how it answers" in self.fake.ran))
        done = [j for j in r.jobs if j.kind == "restore_answers"][0]
        self.assertEqual(done.params, {"round": 3, "teacher": "best.npz", "max_batch_size": done.params["max_batch_size"]})

    def test_memory_cap_is_handed_to_jobs(self):
        seen = []
        self.handlers["quicktest"] = lambda job, *a: seen.append(job.params["max_batch_size"])
        r = self.runner(Plan(hours="any", idle_only=False, pause_on_battery=False, memory="half"))
        r.enqueue(self.job("a"))
        r.start()
        self.assertTrue(wait(lambda: seen))
        self.assertEqual(seen[0], studio_queue.batch_cap(r.plan))

    def test_failed_job_does_not_block_queue(self):
        def boom(job, *a):
            raise ValueError("nope")
        self.handlers["compare"] = boom
        r = self.runner()
        bad = r.enqueue(self.job("bad", "compare"))
        r.enqueue(self.job("ok"))
        r.start()
        self.assertTrue(wait(lambda: self.fake.ran == ["ok"]))
        self.assertEqual((bad.status, bad.error), ("failed", "nope"))


if __name__ == "__main__":
    unittest.main()
