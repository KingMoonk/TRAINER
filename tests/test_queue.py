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


def wait(condition, timeout=20):  # each job starts a worker process, which takes a moment
    end = time.time() + timeout
    while time.time() < end:
        if condition():
            return True
        time.sleep(.01)
    return False


# Handlers run in a worker process, so they are module-level functions that talk to the test through files in the context's root.
def _note(context, line):
    with open(Path(context.root) / "ran.txt", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def long_job(job, context, emit, stop, pause):
    """Emits a step every 5 ms until the test creates the 'gate' file or the job is stopped."""
    _note(context, job.label)
    steps = 0
    while not ((Path(context.root) / "gate").exists() or stop.is_set()):
        while pause.is_set() and not stop.is_set():
            time.sleep(.005)
        steps += 1
        emit({"type": "metric", "step": steps, "total": 1000, "lr": .001})
        time.sleep(.005)


def quick(job, context, emit, stop, pause):
    _note(context, job.label)


def cooldown(job, context, emit, stop, pause):
    _note(context, "cooldown of " + job.params["parent"])


def note_cap(job, context, emit, stop, pause):
    _note(context, f"share {job.params['memory_share']}")


def boom(job, context, emit, stop, pause):
    raise ValueError("nope")


class Gate:
    def __init__(self, path):
        self.path = path

    def set(self):
        self.path.touch()


class Fake:
    """What the handlers did: the labels they logged, and the last step the long job emitted (seen through the runner's events)."""
    def __init__(self, root):
        self.root, self.gate, self.steps = root, Gate(root / "gate"), 0

    @property
    def ran(self):
        try:
            return (self.root / "ran.txt").read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return []

    def listen(self, kind, payload):
        if kind == "train" and payload.get("type") == "metric":
            self.steps = payload["step"]


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "data" / "queue.json"
        self.fake = Fake(Path(self.tmp.name))
        self.handlers = {"pretrain": long_job, "quicktest": quick, "restore_answers": quick, "checkpoint_now": cooldown}

    def runner(self, plan=ANY, **kw):
        r = Runner(self.path, self.handlers, **{"tick": .01, **kw})
        r.set_plan(plan)
        r.subscribe(self.fake.listen)
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
        # the last job's save can land just after the queue reads idle
        self.assertTrue(wait(lambda: [j.status for j in Runner(self.path, self.handlers).jobs] == ["done"] * 3))

    def test_a_reason_a_watchdog_left_on_the_job_becomes_its_error(self):
        r = self.runner()
        r.enqueue(self.job("ended", ended_reason="Ended: the driver stopped responding at round 2, so the run was stopped."))
        r.start()
        self.assertTrue(wait(lambda: r.jobs[0].status == "failed"))
        self.assertEqual(r.jobs[0].error, "Ended: the driver stopped responding at round 2, so the run was stopped.")
        self.assertNotIn("ended_reason", r.jobs[0].params)

    def test_persistence_across_restart_requeues_interrupted_job(self):
        r = self.runner()
        r.enqueue(self.job("long", "pretrain"))
        r.enqueue(self.job("next"))
        r.start()
        self.assertTrue(wait(lambda: self.fake.ran == ["long"]))
        r.shutdown()
        self.assertEqual(r.jobs[0].status, "waiting")  # the worker stopped in time; the job goes back to the front
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
        r.shutdown()  # its thread may still be saving the finished job; Windows refuses to open a file mid-replace
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

    def test_a_job_waiting_for_its_size_is_skipped_until_filled(self):
        r = self.runner()
        sized = r.enqueue(self.job("sizing", sizing=True, pinned="peer"))
        r.enqueue(self.job("b"))
        r.start()
        self.assertTrue(wait(lambda: self.fake.ran == ["b"]))
        self.assertEqual([j.label for j in r.state().up_next], ["sizing"])  # kept, in place, not run
        self.assertTrue(r.fill(sized.id, "filled", {"sizing": True, "steps": 5}))
        self.assertEqual(sized.params, {"steps": 5, "pinned": "peer"})  # a pin set meanwhile stays
        self.assertFalse(r.fill(sized.id, "again", {}))  # filled once

    def test_a_job_filled_after_sizing_runs(self):
        r = self.runner()
        sized = r.enqueue(self.job("sizing", sizing=True))
        r.start()
        r.fill(sized.id, "filled", {"steps": 5})
        self.assertTrue(wait(lambda: self.fake.ran == ["filled"]))

    def test_a_job_whose_size_fails_ends_failed_and_one_left_by_a_restart_too(self):
        r = self.runner()
        seen = []
        r.subscribe(lambda kind, payload: seen.append(payload) if kind == "job" else None)
        a = r.enqueue(self.job("a", sizing=True))
        r.enqueue(self.job("b", sizing=True))
        r.fail(a.id, "The text could not be read.")
        self.assertEqual((a.status, a.error), ("failed", "The text could not be read."))
        self.assertEqual(seen, [{"job": a.id, "status": "failed", "error": "The text could not be read."}])
        self.assertEqual([j.id for j in r.state().failed], [a.id])
        r.shutdown()
        again = Runner(self.path, self.handlers)  # the app stopped while b's size was worked out
        self.assertEqual([(j.label, j.status) for j in again.jobs], [("a", "failed"), ("b", "failed")])
        self.assertIn("Add the run again", again.jobs[1].error)
        self.assertEqual(again.state().up_next, [])

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

    def test_a_job_that_ignores_the_plan_runs_while_the_computer_is_in_use_and_others_wait(self):
        plan = Plan(hours="any", idle_only=True, idle_after_s=30, pause_on_battery=False)
        with mock.patch.object(studio_queue, "idle_seconds", lambda: 1.):  # someone is at the keyboard
            r = self.runner(plan)
            r.enqueue(self.job("plain"))
            r.enqueue(self.job("test", "quicktest", ignore_plan=True))
            r.start()
            self.assertTrue(wait(lambda: r.state().blocked == "computer in use"))
            self.assertTrue(wait(lambda: "plain" not in self.fake.ran and [j.status for j in r.jobs] == ["waiting", "waiting"]))
            r.move_up(r.jobs[1].id)
            self.assertTrue(wait(lambda: self.fake.ran == ["test"]))
            self.assertTrue(wait(lambda: r.jobs[0].status == "done"))
            time.sleep(.05)
            self.assertNotIn("plain", self.fake.ran)  # the plan still holds a job without the param
            self.assertEqual(r.state().blocked, "computer in use")

    def test_the_plan_does_not_pause_a_running_job_that_ignores_it(self):
        plan = Plan(hours="any", idle_only=True, idle_after_s=30, pause_on_battery=False)
        idle = {"s": 120.}
        with mock.patch.object(studio_queue, "idle_seconds", lambda: idle["s"]):
            r = self.runner(plan)
            r.enqueue(self.job("long", "pretrain", ignore_plan=True))
            r.start()
            self.assertTrue(wait(lambda: self.fake.steps > 3))
            idle["s"] = 1.
            self.assertTrue(wait(lambda: r.state().blocked == "computer in use"))
            time.sleep(.05)
            seen = self.fake.steps
            self.assertTrue(wait(lambda: self.fake.steps > seen + 3))
            self.assertFalse(r.pause_event.is_set())
            self.fake.gate.set()

    def test_gate_is_read_once_per_tick_not_per_pass(self):
        # Worker startup and wall-clock sleeps vary with CI load. Drive the same
        # watcher used by both runner loops at exact monotonic times instead.
        with mock.patch.object(studio_queue.time, "monotonic", return_value=0.) as clock, \
                mock.patch.object(studio_queue, "idle_seconds", return_value=120.) as idle:
            r = Runner(self.path, self.handlers, tick=1.)
            r.set_plan(Plan(hours="any", idle_only=True, pause_on_battery=False))
            self.addCleanup(r.shutdown)
            for tick in range(3):
                for fraction in (0., .125, .25, .5, .75, .875):
                    clock.return_value = tick + fraction
                    r._watch()
                    self.assertEqual(idle.call_count, tick + 1, f"gate reads at {clock.return_value}")

            # A changed plan must take effect even before the next tick.
            r.set_plan(Plan(hours="any", idle_only=True, idle_after_s=180., pause_on_battery=False))
            r._watch()
            self.assertEqual(idle.call_count, 4)
            self.assertEqual(r.blocked, "computer in use")
            r._watch()
            self.assertEqual(idle.call_count, 4)

    def test_a_job_id_already_in_the_queue_is_not_added_twice(self):
        r = self.runner()
        first = r.enqueue(Job("quicktest", "a", id="inv1"))
        again = r.enqueue(Job("quicktest", "a again", id="inv1"))
        self.assertIs(again, first)
        self.assertEqual([j.id for j in r.jobs], ["inv1"])
        self.assertEqual([j.id for j in Runner(self.path, self.handlers).jobs], ["inv1"])

    def test_a_slow_save_does_not_hold_up_readers_of_the_queue(self):
        r = self.runner(tick=30)
        r.pause()
        r.start()
        entered, release, real = threading.Event(), threading.Event(), studio_queue.atomic_json

        def slow(path, value):  # a save on Windows that waits out a scanner holding queue.json
            entered.set()
            release.wait(10)
            real(path, value)

        with mock.patch.object(studio_queue, "atomic_json", slow):
            r.enqueue(self.job("a"))
            self.assertTrue(entered.wait(5))
            r.enqueue(self.job("b"))  # saved again while the first save is still writing
            started = time.monotonic()
            self.assertEqual([j.label for j in r.state().up_next], ["a", "b"])
            self.assertLess(time.monotonic() - started, 1)
            self.assertFalse(r.flush(.05))
            release.set()
            self.assertTrue(r.flush(5))
        self.assertEqual([j.label for j in Runner(self.path, self.handlers).jobs], ["a", "b"])

    def test_enqueue_and_resume_wake_the_runner_at_once(self):
        r = self.runner(tick=30)  # no polling to fall back on within the test
        r.start()
        time.sleep(.05)
        r.enqueue(self.job("a"))
        self.assertTrue(wait(lambda: self.fake.ran == ["a"], 10))
        self.assertTrue(wait(lambda: r.state().job is None, 10))  # the end of a worker wakes the runner too
        r.pause()
        r.enqueue(self.job("b"))
        time.sleep(.1)
        self.assertEqual(self.fake.ran, ["a"])
        r.resume()
        self.assertTrue(wait(lambda: self.fake.ran == ["a", "b"], 10))

    def test_plan_change_rereads_the_gate_at_once(self):
        r = self.runner(Plan(hours="set", start_hour=22, end_hour=7, idle_only=False, pause_on_battery=False),
                        clock=lambda: datetime(2026, 1, 1, 12), tick=30)
        r.enqueue(self.job("a"))
        r.start()
        self.assertTrue(wait(lambda: r.state().blocked == "outside hours", 2))
        r.set_plan(ANY)
        self.assertTrue(wait(lambda: self.fake.ran == ["a"], 10))

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

    def test_yield_running_runs_the_waiting_job_first_and_puts_the_running_one_back_in_the_queue(self):
        r = self.runner()
        main = r.enqueue(self.job("long", "pretrain"))
        r.start()
        self.assertTrue(wait(lambda: self.fake.steps > 2))
        urgent = r.enqueue(self.job("urgent"))
        self.assertFalse(r.yield_running("nope"))
        self.assertTrue(r.yield_running(urgent.id))
        self.assertTrue(wait(lambda: self.fake.ran[:2] == ["long", "urgent"]))
        self.assertTrue(wait(lambda: urgent.status == "done"))
        self.assertNotEqual(main.status, "stopped")  # it was stepped aside, not stopped: it runs again from its place
        self.assertTrue(wait(lambda: self.fake.ran == ["long", "urgent", "long"]))
        self.fake.gate.set()
        self.assertTrue(wait(lambda: main.status == "done"))

    def test_yield_running_without_a_job_holds_the_running_one_and_does_nothing_when_idle(self):
        r = self.runner()
        self.assertFalse(r.yield_running())
        main = r.enqueue(self.job("long", "pretrain"))
        r.start()
        self.assertTrue(wait(lambda: self.fake.steps > 2))
        self.assertTrue(r.yield_running())
        self.assertTrue(wait(lambda: main.status == "held"))

    def test_user_text_pretrain_appends_restore_answers(self):
        r = self.runner()
        r.enqueue(Job("pretrain", "night", params={"user_text": True, "previous_best": "best.npz"}))
        self.fake.gate.set()
        r.start()
        self.assertTrue(wait(lambda: "Restore how it answers" in self.fake.ran))
        done = [j for j in r.jobs if j.kind == "restore_answers"][0]
        self.assertEqual(done.params, {"round": 3, "teacher": "best.npz", "memory_share": .75})

    def test_memory_cap_is_handed_to_jobs(self):
        self.handlers["quicktest"] = note_cap
        r = self.runner(Plan(hours="any", idle_only=False, pause_on_battery=False, memory="half"))
        r.enqueue(self.job("a"))
        r.start()
        self.assertTrue(wait(lambda: self.fake.ran))
        self.assertEqual(self.fake.ran, ["share 0.5"])

    def test_failed_job_does_not_block_queue(self):
        self.handlers["compare"] = boom
        r = self.runner()
        bad = r.enqueue(self.job("bad", "compare"))
        r.enqueue(self.job("ok"))
        r.start()
        self.assertTrue(wait(lambda: self.fake.ran == ["ok"]))
        self.assertEqual((bad.status, bad.error), ("failed", "nope"))


if __name__ == "__main__":
    unittest.main()
