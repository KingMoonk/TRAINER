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
        r = Runner(self.path, self.handlers, tick=.01, **kw)
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

    def test_move_up_and_remove(self):
        r = self.runner()
        a, b, c = (r.enqueue(self.job(n)) for n in "abc")
        r.move_up(c.id)
        self.assertEqual([j.label for j in r.state().up_next], ["a", "c", "b"])
        r.move_up(a.id)  # already first
        r.remove(c.id)
        self.assertEqual([j.label for j in r.state().up_next], ["a", "b"])
        self.assertEqual([j.label for j in Runner(self.path, self.handlers).jobs], ["a", "b"])

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
