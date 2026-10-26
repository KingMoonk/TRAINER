"""studio_run_watch and the queue's use of it: a running pretrain job reports a halved batch, low disk or RAM and a projected overrun, from
injected profiles and fake worker events (no GPU, no window)."""
import tempfile
import time
import unittest
from pathlib import Path

import studio_fit
import studio_run_watch as W
from studio_queue import Job, Plan, Runner

G = 2 ** 30
ANY = Plan(hours="any", idle_only=False, pause_on_battery=False)
ROOMY = studio_fit.Profile(device="cuda", vram_total=80 * G, vram_free=78 * G, ram_total=64 * G, ram_free=48 * G, disk_free=500 * G)


def wait(condition, timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        if condition():
            return True
        time.sleep(.01)
    return False


def _hold(context, stop):
    while not ((Path(context.root) / "gate").exists() or stop.is_set()):
        time.sleep(.005)


def halving(job, context, emit, stop, pause):
    emit({"type": "metric", "step": 1, "total": 1000, "lr": .001})
    emit({"type": "info", "kind": "oom", "step": 3, "batch": 4, "accumulation": 2, "text": "Out of memory at step 3: batch size now 4, accumulation 2."})
    emit({"type": "info", "kind": "oom", "step": 9, "batch": 2, "accumulation": 4, "text": "Out of memory at step 9: batch size now 2, accumulation 4."})
    _hold(context, stop)


def steady(job, context, emit, stop, pause):
    emit({"type": "metric", "step": 1, "total": 1000, "lr": .001})
    _hold(context, stop)


def slow(job, context, emit, stop, pause):
    """10 steps of 1000 tokens in 40 s: 250 tokens/s, so the 990 steps left take about 66 min more."""
    emit({"type": "metric", "step": 10, "total": 1000, "lr": .001, "trained_tokens": 10_000, "elapsed_seconds": 40.})
    _hold(context, stop)


class ProjectionTests(unittest.TestCase):
    def project(self, **over):
        args = dict(step=10, first_step=0, total=1000, trained_tokens=10_000, elapsed=40., token_budget=None, window_seconds=3600., now=5.)
        return W.projection(**{**args, **over})

    def test_a_pace_that_ends_past_the_window_is_stated_with_the_overrun(self):
        found = self.project()
        self.assertEqual((found["kind"], found["at"]), ("overrun", 5.))
        self.assertIn("250 tokens/s", found["text"])
        self.assertIn("past the 1.0 h window", found["text"])

    def test_a_run_that_ends_inside_the_window_says_nothing(self):
        self.assertIsNone(self.project(window_seconds=7200.))

    def test_the_token_budget_ends_the_run_before_the_last_step(self):
        self.assertIsNone(self.project(token_budget=100_000))

    def test_no_window_or_no_pace_yet_says_nothing(self):
        self.assertIsNone(self.project(window_seconds=None))
        self.assertIsNone(self.project(elapsed=5.))
        self.assertIsNone(self.project(step=11, first_step=10))  # one step here: not a pace

    def test_a_job_that_re_plans_to_its_wall_clock_has_no_window_to_overrun(self):
        self.assertIsNone(W.window_seconds({"wall_seconds": 3600., "fit": {"window_hours": 1}}))
        self.assertEqual(W.window_seconds({"fit": {"window_hours": 1}}), 3600.)
        self.assertIsNone(W.window_seconds({}))


class ResourceTests(unittest.TestCase):
    def test_free_disk_and_ram_below_the_need_are_reported_and_unknown_is_not(self):
        need = {"disk": 10 * G, "ram": 4 * G}
        low = studio_fit.Profile(device="cpu", ram_free=G, disk_free=2 * G)
        self.assertEqual([n["kind"] for n in W.resources(low, need)], ["disk", "ram"])
        self.assertEqual(W.resources(ROOMY, need), [])
        self.assertEqual(W.resources(studio_fit.Profile(device="cpu"), need), [])

    def test_the_need_comes_from_the_job_config_and_grows_with_the_optimizer_in_host_memory(self):
        from studio_ia import pretraining_plan as plan
        from studio_spec import default_spec, validate_spec
        cfg = plan.pretrain_config(validate_spec(default_spec()), 1000, 8, .05)
        base = W.needs({"config": cfg})
        self.assertGreater(base["disk"], 0)
        self.assertGreater(W.needs({"config": cfg, "offload_optimizer": True})["ram"], base["ram"])
        self.assertEqual(W.needs({"config": None}), {})

    def test_the_oom_event_becomes_a_notice_with_the_new_batch_and_accumulation(self):
        found = W.oom_notice({"type": "info", "kind": "oom", "step": 3, "batch": 4, "accumulation": 2}, now=1.)
        self.assertEqual(found["kind"], "oom")
        self.assertIn("step 3", found["text"])
        self.assertIn("halved to 4 with accumulation 2", found["text"])
        self.assertIsNone(W.oom_notice({"type": "info", "text": "something else"}))


class RunningJobTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "data" / "queue.json"

    def start(self, handler, profile=None, **params):
        from studio_ia import pretraining_plan as plan
        from studio_spec import default_spec, validate_spec
        cfg = plan.pretrain_config(validate_spec(default_spec()), 1000, 8, .05)
        r = Runner(self.path, {"pretrain": handler}, tick=.01, profile=profile)
        r.watch_every = .03
        r.set_plan(ANY)
        self.addCleanup(r.shutdown)
        self.addCleanup(Path(self.tmp.name, "gate").touch)
        r.enqueue(Job("pretrain", "run", params={"config": cfg, "mode": "from_scratch", **params}))
        r.start()
        return r

    def kinds(self, r):
        return [n["kind"] for n in r.state().notices]

    def test_a_batch_halved_after_running_out_of_memory_shows_as_an_event_with_the_new_batch(self):
        r = self.start(halving)
        self.assertTrue(wait(lambda: self.kinds(r) == ["oom", "oom"]))
        texts = [n["text"] for n in r.state().notices]
        self.assertIn("halved to 4 with accumulation 2", texts[0])
        self.assertIn("halved to 2 with accumulation 4", texts[1])
        self.assertIn("halved to 2 with accumulation 4", r.jobs[0].note)  # the job keeps it after the run

    def test_low_disk_during_the_run_is_reported_and_clears_when_space_returns(self):
        seen = [ROOMY]  # roomy when the job is fitted and started; the disk fills while it trains
        r = self.start(steady, profile=lambda: seen[-1])
        self.assertTrue(wait(lambda: r.state().step == 1))
        self.assertEqual(self.kinds(r), [])
        seen.append(studio_fit.Profile(device="cuda", vram_total=80 * G, vram_free=78 * G, ram_total=64 * G, ram_free=48 * G, disk_free=2 ** 20))
        self.assertTrue(wait(lambda: self.kinds(r) == ["disk"]))
        self.assertIn("Free disk is down to", r.jobs[0].note)
        seen.append(ROOMY)
        self.assertTrue(wait(lambda: self.kinds(r) == []))

    def test_low_ram_during_the_run_is_reported(self):
        seen = [ROOMY]
        r = self.start(steady, profile=lambda: seen[-1])
        self.assertTrue(wait(lambda: r.state().step == 1))
        seen.append(studio_fit.Profile(device="cuda", vram_total=80 * G, vram_free=78 * G, ram_total=64 * G, ram_free=1024, disk_free=500 * G))
        self.assertTrue(wait(lambda: self.kinds(r) == ["ram"]))

    def test_a_pace_that_will_overrun_the_window_is_stated_while_the_run_goes(self):
        r = self.start(slow, fit={"window_hours": 1})
        self.assertTrue(wait(lambda: self.kinds(r) == ["overrun"]))
        self.assertIn("past the 1.0 h window", r.state().notices[0]["text"])
        self.assertIn("past the 1.0 h window", r.jobs[0].note)

    def test_the_same_pace_inside_a_longer_window_says_nothing(self):
        r = self.start(slow, fit={"window_hours": 72})
        self.assertTrue(wait(lambda: r.state().step == 10))
        time.sleep(.2)
        self.assertEqual(self.kinds(r), [])

    def test_notices_end_with_the_job(self):
        r = self.start(halving)
        self.assertTrue(wait(lambda: self.kinds(r) == ["oom", "oom"]))
        Path(self.tmp.name, "gate").touch()
        self.assertTrue(wait(lambda: r.jobs[0].status == "done"))
        self.assertEqual(r.state().notices, [])


if __name__ == "__main__":
    unittest.main()
