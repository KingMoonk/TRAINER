"""Continuing a run (studio_ia.continue_run + studio_jobs): more steps on the same data, other data, and a bigger design."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import studio_jobs
import studio_model
from studio_ia import continue_run as cont
from studio_ia import pretraining_plan as plan
from studio_ia import sizing
from studio_spec import default_spec

WORDS = ("cat", "dog", "owl", "fox", "bee", "ant", "elk", "yak")
TEXTS = [{"text": f"the {a} saw the {b} and the {a} ran past the {b} again"} for a in WORDS for b in WORDS if a != b]
OTHER = [{"text": f"a {a} will never eat the {b} because the {b} is too quick"} for a in WORDS for b in WORDS if a != b]


def tiny_spec(width=16, layers=1):
    spec = default_spec()
    spec["dims"].update(d_model=width, n_layer=layers, n_ctx=32, dropout=0.)
    spec["block"]["mixer"]["config"]["heads"] = 2
    spec["train"]["schedule"]["config"]["warmup"] = 0
    return spec


def run_job(root, name, params, events=None):
    events = [] if events is None else events
    job = SimpleNamespace(id=name, kind="pretrain", note="", params={**params, "output_dir": str(root / name)})
    studio_jobs.continue_here(job)
    studio_jobs.pretrain(job, SimpleNamespace(run_dir=str(root)), events.append, None, None)
    return job, root / name, events


def first_run(root, steps=10, spec=None):
    cfg = plan.pretrain_config(spec or tiny_spec(), steps, 4, .2)
    cfg["eval_every"] = 5
    events = []
    job, out, _ = run_job(root, "first", {"config": cfg, "examples": TEXTS, "datasets": ["stories"], "version": 1}, events)
    return out, cfg, events


def losses(events):
    return {e["step"]: e["loss"] for e in events if e["type"] == "metric"}


class ContinueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.enterContext(mock.patch("studio_probe.default_path", return_value=self.root / "device_probe.json"))  # "auto" training probes the GPU: never into data/

    def test_a_run_remembers_the_sets_it_read(self):
        out, _, _ = first_run(self.root)
        self.assertEqual(json.loads((out / "source.json").read_text(encoding="utf-8"))["datasets"], ["stories"])
        run = cont.run_of(out)
        self.assertEqual((run["step"], run["datasets"]), (10, ["stories"]))

    def test_the_same_run_goes_on_from_its_step_with_its_optimizer_and_says_what_it_continues(self):
        out, _, before = first_run(self.root)
        run = cont.run_of(out)
        label, params = cont.same_job(1, run, 6)
        self.assertEqual(label, "Version 1 continued from step 10")
        events = []
        job, second, _ = run_job(self.root, "second", {**params, "examples": TEXTS}, events)
        self.assertEqual(job.params["resume"], run["checkpoint"])
        self.assertEqual(job.params["continue_from"]["total"], 16)
        steps = losses(events)
        self.assertEqual((min(steps), max(steps)), (11, 16))  # the curve carries on from the step it stopped at
        self.assertLess(steps[11], losses(before)[1])  # not back at an untrained model's error
        self.assertEqual(studio_model.load_checkpoint(second / "last.npz")["step"], 16)
        self.assertFalse([e for e in events if e["type"] in ("resumed", "resume_refused")])  # the queue would call this a restart
        self.assertTrue([e for e in events if e["type"] == "info" and "Continuing from step 10" in e["text"]])
        self.assertEqual(json.loads((second / "source.json").read_text(encoding="utf-8"))["datasets"], ["stories"])

    def test_other_data_keeps_the_weights_and_the_step_and_trains_on_the_new_text(self):
        out, _, _ = first_run(self.root)
        run = cont.run_of(out)
        label, params = cont.new_data_job(1, run, 6, ["poems"], own_text=True)
        self.assertEqual(label, "Version 1 continued from step 10 on new data")
        self.assertIs(params["user_text"], True)
        events = []
        job, second, _ = run_job(self.root, "second", {**params, "examples": OTHER}, events)
        self.assertTrue(job.params["resume_rebase"])
        steps = losses(events)
        self.assertEqual((min(steps), max(steps)), (11, 16))
        self.assertEqual(studio_model.load_checkpoint(second / "last.npz")["step"], 16)
        data = (second / "dataset.jsonl").read_text(encoding="utf-8")
        self.assertIn("never eat", data)
        self.assertNotIn("saw the", data)
        self.assertFalse([e for e in events if e["type"] in ("resumed", "resume_refused")])

    def test_a_bigger_design_starts_from_these_weights_not_from_random_ones(self):
        out, _, before = first_run(self.root, steps=20)
        run = cont.run_of(out)
        bigger = sizing.resized_spec(tiny_spec(), 16, 2)
        report = cont.write_warm_start(run, bigger, self.root / "warm" / "warm_start.npz")
        self.assertTrue(report["copied"])
        self.assertTrue(any(name.startswith("block1.") for name in report["zeroed"] + report["fresh"]))  # the new block is new
        grown = SimpleNamespace(number=2, spec=bigger)
        label, params = cont.grow_job(1, grown, run, self.root / "warm" / "warm_start.npz", 5, None, 4, .2)
        self.assertEqual(label, "Version 2 grown from version 1 (step 20)")
        self.assertEqual((params["mode"], params["datasets"], params["version"]), ("catch_up", ["stories"], 2))
        events = []
        _, second, _ = run_job(self.root, "second", {**params, "examples": TEXTS}, events)
        random_start = run_job(self.root, "random", {"config": plan.pretrain_config(bigger, 1, 4, .2), "examples": TEXTS}, [])[2]
        first = losses(events)[1]
        self.assertLess(first, losses(random_start)[1] if random_start else 99)
        self.assertLessEqual(first, losses(before)[20] * 1.5)  # a new block that adds nothing: it begins where the old model ended
        self.assertEqual(studio_model.load_checkpoint(second / "last.npz")["step"], 5)

    def test_a_design_with_another_word_list_cannot_take_the_weights(self):
        out, _, _ = first_run(self.root)
        other = tiny_spec()
        other["tokenizer"] = {"id": "character", "config": {}}
        with self.assertRaisesRegex(cont.ContinueError, "word list"):
            cont.write_warm_start(cont.run_of(out), other, self.root / "w.npz")
        self.assertFalse((self.root / "w.npz").exists())

    def test_a_machine_without_the_checkpoint_adds_only_the_steps_asked_from_the_published_weights(self):
        out, _, _ = first_run(self.root)
        run = cont.run_of(out)
        _, params = cont.same_job(1, run, 4)
        weights = self.root / "pulled.npz"
        shutil.copyfile(run["checkpoint"], weights)
        params = {**params, "init_from": str(weights), "resume": str(self.root / "elsewhere" / "last.npz"), "resume_steps": 14, "examples": TEXTS}
        params["continue_from"] = {**params["continue_from"], "checkpoint": params["resume"]}  # (as the owner's path)
        events = []
        job = SimpleNamespace(id="guest", kind="pretrain", note="", params={**params, "output_dir": str(self.root / "guest")})
        studio_jobs.pretrain(job, SimpleNamespace(run_dir=str(self.root)), events.append, None, None)
        self.assertTrue([e for e in events if e["type"] == "info" and "fresh optimizer" in e["text"]])
        self.assertEqual(max(losses(events)), 4)  # 4 steps added, not the owner's total of 14
        self.assertFalse([e for e in events if e["type"] == "resume_refused"])

    def test_a_restart_of_a_continued_job_resumes_its_own_new_checkpoint_to_the_total(self):
        out, _, _ = first_run(self.root)
        run = cont.run_of(out)
        _, params = cont.same_job(1, run, 6)
        stop_at = []

        class Stopper:
            def is_set(self_inner):
                return bool(stop_at)

        def emit(event):
            if event["type"] == "metric" and event["step"] >= 13:
                stop_at.append(event["step"])
        job = SimpleNamespace(id="c", kind="pretrain", note="", params={**params, "output_dir": str(self.root / "c"), "examples": TEXTS})
        studio_jobs.continue_here(job)
        studio_jobs.pretrain(job, SimpleNamespace(run_dir=str(self.root)), emit, Stopper(), None)
        crashed = studio_model.load_checkpoint(self.root / "c" / "last.npz")["step"]
        self.assertTrue(10 < crashed < 16, crashed)
        again = SimpleNamespace(id="c", kind="pretrain", note="", params={**params, "output_dir": str(self.root / "c"), "examples": TEXTS})
        studio_jobs.continue_here(again)  # the app comes back: the folder of the attempt is moved aside and its checkpoint resumed
        self.assertEqual((again.params["resume_steps"], again.params["resume_local"]), (16, True))
        self.assertNotIn("resume_rebase", again.params)
        events = []
        studio_jobs.pretrain(again, SimpleNamespace(run_dir=str(self.root)), events.append, None, None)
        self.assertEqual((min(losses(events)), max(losses(events))), (crashed + 1, 16))

    def test_a_run_without_a_full_checkpoint_says_so(self):
        (self.root / "empty").mkdir()
        with self.assertRaisesRegex(cont.ContinueError, "no full checkpoint"):
            cont.run_of(self.root / "empty")
        with self.assertRaises(cont.ContinueError):
            cont.same_job(1, {"step": 1, "config": {}, "datasets": None, "checkpoint": "x", "dir": "d"}, 0)
        with self.assertRaisesRegex(cont.ContinueError, "text"):
            cont.new_data_job(1, {"step": 1, "config": {}, "datasets": None, "checkpoint": "x", "dir": "d"}, 5, [])


class ServiceTests(unittest.TestCase):
    """WorkspaceServices.continue_run on a version whose run folder is real; the queue is not started."""

    def setUp(self):
        from studio_ia.services import FixtureServices
        self.tmp = tempfile.TemporaryDirectory()
        self.svc = FixtureServices(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(self.svc.close)  # (runs first)

    def test_each_way_queues_a_pretrain_job_that_names_what_it_continues(self):
        svc = self.svc
        out, _, _ = first_run(Path(self.tmp.name) / "runs")
        svc.store._update(1, runs=[str(out)], best_checkpoint=str(out / "best.npz"), spec=tiny_spec())
        queued = []
        with mock.patch.object(svc, "enqueue", side_effect=queued.append):
            same = svc.continue_run(1, "same", 5)
            other = svc.continue_run(1, "new_data", 5, dataset_ids=["stories"])
            grown = svc.continue_run(1, "grow", 5, width=16, layers=2)
        self.assertEqual([j.label for j in queued], ["Version 1 continued from step 10", "Version 1 continued from step 10 on new data",
                                                     f"Version {grown['version']} grown from version 1 (step 10)"])
        self.assertEqual((same["from_step"], same["kind"]), (10, "same"))
        self.assertEqual(other["kind"], "new_data")
        self.assertEqual({j.kind for j in queued}, {"pretrain"})
        self.assertEqual(queued[0].params["continue_from"]["checkpoint"], str(out / "last.npz"))
        self.assertTrue(Path(queued[2].params["init_from"]).is_file())
        newest = svc.versions()[-1]
        self.assertEqual((newest.number, newest.spec["dims"]["n_layer"]), (grown["version"], 2))
        self.assertIn("grown from version 1", newest.change)

    def test_a_version_without_a_run_or_that_does_not_exist_says_so(self):
        self.svc.store._update(2, runs=[])
        with self.assertRaisesRegex(ValueError, "no training run"):
            self.svc.continue_run(2, "same", 5)
        with self.assertRaisesRegex(ValueError, "No version 9"):
            self.svc.continue_run(9, "same", 5)


if __name__ == "__main__":
    unittest.main()
