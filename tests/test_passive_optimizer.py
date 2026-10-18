"""Passive rounds carry the optimizer state over (studio_model.train(rebase=True), studio_passive's round loop) and say so in the journal."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import studio_model
import studio_passive as passive
from studio_model import load_checkpoint, train
from tests.test_passive import Rig

WORDS = ("yes", "no", "cat", "dog", "sun", "red", "sky", "big")
EXAMPLES = [{"user": f"Copy: {w}", "assistant": w} for w in WORDS] * 3
OTHER = [{"user": f"Copy: {w}", "assistant": w[::-1]} for w in WORDS] * 3
TINY = dict(device="cpu", width=32, layers=2, heads=2, context=32, expansion=2, dropout=0, batch_size=4,
            steps=8, learning_rate=.004, warmup=3, validation=0, eval_every=2, objective="assistant")


class Spy:
    """The size of Adam's first moment when the first update of a run happens."""

    def __init__(self):
        self.first = []

    def __enter__(self):
        real = studio_model.build_optimizer

        def build(model, offload=False):
            optimizer, schedule = real(model, offload)
            update = optimizer.update

            def spying(*args, **kwargs):
                if len(self.first) < self.runs:
                    self.first.append(sum(float(t.abs().sum()) for t in optimizer.m.values()))
                    self.runs = len(self.first)
                return update(*args, **kwargs)
            optimizer.update = spying
            return optimizer, schedule
        self.runs = 1
        self.patch = patch.object(studio_model, "build_optimizer", build)
        self.patch.start()
        return self

    def __exit__(self, *exc):
        self.patch.stop()

    def next_run(self):
        self.runs = len(self.first) + 1


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        train(dict(TINY), EXAMPLES, cls.dir / "base")
        cls.base = cls.dir / "base" / "last.npz"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def rows(self, name):
        return [json_row for json_row in load_rows(self.dir / name / "metrics.jsonl")]

    def test_a_rebased_round_keeps_the_moments_the_step_and_the_schedule(self):
        with Spy() as spy:
            spy.runs = 1
            record = train(dict(TINY, steps=4, learning_rate=.002, batch_size=2, seed=7), OTHER, self.dir / "rebased", resume=self.base, rebase=True)
            spy.next_run()
            train(dict(TINY, steps=4), EXAMPLES, self.dir / "fresh", init_from=self.base)
            spy.next_run()
            train(dict(TINY, steps=12), EXAMPLES, self.dir / "continued", resume=self.base)
        rebased, fresh, continued = spy.first
        self.assertGreater(rebased, 0)  # Adam's moments came across, on other data, another seed, learning rate and batch size
        self.assertGreater(continued, 0)
        self.assertEqual(fresh, 0)  # init_from starts them from zero
        self.assertEqual(record["status"], "completed")
        checkpoint = load_checkpoint(self.dir / "rebased" / "last.npz")
        self.assertEqual((checkpoint["step"], checkpoint["optimizer_steps"]), (12, 12))
        self.assertEqual(record["step"], 12)
        first = lambda name: next(row for row in load_rows(self.dir / name / "metrics.jsonl") if row["step"] == 9)
        self.assertAlmostEqual(first("rebased")["lr"] / .002, first("continued")["lr"] / .004)  # the same place in the same schedule
        self.assertAlmostEqual(next(r for r in load_rows(self.dir / "fresh" / "metrics.jsonl") if r["step"] == 1)["lr"], .004 / 3)  # a fresh run is back in its warmup

    def test_the_best_starts_over_and_no_old_best_is_copied(self):
        train(dict(TINY, steps=4), OTHER, self.dir / "other", resume=self.base, rebase=True)
        self.assertEqual(load_checkpoint(self.dir / "other" / "best.npz")["step"] % 2, 0)
        self.assertNotEqual((self.dir / "other" / "best.npz").read_bytes(), (self.dir / "base" / "best.npz").read_bytes())

    def test_it_still_refuses_another_architecture_and_needs_resume(self):
        with self.assertRaises(ValueError):
            train(dict(TINY, steps=4, width=48), OTHER, self.dir / "wide", resume=self.base, rebase=True)
        with self.assertRaisesRegex(ValueError, "needs resume"):
            train(dict(TINY, steps=4), OTHER, self.dir / "alone", rebase=True)

    def test_a_plain_resume_still_demands_the_same_data(self):
        with self.assertRaisesRegex(ValueError, "original dataset"):
            train(dict(TINY, steps=12), OTHER, self.dir / "plain", resume=self.base)

    def test_default_train_falls_back_to_the_weights_when_the_checkpoint_is_refused(self):
        events = []
        record = passive.default_train(dict(TINY, steps=4, width=48), OTHER, self.dir / "refused", events.append, None, None, resume=self.base)
        self.assertEqual([e for e in events if e["type"] == "resume_refused"][0]["type"], "resume_refused")
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["step"], 4)  # started over from step 0 on the new architecture's fresh weights


def load_rows(path):
    import json
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


class RoundTests(unittest.TestCase):
    def types(self, rig, kind):
        return [e for e in rig.store.journal(rig.id) if e["type"] == kind]

    def test_a_round_after_a_checkpoint_resumes_and_a_step_back_starts_fresh(self):
        rig = Rig(self, [5.0, 4.0, 2.0, 3.0, 3.5, 1.5], stop={"target_error": 1.6}, plateau={"checks": 9, "max_steps": 50})
        state = rig.run()
        paths = [c["path"] for c in state["checkpoints"]]
        modes = [(c["resume"] is not None, c["init_from"] is not None) for c in rig.calls]
        self.assertEqual(modes, [(False, False), (True, False), (True, False), (True, False), (True, False), (False, True)])  # round 6 follows the step back
        self.assertEqual([c["resume"] for c in rig.calls[1:5]], paths[:4])
        self.assertEqual(rig.calls[5]["init_from"], paths[2])  # back to checkpoint 3, weights only
        self.assertEqual([e["optimizer"] for e in self.types(rig, "round")], ["fresh", "resumed", "resumed", "resumed", "resumed", "fresh"])
        self.assertEqual(rig.calls[1]["config"]["steps"], 50)  # steps to add on top of the checkpoint's

    def test_a_refused_checkpoint_is_journalled_as_fresh(self):
        rig = Rig(self, [3.0, 2.0], plateau={"checks": 9}, stop={"target_error": 2.5})

        def refusing(config, examples, output_dir, emit, stop, pause, init_from=None, resume=None):
            if resume:
                emit({"type": "resume_refused", "reason": "no optimizer state"})
            return Rig.train(rig, config, examples, output_dir, emit, stop, pause, init_from=init_from)
        rig.train = refusing
        rig.run()
        refused = self.types(rig, "optimizer")
        self.assertEqual([(e["round"], e["mode"], e["reason"]) for e in refused], [(2, "fresh", "no optimizer state")])


if __name__ == "__main__":
    unittest.main()
