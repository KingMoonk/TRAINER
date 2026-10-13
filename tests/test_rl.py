"""Rounds 2 and 3 (studio_rl), the gate, and the post_round handler through a Runner, on a tiny trained model."""
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import studio_gate
import studio_rl
import studio_rules
from studio_jobs import HANDLERS
from studio_model import AdamW, ScratchTransformer, Tokenizer, load_checkpoint, train
from studio_queue import Job, Plan, Runner
from studio_rlhf import _encode_pair
from tests.fixtures.ia_workspace import _generate_problems

WORDS = ("yes", "no", "cat", "dog", "sun", "red", "sky", "big")
EXAMPLES = [{"user": f"Copy: {w}", "assistant": w} for w in WORDS] * 3
TINY = dict(device="cpu", width=32, layers=1, heads=2, context=32, expansion=2, dropout=0, batch_size=4,
            steps=20, learning_rate=.004, warmup=0, validation=0, eval_every=10, objective="assistant")
USER = [{"role": "user", "content": "Copy: yes"}]


class RLChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        train(TINY, EXAMPLES, cls.dir / "base")
        cls.ckpt = cls.dir / "base" / "last.npz"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def model(self):
        data = load_checkpoint(self.ckpt)
        tokenizer = Tokenizer.restore(data["tokenizer"])
        model = ScratchTransformer(dict(data["config"], dropout=0), tokenizer.vocab_size)
        model.load_weights(data["model"])
        return model, tokenizer

    def rows(self, tokenizer, answers):
        return [_encode_pair(tokenizer, USER, a, 32) for a in answers]

    # ------------------------------------------------------------------ round 2
    def test_policy_gradient_matches_finite_difference(self):
        model, tokenizer = self.model()
        rows, adv = self.rows(tokenizer, ["yes", "no", "cat"]), [1., -1., .5]
        model.zero_grad()
        studio_rl.reinforce_loss(model, rows, adv, backward=True)
        for key in ("block0.f1.w", "token", "final.g"):
            index = tuple(np.unravel_index(np.argmax(np.abs(model.g[key].numpy())), model.g[key].shape))
            original = float(model.p[key][index])
            model.p[key][index] = original + 1e-2
            plus = studio_rl.reinforce_loss(model, rows, adv)
            model.p[key][index] = original - 1e-2
            minus = studio_rl.reinforce_loss(model, rows, adv)
            model.p[key][index] = original
            numeric = (plus - minus) / 2e-2
            self.assertAlmostEqual(float(model.g[key][index]), numeric, delta=max(2e-3, abs(numeric) * .05), msg=key)

    def test_round2_loss_decreases_on_a_copy_rule(self):
        model, tokenizer = self.model()
        optimizer = AdamW(model, 0.)
        answers = ["yes", "no", "cat", "yes", "dog", "yes"]
        rewards = [float(studio_rules.check("copy", "Copy: yes", a)) for a in answers]
        adv = studio_rl.advantages(rewards)
        rows = self.rows(tokenizer, answers)
        start = studio_rl.reinforce_loss(model, rows, adv)
        for _ in range(5):
            model.zero_grad()
            studio_rl.reinforce_loss(model, rows, adv, backward=True)
            optimizer.update(model, 2e-3)
        self.assertLess(studio_rl.reinforce_loss(model, rows, adv), start)

    def test_advantages(self):
        self.assertTrue(np.allclose(studio_rl.advantages([1, 1, 1]), 0))
        self.assertTrue(np.allclose(studio_rl.advantages([1, 0], divide_by_std=False), [.5, -.5]))

    def test_practice_with_checks_runs_and_saves(self):
        events = []
        problems = [{"prompt": f"Copy: {w}", "rule": "copy"} for w in WORDS[:4]]
        record = studio_rl.practice_with_checks(self.ckpt, problems, self.dir / "r2", group=4, steps=3, learning_rate=1e-3,
                                                emit=events.append)
        self.assertEqual(record["status"], "completed")
        self.assertTrue((self.dir / "r2" / "last.npz").exists() and (self.dir / "r2" / "best.npz").exists())
        groups = [e for e in events if e["type"] == "group"]
        self.assertTrue(groups)
        self.assertTrue(all(len(g["answers"]) == 4 for g in groups))
        for g in groups:
            if g["skipped"]:
                self.assertTrue(all(a["push"] == 0 for a in g["answers"]))

    # ------------------------------------------------------------------ round 3
    def test_reverse_kl(self):
        student, tokenizer = self.model()
        rows = self.rows(tokenizer, ["yes", "no", "cat"])
        same = studio_rl.Teacher(self.ckpt)
        kl, _ = studio_rl.reverse_kl(student, same.model, rows, rows)
        self.assertGreaterEqual(kl, -1e-5)
        self.assertLess(kl, 1e-5)
        other = train({**TINY, "steps": 5, "seed": 3}, EXAMPLES, self.dir / "other")
        self.assertTrue(other)
        kl, per_row = studio_rl.reverse_kl(student, studio_rl.Teacher(self.dir / "other" / "last.npz").model, rows, rows)
        self.assertTrue(np.isfinite(kl) and all(np.isfinite(s).all() and np.isfinite(t).all() for s, t in per_row))
        # the privileged teacher sees the correction as a system message; the student does not
        teacher = studio_rl.privileged(self.ckpt, "Always answer yes.")
        t_rows = [_encode_pair(tokenizer, teacher.messages(USER), "yes", 32)]
        kl, _ = studio_rl.reverse_kl(student, teacher.model, self.rows(tokenizer, ["yes"]), t_rows)
        self.assertTrue(np.isfinite(kl))

    def test_learn_from_teacher_runs(self):
        events = []
        record = studio_rl.learn_from_teacher(self.ckpt, [f"Copy: {w}" for w in WORDS[:4]], studio_rl.privileged(self.ckpt, "Copy the word."),
                                              self.dir / "r3", steps=3, learning_rate=1e-3, emit=events.append)
        heat = next(e for e in events if e["type"] == "heat")
        self.assertEqual(len(heat["words"]), len(heat["surprise"]))
        self.assertLessEqual(record["moved"], record["words"])
        self.assertTrue((self.dir / "r3" / "last.npz").exists())

    # ------------------------------------------------------------------ gate
    def test_gate_truth_table(self):
        def run(b0, b1, k0, k1, n0=10, n1=10):
            return studio_gate.passes({"bpc": b0, "pass_k": k0, "pass_n": n0}, {"bpc": b1, "pass_k": k1, "pass_n": n1})
        ok, checks = run(3., 2.9, 4, 5)
        self.assertTrue(ok)
        self.assertEqual([c["ok"] for c in checks], [True, True])
        self.assertEqual(checks[0]["detail"], "2.900 bits per character, was 3.000")
        self.assertEqual(checks[1]["detail"], "5 of 10 held-back problems, was 4 of 10")
        self.assertTrue(run(3., 3., 4, 4)[0])                       # equal is "did not rise / fall"
        ok, checks = run(3., 3.1, 4, 5)
        self.assertFalse(ok)
        self.assertEqual([c["ok"] for c in checks], [False, True])
        self.assertTrue(checks[0]["detail"].endswith(": rose"))
        self.assertEqual([c["ok"] for c in run(3., 2.9, 5, 4)[1]], [True, False])
        self.assertEqual([c["ok"] for c in run(3., 3.1, 5, 4)[1]], [False, False])
        self.assertTrue(run(3., 2.9, 2, 4, n0=4, n1=8)[0])          # 2/4 -> 4/8: same rate
        self.assertFalse(run(3., 2.9, 2, 3, n0=4, n1=8)[0])         # 2/4 -> 3/8: fell
        self.assertFalse(run(None, 2.9, 4, 4)[0])                   # unmeasured error never passes
        self.assertFalse(run(3., float("nan"), 4, 4)[0])
        self.assertTrue(run(3., 2.9, 0, 0, n0=0, n1=0)[0])          # no problems: nothing can have fallen

    # ------------------------------------------------------------------ handler through a Runner
    def test_post_round_handler_completes_through_a_runner(self):
        problems = _generate_problems(10, __import__("random").Random(42))
        services = SimpleNamespace(workspace=SimpleNamespace(run_dir=self.dir / "runs"), current_version=lambda: None,
                                   best_version=lambda: None, post_rows=lambda kind: [])
        events = []
        runner = Runner(self.dir / "queue.json", HANDLERS, services, tick=.01)
        runner.set_plan(Plan(hours="any", idle_only=False, pause_on_battery=False))
        runner.subscribe(lambda kind, payload: events.append(payload) if kind == "train" else None)
        self.addCleanup(runner.shutdown)
        job = runner.enqueue(Job("post_round", "Round 2", params={
            "round": 2, "checkpoint": str(self.ckpt), "output_dir": str(self.dir / "queued"), "problems": problems[:6], "group": 3,
            "steps": 2, "heldout_problems": problems[6:], "heldout_docs": [{"user": "Copy: yes", "assistant": "yes"}, "Copy: no no"]}))
        runner.start()
        end = time.time() + 120
        while time.time() < end and job.status in ("waiting", "running"):
            time.sleep(.05)
        self.assertEqual((job.status, job.error), ("done", ""))
        gate = next(e for e in events if e.get("type") == "gate")
        self.assertEqual(gate["round"], 2)
        self.assertEqual(len(gate["checks"]), 2)
        self.assertEqual(gate["after"]["pass_n"], 4)
        self.assertIsInstance(gate["ok"], bool)


if __name__ == "__main__":
    unittest.main()
