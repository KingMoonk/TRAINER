"""Rounds 2 and 3 (studio_rl), the gate, and the post_round handler through a Runner, on a tiny trained model."""
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

import studio_gate
import studio_rl
import studio_rules
from studio_jobs import post_round
from studio_model import AdamW, ScratchTransformer, Tokenizer, load_checkpoint, train
from studio_queue import Job, Plan, Runner
from studio_rlhf import _encode_pair
from tests.fixtures.ia_workspace import _generate_problems
from tests.torchcheck import counting_backward, watched_models

WORDS = ("yes", "no", "cat", "dog", "sun", "red", "sky", "big")
EXAMPLES = [{"user": f"Copy: {w}", "assistant": w} for w in WORDS] * 3
TINY = dict(device="cpu", width=32, layers=1, heads=2, context=32, expansion=2, dropout=0, batch_size=4,
            steps=20, learning_rate=.004, warmup=0, validation=0, eval_every=10, objective="assistant")
USER = [{"role": "user", "content": "Copy: yes"}]


def one_problem_solved(job, context, emit, stop, pause):
    """post_round in the worker, with the probe saying the model solves one problem (a patch here would not reach the worker)."""
    with mock.patch.object(studio_rl, "pass_at_k", return_value={"problems": 8, "k": 4, "solved": 1, "passed": 1, "answers": 32}):
        return post_round(job, context, emit, stop, pause)


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

    def test_share_of_groups_with_spread(self):
        self.assertEqual(studio_rl.group_spread([[1., 0., 0.], [0., 0., 0.], [1., 1., 1.], [.5, .8]]), (2, 4))
        self.assertEqual(studio_rl.group_spread([]), (0, 0))

    def test_all_fail_and_all_pass_groups_teach_nothing(self):
        self.assertFalse(studio_rl.has_spread([0., 0., 0., 0.]))
        self.assertFalse(studio_rl.has_spread([1., 1.]))
        self.assertTrue(studio_rl.has_spread([0., 0., 1.]))
        self.assertEqual(studio_rl.group_spread([[0.] * 6] * 5), (0, 5))

    def test_loss_is_a_mean_over_tokens_so_length_does_not_scale_it(self):
        def loss(lengths, per_token):
            width = max(lengths)
            valid = torch.tensor([[i < n for i in range(width)] for n in lengths])
            logp = valid.to(torch.float32) * per_token
            return float(studio_rl.length_normalised(logp, valid, torch.tensor([1., -.5][:len(lengths)])))
        self.assertAlmostEqual(loss([2, 2], -.7), loss([20, 20], -.7))
        self.assertAlmostEqual(loss([2, 20], -.7), loss([20, 2], -.7))   # which answer is the long one does not matter
        short, long = loss([3, 3], -1.), loss([30, 30], -1.)
        self.assertAlmostEqual(short, long)
        self.assertAlmostEqual(short, -(1. * -1. + -.5 * -1.) / 2)    # -(advantage * mean token log-prob), averaged over answers

    def test_practice_with_checks_reports_groups_that_taught_something(self):
        events = []
        problems = [{"prompt": f"Copy: {w}", "rule": "copy"} for w in WORDS[:4]]
        record = studio_rl.practice_with_checks(self.ckpt, problems, self.dir / "r2spread", group=4, steps=3, learning_rate=1e-3,
                                                emit=events.append)
        metrics = [e for e in events if e["type"] == "metric"]
        self.assertEqual(sum(m["groups"] for m in metrics), record["groups"])
        self.assertEqual(sum(m["taught"] for m in metrics), record["taught"])
        self.assertEqual((metrics[-1]["groups_total"], metrics[-1]["taught_total"]), (record["groups"], record["taught"]))
        self.assertEqual(record["taught"], sum(1 for e in events if e["type"] == "group" and not e["skipped"]))
        self.assertLessEqual(record["taught"], record["groups"])

    def test_pass_at_k_counts_solved_problems(self):
        problems = [{"prompt": f"Copy: {w}", "rule": "copy"} for w in WORDS[:3]]
        probe = studio_rl.pass_at_k(self.ckpt, problems, count=3, k=2)
        self.assertEqual((probe["problems"], probe["k"], probe["answers"]), (3, 2, 6))
        self.assertLessEqual(probe["solved"], probe["problems"])
        self.assertLessEqual(probe["solved"], probe["passed"])

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
        kl, _ = studio_rl.reverse_kl(student, studio_rl.Teacher(self.ckpt).score(rows), rows)
        self.assertGreaterEqual(kl, -1e-5)
        self.assertLess(kl, 1e-5)
        other = train({**TINY, "steps": 5, "seed": 3}, EXAMPLES, self.dir / "other")
        self.assertTrue(other)
        kl, per_row = studio_rl.reverse_kl(student, studio_rl.Teacher(self.dir / "other" / "last.npz").score(rows), rows)
        self.assertTrue(np.isfinite(kl) and all(np.isfinite(s).all() and np.isfinite(t).all() for s, t in per_row))
        # the privileged teacher sees the correction as a system message; the student does not
        teacher = studio_rl.privileged(self.ckpt, "Always answer yes.")
        t_rows = [_encode_pair(tokenizer, teacher.messages(USER), "yes", 32)]
        kl, _ = studio_rl.reverse_kl(student, teacher.score(t_rows), self.rows(tokenizer, ["yes"]))
        self.assertTrue(np.isfinite(kl))

    def test_a_teacher_with_another_word_list_is_rejected(self):
        train({**TINY, "steps": 2, "tokenizer": "character"}, EXAMPLES, self.dir / "letters")
        teacher = studio_rl.Teacher(self.dir / "letters" / "last.npz")
        student, tokenizer = self.model()
        with self.assertRaisesRegex(ValueError, "different word list"):
            studio_rl.check_teacher(tokenizer, 32, teacher)
        with self.assertRaisesRegex(ValueError, "different word list"):
            studio_rl.learn_from_teacher(self.ckpt, ["Copy: yes"], self.dir / "letters" / "last.npz", self.dir / "r3bad", steps=1)
        self.assertFalse((self.dir / "r3bad").exists())

    def test_a_teacher_with_another_reading_window_is_rejected(self):
        _, tokenizer = self.model()
        teacher = studio_rl.Teacher(self.ckpt)
        studio_rl.check_teacher(tokenizer, 32, teacher)                       # the same checkpoint is a fine teacher
        with self.assertRaisesRegex(ValueError, "reads 32 tokens"):
            studio_rl.check_teacher(tokenizer, 64, teacher)

    def test_another_version_with_the_same_word_list_can_teach(self):
        other = train({**TINY, "steps": 3, "seed": 5}, EXAMPLES, self.dir / "stronger")
        self.assertTrue(other)
        record = studio_rl.learn_from_teacher(self.ckpt, [f"Copy: {w}" for w in WORDS[:2]], self.dir / "stronger" / "last.npz",
                                              self.dir / "r3other", steps=2, learning_rate=1e-3)
        self.assertEqual((record["status"], record["privileged"]), ("completed", False))

    def test_learn_from_teacher_runs(self):
        events = []
        record = studio_rl.learn_from_teacher(self.ckpt, [f"Copy: {w}" for w in WORDS[:4]], studio_rl.privileged(self.ckpt, "Copy the word."),
                                              self.dir / "r3", steps=3, learning_rate=1e-3, emit=events.append)
        heat = next(e for e in events if e["type"] == "heat")
        self.assertEqual(len(heat["words"]), len(heat["surprise"]))
        self.assertLessEqual(record["moved"], record["words"])
        self.assertTrue((self.dir / "r3").joinpath("last.npz").exists())

    def privileged_round(self, name):
        """A short round 3 with the same model and a hint as teacher: (losses per step, record)."""
        events = []
        record = studio_rl.learn_from_teacher(self.ckpt, [f"Copy: {w}" for w in WORDS[:4]], studio_rl.privileged(self.ckpt, "Copy the word."),
                                              self.dir / name, steps=4, learning_rate=1e-3, emit=events.append)
        return [e["loss"] for e in events if e["type"] == "metric"], record

    def assert_unchanged_round(self, losses, record):
        """The numbers from before the teacher left the device between phases (it scored next to the student the whole round)."""
        for got, want in zip(losses, (.044044, .091985, -.053134, -.060713)):
            self.assertAlmostEqual(got, want, delta=1e-3)
        self.assertAlmostEqual(record["kl_before"], .092134, delta=1e-3)
        self.assertAlmostEqual(record["kl_after"], -.025152, delta=1e-3)
        self.assertEqual((record["moved"], record["words"]), (8, 34))

    def test_round_3_losses_do_not_depend_on_where_the_teacher_ran(self):
        self.assert_unchanged_round(*self.privileged_round("r3cpu"))

    def test_the_teacher_is_off_the_device_while_the_student_goes_backward(self):
        for kind, reloads in (("cuda", 0), ("mps", 4)):
            with self.subTest(kind):
                model, live = watched_models(kind)
                loads = []
                load = studio_rl.load_checkpoint
                with mock.patch.object(studio_rl, "ScratchTransformer", model), mock.patch.object(studio_rl, "empty_cache") as emptied, \
                        mock.patch.object(studio_rl, "load_checkpoint", lambda *a, **k: loads.append(a[0]) or load(*a, **k)), \
                        counting_backward(live) as resident:
                    losses, record = self.privileged_round(f"r3{kind}")
                self.assertEqual(resident, [0] * 4)   # four steps, and the teacher was not in memory at any of their backward passes
                self.assertTrue(emptied.called and all(call.args == (kind,) for call in emptied.call_args_list))
                self.assertEqual(len(loads), 2 + reloads)   # the teacher's and the student's loads (the teacher then borrows the student's), plus on mps one reload per later score
                self.assert_unchanged_round(losses, record)
                self.assertEqual(len(live), 0)

    def test_a_cpu_teacher_stays_built(self):
        model, live = watched_models()
        with mock.patch.object(studio_rl, "ScratchTransformer", model):
            teacher = studio_rl.Teacher(self.ckpt)
            _, tokenizer = self.model()
            teacher.score(self.rows(tokenizer, ["yes"]))
            first = teacher.model
            teacher.score(self.rows(tokenizer, ["no"]))
        self.assertIsNotNone(first)
        self.assertIs(teacher.model, first)
        self.assertEqual(len(live), 1)

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
    def test_round_2_is_skipped_when_the_model_solves_nothing(self):
        problems = _generate_problems(10, __import__("random").Random(42))
        events = []
        context = SimpleNamespace(run_dir=self.dir / "runs", post_rows=lambda kind: [])
        job = SimpleNamespace(id="skip", params={"round": 2, "checkpoint": str(self.ckpt), "output_dir": str(self.dir / "skipped"),
                                                 "problems": problems})
        none_solved = {"problems": 8, "k": 4, "solved": 0, "passed": 0, "answers": 32}
        with mock.patch.object(studio_rl, "pass_at_k", return_value=none_solved):
            done = studio_rl.run_round(2, job, context, events.append, threading.Event(), threading.Event())
        self.assertEqual((done["ok"], done["checkpoint"]), (False, str(self.ckpt)))
        self.assertFalse((self.dir / "skipped").exists())                    # nothing trained
        gate = next(e for e in events if e["type"] == "gate")
        self.assertTrue(gate["skipped"] and not gate["ok"])
        self.assertIn("teach nothing", gate["verdict"])
        self.assertEqual(gate["checks"][0]["detail"], "0 of 32 sampled answers passed, on 8 problems")
        self.assertEqual(job.params["result"], {k: v for k, v in gate.items() if k != "type"})   # left for the app to record; it never moves Chat
        self.assertFalse(job.params["result"]["ok"])

    def test_round_1_keeps_a_lora_versions_settings_unless_the_job_says_otherwise(self):
        lora = {"rank": 4, "alpha": 8, "dropout": 0, "targets": ["attn"]}
        context = SimpleNamespace(run_dir=self.dir / "runs", post_rows=lambda kind: [])

        def trained_with(config):
            job = SimpleNamespace(id="r1", params={"round": 1, "checkpoint": str(self.ckpt), "output_dir": str(self.dir / "r1"), "config": config})
            with mock.patch.object(studio_rl, "load_checkpoint", return_value={"config": {"width": 32, "lora": lora}}), \
                 mock.patch.object(studio_rl, "measure", return_value={"bpc": None, "pass_k": 0, "pass_n": 0}), \
                 mock.patch.object(studio_rl, "train", return_value={"checkpoint": str(self.ckpt)}) as fit:
                studio_rl.run_round(1, job, context, lambda event: None, threading.Event(), threading.Event())
            return fit.call_args.args[0]

        self.assertEqual(trained_with({})["lora"], lora)
        self.assertIsNone(trained_with({"lora": None})["lora"])

    def test_the_app_records_a_rounds_result_once_it_ends(self):
        from studio_ia.services import WorkspaceServices
        recorded = []
        services = object.__new__(WorkspaceServices)
        services.record_round = lambda number, version, result: recorded.append((number, version, result["ok"]))
        job = Job("post_round", "r", params={"version": 3, "result": {"round": 2, "ok": True}})
        services.runner = SimpleNamespace(jobs=[job])
        services._record_job("job", {"job": job.id, "status": "running", "error": ""})
        self.assertEqual(recorded, [])
        services._record_job("job", {"job": job.id, "status": "done", "error": ""})
        self.assertEqual(recorded, [(2, 3, True)])

    def test_post_round_handler_completes_through_a_runner(self):
        problems = _generate_problems(10, __import__("random").Random(42))
        events = []
        runner = Runner(self.dir / "data" / "queue.json", {"post_round": one_problem_solved}, tick=.01)  # in a worker process
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
        self.assertEqual(job.params["result"]["round"], 2)   # the result came back from the worker for the app to record
        gate = next(e for e in events if e.get("type") == "gate")
        self.assertEqual(gate["round"], 2)
        self.assertLessEqual(gate["taught"], gate["groups"])
        self.assertEqual(len(gate["checks"]), 2)
        self.assertEqual(gate["after"]["pass_n"], 4)
        self.assertIsInstance(gate["ok"], bool)


if __name__ == "__main__":
    unittest.main()
