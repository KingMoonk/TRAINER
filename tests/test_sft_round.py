"""Round 1 (answers from conversations): its own SFT recipe, the candidate it hands the gate, and the stoppable measurement."""
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import studio_eval
import studio_rl
from studio_model import load_checkpoint, train

WORDS = ("yes", "no", "cat", "dog", "sun", "red", "sky", "big")
CONVERSATIONS = [{"user": f"Copy: {w}", "assistant": w} for w in WORDS] * 3
TINY = dict(device="cpu", width=32, layers=1, heads=2, context=32, expansion=2, dropout=0, batch_size=4,
            steps=20, learning_rate=.004, warmup=0, validation=0, eval_every=10, objective="assistant")
NO_ROWS = {"bpc": None, "conv_loss": None, "eos_rate": None, "pass_k": 0, "pass_n": 0}


class SftRecipe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        train(TINY, CONVERSATIONS, cls.dir / "base")
        cls.ckpt = cls.dir / "base" / "last.npz"
        cls.data = load_checkpoint(cls.ckpt)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def recipe(self, rows=CONVERSATIONS, base=None, **overrides):
        return studio_rl.sft_config(dict(self.data["config"], **(base or {})), rows, self.data["tokenizer"], **overrides)

    def test_the_rate_is_a_tenth_of_the_pretraining_peak_and_never_above_1e_4(self):
        self.assertEqual(self.recipe(base={"learning_rate": 3e-3})["learning_rate"], 1e-4)    # a high peak is capped
        self.assertAlmostEqual(self.recipe(base={"learning_rate": 5e-5})["learning_rate"], 5e-6)   # a low one gives a tenth
        self.assertEqual(self.recipe(base={"learning_rate": 1e-3})["learning_rate"], 1e-4)

    def test_steps_are_epochs_over_the_conversation_tokens_up_to_a_cap(self):
        from studio_model import Tokenizer, encode_example
        tokenizer = Tokenizer.restore(self.data["tokenizer"])
        tokens = sum(len(encode_example(r, tokenizer)[0]) for r in CONVERSATIONS)
        cfg = self.recipe()
        batch, context = self.data["config"]["batch_size"], self.data["config"]["context"]
        self.assertEqual(cfg["steps"], -(-studio_rl.SFT_EPOCHS * tokens // (batch * context)))
        self.assertEqual(cfg["warmup"], max(1, round(.03 * cfg["steps"])))
        self.assertEqual(cfg["eval_every"], max(1, cfg["steps"] // 10))
        self.assertEqual(cfg["sample_prompt"], studio_rl.SFT_SAMPLE_PROMPT)
        self.assertEqual(cfg["objective"], "assistant")
        self.assertEqual(self.recipe(CONVERSATIONS * 5000)["steps"], studio_rl.SFT_MAX_STEPS)   # a big set stops at the cap
        self.assertEqual(self.recipe([])["steps"], 1)

    def test_conversations_are_held_out_of_training_for_the_evaluation(self):
        self.assertEqual(self.recipe()["validation"], .1)                  # the base held none back (validation 0)
        self.assertEqual(self.recipe(CONVERSATIONS[:5])["validation"], 0)  # too few to spare any

    def test_the_jobs_own_settings_win(self):
        cfg = self.recipe(steps=7, learning_rate=2e-3, warmup=0, sample_prompt="Hi")
        self.assertEqual((cfg["steps"], cfg["learning_rate"], cfg["warmup"], cfg["sample_prompt"]), (7, 2e-3, 0, "Hi"))
        self.assertEqual(cfg["eval_every"], 1)                             # derived from the overridden steps


class RoundOne(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        train(TINY, CONVERSATIONS, cls.dir / "base")
        cls.ckpt = cls.dir / "base" / "last.npz"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_it(self, name, trained=None, rows=CONVERSATIONS, docs=(), params=None, measured=None, stop=None, held=()):
        out = self.dir / name
        job = SimpleNamespace(id=name, params={"round": 1, "checkpoint": str(self.ckpt), "output_dir": str(out), **(params or {})})
        context = SimpleNamespace(run_dir=self.dir / "runs", post_rows=lambda kind: list(rows) if kind == "conversations" else list(docs) if kind == "heldout_docs" else list(held) if kind == "heldout_conversations" else [])
        events = []

        def fake_train(cfg, examples, directory, **kwargs):
            if trained:
                trained(Path(directory))
            return {"checkpoint": str(self.ckpt)}
        with mock.patch.object(studio_rl, "train", side_effect=fake_train) as fit, \
                mock.patch.object(studio_rl, "measure", side_effect=measured or (lambda *a, **k: dict(NO_ROWS))) as measure:
            done = studio_rl.run_round(1, job, context, events.append, stop or threading.Event(), threading.Event())
        return done, job, fit, measure, events

    def test_with_no_overrides_it_trains_with_the_sft_recipe_and_keeps_the_job_limits(self):
        _, _, fit, _, _ = self.run_it("recipe", params={"token_budget": 5000, "wall_seconds": 90})
        cfg = fit.call_args.args[0]
        self.assertLessEqual(cfg["learning_rate"], 1e-4)
        self.assertLessEqual(cfg["learning_rate"], .1 * TINY["learning_rate"])
        self.assertLessEqual(cfg["steps"], studio_rl.SFT_MAX_STEPS)
        self.assertEqual(cfg["sample_prompt"], studio_rl.SFT_SAMPLE_PROMPT)
        self.assertEqual((fit.call_args.kwargs["token_budget"], fit.call_args.kwargs["wall_seconds"]), (5000, 90))
        self.assertEqual(fit.call_args.kwargs["init_from"], str(self.ckpt))

    def test_job_params_override_the_recipe(self):
        _, _, fit, _, _ = self.run_it("override", params={"steps": 3, "learning_rate": 2e-3, "config": {"warmup": 0}})
        cfg = fit.call_args.args[0]
        self.assertEqual((cfg["steps"], cfg["learning_rate"], cfg["warmup"]), (3, 2e-3, 0))

    def test_the_candidate_is_best_npz_when_the_run_wrote_one(self):
        done, job, _, measure, _ = self.run_it("best", trained=lambda directory: (directory / "best.npz").parent.mkdir(parents=True, exist_ok=True) or (directory / "best.npz").write_bytes(b""))
        self.assertEqual(done["checkpoint"], str(self.dir / "best" / "best.npz"))
        self.assertEqual(measure.call_args_list[-1].args[0], str(self.dir / "best" / "best.npz"))
        self.assertEqual(job.params["result"]["checkpoint"], done["checkpoint"])

    def test_round_1_measures_the_held_back_conversations_and_the_gate_gets_its_number(self):
        held = [{"user": "Copy: yes", "assistant": "yes"}]
        with mock.patch.object(studio_rl.studio_gate if hasattr(studio_rl, "studio_gate") else __import__("studio_gate"), "passes", return_value=(True, [])) as gate:
            _, _, _, measure, _ = self.run_it("heldconv", held=held)
        self.assertEqual(measure.call_args_list[0].args[4], held)
        self.assertEqual(gate.call_args.kwargs["number"], 1)

    def test_the_candidate_is_the_runs_checkpoint_without_a_best(self):
        done, _, _, _, _ = self.run_it("nobest")
        self.assertEqual(done["checkpoint"], str(self.ckpt))

    def test_held_back_text_is_scored_twice_and_the_band_comes_from_the_first(self):
        docs = [{"user": "Copy: yes", "assistant": "yes"}, "Copy: no no", "Copy: cat cat cat"]
        real = studio_eval.score_docs
        with mock.patch.object(studio_eval, "score_docs", side_effect=real) as scored:
            done, job, _, _, _ = self.run_it("twice", docs=docs, measured=studio_rl_measure_with_scores)
        self.assertEqual(scored.call_count, 2)    # before (the band reads these scores too), then the candidate
        self.assertEqual(len(job.params["result"]["band"]), 2)
        self.assertLessEqual(*job.params["result"]["band"])

    def test_a_stop_while_measuring_ends_the_round_without_a_result(self):
        stop = threading.Event()
        stop.set()
        done, job, fit, _, events = self.run_it("stopped", docs=["Copy: yes"], stop=stop)
        self.assertTrue(done["stopped"] and not done["ok"])
        self.assertNotIn("result", job.params)
        self.assertEqual(events, [])
        fit.assert_not_called()

    def test_the_verdict_follows_the_gate(self):
        check = {"name": "Held-back text: error did not rise", "detail": "", "ok": False}
        self.assertEqual(studio_rl.verdict_for("Answers now look like answers.", True, []), "Answers now look like answers.")
        failed = studio_rl.verdict_for("Answers now look like answers.", False, [check, dict(check, name="Other", ok=None)])
        self.assertNotIn("look like answers", failed)
        self.assertIn("held-back text: error did not rise", failed)
        self.assertNotIn("other", failed)

    def test_round_1_s_verdict_in_the_gate_event_is_the_failed_one_when_the_gate_fails(self):
        worse = iter([{"bpc": 2.318, "conv_loss": 3., "eos_rate": .5, "pass_k": 0, "pass_n": 0},
                      {"bpc": 2.673, "conv_loss": 1., "eos_rate": .9, "pass_k": 0, "pass_n": 0}])
        with mock.patch.object(studio_eval, "band_from_scores", return_value=(2.3, 2.33)):   # the band capped at 2.330, as in the audit
            done, job, _, _, events = self.run_it("verdict", docs=["Copy: yes", "Copy: no"], measured=lambda *a, **k: next(worse))
        self.assertFalse(done["ok"])
        self.assertNotIn("look like answers", job.params["result"]["verdict"])
        self.assertEqual(events[-1]["verdict"], job.params["result"]["verdict"])


def studio_rl_measure_with_scores(*args, **kwargs):
    """measure() as the real one scores the held-back text (the scores handed in are reused), the model-free parts left out."""
    checkpoint, docs = args[0], args[1]
    scores = kwargs.get("scores") if kwargs.get("scores") is not None else (args[5] if len(args) > 5 else None)
    scores = scores if scores is not None else studio_eval.score_docs(checkpoint, [studio_rl._doc(d) for d in docs])
    return dict(NO_ROWS, bpc=studio_eval.corpus_bpc(scores))


class Measure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        train(TINY, CONVERSATIONS, Path(cls.tmp.name) / "base")
        cls.ckpt = Path(cls.tmp.name) / "base" / "last.npz"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_it_returns_bpc_conv_loss_and_eos_rate(self):
        held = [{"user": "Copy: yes", "assistant": "yes"}, {"messages": [{"role": "user", "content": "Copy: no"}, {"role": "assistant", "content": "no"}]}]
        got = studio_rl.measure(self.ckpt, ["Copy: yes yes", "Copy: no no"], [], heldout_conversations=held)
        self.assertEqual(set(got), {"bpc", "conv_loss", "eos_rate", "pass_k", "pass_n", "letters"})
        self.assertGreater(got["bpc"], 0)
        self.assertGreater(got["conv_loss"], 0)
        self.assertTrue(0 <= got["eos_rate"] <= 1)

    def test_without_held_back_conversations_those_two_are_none(self):
        got = studio_rl.measure(self.ckpt, ["Copy: yes yes"], [])
        self.assertEqual((got["conv_loss"], got["eos_rate"]), (None, None))
        self.assertIsNotNone(got["bpc"])
        self.assertIsNone(studio_rl.measure(self.ckpt, [], [])["bpc"])

    def test_scores_in_hand_are_not_computed_again(self):
        scores = studio_rl.doc_scores(self.ckpt, ["Copy: yes yes"])
        with mock.patch.object(studio_eval, "score_docs", side_effect=AssertionError("scored again")):
            got = studio_rl.measure(self.ckpt, ["Copy: yes yes"], [], scores=scores)
        self.assertEqual(got["bpc"], studio_eval.corpus_bpc(scores))

    def test_it_honours_stop(self):
        stop = threading.Event()
        stop.set()
        row = {"user": "Copy: yes", "assistant": "yes"}
        for kwargs in ({"docs": ["Copy: yes yes"]}, {"docs": [], "heldout_conversations": [row]},
                       {"docs": [], "problems": [{"rule": "copy", "prompt": "Copy: yes"}]}):
            with self.subTest(kwargs), self.assertRaises(studio_eval.Stopped):
                studio_rl.measure(self.ckpt, kwargs["docs"], kwargs.get("problems", []), stop=stop, heldout_conversations=kwargs.get("heldout_conversations"))


if __name__ == "__main__":
    unittest.main()
