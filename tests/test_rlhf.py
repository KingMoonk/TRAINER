"""DPO/IPO gradients, preference storage, best-of-N pairs, gating and the teacher protocol (Ollama is mocked)."""
import json
import random
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import studio_rlhf
from tests.torchcheck import counting_backward, to_np, watched_models
from studio_model import train
from studio_rlhf import (DEFAULT_RUBRIC, OllamaTeacher, PreferenceStore, TeacherError, _encode_pair, _sequence_logps,
                         auto_rlhf, compare_checkpoints, dpo_train, evaluate_checkpoint, load_settings, preference_batch,
                         preference_terms, prompts_from, reference_logps, save_settings, teacher_pairs, temperature_spread)

EXAMPLES = [{"user": f"say {w}", "assistant": w} for w in ("yes", "no", "cat", "dog", "sun", "red", "sky", "big")] * 3
TINY = dict(device="cpu", width=32, layers=1, heads=2, context=32, expansion=2, dropout=0, batch_size=4,
            steps=30, learning_rate=.004, warmup=0, validation=0, eval_every=15, objective="assistant")
PROMPT = {"messages": [{"role": "user", "content": "say yes"}], "reference": "yes", "source_id": "ex1"}
CFG = {**studio_rlhf.AUTO_DEFAULTS, "candidates": 4, "max_tokens": 8}


def pretrained(directory, name="base", steps=30):
    train({**TINY, "steps": steps}, EXAMPLES, Path(directory) / name)
    return Path(directory) / name / "last.npz"


class ScriptedModel:
    """Stands in for LocalModel: returns the scripted replies in order, one per generate() call."""

    def __init__(self, replies):
        self.replies, self.calls = list(replies), 0

    def generate(self, messages, max_tokens=8, temperature=.8, top_k=40, stop=None):
        reply = self.replies[self.calls % len(self.replies)] if self.calls < len(self.replies) else f"extra{self.calls}"
        self.calls += 1
        return reply

    def generate_group(self, messages, n, **kwargs):
        return [self.generate(messages) for _ in range(n)]


class ScriptedTeacher(OllamaTeacher):
    """Grades by lookup: batch/single scoring from `grades`, pairwise judging by the same table."""

    def __init__(self, grades, rewrite="the right answer", rewrite_grade=9, **kwargs):
        super().__init__(model="mock", retry_delay=0, **kwargs)
        self.grades, self.rewrite_text, self.rewrite_grade = dict(grades), rewrite, rewrite_grade
        self.systems = []
        self.batch_calls = 0

    def grade(self, text):
        return self.rewrite_grade if text == self.rewrite_text else self.grades.get(text, 1)

    def chat(self, system, user, json_mode=True, temperature=0.):
        self.systems.append(system)
        if "ideal assistant reply" in system:
            return {"answer": self.rewrite_text}
        if "numbered replies" in system:
            self.batch_calls += 1
            blocks = [b.split("\n", 1)[1] for b in user.split("\n\nREPLY ")[1:]] if "\n\nREPLY " in user else []
            return {"scores": [self.grade(b.strip()) for b in blocks], "reason": "scripted"}
        if "score_a" in system:
            a = user.split("REPLY A\n", 1)[1].split("\n\nREPLY B\n", 1)
            return {"score_a": self.grade(a[0].strip()), "score_b": self.grade(a[1].strip()), "reason": "scripted"}
        return {"score": self.grade(user.split("REPLY\n", 1)[1].strip())}


class RLHFChecks(unittest.TestCase):
    # ------------------------------------------------------------------ gradients

    def test_dpo_gradient_matches_finite_difference(self):
        from studio_model import ScratchTransformer, Tokenizer, batch_arrays, load_checkpoint
        with tempfile.TemporaryDirectory() as directory:
            data = load_checkpoint(pretrained(directory))
            cfg = dict(data["config"], dropout=0)
            tokenizer = Tokenizer.restore(data["tokenizer"])
            model = ScratchTransformer(cfg, tokenizer.vocab_size)
            model.load_weights(data["model"])
            rows = [_encode_pair(tokenizer, [{"role": "user", "content": "say yes"}], "yes", 32),
                    _encode_pair(tokenizer, [{"role": "user", "content": "say yes"}], "no", 32)]
            x, y = batch_arrays(rows)
            weights = torch.tensor([.7, -.4])  # arbitrary d(loss)/d(logp) per sequence

            def objective():
                logps, _ = _sequence_logps(model, x, y)
                return float((weights * logps).sum())

            logps, (_, _, _, cache) = _sequence_logps(model, x, y, tape=True)
            model.zero_grad()
            # backward_loss accumulates d(-objective); compare sign-flipped
            model.backward_loss(-(weights * logps).sum(), cache)
            for key in ("block0.f1.w", "token", "final.g"):
                index = tuple(np.unravel_index(np.argmax(np.abs(to_np(model.g[key]))), model.g[key].shape))
                original = float(model.p[key][index])
                with torch.no_grad():
                    model.p[key][index] = original + 1e-2
                plus = objective()
                with torch.no_grad():
                    model.p[key][index] = original - 1e-2
                minus = objective()
                with torch.no_grad():
                    model.p[key][index] = original
                numeric = -(plus - minus) / 2e-2
                self.assertAlmostEqual(float(model.g[key][index]), numeric, delta=max(2e-3, abs(numeric) * .05), msg=key)

    def test_preference_loss_gradient_variants(self):
        """d(loss)/d(logp) is exact for DPO and IPO, with and without length normalization and SFT."""
        rng = np.random.default_rng(3)
        count = 3
        pl, rl = rng.normal(-12, 2, 2 * count), rng.normal(-12, 2, 2 * count)
        lengths = rng.integers(2, 11, 2 * count).astype(float)
        for loss_type in ("dpo", "ipo"):
            for normalize in (False, True):
                for sft in (0., .3):
                    cfg = {"beta": .4, "loss_type": loss_type, "length_normalize": normalize, "sft_weight": sft}
                    _, g = preference_terms(pl, rl, lengths, count, cfg)
                    for k in range(2 * count):
                        step = np.zeros(2 * count)
                        step[k] = 1e-6
                        plus = preference_terms(pl + step, rl, lengths, count, cfg)[0]["loss"]
                        minus = preference_terms(pl - step, rl, lengths, count, cfg)[0]["loss"]
                        self.assertAlmostEqual(g[k], (plus - minus) / 2e-6, places=5, msg=f"{loss_type} norm={normalize} sft={sft} k={k}")

    def test_ipo_formula(self):
        pl, rl = np.array([-5., -9.]), np.array([-6., -8.])
        metrics, _ = preference_terms(pl, rl, np.array([1., 1.]), 1, {"beta": .5, "loss_type": "ipo", "length_normalize": False, "sft_weight": 0})
        self.assertAlmostEqual(metrics["loss"], (2 - 1 / (2 * .5)) ** 2 * 1.0)  # raw = (1) - (-1) = 2

    def test_length_normalization_removes_length_advantage(self):
        # the same log-ratio spread over twice the tokens counts half when normalized
        pl, rl = np.array([-10., -20.]), np.array([-12., -20.])
        lengths = np.array([4., 8.])
        plain = preference_terms(pl, rl, lengths, 1, {"beta": 1., "loss_type": "dpo", "length_normalize": False, "sft_weight": 0})[0]
        normed = preference_terms(pl, rl, lengths, 1, {"beta": 1., "loss_type": "dpo", "length_normalize": True, "sft_weight": 0})[0]
        self.assertAlmostEqual(plain["margin"], 2.)
        self.assertAlmostEqual(normed["margin"], 2 / 4)
        self.assertEqual(normed["chosen_len"], 4.)
        self.assertEqual(normed["rejected_len"], 8.)

    def test_model_gradient_for_ipo_and_length_normalized(self):
        from studio_model import ScratchTransformer, Tokenizer, load_checkpoint
        with tempfile.TemporaryDirectory() as directory:
            data = load_checkpoint(pretrained(directory))
            cfg = dict(data["config"], dropout=0)
            tokenizer = Tokenizer.restore(data["tokenizer"])
            policy, reference = (ScratchTransformer(cfg, tokenizer.vocab_size) for _ in range(2))
            policy.load_weights(data["model"])
            reference.load_weights(data["model"])
            noise = np.random.default_rng(1).normal(0, .05, policy.p["token"].shape).astype(np.float32)
            with torch.no_grad():
                policy.p["token"].add_(torch.as_tensor(noise))  # policy != reference so the gradient is not degenerate
            messages = [{"role": "user", "content": "say yes"}]
            rows = [_encode_pair(tokenizer, messages, "yes", 32), _encode_pair(tokenizer, messages, "yes yes", 32),
                    _encode_pair(tokenizer, messages, "no", 32), _encode_pair(tokenizer, messages, "no no no", 32)]
            reference_logp = reference_logps(reference, rows)
            for loss_type, normalize in (("dpo", True), ("ipo", False), ("ipo", True)):
                config = {"beta": .3, "loss_type": loss_type, "length_normalize": normalize, "sft_weight": .2}
                policy.zero_grad()
                preference_batch(policy, reference_logp, rows, 2, config, backward=True)
                for key in ("block0.f1.w", "token", "final.g"):
                    index = tuple(np.unravel_index(np.argmax(np.abs(to_np(policy.g[key]))), policy.g[key].shape))
                    original = float(policy.p[key][index])
                    with torch.no_grad():
                        policy.p[key][index] = original + 1e-2
                    plus = preference_batch(policy, reference_logp, rows, 2, config)["loss"]
                    with torch.no_grad():
                        policy.p[key][index] = original - 1e-2
                    minus = preference_batch(policy, reference_logp, rows, 2, config)["loss"]
                    with torch.no_grad():
                        policy.p[key][index] = original
                    numeric = (plus - minus) / 2e-2
                    self.assertAlmostEqual(float(policy.g[key][index]), numeric, delta=max(3e-3, abs(numeric) * .06), msg=f"{loss_type} {normalize} {key}")

    def test_dpo_learns_preferences_and_readiness(self):
        with tempfile.TemporaryDirectory() as directory:
            base = pretrained(directory)
            report = evaluate_checkpoint(base, EXAMPLES, device="cpu")
            self.assertTrue(0 <= report["token_error"] <= 1)
            store = PreferenceStore(directory)
            for word, other in [("yes", "no"), ("cat", "dog"), ("sun", "sky"), ("red", "big")]:
                store.add([{"role": "user", "content": f"say {word}"}], word.upper(), word, "human")
            pairs = store.all()
            self.assertEqual(store.stats()["human"], 4)
            result = dpo_train(base, pairs, Path(directory) / "dpo", {"steps": 40, "learning_rate": 3e-3, "device": "cpu", "validation": 0})
            self.assertEqual(result["status"], "completed")
            history = [json.loads(line) for line in (Path(result["directory"]) / "metrics.jsonl").read_text().splitlines()]
            self.assertLess(history[-1]["loss"], history[0]["loss"])
            self.assertEqual(history[-1]["reward_accuracy"], 1.0)
            self.assertIn("chosen_len", history[-1])
            from studio_model import LocalModel
            LocalModel(result["checkpoint"], device="cpu")  # loads as an ordinary checkpoint

    def dpo_losses(self, directory, base, **settings):
        """(training losses, validation losses) of a short run on 12 pairs; the numbers below come from the reference model scoring
        every batch next to the policy, before it was scored once up front."""
        store = PreferenceStore(directory)
        for word in ("yes", "no", "cat", "dog", "sun", "red", "sky", "big", "yes", "no", "cat", "dog"):
            store.add([{"role": "user", "content": f"say {word}"}], word, word + " " + word + " " + word, "human", note=word)
        result = dpo_train(base, store.all(), Path(directory) / "dpo",
                           {"steps": 6, "eval_every": 3, "learning_rate": 2e-3, "device": "cpu", "validation": .2, "beta": .5, **settings})
        rows = [json.loads(line) for line in (Path(result["directory"]) / "metrics.jsonl").read_text().splitlines()]
        return [r["loss"] for r in rows], [r["val_loss"] for r in rows if r["val_loss"] is not None]

    def test_dpo_losses_do_not_depend_on_where_the_reference_ran(self):
        with tempfile.TemporaryDirectory() as directory:
            losses, validation = self.dpo_losses(directory, pretrained(directory))
        for got, want in zip(losses, (1.27481, 1.30112, .67983, .67175, .64724, .58553, .58737)):
            self.assertAlmostEqual(got, want, delta=1e-3)
        for got, want in zip(validation, (1.27481, .66271, .56367)):
            self.assertAlmostEqual(got, want, delta=1e-3)

    def test_steps_scale_with_the_pair_count_and_the_default_rate_is_small(self):
        self.assertEqual([studio_rlhf.dpo_steps(n, 4) for n in (20, 21, 100, 1000)], [15, 16, 75, 300])
        with tempfile.TemporaryDirectory() as directory:
            base = pretrained(directory)
            store = PreferenceStore(directory)
            for word in ("yes", "no", "cat", "dog", "sun", "red", "sky", "big"):
                store.add([{"role": "user", "content": f"say {word}"}], word, word + " " + word + " " + word, "human", note=word)
            result = dpo_train(base, store.all(), Path(directory) / "dpo", {"device": "cpu", "eval_every": 3})
        self.assertEqual(result["steps"], 6)   # ceil(3 x 8 pairs / batch 4)
        self.assertEqual(result["dpo"]["learning_rate"], studio_rlhf.DPO_LR_FULL)
        self.assertEqual(result["validation_pairs"], 2)   # 10 % of 8 pairs is 0, but at least 2 are always held back

    def test_the_shipped_checkpoint_is_the_best_one_only_when_pairs_were_held_back(self):
        with tempfile.TemporaryDirectory() as directory:
            base = pretrained(directory)
            store = PreferenceStore(directory)
            for word in ("yes", "no", "cat", "dog", "sun", "red"):
                store.add([{"role": "user", "content": f"say {word}"}], word, word + " " + word + " " + word, "human", note=word)
            for validation, name in ((.2, "best.npz"), (0, "last.npz")):
                result = dpo_train(base, store.all(), Path(directory) / name, {"steps": 4, "eval_every": 2, "device": "cpu", "validation": validation})
                self.assertEqual(Path(result["checkpoint"]).name, name)

    def test_the_reference_is_gone_before_the_policy_goes_backward(self):
        model, live = watched_models()
        with tempfile.TemporaryDirectory() as directory:
            base = pretrained(directory)
            with patch.object(studio_rlhf, "ScratchTransformer", model), counting_backward(live) as resident:
                self.dpo_losses(directory, base)
        self.assertEqual(resident, [0] * 6)

    def test_ipo_and_length_normalized_training_run(self):
        with tempfile.TemporaryDirectory() as directory:
            base = pretrained(directory)
            store = PreferenceStore(directory)
            for word in ("yes", "no", "cat", "dog", "sun", "red", "sky", "big", "yes", "no", "cat", "dog"):
                store.add([{"role": "user", "content": f"say {word}"}], word, word + " " + word + " " + word, "human", note=word)
            for loss_type in ("ipo", "dpo"):
                result = dpo_train(base, store.all(), Path(directory) / loss_type,
                                   {"steps": 12, "eval_every": 6, "learning_rate": 2e-3, "device": "cpu", "loss_type": loss_type,
                                    "length_normalize": True, "validation": .2, "beta": .5})
                self.assertEqual(result["status"], "completed")
                rows = [json.loads(line) for line in (Path(result["directory"]) / "metrics.jsonl").read_text().splitlines()]
                evals = [r for r in rows if r.get("val_chosen_len") is not None]
                self.assertTrue(evals)
                self.assertLess(evals[-1]["val_chosen_len"], evals[-1]["val_rejected_len"])
            with self.assertRaises(ValueError):
                dpo_train(base, store.all(), Path(directory) / "bad", {"loss_type": "nope", "device": "cpu"})

    # ------------------------------------------------------------------ store

    def test_store_update_flip_delete_and_stats(self):
        with tempfile.TemporaryDirectory() as directory:
            store = PreferenceStore(directory)
            prompt = [{"role": "user", "content": "q"}]
            a = store.add(prompt, "good", "bad", "human", kind="comparison", reference="good", source_id="s1")
            b = store.add(prompt, "better", "worse", "teacher:m", kind="rewrite")
            c = store.add(prompt, "x", "y", "teacher:m", kind="comparison")
            self.assertEqual(a["reference"], "good")
            updated = store.update(a["id"], chosen="great")
            self.assertEqual(updated["chosen"], "great")
            self.assertIn("editedAt", updated)
            self.assertEqual(store.get(a["id"])["chosen"], "great")
            self.assertEqual(store.get(a["id"])["id"], a["id"])
            with self.assertRaises(ValueError):
                store.update(a["id"], chosen="bad")  # equals rejected
            with self.assertRaises(ValueError):
                store.update(a["id"], chosen="  ")
            with self.assertRaises(KeyError):
                store.update("nope", chosen="z")
            flipped = store.flip(b["id"])
            self.assertEqual((flipped["chosen"], flipped["rejected"]), ("worse", "better"))
            self.assertEqual(store.flip(b["id"])["chosen"], "better")
            stats = store.stats()
            self.assertEqual((stats["total"], stats["human"], stats["teacher"], stats["rewrites"]), (3, 1, 2, 1))
            self.assertEqual(stats["by_kind"], {"comparison": 2, "rewrite": 1})
            self.assertEqual(stats["by_source"], {"human": 1, "teacher:m": 2})
            self.assertEqual(store.delete([a["id"], c["id"], "missing"]), 2)
            self.assertEqual([r["id"] for r in store.all()], [b["id"]])
            self.assertFalse(list(store.path.parent.glob("*.tmp")), "temp files must be replaced, not left behind")
            self.assertEqual(store.delete(["missing"]), 0)

    def test_store_keeps_unreadable_lines_on_rewrite(self):
        with tempfile.TemporaryDirectory() as directory:
            store = PreferenceStore(directory)
            record = store.add([{"role": "user", "content": "q"}], "a", "b", "human")
            with store.path.open("a", encoding="utf-8") as stream:
                stream.write("{not json\n")
            store.flip(record["id"])
            self.assertIn("{not json", store.path.read_text(encoding="utf-8"))
            self.assertEqual(len(store.all()), 1)

    def test_store_export_import_dedupes_by_content(self):
        with tempfile.TemporaryDirectory() as directory:
            one, two = PreferenceStore(Path(directory) / "one"), PreferenceStore(Path(directory) / "two")
            prompt = [{"role": "user", "content": "q"}]
            first = one.add(prompt, "a", "b", "human")
            one.add(prompt, "c", "d", "teacher:m", kind="rewrite")
            target = Path(directory) / "out" / "prefs.jsonl"
            self.assertEqual(one.export(target), 2)
            two.add(prompt, "a", "b", "human")  # same content as `first`, different id
            result = two.import_file(target)
            self.assertEqual(result, {"added": 1, "duplicates": 1, "invalid": 0})
            self.assertEqual(len(two.all()), 2)
            self.assertEqual(two.import_file(target)["added"], 0)  # idempotent
            self.assertEqual(len({r["id"] for r in two.all()}), 2)
            with target.open("a", encoding="utf-8") as stream:
                stream.write("garbage\n" + json.dumps({"messages": [], "chosen": "x", "rejected": "y"}) + "\n")
            self.assertEqual(two.import_file(target)["invalid"], 2)
            self.assertEqual(one.export(Path(directory) / "subset.jsonl", ids=[first["id"]]), 1)

    # ------------------------------------------------------------------ settings and rubric

    def test_rubric_settings_roundtrip_and_prompt_injection(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(load_settings(directory)["rubric"], DEFAULT_RUBRIC)
            custom = [{"text": "Answers in one word", "weight": 5}, {"text": "  ", "weight": 2}, {"text": "Polite", "weight": 99}]
            saved = save_settings(directory, {"rubric": custom, "judge_model": "j"})
            self.assertEqual([c["text"] for c in saved["rubric"]], ["Answers in one word", "Polite"])
            self.assertEqual(saved["rubric"][1]["weight"], 10)
            loaded = load_settings(directory)
            self.assertEqual(loaded["judge_model"], "j")
            self.assertTrue((Path(directory) / "data" / "rlhf_settings.json").exists())
            teacher = ScriptedTeacher({"a": 5, "b": 7}, rubric=loaded["rubric"])
            teacher.score_many(PROMPT["messages"], ["a", "b"])
            teacher.judge(PROMPT["messages"], "a", "b", both_orders=False)
            teacher.score(PROMPT["messages"], "a")
            teacher.rewrite(PROMPT["messages"])
            self.assertEqual(len(teacher.systems), 4)
            for system in teacher.systems:
                self.assertIn("Answers in one word (weight 5)", system)
                self.assertNotIn("{criteria}", system)

    def test_temperature_spread(self):
        self.assertEqual(len(temperature_spread(4)), 4)
        spread = temperature_spread(5)
        self.assertEqual(spread, sorted(spread))
        self.assertEqual(temperature_spread(3, [.7, 1.1, .9]), [.7, 1.1, .9])
        self.assertEqual(temperature_spread(3, [.7, 1.1]), [.7, .9, 1.1])

    # ------------------------------------------------------------------ best-of-N

    def test_best_of_n_pairs_best_vs_worst(self):
        teacher = ScriptedTeacher({"good": 9, "medium": 6, "meh": 5, "bad": 2})
        model = ScriptedModel(["medium", "good", "bad", "meh"])
        pairs, info = teacher_pairs(teacher, model, PROMPT, CFG, random.Random(1), threading.Event())
        self.assertEqual(len(pairs), 1)
        self.assertEqual((pairs[0]["chosen"], pairs[0]["rejected"], pairs[0]["kind"]), ("good", "bad", "comparison"))
        self.assertEqual(pairs[0]["reference"], "yes")
        self.assertEqual(pairs[0]["source_id"], "ex1")
        self.assertEqual(teacher.batch_calls, 1, "all candidates are graded in one call")
        self.assertEqual(sorted(info["scores"]), [2, 5, 6, 9])
        extra, _ = teacher_pairs(teacher, ScriptedModel(["medium", "good", "bad", "meh"]), PROMPT, {**CFG, "extra_pairs": True}, random.Random(1), threading.Event())
        self.assertEqual({(p["chosen"], p["rejected"]) for p in extra}, {("good", "bad"), ("good", "medium"), ("good", "meh")})
        small = teacher_pairs(teacher, ScriptedModel(["good", "bad"]), PROMPT, {**CFG, "candidates": 2}, random.Random(1), threading.Event())[0]
        self.assertEqual((small[0]["chosen"], small[0]["rejected"]), ("good", "bad"))

    def test_top_two_tie_is_settled_pairwise(self):
        class Judging(ScriptedTeacher):
            def chat(self, system, user, json_mode=True, temperature=0.):
                if "score_a" in system:  # pairwise: prefer the reply containing "sharp"
                    a, b = user.split("REPLY A\n", 1)[1].split("\n\nREPLY B\n", 1)
                    return {"score_a": 8 if "sharp" in a else 6, "score_b": 8 if "sharp" in b else 6, "reason": "sharper"}
                return super().chat(system, user, json_mode, temperature)
        teacher = Judging({"plain one": 8, "sharp one": 8, "poor": 3, "weak": 4})
        pairs, info = teacher_pairs(teacher, ScriptedModel(["plain one", "sharp one", "poor", "weak"]), PROMPT, CFG, random.Random(2), threading.Event())
        self.assertEqual(pairs[0]["chosen"], "sharp one")
        self.assertEqual(pairs[0]["rejected"], "poor")

    def test_skips_when_scores_are_too_close_or_replies_identical(self):
        teacher = ScriptedTeacher({"a": 6, "b": 6, "c": 6})
        pairs, info = teacher_pairs(teacher, ScriptedModel(["a", "b", "c", "a"]), PROMPT, CFG, random.Random(1), threading.Event())
        self.assertEqual((pairs, info["skip"]), ([], "scores too close"))
        pairs, info = teacher_pairs(teacher, ScriptedModel(["same"]), PROMPT, {**CFG, "max_tokens": 4}, random.Random(1), threading.Event())
        # ScriptedModel emits "extra<N>" after its script, so re-sampling finds distinct replies; use a constant model instead
        class Constant:
            def generate(self, *args, **kwargs):
                return "same"

            def generate_group(self, messages, n, **kwargs):
                return [self.generate(messages) for _ in range(n)]
        pairs, info = teacher_pairs(teacher, Constant(), PROMPT, CFG, random.Random(1), threading.Event())
        self.assertEqual((pairs, info["skip"]), ([], "model gave identical replies"))

    def test_rewrite_is_regraded_and_must_beat_best_candidate(self):
        low = {"p": 2, "q": 3, "r": 1, "s": 2}
        # the rewrite really is better -> used as chosen
        teacher = ScriptedTeacher(low, rewrite="the right answer", rewrite_grade=9)
        pairs, info = teacher_pairs(teacher, ScriptedModel(["p", "q", "r", "s"]), PROMPT, CFG, random.Random(1), threading.Event())
        self.assertEqual((pairs[0]["kind"], pairs[0]["chosen"], pairs[0]["rejected"]), ("rewrite", "the right answer", "r"))
        self.assertEqual(info["rewrite"]["scores"], [9, 3])
        self.assertGreaterEqual(teacher.batch_calls, 2, "the rewrite is graded again")
        # the rewrite is worse than the model's own best attempt -> ignored, plain best-vs-worst comparison
        teacher = ScriptedTeacher(low, rewrite="the right answer", rewrite_grade=2)
        pairs, info = teacher_pairs(teacher, ScriptedModel(["p", "q", "r", "s"]), PROMPT, CFG, random.Random(1), threading.Event())
        self.assertEqual((pairs[0]["kind"], pairs[0]["chosen"]), ("comparison", "q"))
        # nothing separates the candidates and the rewrite is no better -> no pair at all
        flat = {"p": 2, "q": 2, "r": 2, "s": 2}
        teacher = ScriptedTeacher(flat, rewrite="the right answer", rewrite_grade=2)
        pairs, info = teacher_pairs(teacher, ScriptedModel(["p", "q", "r", "s"]), PROMPT, CFG, random.Random(1), threading.Event())
        self.assertEqual((pairs, info["skip"]), ([], "rewrite no better than best reply"))

    # ------------------------------------------------------------------ robustness

    def test_malformed_teacher_replies_are_retried_then_reported(self):
        teacher = OllamaTeacher(model="mock", retry_delay=0, retries=2)
        replies = iter([["not a dict"], {"score": "ten"}, {"score": 7}])
        with patch.object(teacher, "chat", side_effect=lambda *a, **k: next(replies)):
            self.assertEqual(teacher.score([{"role": "user", "content": "q"}], "r"), 7.)
        with patch.object(teacher, "chat", side_effect=RuntimeError("The teacher did not return JSON: ???")) as chat:
            with self.assertRaises(TeacherError) as caught:
                teacher.score([{"role": "user", "content": "q"}], "r")
            self.assertEqual(chat.call_count, 3)
            self.assertFalse(caught.exception.fatal)
        with patch.object(teacher, "chat", side_effect=RuntimeError("Cannot reach Ollama at x")) as chat:
            with self.assertRaises(TeacherError) as caught:
                teacher.score([{"role": "user", "content": "q"}], "r")
            self.assertEqual(chat.call_count, 1)
            self.assertTrue(caught.exception.fatal)

    def test_batch_scoring_falls_back_to_single_grades(self):
        teacher = ScriptedTeacher({})
        outputs = {"numbered replies": {"scores": [5]}}  # wrong length

        def chat(system, user, json_mode=True, temperature=0.):
            for marker, value in outputs.items():
                if marker in system:
                    return value
            return {"score": 4 if "REPLY\nbb" in user else 8}
        with patch.object(teacher, "chat", side_effect=chat):
            self.assertEqual(teacher.score_many(PROMPT["messages"], ["aa", "bb", ""], rng=random.Random(0)), [8, 4, 1.0])

    def test_auto_loop_survives_bad_replies(self):
        with tempfile.TemporaryDirectory() as directory:
            base = pretrained(directory)
            teacher = ScriptedTeacher({}, rewrite="yes")
            state = {"n": 0}

            def chat(system, user, json_mode=True, temperature=0.):
                state["n"] += 1
                if "numbered replies" in system:
                    if state["n"] % 4 == 1:
                        return {"unexpected": True}  # malformed batch reply -> falls back to single grades
                    return {"scores": [9 if i == 0 else 2 for i in range(user.count("REPLY "))]}
                if state["n"] % 7 == 0:
                    raise RuntimeError("The teacher did not return JSON: oops")
                if "score_a" in system:
                    return {"score_a": 7, "score_b": 3, "reason": "A"}
                return {"score": 5}
            with patch.object(teacher, "chat", side_effect=chat):
                result = auto_rlhf(base, prompts_from(EXAMPLES), teacher, PreferenceStore(directory), Path(directory) / "runs",
                                   {"rounds": 1, "prompts_per_round": 8, "dpo_steps": 3, "eval_prompts": 2, "max_tokens": 6,
                                    "candidates": 3, "gate_samples": 1, "keep_only_improvements": False},
                                   {"device": "cpu"}, stop=threading.Event())
            self.assertIsNone(result["aborted"])
            self.assertEqual(len(result["rounds"]), 1)
            self.assertIsInstance(result["skipped"], dict)

    def test_auto_loop_stops_gracefully_when_teacher_is_down(self):
        with tempfile.TemporaryDirectory() as directory:
            base = pretrained(directory)
            teacher = OllamaTeacher(model="mock", retry_delay=0)
            events = []
            with patch.object(teacher, "chat", side_effect=RuntimeError("Cannot reach Ollama at x")):
                result = auto_rlhf(base, prompts_from(EXAMPLES), teacher, PreferenceStore(directory), Path(directory) / "runs",
                                   {"rounds": 2, "prompts_per_round": 20, "eval_prompts": 2, "max_tokens": 4, "max_failures": 3},
                                   {"device": "cpu"}, emit=events.append, stop=threading.Event())
            self.assertIn("failed 3 prompts in a row", result["aborted"])
            self.assertEqual(result["checkpoint"], str(base))
            self.assertEqual(result["skipped"], {"teacher unreachable": 3})
            self.assertEqual(sum(e["type"] == "rlhf_progress" for e in events), 3)

    # ------------------------------------------------------------------ gating

    def test_compare_checkpoints_counts_and_standalone(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = pretrained(directory, "a", 30), pretrained(directory, "b", 3)
            prompts = prompts_from(EXAMPLES)[:5]

            class Fixed:
                def __init__(self, scores):
                    self.scores, self.calls = scores, 0

                def judge(self, messages, first, second, reference=None, both_orders=True, rng=None):
                    self.calls += 1
                    return {"scores": self.scores, "reason": "fixed"}
            win = compare_checkpoints(a, b, prompts, Fixed([8, 2]), samples=2, max_tokens=4, temperature=1.2, device="cpu")
            self.assertEqual(win["n"], 10)
            self.assertEqual(win["wins"] + win["ties"], 10)
            self.assertEqual(win["losses"], 0)
            self.assertAlmostEqual(win["win_rate"], (win["wins"] + .5 * win["ties"]) / 10)
            loss = compare_checkpoints(a, b, prompts, Fixed([2, 8]), samples=2, max_tokens=4, temperature=1.2, device="cpu")
            self.assertEqual(loss["wins"], 0)
            self.assertEqual(loss["losses"] + loss["ties"], 10)
            tie = compare_checkpoints(a, a, prompts, Fixed([8, 2]), samples=1, max_tokens=4, temperature=0, device="cpu")
            self.assertEqual((tie["ties"], tie["n"], tie["win_rate"]), (5, 5, .5))
            self.assertTrue(win["examples"] and {"prompt", "a", "b", "outcome"} <= set(win["examples"][0]))

    def test_compare_checkpoints_frees_both_models_even_when_the_judge_fails(self):
        import gc
        import weakref
        made = []
        real = studio_rlhf.LocalModel

        def remember(*args, **kwargs):
            model = real(*args, **kwargs)
            made.append(weakref.ref(model))
            return model

        class Broken:
            def judge(self, *args, **kwargs):
                raise studio_rlhf.TeacherError("down", fatal=True)
        with tempfile.TemporaryDirectory() as directory:
            a, b = pretrained(directory, "a", 30), pretrained(directory, "b", 3)
            freed = []  # whether both models were gone when the cache was emptied, i.e. while the failure was still propagating
            with patch.object(studio_rlhf, "LocalModel", remember), \
                    patch.object(studio_rlhf, "empty_cache", side_effect=lambda device: freed.append((device, [ref() is None for ref in made]))):
                with self.assertRaises(studio_rlhf.TeacherError):
                    compare_checkpoints(a, b, prompts_from(EXAMPLES)[:2], Broken(), samples=1, max_tokens=4, temperature=1.2, device="cpu")
            gc.collect()
            self.assertEqual(freed, [("cpu", [True, True])])

    def test_gate_accepts_and_rejects_by_win_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            base = pretrained(directory)
            teacher = ScriptedTeacher({}, rewrite="yes")
            config = {"rounds": 3, "prompts_per_round": 6, "dpo_steps": 2, "eval_prompts": 2, "max_tokens": 6, "candidates": 3,
                      "win_margin": .1}
            outcomes = iter([.7, .55, .62])  # needs >= 0.6

            def fake_compare(candidate, current, *args, **kwargs):
                rate = next(outcomes)
                return {"wins": int(rate * 10), "ties": 0, "losses": 10 - int(rate * 10), "n": 10, "win_rate": rate, "examples": []}

            def chat(system, user, json_mode=True, temperature=0.):
                if "numbered replies" in system:
                    return {"scores": [9] + [2] * (user.count("REPLY ") - 1)}
                if "ideal assistant reply" in system:
                    return {"answer": "yes"}
                return {"score_a": 7, "score_b": 3, "reason": "A"}
            with patch.object(teacher, "chat", side_effect=chat), patch.object(studio_rlhf, "compare_checkpoints", side_effect=fake_compare):
                result = auto_rlhf(base, prompts_from(EXAMPLES), teacher, PreferenceStore(directory), Path(directory) / "runs",
                                   config, {"device": "cpu"}, stop=threading.Event())
            kept = [r["accepted"] for r in result["rounds"]]
            self.assertEqual(kept, [True, False, True])
            self.assertEqual(result["checkpoint"], result["rounds"][2]["checkpoint"])
            self.assertEqual(result["win_rate"], .62)
            self.assertNotEqual(result["checkpoint"], result["rounds"][1]["checkpoint"])
            self.assertTrue(all(r["pairs"] >= 2 for r in result["rounds"]))

    def test_gate_rejects_when_nothing_could_be_judged(self):
        with tempfile.TemporaryDirectory() as directory:
            base = pretrained(directory)
            teacher = ScriptedTeacher({}, rewrite="yes")

            def chat(system, user, json_mode=True, temperature=0.):
                if "numbered replies" in system:
                    return {"scores": [9] + [2] * (user.count("REPLY ") - 1)}
                raise RuntimeError("The teacher did not return JSON: x")  # pairwise judging always fails
            # ties in tie-break are swallowed, gate judging fails -> gate errors -> no win rate -> rejected
            with patch.object(teacher, "chat", side_effect=chat):
                result = auto_rlhf(base, prompts_from(EXAMPLES), teacher, PreferenceStore(directory), Path(directory) / "runs",
                                   {"rounds": 1, "prompts_per_round": 6, "dpo_steps": 2, "eval_prompts": 2, "max_tokens": 6,
                                    "candidates": 3, "gate_samples": 1, "temperatures": [1.5, 1.5, 1.5]},
                                   {"device": "cpu"}, stop=threading.Event())
            self.assertEqual(result["checkpoint"], str(base))
            self.assertFalse(result["rounds"][0]["accepted"])

    def test_separate_judge_model(self):
        teacher = OllamaTeacher("http://x:1", "writer", rubric=[{"text": "Short", "weight": 2}], judge_model="grader")
        judge = teacher.judge_teacher()
        self.assertEqual((judge.model, judge.url, judge.rubric), ("grader", "http://x:1", teacher.rubric))
        self.assertIs(OllamaTeacher(model="same").judge_teacher().__class__, OllamaTeacher)
        solo = OllamaTeacher(model="solo")
        self.assertIs(solo.judge_teacher(), solo)

    def test_auto_loop_with_mock_teacher(self):
        with tempfile.TemporaryDirectory() as directory:
            base = pretrained(directory)
            teacher = ScriptedTeacher({}, rewrite="yes")

            def chat(system, user, json_mode=True, temperature=0.):
                if "ideal assistant reply" in system:
                    return {"answer": "yes"}
                if "numbered replies" in system:
                    n = user.count("REPLY ")
                    return {"scores": [9 if i == 0 else 3 for i in range(n)]}
                if "score_a" in system:
                    return {"score_a": 7, "score_b": 3, "reason": "A is closer"}
                return {"score": 5}
            events = []
            with patch.object(teacher, "chat", side_effect=chat):
                result = auto_rlhf(base, prompts_from(EXAMPLES), teacher, PreferenceStore(directory), Path(directory) / "runs",
                                   {"rounds": 1, "prompts_per_round": 6, "dpo_steps": 5, "eval_prompts": 2, "max_tokens": 8, "both_orders": False,
                                    "gate_samples": 1},
                                   {"device": "cpu"}, emit=events.append, stop=threading.Event())
            self.assertEqual(len(result["rounds"]), 1)
            self.assertGreaterEqual(result["pairs"], 2)
            row = result["rounds"][0]
            self.assertIn("win_rate", row)
            self.assertEqual(row["wins"] + row["ties"] + row["losses"], row["n"])
            self.assertTrue(any(e["type"] == "rlhf_gate" for e in events))
            stored = PreferenceStore(directory).all()
            self.assertTrue(all(p["source"] == "teacher:mock" and "reference" in p for p in stored))


if __name__ == "__main__":
    unittest.main()
