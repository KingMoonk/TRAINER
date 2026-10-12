"""Functional checks for data integrity, causal attention, learning and resumption."""
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np

from studio_core import Workspace, fingerprint, normalize, parse_file
from studio_model import (BOS, DEFAULT_CONFIG, LocalModel, ScratchTransformer, Tokenizer, WindowStore, batch_arrays,
                          encode_example, load_checkpoint, samples_from, train, validate_config)


class DataChecks(unittest.TestCase):
    def test_csv_roundtrip_preserves_conversation_and_corpus(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Workspace(directory)
            dataset = workspace.create_dataset("Import check")
            records = [{"messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "café?"}, {"role": "assistant", "content": "☕"}, {"role": "user", "content": "more?"}, {"role": "assistant", "content": "yes"}]}, {"text": "Original raw prose\n第二行"}]
            self.assertEqual(workspace.add(dataset, records)[0], 2)
            destination = Path(directory) / "data.csv"
            workspace.export(dataset, destination)
            restored = [normalize(row) for row in parse_file(destination)]
            self.assertEqual([fingerprint(e) for e in workspace.examples(dataset)], [fingerprint(e) for e in restored])
            self.assertEqual(workspace.add(dataset, records), (0, 2, 0))
            self.assertEqual(len(Workspace(directory).examples(dataset)), 2)

    def test_mixed_text_and_invalid_examples(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mixed.txt"
            path.write_text("Corpus paragraph.\n\nUser: two minus two?\nAssistant: 0", encoding="utf-8")
            rows = [normalize(row) for row in parse_file(path)]
            self.assertEqual(rows[0]["text"], "Corpus paragraph.")
            self.assertEqual(rows[1]["assistant"], "0")
        self.assertIsNone(normalize({"user": "  ", "assistant": "  "}))
        self.assertEqual(normalize({"user": "two minus two", "assistant": 0})["assistant"], "0")


class TrainingChecks(unittest.TestCase):
    def config(self, **values):
        cfg = dict(DEFAULT_CONFIG, device="cpu", width=32, layers=1, heads=2, context=32, expansion=2,
                    dropout=0., batch_size=4, steps=45, learning_rate=.004, warmup=0,
                    validation=0., eval_every=15, objective="assistant")
        return dict(cfg, **values)

    def test_tokenizer_and_loss_mask(self):
        tokenizer = Tokenizer("byte")
        text = "A café ☕ 中文\n"
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
        sample = normalize({"user": "question", "assistant": "answer"})
        ids, mask = encode_example(sample, tokenizer, True)
        selected = [token for token, keep in zip(ids, mask) if keep]
        self.assertEqual(tokenizer.decode(selected), "answer")

    def test_attention_cannot_see_future_tokens(self):
        model = ScratchTransformer(self.config(), 40)
        first = np.array([[1, 3, 8, 9, 10, 11]])
        second = np.array([[1, 3, 8, 20, 21, 22]])
        a, b = model.forward(first), model.forward(second)
        np.testing.assert_allclose(a[:, :3], b[:, :3], atol=1e-6, rtol=1e-6)

    def test_handwritten_gradients_against_finite_differences(self):
        model = ScratchTransformer(dict(self.config(), width=16), 12)
        # Double precision isolates differentiation errors from float32 rounding.
        model.p = {key: value.astype(np.float64) for key, value in model.p.items()}
        model.g = {key: np.zeros_like(value) for key, value in model.p.items()}
        x = np.array([[1, 3, 8, 2, 4, 9], [1, 3, 10, 2, 4, 11]])
        y = np.array([[-100, -100, -100, -100, 9, 2], [-100, -100, -100, -100, 11, 2]])
        model.loss_and_backward(x, y)
        rng = np.random.default_rng(77)
        for key, parameter in model.p.items():
            for _ in range(3):
                direction = rng.normal(size=parameter.shape)
                direction /= np.linalg.norm(direction)
                expected = float((model.g[key] * direction).sum())
                original = parameter.copy()
                epsilon = 1e-5
                parameter[...] = original + epsilon * direction
                positive = model.loss(x, y)
                parameter[...] = original - epsilon * direction
                negative = model.loss(x, y)
                parameter[...] = original
                actual = (positive - negative) / (2 * epsilon)
                self.assertAlmostEqual(expected, actual, delta=2e-6, msg=key)

    def test_learning_checkpoint_inference_resume_and_data_guard(self):
        examples = [normalize({"user": "a", "assistant": "b"})]
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "first"
            cfg = self.config()
            result = train(cfg, examples, run_dir)
            data = load_checkpoint(run_dir / "last.npz")
            self.assertEqual(result["step"], 45)
            self.assertLess(data["history"][-1]["loss"], data["history"][0]["loss"] * .5)
            local = LocalModel(run_dir / "last.npz", device="cpu")
            self.assertEqual(local.generate([{"role": "user", "content": "a"}], 5, 0), "b")
            resumed = dict(cfg, steps=50)
            second = Path(directory) / "resumed"
            train(resumed, examples, second, resume=run_dir / "last.npz")
            after = load_checkpoint(second / "last.npz")
            self.assertEqual(after["step"], 50)
            self.assertFalse(np.array_equal(data["model"]["token"], after["model"]["token"]))
            with self.assertRaisesRegex(ValueError, "original dataset"):
                train(resumed, [normalize({"user": "changed", "assistant": "b"})], Path(directory) / "invalid", resume=run_dir / "last.npz")

    def test_cancel_saves_usable_checkpoint(self):
        stop = threading.Event()
        def event(value):
            if value.get("type") == "metric" and value.get("step") == 1:
                stop.set()
        with tempfile.TemporaryDirectory() as directory:
            result = train(self.config(), [normalize({"user": "x", "assistant": "y"})], directory, event, stop)
            self.assertEqual(result["status"], "stopped")
            self.assertEqual(load_checkpoint(Path(directory) / "last.npz")["step"], 1)

    def test_resume_restores_rng_and_optimizer_exactly(self):
        examples = [normalize({"user": "a", "assistant": "b"}), normalize({"user": "c", "assistant": "d"})]
        cfg = self.config(steps=25, dropout=.2, accumulation=2)
        stop = threading.Event()
        def event(value):
            if value.get("type") == "metric" and value.get("step") == 10:
                stop.set()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train(cfg, examples, root / "full")
            train(cfg, examples, root / "partial", emit=event, stop=stop)
            train(cfg, examples, root / "resumed", resume=root / "partial" / "last.npz")
            full = load_checkpoint(root / "full" / "last.npz")
            resumed = load_checkpoint(root / "resumed" / "last.npz")
            for key in full["model"]:
                np.testing.assert_array_equal(full["model"][key], resumed["model"][key], err_msg=key)
            self.assertEqual(full["optimizer_steps"], resumed["optimizer_steps"])
            with self.assertRaisesRegex(ValueError, "new output folder"):
                train(cfg, examples, root / "full")


class WindowingChecks(unittest.TestCase):
    def cfg(self, **values):
        return validate_config(dict(DEFAULT_CONFIG, context=32, **values))

    def chat(self, prompt, answer, extra=None):
        messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}] + (extra or [])
        return normalize({"messages": messages})

    def test_new_defaults_and_validation(self):
        cfg = validate_config({})
        self.assertEqual((cfg["tokenizer"], cfg["context"], cfg["windowing"], cfg["objective"]), ("byte", 256, "anchored", "assistant"))
        with self.assertRaises(ValueError):
            validate_config({"windowing": "diagonal"})

    def test_anchored_keeps_prompt_start_and_targets_answer_once(self):
        tokenizer, cfg = Tokenizer("byte"), self.cfg()
        store = samples_from([self.chat("p" * 5, "a" * 15)], tokenizer, cfg)  # fits one window
        self.assertIsInstance(store, WindowStore)
        self.assertEqual(len(store), 1)
        inputs, targets = store[0]
        self.assertEqual(int(inputs[0]), BOS)
        self.assertEqual(tokenizer.decode([t for t in targets if t != -100]), "a" * 15)
        # Sequential windows train the answer tail without its prompt.
        old = samples_from([self.chat("p" * 40, "a" * 20)], tokenizer, self.cfg(windowing="sequential"))
        self.assertNotEqual(int(old[0][0][0]), BOS)

    def test_long_answer_splits_with_context_and_each_token_once(self):
        tokenizer, cfg = Tokenizer("byte"), self.cfg()
        answer = "".join(chr(97 + i % 26) for i in range(200))
        store = samples_from([self.chat("question " * 3, answer)], tokenizer, cfg)
        self.assertGreater(len(store), 3)
        trained = []
        for index, (inputs, targets) in enumerate(store):
            self.assertLessEqual(len(inputs), 32)
            if index:
                self.assertEqual(len(inputs), 32)  # follow-up windows are filled with preceding context
            trained += [t for t in targets if t != -100]
        self.assertEqual(tokenizer.decode(trained), answer)
        self.assertEqual(int(store[0][0][0]), BOS)

    def test_truncated_window_keeps_role_token_and_all_objective_covers_every_token(self):
        tokenizer = Tokenizer("byte")
        window = samples_from([self.chat("p" * 150, "a" * 10)], tokenizer, self.cfg())[0][0]
        self.assertIn(int(window[0]), (3, 4))  # USER or ASSISTANT role token leads a mid-message cut
        example = self.chat("p" * 100, "a" * 100)
        store = samples_from([example], tokenizer, self.cfg(objective="all"))
        ids, mask = encode_example(example, tokenizer, False)
        trained = [t for _, targets in store for t in targets if t != -100]
        self.assertEqual(trained, [t for t, keep in zip(ids, mask) if keep])

    def test_multi_turn_answers_each_targeted_once_and_text_is_sequential(self):
        tokenizer = Tokenizer("byte")
        example = self.chat("one", "first", [{"role": "user", "content": "two"}, {"role": "assistant", "content": "second"}])
        store = samples_from([example], tokenizer, self.cfg())
        self.assertEqual(len(store), 2)
        self.assertEqual("".join(tokenizer.decode([t for t in tg if t != -100]) for _, tg in store), "firstsecond")
        self.assertEqual(len(samples_from([normalize({"text": "x" * 70})], tokenizer, self.cfg())), 3)

    def test_compact_store_matches_batch_semantics(self):
        tokenizer = Tokenizer("byte")
        store = samples_from([self.chat("q" * i, "a" * (i + 3)) for i in range(1, 30)], tokenizer, self.cfg())
        self.assertEqual(store.tokens.dtype, np.uint16)
        self.assertEqual(store.offsets.dtype, np.int64)
        x, y = batch_arrays(store[2:6])
        self.assertEqual((x.shape, x.dtype), (y.shape, np.dtype(np.int64)))
        self.assertEqual(len(list(store)), len(store))
        self.assertEqual(store.describe()["windows"], len(store))
        self.assertLess(store.nbytes, 3 * (store.tokens.size + len(store)) * 2)

    def test_legacy_checkpoint_without_windowing_resumes_sequential(self):
        import studio_model
        examples = [self.chat("a" * 40, "b" * 40)]
        cfg = dict(DEFAULT_CONFIG, device="cpu", width=32, layers=1, heads=2, context=32, expansion=2, dropout=0.,
                   batch_size=2, steps=4, warmup=0, validation=0., eval_every=2, windowing="sequential")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train(cfg, examples, root / "first")
            legacy = {k: v for k, v in load_checkpoint(root / "first" / "last.npz")["config"].items() if k != "windowing"}
            original = studio_model.load_checkpoint
            studio_model.load_checkpoint = lambda path: dict(original(path), config=legacy)
            try:
                train(dict(cfg, steps=6, windowing="anchored"), examples, root / "second", resume=root / "first" / "last.npz")
            finally:
                studio_model.load_checkpoint = original
            self.assertEqual(load_checkpoint(root / "second" / "last.npz")["config"]["windowing"], "sequential")


if __name__ == "__main__":
    unittest.main()
