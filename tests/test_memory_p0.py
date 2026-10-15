"""Issue #13 P0: weights-only loads, KV-cached generation, the chat-model LRU, OOM recovery in train()."""
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

import studio_model
from studio_compile import CompiledModel
from studio_estimate import estimate
from studio_ia.services import CHAT_MODELS, WorkspaceServices
from studio_model import LocalModel, ScratchTransformer, load_checkpoint, train, validate_config
from studio_modules.tokenizer.byte import Tokenizer
from studio_core import Workspace

EXAMPLES = [{"text": f"Counting up from {n}: {n}, {n + 1}, {n + 2}."} for n in range(12)]
CONFIG = {"width": 16, "layers": 2, "heads": 2, "context": 32, "expansion": 2, "dropout": 0., "batch_size": 4,
          "steps": 5, "learning_rate": .003, "warmup": 1, "validation": 0, "eval_every": 5, "seed": 5,
          "device": "cpu", "sample_every": 0}
POSITIONS = ("learned", "rope", "alibi", "sinusoidal", "none")
MIXERS = ("attention", "gqa", "linear", "conv", "gru", "lstm", "ssm", "none")
CACHED = ("attention", "gqa", "linear")  # mixers with a cache path; the rest re-run the whole window


def positions(mixer):
    """RoPE and ALiBi act on attention scores, which linear attention does not have."""
    return POSITIONS if mixer != "linear" else ("learned", "sinusoidal", "none")


def local(mixer="attention", position="learned"):
    """A LocalModel with random weights and no file."""
    spec = {"schema": 1, "dims": {"d_model": 16, "n_layer": 2, "n_ctx": 32, "dropout": 0}, "position": position,
            "block": {"mixer": {"id": mixer, "config": {"heads": 4, "kv_heads": 2} if mixer in ("attention", "gqa") else {}}}}
    if mixer == "attention":
        spec["block"]["mixer"]["config"] = {"heads": 4}
    config = validate_config({"architecture": spec, "device": "cpu", "seed": 3})
    result = LocalModel.__new__(LocalModel)
    result.config, result.tokenizer = config, Tokenizer("byte")
    result.model = ScratchTransformer(config, result.tokenizer.vocab_size)
    result.rng, result.step = np.random.default_rng(1), 0
    return result


def generate(model, cached, max_tokens=48):
    with mock.patch.object(CompiledModel, "can_cache", new_callable=mock.PropertyMock, return_value=cached and model.model.can_cache):
        return model.generate([{"role": "user", "content": "Count from 4 to 9, slowly."}], temperature=0, max_tokens=max_tokens)


class KVCache(unittest.TestCase):
    def test_cached_logits_equal_uncached(self):
        ids = np.random.default_rng(0).integers(6, 200, size=30)
        for mixer in CACHED:
            for position in positions(mixer):
                with self.subTest(mixer=mixer, position=position):
                    model = local(mixer, position).model
                    self.assertTrue(model.can_cache)
                    cache = {"n": 0}
                    got = [model.forward(ids[None, :6], cache=cache)[0, -1]]
                    got += [model.forward(ids[None, i:i + 1], cache=cache)[0, -1] for i in range(6, 30)]
                    for i, row in zip(range(6, 31), got):
                        torch.testing.assert_close(row, model.forward(ids[None, :i])[0, -1], atol=1e-5, rtol=1e-4)
                    self.assertEqual(cache["n"], 30)

    def test_greedy_output_is_identical_and_survives_a_sliding_window(self):
        for mixer in CACHED:
            for position in positions(mixer):
                with self.subTest(mixer=mixer, position=position):
                    model = local(mixer, position)
                    self.assertEqual(generate(model, True), generate(model, False))  # prompt + 48 new tokens passes the 32-token context

    def test_mixers_without_support_fall_back_and_still_match(self):
        for mixer in MIXERS:
            with self.subTest(mixer=mixer):
                model = local(mixer)
                self.assertEqual(model.model.can_cache, mixer in CACHED + ("none",))
                self.assertEqual(generate(model, True), generate(model, False))

    def test_greedy_tokens_match_one_by_one(self):
        """Token for token, not just the decoded text: CPU, float32, fixed seed, EOS blocked so every run makes 20 tokens."""
        real = studio_model.last_logits
        for mixer in CACHED:
            with self.subTest(mixer=mixer):
                model, seen = local(mixer, "learned" if mixer == "linear" else "rope"), {}
                for cached in (True, False):
                    windows = seen[cached] = []
                    with mock.patch.object(studio_model, "last_logits", lambda *a, out=windows: out.append(list(a[1])) or real(*a)), \
                            mock.patch.object(studio_model, "unsayable", lambda tokenizer: [0, 1, 2, 3, 4, 5, 6]):
                        text = generate(model, cached, max_tokens=20)
                    seen[cached, "text"] = text
                self.assertEqual(len(seen[True]), 20)
                self.assertEqual(seen[True], seen[False])  # the same tokens fed back at every step
                self.assertEqual(seen[True, "text"], seen[False, "text"])

    def test_cache_memory_matches_the_estimate(self):
        """Attention and GQA keep Estimate.kv_bytes per layer per token; linear attention keeps a fixed state of that size."""
        for mixer in CACHED:
            with self.subTest(mixer=mixer):
                model = local(mixer).model
                per_token = estimate(model.spec, model.dims.vocab)["kv_bytes_per_token"]
                cache = {"n": 0}
                model.forward(np.arange(7, 27)[None], cache=cache)
                for token in range(27, 31):
                    model.forward(np.array([[token]]), cache=cache)
                stored = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes() for key, state in cache.items() if key != "n" for t in state}
                expected = per_token * (cache["n"] if mixer != "linear" else 1)
                self.assertAlmostEqual(sum(stored.values()) / expected, 1., delta=.01)

    def test_complete_matches_uncached(self):
        model = local("gqa", "rope")
        with mock.patch.object(CompiledModel, "can_cache", new_callable=mock.PropertyMock, return_value=False):
            expected = model.complete("The ", 40)
        self.assertEqual(model.complete("The ", 40), expected)


class Checkpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        train(CONFIG, EXAMPLES, cls.root / "run", telemetry_every=0)
        cls.checkpoint = cls.root / "run" / "last.npz"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_weights_only_matches_a_full_load_and_skips_optimizer_arrays(self):
        full = load_checkpoint(self.checkpoint)
        read = []
        real = np.load

        class Spy:
            def __init__(self, archive):
                self.archive, self.files = archive, archive.files

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                self.archive.close()

            def __getitem__(self, key):
                read.append(key)
                return self.archive[key]

        with mock.patch.object(studio_model.np, "load", lambda *a, **k: Spy(real(*a, **k))):
            light = load_checkpoint(self.checkpoint, weights_only=True)
        self.assertTrue(full["m"] and not light["m"] and not light["v"] and not light["optimizer_state"])
        self.assertFalse([key for key in read if key.startswith("adam_")])
        self.assertEqual(set(light["model"]), set(full["model"]))
        self.assertTrue(all(np.array_equal(light["model"][k], full["model"][k]) for k in full["model"]))

    def test_chat_models_are_an_lru(self):
        paths = []
        for name in "abcd":
            paths.append(str(self.root / f"{name}.npz"))
            shutil.copy(self.checkpoint, paths[-1])
        self.assertEqual(CHAT_MODELS, 3)
        svc = WorkspaceServices(Workspace(self.root / "ws"))
        self.addCleanup(svc.close)
        messages, ask = [{"role": "user", "content": "Counting"}], lambda p: svc.chat(p, messages, None, None)
        with mock.patch.object(svc, "_checkpoint", side_effect=lambda v: v):
            first = ask(paths[0])
            for path in paths[1:]:
                ask(path)
            self.assertEqual(list(svc._models), paths[1:])  # a was evicted, at most CHAT_MODELS stay loaded
            self.assertEqual(ask(paths[0]), first)  # reloaded: same weights, same seed, same answer
            self.assertEqual(list(svc._models), [paths[2], paths[3], paths[0]])


class OutOfMemory(unittest.TestCase):
    def test_a_forced_oom_shrinks_the_batch_and_the_run_finishes(self):
        real, calls, events = ScratchTransformer.forward, [], []

        def forward(self, ids, *args, **kwargs):
            if kwargs.get("training"):
                calls.append(len(ids))
                if len(calls) == 4:  # the 4th forward is step 4; raise once
                    raise torch.cuda.OutOfMemoryError("CUDA out of memory.")
            return real(self, ids, *args, **kwargs)

        with tempfile.TemporaryDirectory() as d, mock.patch.object(ScratchTransformer, "forward", forward):
            root = Path(d)
            record = train(dict(CONFIG, steps=6, eval_every=6), EXAMPLES, root / "r", emit=events.append, telemetry_every=0)
            self.assertEqual(record["status"], "completed")
            self.assertEqual(record["step"], 6)
            info = [e["text"] for e in events if e["type"] == "info" and "Out of memory" in e["text"]]
            self.assertEqual(len(info), 1)
            self.assertIn("batch size now 2, accumulation 2", info[0])
            self.assertEqual(calls[:4], [4, 4, 4, 4])
            self.assertEqual(set(calls[4:]), {2})  # the retry and every later step run as 2 x 2
            self.assertEqual(len(calls[4:]), 3 * 2)  # step 4 retried plus steps 5 and 6, two forwards each
            losses = [e["loss"] for e in events if e["type"] == "metric"]
            self.assertTrue(losses and all(np.isfinite(losses)))
            # the declared batch settings are what the checkpoint keeps, so the shrunk run still resumes
            self.assertEqual(load_checkpoint(root / "r" / "last.npz")["config"]["batch_size"], 4)
            self.assertEqual(train(dict(CONFIG, steps=8, eval_every=8), EXAMPLES, root / "r2", resume=root / "r" / "last.npz", telemetry_every=0)["status"], "completed")

    def test_batch_of_one_reraises(self):
        def forward(self, ids, *args, **kwargs):
            raise RuntimeError("MPS backend out of memory")

        with tempfile.TemporaryDirectory() as d, mock.patch.object(ScratchTransformer, "forward", forward):
            with self.assertRaises(RuntimeError):
                train(dict(CONFIG, batch_size=2, validation=0), EXAMPLES, Path(d) / "r", telemetry_every=0)
            self.assertEqual(studio_model.read_metrics(Path(d) / "r" / "metrics.jsonl"), [])


if __name__ == "__main__":
    unittest.main()
