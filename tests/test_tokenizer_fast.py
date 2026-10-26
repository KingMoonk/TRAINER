"""The fast subword tokenizer: learned from a stream, byte-exact, hashed, and usable by training, warm starts and DiLoCo."""
import importlib.util
import json
import os
import random
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np

import studio_ia.design as D
from studio_model import LocalModel, Tokenizer, load_checkpoint, remap_vocabulary, train, vocabulary_guess
from studio_modules.tokenizer.bpe_fast import FastBPETokenizer, sample_texts, suggest_vocab
from studio_spec import config_from_spec, validate_spec

# CI always installs the package (requirements-training.txt); a machine that has not yet run pip install skips instead of failing.
NEEDS = unittest.skipUnless(importlib.util.find_spec("tokenizers") or os.environ.get("CI"), "the `tokenizers` package is not installed")

SYLLABLES = ["ka", "to", "mi", "re", "su", "na", "lo", "vi", "pe", "dar", "ten", "ul", "or", "ix", "be", "chi"]


def corpus(documents=400, seed=1):
    """A fake corpus of pseudo-words with a Zipf-like spread, as a generator (it is never held whole)."""
    rng = random.Random(seed)
    words = ["".join(rng.choice(SYLLABLES) for _ in range(rng.randint(1, 4))) for _ in range(3000)]
    for _ in range(documents):
        yield {"text": " ".join(words[min(len(words) - 1, int(rng.paretovariate(1.1)) - 1 + rng.randint(0, 40))] for _ in range(120)) + "."}


class Declared(unittest.TestCase):
    def test_the_dependency_is_in_the_requirements_ci_installs(self):
        text = (Path(__file__).resolve().parents[1] / "requirements-training.txt").read_text()
        self.assertIn("tokenizers", text)

    def test_the_module_is_registered_and_its_vocabulary_is_read_from_its_setting(self):
        from studio_registry import get
        module = get("tokenizer", "bpe_fast")
        self.assertIsNone(module.vocab(module.parse()))
        spec = validate_spec({"tokenizer": {"id": "bpe_fast", "config": {"vocab_size": 4096}}})
        self.assertEqual(vocabulary_guess({"architecture": spec}), 4096)

    def test_the_vocabulary_is_sized_to_the_model(self):
        sizes = [suggest_vocab(d, layers) for d, layers in ((32, 1), (64, 2), (256, 6), (768, 12), (1024, 24))]
        self.assertEqual(sizes, sorted(sizes))
        self.assertEqual((sizes[0], sizes[-1]), (1024, 32768))
        for d, layers in ((64, 2), (256, 6), (768, 12)):   # the word table stays a minor part of a model big enough for it
            self.assertLessEqual(suggest_vocab(d, layers) * d, max(1024 * d, layers * 12 * d * d / 3 + 512 * d))

    def test_the_sample_spans_the_corpus_and_stops_at_its_budget(self):
        texts = [{"text": "x" * 100} for _ in range(1000)]
        picked = list(sample_texts(texts, 10_000))
        self.assertLessEqual(sum(map(len, picked)), 10_100)
        self.assertGreater(len(picked), 90)
        taken = []
        stream = ({"text": f"{i:04d}" + "y" * 96} for i in range(100_000))
        for text in sample_texts(stream, 5_000):
            taken.append(text)
        self.assertLessEqual(len(taken), 51)   # a stream is read only as far as the budget

    def test_a_rows_like_corpus_is_read_only_at_the_sampled_rows(self):
        class CountingRows:
            def __init__(self, count):
                self.rows, self.reads = [{"text": "x" * 100} for _ in range(count)], 0

            def __len__(self):
                return len(self.rows)

            def __getitem__(self, position):
                self.reads += 1
                return self.rows[position]

            def characters(self):
                return sum(len(row["text"]) for row in self.rows)

        rows, heard = CountingRows(1000), []
        picked = list(sample_texts(rows, 10_000, lambda done, total: heard.append((done, total))))
        self.assertEqual(picked, list(sample_texts([{"text": "x" * 100} for _ in range(1000)], 10_000)))
        self.assertLessEqual(rows.reads, len(picked))
        self.assertEqual(heard[-1][1], 10_000)
        self.assertGreaterEqual(heard[-1][0], 10_000 - 100)
        small = CountingRows(5)
        self.assertEqual(len(list(sample_texts(small, 10_000))), 5)   # fits: every row, each read once
        self.assertEqual(small.reads, 5)


@NEEDS
class Fitting(unittest.TestCase):
    def test_a_32k_vocabulary_is_learned_from_a_streamed_corpus_and_is_deterministic(self):
        rng = random.Random(5)
        letters = "abcdefghijklmnopqrstuvwxyz"
        words = ["".join(rng.choice(letters) for _ in range(rng.randint(3, 9))) for _ in range(60000)]

        def stream():
            local = random.Random(9)
            for _ in range(3000):
                yield {"text": " ".join(local.choice(words) for _ in range(200))}
        first = FastBPETokenizer.fit(stream(), vocab_size=32768, fit_chars=2_000_000)
        second = FastBPETokenizer.fit(stream(), vocab_size=32768, fit_chars=2_000_000)
        self.assertEqual(first.vocab_size, 32768)
        self.assertEqual(first.merges, second.merges)
        self.assertEqual(first.digest, second.digest)

    def test_a_small_corpus_stops_early_instead_of_inventing_pieces(self):
        tokenizer = FastBPETokenizer.fit(list(corpus(20)), vocab_size=65536)
        self.assertLess(tokenizer.vocab_size, 65536)
        self.assertGreater(tokenizer.vocab_size, 263)

    def test_it_cuts_text_into_far_fewer_pieces_than_bytes(self):
        examples = list(corpus())
        tokenizer = FastBPETokenizer.fit(examples, vocab_size=2048)
        self.assertGreater(tokenizer.characters_per_token([e["text"] for e in examples]), 3.)


@NEEDS
class Exact(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = FastBPETokenizer.fit(list(corpus()), vocab_size=1500)

    def test_every_byte_round_trips_with_no_unknown_token(self):
        t = self.tokenizer
        data = bytes(range(256)) * 3
        ids = t.encode_bytes(data)
        self.assertEqual(t.decode_bytes(ids), data)
        self.assertTrue(all(7 <= i < t.vocab_size for i in ids))

    def test_emoji_and_invalid_utf8_round_trip(self):
        t = self.tokenizer
        for data in ("hello 😀 wörld 日本語 \x00\n\t ".encode(), b"\xff\xfe\xc3(\xe2\x82", "ok".encode() + b"\x80" + "😀".encode()[:3], b""):
            self.assertEqual(t.decode_bytes(t.encode_bytes(data)), data, data)
        text = "Zażółć gęślą jaźń 🧪 — done"
        self.assertEqual(t.decode(t.encode(text)), text)

    def test_the_array_form_agrees_with_the_list_form(self):
        text = "kato mire sunalo"
        self.assertEqual(self.tokenizer.encode_array(text).tolist(), self.tokenizer.encode(text))
        self.assertEqual(self.tokenizer.encode_batch([text, "x"]), [self.tokenizer.encode(text), self.tokenizer.encode("x")])


@NEEDS
class Sharing(unittest.TestCase):
    def test_a_checkpoint_form_restores_to_the_same_cut_and_a_changed_one_is_refused(self):
        t = FastBPETokenizer.fit(list(corpus(60)), vocab_size=1200)
        data = json.loads(json.dumps(t.serialize()))
        back = Tokenizer.restore(data)
        self.assertEqual(back.kind, "bpe_fast")
        self.assertEqual(back.encode("kato mire"), t.encode("kato mire"))
        data["merges"][0][0] = (data["merges"][0][0] + 1) % 256
        with self.assertRaises(ValueError):
            Tokenizer.restore(data)

    def test_the_artifact_carries_the_hash_peers_compare(self):
        t = FastBPETokenizer.fit(list(corpus(60)), vocab_size=1200)
        with tempfile.TemporaryDirectory() as tmp:
            digest = t.save_artifact(Path(tmp) / "tok" / "t.json")
            self.assertEqual(digest, t.digest)
            self.assertEqual(FastBPETokenizer.load_artifact(Path(tmp) / "tok" / "t.json", digest).merges, t.merges)
            with self.assertRaises(ValueError):
                FastBPETokenizer.load_artifact(Path(tmp) / "tok" / "t.json", "0" * 64)


def small_spec(vocab=1024):
    spec = validate_spec({"schema": 1})
    spec["dims"].update(d_model=16, n_layer=1, n_ctx=32, dropout=0.)
    spec["tokenizer"] = {"id": "bpe_fast", "config": {"vocab_size": vocab}}
    spec["block"]["mixer"]["config"].update(heads=2)
    return validate_spec(spec)


RUN = {"batch_size": 2, "steps": 4, "validation": 0, "eval_every": 4, "seed": 3, "objective": "all", "device": "cpu"}


@NEEDS
class Training(unittest.TestCase):
    def test_trains_saves_the_tokenizer_resumes_and_generates(self):
        examples = list(corpus(60))
        cfg = {**config_from_spec(small_spec()), **RUN}
        with tempfile.TemporaryDirectory() as tmp:
            first, second = Path(tmp) / "a", Path(tmp) / "b"
            train(cfg, examples, first)
            saved = load_checkpoint(first / "last.npz")
            self.assertEqual(saved["tokenizer"]["kind"], "bpe_fast")
            self.assertEqual(json.loads((first / "tokenizer.json").read_text())["sha256"], saved["tokenizer"]["sha256"])
            train(dict(cfg, steps=6, eval_every=6), examples, second, resume=first / "last.npz")
            self.assertEqual(load_checkpoint(second / "last.npz")["tokenizer"]["sha256"], saved["tokenizer"]["sha256"])
            reply = LocalModel(second / "last.npz", device="cpu").generate([{"role": "user", "content": "kato"}], max_tokens=5)
            self.assertIsInstance(reply, str)

    def test_a_byte_checkpoint_warm_starts_a_subword_design_by_matching_pieces(self):
        examples = list(corpus(60))
        byte = validate_spec({"schema": 1})
        byte["dims"].update(d_model=16, n_layer=1, n_ctx=32, dropout=0.)
        byte["block"]["mixer"]["config"].update(heads=2)
        with tempfile.TemporaryDirectory() as tmp:
            old, new = Path(tmp) / "old", Path(tmp) / "new"
            train({**config_from_spec(validate_spec(byte)), **RUN}, examples, old)
            cfg = {**config_from_spec(small_spec()), **RUN}
            train(cfg, examples, new, init_from=old / "last.npz")
            before, after = load_checkpoint(old / "last.npz"), load_checkpoint(new / "last.npz")
            self.assertEqual(after["tokenizer"]["kind"], "bpe_fast")
            fresh = {k: v for k, v in after["model"].items()}
            moved = remap_vocabulary(before["model"], Tokenizer.restore(before["tokenizer"]), Tokenizer.restore(after["tokenizer"]), fresh)
            self.assertEqual(moved.keys(), fresh.keys())

    def test_the_sharded_path_cuts_with_it(self):
        import studio_model
        examples = list(corpus(60))
        cfg = {**config_from_spec(small_spec()), **RUN}
        with tempfile.TemporaryDirectory() as tmp:
            train(cfg, examples, Path(tmp) / "run", stream_chars=1000, shard_dir=Path(tmp) / "shards")
            self.assertTrue(any(Path(tmp, "shards").iterdir()))
            self.assertTrue(studio_model.load_checkpoint(Path(tmp) / "run" / "last.npz")["tokenizer"]["kind"] == "bpe_fast")


@NEEDS
class Distributed(unittest.TestCase):
    def test_a_member_cuts_text_with_the_coordinators_tokenizer(self):
        from fastapi.testclient import TestClient

        import studio_diloco
        from tests import diloco_support as support
        examples = list(corpus(60))
        config = {**support.CONFIG, **config_from_spec(small_spec()), "device": "cpu", "batch_size": 2}
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            coordinator = studio_diloco.Coordinator.build("fast", root / "diloco", config, examples, support.settings(inner_steps=2, rounds=2))
            studio_diloco.COORDINATORS["fast"] = coordinator
            self.addCleanup(studio_diloco.COORDINATORS.pop, "fast", None)
            shipped = json.loads(json.dumps(coordinator.tokenizer.serialize()))
            self.assertEqual(Tokenizer.restore(shipped).digest, coordinator.tokenizer.digest)
            with TestClient(support.app()) as client:
                studio_diloco.run_member(support.Peer(client), "fast", "solo", 0, root, root / "member", lambda event: None, threading.Event(), wait=.2)
            self.assertTrue(coordinator.finished)
            self.assertTrue(all(np.isfinite(v).all() for v in coordinator.theta.values()))


@NEEDS
class Builder(unittest.TestCase):
    def test_picking_the_small_word_list_writes_the_learned_tokenizer_sized_to_the_design(self):
        spec = validate_spec({"schema": 1})
        spec["dims"].update(d_model=256, n_layer=6)
        spec = validate_spec(spec)
        choices = D.read_spec(spec)
        written = D.write_spec(spec, dict(choices, V=1))
        self.assertEqual(written["tokenizer"]["id"], "bpe_fast")
        self.assertEqual(written["tokenizer"]["config"]["vocab_size"], suggest_vocab(256, 6))
        self.assertEqual(D.read_spec(written)["V"], 1)

    def test_a_design_saved_with_the_older_word_list_still_reads_as_the_small_word_list(self):
        spec = validate_spec({"tokenizer": {"id": "bpe_custom", "config": {"vocab_size": 512}}})
        self.assertEqual(D.read_spec(spec)["V"], 1)
        self.assertEqual(D.write_spec(spec, D.read_spec(spec))["tokenizer"]["id"], "bpe_custom")


if __name__ == "__main__":
    unittest.main()
