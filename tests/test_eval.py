import json
import math
import random
import tempfile
import threading
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from studio_compile import CompiledModel
from studio_eval import (LN2, Stopped, MIN_DOCS, band_from_scores, bits_per_char, corpus_bpc, heldout_split, memorise_one_batch, noise_band, paired_gain,
                         run_status, sanity_checks, score_docs, score_windows, smooth, training_saved, uniform_bpc, unigram_bpc)
from studio_model import LocalModel, Tokenizer
from studio_registry import for_slot
from studio_slots import BOS, OFFSET, UNK
from studio_spec import validate_spec

FIXTURE = Path(__file__).parent / "fixtures" / "sample.jsonl"


class Formulas(unittest.TestCase):
    def test_split_dedups_by_document(self):
        docs = ["a", "b", "a", "c", "d", "b", "e", "f", "g", "h", "i", "j"]
        train, held = heldout_split(docs, .2)
        self.assertEqual(sorted(train + held), sorted(set(docs)))
        self.assertEqual(len(held), 2)
        self.assertEqual((train, held), heldout_split(docs, .2))

    def test_uniform_and_unigram(self):
        tok = Tokenizer("byte")
        docs = ["ab" * 10, "ba" * 10]
        self.assertAlmostEqual(uniform_bpc(tok, docs), math.log(263) / LN2)
        self.assertAlmostEqual(unigram_bpc(tok, docs), 1.0)  # two equiprobable bytes: 1 bit each
        self.assertAlmostEqual(unigram_bpc(tok, ["aaaa"]), 0.0)
        self.assertAlmostEqual(unigram_bpc(tok, ["aaaa"], ["b"]), math.log(263) / LN2)  # unseen: uniform

    def test_unknown_characters_scored_uniform(self):
        tok = Tokenizer("character", ["a", "b"])
        self.assertAlmostEqual(unigram_bpc(tok, ["abz"]), math.log(tok.vocab_size) / LN2 * (1 / 3) + 2 / 3)

    def test_band(self):
        lo, hi = band_from_scores([(math.log(2) * 8, 8)] * 5)
        self.assertAlmostEqual(lo, 1.0)
        self.assertAlmostEqual(hi, 1.0)
        lo, hi = band_from_scores([(LN2 * c, 10) for c in (1, 5, 9, 2, 8, 4, 6)])
        self.assertLess(lo, hi)

    def test_status(self):
        falling = [{"loss": 5 - .05 * i + (.02 if i % 2 else 0)} for i in range(40)]
        self.assertEqual(run_status(falling), "learning")
        noisy = [{"loss": 3 + (.3 if i % 2 else -.3)} for i in range(40)]
        self.assertEqual(run_status(noisy), "flat")
        memo = [{"loss": 3 - .05 * i, "val_loss": 3 + .1 * i if i % 10 == 0 else None} for i in range(40)]
        self.assertEqual(run_status(memo), "memorising")
        self.assertEqual(run_status([{"loss": 5}]), "learning")

    def test_smooth(self):
        self.assertEqual(smooth([1, 1, 1]), [1, 1, 1])
        self.assertLess(smooth([0, 10])[1], 10)

    def test_training_saved(self):
        old = [(0, 5.), (50, 3.), (100, 2.)]
        new = [(0, 5.), (50, 2.), (100, 1.5)]
        r = training_saved(new, old, "cos", "cos")
        self.assertTrue(r["comparable"])
        self.assertAlmostEqual(r["percent"], 50.)
        self.assertFalse(training_saved(new, old, "cos", "lin")["comparable"])
        self.assertFalse(training_saved(new[:2], old)["comparable"])
        self.assertEqual(training_saved([(0, 5.), (100, 4.)], old)["percent"], 0.)

    def test_sanity_from_history(self):
        good = [{"loss": math.log(263)}, {"loss": 4.}, {"loss": 3.}]
        checks = {c["id"]: c["ok"] for c in sanity_checks(good, 263, {"start": 5, "end": 1, "ok": True})}
        self.assertEqual(checks, {"starts_near_uniform": True, "finite": True, "memorises_one_batch": True, "error_falls": True})
        bad = {c["id"]: c["ok"] for c in sanity_checks([{"loss": 1.}, {"loss": float("nan")}], 263)}
        self.assertFalse(bad["starts_near_uniform"])
        self.assertFalse(bad["finite"])
        self.assertIsNone(bad["memorises_one_batch"])


def one_window_at_a_time(local, docs):
    """score_docs as it was first written: back-to-back context windows, each starting with no context."""
    tok, context, uniform = local.tokenizer, local.config["context"], math.log(local.tokenizer.vocab_size)
    out = []
    for text in docs:
        ids = np.array([BOS] + tok.encode(text), dtype=np.int64)
        nats = 0.
        for start in range(0, len(ids) - 1, context):
            x, y = ids[start:start + context], ids[start + 1:start + context + 1]
            logits = local.model.backend.to_cpu(local.model.forward(x[None, :len(y)]))[0].astype(np.float64)
            logits -= logits.max(axis=-1, keepdims=True)
            logp = logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))
            picked = logp[np.arange(len(y)), y]
            unknown = y == UNK
            nats += float(-picked[~unknown].sum()) + uniform * int(unknown.sum())
        out.append((nats, len(text)))
    return out


def strided_one_window_at_a_time(local, docs):
    """The sliding-window score written out plainly: one forward pass per window, no batching, each new half after the first read with the half before it."""
    tok, context, uniform = local.tokenizer, local.config["context"], math.log(local.tokenizer.vocab_size)
    out = []
    for text in docs:
        ids = np.array([BOS] + tok.encode(text), dtype=np.int64)
        nats, done = 0., 0  # done: ids[1:done + 1] are scored
        while done < len(ids) - 1:
            new = context if done == 0 else context // 2
            end = min(len(ids) - 1, done + new)  # the last target scored by this window
            begin = max(0, end - context)
            x, y = ids[begin:end], ids[begin + 1:end + 1]
            logits = local.model.backend.to_cpu(local.model.forward(x[None]))[0].astype(np.float64)
            logits[:, :OFFSET] = -np.inf  # the special rows are not text: the probability is shared among the real pieces
            logits -= logits.max(axis=-1, keepdims=True)
            logp = logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))
            fresh = np.arange(begin + 1, end + 1) > done  # the targets not scored by an earlier window
            picked, unknown = logp[np.arange(len(y)), y], y == UNK
            nats += float(-picked[fresh & ~unknown].sum()) + uniform * int((fresh & unknown).sum())
            done = end
        out.append((nats, len(text)))
    return out


def unigram_with_a_counter(tokenizer, docs, eval_docs=None):
    counts = Counter(t for d in docs for t in tokenizer.encode(d) if t != UNK)
    total, uniform = sum(counts.values()), math.log(tokenizer.vocab_size)
    nats = chars = 0
    for d in docs if eval_docs is None else eval_docs:
        chars += len(d)
        for t in tokenizer.encode(d):
            nats += math.log(total / counts[t]) if counts.get(t) else uniform
    return nats / chars / LN2


class Vectorised(unittest.TestCase):
    TEXTS = ["", "a", "hello world, hello world", "naïve café ✓ 日本語 \U0001f600", "x" * 200, "tab\tnew\nline" * 30, "ab" * 100]

    def test_byte_encode_matches_the_per_byte_loop(self):
        tok = Tokenizer("byte")
        for text in self.TEXTS:
            expected = [b + 7 for b in text.encode("utf-8")]
            self.assertEqual(tok.encode(text), expected)
            self.assertEqual(tok.encode_array(text).tolist(), expected)
            self.assertEqual(tok.encode_array(text).dtype, np.int64)
        self.assertEqual(tok.encode_array("\U0001f600").max(), 0xF0 + 7)  # uint8 + 7 would wrap past 248

    def test_character_encode_array_marks_unknown_characters(self):
        tok = Tokenizer("character", ["a", "b"])
        self.assertEqual(tok.encode_array("abz").tolist(), [7, 8, UNK])
        self.assertEqual(tok.encode_array("").tolist(), [])

    def test_unigram_matches_the_counter_version(self):
        rng = random.Random(3)
        docs = ["".join(rng.choice("abcde fgh\n") for _ in range(rng.randint(0, 60))) for _ in range(40)]
        held = docs[::3] + ["zzz héllo"]
        for tok in (Tokenizer("byte"), Tokenizer("character", sorted(set("".join(docs[:30]))))):
            self.assertAlmostEqual(unigram_bpc(tok, docs[:30], held), unigram_with_a_counter(tok, docs[:30], held), places=9)
            self.assertAlmostEqual(unigram_bpc(tok, docs), unigram_with_a_counter(tok, docs), places=9)

    def test_unigram_tallies_across_the_thousand_document_chunks(self):
        tok = Tokenizer("byte")
        docs = ["ab", "b", "cc"] * 700  # 2100 documents: three chunks
        self.assertAlmostEqual(unigram_bpc(tok, docs), unigram_with_a_counter(tok, docs), places=9)

    def test_padding_never_reaches_a_real_position_in_any_mixer(self):
        rng = np.random.default_rng(0)
        for module in for_slot("mixer"):
            with self.subTest(mixer=module.short):
                spec = validate_spec({"dims": {"d_model": 16, "n_layer": 2, "n_ctx": 32, "dropout": 0}, "block": {"mixer": {"id": module.short}}})
                model = CompiledModel(spec, 23, device="cpu")
                ids = rng.integers(7, 23, (1, 20))
                padded = np.concatenate([ids, np.zeros((1, 12), dtype=np.int64)], axis=1)
                np.testing.assert_allclose(model.forward(padded)[:, :20].numpy(), model.forward(ids).numpy(), atol=1e-5)

    def test_batched_scores_match_one_window_at_a_time_with_a_stride(self):
        from studio_model import train
        rows = [json.loads(line) for line in FIXTURE.read_text().splitlines()][:30]
        docs = [r["user"] + "\n" + r["assistant"] for r in rows]
        held = [docs[0], "", "é", docs[1] * 6, docs[2] + " ✓ unseen", docs[3] * 3, docs[4]]  # empty, 1 piece, longer than the 48-piece window
        for tokenizer in ("byte", "character"):
            with self.subTest(tokenizer=tokenizer), tempfile.TemporaryDirectory() as tmp:
                cfg = dict(width=16, layers=1, heads=2, context=48, steps=4, batch_size=4, eval_every=4, validation=.2, device="cpu", tokenizer=tokenizer)
                train(cfg, [{"text": d} for d in docs[5:]], Path(tmp) / "r")
                local = LocalModel(Path(tmp) / "r" / "last.npz", device="cpu")
                expected, got = strided_one_window_at_a_time(local, held), score_docs(local, held)
                self.assertEqual([c for _, c in got], [c for _, c in expected])
                for (a, _), (b, _) in zip(got, expected):
                    self.assertAlmostEqual(a, b, delta=1e-5 * max(1., abs(b)))  # float32 logits: batch shape moves the last bits
                if tokenizer == "character":
                    self.assertIn(UNK, local.tokenizer.encode(held[4]))  # the unknown-character path ran


class StopScoring(unittest.TestCase):
    """Issue #755: score_docs checks its stop event between documents and batches."""

    def stub(self, on_forward=None):
        tok = Tokenizer("byte")
        local = mock.Mock(spec=LocalModel)
        local.tokenizer, local.config = tok, {"context": 8}
        local.model = mock.Mock()
        local.model.forward.side_effect = lambda x: (on_forward and on_forward(), torch.zeros(*x.shape, tok.vocab_size))[1]
        return local

    def test_a_stop_set_before_the_first_document_scores_nothing(self):
        stop, local = threading.Event(), self.stub()
        stop.set()
        with self.assertRaises(Stopped):
            score_docs(local, ["abc", "def"], stop=stop)
        local.model.forward.assert_not_called()

    def test_a_stop_set_mid_run_ends_before_the_remaining_documents(self):
        stop = threading.Event()
        local = self.stub(on_forward=stop.set)
        with self.assertRaises(Stopped):
            score_docs(local, ["a"] * 2000, stop=stop)
        self.assertGreater(local.model.forward.call_count, 0)
        self.assertLess(local.model.forward.call_count, 50)  # a few batches of the first pass, not all 2000 documents

    def test_without_a_stop_every_document_is_scored(self):
        rows = score_docs(self.stub(), ["a", "bc"], stop=threading.Event())
        self.assertEqual([c for _, c in rows], [1, 2])


class Strided(unittest.TestCase):
    def test_windows_cover_every_piece_exactly_once(self):
        for context in (1, 2, 3, 8, 9, 48):
            for length in (1, 2, 3, context, context + 1, context + 2, 2 * context, 3 * context + 1, 100):
                with self.subTest(context=context, length=length):
                    windows = score_windows(length, context)
                    scored = [t for start, stop, skip in windows for t in range(start + 1 + skip, stop + 1)]
                    self.assertEqual(scored, list(range(1, length)))  # every id after BOS, once, in order
                    for index, (start, stop, skip) in enumerate(windows):
                        self.assertLessEqual(stop - start, context)
                        if index:  # a later window scores at most half a window of new pieces, with the pieces before them as context
                            self.assertLessEqual(stop - start - skip, max(1, context // 2))
                            self.assertGreaterEqual(skip, min(stop - start, context) - max(1, context // 2))

    def test_every_piece_is_scored_once_by_the_model_pass(self):
        from studio_model import LocalModel
        tok = Tokenizer("character", ["a", "b"])

        class Flat:  # every real piece equally likely: a document's nats are ln (V - 7) times the pieces scored
            def forward(self, x):
                return torch.zeros(x.shape[0], x.shape[1], tok.vocab_size)

        local = LocalModel.__new__(LocalModel)
        local.model, local.tokenizer, local.config = Flat(), tok, {"context": 8}
        docs = ["", "a", "ab" * 3, "ab" * 4, "ab" * 4 + "a", "ab" * 20, "abz" * 13]  # z is unknown: scored at ln V too
        for (nats, chars), doc in zip(score_docs(local, docs), docs):
            self.assertEqual(chars, len(doc))
            unknown = doc.count("z")  # an unknown character is scored at ln V
            self.assertAlmostEqual(nats, (len(doc) - unknown) * math.log(tok.vocab_size - OFFSET) + unknown * math.log(tok.vocab_size), places=6)

    def test_strided_bits_per_character_do_not_exceed_the_old_back_to_back_value(self):
        from studio_model import train
        # "aabb" repeating, each document starting at a different phase: the next letter needs the two before it, which a
        # window starting cold does not have for its first prediction and a window re-reading the previous half does.
        pattern = "aabb" * 80
        fit = [pattern[r % 4:r % 4 + 100 + r] for r in range(40)]
        held = [pattern[r % 4:r % 4 + 130 + 7 * r] for r in range(6)]  # several 32-piece windows each
        with tempfile.TemporaryDirectory() as tmp:
            cfg = dict(width=16, layers=1, heads=2, context=32, steps=600, batch_size=8, eval_every=600, validation=.2, device="cpu",
                       tokenizer="character", learning_rate=.01, warmup=5)
            train(cfg, [{"text": d} for d in fit], Path(tmp) / "r")
            local = LocalModel(Path(tmp) / "r" / "last.npz", device="cpu")
            old, new = corpus_bpc(one_window_at_a_time(local, held)), corpus_bpc(score_docs(local, held))
        self.assertGreater(min(len(local.tokenizer.encode(d)) for d in held), 2 * 32)
        self.assertLessEqual(new, old)
        self.assertLess(new, old)  # and the learned context does help here: the gap is not rounding


class PairedGain(unittest.TestCase):
    @staticmethod
    def scores(bits, chars=20):
        return [(b * chars * LN2, chars) for b in bits]

    def test_a_clearly_better_model_is_better(self):
        rng = np.random.default_rng(1)
        hard = rng.uniform(1.5, 4., 60)  # documents differ a lot in difficulty; the two models share that
        parent, candidate = self.scores(hard), self.scores(hard - .1 + rng.normal(0, .02, 60))
        got = paired_gain(parent, candidate)
        self.assertTrue(got["better"])
        self.assertLess(got["upper"], 0)
        self.assertAlmostEqual(got["diff"], -.1, delta=.02)
        self.assertFalse(paired_gain(candidate, parent)["better"])  # the other way round is worse, not better

    def test_identical_models_are_not_sure(self):
        scores = self.scores(np.random.default_rng(2).uniform(1., 4., 40))
        got = paired_gain(scores, list(scores))
        self.assertFalse(got["better"])
        self.assertEqual((got["diff"], got["upper"]), (0., 0.))

    def test_a_gap_within_luck_is_not_sure(self):
        rng = np.random.default_rng(3)
        base = rng.uniform(1.5, 4., 60)
        noisy = base + rng.normal(0, .5, 60) - .02  # a hair lower on average, document by document all over the place
        got = paired_gain(self.scores(base), self.scores(noisy))
        self.assertLess(got["diff"], 0)
        self.assertFalse(got["better"])

    def test_pairing_sees_what_the_unpaired_band_cannot(self):
        rng = np.random.default_rng(4)
        hard = rng.uniform(.5, 6., 80)  # a wide spread of difficulty swamps a small steady gain unless the documents are paired
        parent, candidate = self.scores(hard), self.scores(hard - .05)
        lo, _ = band_from_scores(parent)
        self.assertGreater(corpus_bpc(candidate), lo)  # inside the parent's own band: the old gate could not see it
        self.assertTrue(paired_gain(parent, candidate)["better"])

    def test_the_confidence_level_and_documents_must_match(self):
        rng = np.random.default_rng(5)
        base = rng.uniform(1.5, 4., 50)
        parent, candidate = self.scores(base), self.scores(base + rng.normal(-.03, .1, 50))
        loose, strict = paired_gain(parent, candidate, confidence=.8), paired_gain(parent, candidate, confidence=.999)
        self.assertLess(loose["upper"], strict["upper"])
        with self.assertRaises(ValueError):
            paired_gain(parent, candidate[:-1])
        with self.assertRaises(ValueError):
            paired_gain(parent, self.scores(base, chars=21))
        self.assertFalse(paired_gain(parent[:1], candidate[:1])["better"])  # one document says nothing about luck

    def test_too_few_documents_are_never_better(self):
        base = np.random.default_rng(7).uniform(1.5, 4., 40)
        for n in (2, 5, MIN_DOCS - 1):  # a gain far bigger than luck, but on too little text to tell
            got = paired_gain(self.scores(base[:n]), self.scores(base[:n] - 1.))
            self.assertFalse(got["better"])
            self.assertEqual((got["reason"], got["docs"]), ("few_docs", n))
            self.assertAlmostEqual(got["diff"], -1., places=6)  # the measured gap is still reported
            self.assertTrue(math.isnan(got["upper"]))
        got = paired_gain(self.scores(base[:MIN_DOCS]), self.scores(base[:MIN_DOCS] - 1.))
        self.assertTrue(got["better"])  # the same gain with MIN_DOCS documents is judged
        self.assertNotIn("reason", got)

    def test_the_same_scores_give_the_same_answer(self):
        base = np.random.default_rng(6).uniform(1.5, 4., 50)
        a, b = self.scores(base), self.scores(base - .05)
        self.assertEqual(paired_gain(a, b), paired_gain(a, b))


class TinyRun(unittest.TestCase):
    def test_real_run(self):
        from studio_model import train
        rows = [json.loads(line) for line in FIXTURE.read_text().splitlines()][:40]
        docs = [r["user"] + "\n" + r["assistant"] for r in rows]
        fit, held = heldout_split(docs, .2)
        cfg = dict(width=16, layers=1, heads=2, context=64, steps=30, batch_size=4, eval_every=10, validation=.2, device="cpu")
        with tempfile.TemporaryDirectory() as tmp:
            train(cfg, [{"text": d} for d in fit], Path(tmp) / "r")
            ckpt = Path(tmp) / "r" / "last.npz"
            bpc = bits_per_char(ckpt, held)
            lo, hi = noise_band(ckpt, held)
            checks = sanity_checks(Path(tmp) / "r")
        self.assertAlmostEqual(bpc, uniform_bpc(Tokenizer("byte"), held), delta=.15 * uniform_bpc(Tokenizer("byte"), held))
        self.assertLessEqual(lo, hi)
        self.assertTrue(lo - .5 < bpc < hi + .5)
        self.assertTrue(all(c["ok"] in (True, None) for c in checks), checks)

    def test_memorise_one_batch(self):
        result = memorise_one_batch(dict(width=16, layers=1, heads=2, context=32), ["hello world, hello world"] * 2)
        self.assertTrue(result["ok"], result)


if __name__ == "__main__":
    unittest.main()
