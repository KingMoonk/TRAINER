import json
import math
import tempfile
import unittest
from pathlib import Path

from studio_eval import (LN2, band_from_scores, bits_per_char, heldout_split, memorise_one_batch, noise_band,
                         run_status, sanity_checks, smooth, training_saved, uniform_bpc, unigram_bpc)
from studio_model import Tokenizer

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
