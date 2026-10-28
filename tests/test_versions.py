import json
import tempfile
import unittest
from pathlib import Path

import studio_spec
from studio_versions import VersionStore, change_text, spec_diff


def make_spec(layers=4, heads=4, d_ff=0, tokenizer="byte"):
    spec = studio_spec.default_spec()
    spec["dims"]["n_layer"] = layers
    spec["tokenizer"] = {"id": tokenizer}
    spec["block"]["mixer"] = {"id": "attention", "config": {"heads": heads}}
    spec["block"]["channel"] = {"id": "mlp_gelu", "config": {"d_ff": d_ff} if d_ff else {}}
    return studio_spec.validate_spec(spec)


class Diff(unittest.TestCase):
    def test_plain_words(self):
        self.assertEqual(spec_diff(make_spec(4), make_spec(6)), ["more blocks (4 → 6)"])
        self.assertEqual(spec_diff(make_spec(6), make_spec(4)), ["fewer blocks (6 → 4)"])
        self.assertEqual(spec_diff(make_spec(heads=4), make_spec(heads=8)), ["look-back heads 4 → 8"])
        self.assertEqual(spec_diff(make_spec(), make_spec(d_ff=512)), ["wide thinking room"])
        self.assertEqual(spec_diff(make_spec(), make_spec(d_ff=128)), ["narrow thinking room"])

    def test_combined_and_unchanged(self):
        self.assertEqual(change_text(make_spec(4), make_spec(6, heads=8)), "more blocks (4 → 6), look-back heads 4 → 8")
        self.assertEqual(change_text(make_spec(), make_spec()), "no change")

    def test_word_list_wording(self):
        self.assertEqual(spec_diff(make_spec(tokenizer="byte"), make_spec(tokenizer="bpe_gpt2")), ["GPT-2 BPE"])


class Store(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = VersionStore(self.tmp.name)

    def test_numbering_parent_and_change(self):
        self.assertIsNone(self.store.current_version())
        one = self.store.save_version(make_spec(4))
        two = self.store.save_version(make_spec(6))
        self.assertEqual((one["number"], one["parent"], one["change"], one["status"]), (1, None, "first design", "saved"))
        self.assertEqual((two["number"], two["parent"], two["change"]), (2, 1, "more blocks (4 → 6)"))
        self.assertEqual(self.store.current_version()["number"], 2)
        self.assertEqual([v["number"] for v in VersionStore(self.tmp.name).versions()], [1, 2])   # persisted
        json.loads((Path(self.tmp.name) / "data" / "versions.json").read_text())
        self.assertEqual(self.store.save_version(make_spec(6), "my words")["change"], "my words")

    def test_status_transitions(self):
        self.store.save_version(make_spec())
        for status in ("queued", "training", "trained", "queued"):
            self.assertEqual(self.store.set_status(1, status)["status"], status)
        for bad in ("draft", "trained"):
            with self.assertRaises(ValueError):
                self.store.set_status(1, bad)
        self.assertEqual(self.store.get(1)["status"], "queued")
        with self.assertRaises(KeyError):
            self.store.set_status(9, "queued")

    def test_best_version_needs_checkpoint_and_lowest_bpc(self):
        for _ in range(3):
            self.store.save_version(make_spec())
        self.assertIsNone(self.store.best_version())
        for number, bpc in ((1, 3.0), (2, 2.5), (3, 2.0)):
            self.store.set_scorecard(number, {"bpc": bpc})
        for number in (1, 2):
            self.store.set_best_checkpoint(number, f"run{number}/best.npz")
        self.assertEqual(self.store.best_version()["number"], 2)   # 3 is lower but has no checkpoint
        self.store.add_run(2, "runs/a")
        self.store.set_verdict(2, "better")
        self.assertEqual((self.store.get(2)["runs"], self.store.get(2)["verdict"]), (["runs/a"], "better"))

    def test_new_weights_invalidate_the_old_assistant_card(self):
        self.store.save_version(make_spec())
        self.store.set_best_checkpoint(1, "first/best.npz")
        self.store.set_scorecard(1, {"bpc": 2.0, "assistant_card": {"suite": "frozen"}})
        self.store.set_best_checkpoint(1, "first/best.npz")
        self.assertEqual(self.store.get(1)["scorecard"]["assistant_card"]["suite"], "frozen")
        self.store.set_best_checkpoint(1, "second/best.npz")
        self.assertEqual(self.store.get(1)["scorecard"], {"bpc": 2.0})


if __name__ == "__main__":
    unittest.main()
