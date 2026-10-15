"""studio_model boundaries: config validation limits, tokenizers, checkpoint file guards, parameter estimates."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from studio_model import Tokenizer, load_checkpoint, parameter_estimate, train, validate_config

TINY = dict(device="cpu", width=16, layers=1, heads=2, context=32, expansion=2, dropout=0, batch_size=2,
            steps=4, learning_rate=.004, warmup=0, validation=0, eval_every=2, objective="assistant")
EXAMPLES = [{"user": f"say {w}", "assistant": w} for w in ("yes", "no", "cat", "dog")] * 2


class ValidateConfig(unittest.TestCase):
    def test_each_numeric_bound_is_enforced_on_both_sides(self):
        integers = {"width", "layers", "context", "batch_size", "steps"}
        for key, low, high in (("width", 16, 512), ("layers", 1, 12), ("context", 32, 2048), ("batch_size", 1, 128), ("steps", 1, 1000000),
                               ("dropout", 0, .5), ("validation", 0, .4), ("learning_rate", 1e-7, .1), ("weight_decay", 0, 1)):
            base = {"heads": 1, "width": 32}
            for good in (low, high):
                self.assertEqual(float(validate_config({**base, key: good})[key]), float(good), (key, good))
            for bad in (low - 1 if key in integers else low / 2 - 1, high + 1):
                with self.assertRaisesRegex(ValueError, key, msg=(key, bad)):
                    validate_config({**base, key: bad})

    def test_string_values_from_the_form_are_coerced_and_junk_rejected(self):
        cfg = validate_config({"width": "32", "heads": "2", "steps": "10", "learning_rate": "0.01"})
        self.assertEqual((cfg["width"], cfg["steps"], cfg["learning_rate"]), (32, 10, .01))
        for junk in ("ten", ""):
            with self.assertRaises(ValueError):
                validate_config({"steps": junk})

    def test_head_divisibility_and_enumerations(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            validate_config({"width": 30, "heads": 4})
        for key, bad in (("tokenizer", "word"), ("objective", "some"), ("device", "tpu"), ("windowing", "x")):
            with self.assertRaises(ValueError, msg=key):
                validate_config({key: bad})

    def test_unknown_keys_are_dropped_and_input_not_mutated(self):
        source = {"width": 32, "heads": 2, "surprise": 1}
        cfg = validate_config(source)
        self.assertNotIn("surprise", cfg)
        self.assertEqual(source, {"width": 32, "heads": 2, "surprise": 1})

    def test_architecture_spec_may_be_blank_or_json_string(self):
        for blank in ("", "None", "null", None):
            self.assertIsNone(validate_config({"architecture": blank})["architecture"])
        with self.assertRaises(ValueError):
            validate_config({"architecture": "{not json"})


class Tokens(unittest.TestCase):
    def test_byte_round_trip_including_multibyte_text(self):
        tokenizer = Tokenizer("byte")
        for text in ("hello", "héllo ✓ \U0001f600", ""):
            self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_character_vocabulary_unknowns_and_serialisation(self):
        tokenizer = Tokenizer.fit("character", [{"user": "ab", "assistant": "ba"}, {"text": "c"}])
        self.assertEqual(tokenizer.characters, ["a", "b", "c"])
        self.assertEqual(tokenizer.decode(tokenizer.encode("abc")), "abc")
        self.assertEqual(tokenizer.decode(tokenizer.encode("azb")), "ab")           # unseen characters decode to nothing
        again = Tokenizer.restore(json.loads(json.dumps(tokenizer.serialize())))
        self.assertEqual(again.encode("cab"), tokenizer.encode("cab"))
        self.assertEqual(again.vocab_size, tokenizer.vocab_size)
        with self.assertRaises(ValueError):
            Tokenizer.restore({"kind": "word"})

    def test_huge_character_vocabulary_is_refused(self):
        rows = [{"text": "".join(chr(0x4e00 + i) for i in range(start, start + 1000))} for start in range(0, 17000, 1000)]
        with self.assertRaisesRegex(ValueError, "byte tokenizer"):
            Tokenizer.fit("character", rows)


class Checkpoints(unittest.TestCase):
    def test_wrong_and_corrupt_files_are_refused_with_clear_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            foreign = root / "foreign.npz"
            np.savez(foreign, metadata=np.array(json.dumps({"architecture": "other"})))
            with self.assertRaisesRegex(ValueError, "Groundwork"):
                load_checkpoint(foreign)
            future = root / "future.npz"
            np.savez(future, metadata=np.array(json.dumps({"architecture": "groundwork_numpy_decoder", "format_version": 2})))
            with self.assertRaisesRegex(ValueError, "Groundwork"):
                load_checkpoint(future)
            garbage = root / "garbage.npz"
            garbage.write_bytes(b"not a zip at all")
            with self.assertRaises(Exception):
                load_checkpoint(garbage)
            with self.assertRaises(OSError):
                load_checkpoint(root / "missing.npz")

    def test_saved_checkpoint_contains_weights_optimizer_and_estimate_matches(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "run"
            train(TINY, EXAMPLES, out)
            checkpoint = load_checkpoint(out / "last.npz")
            self.assertEqual(checkpoint["step"], TINY["steps"])
            self.assertEqual(set(checkpoint["model"]), set(checkpoint["m"]))
            self.assertEqual(set(checkpoint["model"]), set(checkpoint["v"]))
            counted = sum(int(w.size) for w in checkpoint["model"].values())
            cfg = validate_config(TINY)
            self.assertEqual(parameter_estimate(cfg, Tokenizer("byte").vocab_size), counted)
            record = json.loads((out / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(record["parameters"], counted)
            self.assertEqual(sorted(p.name for p in out.glob("*.tmp")), [])           # no half-written files left behind

    def test_resume_continues_after_the_saved_step(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "a", Path(directory) / "b"
            train(TINY, EXAMPLES, first)
            events = []
            train(dict(TINY, steps=6), EXAMPLES, second, events.append, resume=first / "last.npz")
            steps = [e["step"] for e in events if e["type"] == "metric"]
            self.assertTrue(steps and min(steps) > TINY["steps"] and max(steps) == 6, steps)
            self.assertEqual(load_checkpoint(second / "last.npz")["step"], 6)


if __name__ == "__main__":
    unittest.main()
