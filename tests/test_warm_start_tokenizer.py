"""A warm start (train(init_from=...)) keeps the checkpoint's tokenizer instead of refitting it on the new data."""
import tempfile
import unittest
from pathlib import Path

from studio_model import load_checkpoint, train
from studio_spec import config_from_spec, validate_spec
from tests.test_tokenizer_fast import NEEDS, corpus

RUN = {"batch_size": 2, "steps": 2, "validation": 0, "eval_every": 2, "seed": 3, "objective": "all", "device": "cpu"}
CONVERSATIONS = [{"messages": [{"role": "user", "content": f"zebra quartz {i}, wxyz?"},
                               {"role": "assistant", "content": f"Jumbo vexing {i} qwerty."}]} for i in range(40)]


def spec_for(tokenizer):
    spec = validate_spec({"schema": 1})
    spec["dims"].update(d_model=16, n_layer=1, n_ctx=32, dropout=0.)
    spec["tokenizer"] = {"id": tokenizer, "config": {"vocab_size": 1024 if tokenizer == "bpe_fast" else 400} if tokenizer.startswith("bpe") else {}}
    spec["block"]["mixer"]["config"].update(heads=2)
    return validate_spec(spec)


def fingerprint(tokenizer):
    return tokenizer.get("sha256") or tokenizer


@NEEDS
class WarmStartTokenizer(unittest.TestCase):
    def warm_start(self, tokenizer, lora=None, **extra):
        with tempfile.TemporaryDirectory() as tmp:
            base, child = Path(tmp) / "base", Path(tmp) / "child"
            cfg = {**config_from_spec(spec_for(tokenizer)), **RUN}
            train(cfg, list(corpus(40)), base)
            before = load_checkpoint(base / "last.npz")
            second = {**before["config"], "objective": "assistant", **extra}
            if lora:
                second["lora"] = lora
            train(second, CONVERSATIONS, child, init_from=base / "last.npz")
            return before["tokenizer"], load_checkpoint(child / "last.npz")["tokenizer"]

    def test_the_checkpoint_tokenizer_survives_a_warm_start(self):
        for tokenizer in ("bpe_fast", "bpe_custom", "word", "character"):
            for lora in (None, {"rank": 2, "alpha": 4}):
                with self.subTest(tokenizer=tokenizer, lora=bool(lora)):
                    before, after = self.warm_start(tokenizer, lora)
                    self.assertEqual(fingerprint(after), fingerprint(before))

    def test_refit_tokenizer_fits_a_new_one_on_the_new_data(self):
        for tokenizer in ("bpe_fast", "character"):
            with self.subTest(tokenizer=tokenizer):
                before, after = self.warm_start(tokenizer, refit_tokenizer=True)
                self.assertNotEqual(fingerprint(after), fingerprint(before))


if __name__ == "__main__":
    unittest.main()
