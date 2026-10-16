"""Builder ready-made designs: what each costs before it is loaded, its recipe, and loading it as an unsaved draft. No Tk window."""
import tempfile
import unittest
from unittest import mock

from studio_arch import model_spec_presets
from studio_glossary import GLOSSARY
from studio_ia import design as D
from studio_ia.pages import builder as B
from studio_spec import default_spec, validate_spec
from studio_versions import VersionStore

GB = 1e9


def preset(key, gpt2_words=False):
    return dict(D.preset_specs(gpt2_words))[key]


class Choices(unittest.TestCase):
    def test_every_studio_arch_preset_is_offered_with_words_and_a_glossary_entry(self):
        keys = [key for key, _ in D.preset_specs()]
        self.assertEqual(set(keys), set(model_spec_presets()))
        self.assertEqual(keys[:1], ["starter"])
        self.assertIn("gpt3_small_style", keys)
        for key in keys:
            self.assertTrue(B.T["preset_" + key].strip(), key)
            self.assertTrue(GLOSSARY[D.preset_term(key)][1].strip(), key)
        self.assertIn("2,048", GLOSSARY[D.preset_term("gpt3_small_style")][1])

    def test_gpt2_word_list_is_used_only_when_it_is_on_this_computer(self):
        self.assertEqual(preset("gpt3_small_style")["tokenizer"]["id"], "byte")
        words = preset("gpt3_small_style", gpt2_words=True)
        self.assertEqual(words["tokenizer"]["id"], "bpe_gpt2")
        self.assertIn("GPT-2 word list", words["name"])
        self.assertEqual(preset("starter", gpt2_words=True)["tokenizer"]["id"], "byte")  # only the GPT-2/3 shapes had it
        self.assertEqual(preset("tiny_llama_style", gpt2_words=True)["tokenizer"]["id"], "byte")
        with tempfile.TemporaryDirectory() as root, mock.patch("studio_core.ROOT", __import__("pathlib").Path(root)):
            self.assertFalse(D.gpt2_words_ready())  # nothing downloaded to find out


class Cost(unittest.TestCase):
    def test_starter_fits_any_computer_and_is_quick(self):
        cost = D.preset_cost(preset("starter"), ram=8 * GB, gpu="")
        self.assertTrue(cost["fits"])
        self.assertFalse(cost["slow"])
        self.assertEqual(cost["rows"], D.ROWS)
        self.assertTrue(cost["lines"][1].startswith("Fits this computer"))

    def test_gpt3_small_says_how_many_rows_fit_and_that_cpu_is_very_slow(self):
        spec = preset("gpt3_small_style")
        cost = D.preset_cost(spec, ram=16 * GB, gpu="")
        c = D.sizes(spec, D.read_spec(spec)["V"])
        self.assertEqual(cost["params"], c["params"])
        self.assertFalse(cost["fits"])
        self.assertEqual(cost["rows"], int((16 * GB - c["memory"]) // c["activations"]))
        self.assertGreater(cost["rows"], 0)
        self.assertIn(f"It fits at {cost['rows']} rows per step", cost["lines"][1])
        self.assertTrue(cost["slow"])
        self.assertTrue(cost["lines"][2].startswith("Without a GPU"))
        self.assertIn("Very slow", cost["lines"][2])

    def test_gpt3_medium_does_not_fit_a_laptop_at_all(self):
        cost = D.preset_cost(preset("gpt3_medium_style"), ram=16 * GB, gpu="")
        self.assertFalse(cost["fits"])
        self.assertEqual(cost["rows"], 0)
        self.assertTrue(cost["lines"][1].startswith("Won't fit this computer"))
        self.assertTrue(cost["slow"])

    def test_with_the_gpt2_word_list_the_sizes_match_the_originals(self):
        self.assertAlmostEqual(D.preset_cost(preset("gpt2_small_style", True), ram=GB)["params"] / 1e6, 124, delta=1)
        self.assertAlmostEqual(D.preset_cost(preset("gpt3_small_style", True), ram=GB)["params"] / 1e6, 125, delta=1)
        self.assertAlmostEqual(D.preset_cost(preset("gpt3_medium_style", True), ram=GB)["params"] / 1e6, 350, delta=10)

    def test_a_gpu_is_faster_and_said_so(self):
        spec = preset("gpt3_small_style")
        cpu, gpu = D.preset_cost(spec, ram=GB, gpu=""), D.preset_cost(spec, ram=GB, gpu="cuda")
        self.assertLess(gpu["step"], cpu["step"])
        self.assertTrue(gpu["lines"][2].startswith("With the GPU"))

    def test_gpu_is_only_read_from_an_earlier_check_never_probed(self):
        with mock.patch.dict("sys.modules", {"studio_backend": None}):
            self.assertEqual(D.gpu_kind(), "")  # torch not loaded here: no import, no Metal/CUDA work
        status = mock.Mock(return_value={"available": True, "kind": "mps"})
        status.cache_info.return_value.currsize = 0
        with mock.patch.dict("sys.modules", {"studio_backend": mock.Mock(gpu_status=status)}):
            self.assertEqual(D.gpu_kind(), "")
            status.cache_info.return_value.currsize = 1
            self.assertEqual(D.gpu_kind(), "mps")
        self.assertEqual(status.call_count, 1)

    def test_unknown_memory_is_said_plainly(self):
        with mock.patch.object(D, "ram_bytes", return_value=None):
            cost = D.preset_cost(preset("starter"))
        self.assertIsNone(cost["fits"])
        self.assertEqual(cost["lines"][1], B.T["preset_fit_unknown"])

    def test_readable_amounts(self):
        self.assertEqual(D.amount(86_812_416), "86.8M")
        self.assertEqual(D.gigabytes(1.39e9), "1.4 GB")
        self.assertEqual(D.duration(274), "about 5 minutes")
        self.assertEqual(D.duration(274_000), "about 3 days")
        self.assertEqual(D.duration(.2), "under a second")


class Recipe(unittest.TestCase):
    def test_gpt3_recipe_in_one_line(self):
        text = D.recipe_text(preset("gpt3_small_style"))
        for part in ("AdamW", "learning rate 0.0006", "β2 0.95", "weight decay 0.1", "to 10%"):
            self.assertIn(part, text)

    def test_loading_a_preset_is_an_unsaved_draft_that_keeps_its_recipe(self):
        base = validate_spec(default_spec())
        spec = preset("gpt3_small_style")
        draft, anchor = B.preset_draft(spec)
        self.assertEqual(anchor[0], spec)
        written = D.write_spec(anchor[0], draft, anchor[1])
        self.assertEqual(written, spec)  # width, reading window and recipe come with it
        chips = D.change_chips(base, written, D.read_spec(base), draft)
        self.assertIn("longer reading window (256 → 2048)", chips)
        self.assertIn(B.T["recipe_change"].format(recipe=D.recipe_text(spec)), chips)
        heads = D.write_spec(anchor[0], dict(draft, L=10), anchor[1])  # a later edit keeps the recipe too
        self.assertEqual(heads["train"]["optimizer"]["config"]["beta2"], .95)
        self.assertEqual(heads["dims"]["n_layer"], 10)

    def test_the_same_recipe_gets_no_chip(self):
        base = validate_spec(default_spec())
        draft, anchor = B.preset_draft(preset("starter"))
        chips = D.change_chips(base, D.write_spec(anchor[0], draft, anchor[1]), D.read_spec(base), draft)
        self.assertFalse(any(c.startswith("training recipe") for c in chips))

    def test_saving_keeps_the_recipe(self):
        spec = preset("gpt3_small_style")
        with tempfile.TemporaryDirectory() as root:
            store = VersionStore(root)
            store.save_version(validate_spec(default_spec()), "first design")
            saved = store.save_version(spec)
            stored = store.versions()[-1]["spec"]
        self.assertEqual(saved["spec"]["train"], spec["train"])
        self.assertEqual(stored["train"]["optimizer"]["config"], {"lr": 6e-4, "beta1": .9, "beta2": .95, "eps": 1e-8, "weight_decay": .1})
        self.assertEqual(stored["train"]["schedule"]["config"]["min_lr"], .1)
        cfg = B.config_from_spec(stored)  # what a catch-up job trains with
        self.assertEqual((cfg["learning_rate"], cfg["weight_decay"], cfg["context"]), (6e-4, .1, 2048))


if __name__ == "__main__":
    unittest.main()
