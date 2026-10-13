"""Estimator conventions and exact default-model count gates."""
import copy
import unittest

import studio_registry as registry
from studio_compile import CompiledModel
from studio_estimate import VOCAB_SLOTS, estimate, per_block
from studio_spec import validate_spec


def small_spec():
    return validate_spec({"dims": {"d_model": 16, "n_layer": 2, "n_ctx": 32, "dropout": 0}})


class Estimates(unittest.TestCase):
    def test_baseline_exact_count_and_units(self):
        spec = validate_spec({"dims": {"d_model": 256, "n_layer": 4, "n_ctx": 256, "dropout": 0},
                              "block": {"mixer": {"id": "attention", "config": {"heads": 4}}}})
        result = estimate(spec, vocab=50257)
        self.assertEqual(result["total"], 16_090_880)
        self.assertEqual(result["active"], result["total"])
        self.assertEqual(result["embedding_params"], 12_931_328)
        self.assertEqual(result["block_params"], 3_159_552)
        self.assertEqual(per_block(result), [789_760] * 4)
        self.assertEqual(result["train_flops_per_token"], 20_530_176)
        self.assertEqual(result["kv_bytes_per_token"], 8192)
        self.assertEqual(result["activation_bytes_per_token"], 100_352)
        # Units: FLOPs cover approximate forward+backward per token, KV and
        # activation storage are bytes per token. The tied head adds no weights.
        self.assertEqual(next(row for row in result["rows"] if row["slot"] == "head")["params"], 0)
        self.assertEqual(result["train_flops_per_token"], 3 * sum(row["flops_per_token"] for row in result["rows"]))
        self.assertTrue(all(row["flops_per_token"] == 0 for row in result["rows"] if row["slot"] in VOCAB_SLOTS))

    def test_registry_module_counts_match_compiled_weights(self):
        base = small_spec()
        for slot in ("embedding", "position", "norm", "mixer", "channel", "head"):
            for module in registry.for_slot(slot):
                with self.subTest(module=module.id):
                    spec = copy.deepcopy(base)
                    destination = spec["block"] if slot in ("norm", "mixer", "channel") else spec
                    destination[slot] = {"id": module.short}
                    model = CompiledModel(spec, 31, seed=7, device="cpu")
                    result = estimate(spec, vocab=31)
                    self.assertEqual(result["total"], model.parameter_count)
                    self.assertLessEqual(result["active"], result["total"])
                    self.assertEqual(sum(row["params"] for row in result["rows"]), result["total"])
                    for row in result["rows"]:
                        self.assertGreaterEqual(row["activation_bytes"], 0)
                        self.assertGreaterEqual(row["kv_bytes"], 0)

    def test_skipped_branches_omit_their_norms(self):
        spec = small_spec()
        both = estimate(spec, vocab=31)
        for slot in ("mixer", "channel"):
            with self.subTest(slot=slot):
                changed = copy.deepcopy(spec)
                changed["block"][slot] = {"id": "none"}
                result = estimate(changed, vocab=31)
                rows = [row for row in result["rows"] if row["block"] is not None]
                self.assertEqual(sum(row["slot"] == "norm" for row in rows), 2)
                self.assertTrue(all(row["skip"] and row["params"] == 0 for row in rows if row["slot"] == slot))
                removed = 2 * (next(row["params"] for row in both["rows"] if row["slot"] == slot) + 32)
                self.assertEqual(both["total"] - result["total"], removed)
                self.assertEqual(result["total"], CompiledModel(changed, 31, device="cpu").parameter_count)

    def test_hybrid_rows_and_precision_memory(self):
        spec = small_spec()
        spec["blocks"] = [{"index": 1, "mixer": {"id": "none"}}]
        normalized = validate_spec(spec)
        result = estimate(normalized, vocab=31)
        mixer = next(row["params"] for row in result["rows"] if row["block"] == 0 and row["slot"] == "mixer")
        norm = next(row["params"] for row in result["rows"] if row["block"] == 0 and row["slot"] == "norm")
        self.assertEqual(per_block(result)[1], per_block(result)[0] - mixer - norm)
        self.assertEqual(result["total"], CompiledModel(normalized, 31, device="cpu").parameter_count)
        double = copy.deepcopy(normalized)
        double["train"]["precision"] = {"id": "fp64"}
        double_result = estimate(double, vocab=31)
        self.assertEqual(double_result["total"], result["total"])
        self.assertEqual(double_result["kv_bytes_per_token"], 2 * result["kv_bytes_per_token"])
        self.assertEqual(double_result["activation_bytes_per_token"], 2 * result["activation_bytes_per_token"])


if __name__ == "__main__":
    unittest.main()
