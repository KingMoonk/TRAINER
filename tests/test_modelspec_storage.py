"""ModelSpec migration and storage alongside the legacy architecture UI API."""
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import numpy as np

from studio_arch import default_spec, model_spec_presets
from studio_archdb import ArchitectureStore, SCHEMA, spec_hash
from studio_spec import from_legacy, validate_spec as validate_model_spec


class ModelSpecStorage(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = ArchitectureStore(self.root)
        # Exercise the real pre-migration table layout, including no legacy JSON column.
        self.store.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.store.path)) as connection:
            connection.executescript(SCHEMA.replace(", legacy_spec_json TEXT", ""))

    def tearDown(self):
        self.temp.cleanup()

    def test_transactional_migration_preserves_legacy_views_links_and_backup_restore(self):
        first = default_spec("First")
        second = default_spec("Second")
        second["embedding"]["width"] = 128
        arch = self.store.create("Network", first)
        self.store.commit(arch, second, note="wider")
        weights = self.root / "sample.npz"
        np.savez(weights, value=np.arange(4))
        version = self.store.add_weights(arch, weights, revision=1, step=7)
        invalid = {"format": "groundwork-arch/1", "blocks": []}
        incomplete = self.store.create("Incomplete historical row", invalid)
        before = self.store.revisions(arch)
        backup = self.store.backup("before-migration")

        with patch("studio_archdb.upgrade", side_effect=RuntimeError("interrupted")):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                self.store.migrate_model_specs()
        with closing(sqlite3.connect(self.store.path)) as connection:
            self.assertEqual(connection.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0], "1")
            self.assertNotIn("legacy_spec_json", {row[1] for row in connection.execute("PRAGMA table_info(revisions)")})

        self.assertEqual(self.store.migrate_model_specs(), {"migrated": 2, "skipped": 1})
        self.assertEqual(self.store.migrate_model_specs(), {"migrated": 0, "skipped": 1})
        self.assertEqual(self.store.get(arch)["spec"], second)
        self.assertEqual(self.store.revision(arch, 1), first)
        self.assertEqual(self.store.get(incomplete)["spec"], invalid)
        self.assertEqual(self.store.get_model_spec(arch), from_legacy(second))
        self.assertEqual(self.store.revision_model_spec(arch, 1), from_legacy(first))
        self.assertEqual([(row["number"], row["parent"]) for row in self.store.revisions(arch)],
                         [(row["number"], row["parent"]) for row in before])
        self.assertEqual(self.store.weights(arch)[0]["revision"], 1)
        self.assertEqual(Path(self.store.weight_path(arch, version)).read_bytes(), weights.read_bytes())
        with closing(sqlite3.connect(self.store.path)) as connection:
            raw, digest = connection.execute(
                "SELECT spec_json, spec_hash FROM revisions WHERE architecture_id=? AND number=2", (arch,)).fetchone()
            self.assertEqual(validate_model_spec(json.loads(raw)), from_legacy(second))
            self.assertEqual(digest, spec_hash(from_legacy(second)))

        # An old snapshot remains restorable, and can then be migrated again.
        self.store.restore(backup)
        self.assertEqual(self.store.get(arch)["spec"], second)
        self.assertEqual(self.store.migrate_model_specs(), {"migrated": 2, "skipped": 1})
        self.assertEqual(Path(self.store.weight_path(arch, version)).read_bytes(), weights.read_bytes())

    def test_normalized_model_spec_create_commit_export_import(self):
        spec = model_spec_presets()["starter"]
        arch = self.store.create_model_spec("Modern", spec)
        self.assertEqual(self.store.get_model_spec(arch), spec)
        self.assertEqual(self.store.commit_model_spec(arch, {"schema": 1, "name": "Starter decoder",
                                                           "dims": {"d_model": 64, "n_layer": 2, "n_ctx": 256, "dropout": .1}}), 1)
        changed = dict(spec, name="Edited")
        self.assertEqual(self.store.commit_model_spec(arch, changed), 2)
        self.assertEqual(self.store.get_model_spec(arch)["name"], "Edited")
        self.assertEqual(self.store.checkout(arch, 1), 3)
        self.assertEqual(self.store.get_model_spec(arch), spec)
        self.store.commit_model_spec(arch, changed)
        path = self.root / "modern.json"
        self.store.export_model_spec(arch, path)
        self.assertEqual(json.loads(path.read_text())["format"], "groundwork-model-spec-export/1")
        copied = self.store.import_model_spec(path)
        self.assertEqual(self.store.get_model_spec(copied), self.store.get_model_spec(arch))
        self.assertEqual(self.store.get(copied)["name"], "Modern")

    def test_presets_use_registered_modules_and_explicit_sizes(self):
        presets = model_spec_presets()
        self.assertEqual(presets["gpt2_small_style"]["dims"]["d_model"], 768)
        self.assertEqual(presets["gpt2_small_style"]["block"]["mixer"]["config"]["heads"], 12)
        for spec in presets.values():
            self.assertEqual(validate_model_spec(spec), spec)
        if "tiny_llama_style" in presets:
            self.assertEqual(presets["tiny_llama_style"]["position"]["id"], "rope")

    def test_gpt3_presets_carry_the_paper_shape_and_recipe(self):
        presets = model_spec_presets()
        for key, (width, layers, heads, lr) in {"gpt3_small_style": (768, 12, 12, 6e-4), "gpt3_medium_style": (1024, 24, 16, 3e-4)}.items():
            spec = presets[key]
            self.assertEqual(spec["dims"], {"d_model": width, "n_layer": layers, "n_ctx": 2048, "dropout": .1})
            self.assertEqual((spec["block"]["mixer"]["id"], spec["block"]["mixer"]["config"]["heads"]), ("attention", heads))
            self.assertEqual((spec["block"]["channel"]["id"], spec["block"]["channel"]["config"]["d_ff"]), ("mlp_gelu", 4 * width))
            self.assertEqual((spec["block"]["norm"]["id"], spec["position"]["id"], spec["head"]["id"]), ("layernorm", "learned", "tied"))
            self.assertEqual(spec["tokenizer"]["id"], presets["gpt2_small_style"]["tokenizer"]["id"])
            optimizer = spec["train"]["optimizer"]
            self.assertEqual(optimizer["id"], "adamw")
            self.assertEqual({k: optimizer["config"][k] for k in ("lr", "beta1", "beta2", "eps", "weight_decay")},
                             {"lr": lr, "beta1": .9, "beta2": .95, "eps": 1e-8, "weight_decay": .1})
            self.assertEqual((spec["train"]["schedule"]["id"], spec["train"]["schedule"]["config"]["min_lr"]), ("cosine", .1))
            self.assertGreater(spec["train"]["schedule"]["config"]["warmup"], 0)
        # The recipe rides on the preset: a plain spec keeps the global defaults.
        starter = presets["starter"]["train"]["optimizer"]["config"]
        self.assertEqual((starter["beta2"], starter["weight_decay"]), (.999, .01))


if __name__ == "__main__":
    unittest.main()
