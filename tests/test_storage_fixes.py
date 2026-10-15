"""Regression tests for storage-layer and trainer fixes (core, catalog, archdb, model)."""
import json
import random
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import studio_arch
import studio_archdb
import studio_catalog
import studio_core
import studio_model
from studio_core import Workspace, atomic_json, normalize, parse_file


class CoreFixes(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def test_null_quality_survives_move_to_disk(self):
        ws = Workspace(self.dir)
        ds = ws.create_dataset("d")
        ws.state["examples"].append({"id": "ex_1", "datasetId": ds, "messages": [], "text": "", "user": "q", "assistant": "a", "intent": "x", "quality": None})
        ws.use_disk_storage(ds)
        self.assertEqual(ws.count(ds), 1)
        self.assertEqual(normalize({"user": "a", "assistant": "b", "quality": None})["quality"], "Review")

    def test_failed_move_keeps_memory_copy(self):
        ws = Workspace(self.dir)
        ds = ws.create_dataset("d")
        ws.state["examples"].append({"id": "ex_1", "datasetId": ds, "user": "q", "assistant": "a", "intent": "x"})
        with patch.object(Workspace, "_db_row", side_effect=lambda e: ("ex_1", ds, "f", "{}", None, "", "", "", 0, "text", 0)):
            with self.assertRaises(Exception):
                ws.use_disk_storage(ds)
        self.assertEqual(len(ws.state["examples"]), 1)
        self.assertNotEqual(ws._dataset(ds).get("storage"), "sqlite")

    def test_jsonl_keeps_unicode_line_separators_inside_strings(self):
        path = self.dir / "a.jsonl"
        path.write_bytes(('{"user":"a b\x85c ","assistant":"x"}\r\n{"user":"c","assistant":"d"}\r\n').encode("utf-8"))
        rows = parse_file(path)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["user"], "a b\x85c ")

    def test_failed_large_import_leaves_no_dataset_and_keeps_selection(self):
        ws = Workspace(self.dir)
        keep = ws.create_dataset("keep")
        path = self.dir / "big.jsonl"
        path.write_text('{"user":"a","assistant":"b"}\n{broken\n' + " " * (11 * 1024 * 1024) + "\n", encoding="utf-8")
        report = ws.import_files([path])
        self.assertIn("invalid JSON", report[0])
        self.assertEqual([d["id"] for d in ws.state["datasets"]], [keep])
        self.assertEqual(ws.state["selectedDatasetId"], keep)
        self.assertEqual(ws.count(keep), 0)

    def test_atomic_json_uses_unique_temp_and_cleans_up(self):
        target = self.dir / "s.json"
        names = []
        real = studio_core.replace_file

        def spy(source, dest, *a):
            names.append(Path(source).name)
            return real(source, dest, *a)
        with patch.object(studio_core, "replace_file", spy):
            atomic_json(target, {"a": 1})
            atomic_json(target, {"a": 2})
        self.assertNotEqual(names[0], names[1])
        self.assertNotEqual(names[0], "s.json.tmp")
        with patch.object(studio_core, "replace_file", side_effect=OSError("boom")):
            with self.assertRaises(OSError):
                atomic_json(target, {"a": 3})
        self.assertEqual([p.name for p in self.dir.iterdir()], ["s.json"])
        self.assertEqual(json.loads(target.read_text()), {"a": 2})

    def test_synthetic_string_false_is_false(self):
        self.assertFalse(normalize({"user": "a", "assistant": "b", "synthetic": "False"})["synthetic"])
        self.assertTrue(normalize({"user": "a", "assistant": "b", "synthetic": "true"})["synthetic"])


class CatalogFixes(unittest.TestCase):
    @staticmethod
    def fetcher(response):
        def fake(url, path, stop=None, emit=None, maximum=None, **kw):
            if Path(path).name == "SOURCE_README.md":
                raise OSError("card unavailable")
            payload = (json.dumps({"instruction": "Say hello", "response": response, "context": "", "category": "g"}) + "\n").encode() if Path(path).suffix == ".jsonl" else b"x"
            Path(path).write_bytes(payload)
            return {"url": url, "filename": Path(path).name, "bytes": len(payload), "sha256": studio_catalog.file_sha256(path)}
        return fake

    def test_missing_card_does_not_abort_and_reregistration_replaces_stale_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("studio_catalog._fetch", side_effect=self.fetcher("Hello")):
                first = studio_catalog.download_dataset("dolly15k", root=directory)
            ws = Workspace(directory)
            dataset_id = studio_catalog.register_download(ws, first)
            self.assertEqual(ws.examples(dataset_id)[0]["assistant"], "Hello")
            for name in ("records.jsonl", "manifest.json", "source.json", "databricks-dolly-15k.jsonl"):
                (Path(directory) / "data" / "downloads" / "dolly15k" / name).unlink(missing_ok=True)
            with patch("studio_catalog._fetch", side_effect=self.fetcher("Hi there")):
                second = studio_catalog.download_dataset("dolly15k", root=directory)
            studio_catalog.register_download(ws, second)
            self.assertEqual([e["assistant"] for e in ws.examples(dataset_id)], ["Hi there"])
            self.assertEqual(ws._dataset(dataset_id)["sourceSha256"], second["sha256"])

    def test_old_workspace_backups_are_pruned(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("studio_catalog._fetch", side_effect=self.fetcher("Hello")):
                result = studio_catalog.download_dataset("dolly15k", root=directory)
            ws = Workspace(directory)
            ws.save()
            for i in range(8):
                (ws.path.parent / f"workspace.before-catalog-2020010{i}-000000.json").write_text("{}")
            studio_catalog.register_download(ws, result)
            self.assertLessEqual(len(list(ws.path.parent.glob("workspace.before-catalog-*.json"))), 3)

    def test_oasst_synthetic_string_false(self):
        entry = next(e for e in studio_catalog.CATALOG if e["parser"] == "oasst")
        raw = {"message_tree_id": "t", "prompt": {"role": "prompter", "lang": "en", "text": "q", "synthetic": "False", "replies": [{"role": "assistant", "text": "a", "message_id": "m"}]}}
        self.assertFalse(list(studio_catalog.convert_record(raw, entry))[0]["synthetic"])


class ArchDbFixes(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.store = studio_archdb.ArchitectureStore(self.root)

    def test_member_pattern_rejects_trailing_newline(self):
        self.assertTrue(studio_archdb.MEMBER.match("weights/a/v0001.npz"))
        self.assertIsNone(studio_archdb.MEMBER.match("architectures.sqlite\n"))
        self.assertIsNone(studio_archdb.MEMBER.match("weights/a/v0001.npz\n"))

    def test_import_spec_reads_bom_and_validates(self):
        good = studio_arch.default_spec("Net")
        path = self.root / "good.json"
        path.write_bytes(b"\xef\xbb\xbf" + json.dumps({"format": studio_archdb.EXPORT_FORMAT, "name": "Net", "spec": good}).encode())
        self.assertEqual(self.store.get(self.store.import_spec(path))["name"], "Net")
        bad = self.root / "bad.json"
        bad.write_text(json.dumps({"format": studio_archdb.EXPORT_FORMAT, "name": "Bad", "spec": {"blocks": 3}}), encoding="utf-8")
        with self.assertRaises(ValueError):
            self.store.import_spec(bad)
        self.assertEqual(len(self.store.list()), 1)

    def test_failed_swap_keeps_old_weights_and_database(self):
        arch = self.store.create("Keep", {"format": "groundwork-arch/1", "name": "k", "embedding": {}, "blocks": []})
        weights = self.store.weights_dir / arch / "v0001.npz"
        weights.parent.mkdir(parents=True)
        weights.write_bytes(b"old")
        staging = self.root / "staging"
        (staging / "weights").mkdir(parents=True)
        (staging / "weights" / "new.npz").write_bytes(b"new")
        (staging / "architectures.sqlite").write_bytes(b"x")
        real = studio_archdb.replace_file

        def flaky(source, target, *a):
            if Path(target) == self.store.path:
                raise OSError("disk full")
            return real(source, target, *a)
        with patch.object(studio_archdb, "replace_file", flaky):
            with self.assertRaises(OSError):
                self.store._swap(staging, True)
        self.assertEqual(weights.read_bytes(), b"old")
        self.assertEqual(self.store.get(arch)["name"], "Keep")


TINY = dict(device="cpu", width=16, layers=1, heads=2, context=32, expansion=2, dropout=0, batch_size=2,
            steps=4, learning_rate=.004, warmup=0, validation=0, eval_every=2, objective="assistant")
EXAMPLES = [{"user": f"say {w}", "assistant": w} for w in ("yes", "no", "cat", "dog")] * 2


class ModelFixes(unittest.TestCase):
    def test_weight_decay_skips_gains_and_biases(self):
        model = studio_model.ScratchTransformer({**TINY, "weight_decay": .5}, 16)
        before = {k: v.detach().clone() for k, v in model.p.items()}
        studio_model.AdamW(model, .5).update(model, .1)  # gradients are all zero: only decay can move a parameter
        for key, value in model.p.items():
            if key.endswith((".g", ".b")):
                self.assertTrue(bool((value == before[key]).all()), key)
            elif key.endswith(".w") or key in ("token", "position"):
                self.assertTrue(bool((value != before[key]).any()), key)

    def test_best_checkpoint_without_validation_uses_fixed_windows_not_one_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            examples = [studio_core.normalize(e) for e in EXAMPLES]
            record = studio_model.train(TINY, examples, out)
            cfg = studio_model.validate_config(TINY)
            data = studio_model.load_checkpoint(out / "best.npz")
            tokenizer = studio_model.Tokenizer.restore(data["tokenizer"])
            model = studio_model.ScratchTransformer(cfg, tokenizer.vocab_size)
            model.load_weights(data["model"])
            kept, _, _ = studio_model.partition_examples([json.loads(line) for line in (out / "dataset.jsonl").read_text(encoding="utf-8").splitlines()], 0, random.Random(cfg["seed"]))
            samples = studio_model.samples_from(kept, tokenizer, cfg)
            total = count = 0
            for start in range(0, min(len(samples), cfg["batch_size"] * 20), cfg["batch_size"]):
                x, y = studio_model.batch_arrays(samples[start:start + cfg["batch_size"]])
                n = int((y != -100).sum())
                total += model.loss(x, y) * n
                count += n
            self.assertAlmostEqual(record["best_loss"], total / count, places=5)

    def test_interrupted_evaluation_never_writes_best_checkpoint(self):
        stop, armed = threading.Event(), []
        real = studio_model.ScratchTransformer.loss

        def loss(self, x, y):
            value = real(self, x, y)
            if armed:
                stop.set()
            return value
        with tempfile.TemporaryDirectory() as directory, patch.object(studio_model.ScratchTransformer, "loss", loss):
            studio_model.train(TINY, [studio_core.normalize(e) for e in EXAMPLES], Path(directory), lambda e: armed.append(1) if e["type"] == "metric" else None, stop=stop)
            self.assertTrue(stop.is_set())
            self.assertFalse((Path(directory) / "best.npz").exists())


if __name__ == "__main__":
    unittest.main()
