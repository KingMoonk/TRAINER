"""Architecture database checks: revisions, weights, backups, restore safety and concurrency."""
import hashlib
import json
import os
import tempfile
import threading
import unittest
import zipfile
from pathlib import Path

import numpy as np

from studio_archdb import ArchitectureStore


def spec(name, width=64):
    return {"format": "groundwork-arch/1", "name": name, "embedding": {"width": width}, "blocks": []}


def fake_weights(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, w=np.full((4, 4), value, dtype=np.float32))
    return path


def sha_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ArchitectureDatabaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = ArchitectureStore(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_create_list_get_rename_delete(self):
        arch = self.store.create("Alpha", spec("alpha"))
        listing = self.store.list()
        self.assertEqual([entry["id"] for entry in listing], [arch])
        self.assertEqual(listing[0]["head_revision"], 1)
        self.assertEqual(listing[0]["revisions"], 1)
        self.assertEqual(listing[0]["weight_versions"], 0)
        record = self.store.get(arch)
        self.assertEqual(record["name"], "Alpha")
        self.assertEqual(record["spec"], spec("alpha"))
        self.store.rename(arch, "Beta")
        self.assertEqual(self.store.get(arch)["name"], "Beta")
        with self.assertRaises(ValueError):
            self.store.get("arch_missing")
        self.store.delete(arch)
        self.assertEqual(self.store.list(), [])

    def test_commit_dedupes_and_orders_revisions(self):
        arch = self.store.create("Alpha", spec("a", 64))
        self.assertEqual(self.store.commit(arch, spec("a", 64)), 1)
        self.assertEqual(self.store.commit(arch, spec("a", 128)), 2)
        self.assertEqual(self.store.commit(arch, spec("a", 256), note="wider"), 3)
        revisions = self.store.revisions(arch)
        self.assertEqual([r["number"] for r in revisions], [3, 2, 1])
        self.assertEqual(revisions[0]["note"], "wider")
        self.assertEqual(revisions[2]["note"], "created")
        self.assertIsNone(revisions[2]["parent"])
        self.assertEqual(revisions[1]["parent"], 1)
        self.assertEqual(self.store.revision(arch, 2), spec("a", 128))

    def test_checkout_creates_new_head_with_old_spec(self):
        arch = self.store.create("Alpha", spec("a", 64))
        self.store.commit(arch, spec("a", 128))
        self.assertEqual(self.store.checkout(arch, 1), 3)
        self.assertEqual(self.store.get(arch)["spec"], spec("a", 64))
        self.assertEqual(self.store.revisions(arch)[0]["note"], "restored from r1")
        self.assertEqual(self.store.checkout(arch, 3), 3)

    def test_add_weights_versions_and_path(self):
        arch = self.store.create("Alpha", spec("a"))
        first = fake_weights(self.root / "src" / "one.npz", 1.0)
        second = fake_weights(self.root / "src" / "two.npz", 2.0)
        self.assertEqual(self.store.add_weights(arch, first, step=10, loss=2.5, run_dir="runs/x", note="first"), 1)
        self.assertEqual(self.store.add_weights(arch, second, step=20, loss=1.5), 2)
        weights = self.store.weights(arch)
        self.assertEqual([w["version"] for w in weights], [2, 1])
        self.assertEqual(weights[0]["revision"], 1)
        self.assertEqual(weights[0]["sha256"], sha_of(second))
        self.assertTrue(weights[0]["path"].startswith("data/weights/"))
        path = self.store.weight_path(arch, 2)
        self.assertTrue(Path(path).is_absolute())
        self.assertEqual(Path(path).read_bytes(), second.read_bytes())
        with self.assertRaises(ValueError):
            self.store.add_weights(arch, self.root / "missing.npz")

    def test_corrupted_weight_file_is_detected(self):
        arch = self.store.create("Alpha", spec("a"))
        self.store.add_weights(arch, fake_weights(self.root / "src" / "one.npz", 1.0))
        self.store.add_weights(arch, fake_weights(self.root / "src" / "two.npz", 2.0))
        with open(self.store.weight_path(arch, 1), "ab") as handle:
            handle.write(b"junk")
        with self.assertRaisesRegex(ValueError, "corrupted or missing"):
            self.store.weight_path(arch, 1)
        self.store.weight_path(arch, 2)
        os.remove(self.store.weight_path(arch, 2))
        with self.assertRaisesRegex(ValueError, "corrupted or missing"):
            self.store.weight_path(arch, 2)

    def test_backup_mutate_restore(self):
        arch = self.store.create("Alpha", spec("alpha"))
        self.store.commit(arch, spec("alpha2"))
        weights = fake_weights(self.root / "src" / "one.npz", 3.0)
        self.store.add_weights(arch, weights)
        other = self.store.create("Other", spec("other"))
        original_bytes = weights.read_bytes()

        zip_path = self.store.backup("nightly", include_weights=True)
        self.assertIn("-nightly.zip", Path(zip_path).name)
        self.assertEqual(self.store.backups()[0]["path"], zip_path)
        self.assertEqual(self.store.backups()[0]["label"], "nightly")

        self.store.rename(arch, "Renamed")
        self.store.commit(arch, spec("alpha3"))
        self.store.delete(other)
        os.remove(self.store.weight_path(arch, 1))

        result = self.store.restore(zip_path)
        self.assertEqual(result["restored"], zip_path)
        self.assertEqual(result["architectures"], 2)
        self.assertTrue(result["weights_restored"])
        self.assertTrue(Path(result["pre_restore_backup"]).is_file())
        self.assertIn("pre-restore", Path(result["pre_restore_backup"]).name)
        # The weight deleted before restore is recorded as skipped in the pre-restore backup, not fatal.
        pre = next(b for b in self.store.backups() if b["path"] == result["pre_restore_backup"])
        self.assertEqual(pre["skipped_weights"], 1)

        restored = self.store.get(arch)
        self.assertEqual(restored["name"], "Alpha")
        self.assertEqual(restored["head_revision"], 2)
        self.assertEqual(restored["spec"], spec("alpha2"))
        self.assertEqual(self.store.get(other)["name"], "Other")
        self.assertEqual(Path(self.store.weight_path(arch, 1)).read_bytes(), original_bytes)

    def test_malicious_zip_members_are_rejected(self):
        arch = self.store.create("Keep", spec("keep"))
        for member in ("../escape-archdb.txt", "weights/../../escape-archdb.txt"):
            evil = self.root / "evil.zip"
            payload = b"owned"
            manifest = {"format": "groundwork-arch-backup/1", "created_at": "x", "label": "evil",
                        "files": {member: hashlib.sha256(payload).hexdigest(), "architectures.sqlite": "0" * 64}}
            with zipfile.ZipFile(evil, "w") as archive:
                archive.writestr(member, payload)
                archive.writestr("manifest.json", json.dumps(manifest))
            with self.assertRaises(ValueError):
                self.store.restore(evil)
            self.assertFalse((self.root / "escape-archdb.txt").exists())
            self.assertFalse((self.root.parent / "escape-archdb.txt").exists())
        self.assertEqual(self.store.get(arch)["name"], "Keep")
        self.assertEqual(self.store.backups(), [])

    def test_export_import_roundtrip(self):
        arch = self.store.create("Export me", spec("x"))
        self.store.commit(arch, spec("y"))
        path = self.root / "exports" / "net.json"
        self.store.export_spec(arch, path)
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["format"], "groundwork-arch-export/1")
        copied = self.store.import_spec(path, name="Copy")
        self.assertEqual(self.store.get(copied)["spec"], spec("y"))
        self.assertEqual(self.store.get(copied)["name"], "Copy")
        defaulted = self.store.import_spec(path)
        self.assertEqual(self.store.get(defaulted)["name"], "Export me")

    def test_concurrent_commits_get_unique_sequential_numbers(self):
        arch = self.store.create("Race", spec("base"))
        stores = [ArchitectureStore(self.root) for _ in range(4)]
        numbers, errors = [], []
        guard = threading.Lock()

        def worker(index, store):
            try:
                for step in range(5):
                    number = store.commit(arch, spec(f"w{index}-{step}"))
                    with guard:
                        numbers.append(number)
            except Exception as error:
                with guard:
                    errors.append(error)

        threads = [threading.Thread(target=worker, args=(i, s)) for i, s in enumerate(stores)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])
        self.assertEqual(sorted(numbers), list(range(2, 22)))
        self.assertEqual(self.store.get(arch)["head_revision"], 21)


if __name__ == "__main__":
    unittest.main()
