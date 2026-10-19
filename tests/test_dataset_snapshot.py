"""Dataset snapshots (studio_dataset_snapshot.py): one stored file shared by every attempt of the same data. Window-free."""
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import studio_core
import studio_dataset_snapshot as snapshots
import studio_model

TINY = dict(device="cpu", width=16, layers=1, heads=2, context=32, expansion=2, dropout=0, batch_size=2,
            steps=2, learning_rate=.004, warmup=0, validation=0, eval_every=2, objective="assistant")
EXAMPLES = [studio_core.normalize({"user": f"say {w}", "assistant": w}) for w in ("yes", "no", "cat", "dog")] * 2


def same_file(a, b):
    return os.path.samefile(a, b)


class Store(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def test_the_file_is_named_by_the_hash_of_its_bytes_and_split_on_newlines_only(self):
        rows = [{"user": "a b", "assistant": "ok"}, {"text": "two"}]
        path, created = snapshots.store(rows, self.root / "store")
        self.assertTrue(created)
        data = path.read_bytes()
        self.assertEqual(path.name, hashlib.sha256(data).hexdigest() + ".jsonl")
        self.assertEqual(snapshots.hash_of(path), path.stem)
        self.assertEqual(data, ("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n").encode())  # the bytes it always had
        self.assertEqual(list(snapshots.read(path)), rows)

    def test_the_same_data_is_stored_once_and_other_data_beside_it(self):
        first, created = snapshots.store(({"text": str(i)} for i in range(5)), self.root / "store")
        again, created_again = snapshots.store(({"text": str(i)} for i in range(5)), self.root / "store")
        other, _ = snapshots.store([{"text": "other"}], self.root / "store")
        self.assertEqual((first, created, created_again), (again, True, False))
        self.assertNotEqual(first, other)
        self.assertEqual(sorted(p.name for p in (self.root / "store").iterdir()), sorted([first.name, other.name]))  # no temporary file is left

    def test_a_snapshot_is_linked_not_copied(self):
        (self.root / "a").mkdir()
        (self.root / "b").mkdir()
        path, _, linked_a = snapshots.adopt([{"text": "x"}], self.root / "store", self.root / "a" / "dataset.jsonl")
        _, created, linked_b = snapshots.adopt([{"text": "x"}], self.root / "store", self.root / "b" / "dataset.jsonl")
        self.assertEqual((linked_a, created, linked_b), (True, False, True))
        self.assertTrue(same_file(path, self.root / "a" / "dataset.jsonl") and same_file(path, self.root / "b" / "dataset.jsonl"))
        self.assertEqual(path.stat().st_nlink, 3)
        self.assertEqual(snapshots.listing(self.root / "store"), [(path, path.stat().st_size, 3)])

    def test_a_filesystem_that_cannot_link_gets_a_copy(self):
        (self.root / "a").mkdir()
        with patch("studio_dataset_snapshot.os.link", side_effect=OSError("cross-device")):
            path, _, linked = snapshots.adopt([{"text": "x"}], self.root / "store", self.root / "a" / "dataset.jsonl")
        self.assertFalse(linked)
        self.assertFalse(same_file(path, self.root / "a" / "dataset.jsonl"))
        self.assertEqual((self.root / "a" / "dataset.jsonl").read_bytes(), path.read_bytes())
        self.assertEqual(sorted(p.name for p in (self.root / "a").iterdir()), ["dataset.jsonl"])

    def test_without_a_store_the_file_is_written_where_it_belongs(self):
        (self.root / "run").mkdir()
        _, created, linked = snapshots.adopt([{"text": "x"}], None, self.root / "run" / "dataset.jsonl")
        self.assertEqual((created, linked), (True, False))
        self.assertEqual(sorted(p.name for p in (self.root / "run").iterdir()), ["dataset.jsonl"])

    def test_the_store_is_made_only_beside_a_runs_folder(self):
        self.assertEqual(snapshots.folder_for(self.root / "runs" / "queue_1"), self.root / "runs" / ".snapshots")
        self.assertIsNone(snapshots.folder_for(self.root / "elsewhere" / "queue_1"))

    def test_examples_are_streamed_one_at_a_time(self):
        seen = []

        def rows():
            for i in range(3):
                seen.append(i)
                yield {"text": str(i)}
        path, _ = snapshots.store(rows(), self.root / "store")
        self.assertEqual(seen, [0, 1, 2])
        self.assertEqual(len(list(snapshots.read(path))), 3)


class Training(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.runs = Path(self._tmp.name) / "runs"

    def test_two_attempts_of_the_same_data_share_one_file_on_disk(self):
        events = []
        for name in ("queue_1", "queue_1-before-restart", "queue_2"):
            studio_model.train(TINY, [dict(e) for e in EXAMPLES], self.runs / name, events.append)
        links = [self.runs / name / "dataset.jsonl" for name in ("queue_1", "queue_1-before-restart", "queue_2")]
        self.assertTrue(all(same_file(links[0], other) for other in links[1:]))
        (snapshot, size, count), = snapshots.listing(self.runs / ".snapshots")
        self.assertEqual((count, size), (4, links[0].stat().st_size))  # the store's name and the three run folders
        self.assertEqual(len(list(snapshots.read(links[0]))), 4)  # the four different examples, not the eight rows
        said = [e["text"] for e in events if e["type"] == "info" and e["text"].startswith("Dataset snapshot")]
        self.assertTrue(said[0].endswith("written") and all(s.endswith("already stored, linked") for s in said[1:]), said)

    def test_different_data_gets_its_own_snapshot(self):
        studio_model.train(TINY, [dict(e) for e in EXAMPLES], self.runs / "a")
        studio_model.train(TINY, [studio_core.normalize({"user": "say moo", "assistant": "moo"})] * 4, self.runs / "b")
        self.assertEqual(len(snapshots.listing(self.runs / ".snapshots")), 2)
        self.assertFalse(same_file(self.runs / "a" / "dataset.jsonl", self.runs / "b" / "dataset.jsonl"))

    def test_the_callers_rows_are_not_changed_by_merging_duplicates(self):
        first = studio_core.normalize({"user": "say yes", "assistant": "yes"})
        copy = dict(first, id="other_id", groupId="g2")
        before = json.dumps(first, sort_keys=True)
        studio_model.train(TINY, [first, copy, *[dict(e) for e in EXAMPLES]], self.runs / "a")
        self.assertEqual(json.dumps(first, sort_keys=True), before)  # the merge happened on a private copy
        merged = [r for r in snapshots.read(self.runs / "a" / "dataset.jsonl") if r["id"] == first["id"]]
        self.assertIn("other_id", merged[0]["provenance"]["source_ids"])

    def test_a_folder_outside_runs_keeps_a_plain_file(self):
        out = Path(self._tmp.name) / "plain"
        studio_model.train(TINY, [dict(e) for e in EXAMPLES], out)
        self.assertEqual(sorted(p.name for p in out.iterdir() if p.name == "dataset.jsonl"), ["dataset.jsonl"])
        self.assertEqual(list(Path(self._tmp.name).glob(".snapshots")), [])


if __name__ == "__main__":
    unittest.main()
