"""Summaries are kept per dataset beside the dataset's rev: a change reads only the datasets it touched. Window-free."""
import json
import random
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest import mock

import studio_catalog
import studio_data_kinds as dk
from studio_core import Workspace
from studio_data_cache import SummaryCache
from studio_ia.services import SUMMARY_FORMAT, WorkspaceServices
from studio_rules import generate

STORY = "the old lighthouse keeper climbed the long spiral stairs every evening to light the great lamp above the grey sea"


class PerDataset(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ws = Workspace(self.root)
        self.svc = WorkspaceServices(self.ws)
        self.addCleanup(self.svc.close)
        self.a = self.ws.create_dataset("Stories")
        self.b = self.ws.create_dataset("Talk", storage="sqlite")
        self.c = self.ws.create_dataset("Rules")
        self.ws.add(self.a, [STORY, STORY + " today", "a different text about cooking pasta with garlic and olive oil slowly"])
        self.ws.add(self.b, [{"user": f"q{i}", "assistant": f"a{i}", "quality": "Good"} for i in range(6)] + [STORY + " again", {"user": STORY, "assistant": "yes"}])
        dk.add_problems(self.ws, self.c, generate("copy", random.Random(1), 3))
        self.saved = self.root / "data" / "summaries.sqlite"

    def scanned(self):
        """The dataset_ids of every dk.scan call made while `work` runs."""
        return mock.patch.object(dk, "scan", wraps=dk.scan)

    def entries(self):
        return SummaryCache(self.saved, SUMMARY_FORMAT).load()[0]

    def stored_rows(self):
        return self.sql("SELECT dataset_id, kind, COUNT(*) FROM rows GROUP BY dataset_id, kind")

    def sql(self, statement, *values):
        with closing(sqlite3.connect(self.saved)) as connection, connection:
            return connection.execute(statement, values).fetchall()

    def everything(self, svc=None):
        svc = svc or self.svc
        return {k: svc.data(k) for k in dk.KINDS}, svc.post_training(None)["example"]

    def test_editing_one_dataset_rescans_only_that_dataset(self):
        self.everything()
        before = self.entries()
        with self.scanned() as scan:
            self.ws.add(self.a, ["one more story about a fox"])
            self.assertEqual(self.svc.data("text").count, 5)
        self.assertEqual([call.args[2] if len(call.args) > 2 else call.kwargs["dataset_ids"] for call in scan.call_args_list], [[self.a]])
        after = self.entries()
        self.assertEqual((after[self.b], after[self.c]), (before[self.b], before[self.c]))  # their cached entries were not touched
        self.assertNotEqual(after[self.a]["rev"], before[self.a]["rev"])

    def test_every_write_path_moves_the_rev_of_its_dataset_only(self):
        revs = lambda: {d["id"]: d.get("rev", 0) for d in self.ws.state["datasets"]}  # noqa: E731
        for name, write, touched in (
            ("add to a json set", lambda: self.ws.add(self.a, ["a fresh story about a heron"]), {self.a}),
            ("add to a disk set", lambda: self.ws.add(self.b, [{"user": "new q", "assistant": "new a"}]), {self.b}),
            ("edit in a disk set", lambda: self.ws.update_example({**self.ws.examples(self.b)[0], "quality": "Review"}), {self.b}),
            ("edit in a json set", lambda: self.ws.update_example({**self.ws.examples(self.a)[0], "quality": "Good"}), {self.a}),
            ("remove from a disk set", lambda: self.ws.remove_examples([self.ws.examples(self.b)[1]["id"]]), {self.b}),
            ("remove from a json set", lambda: self.ws.remove_examples([self.ws.examples(self.a)[1]["id"]]), {self.a}),
            ("problems into a json set", lambda: dk.add_problems(self.ws, self.c, generate("sort", random.Random(3), 2)), {self.c}),
            ("rename", lambda: self.ws.set_dataset_metadata(self.c, name="Renamed"), {self.c}),
        ):
            before = revs()
            write()
            after = revs()
            self.assertEqual({k for k in after if after[k] != before[k]}, touched, name)
        before = revs()
        self.ws.add(self.a, [STORY])  # a duplicate: nothing changed, so no rev moved
        self.assertEqual(revs(), before)

    def test_revs_are_saved_and_revision_never_goes_down(self):
        seen = [self.ws.revision]
        self.ws.add(self.a, ["another story about the sea"])
        seen.append(self.ws.revision)
        self.ws.remove_dataset(self.c)
        seen.append(self.ws.revision)
        self.ws.create_dataset("Again")
        seen.append(self.ws.revision)
        self.assertEqual(seen, sorted(set(seen)))
        reloaded = Workspace(self.root)
        self.assertEqual(reloaded.revision, self.ws.revision)
        self.assertEqual({d["id"]: d["rev"] for d in reloaded.state["datasets"]}, {d["id"]: d["rev"] for d in self.ws.state["datasets"]})

    def test_cached_summary_equals_a_direct_read_before_and_after_changes(self):
        for step in range(3):
            direct = {k: dk.summarize(self.ws, k, limit=None) for k in dk.KINDS}
            cached, _ = self.everything()
            self.assertEqual(cached, direct, step)
            if step == 0:
                self.assertGreater(direct["text"].near_duplicates, 0)
                self.ws.add(self.b, [{"user": STORY + " and then went home", "assistant": "ok"}])
            elif step == 1:
                self.ws.remove_dataset(self.a)

    def test_a_launch_reads_nothing_and_a_change_made_elsewhere_reads_one_dataset(self):
        first = self.everything()
        with self.scanned() as scan:
            again = WorkspaceServices(Workspace(self.root))
            self.addCleanup(again.close)
            self.assertEqual(self.everything(again), first)
            self.assertEqual(scan.call_count, 0)
            Workspace(self.root).add(self.a, ["written by another window"])
            later = WorkspaceServices(Workspace(self.root))
            self.addCleanup(later.close)
            self.assertEqual(later.data("text").count, 5)
            self.assertEqual([call.args[2] for call in scan.call_args_list], [[self.a]])

    def test_files_changed_without_a_rev_means_rescan_everything(self):
        self.everything()
        with self.scanned() as scan:
            with self.ws._connect() as connection:  # what an older app did: write rows and not the rev
                connection.execute("DELETE FROM records WHERE dataset_id=? AND rowid=(SELECT MIN(rowid) FROM records WHERE dataset_id=?)", (self.b, self.b))
            for d in self.ws.state["datasets"]:
                d["rev"] = self.entries()[d["id"]]["rev"]
            self.ws.save()
            again = WorkspaceServices(Workspace(self.root))
            self.addCleanup(again.close)
            self.assertEqual(again.data("conversations").count, 6)  # seven chats, one deleted behind the cache's back
            self.assertEqual(scan.call_count, 3)

    def test_older_summary_formats_and_garbage_are_rebuilt_once(self):
        expected = self.everything()
        legacy = self.root / "data" / "summaries.json"

        def with_format(number):
            def write():
                self.sql("UPDATE meta SET value=? WHERE key='format'", str(number))
            return write

        def with_entry(entry):
            def write():
                self.sql("UPDATE datasets SET entry=? WHERE id=?", json.dumps(entry), self.a)
            return write
        for name, damage in (("older format", with_format(SUMMARY_FORMAT - 1)), ("not a database", lambda: self.saved.write_bytes(b"not a database" * 100)),
                             ("empty file", lambda: self.saved.write_bytes(b"")), ("an entry of another shape", with_entry({"rev": self.ws._dataset(self.a)["rev"]}))):
            damage()
            legacy.write_text("{}", encoding="utf-8")
            with self.scanned() as scan:
                svc = WorkspaceServices(Workspace(self.root))
                self.addCleanup(svc.close)
                self.assertEqual(self.everything(svc), expected, name)
                self.assertEqual(scan.call_count, 3, name)
                self.assertEqual(SummaryCache(self.saved, SUMMARY_FORMAT).load()[0].keys(), {self.a, self.b, self.c}, name)
                self.assertFalse(legacy.exists(), name)  # the one large file an older app wrote is not kept
                self.assertEqual(WorkspaceServices(Workspace(self.root)).data("text").count, 4)
                self.assertEqual(scan.call_count, 3)  # the rebuilt file is read as it is

    def test_a_dataset_that_shrinks_gives_its_budget_to_the_next(self):
        ws = Workspace(self.root / "small")
        cache = SummaryCache(self.root / "small" / "data" / "summaries.sqlite", SUMMARY_FORMAT)
        first, second = ws.create_dataset("first"), ws.create_dataset("second")
        ws.add(first, [f"first story number {i} " + "alpha beta gamma delta " * 3 for i in range(5)])
        ws.add(second, [f"second story number {i} " + "kappa omicron sigma tau " * 3 for i in range(5)])

        def update(scan_limit, cached):
            entries, reread = dk.refresh(ws, cached, scan_limit=scan_limit)
            view = cache.save(ws.state["datasets"], entries, [], scan_limit)
            return entries, reread, view

        entries, reread, view = update(6, {})
        self.assertEqual((reread, [entries[d]["kept"]["text"] for d in (first, second)]), ([first, second], [5, 1]))  # the second keeps what the budget takes
        self.assertEqual(dk.assemble(ws.state["datasets"], entries, view, 6)[0]["text"].scanned, 6)
        ws.remove_examples([e["id"] for e in ws.examples(first)[:3]])
        entries, reread, view = update(6, entries)
        self.assertEqual(reread, [first, second])  # the second one kept one row and is now asked for four
        self.assertEqual([entries[d]["kept"]["text"] for d in (first, second)], [2, 4])
        self.assertEqual(dk.assemble(ws.state["datasets"], entries, view, 6)[0]["text"].scanned, 6)
        self.assertEqual(dk.summarize(ws, "text", scan_limit=6).rows, dk.assemble(ws.state["datasets"], entries, view, 6)[0]["text"].rows)
        self.assertEqual(update(6, entries)[1], [])  # nothing moved: nothing is read

    def test_the_cache_keeps_the_rows_the_budget_takes_and_a_search_reaches_all_of_them(self):
        ws = Workspace(self.root / "wide")
        cache = SummaryCache(self.root / "wide" / "data" / "summaries.sqlite", SUMMARY_FORMAT)
        first, second = ws.create_dataset("first", storage="sqlite"), ws.create_dataset("second", storage="sqlite")
        ws.add(first, [{"user": f"question {i} about topic{i}", "assistant": f"answer {i} " + "word " * 9 + f"unique{i}"} for i in range(30)])
        ws.add(second, [{"user": f"other {i} about thing{i}", "assistant": f"reply {i} " + "term " * 9 + f"single{i}"} for i in range(30)])
        entries, _ = dk.refresh(ws, {}, scan_limit=40)
        view = cache.save(ws.state["datasets"], entries, [], 40)
        self.assertEqual(entries[second]["kept"]["conversations"], 10)  # 30 + 10 is the budget: the other 20 are not kept
        self.saved = cache.path
        self.assertEqual(self.sql("SELECT dataset_id, COUNT(*) FROM rows GROUP BY dataset_id ORDER BY dataset_id"), sorted([(first, 30), (second, 10)]))
        self.assertEqual(len(view["rows"]["conversations"]), 40)
        names = {first: "first", second: "second"}
        found = cache.search("conversations", " THING9 ", view["takes"]["conversations"], names)
        self.assertEqual([(r["a"], r["dataset"]) for r in found], [("other 9 about thing9", "second")])
        self.assertEqual(cache.search("conversations", "thing19", view["takes"]["conversations"], names), [])  # past the budget
        self.assertEqual(len(cache.search("conversations", "word", view["takes"]["conversations"], names, limit=7)), 7)
        direct = dk.summarize(ws, "conversations", "about", limit=None, scan_limit=40)
        self.assertEqual(cache.search("conversations", "about", view["takes"]["conversations"], names, limit=1000), direct.rows)

    def test_disk_datasets_are_counted_by_the_kind_column_not_read_whole(self):
        with mock.patch.object(Workspace, "iter_examples", side_effect=AssertionError("read the whole dataset")):
            out, pairs = dk.scan(self.ws, dataset_ids=[self.b])
        self.assertEqual((out["conversations"].count, out["text"].count, out["conversations"].characters),
                         (7, 1, sum(len(f"q{i}") + len(f"a{i}") for i in range(6)) + len(STORY) + len("yes")))
        self.assertEqual(len(pairs["conversations"]), 7)
        self.assertEqual(out["conversations"].held_back, sum(dk.held_back(e) for e in self.ws.examples(self.b) if dk.classify(e) == "conversations"))
        with mock.patch.object(Workspace, "iter_kind", side_effect=AssertionError("read rows nobody asked for")):
            self.assertEqual(dk.scan(self.ws, scan_limit=0, dataset_ids=[self.b])[0]["conversations"].count, 7)

    def test_signatures_round_trip(self):
        found = dk.signature(STORY)
        self.assertEqual(dk.unpack_signature(dk.pack_signature(*found)), found)
        self.assertEqual(dk.match_signatures([("a", *found), ("b", *dk.signature(STORY + " today"))]), {"b": "a"})


class CatalogSwap(unittest.TestCase):
    def test_registering_a_download_again_moves_the_rev(self):
        def fetcher(response):
            def fake(url, path, stop=None, emit=None, maximum=None, **kw):
                payload = (json.dumps({"instruction": "Say hello", "response": response, "context": "", "category": "g"}) + "\n").encode() if Path(path).suffix == ".jsonl" else b"x"
                if Path(path).name == "SOURCE_README.md":
                    raise OSError("card unavailable")
                Path(path).write_bytes(payload)
                return {"url": url, "filename": Path(path).name, "bytes": len(payload), "sha256": studio_catalog.file_sha256(path)}
            return fake
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch("studio_catalog._fetch", side_effect=fetcher("Hello")):
                first = studio_catalog.download_dataset("dolly15k", root=directory)
            ws = Workspace(directory)
            svc = WorkspaceServices(ws)
            self.addCleanup(svc.close)
            other = ws.create_dataset("Other")
            ws.add(other, [STORY])
            dataset_id = studio_catalog.register_download(ws, first)
            self.assertEqual(svc.data("conversations").rows[0]["b"], "Hello")
            seen = ws._dataset(dataset_id)["rev"]
            for name in ("records.jsonl", "manifest.json", "source.json", "databricks-dolly-15k.jsonl"):
                (Path(directory) / "data" / "downloads" / "dolly15k" / name).unlink(missing_ok=True)
            with mock.patch("studio_catalog._fetch", side_effect=fetcher("Hi there")):
                second = studio_catalog.download_dataset("dolly15k", root=directory)
            with mock.patch.object(dk, "scan", wraps=dk.scan) as scan:
                studio_catalog.register_download(ws, second)
                self.assertGreater(ws._dataset(dataset_id)["rev"], seen)
                self.assertEqual(svc.data("conversations").rows[0]["b"], "Hi there")
                self.assertEqual([call.args[2] for call in scan.call_args_list], [[dataset_id]])
            # a removed and re-registered dataset never reuses a rev an old cache entry could still hold
            high = ws._dataset(dataset_id)["rev"]
            ws.remove_dataset(dataset_id)
            self.assertGreater(ws.next_rev(), high)

    def test_a_reader_of_another_dataset_is_not_blocked_while_a_download_is_swapped_in(self):
        """The comment in register_download promises WAL readers keep going during the transaction; this pins it."""
        def fake(url, path, stop=None, emit=None, maximum=None, **kw):
            if Path(path).name == "SOURCE_README.md":
                raise OSError("card unavailable")
            payload = (json.dumps({"instruction": "Say hello", "response": "Hello", "context": "", "category": "g"}) + "\n").encode() if Path(path).suffix == ".jsonl" else b"x"
            Path(path).write_bytes(payload)
            return {"url": url, "filename": Path(path).name, "bytes": len(payload), "sha256": studio_catalog.file_sha256(path)}
        with tempfile.TemporaryDirectory() as directory, mock.patch("studio_catalog._fetch", side_effect=fake):
            result = studio_catalog.download_dataset("dolly15k", root=directory)
            ws = Workspace(directory)
            other = ws.create_dataset("Other", storage="sqlite")
            ws.add(other, [STORY, "a second document about the sea"])
            read, insert = [], Workspace._insert_rows

            def insert_then_read(connection, rows, ignore=True):
                stored = insert(connection, rows, ignore)  # the transaction is open and holds the write lock here
                read.append(len(list(ws.iter_examples(other))))
                return stored
            with mock.patch.object(Workspace, "_insert_rows", staticmethod(insert_then_read)):
                studio_catalog.register_download(ws, result)
            self.assertEqual(read[0], 2)


if __name__ == "__main__":
    unittest.main()
