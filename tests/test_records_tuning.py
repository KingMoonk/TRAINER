"""records.sqlite tuning: WAL, one connection per call, the stats index, and bulk writes in one transaction. Window-free."""
import random
import sqlite3
import tempfile
import threading
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import studio_cleanup
import studio_core
import studio_data_kinds as dk
from studio_core import Workspace
from studio_rules import generate

STATS_SQL = ("SELECT kind,COUNT(*),COALESCE(SUM(characters),0),COALESCE(SUM(holdout),0) FROM records WHERE dataset_id=? GROUP BY kind")
REVIEW_SQL = "SELECT COUNT(*) FROM records WHERE dataset_id=? AND quality='Review'"


def plan(db_path, sql):
    with closing(sqlite3.connect(db_path)) as connection:
        return " | ".join(row[3] for row in connection.execute("EXPLAIN QUERY PLAN " + sql, ("d",)))


class Tuning(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.ws = Workspace(self.root)
        self.ds = self.ws.create_dataset("disk", storage="sqlite")
        self.ws.add(self.ds, [{"text": f"story {i}"} for i in range(20)] + [{"user": f"q{i}", "assistant": f"a{i}"} for i in range(20)])

    def pragma(self, name):
        with self.ws._connect() as connection:
            return connection.execute(f"PRAGMA {name}").fetchone()[0]

    def test_the_file_is_in_wal_mode_and_every_connection_is_tuned(self):
        with closing(sqlite3.connect(self.ws.db_path)) as connection:
            self.assertEqual(connection.execute("PRAGMA journal_mode").fetchone()[0], "wal")  # read from the file, not set by this connection
        self.assertEqual(self.pragma("synchronous"), 1)  # NORMAL
        self.assertGreaterEqual(self.pragma("busy_timeout"), 30000)

    def test_stats_and_review_count_read_the_index_alone(self):
        for sql in (STATS_SQL, REVIEW_SQL):
            self.assertIn("COVERING INDEX records_stats", plan(self.ws.db_path, sql))

    def test_a_schema_2_database_gets_the_index_without_touching_its_rows(self):
        with closing(sqlite3.connect(self.ws.db_path)) as connection:
            connection.execute("DROP INDEX records_stats")
            connection.execute("PRAGMA user_version=2")
            connection.commit()
            before = connection.execute("SELECT * FROM records").fetchall()
        self.assertTrue(Workspace(self.root).schema_pending())
        later = Workspace(self.root)
        later.ensure_schema()
        self.assertFalse(later.schema_pending())
        self.assertIn("COVERING INDEX records_stats", plan(self.ws.db_path, STATS_SQL))
        with closing(sqlite3.connect(self.ws.db_path)) as connection:
            self.assertEqual(connection.execute("SELECT * FROM records").fetchall(), before)
        self.assertEqual(later.kind_stats(self.ds)["text"][0], 20)

    def test_readers_on_two_threads_and_a_writer_do_not_collide(self):
        errors, stop = [], threading.Event()

        def read():
            try:
                while not stop.is_set():
                    self.ws.count(self.ds)
                    self.ws.kind_stats(self.ds)
                    list(self.ws.iter_examples(self.ds))
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        readers = [threading.Thread(target=read) for _ in range(2)]
        for thread in readers:
            thread.start()
        try:
            for i in range(30):
                self.ws.add(self.ds, [{"text": f"written while reading {i}"}])
        finally:
            stop.set()
            for thread in readers:
                thread.join()
        self.assertEqual(errors, [])
        self.assertEqual(self.ws.count(self.ds), 70)

    def test_each_call_opens_its_own_connection(self):
        seen = []
        real = sqlite3.connect

        def connect(*args, **kwargs):
            seen.append(real(*args, **kwargs))
            return seen[-1]
        with patch.object(studio_core.sqlite3, "connect", connect):
            self.ws.count(self.ds)
            self.ws.count(self.ds)
        self.assertEqual(len(seen), 2)
        with self.assertRaises(sqlite3.ProgrammingError):
            seen[0].execute("SELECT 1")  # closed when the call ended: nothing is kept for another thread to pick up

    def test_a_bulk_add_is_one_transaction_in_batches(self):
        def rows():
            for i in range(10):
                yield {"text": f"bulk {i}"}
            yield {"text": "bulk 3"}  # a duplicate
            yield {"user": "", "assistant": ""}  # invalid
        with patch.object(studio_core, "INSERT_BATCH", 3):
            self.assertEqual(self.ws.add(self.ds, rows()), (10, 1, 1))
        self.assertEqual(self.ws.count(self.ds), 50)

        def failing():
            yield {"text": "doomed 1"}
            yield {"text": "doomed 2"}
            yield {"text": "doomed 3"}
            yield {"text": "doomed 4"}
            raise RuntimeError("the file broke")
        with patch.object(studio_core, "INSERT_BATCH", 2), self.assertRaises(RuntimeError):
            self.ws.add(self.ds, failing())
        self.assertEqual(self.ws.count(self.ds), 50)  # the batches already inserted were rolled back with the rest

    def test_mark_reviewed_saves_once(self):
        ids = [e["id"] for e in self.ws.examples(self.ds, limit=5)]
        small = self.ws.create_dataset("memory")
        self.ws.add(small, [{"text": "a note"}])
        ids.append(self.ws.examples(small)[0]["id"])
        with patch.object(Workspace, "save", autospec=True, side_effect=Workspace.save) as save:
            self.ws.mark_reviewed(ids)
        self.assertEqual(save.call_count, 1)
        self.assertEqual({self.ws.get_example(i)["quality"] for i in ids}, {"Good"})
        self.assertEqual(self.ws.review_count(self.ds), 35)
        self.assertEqual(self.ws.review_count(small), 0)

    def test_sample_can_keep_one_kind(self):
        talk = self.ws.sample(self.ds, limit=5, kind="conversations")
        self.assertEqual({dk.classify(e) for e in talk}, {"conversations"})
        self.assertEqual(len(talk), 5)
        self.assertEqual(self.ws.sample(self.ds, limit=100, kind="problems"), [])
        self.assertEqual(self.ws.sample(self.ds, limit=100, reviewed_only=True), [])  # nothing is Good yet


class AddProblems(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.ws = Workspace(self._tmp.name)

    def check(self, storage):
        ds = self.ws.create_dataset(f"problems {storage}", storage=storage)
        items = generate("copy", random.Random(1), 5)
        self.assertEqual(dk.add_problems(self.ws, ds, items), (5, 0))
        self.assertEqual(dk.add_problems(self.ws, ds, items), (0, 5))  # the same items again
        more = generate("reverse", random.Random(2), 3)
        fresh = len({item["prompt"] for item in more})
        self.assertEqual(dk.add_problems(self.ws, ds, more + more[:2]), (fresh, 5 - fresh))  # a repeat inside one call is a duplicate too
        self.assertEqual(self.ws.count(ds), 5 + fresh)
        self.assertEqual({dk.classify(e) for e in self.ws.iter_examples(ds)}, {"problems"})
        self.assertEqual({dk.rule_of(e) for e in self.ws.iter_examples(ds)}, {"copy", "reverse"})
        if storage == "sqlite":
            self.assertEqual(self.ws._dataset(ds)["recordCount"], 5 + fresh)

    def test_in_memory_dataset(self):
        self.check("json")

    def test_disk_dataset_relies_on_the_unique_key(self):
        self.check("sqlite")


class CleanupSidecars(unittest.TestCase):
    def test_deleting_the_data_takes_the_wal_files_with_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(tmp)
            ws.add(ws.create_dataset("disk", storage="sqlite"), ["one"])
            for suffix in ("-wal", "-shm"):  # present while the app has the store open
                ws.db_path.with_name(ws.db_path.name + suffix).write_bytes(b"x")
            names = {p.name for p in studio_cleanup.plan(tmp, data=True)}
            self.assertLessEqual({"records.sqlite", "records.sqlite-wal", "records.sqlite-shm"}, names)
            studio_cleanup.clean(ws, data=True)
            self.assertEqual(list(Path(tmp, "data").glob("records.sqlite*")), [])


if __name__ == "__main__":
    unittest.main()
