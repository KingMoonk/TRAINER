"""set_counts: the per-set counts the text sets list answers from stay exact as rows come and go, without counting again. Window-free."""
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from studio_core import Workspace
from studio_ia.services import WorkspaceServices


def grouped(ws, dataset_id):
    """What a full count of the rows says: the answer set_counts has to give without it."""
    with closing(sqlite3.connect(ws.db_path)) as connection:
        rows = connection.execute("SELECT kind,COUNT(*),COALESCE(SUM(characters),0),COALESCE(SUM(holdout),0) FROM records "
                                  "WHERE dataset_id=? GROUP BY kind", (dataset_id,)).fetchall()
    return {kind: tuple(rest) for kind, *rest in rows}


class SetCounts(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.ws = Workspace(self.root)
        self.a = self.ws.create_dataset("a", storage="sqlite")
        self.b = self.ws.create_dataset("b", storage="sqlite")

    def check(self):
        for dataset in (self.a, self.b):
            self.assertEqual(self.ws.kind_stats(dataset), grouped(self.ws, dataset) or {})

    def test_counts_follow_adds_removals_edits_and_a_removed_set(self):
        self.ws.add(self.a, [{"text": f"story {i}"} for i in range(30)] + [{"user": f"q{i}", "assistant": f"a{i}"} for i in range(10)])
        self.ws.add(self.b, [{"text": "other story"}])
        self.check()
        self.assertEqual(self.ws.kind_stats(self.a)["text"][0], 30)
        ids = [e["id"] for e in self.ws.examples(self.a, limit=5)]
        self.ws.remove_examples(ids)
        self.check()
        self.assertEqual(self.ws.kind_stats(self.a)["text"][0], 25)
        changed = dict(self.ws.examples(self.a, limit=1)[0], text="a much longer story than it was", user="a much longer story than it was")
        self.ws.update_example(changed)
        self.check()
        self.ws.add(self.a, [{"text": "story 10"}])  # a duplicate is ignored and must not be counted
        self.check()
        self.assertEqual(self.ws.kind_stats(self.a)["text"][0], 25)
        self.ws.remove_dataset(self.b)
        self.assertEqual(self.ws.kind_stats(self.b), {})

    def test_asking_never_groups_the_rows(self):
        self.ws.add(self.a, [{"text": f"story {i}"} for i in range(50)])
        seen = []
        real = self.ws._connect

        def traced(*args, **kwargs):
            manager = real(*args, **kwargs)

            class Traced:
                def __enter__(self_inner):
                    connection = manager.__enter__()
                    connection.set_trace_callback(seen.append)
                    return connection

                def __exit__(self_inner, *exc):
                    return manager.__exit__(*exc)
            return Traced()
        self.ws._connect = traced
        self.assertEqual(self.ws.kind_stats(self.a)["text"][0], 50)
        self.assertTrue(seen)
        self.assertFalse([sql for sql in seen if "FROM records" in sql or "COUNT(" in sql.upper()], seen)

    def test_a_schema_3_database_is_counted_once_and_then_kept(self):
        self.ws.add(self.a, [{"text": f"story {i}"} for i in range(12)])
        with closing(sqlite3.connect(self.ws.db_path)) as connection:
            connection.executescript("DROP TRIGGER records_counts_insert; DROP TRIGGER records_counts_delete; "
                                     "DROP TRIGGER records_counts_update; DROP TABLE set_counts; PRAGMA user_version=3;")
        later = Workspace(self.root)
        self.assertTrue(later.schema_pending())
        later.ensure_schema()
        self.assertEqual(later.kind_stats(self.a)["text"][0], 12)
        later.add(self.a, [{"text": "one more"}])
        self.assertEqual(later.kind_stats(self.a)["text"][0], 13)

    def test_a_row_without_a_kind_sends_the_caller_to_the_rows(self):
        self.ws.add(self.a, [{"text": "story"}])
        with closing(sqlite3.connect(self.ws.db_path)) as connection:
            connection.execute("UPDATE records SET kind=NULL")
            connection.commit()
        self.assertIsNone(self.ws.kind_stats(self.a))


class TextSets(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.ws = Workspace(Path(self._tmp.name))
        self.services = WorkspaceServices.reader(self.ws.root)
        self.services.workspace = self.ws

    def test_the_list_matches_the_summary_and_follows_the_rows(self):
        disk = self.ws.create_dataset("Disk", storage="sqlite")
        memory = self.ws.create_dataset("Memory")
        talk = self.ws.create_dataset("Talk", storage="sqlite")
        self.ws.add(disk, [{"text": f"story {i}"} for i in range(7)])
        self.ws.add(memory, [{"text": "a"}, {"text": "b"}])
        self.ws.add(talk, [{"user": "q", "assistant": "a"}])
        self.assertEqual(self.services.text_sets(), [{"id": disk, "name": "Disk", "count": 7}, {"id": memory, "name": "Memory", "count": 2}])
        self.assertEqual(self.services.text_sets(), [{"id": s["id"], "name": s["name"], "count": s["count"]} for s in self.services.data("text").sets])
        self.ws.add(disk, [{"text": "eighth"}])
        self.ws.remove_examples([e["id"] for e in self.ws.examples(disk, limit=3)])
        self.assertEqual(self.services.text_sets()[0]["count"], 5)

    def test_the_summary_is_never_read(self):
        self.ws.add(self.ws.create_dataset("Disk", storage="sqlite"), [{"text": "x"}])
        self.services._summaries = lambda: self.fail("the summary was read")
        self.assertEqual(self.services.text_sets()[0]["count"], 1)


if __name__ == "__main__":
    unittest.main()
