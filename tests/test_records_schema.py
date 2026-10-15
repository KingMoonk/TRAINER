"""Records schema 2: kind and holdout columns, the migration from user_version 0, and post_rows as one indexed query. Window-free."""
import json
import random
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import studio_core
import studio_data_kinds as dk
from studio_core import Workspace, fingerprint, normalize
from studio_ia.services import WorkspaceServices
from studio_rules import generate

OLD_TABLE = """CREATE TABLE records (
    id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL, fingerprint TEXT NOT NULL,
    payload TEXT NOT NULL, user TEXT NOT NULL, assistant TEXT NOT NULL,
    intent TEXT NOT NULL, quality TEXT NOT NULL, characters INTEGER NOT NULL,
    UNIQUE(dataset_id, fingerprint))"""


def mixed_rows(dataset_id, n=120):
    """normalized examples of all three kinds, as an older app would have stored them (user/assistant inside the payload)."""
    rows = [normalize({"text": f"story number {i} about a cat"}, dataset_id) for i in range(n)]
    rows += [normalize({"user": f"question {i}", "assistant": f"answer {i}"}, dataset_id) for i in range(n)]
    for item in generate("copy", random.Random(1), n):
        row = dk.make_problem(item)
        example = normalize(row, dataset_id, "synthetic")
        example["rule"], example["params"] = row["rule"], row["params"]
        rows.append(example)
    unique = {}
    for example in rows:
        unique.setdefault(fingerprint(example), example)
    return list(unique.values())


def write_v0_database(db_path, rows):
    """A records.sqlite exactly as the app wrote it before schema 2: nine columns, user_version 0, payload holding everything."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(db_path)) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute(OLD_TABLE)
        connection.execute("CREATE INDEX records_dataset ON records(dataset_id)")
        connection.executemany("INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?)", [
            (e["id"], e["datasetId"], fingerprint(e), json.dumps(e, ensure_ascii=False), e["user"], e["assistant"],
             e["intent"], e["quality"], len(e["user"]) + len(e["assistant"])) for e in rows])
        connection.commit()


def stored(db_path, sql="SELECT * FROM records"):
    with closing(sqlite3.connect(db_path)) as connection:
        return connection.execute(sql).fetchall()


class Migration(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.ws = Workspace(self.root)
        self.ds = self.ws.create_dataset("old disk data", storage="sqlite")
        self.rows = mixed_rows(self.ds)
        write_v0_database(self.ws.db_path, self.rows)
        self.assertEqual(stored(self.ws.db_path, "PRAGMA user_version"), [(0,)])

    def check_columns(self):
        by_id = {e["id"]: e for e in self.rows}
        got = stored(self.ws.db_path, "SELECT id, kind, holdout FROM records")
        self.assertEqual(len(got), len(self.rows))
        for example_id, kind, holdout in got:
            self.assertEqual((kind, holdout), (dk.classify(by_id[example_id]), int(dk.held_back(by_id[example_id]))))
        self.assertEqual({k for _, k, _ in got}, {"text", "conversations", "problems"})
        self.assertIn(1, {h for _, _, h in got})  # the fixture does hold some rows back

    def test_a_version_0_database_is_migrated_and_keeps_working(self):
        self.assertTrue(self.ws.schema_pending())
        self.assertEqual(self.ws.count(self.ds), len(self.rows))  # the first read migrates
        self.assertEqual(stored(self.ws.db_path, "PRAGMA user_version"), [(studio_core.SCHEMA_VERSION,)])
        self.assertFalse(self.ws.schema_pending())
        self.check_columns()
        indexes = {row[0] for row in stored(self.ws.db_path, "SELECT name FROM sqlite_master WHERE type='index'")}
        self.assertTrue({"records_pick", "records_dataset"} <= indexes)
        self.assertEqual(list(self.ws.iter_examples(self.ds)), self.rows)  # nothing about a row changes for its readers
        self.assertEqual(self.ws.get_example(self.rows[3]["id"]), self.rows[3])

    def test_migrating_again_changes_nothing(self):
        self.ws.ensure_schema()
        before = stored(self.ws.db_path)
        with patch.object(Workspace, "_migrate", side_effect=AssertionError("ran twice")):
            Workspace(self.root).ensure_schema()
        self.assertEqual(stored(self.ws.db_path), before)

    def test_an_interrupted_migration_resumes_where_it_stopped(self):
        seen = []

        def crash(text, fraction):
            seen.append(fraction)
            if len(seen) == 2:
                raise KeyboardInterrupt  # the app is closed part-way
        self.ws.progress = crash
        with patch.object(studio_core, "MIGRATE_BATCH", 50):
            with self.assertRaises(KeyboardInterrupt):
                self.ws.ensure_schema()
        self.assertEqual(stored(self.ws.db_path, "PRAGMA user_version"), [(0,)])  # not marked done
        filled = stored(self.ws.db_path, "SELECT COUNT(*) FROM records WHERE kind IS NOT NULL")[0][0]
        self.assertEqual(filled, 100)
        later = []
        reopened = Workspace(self.root)
        reopened.progress = lambda text, fraction: later.append(fraction)
        with patch.object(studio_core, "MIGRATE_BATCH", 50):
            reopened.ensure_schema()
        self.assertEqual(later[-1], 1.0)
        self.assertEqual(len(later), (len(self.rows) - 100 + 49) // 50)  # only the rest was read
        self.assertEqual(stored(self.ws.db_path, "PRAGMA user_version"), [(studio_core.SCHEMA_VERSION,)])
        self.check_columns()

    def test_a_database_with_the_columns_but_no_version_is_finished_not_broken(self):
        with closing(sqlite3.connect(self.ws.db_path)) as connection:  # an earlier attempt that added the columns and stopped
            connection.execute("ALTER TABLE records ADD COLUMN kind TEXT")
            connection.execute("ALTER TABLE records ADD COLUMN holdout INTEGER")
            connection.commit()
        self.ws.ensure_schema()
        self.check_columns()

    def test_a_new_workspace_starts_at_the_current_schema(self):
        fresh = Workspace(self.root / "fresh")
        ds = fresh.create_dataset("new", storage="sqlite")
        fresh.add(ds, [{"user": "q", "assistant": "a"}])
        self.assertEqual(stored(fresh.db_path, "PRAGMA user_version"), [(studio_core.SCHEMA_VERSION,)])
        self.assertEqual(stored(fresh.db_path, "SELECT kind, holdout FROM records")[0][0], "conversations")


class LeanRows(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.ws = Workspace(self._tmp.name)
        self.ds = self.ws.create_dataset("disk", storage="sqlite")

    def test_the_payload_leaves_out_what_the_columns_hold(self):
        self.ws.add(self.ds, [{"user": "what is two plus two", "assistant": "four"}, {"text": "once upon a time"}])
        for payload, user, assistant in stored(self.ws.db_path, "SELECT payload, user, assistant FROM records"):
            keys = json.loads(payload)
            self.assertNotIn("user", keys)
            self.assertNotIn("assistant", keys)
            self.assertTrue(user)
        read = {e.get("text") or e["user"]: e for e in self.ws.iter_examples(self.ds)}
        self.assertEqual((read["what is two plus two"]["assistant"], read["once upon a time"]["user"]), ("four", "once upon a time"))
        self.assertEqual(self.ws.examples(self.ds, query="two plus")[0]["assistant"], "four")  # search still works on the columns

    def test_editing_a_row_updates_its_kind_and_holdout(self):
        self.ws.add(self.ds, [{"user": "q", "assistant": "a"}])
        row = self.ws.examples(self.ds)[0]
        messages = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a different answer"}]
        edited = self.ws.update_example({**row, "messages": messages})
        kind, holdout = stored(self.ws.db_path, "SELECT kind, holdout FROM records")[0]
        self.assertEqual((kind, holdout), ("conversations", int(dk.held_back(edited))))
        self.assertEqual(self.ws.get_example(row["id"])["assistant"], "a different answer")

    def test_json_datasets_carry_the_stamps_and_exports_leave_them_out(self):
        json_ds = self.ws.create_dataset("memory")
        self.ws.add(json_ds, [{"user": "q", "assistant": "a"}])
        example = self.ws.examples(json_ds)[0]
        self.assertEqual((example["kind"], example["holdout"]), ("conversations", int(dk.held_back(example))))
        out = Path(self._tmp.name) / "out.jsonl"
        self.ws.export(json_ds, out)
        self.assertEqual(set(json.loads(out.read_text())), {"messages", "text", "intent", "quality", "synthetic", "source"})


def old_post_rows(ws, kind):
    """post_rows as it was before schema 2: every example of every dataset, classified and hashed in Python."""
    want = "problems" if kind.endswith("problems") else "conversations"
    held = kind == "heldout_problems"
    out = []
    for d in ws.state["datasets"]:
        for e in ws.iter_examples(d["id"]):
            if dk.classify(e) == want and dk.held_back(e) == held:
                out.append({"prompt": e.get("user", ""), "rule": dk.rule_of(e), "params": e.get("params") or (e.get("metadata") or {}).get("params", {})}
                           if want == "problems" else e)
    return out


class PostRows(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.ws = Workspace(self._tmp.name)
        self.svc = WorkspaceServices(self.ws)
        self.addCleanup(self.svc.close)
        self.small = self.ws.create_dataset("in memory")  # JSON
        self.big = self.ws.create_dataset("on disk", storage="sqlite")
        for ds in (self.small, self.big):
            self.ws.add(ds, [{"text": f"story {i} of {ds}"} for i in range(40)])
            self.ws.add(ds, [{"user": f"q{i} {ds}", "assistant": f"a{i}"} for i in range(150)])
            dk.add_problems(self.ws, ds, generate("copy", random.Random(2), 80) + generate("reverse", random.Random(3), 80))

    def test_matches_the_old_python_filter_on_json_and_sqlite_datasets(self):
        for kind in ("conversations", "problems", "heldout_problems"):
            expected = old_post_rows(self.ws, kind)
            self.assertTrue(expected, kind)
            self.assertEqual(self.svc.post_rows(kind), expected, kind)

    def test_held_back_rows_never_reach_training_and_are_all_in_the_held_set(self):
        held_ids = {e["id"] for d in (self.small, self.big) for e in self.ws.iter_examples(d) if dk.held_back(e)}
        self.assertTrue(held_ids)
        self.assertFalse({e["id"] for e in self.svc.post_rows("conversations")} & held_ids)
        held_prompts = {p["prompt"] for p in self.svc.post_rows("heldout_problems")}
        self.assertEqual(held_prompts, {e["user"] for d in (self.small, self.big) for e in self.ws.iter_examples(d)
                                        if e["id"] in held_ids and dk.classify(e) == "problems"})
        self.assertFalse(held_prompts & {p["prompt"] for p in self.svc.post_rows("problems")})

    def test_the_disk_side_is_one_query_that_uses_the_index(self):
        statements = []
        real = sqlite3.connect

        def traced(*args, **kwargs):
            connection = real(*args, **kwargs)
            connection.set_trace_callback(statements.append)
            return connection
        with patch("studio_core.sqlite3.connect", traced):
            self.svc.post_rows("conversations")
        selects = [s for s in statements if s.lstrip().startswith("SELECT") and "FROM records" in s and "user_version" not in s]
        self.assertEqual(len(selects), 1)
        with closing(real(self.ws.db_path)) as connection:
            plan = " ".join(str(r) for r in connection.execute("EXPLAIN QUERY PLAN " + selects[0]))
        self.assertIn("records_pick", plan)

    def test_a_migrated_version_0_dataset_gives_the_same_rows(self):
        root = Path(self._tmp.name) / "old"
        old = Workspace(root)
        ds = old.create_dataset("old", storage="sqlite")
        rows = mixed_rows(ds)
        write_v0_database(old.db_path, rows)
        svc = WorkspaceServices(old)  # migrates in the background; reads wait for it
        self.addCleanup(svc.close)
        for kind in ("conversations", "problems", "heldout_problems"):
            self.assertEqual(svc.post_rows(kind), old_post_rows(old, kind), kind)


if __name__ == "__main__":
    unittest.main()
