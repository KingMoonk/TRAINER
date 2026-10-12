"""studio_core: record normalising, file parsing, workspace storage (JSON and SQLite), import/export, atomic writes; train.py CLI."""
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import studio_core
from studio_core import Workspace, atomic_json, module_command, normalize, parse_file, replace_file

ROOT = Path(__file__).resolve().parent.parent
PAIRS = [{"user": f"q{i}", "assistant": f"a{i}"} for i in range(4)]


class Normalize(unittest.TestCase):
    def test_field_aliases_and_roles(self):
        self.assertEqual(normalize({"prompt": "hi", "completion": "yo"})["assistant"], "yo")
        self.assertEqual(normalize({"instruction": "sum", "input": "1 2", "output": "3"})["user"], "sum\n1 2")
        convo = normalize({"conversations": [{"from": "human", "value": "a"}, {"from": "gpt", "value": "b"}, {"from": "tool", "value": "x"}]})
        self.assertEqual([m["role"] for m in convo["messages"]], ["user", "assistant"])
        self.assertEqual(normalize({"messages": json.dumps([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}])})["assistant"], "b")

    def test_text_records_and_rejections(self):
        self.assertEqual(normalize("plain corpus")["text"], "plain corpus")
        for bad in (None, 5, {}, {"user": "only a question"}, {"messages": [{"role": "user", "content": "x"}]}, {"text": "  "}, {"messages": "[not json"}):
            self.assertIsNone(normalize(bad), bad)

    def test_provenance_fields_survive_and_synthetic_flag(self):
        row = normalize({"user": "a", "assistant": "b", "license": "MIT", "sourceId": "s1"}, "d1", "synthetic")
        self.assertEqual((row["license"], row["sourceId"], row["synthetic"], row["datasetId"]), ("MIT", "s1", True, "d1"))
        self.assertFalse(normalize({"user": "a", "assistant": "b"})["synthetic"])


class ParseFile(unittest.TestCase):
    def parse(self, name, content):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / name
            path.write_text(content, encoding="utf-8")
            return parse_file(path)

    def test_formats(self):
        self.assertEqual(len(self.parse("a.jsonl", '{"user":"a","assistant":"b"}\n\n{"user":"c","assistant":"d"}\n')), 2)
        self.assertEqual(len(self.parse("a.json", '{"examples":[{"user":"a","assistant":"b"}]}')), 1)
        self.assertEqual(self.parse("a.json", '{"user":"a","assistant":"b"}'), [{"user": "a", "assistant": "b"}])
        rows = self.parse("a.csv", "User, Assistant\nhi,yo\n")
        self.assertEqual(rows, [{"user": "hi", "assistant": "yo"}])
        self.assertEqual(self.parse("a.tsv", "user\tassistant\nhi\tyo\n")[0]["assistant"], "yo")

    def test_txt_tagged_turns_and_plain_corpus(self):
        rows = self.parse("a.txt", "User: hello\nthere\nAssistant: hi\n\njust some prose\n")
        self.assertEqual(rows[0]["messages"], [{"role": "user", "content": "hello\nthere"}, {"role": "assistant", "content": "hi"}])
        self.assertEqual(rows[1], {"text": "just some prose"})

    def test_bad_inputs(self):
        with self.assertRaisesRegex(ValueError, "line 2"):
            self.parse("a.jsonl", '{"user":"a"}\n{broken\n')
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            self.parse("a.xlsx", "x")
        with self.assertRaises(ValueError):
            self.parse("a.json", "3")

    def test_utf8_bom_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bom.jsonl"
            path.write_bytes(b'\xef\xbb\xbf{"user":"a","assistant":"b"}\n')
            self.assertEqual(parse_file(path), [{"user": "a", "assistant": "b"}])


class AtomicWrites(unittest.TestCase):
    def test_atomic_json_replaces_and_leaves_no_temp_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "deep" / "x.json"
            atomic_json(path, {"a": 1})
            atomic_json(path, {"a": "é"})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"a": "é"})
            self.assertEqual([p.name for p in path.parent.iterdir()], ["x.json"])

    def test_replace_file_retries_through_transient_permission_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "s", Path(directory) / "t"
            source.write_text("new")
            real, calls = Path.replace, []

            def flaky(self, destination):          # a Windows sharing violation that clears after two tries
                calls.append(1)
                if len(calls) < 3:
                    raise PermissionError("locked")
                return real(self, destination)
            with patch.object(Path, "replace", flaky), patch("studio_core.time.sleep"):
                replace_file(source, target)
            self.assertEqual((len(calls), target.read_text()), (3, "new"))

    def test_replace_file_gives_up_eventually(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "s"
            source.write_text("x")
            with patch.object(Path, "replace", side_effect=PermissionError("locked")), patch("studio_core.time.sleep"):
                with self.assertRaises(PermissionError):
                    replace_file(source, Path(directory) / "t", attempts=3)


class WorkspaceStorage(unittest.TestCase):
    """The same behaviour must hold for JSON-backed and SQLite-backed datasets."""

    def each_storage(self):
        for storage in ("json", "sqlite"):
            with tempfile.TemporaryDirectory() as directory:
                workspace = Workspace(directory)
                yield storage, directory, workspace, workspace.create_dataset("D", storage=storage)

    def test_add_dedupes_counts_invalid_and_persists(self):
        for storage, directory, workspace, dataset in self.each_storage():
            with self.subTest(storage):
                self.assertEqual(workspace.add(dataset, PAIRS + [PAIRS[0], {"user": "no answer"}]), (4, 1, 1))
                self.assertEqual(workspace.add(dataset, PAIRS), (0, 4, 0))
                self.assertEqual(workspace.count(dataset), 4)
                self.assertEqual(workspace.audit(dataset)["records"], 4)
                self.assertEqual(workspace.audit(dataset)["review"], 4)
                self.assertEqual(Workspace(directory).count(dataset), 4)       # reload from disk

    def test_search_paging_and_sampling(self):
        for storage, _, workspace, dataset in self.each_storage():
            with self.subTest(storage):
                workspace.add(dataset, PAIRS + [{"user": "50% off_sale", "assistant": "ok"}])
                self.assertEqual([e["user"] for e in workspace.examples(dataset, query="q2")], ["q2"])
                self.assertEqual(workspace.count(dataset, query="50%"), 1)         # LIKE wildcards are literal
                self.assertEqual(workspace.count(dataset, query="off_s"), 1)
                self.assertEqual(workspace.count(dataset, query="o%s"), 0)
                self.assertEqual([e["user"] for e in workspace.examples(dataset, limit=2, offset=1)], ["q1", "q2"])
                self.assertEqual(workspace.examples(dataset, limit=2, offset=-5)[0]["user"], "q0")
                first = workspace.sample(dataset, limit=3, seed=1)
                self.assertEqual(len(first), 3)
                self.assertEqual(first, workspace.sample(dataset, limit=3, seed=1))
                self.assertEqual(len(workspace.sample(dataset, limit=99)), 5)
                self.assertEqual(workspace.sample(dataset, limit=0), [])
                self.assertEqual(workspace.sample(dataset, limit=5, reviewed_only=True), [])

    def test_update_review_and_duplicate_guard(self):
        for storage, _, workspace, dataset in self.each_storage():
            with self.subTest(storage):
                workspace.add(dataset, PAIRS)
                one, two = workspace.examples(dataset)[:2]
                edited = workspace.update_example(dict(normalize({"user": "q0", "assistant": "changed"}), id=one["id"]))
                self.assertEqual((edited["id"], workspace.get_example(one["id"])["assistant"]), (one["id"], "changed"))
                with self.assertRaisesRegex(ValueError, "duplicates"):
                    workspace.update_example(dict(normalize({"user": "q0", "assistant": "changed"}), id=two["id"]))
                with self.assertRaises(ValueError):
                    workspace.update_example({"id": "missing", "user": "a", "assistant": "b"})
                with self.assertRaises(ValueError):
                    workspace.update_example({"id": two["id"], "messages": [{"role": "user", "content": "no reply"}]})
                workspace.mark_reviewed([one["id"], "nonexistent"])
                self.assertEqual(workspace.audit(dataset)["review"], 3)
                self.assertEqual(len(workspace.sample(dataset, reviewed_only=True)), 1)

    def test_remove_examples_and_dataset(self):
        for storage, directory, workspace, dataset in self.each_storage():
            with self.subTest(storage):
                other = workspace.create_dataset("Other", storage=storage)
                workspace.add(dataset, PAIRS)
                workspace.add(other, PAIRS)                                      # same text in a second dataset is fine
                self.assertEqual(workspace.count(other), 4)
                ids = [e["id"] for e in workspace.examples(dataset)[:2]]
                workspace.remove_examples(ids)
                self.assertEqual((workspace.count(dataset), workspace.count(other)), (2, 4))
                workspace.remove_dataset(dataset)
                self.assertEqual(workspace.count(other), 4)
                self.assertEqual([d["id"] for d in Workspace(directory).state["datasets"]], [other])

    def test_catalog_dataset_removal_writes_a_tombstone(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Workspace(directory)
            dataset = workspace.create_dataset("Cat", catalogId="x")
            atomic_json(Path(directory) / "data" / "catalog_registry.json", [{"id": dataset, "name": "Cat"}, {"id": "keep", "name": "K"}])
            workspace.remove_dataset(dataset)
            self.assertEqual(json.loads((Path(directory) / "data" / "catalog_removed.json").read_text()), [dataset])
            registry = json.loads((Path(directory) / "data" / "catalog_registry.json").read_text())
            self.assertEqual([e["id"] for e in registry], ["keep"])

    def test_registry_entries_are_merged_but_not_duplicated(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Workspace(directory)
            atomic_json(Path(directory) / "data" / "catalog_registry.json", [{"id": "catalog_a", "name": "A"}])
            self.assertEqual([d["id"] for d in Workspace(directory).state["datasets"]], ["catalog_a"])
            workspace.save()
            self.assertEqual(len(Workspace(directory).state["datasets"]), 1)

    def test_json_dataset_moves_to_disk_storage_keeping_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Workspace(directory)
            dataset = workspace.create_dataset("Big")
            workspace.add(dataset, PAIRS)
            before = [e["id"] for e in workspace.examples(dataset)]
            workspace.use_disk_storage(dataset)
            workspace.use_disk_storage(dataset)                                   # idempotent
            self.assertEqual([e["id"] for e in workspace.examples(dataset)], before)
            self.assertEqual(workspace.state["examples"], [])
            self.assertEqual(Workspace(directory).count(dataset), 4)
            with self.assertRaises(ValueError):
                workspace.use_disk_storage("nope")

    def test_validation_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Workspace(directory)
            with self.assertRaisesRegex(ValueError, "name"):
                workspace.create_dataset("   ")
            with self.assertRaises(ValueError):
                workspace.add("missing", PAIRS)
            with self.assertRaises(ValueError):
                workspace.set_dataset_metadata("missing", license="x")
            dataset = workspace.create_dataset("E")
            with self.assertRaisesRegex(ValueError, "empty"):
                workspace.export(dataset, Path(directory) / "x.jsonl")

    def test_corrupt_workspace_file_is_refused_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data" / "workspace.json"
            path.parent.mkdir()
            path.write_text('{"datasets": 3}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid workspace"):
                Workspace(directory)
            self.assertEqual(path.read_text(encoding="utf-8"), '{"datasets": 3}')

    def test_runs_listing_skips_corrupt_records_and_sorts_newest_first(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Workspace(directory)
            for name, body in (("a", '{"createdAt": "2020"}'), ("b", '{"createdAt": "2024"}'), ("bad", "{nope")):
                (workspace.run_dir / name).mkdir(parents=True)
                (workspace.run_dir / name / "run.json").write_text(body, encoding="utf-8")
            self.assertEqual([Path(r["directory"]).name for r in workspace.runs()], ["b", "a"])


class ImportExport(unittest.TestCase):
    def test_import_report_covers_good_empty_unsupported_and_missing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "good.jsonl").write_text('{"user":"a","assistant":"b"}\n{"user":"a","assistant":"b"}\n{"user":"x"}\n', encoding="utf-8")
            (root / "empty.json").write_text("[]", encoding="utf-8")
            (root / "bad.exe").write_text("x")
            workspace = Workspace(root / "ws")
            report = workspace.import_files([root / "good.jsonl", root / "empty.json", root / "bad.exe", root / "missing.csv"])
            self.assertEqual(report[0], "good.jsonl: 1 added, 1 duplicates, 1 invalid")
            self.assertIn("No chat pairs", report[1])
            self.assertIn("Unsupported", report[2])
            self.assertEqual(len(report), 4)
            self.assertEqual([d["name"] for d in workspace.state["datasets"]], ["good"])   # failed files create no dataset

    def test_export_round_trips_through_import_for_both_formats(self):
        for suffix in (".jsonl", ".csv"):
            with self.subTest(suffix), tempfile.TemporaryDirectory() as directory:
                workspace = Workspace(Path(directory) / "ws")
                dataset = workspace.create_dataset("Src")
                workspace.add(dataset, PAIRS + [{"text": "a corpus, with comma\nand newline"},
                                                {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]}])
                out = Path(directory) / f"out{suffix}"
                workspace.export(dataset, out)
                copy = Workspace(Path(directory) / "ws2")
                target = copy.create_dataset("Dst")
                self.assertEqual(copy.import_files([out], target), [f"out{suffix}: 6 added, 0 duplicates, 0 invalid"])
                original = {(e["text"], e["user"], e["assistant"]) for e in workspace.examples(dataset)}
                self.assertEqual({(e["text"], e["user"], e["assistant"]) for e in copy.examples(target)}, original)
                if suffix == ".csv":
                    with out.open(encoding="utf-8", newline="") as stream:
                        self.assertEqual(next(csv.reader(stream)), ["user", "assistant", "text", "intent", "messages"])
                else:
                    first = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
                    self.assertTrue({"id", "datasetId"}.isdisjoint(first))

    def test_import_into_chosen_dataset_and_sqlite_export(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Workspace(directory)
            dataset = workspace.create_dataset("S", storage="sqlite")
            workspace.add(dataset, PAIRS)
            out = Path(directory) / "s.jsonl"
            workspace.export(dataset, out)
            self.assertEqual(len(out.read_text(encoding="utf-8").splitlines()), 4)


class TrainCli(unittest.TestCase):
    def run_cli(self, *args, timeout=120):
        return subprocess.run([sys.executable, str(ROOT / "train.py"), *args], capture_output=True, text=True, timeout=timeout, cwd=ROOT)

    def test_list_devices_reports_cpu_and_auto(self):
        result = self.run_cli("--list-devices")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("cpu: NumPy CPU (available)", result.stdout)
        self.assertIn("auto: prefers CUDA", result.stdout)
        self.assertRegex(result.stdout, r"cuda: ")

    def test_bad_arguments_exit_with_usage_errors(self):
        result = self.run_cli()
        self.assertEqual(result.returncode, 2)
        self.assertIn("--data is required", result.stderr)
        result = self.run_cli("--device", "tpu", "--data", "x")
        self.assertEqual(result.returncode, 2)
        self.assertIn("invalid choice", result.stderr)

    def test_train_then_resume_through_the_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "d.jsonl"
            data.write_text("".join(json.dumps(p) + "\n" for p in PAIRS * 3), encoding="utf-8")
            config = root / "c.json"
            config.write_text(json.dumps(dict(width=16, layers=1, heads=2, context=32, expansion=2, dropout=0, batch_size=2,
                                              learning_rate=.004, warmup=0, validation=0, eval_every=5, objective="assistant")), encoding="utf-8")
            out = root / "run"
            result = self.run_cli("--data", str(data), "--config", str(config), "--steps", "6", "--device", "cpu", "--output", str(out))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("step=6 loss=", result.stdout)
            self.assertIn(f"completed: {out}", result.stdout)
            self.assertTrue((out / "last.npz").exists())
            record = json.loads((out / "run.json").read_text(encoding="utf-8"))
            self.assertEqual((record["status"], record["step"]), ("completed", 6))
            # resuming continues from the saved step and reads the dataset copy stored beside the checkpoint
            resumed = root / "run2"
            result = self.run_cli("--resume", str(out / "last.npz"), "--steps", "9", "--device", "cpu", "--output", str(resumed))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("step=9 loss=", result.stdout)
            self.assertNotIn("step=3 ", result.stdout)
            self.assertEqual(json.loads((resumed / "run.json").read_text(encoding="utf-8"))["step"], 9)


class SelfLaunch(unittest.TestCase):
    def test_module_command_from_source_is_python_dash_m(self):
        self.assertEqual(module_command("studio_passive", "run", "id", "--root", "/r"), [sys.executable, "-m", "studio_passive", "run", "id", "--root", "/r"])
        self.assertEqual(module_command("studio_brain3d"), [sys.executable, "-m", "studio_brain3d"])

    def test_module_command_frozen_dispatches_on_the_executable(self):
        with patch.object(sys, "frozen", True, create=True):
            self.assertEqual(module_command("studio_passive", "run", "id"), [sys.executable, "--run-module", "studio_passive", "run", "id"])

    def test_app_dispatch_runs_a_module_with_its_own_arguments(self):
        code = subprocess.run([sys.executable, str(ROOT / "app.py"), "--run-module", "studio_passive", "run", "--help"], cwd=ROOT, capture_output=True, text=True, timeout=60)
        self.assertEqual(code.returncode, 0, code.stderr)
        self.assertIn("usage: studio_passive run", code.stdout)

    def test_workspace_root_is_the_source_folder_but_a_user_folder_when_frozen(self):
        self.assertEqual(studio_core.ROOT, ROOT)
        with tempfile.TemporaryDirectory() as home, patch.object(sys, "frozen", True, create=True), patch("pathlib.Path.home", return_value=Path(home)), \
                patch.dict("os.environ", {"APPDATA": home, "XDG_DATA_HOME": home}):
            root = studio_core._root()
            self.assertEqual((root.name, root.parent.is_relative_to(home), root.is_dir()), ("Groundwork", True, True))
            self.assertNotEqual(root, ROOT)


if __name__ == "__main__":
    unittest.main()
