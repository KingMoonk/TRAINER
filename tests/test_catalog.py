"""Offline checks for large-corpus storage, provenance and atomic downloads."""
import io
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from studio_catalog import (ALL_CATALOG, STARTER_IDS, approve_catalog_rows, CATALOG, LEGACY_CATALOG, STAGES, DownloadCancelled, _fetch, clean_wikitext, convert_record, default_catalog_ids,
                            download_dataset, file_sha256, iter_json_array, iter_stories, list_downloads, register_download,
                            source_records, wikitext_paragraphs)
from studio_core import Workspace, fingerprint, normalize, parse_file


class CorpusStorageTests(unittest.TestCase):
    def test_sqlite_paging_sampling_editing_and_metadata_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Workspace(directory)
            small = workspace.create_dataset("Manual")
            workspace.add(small, [{"user": "old data", "assistant": "preserved"}])
            large = workspace.create_dataset("Disk corpus", storage="sqlite")
            rows = [{"user": f"Question {i}", "assistant": f"Answer {i}", "intent": "even" if i % 2 == 0 else "odd",
                     "quality": "Good" if i % 3 == 0 else "Review", "synthetic": True,
                     "provenance": {"source_ids": [str(i)]}, "groupId": f"group:{i // 2}"} for i in range(211)]
            self.assertEqual(workspace.add(large, rows + [rows[0], {}]), (211, 1, 1))
            self.assertEqual(len(workspace.state["examples"]), 1)
            self.assertEqual(workspace.count(large), 211)
            page = workspace.examples(large, limit=10, offset=200)
            self.assertEqual(len(page), 10)
            self.assertEqual(page[0]["user"], "Question 200")
            self.assertEqual(workspace.count(large, query="Question 20"), 11)
            self.assertEqual(workspace.count(large, query="%"), 0)
            self.assertEqual(workspace.audit(large)["review"], 140)
            self.assertEqual(workspace.audit(large)["categories"], 2)
            first = workspace.sample(large, limit=25, seed=17)
            second = workspace.sample(large, limit=25, seed=17)
            self.assertEqual([e["id"] for e in first], [e["id"] for e in second])
            self.assertEqual(len({e["id"] for e in first}), 25)
            self.assertTrue(all(e["quality"] == "Good" for e in workspace.sample(large, limit=100, reviewed_only=True)))
            row = page[0]
            row["messages"][-1]["content"] = "Edited answer"
            updated = workspace.update_example(row)
            self.assertEqual(updated["assistant"], "Edited answer")
            self.assertEqual(updated["provenance"], {"source_ids": ["200"]})
            self.assertTrue(updated["synthetic"])
            workspace.mark_reviewed([row["id"]])
            self.assertEqual(workspace.get_example(row["id"])["quality"], "Good")
            exported = Path(directory) / "export.jsonl"
            workspace.export(large, exported)
            restored = [normalize(record) for record in parse_file(exported)]
            self.assertEqual([fingerprint(e) for e in workspace.examples(large)], [fingerprint(e) for e in restored])
            self.assertEqual(restored[0]["groupId"], "group:0")
            workspace.remove_examples([row["id"]])
            self.assertEqual(workspace.count(large), 210)
            self.assertIsNone(workspace.get_example(row["id"]))
            self.assertEqual(Workspace(directory).count(large), 210)
            workspace.remove_dataset(large)
            self.assertEqual(len(workspace.examples(small)), 1)

    def test_duplicate_edit_fails_without_overwriting_existing_record(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Workspace(directory)
            target = workspace.create_dataset("Disk", storage="sqlite")
            workspace.add(target, [{"user": "a", "assistant": "b"}, {"user": "c", "assistant": "d"}])
            one, two = workspace.examples(target)
            with self.assertRaisesRegex(ValueError, "duplicates"):
                workspace.update_example({**two, "messages": one["messages"]})
            self.assertEqual(workspace.get_example(two["id"])["user"], "c")


class SourceConversionTests(unittest.TestCase):
    def test_streaming_json_handles_chunks_unicode_and_rejects_truncation(self):
        records = [{"value": "café [\\\"] and braces {}"}, {"nested": [1, 2, 3]}, {}]
        self.assertEqual(list(iter_json_array(io.StringIO(json.dumps(records)), chunk_size=5)), records)
        for broken in ('[{"a":1}', '[{},]', '[{}] garbage', '[{} {}]'):
            with self.assertRaises(ValueError):
                list(iter_json_array(io.StringIO(broken), chunk_size=3))

    def test_squad_keeps_passage_and_groups_related_questions(self):
        entry = _entry("squad2")
        base = {"context": "Aster is a blue star.", "question": "What color?", "id": "one", "answers": [{"text": "blue"}], "is_impossible": False}
        answer = list(convert_record(base, entry))[0]
        unknown = list(convert_record({**base, "id": "two", "question": "How old?", "is_impossible": True, "answers": []}, entry))[0]
        self.assertEqual(answer["assistant"], "blue")
        self.assertIn(base["context"], answer["user"])
        self.assertEqual(answer["groupId"], unknown["groupId"])
        self.assertIn("does not contain", unknown["assistant"])

    def test_openassistant_preserves_branches_and_filters_rejected_subtrees(self):
        entry = _entry("oasst1")
        tree = {"message_tree_id": "tree-1", "prompt": {"role": "prompter", "lang": "en", "text": "hello", "replies": [
            {"role": "assistant", "text": "hi", "message_id": "a", "replies": [
                {"role": "prompter", "text": "more", "replies": [{"role": "assistant", "text": "yes", "message_id": "b", "synthetic": True}]}]},
            {"role": "assistant", "text": "bad", "review_result": False, "replies": []}]}}
        converted = list(convert_record(tree, entry))
        self.assertEqual(len(converted), 2)
        self.assertEqual(len(converted[1]["messages"]), 4)
        self.assertEqual(converted[0]["groupId"], "oasst1:tree-1")
        self.assertFalse(converted[0]["synthetic"])
        self.assertTrue(converted[1]["synthetic"])


class DownloadLifecycleTests(unittest.TestCase):
    @staticmethod
    def fake_fetch(url, path, stop=None, emit=None, maximum=None):
        payload = (json.dumps({"instruction": "Say hello", "response": "Hello", "context": "", "category": "greeting"}) + "\n").encode() if Path(path).suffix == ".jsonl" else b"Original dataset attribution."
        Path(path).write_bytes(payload)
        return {"url": url, "filename": Path(path).name, "bytes": len(payload), "sha256": file_sha256(path)}

    def test_catalogue_sets_registered_as_drafts_are_approved_and_other_sets_are_left_alone(self):
        with tempfile.TemporaryDirectory() as directory, patch("studio_catalog._fetch", side_effect=self.fake_fetch):
            workspace = Workspace(directory)
            dataset_id = register_download(workspace, download_dataset("dolly15k", root=directory))
            own = workspace.create_dataset("Generated", storage="sqlite")
            workspace.add(own, [{"text": "a draft of my own", "quality": "Review", "synthetic": True}])
            with workspace._connect({dataset_id}) as connection:  # as registered before rows were approved on import
                connection.execute("UPDATE records SET quality='Review', payload=json_set(payload, '$.quality', 'Review') WHERE dataset_id=?", (dataset_id,))
            self.assertEqual(workspace.review_count(dataset_id), 1)
            rev = workspace._dataset(dataset_id)["rev"]
            self.assertEqual(approve_catalog_rows(workspace), [dataset_id])
            self.assertEqual((workspace.review_count(dataset_id), workspace.review_count(own)), (0, 1))
            self.assertEqual(workspace.examples(dataset_id)[0]["quality"], "Good")
            self.assertGreater(workspace._dataset(dataset_id)["rev"], rev)
            self.assertEqual(approve_catalog_rows(workspace), [])

    def test_cache_checksums_registration_and_legacy_workspace_recovery(self):
        with tempfile.TemporaryDirectory() as directory, patch("studio_catalog._fetch", side_effect=self.fake_fetch) as fetch:
            result = download_dataset("dolly15k", root=directory)
            self.assertEqual(result["records"], 1)
            self.assertEqual(len(list_downloads(directory)), 1)
            calls = fetch.call_count
            again = download_dataset("dolly15k", root=directory)
            self.assertTrue(again["cached"])
            self.assertEqual(fetch.call_count, calls)
            workspace = Workspace(directory)
            dataset_id = register_download(workspace, result)
            self.assertEqual(register_download(workspace, result), dataset_id)
            self.assertEqual(workspace.count(dataset_id), 1)
            self.assertEqual(len(workspace.state["examples"]), 0)
            self.assertEqual(workspace.examples(dataset_id)[0]["quality"], "Good")  # a catalogue corpus needs no review
            self.assertEqual(workspace.review_count(dataset_id), 0)
            # A previous-version GUI can save old JSON without losing the disk corpus.
            workspace.path.write_text(json.dumps({"datasets": [], "examples": []}), encoding="utf-8")
            restored = Workspace(directory)
            self.assertEqual(restored.count(dataset_id), 1)
            self.assertEqual(restored.examples(dataset_id)[0]["license"], "CC BY-SA 3.0")
            Path(result["path"]).write_text("corrupt", encoding="utf-8")
            repaired = download_dataset("dolly15k", root=directory)
            self.assertEqual(repaired["records"], 1)
            self.assertEqual(fetch.call_count, calls)  # intact upstream file reused

    def test_pinned_checksum_mismatch_is_rejected(self):
        entry = _entry("dolly15k")
        with tempfile.TemporaryDirectory() as directory, patch("studio_catalog._fetch", side_effect=self.fake_fetch), \
                patch.dict(entry, {"sha256": "0" * 64}):
            with self.assertRaisesRegex(ValueError, "pinned checksum"):
                download_dataset("dolly15k", root=directory)
            self.assertFalse((Path(directory) / "data" / "downloads" / "dolly15k" / "source.json").exists())

    def test_toolace_download_keeps_each_tool_conversation_in_its_own_group(self):
        rows = [{"system": "Use the tools listed here.", "conversations": [
            {"from": "user", "value": f"Weather in city {index}?"},
            {"from": "assistant", "value": f"[weather(city={index})]"},
            {"from": "tool", "value": f"{{'temp': {index}}}"},
            {"from": "assistant", "value": f"It is {index} degrees."}]} for index in range(2)]

        def fetch(url, path, stop=None, emit=None, maximum=None, range_bytes=None):
            payload = json.dumps(rows).encode() if Path(path).name == "data.json" else b"ToolACE source card"
            Path(path).write_bytes(payload)
            return {"url": url, "filename": Path(path).name, "bytes": len(payload), "sha256": file_sha256(path)}

        with tempfile.TemporaryDirectory() as directory, patch("studio_catalog._fetch", side_effect=fetch), \
                patch.dict(_entry("toolace"), {"sha256": ""}):
            result = download_dataset("toolace", root=directory)
            converted = [json.loads(line) for line in Path(result["path"]).read_text(encoding="utf-8").splitlines()]
        self.assertEqual(result["records"], 2)
        self.assertEqual(len({row["groupId"] for row in converted}), 2)
        self.assertTrue(all(row["intent"] == "tool_use" for row in converted))
        self.assertTrue(all(row["messages"][2]["content"].startswith("<tool_call>") for row in converted))

    def test_failed_license_fetch_is_tolerated_and_recorded(self):
        def fetch(url, path, stop=None, emit=None, maximum=None):
            if "LICENSE" in url:
                raise OSError("offline")
            payload = (json.dumps({"question": "1+1?", "answer": "2"}) + "\n").encode()
            Path(path).write_bytes(payload)
            return {"url": url, "filename": Path(path).name, "bytes": len(payload), "sha256": file_sha256(path)}
        with tempfile.TemporaryDirectory() as directory, patch("studio_catalog._fetch", side_effect=fetch):
            self.assertEqual(download_dataset("gsm8k", root=directory)["license_file"], "unavailable")

    def test_corrupt_state_rebuilds_and_line_endings_are_normalised(self):
        def fetch(url, path, stop=None, emit=None, maximum=None):
            payload = (json.dumps({"instruction": "Say\r\nhi", "response": "Hi\rthere", "context": ""}) + "\n").encode()
            Path(path).write_bytes(payload)
            return {"url": url, "filename": Path(path).name, "bytes": len(payload), "sha256": file_sha256(path)}
        with tempfile.TemporaryDirectory() as directory, patch("studio_catalog._fetch", side_effect=fetch):
            first = download_dataset("dolly15k", root=directory)
            folder = Path(directory) / "data" / "downloads" / "dolly15k"
            for name in ("manifest.json", "source.json"):
                (folder / name).write_text("{", encoding="utf-8")
            with self.assertLogs("studio_catalog", "WARNING"):
                again = download_dataset("dolly15k", root=directory)
            self.assertEqual(again["sha256"], first["sha256"])
            row = json.loads(Path(again["path"]).read_text(encoding="utf-8"))
            self.assertEqual((row["user"], row["assistant"]), ("Say\nhi", "Hi\nthere"))

    def test_registration_checks_the_actual_row_count(self):
        with tempfile.TemporaryDirectory() as directory, patch("studio_catalog._fetch", side_effect=self.fake_fetch):
            result = download_dataset("dolly15k", root=directory)
            workspace = Workspace(directory)
            dataset_id = register_download(workspace, result)
            with workspace._connect() as connection:
                connection.execute("DELETE FROM records WHERE dataset_id=?", (dataset_id,))
            register_download(workspace, result)
            self.assertEqual(workspace.count(dataset_id), 1)

    def test_cancelled_normalization_never_advertises_a_complete_dataset(self):
        with tempfile.TemporaryDirectory() as directory, patch("studio_catalog._fetch", side_effect=self.fake_fetch):
            stop = threading.Event()
            def on_progress(progress):
                if progress["text"].startswith("Preparing"):
                    stop.set()
            with self.assertRaises(DownloadCancelled):
                download_dataset("dolly15k", root=directory, stop=stop, emit=on_progress)
            self.assertEqual(list_downloads(directory), [])
            self.assertFalse(list(Path(directory).rglob("*.part")))

    def test_interrupted_transfer_removes_partial_file(self):
        with tempfile.TemporaryDirectory() as directory:
            stop = threading.Event()
            class Response(io.BytesIO):
                headers = {"Content-Length": "4"}
                def read(self, size=-1):
                    stop.set()
                    return super().read(size)
            path = Path(directory) / "source.jsonl"
            with patch("urllib.request.urlopen", return_value=Response(b"data")):
                with self.assertRaises(DownloadCancelled):
                    _fetch("https://example.invalid/source", path, stop=stop)
            self.assertFalse(path.exists())
            self.assertFalse(path.with_suffix(".jsonl.part").exists())


def _entry(catalog_id):
    return next(e for e in ALL_CATALOG if e["id"] == catalog_id)


def _convert(catalog_id, raw):
    return list(convert_record(raw, _entry(catalog_id)))


class CatalogMetadataTests(unittest.TestCase):
    def test_every_entry_is_pinned_licensed_staged_and_unique(self):
        ids = [e["id"] for e in CATALOG]
        self.assertEqual(len(ids), len(set(ids)))
        for entry in CATALOG:
            with self.subTest(entry["id"]):
                self.assertIn(entry["stage"], STAGES)
                self.assertIsInstance(entry["default"], bool)
                self.assertRegex(entry["revision"], r"^[0-9a-f]{40}$")
                self.assertTrue(entry["license"] and entry["source_url"] and entry["split"])
                if "non-commercial" in entry["license"].lower():
                    self.assertFalse(entry["default"])
                    self.assertIn("NON-COMMERCIAL", entry["description"])

    def test_default_ids_are_the_nanochat_recipe_and_stay_in_stage_order(self):
        defaults = default_catalog_ids()
        self.assertEqual(defaults, ["fineweb_edu_00", "fineweb_edu_01", "fineweb_edu_02", "fineweb_edu_03", "cosmopedia_khan", "cosmopedia_openstax",
                                    "mmlu_auxiliary", "gsm8k", "smol_smoltalk", "toolace"])
        self.assertEqual(STARTER_IDS, tuple(defaults))
        for catalog_id in defaults:
            self.assertTrue(_entry(catalog_id)["license"])
        stages = [STAGES.index(_entry(i)["stage"]) for i in defaults]
        self.assertEqual(stages, sorted(stages))
        self.assertEqual({e["stage"] for e in CATALOG}, {"language", "instructions", "knowledge", "reasoning", "conversation"})
        self.assertTrue(all(not e["default"] for e in LEGACY_CATALOG))
        optional = {e["id"] for e in CATALOG if not e["default"]}
        self.assertEqual(optional, {"fineweb_edu_04", "fineweb_edu_05", "fineweb_edu_06", "fineweb_edu_07", "cosmopedia_openstax_b",
                                    "dolly15k", "squad2", "oasst1", "smol_smoltalk_b"})

    def test_hugging_face_entries_are_hashed_pinned_and_under_the_download_cap(self):
        for entry in CATALOG:
            if entry.get("repo"):
                with self.subTest(entry["id"]):
                    self.assertTrue(entry["id"] in {"dolly15k", "oasst1"} or "sha256" in entry)
                    if "sha256" in entry:
                        self.assertRegex(entry["sha256"], r"^[0-9a-f]{64}$")
        shard = _entry("fineweb_edu_00")
        self.assertEqual((shard["repo"], shard["filename"], shard["size_hint"]), ("karpathy/fineweb-edu-100b-shuffle", "shard_00000.parquet", "95 MB download (one shard)"))
        self.assertEqual(_entry("fineweb_edu_07")["filename"], "shard_00007.parquet")
        self.assertEqual(_entry("mmlu_auxiliary")["filename"], "auxiliary_train/train-00000-of-00001.parquet")

    def test_existing_entries_keep_their_pinned_revisions(self):
        self.assertEqual(_entry("dolly15k")["revision"], "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a")
        self.assertEqual(_entry("oasst1")["filename"], "2023-04-12_oasst_ready.trees.jsonl.gz")
        self.assertEqual(_entry("gsm8k")["revision"], "3101c7d5072418e28b9008a6636bde82a006892c")


class CheckableProblemTests(unittest.TestCase):
    """Problem-stage converters: each row is a problem whose stored answer passes its own keyed rule."""

    def _problem(self, catalog_id, raw):
        import studio_data_kinds as dk
        import studio_rules
        from studio_core import normalize
        rows = _convert(catalog_id, raw)
        self.assertEqual(len(rows), 1, raw)
        example = normalize({**rows[0], "quality": "Review"}, "d", "catalog")
        self.assertEqual(dk.classify(example), "problems")
        params = example["metadata"]["params"]
        self.assertTrue(studio_rules.check(dk.rule_of(example), example["user"], example["assistant"], params))
        return example, dk.rule_of(example), params["answer"]

    def test_numeric_keys(self):
        ex, rule, key = self._problem("gsm8k_check", {"question": "How many?", "answer": "48/2 = <<48/2=24>>24\n#### 1,072"})
        self.assertEqual((rule, key, ex["assistant"]), ("final_number", "1072", "1072"))
        self.assertTrue(ex["user"].startswith("How many?") and ex["user"].endswith("as a number."))
        self.assertEqual(self._problem("svamp_check", {"ID": "s1", "question_concat": "Some apples?", "Answer": 145.0})[2], "145")
        self.assertEqual(self._problem("metamath_check", {"query": "Q", "response": "Work.\nThe answer is: 2.5"})[2], "2.5")
        for raw in ({"query": "Q", "response": "The answer is: \\frac{1}{2}"}, {"query": "Q", "response": "no key"}):
            self.assertEqual(_convert("metamath_check", raw), [])
        self.assertEqual(_convert("gsm8k_check", {"question": "Q", "answer": "#### seven"}), [])

    def test_lettered_keys(self):
        ex, rule, key = self._problem("aqua_check", {"question": "x?", "options": ["A)21", "B)21.5", "C)22"], "correct": "C"})
        self.assertEqual((rule, key, ex["assistant"]), ("choice", "C", "C) 22"))
        self.assertIn("A) 21\nB) 21.5\nC) 22", ex["user"])
        self.assertEqual(self._problem("qasc_check", {"id": "q", "question": "Rain?", "answerKey": "B",
                                                      "choices": {"label": ["A", "B"], "text": ["dust", "water"]}})[0]["assistant"], "B) water")
        self.assertEqual(self._problem("hellaswag_check", {"activity_label": "Cooking", "ctx": "He stirs", "label": "1",
                                                           "endings": ["the sky.", "the soup."]})[0]["assistant"], "B) The soup.")
        boolq = self._problem("boolq_check", {"question": "is water wet", "passage": "Water is wet.", "answer": False})[0]
        self.assertEqual(boolq["assistant"], "B) no")
        self.assertIn("Question: Is water wet?", boolq["user"])
        self.assertEqual(self._problem("winogrande_check", {"sentence": "A _ b", "option1": "x", "option2": "y", "answer": "2"})[2], "B")
        self.assertEqual(_convert("aqua_check", {"question": "x?", "options": ["A)1", "B)2"], "correct": "E"}), [])
        self.assertEqual(_convert("hellaswag_check", {"ctx": "c", "label": "", "endings": ["a", "b"]}), [])

    def test_legacy_problem_sources_are_not_in_the_replacement_catalog(self):
        problems = [e for e in LEGACY_CATALOG if e["stage"] == "problems"]
        self.assertTrue(all(e["parser"].endswith("_check") for e in problems))
        self.assertFalse(_entry("winogrande_check")["default"])
        self.assertFalse(any(e["default"] for e in problems))
        self.assertEqual(_convert("simplewiki", {"id": "7", "title": "T", "text": "short"}), [])
        article = _convert("simplewiki", {"id": "7", "title": "T", "text": "word " * 60})[0]
        self.assertEqual((article["intent"], article["metadata"]["title"]), ("encyclopedia", "T"))
        chat = _convert("ultrachat_a", {"prompt_id": "p", "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]})[0]
        self.assertEqual((chat["groupId"], len(chat["messages"])), ("ultrachat:p", 2))


class NewConverterTests(unittest.TestCase):
    def test_multiple_choice_maps_numeric_labels_to_letters_and_skips_bad_keys(self):
        raw = {"id": "q1", "question": "Which is a liquid?", "answerKey": "2",
               "choices": {"label": ["1", "2", "3"], "text": ["rock", "water", "iron"]}}
        row = _convert("arc_easy", raw)[0]
        self.assertEqual(row["assistant"], "B) water")
        self.assertIn("A) rock\nB) water\nC) iron", row["user"])
        self.assertEqual(row["sourceId"], "q1")
        self.assertEqual(_convert("arc_easy", {**raw, "answerKey": "Z"}), [])
        self.assertEqual(_convert("arc_easy", {**raw, "question": " "}), [])
        csqa = _convert("commonsenseqa", {"id": "c", "question": "Where?", "question_concept": "x", "answerKey": "E",
                                          "choices": {"label": list("ABCDE"), "text": list("vwxyz")}})[0]
        self.assertEqual(csqa["assistant"], "E) z")
        self.assertEqual(csqa["intent"], "commonsense_qa")

    def test_sciq_shuffle_is_deterministic_and_keeps_the_correct_answer_and_support(self):
        raw = {"question": "What is H2O?", "correct_answer": "water", "distractor1": "salt", "distractor2": "oil",
               "distractor3": "sand", "support": "Water is H2O."}
        first, second = _convert("sciq", raw)[0], _convert("sciq", dict(raw))[0]
        self.assertEqual(first, second)
        self.assertTrue(first["assistant"].endswith(") water"))
        self.assertIn(first["assistant"], first["user"])
        self.assertIn("Background:\nWater is H2O.", first["user"])
        self.assertEqual(sum(1 for letter in "ABCD" if f"\n{letter}) " in first["user"]), 4)
        no_answer = _convert("sciq", {**raw, "correct_answer": ""})
        self.assertEqual(no_answer, [])

    def test_mbpp_embeds_tests_and_normalises_whitespace(self):
        row = _convert("mbpp", {"task_id": 7, "text": "Write a function to add.", "code": "def add(a, b):\r\n\treturn a + b \r\n",
                                "test_list": ["assert add(1, 2) == 3", "assert add(0, 0) == 0"]})[0]
        self.assertIn("assert add(1, 2) == 3", row["user"])
        self.assertEqual(row["assistant"], "def add(a, b):\n    return a + b")
        self.assertEqual(row["sourceId"], "mbpp:7")

    def test_alpaca_metamath_story_and_no_robots(self):
        alpaca = _convert("alpaca_cleaned", {"instruction": "Translate", "input": "hello", "output": "hola"})[0]
        self.assertEqual(alpaca["user"], "Translate\n\nInput:\nhello")
        plain = _convert("alpaca_cleaned", {"instruction": "Say hi", "input": "", "output": "Hi"})[0]
        self.assertEqual(plain["user"], "Say hi")
        self.assertEqual(_convert("alpaca_cleaned", {"instruction": "Say hi", "input": "", "output": " "}), [])
        math = _convert("metamathqa", {"query": "1+1?", "response": "2", "type": "GSM_Rephrased"})[0]
        self.assertEqual((math["user"], math["assistant"], math["metadata"]["type"]), ("1+1?", "2", "GSM_Rephrased"))
        self.assertEqual(_convert("metamathqa", {"query": "", "response": "2"}), [])
        self.assertEqual(_convert("tinystories", {"text": "Once upon a time."})[0]["text"], "Once upon a time.")
        robots = _convert("no_robots", {"prompt_id": "p1", "category": "Brainstorm", "messages": [
            {"role": "user", "content": "Ideas?"}, {"role": "assistant", "content": "Many."}]})[0]
        self.assertEqual(robots["intent"], "brainstorm")
        self.assertEqual(robots["groupId"], "no_robots:p1")
        row = normalize(robots, source="catalog:no_robots")
        self.assertEqual((row["user"], row["assistant"]), ("Ideas?", "Many."))

    def test_toolace_keeps_tool_schemas_calls_results_and_turn_order(self):
        raw = {"system": "Available tools: [{\"name\":\"weather\"}]", "conversations": [
            {"from": "user", "value": "Will it rain?"},
            {"from": "assistant", "value": "[weather(city=\"Boston\")]"},
            {"from": "tool", "value": "{\"rain\":true}"},
            {"from": "assistant", "value": "Yes, the tool reports rain."}]}
        row = _convert("toolace", raw)[0]
        normalized = normalize(row, source="catalog:toolace")
        self.assertEqual(normalized["intent"], "tool_use")
        self.assertEqual([message["role"] for message in normalized["messages"]], ["system", "user", "assistant", "user", "assistant"])
        self.assertIn("Available tools", normalized["messages"][0]["content"])
        self.assertIn("<tool_call>", normalized["messages"][2]["content"])
        self.assertIn("<tool_response>", normalized["messages"][3]["content"])
        self.assertNotIn("groupId", row)
        self.assertEqual(_convert("toolace", {**raw, "conversations": raw["conversations"][:1]}), [])

    def test_story_rows_normalise_as_plain_text_records(self):
        row = normalize(_convert("tinystories", {"text": "Tim had a red ball."})[0], source="catalog:tinystories")
        self.assertEqual(row["text"], "Tim had a red ball.")
        self.assertEqual(row["messages"], [])

    def test_story_splitter_handles_chunk_boundaries_and_truncated_prefixes(self):
        text = "One story.<|endoftext|>\nTwo story.<|endoftext|>\nThree is cut o"
        for size in (1, 4, 7, 1000):
            # Text after the last marker may be a truncated story, so it is always dropped.
            self.assertEqual(list(iter_stories(io.StringIO(text), chunk_size=size)), ["One story.", "Two story."])

    def test_truncated_json_prefix_yields_only_whole_records(self):
        full = json.dumps([{"a": "x" * 10}, {"a": "y" * 10}, {"a": "z" * 10}])
        for cut in range(2, len(full)):
            with self.subTest(cut=cut):
                parsed = list(iter_json_array(io.StringIO(full[:cut]), chunk_size=4, truncated=True))
                self.assertEqual(parsed, [{"a": "x" * 10}, {"a": "y" * 10}, {"a": "z" * 10}][:len(parsed)])
        with self.assertRaises(ValueError):
            list(iter_json_array(io.StringIO(full[:-5]), chunk_size=4))

    def test_wikitext_paragraphs_track_titles_and_detokenize(self):
        rows = [{"text": ""}, {"text": " = Aster Station = \n"}, {"text": " = = History = = \n"},
                {"text": " The station was built in 1990 @-@ 91 , and it , like most , served the region 's trains . " * 2},
                {"text": " short . \n"}, {"text": " = Other Page = \n"}, {"text": " " + "Words go here and there , again . " * 4}]
        out = list(wikitext_paragraphs(rows))
        self.assertEqual([(o["title"], o["section"]) for o in out], [("Aster Station", "History"), ("Other Page", "")])
        self.assertIn("1990-91, and it, like most, served the region's trains.", out[0]["text"])
        self.assertNotIn(" , ", out[1]["text"])
        self.assertEqual(clean_wikitext(" He said \" hello \" ( really ) . "), 'He said "hello" (really).')
        converted = _convert("wikitext2", out[0])[0]
        self.assertEqual(converted["groupId"], "wikitext2:Aster Station")


class HigherQualityCorpusTests(unittest.TestCase):
    """Converters for the second wave of corpora, each fed a small row in its source's real schema."""

    def test_large_legacy_sources_are_not_in_the_replacement_catalog(self):
        defaults = set(default_catalog_ids())
        for catalog_id in ("fineweb_edu", "openwebmath", "gutenberg", "orca_math",
                           "stackexchange_math", "magicoder_oss", "stackexchange_unix", "smoltalk_magpie", "reddit_instruct"):
            self.assertNotIn(catalog_id, defaults)
            self.assertNotIn(catalog_id, {entry["id"] for entry in CATALOG})
            self.assertRegex(_entry(catalog_id)["revision"], r"^[0-9a-f]{40}$")
        self.assertFalse(_entry("openbookqa")["default"])
        self.assertIn("no licence", _entry("openbookqa")["description"])
        self.assertEqual(_entry("gutenberg")["batch_size"], 8)

    def test_stack_exchange_html_becomes_text_with_the_accepted_answer(self):
        raw = {"qid": 77, "date": "2012/01/01", "metadata": ["https://math.stackexchange.com/questions/77"],
               "question": "<p>Why is <code>x &lt; y</code> true?</p>\n\n<pre><code>print(1 &lt; 2)\n</code></pre>\n<ul><li>one</li></ul>",
               "answers": [{"answer_id": 1, "author": "a", "author_id": 1, "author_profile": "", "pm_score": 9, "selected": False, "text": "<p>Top.</p>"},
                           {"answer_id": 2, "author": "b", "author_id": 2, "author_profile": "", "pm_score": 3, "selected": True,
                            "text": "<p>Because $x&lt;y$.<br>See <a href=\"http://e.org\">this</a>.</p>"}]}
        row = _convert("stackexchange_math", raw)[0]
        self.assertEqual(row["user"], "Why is `x < y` true?\n\n```\nprint(1 < 2)\n```\n\n- one")
        self.assertEqual(row["assistant"], "Because $x<y$.\nSee this.")
        self.assertEqual((row["intent"], row["sourceId"], row["metadata"]["accepted"]), ("forum_qa", "stackexchange_math:77", True))
        best = _convert("stackexchange_unix", {**raw, "answers": [{**raw["answers"][0]}, {**raw["answers"][1], "selected": False}]})[0]
        self.assertEqual(best["assistant"], "Top.")  # no accepted answer: the best-scored one
        self.assertEqual(_convert("stackexchange_math", {**raw, "answers": [{**raw["answers"][1], "pm_score": 0}]}), [])
        self.assertEqual(_convert("stackexchange_math", {**raw, "answers": []}), [])

    def test_reddit_instruct_skips_removed_posts(self):
        raw = {"post_title": "What is billet?", "post_text": "I cannot find an explanation.", "post_scores": 24,
               "comment_text": "A chunk of metal.", "comment_score": 54}
        row = _convert("reddit_instruct", raw)[0]
        self.assertEqual((row["user"], row["assistant"]), ("What is billet?\n\nI cannot find an explanation.", "A chunk of metal."))
        self.assertEqual(row["metadata"], {"post_score": 24, "comment_score": 54})
        self.assertEqual(_convert("reddit_instruct", {**raw, "post_text": ""})[0]["user"], "What is billet?")
        self.assertEqual(_convert("reddit_instruct", {**raw, "comment_text": "[removed]"}), [])
        self.assertEqual(_convert("reddit_instruct", {**raw, "post_text": "[deleted]"}), [])

    def test_web_text_corpora_keep_pages_and_skip_stubs(self):
        page = "Bromide in rivers. " * 20
        fine = _convert("fineweb_edu", {"text": page, "id": "<urn:uuid:1>", "dump": "CC-MAIN-2013-20", "url": "http://a.org", "language": "en",
                                        "language_score": 0.9, "token_count": 80, "score": 3.2, "int_score": 3})[0]
        self.assertEqual((fine["text"], fine["intent"], fine["sourceId"]), (page.strip(), "web_text", "<urn:uuid:1>"))
        self.assertEqual(_convert("fineweb_edu", {"text": "short", "id": "x"}), [])
        shuffle = [_convert("fineweb_edu_00", {"text": page + str(n)})[0] for n in (1, 2)]  # the shuffle shards carry only "text"
        self.assertNotEqual(shuffle[0]["sourceId"], shuffle[1]["sourceId"])
        self.assertEqual(shuffle[0]["groupId"], "fineweb_edu_00:" + shuffle[0]["sourceId"])
        math = _convert("openwebmath", {"url": "http://m.org/bayes", "text": "Bayes. $P(A|B)$ " * 20, "date": "2020-01-01", "metadata": "{}"})[0]
        self.assertEqual((math["intent"], math["metadata"]["url"]), ("math_text", "http://m.org/bayes"))
        self.assertTrue(math["groupId"].startswith("openwebmath:"))
        self.assertEqual(_convert("openwebmath", {"url": "u", "text": " "}), [])
        book = _convert("cosmopedia_khan", {"prompt": "Write a unit", "text": " Lesson Objective: count.", "text_token_length": 5,
                                            "seed_data": "khanacademy", "format": "textbook", "audience": "middle_school"})[0]
        self.assertEqual((book["text"], book["intent"], book["metadata"]["audience"]), ("Lesson Objective: count.", "textbook", "middle_school"))
        self.assertEqual(_convert("cosmopedia_khan", {"prompt": "p", "text": ""}), [])

    def test_gutenberg_rejoins_hard_wrapped_lines_and_cuts_passages(self):
        paragraph = "\r\n\r\n".join(["The quick brown fox jumps over the lazy dog and keeps running far"] * 10)
        wrapped = "\r\n".join([paragraph] * 20)
        meta = json.dumps({"language": "en", "text_id": 22428, "title": "Essays\nA Series", "authors": "Wallace, Alfred Russel"})
        passages = _convert("gutenberg", {"TEXT": wrapped, "SOURCE": "gutenberg", "METADATA": meta})
        self.assertEqual(len(passages), 5)  # four ~650-character paragraphs per passage; the short remainder is dropped
        self.assertTrue(all(2000 <= len(p["text"]) <= 3300 for p in passages))
        self.assertNotIn("\n\n", passages[0]["text"])
        self.assertTrue(passages[0]["text"].startswith("The quick brown fox jumps over the lazy dog and keeps running far The quick"))
        self.assertEqual((passages[0]["sourceId"], passages[1]["sourceId"], passages[0]["groupId"]), ("gutenberg:22428:1", "gutenberg:22428:2", "gutenberg:22428"))
        self.assertEqual((passages[0]["metadata"]["title"], passages[0]["intent"]), ("Essays", "book"))
        paragraphs = "\n\n".join(["A paragraph that is long enough to be a whole line of its own in this book. " * 6] * 5)
        self.assertEqual(len(_convert("gutenberg", {"TEXT": paragraphs, "METADATA": "not json"})), 1)  # real paragraphs are left alone
        self.assertEqual(_convert("gutenberg", {"TEXT": "A tiny book.", "METADATA": meta}), [])

    def test_instruction_corpora_map_their_fields(self):
        orca = _convert("orca_math", {"question": "Find the number of people ahead of 5th place.", "answer": "4 people crossed first."})[0]
        self.assertEqual((orca["user"], orca["assistant"], orca["intent"]), ("Find the number of people ahead of 5th place.", "4 people crossed first.", "math_reasoning"))
        self.assertEqual(_convert("orca_math", {"question": "Q", "answer": " "}), [])
        code = _convert("magicoder_oss", {"lang": "cpp", "raw_index": 101533, "index": 4626, "seed": "int n;", "openai_fingerprint": "fp_eeff13170a",
                                          "problem": "Write convolution.", "solution": "```cpp\nint main() {}\n```"})[0]
        self.assertEqual((code["user"], code["assistant"], code["intent"], code["sourceId"]),
                         ("Write convolution.", "```cpp\nint main() {}\n```", "coding", "magicoder:cpp:4626"))
        self.assertEqual(code["metadata"], {"language": "cpp"})
        self.assertEqual(_convert("magicoder_oss", {"lang": "cpp", "problem": "", "solution": "x"}), [])

    def test_smoltalk_keeps_only_good_chats_and_openbookqa_maps_stems(self):
        chat = {"messages": [{"role": "user", "content": "Plan a lake trip."}, {"role": "assistant", "content": "Weigh rain and cost."}],
                "category": "planning", "difficulty": "easy", "quality": "good", "reward_model_score": 0.1, "conversation_tokens": 20}
        row = _convert("smoltalk_magpie", chat)[0]
        self.assertEqual((row["intent"], len(row["messages"]), row["metadata"]["quality"]), ("planning", 2, "good"))
        self.assertEqual(row["groupId"], _convert("smoltalk_magpie", {**chat, "quality": "excellent"})[0]["groupId"])
        for bad in ("average", "poor", "very poor"):
            self.assertEqual(_convert("smoltalk_magpie", {**chat, "quality": bad}), [])
        obqa = _convert("openbookqa", {"id": "7-980", "question_stem": "The sun is responsible for", "answerKey": "D",
                                       "choices": {"text": ["puppies", "children", "flowers", "plants sprouting"], "label": ["A", "B", "C", "D"]}})[0]
        self.assertEqual((obqa["assistant"], obqa["sourceId"], obqa["intent"]), ("D) plants sprouting", "7-980", "science_qa"))
        self.assertIn("Question: The sun is responsible for", obqa["user"])

    def test_catalog_rows_normalise_into_training_records(self):
        for catalog_id, raw in (("stackexchange_math", {"qid": 1, "question": "<p>Q?</p>", "answers": [{"pm_score": 2, "selected": True, "text": "<p>A.</p>"}]}),
                                ("smoltalk_magpie", {"quality": "good", "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]}),
                                ("fineweb_edu", {"id": "i", "text": "word " * 60}), ("reddit_instruct", {"post_title": "T", "comment_text": "C"})):
            with self.subTest(catalog_id):
                row = normalize(_convert(catalog_id, raw)[0], source=f"catalog:{catalog_id}")
                self.assertTrue(row["text"] or (row["user"] and row["assistant"]))


class BoundedDownloadTests(unittest.TestCase):
    def test_range_fetch_requests_a_prefix_and_never_keeps_more_than_asked(self):
        seen = {}

        class Response(io.BytesIO):
            headers = {"Content-Length": "1000"}  # a server that ignores Range and sends everything

        def opener(request, timeout=None):
            seen.update(request.headers)
            return Response(b"0123456789" * 100)

        with tempfile.TemporaryDirectory() as directory, patch("urllib.request.urlopen", side_effect=opener):
            receipt = _fetch("https://example.invalid/big.txt", Path(directory) / "big.txt", range_bytes=25)
            self.assertEqual(Path(directory, "big.txt").read_bytes(), b"0123456789012345678901234")
            self.assertEqual(receipt["bytes"], 25)
            self.assertEqual(receipt["sha256"], file_sha256(Path(directory) / "big.txt"))
        self.assertEqual(seen.get("Range"), "bytes=0-24")

    def test_stories_download_drops_cut_off_story_honours_max_records_and_records_range(self):
        payload = "A first story.<|endoftext|>\nA second story.<|endoftext|>\nA third story.<|endoftext|>\nA cut-off fou"
        fetches = []

        def fake(url, path, stop=None, emit=None, maximum=None, range_bytes=None):
            fetches.append(range_bytes)
            data = payload.encode() if Path(path).suffix == ".txt" else b"Card."
            Path(path).write_bytes(data)
            return {"url": url, "filename": Path(path).name, "bytes": len(data), "sha256": file_sha256(path)}

        entry = _entry("tinystories")
        with tempfile.TemporaryDirectory() as directory, patch("studio_catalog._fetch", side_effect=fake), \
                patch.dict(entry, {"max_records": 2}):
            result = download_dataset("tinystories", root=directory)
            self.assertEqual(result["records"], 2)  # three whole stories available, capped at two
            self.assertEqual(result["stage"], "language")
            rows = [json.loads(line) for line in Path(result["path"]).read_text(encoding="utf-8").splitlines()]
            self.assertEqual([r["text"] for r in rows], ["A first story.", "A second story."])
            self.assertEqual(rows[0]["license"], "CDLA-Sharing-1.0")
            self.assertEqual(rows[0]["provenance"]["revision"], entry["revision"])
            self.assertEqual(fetches[0], entry["range_bytes"])
            receipt = json.loads((Path(directory) / "data" / "downloads" / "tinystories" / "source.json").read_text())
            self.assertEqual(receipt["range_bytes"], entry["range_bytes"])
            self.assertTrue(download_dataset("tinystories", root=directory)["cached"])

    def test_parquet_sources_stream_through_pyarrow_and_convert(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("pyarrow is not installed")
        table = pa.table({"id": ["a", "b"], "question": ["Pick one", "Pick two"],
                          "choices": [{"label": ["A", "B"], "text": ["x", "y"]}, {"label": ["A", "B"], "text": ["p", "q"]}],
                          "answerKey": ["B", "A"]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.parquet"
            pq.write_table(table, path)
            rows = list(source_records(path, _entry("arc_easy")))
            self.assertEqual(len(rows), 2)
            converted = [r for raw in rows for r in convert_record(raw, _entry("arc_easy"))]
            self.assertEqual([c["assistant"] for c in converted], ["B) y", "A) p"])


class NanochatRecipeParserTests(unittest.TestCase):
    def test_smol_smoltalk_keeps_the_chat_and_its_source(self):
        chat = {"messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello."}], "source": "smol-rewrite"}
        row = _convert("smol_smoltalk", chat)[0]
        self.assertEqual(([m["role"] for m in row["messages"]], row["intent"], row["metadata"]), (["system", "user", "assistant"], "smol-rewrite", {"source": "smol-rewrite"}))
        self.assertTrue(row["sourceId"].startswith("smol_smoltalk:") and row["groupId"] == row["sourceId"])
        self.assertEqual(_convert("smol_smoltalk", {**chat, "source": ""})[0]["intent"], "conversation")
        self.assertEqual(_convert("smol_smoltalk", {"messages": [], "source": "x"}), [])
        self.assertEqual(_convert("smol_smoltalk", {"messages": [{"role": "assistant", "content": "Hello."}]}), [])

    def test_mmlu_auxiliary_unwraps_the_train_column_into_lettered_choices(self):
        inner = {"answer": 1, "choices": ["Adams only.", "Brooks only.", "Case only."], "question": " Who was hurt? ", "subject": ""}
        row = _convert("mmlu_auxiliary", {"train": inner})[0]
        self.assertEqual(row["assistant"], "B) Brooks only.")
        self.assertTrue(row["user"].endswith("Question: Who was hurt?\n\nA) Adams only.\nB) Brooks only.\nC) Case only."))
        self.assertEqual((row["intent"], row["sourceId"]), ("multiple_choice", row["groupId"]))
        self.assertEqual(_convert("mmlu_auxiliary", inner)[0]["assistant"], "B) Brooks only.")  # a flat row works too
        for bad in ({**inner, "answer": 3}, {**inner, "answer": -1}, {**inner, "answer": None}, {**inner, "question": " "}, {**inner, "choices": ["a", ""]}, {**inner, "choices": ["a"]}):
            self.assertEqual(_convert("mmlu_auxiliary", {"train": bad}), [])

    def test_mmlu_auxiliary_reads_the_real_parquet_layout(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        inner = {"answer": 0, "choices": ["x", "y"], "question": "Q?", "subject": ""}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.parquet"
            pq.write_table(pa.table({"train": pa.array([inner])}), path)
            rows = list(source_records(path, _entry("mmlu_auxiliary")))
        self.assertEqual([r["assistant"] for raw in rows for r in convert_record(raw, _entry("mmlu_auxiliary"))], ["A) x"])


class LanguageCorpusSizeTests(unittest.TestCase):
    def test_replacement_catalog_language_text_is_fineweb_edu_and_cosmopedia(self):
        defaults = {e["id"] for e in CATALOG if e["stage"] == "language" and e["default"]}
        self.assertEqual(defaults, {"fineweb_edu_00", "fineweb_edu_01", "fineweb_edu_02", "fineweb_edu_03", "cosmopedia_khan", "cosmopedia_openstax"})

    def test_whole_shards_are_unbounded_pinned_hashed_and_licensed(self):
        for catalog_id in ("fineweb_edu_00", "fineweb_edu_07", "cosmopedia_openstax", "smol_smoltalk", "mmlu_auxiliary", "fineweb_edu", "openwebmath", "openwebmath_b", "wikipedia_en_a", "wikipedia_en_b", "gutenberg", "gutenberg_b"):
            with self.subTest(catalog_id):
                entry = _entry(catalog_id)
                self.assertNotIn("max_records", entry)
                self.assertRegex(entry["revision"], r"^[0-9a-f]{40}$")
                self.assertRegex(entry["sha256"], r"^[0-9a-f]{64}$")
                self.assertTrue(entry["license"])
                self.assertTrue(entry["source_url"].startswith("https://huggingface.co/datasets/"))

    def test_english_wikipedia_groups_by_its_own_entry_id(self):
        article = _convert("wikipedia_en_a", {"id": "12", "title": "Anarchism", "text": "word " * 60})[0]
        self.assertEqual((article["groupId"], article["sourceId"]), ("wikipedia_en_a:12", "12"))
        self.assertEqual(_convert("simplewiki", {"id": "12", "title": "T", "text": "word " * 60})[0]["groupId"], "simplewiki:12")


if __name__ == "__main__":
    unittest.main()
