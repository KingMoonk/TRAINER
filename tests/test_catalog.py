"""Offline checks for large-corpus storage, provenance and atomic downloads."""
import io
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from studio_catalog import (CATALOG, STAGES, DownloadCancelled, _fetch, clean_wikitext, convert_record, default_catalog_ids,
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
        entry = next(e for e in CATALOG if e["id"] == "squad2")
        base = {"context": "Aster is a blue star.", "question": "What color?", "id": "one", "answers": [{"text": "blue"}], "is_impossible": False}
        answer = list(convert_record(base, entry))[0]
        unknown = list(convert_record({**base, "id": "two", "question": "How old?", "is_impossible": True, "answers": []}, entry))[0]
        self.assertEqual(answer["assistant"], "blue")
        self.assertIn(base["context"], answer["user"])
        self.assertEqual(answer["groupId"], unknown["groupId"])
        self.assertIn("does not contain", unknown["assistant"])

    def test_openassistant_preserves_branches_and_filters_rejected_subtrees(self):
        entry = next(e for e in CATALOG if e["id"] == "oasst1")
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
        entry = next(e for e in CATALOG if e["id"] == "dolly15k")
        with tempfile.TemporaryDirectory() as directory, patch("studio_catalog._fetch", side_effect=self.fake_fetch), \
                patch.dict(entry, {"sha256": "0" * 64}):
            with self.assertRaisesRegex(ValueError, "pinned checksum"):
                download_dataset("dolly15k", root=directory)
            self.assertFalse((Path(directory) / "data" / "downloads" / "dolly15k" / "source.json").exists())

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
    return next(e for e in CATALOG if e["id"] == catalog_id)


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

    def test_default_ids_cover_the_original_core_and_stay_in_stage_order(self):
        defaults = default_catalog_ids()
        for legacy in ("dolly15k", "squad2", "codealpaca20k", "gsm8k", "oasst1"):
            self.assertIn(legacy, defaults)
        self.assertNotIn("wizard143k", defaults)
        self.assertIn("tinystories", defaults)
        stages = [STAGES.index(_entry(i)["stage"]) for i in defaults]
        self.assertEqual(stages, sorted(stages))
        self.assertEqual({e["stage"] for e in CATALOG}, set(STAGES))

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

    def test_problem_sources_default_except_unlicensed_and_enrichment_parsers(self):
        problems = [e for e in CATALOG if e["stage"] == "problems"]
        self.assertTrue(all(e["parser"].endswith("_check") for e in problems))
        self.assertFalse(_entry("winogrande_check")["default"])
        self.assertGreaterEqual(sum(e["default"] for e in problems), 10)
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


if __name__ == "__main__":
    unittest.main()
