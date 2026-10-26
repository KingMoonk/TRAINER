"""Decision logic of the training pages, window-free."""
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import studio_gate
from studio_core import Workspace
from studio_data_kinds import add_problems
from studio_ia.pretraining_plan import estimated_pieces, size_advice, enough_text, held_out, ratio_text, source_action, source_status, can_continue, forced_mode, word_list_changed
from studio_ia.services import WorkspaceServices
from studio_spec import default_spec

BYTE, BPE = {"id": "byte", "config": {}}, {"id": "bpe_gpt2", "config": {}}


def version(tokenizer, checkpoint=None):
    return SimpleNamespace(spec={"tokenizer": tokenizer}, best_checkpoint=checkpoint)


class PretrainingMode(unittest.TestCase):
    def test_same_word_list_continues(self):
        v, parent = version(BYTE), version(BYTE, "p.npz")
        self.assertTrue(can_continue(v, parent))
        self.assertEqual(forced_mode(v, parent, "catch_up"), "catch_up")

    def test_word_list_change_forces_scratch(self):
        v, parent = version(BPE), version(BYTE, "p.npz")
        self.assertTrue(word_list_changed(v, parent))
        self.assertEqual(forced_mode(v, parent, "catch_up"), "from_scratch")

    def test_untrained_or_missing_parent_forces_scratch(self):
        self.assertEqual(forced_mode(version(BYTE), version(BYTE), "catch_up"), "from_scratch")
        self.assertEqual(forced_mode(version(BYTE), None, "catch_up"), "from_scratch")

    def test_scratch_is_always_allowed(self):
        self.assertEqual(forced_mode(version(BYTE), version(BYTE, "p.npz"), "from_scratch"), "from_scratch")


class RecommendedCollection(unittest.TestCase):
    @staticmethod
    def fake_fetch(url, path, stop=None, emit=None, maximum=None, range_bytes=None):
        from pathlib import Path
        from studio_catalog import file_sha256
        if Path(path).suffix == ".txt":
            payload = b"a first story.<|endoftext|>a second story.<|endoftext|>"
        elif Path(path).suffix == ".jsonl":
            payload = b'{"instruction":"Say hello","response":"Hello","context":"","category":"greeting"}\n'
        else:
            payload = b"attribution"
        Path(path).write_bytes(payload)
        return {"url": url, "filename": Path(path).name, "bytes": len(payload), "sha256": file_sha256(path)}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.svc = WorkspaceServices(Workspace(self.tmp.name))
        self.addCleanup(self.svc.close)
        self.events = []
        self.svc.watch(lambda kind, payload: self.events.append((kind, payload)))

    def states(self, kind="text"):
        return {r["id"]: r["state"] for r in self.svc.collection(kind)}

    def test_row_labels(self):
        self.assertEqual(source_action({"state": "not downloaded"})[1], "download")
        self.assertEqual(source_action({"state": "downloaded"})[1], "import")
        self.assertEqual(source_action({"state": "downloading"}), (None, None))
        self.assertEqual(source_action({"state": "imported"}), (None, None))
        self.assertEqual(source_status({"state": "downloading"}, "Downloading x · 1.0 MB"), "Downloading x · 1.0 MB")
        self.assertIn("Could not finish: no network.", source_status({"state": "not downloaded", "error": "no network"}))

    def test_download_then_import_into_the_conversation_sets(self):
        from unittest.mock import patch
        self.assertEqual(set(self.states("conversations").values()), {"not downloaded"})
        with patch("studio_catalog._fetch", side_effect=self.fake_fetch):
            self.assertTrue(self.svc.download_sources(["dolly15k"]))
            self.svc._worker.join(10)
            self.assertEqual(self.states("conversations")["dolly15k"], "downloaded")
            self.assertIn(("download_done", {"id": "dolly15k"}), self.events)
            self.assertFalse(self.svc.import_source("wikitext2"))  # not downloaded
            self.assertTrue(self.svc.import_source("dolly15k"))
            self.svc._worker.join(10)
        self.assertEqual(self.states("conversations")["dolly15k"], "imported")
        self.assertEqual(self.svc.data("conversations").count, 1)

    def test_text_downloads_and_adds_in_one_step(self):
        from unittest.mock import patch
        with patch("studio_catalog._fetch", side_effect=self.fake_fetch):
            self.assertTrue(self.svc.download_sources(["tinystories"], install=True))
            self.svc._worker.join(10)
        self.assertEqual(self.states()["tinystories"], "imported")
        self.assertEqual(self.svc.data("text").count, 2)

    def test_validation_and_test_splits_are_held_out(self):
        self.assertTrue(held_out("catalog_tinystories_valid"))
        self.assertFalse(held_out("catalog_tinystories"))
        self.assertFalse(held_out("d1"))  # the person's own set

    def test_collections_by_kind_and_problem_install(self):
        from pathlib import Path
        from unittest.mock import patch
        import studio_catalog as cat
        stage = {e["id"]: e["stage"] for e in cat.CATALOG}
        self.assertEqual(self.svc.collection_ids("text"), ["fineweb_edu_00", "fineweb_edu_01", "fineweb_edu_02", "fineweb_edu_03", "cosmopedia_khan", "cosmopedia_v2"])
        self.assertEqual(self.svc.collection_ids("problems"), [])
        self.assertNotIn("winogrande_check", self.svc.collection_ids("problems"))
        self.assertEqual(set(self.svc.collection_ids("conversations")), {"mmlu_auxiliary", "gsm8k", "smol_smoltalk", "everyday_conversations", "smoltalk_systemchats", "toolace"})
        self.assertTrue(all(stage[catalog_id] != "language" for catalog_id in self.svc.collection_ids("conversations")))

        def fetch(url, path, stop=None, emit=None, maximum=None, range_bytes=None):
            if Path(path).name == "train.jsonl":
                Path(path).write_text('{"question": "What is 2 + 3?", "answer": "2 + 3 = 5\\n#### 5"}\n', encoding="utf-8")
                return {"url": url, "filename": "train.jsonl", "bytes": 1, "sha256": cat.file_sha256(path)}
            return self.fake_fetch(url, path, stop, emit, maximum, range_bytes)
        with patch("studio_catalog._fetch", side_effect=fetch):
            self.assertTrue(self.svc.download_sources(["gsm8k_check"], kind="problems", install=True))
            self.svc._worker.join(10)
        self.assertEqual({r["id"]: r["state"] for r in self.svc.collection("problems")}["gsm8k_check"], "imported")
        self.assertEqual(self.svc.data("problems").count, 1)
        rows = self.svc.post_rows("problems") + self.svc.post_rows("heldout_problems")
        self.assertEqual([(r["rule"], r["params"]) for r in rows], [("final_number", {"answer": "5"})])

    def test_failure_is_reported_and_retryable(self):
        from unittest.mock import patch
        import urllib.error
        offline = "Could not reach the download server. Check the connection and try again."
        with patch("studio_catalog._fetch", side_effect=urllib.error.URLError("no network")):
            self.svc.download_sources(["dolly15k"])
            self.svc._worker.join(10)
        row = next(r for r in self.svc.collection("conversations") if r["id"] == "dolly15k")
        self.assertEqual((row["state"], row["error"]), ("not downloaded", offline))
        self.assertIn(("download_failed", {"id": "dolly15k", "error": offline}), self.events)
        with patch("studio_catalog._fetch", side_effect=self.fake_fetch):
            self.assertTrue(self.svc.download_sources(["dolly15k"]))  # nothing left running
            self.svc._worker.join(10)
        self.assertEqual(self.states("conversations")["dolly15k"], "downloaded")


class GateRules(unittest.TestCase):
    def test_rise_inside_the_noise_band_passes_beyond_it_fails(self):
        before, band = {"bpc": 2.0, "pass_k": 3, "pass_n": 5}, (1.95, 2.1)
        self.assertTrue(studio_gate.passes(before, {"bpc": 2.05, "pass_k": 3, "pass_n": 5}, band)[0])
        self.assertFalse(studio_gate.passes(before, {"bpc": 2.2, "pass_k": 3, "pass_n": 5}, band)[0])
        self.assertFalse(studio_gate.passes(before, {"bpc": 2.05, "pass_k": 3, "pass_n": 5})[0])
        self.assertFalse(studio_gate.passes(before, {"bpc": 1.9, "pass_k": 2, "pass_n": 5}, band)[0])  # pass rate fell

    def test_chat_moves_only_when_the_paired_test_says_better(self):
        better, unsure = {"better": True, "diff": -.1, "upper": -.02}, {"better": False, "diff": -.1, "upper": .02}
        self.assertTrue(studio_gate.moves_chat(1.0, None, None))
        self.assertFalse(studio_gate.moves_chat(1.97, 2.0, unsure))   # lower, but not sure it is more than luck
        self.assertTrue(studio_gate.moves_chat(1.9, 2.0, better))
        self.assertFalse(studio_gate.moves_chat(None, 2.0, better))
        self.assertTrue(studio_gate.moves_chat(1.97, 2.0))             # no held-back text to pair on: lower is enough
        self.assertFalse(studio_gate.moves_chat(2.03, 2.0))


class WorkspaceRows(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ws = Workspace(self.tmp.name)
        self.svc = WorkspaceServices(self.ws)
        self.addCleanup(self.svc.close)
        self.text = self.ws.create_dataset("stories")
        self.ws.add(self.text, [{"text": f"story number {i} " * 5} for i in range(20)])
        self.conv = self.ws.create_dataset("talk")
        self.ws.add(self.conv, [{"user": f"q{i}", "assistant": f"a{i}"} for i in range(30)])
        self.prob = self.ws.create_dataset("problems")
        add_problems(self.ws, self.prob, _problems())

    def test_waiting_jobs_from_an_earlier_session_start_again(self):
        from unittest import mock
        import studio_queue
        with mock.patch.object(studio_queue.Runner, "start") as start:
            self.svc.start_queue()
            start.assert_not_called()  # nothing waiting: the first enqueue starts it
            self.svc.runner.jobs.append(studio_queue.Job("quicktest", "left over", status="held"))
            self.svc.start_queue()
            start.assert_not_called()  # held jobs do not count
            self.svc.runner.jobs.append(studio_queue.Job("quicktest", "left over"))
            self.svc.start_queue()
            start.assert_called_once()

    def test_preview_is_cached_on_disk_and_never_computed_by_the_cached_read(self):
        self.svc.save_version(default_spec(), "first design")
        self.assertIsNone(self.svc.cached_preview([self.text], rows=6))  # unknown yet: no work done to find out
        preview = self.svc.pretrain_preview([self.text], rows=6)
        self.assertGreater(preview["pieces"], 0)
        again = WorkspaceServices(Workspace(self.tmp.name))  # a new session, as after a restart
        self.addCleanup(again.close)
        self.assertEqual(again.cached_preview([self.text], rows=6)["pieces"], preview["pieces"])
        self.ws.add(self.conv, [{"user": "another question", "assistant": "another answer"}])   # a set not chosen
        self.assertEqual(self.svc.cached_preview([self.text], rows=6)["pieces"], preview["pieces"])
        self.ws.add(self.text, [{"text": "a new story changes the stored data"}])  # a newer rev: the old measurement serves, with the live count
        after = self.svc.cached_preview([self.text], rows=6)
        self.assertGreater(after["pieces"], preview["pieces"])
        self.assertTrue(after["estimated"])

    def test_a_set_of_unreviewed_generated_text_is_reported_as_having_nothing_to_train_on(self):
        drafts = self.ws.create_dataset("generated stories")
        self.ws.add(drafts, [{"text": f"a generated tale {i} " * 6, "synthetic": True, "quality": "Review"} for i in range(12)])
        self.svc.save_version(default_spec(), "first design")
        self.assertEqual(self.svc.no_usable_text([drafts]), [])  # not measured yet: nothing is known
        self.svc.pretrain_preview([drafts], rows=6)
        self.assertEqual(self.svc.no_usable_text([drafts]), [("generated stories", 12)])
        self.assertEqual(self.svc.no_usable_text([self.text]), [])
        self.svc.pretrain_preview([self.text], rows=6)
        self.assertEqual(self.svc.no_usable_text([self.text, drafts]), [("generated stories", 12)])

    def test_approving_a_catalogue_set_forgets_its_empty_measurement_so_it_is_measured_again(self):
        drafts = self.ws.create_dataset("tales", storage="sqlite")
        self.ws.add(drafts, [{"text": f"a catalogue tale {i} " * 6, "synthetic": True, "quality": "Review"} for i in range(12)])
        self.ws._dataset(drafts)["catalogId"] = "tinystories"
        self.svc.save_version(default_spec(), "first design")
        self.svc.pretrain_preview([drafts], rows=6)
        self.assertEqual(self.svc.no_usable_text([drafts]), [("tales", 12)])
        self.assertEqual(self.svc.approve_catalog_sets(), [drafts])
        self.assertIsNone(self.svc.cached_preview([drafts], rows=6))  # forgotten: it is measured again, not read as empty
        self.assertGreater(self.svc.pretrain_preview([drafts], rows=6)["pieces"], 0)
        self.assertEqual(self.svc.no_usable_text([drafts]), [])

    def test_any_combination_of_measured_sets_is_served_without_the_tokenizer(self):
        from unittest import mock
        import studio_spec
        other = self.ws.create_dataset("more stories")
        self.ws.add(other, [{"text": f"another tale {i} " * 9} for i in range(10)])
        self.svc.save_version(default_spec(), "first design")
        first, second = (self.svc.pretrain_preview([i], rows=6) for i in (self.text, other))
        module, _ = studio_spec.module_of("tokenizer", studio_spec.validate_spec(default_spec())["tokenizer"])
        with mock.patch.object(module, "fit", side_effect=AssertionError("fit")), mock.patch.object(self.svc, "measure_set", side_effect=AssertionError("measure")):
            both = self.svc.cached_preview([self.text, other], rows=6)
            self.assertEqual(self.svc.pretrain_preview([other, self.text], rows=6)["pieces"], both["pieces"])
        self.assertAlmostEqual(both["pieces"], first["pieces"] + second["pieces"], delta=2)

    def test_the_precacher_measures_each_text_set_once_and_waits_for_a_busy_set_to_settle(self):
        self.svc.save_version(default_spec(), "first design")
        self.assertEqual(self.svc.precache_set_sizes(now=1000), [self.text])
        self.assertEqual(self.svc.precache_set_sizes(now=1001), [])  # measured: nothing to do
        self.assertIsNotNone(self.svc.cached_preview([self.text], rows=6))
        self.ws.add(self.text, [{"text": "the collector adds another"}])  # rev up: seen now, so it is not yet redone
        self.assertEqual(self.svc.precache_set_sizes(now=2000), [])
        self.assertEqual(self.svc.precache_set_sizes(now=2000 + 601), [self.text])
        self.assertEqual(self.svc.precache_set_sizes(now=3000), [])

    def test_text_size_is_fast_and_measured_once_a_preview_has_counted(self):
        spec = default_spec()
        self.svc.save_version(spec, "first design")
        first = self.svc.text_size([self.text], spec)
        self.assertEqual(first["characters"], sum(len((f"story number {i} " * 5).strip()) for i in range(20)))
        self.assertGreater(first["params"], 0)
        self.assertIsNone(first["chars_per_piece"])
        self.assertIsNone(self.svc.cached_preview([self.text], rows=6))  # reading the size did no tokenizing
        preview = self.svc.pretrain_preview([self.text], rows=6)
        self.assertEqual(self.svc.text_size([self.text], spec)["chars_per_piece"], preview["chars_per_piece"])

    def test_preview_normalizes_legacy_specs_without_changing_the_version(self):
        version = self.svc.save_version(default_spec(), "first design")
        for sparse, context in (({"tokenizer": BYTE}, 256), ({"dims": {"n_ctx": 128}}, 128)):
            with self.subTest(spec=sparse):
                version.spec = sparse
                with mock.patch.object(self.svc, "current_version", return_value=version):
                    preview = self.svc.pretrain_preview([self.text], rows=6)
                    self.assertEqual(preview["context"], context)
                    self.assertGreater(preview["pieces"], 0)
                    self.assertGreater(preview["params"], 0)
                    self.assertEqual(self.svc.cached_preview([self.text], rows=6)["context"], context)
                self.assertEqual(version.spec, sparse)
                self.assertNotIn("d_model", sparse.get("dims", {}))

    def test_held_back_text_is_fixed_and_never_trained_on(self):
        train = self.svc.examples({"datasets": [self.text], "heldout_fraction": .2})
        held = set(self.svc._heldout())
        self.assertTrue(held)
        self.assertFalse(held & {e["text"] for e in train})
        self.assertEqual(len(train) + len(held), 20)
        again = self.svc.examples({"datasets": [self.text], "heldout_fraction": .5})
        self.assertEqual(set(self.svc._heldout()), held)  # fixed on first use
        self.assertEqual(len(again), len(train))

    def test_the_stream_gives_the_rows_examples_gives_one_at_a_time(self):
        first = self.svc.example_stream({"datasets": [self.text], "heldout_fraction": .2})  # no held-back file yet: examples() makes it
        listed = self.svc.examples({"datasets": [self.text]})
        self.assertEqual(list(first), listed)
        stream = self.svc.example_stream({"datasets": [self.text]})
        self.assertEqual(stream.expected_characters, sum(len(e["text"]) for e in listed))
        self.assertEqual(list(stream), listed)
        self.ws.add(self.text, [{"text": "an unreviewed row", "quality": "Review"}])
        self.assertEqual([e["text"] for e in self.svc.example_stream({"datasets": [self.text], "reviewed_only": True})],
                         [e["text"] for e in self.svc.examples({"datasets": [self.text], "reviewed_only": True})])

    def three_sets(self):
        """Three text sets whose rows are 100, 200 and 400 characters long, 30 rows each: shares of 1/7, 2/7 and 4/7."""
        ids = []
        for name, size in (("short", 100), ("middle", 200), ("long", 400)):
            ids.append(self.ws.create_dataset(name))
            self.ws.add(ids[-1], [{"text": (f"{name} {i} " * 100)[:size]} for i in range(30)])
        self.svc.examples({"datasets": ids})  # fixes the held-back split once
        return ids

    def test_a_budget_smaller_than_the_selection_draws_from_every_set_by_share(self):
        import studio_corpus
        ids = self.three_sets()
        total = self.svc.example_stream({"datasets": ids}).expected_characters
        token_budget = total // 2 // studio_corpus.CHARS_PER_TOKEN
        read, taken = 0, {"short": 0, "middle": 0, "long": 0}
        for example in self.svc.example_stream({"datasets": ids, "token_budget": token_budget}):
            taken[example["text"].split()[0]] += len(example["text"])
            read += len(example["text"])
            if read >= studio_corpus.budget_chars(token_budget):
                break
        sizes = {name: sum(len(e["text"]) for e in self.svc.examples({"datasets": [i]})) for name, i in zip(taken, ids)}
        for name in taken:
            self.assertAlmostEqual(taken[name] / read, sizes[name] / sum(sizes.values()), delta=.1, msg=name)
            self.assertGreater(taken[name], 0, name)

    def test_without_a_budget_the_rows_stay_in_dataset_order(self):
        ids = self.three_sets()
        in_order = [e["text"] for i in ids for e in self.svc.examples({"datasets": [i]})]
        self.assertEqual([e["text"] for e in self.svc.example_stream({"datasets": ids})], in_order)
        roomy = [e["text"] for e in self.svc.example_stream({"datasets": ids, "token_budget": 10 ** 9})]
        self.assertEqual(roomy, in_order)  # a budget the text does not exceed changes nothing

    @staticmethod
    def _words(seed, count=40):
        import random
        pick = random.Random(seed)
        return " ".join(f"w{pick.randrange(10 ** 6)}" for _ in range(count))

    def test_a_big_text_is_split_as_a_stream_and_a_small_one_as_a_list(self):
        import studio_eval
        small = self.svc.examples({"datasets": [self.text], "heldout_fraction": .2})
        listed = self.svc._heldout()
        self.assertEqual(len(small) + len(listed), 20)
        (self.svc.root / "data" / "heldout.json").unlink()
        self.svc._heldout_hit = (None,)
        with mock.patch.object(studio_eval, "heldout_split", side_effect=AssertionError("listed")), \
                mock.patch("studio_model.STREAM_CHARS", 100):
            streamed = self.svc.examples({"datasets": [self.text], "heldout_fraction": .2})
        held = set(self.svc._heldout())
        self.assertTrue(held)
        self.assertFalse(held & {e["text"] for e in streamed})
        self.assertEqual(len(streamed) + len(held), 20)
        self.assertEqual(streamed, self.svc.examples({"datasets": [self.text], "heldout_fraction": .5}))  # fixed once made

    def test_the_streamed_split_still_leaves_out_a_held_back_text_that_shares_a_13_gram_with_training(self):
        import studio_decontam
        import studio_eval
        rows = [self._words(n) for n in range(300)]
        held_of = lambda texts: set(studio_eval.streamed_heldout_split(texts, studio_decontam.Index(), .2))
        held = held_of(rows)
        self.assertTrue(held)
        victim = next(t for t in rows if t in held)
        phrase = " ".join(victim.split()[5:25])
        for k in range(100):  # a training row that quotes the victim (and, by chance, still is not held back itself)
            quoting = f"{self._words(1000 + k, 10)} {phrase} tail{k}"
            if quoting not in held_of(rows + [quoting]):
                break
        data = [{"text": t} for t in rows + [quoting]]
        with mock.patch("studio_data_kinds.pretraining_examples", return_value=iter(data)), mock.patch("studio_model.STREAM_CHARS", 100):
            self.svc._fix_heldout({"datasets": [self.text], "heldout_fraction": .2})
        written = {d["text"] for d in __import__("json").loads((self.svc.root / "data" / "heldout.json").read_text(encoding="utf-8"))}
        self.assertNotIn(victim, written)
        self.assertEqual(written, held - {victim})

    def test_the_held_back_side_stops_at_the_cap_however_big_the_corpus(self):
        import json
        import studio_eval
        rows = [{"text": f"{n} " + "x" * 98} for n in range(2000)]  # 200 thousand characters, 5% held would be 100 documents
        with mock.patch("studio_data_kinds.pretraining_examples", return_value=iter(rows)), mock.patch("studio_model.STREAM_CHARS", 100),                 mock.patch.object(studio_eval, "HELDOUT_MAX_DOCS", 30):
            self.svc._fix_heldout({"datasets": [self.text], "heldout_fraction": .05})
        written = json.loads((self.svc.root / "data" / "heldout.json").read_text(encoding="utf-8"))
        self.assertEqual(len(written), 30)
        held = studio_eval.streamed_heldout_split((r["text"] for r in rows), __import__("studio_decontam").Index(), .5, max_chars=1000)
        self.assertTrue(0 < sum(map(len, held)) <= 1000)
        train, held = studio_eval.heldout_split([r["text"] for r in rows], .5, max_chars=1000)
        self.assertTrue(0 < sum(map(len, held)) <= 1000)
        self.assertEqual(len(held) + len(train), 2000)

    def test_two_example_streams_read_the_held_back_file_once(self):
        import json
        self.svc.example_stream({"datasets": [self.text], "heldout_fraction": .2})
        self.svc._heldout_hit = (None,)
        loads = mock.Mock(wraps=json.loads)
        with mock.patch("studio_ia.services.json.loads", loads):
            first = list(self.svc.example_stream({"datasets": [self.text]}))
            second = list(self.svc.example_stream({"datasets": [self.text]}))
        self.assertEqual(first, second)
        self.assertEqual(loads.call_count, 1)

    def test_the_first_split_of_a_big_text_holds_the_held_back_side_not_the_text(self):
        import json
        import tracemalloc
        count, size = 3000, 5000  # 15 million characters
        data = ({"text": f"{n} " + "x" * size} for n in range(count))
        with mock.patch("studio_data_kinds.pretraining_examples", return_value=data), mock.patch("studio_model.STREAM_CHARS", 1_000_000):
            tracemalloc.start()
            try:
                self.svc._fix_heldout({"datasets": [self.text], "heldout_fraction": .05})
                _, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
        written = json.loads((self.svc.root / "data" / "heldout.json").read_text(encoding="utf-8"))
        self.assertTrue(0 < len(written) < count // 5)
        self.assertLess(peak, 4 * 1024 ** 2)  # the whole text would hold 15 MB of strings

    def test_post_rows_keep_held_back_apart(self):
        train, held = self.svc.post_rows("problems"), self.svc.post_rows("heldout_problems")
        self.assertEqual(len(train) + len(held), 6)
        self.assertTrue(all(p["rule"] and p["prompt"] for p in train + held))
        convs = self.svc.post_rows("conversations")
        self.assertTrue(0 < len(convs) <= 30)

    def test_post_rows_leave_out_drafts_and_have_held_back_conversations(self):
        drafts = self.ws.create_dataset("drafts")
        self.ws.add(drafts, [{"user": f"draft q{i}", "assistant": f"draft a{i}", "synthetic": True, "quality": "Review"} for i in range(60)])
        self.ws.add(drafts, [{"user": f"good q{i}", "assistant": f"good a{i}", "synthetic": True, "quality": "Good"} for i in range(60)])
        everything = self.svc.post_rows("conversations") + self.svc.post_rows("heldout_conversations")
        self.assertFalse([e for e in everything if e["user"].startswith("draft")])
        self.assertTrue([e for e in everything if e["user"].startswith("good")])
        held = self.svc.post_rows("heldout_conversations")
        self.assertTrue(held)
        self.assertTrue(all(e["holdout"] for e in held))
        self.assertFalse(any(e["holdout"] for e in self.svc.post_rows("conversations")))

    def test_corrections_and_pairs_are_saved_for_rounds_3_and_4(self):
        messages = [{"role": "user", "content": "Largest planet?"}]
        self.svc.teach_correction(None, messages, "Jupiter")
        self.assertEqual(self.svc.post_rows("corrections")[0]["correction"], "Jupiter")
        self.svc.teach_correction(None, messages, "Jupiter is the largest.", whole_answer=True)
        self.assertGreater(self.svc.data("conversations").count, 30)
        self.svc.rate_pair(messages, {"version": 2, "reply": "Jupiter"}, {"version": 1, "reply": "Mars"})
        self.assertEqual(len(self.svc.post_rows("pairs")), 1)
        with self.assertRaises(ValueError):
            self.svc.rate_pair(messages, {"version": 2, "reply": "same"}, {"version": 1, "reply": "same"})

    def test_only_a_passed_gate_moves_the_checkpoint(self):
        self.svc.save_version(default_spec(), "first design")
        result = {"ok": False, "checks": [], "verdict": "v", "before": {"bpc": 2.}, "after": {"bpc": 2.5}, "checkpoint": "/x/last.npz"}
        self.svc.record_round(1, 1, result)
        self.assertIsNone(self.svc.versions()[0].best_checkpoint)
        state = self.svc.post_training(1)
        self.assertEqual((state["rounds"][1]["status"], state["rounds"][1]["note"]), ("done", "kept, not live"))
        self.svc.record_round(1, 1, {**result, "ok": True, "after": {"bpc": 1.5, "pass_k": 2, "pass_n": 4}})
        v = self.svc.versions()[0]
        self.assertTrue(v.best_checkpoint.endswith("last.npz"))
        self.assertEqual((v.scorecard["bpc"], v.scorecard["answers"], v.scorecard["problems"]), (1.5, True, [2, 4]))
        self.assertEqual(self.svc.post_training(1)["rounds"][2]["status"], "idle")

    def test_verdict_needs_finished_runs(self):
        self.svc.save_version(default_spec(), "a")
        self.svc.save_version(default_spec(), "b")
        res = self.svc.verdict(1, 2)
        self.assertFalse(res["comparable"])
        self.assertIsNone(res["percent"])


def _problems():
    import random

    from studio_rules import generate
    return generate("copy", random.Random(1), 6)


if __name__ == "__main__":
    unittest.main()


class SizeAdviceTest(unittest.TestCase):
    def verdict(self, ratio):
        return size_advice(ratio * 1_000_000, 1_000_000)[0]

    def test_thresholds(self):
        self.assertEqual([self.verdict(r) for r in (0.5, 9.9, 10, 20, 40, 40.1, 500)], ["few", "few", "right", "right", "right", "many", "many"])

    def test_target_is_the_size_that_makes_twenty_per_parameter(self):
        self.assertEqual(size_advice(200_000_000, 100_000_000), ("few", 2.0, 10_000_000))
        self.assertEqual(size_advice(2_000_000_000, 18_000_000)[2], 100_000_000)

    def test_pieces_are_a_quarter_of_the_characters_unless_measured(self):
        self.assertEqual(estimated_pieces(8_000_000), 2_000_000)
        self.assertEqual(estimated_pieces(8_000_000, 3.2), 2_500_000)


class EnoughTextTest(unittest.TestCase):
    def verdict(self, ratio):
        return enough_text(ratio)[1]

    def test_headline_follows_the_rule_of_thumb(self):
        self.assertEqual(self.verdict(2), "little")
        self.assertEqual(self.verdict(15.9), "little")
        self.assertEqual(self.verdict(16), "about_right")
        self.assertEqual(self.verdict(20), "about_right")
        self.assertEqual(self.verdict(30), "about_right")
        self.assertEqual(self.verdict(30.5), "plenty")
        self.assertEqual(self.verdict(549.54), "plenty")

    def test_far_above_the_rule_says_plenty_not_about_as_much(self):
        headline, _ = enough_text(549.54)
        self.assertIn("plenty", headline)
        self.assertIn("27", headline)
        self.assertNotIn("about as much", headline)
        self.assertIn("about 10.0", enough_text(2)[0])

    def test_ratio_reads_plainly(self):
        self.assertEqual(ratio_text(2.0), "2.00")
        self.assertEqual(ratio_text(27.46), "27.5")
        self.assertEqual(ratio_text(549.54), "about 550")
        self.assertEqual(ratio_text(12345.6), "about 12,000")
