"""GH 18 items 6-9 and 30: builder copy, version names, English-only OASST, HellaSwag punctuation, rule-sized token budgets."""
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import studio_rl
import studio_rules
from studio_catalog import CATALOG, FORMAT_VERSION, convert_record, download_dataset, file_sha256, list_downloads, refresh_downloads, register_download
from studio_core import Workspace
from studio_ia.copy import BUILDER
from studio_ia import design as D
from studio_ia.pages import builder as B
from studio_ia.services import Job
from studio_model import LocalModel, train


def entry(catalog_id):
    return next(e for e in CATALOG if e["id"] == catalog_id)


def tree(tree_id, lang, text="hello"):
    return {"message_tree_id": tree_id, "prompt": {"role": "prompter", "lang": lang, "text": text, "replies": [
        {"role": "assistant", "lang": lang, "text": "hi", "message_id": tree_id + "-a"}]}}


class BuilderCopy(unittest.TestCase):
    def test_6_memory_verdict_is_not_a_placeholder(self):
        verdict = BUILDER["training_memory_fits"]
        self.assertNotIn("[", verdict)
        self.assertIn("unknown", verdict)
        with patch("os.sysconf", side_effect=OSError, create=True):  # the case the text covers: this computer does not report its memory
            self.assertIsNone(D.fits({}))

    def test_7_no_version_yet_is_not_version_0(self):
        self.assertEqual(B.version_name(0), "no version yet")
        self.assertEqual(B.version_name(3), "version 3")
        job = Job("j", "pretrain", "Train", "train", {})
        self.assertNotIn("0", B.pill_text(job, None))
        self.assertEqual(B.pill_text(job, None), "Training (no version yet)")
        self.assertEqual(B.pill_text(Job("j", "pretrain", "Train", "train", {"version": 2, "mode": "from_scratch"}), None), "Version 2 is training")


class Converters(unittest.TestCase):
    def test_8_oasst_keeps_english_trees_only(self):
        self.assertEqual(list(convert_record(tree("t-de", "de"), entry("oasst1"))), [])
        self.assertEqual(list(convert_record(tree("t-es", "es"), entry("oasst1"))), [])
        english = list(convert_record(tree("t-en", "en"), entry("oasst1")))
        self.assertEqual(len(english), 1)
        self.assertEqual(english[0]["groupId"], "oasst1:t-en")

    def test_30_hellaswag_text_is_capitalised_and_punctuated(self):
        raw = {"activity_label": "Cooking", "ctx": "he stirs the pot", "label": "2", "source_id": "h1",
               "endings": ["then he tastes it", "the soup is ready.", "what a mess!", "he says \"done\""]}
        rows = list(convert_record(raw, entry("hellaswag_check")))
        user = rows[0]["user"]
        self.assertIn("Cooking: He stirs the pot", user)
        for line in ("A) Then he tastes it.", "B) The soup is ready.", "C) What a mess!", 'D) He says "done".'):
            self.assertIn(line, user)
        self.assertEqual(rows[0]["assistant"], "C) What a mess!")


class RefreshDownloads(unittest.TestCase):
    @staticmethod
    def fake_fetch(url, path, stop=None, emit=None, maximum=None):
        if Path(path).suffix == ".gz":
            payload = gzip.compress("".join(json.dumps(t) + "\n" for t in (tree("t-en", "en"), tree("t-de", "de", "hallo"))).encode())
        else:
            payload = b"Original dataset attribution."
        Path(path).write_bytes(payload)
        return {"url": url, "filename": Path(path).name, "bytes": len(payload), "sha256": file_sha256(path)}

    def test_8_and_30_refresh_rebuilds_old_downloads_once(self):
        with tempfile.TemporaryDirectory() as directory, patch("studio_catalog._fetch", side_effect=self.fake_fetch) as fetch:
            result = download_dataset("oasst1", root=directory)
            self.assertEqual(result["format_version"], FORMAT_VERSION)
            self.assertEqual(result["records"], 1)
            # An older app converted this download: its manifest says version 2 and the file still holds the German tree.
            folder = Path(directory) / "data" / "downloads" / "oasst1"
            rows = [json.loads(line) for line in (folder / "records.jsonl").read_text(encoding="utf-8").splitlines()]
            german = {**rows[0], "sourceId": "t-de-a", "groupId": "oasst1:t-de", "messages": [{"role": "user", "content": "hallo"}, {"role": "assistant", "content": "hi"}]}
            (folder / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows + [german]), encoding="utf-8")
            old = {**result, "format_version": 2, "records": 2, "bytes": (folder / "records.jsonl").stat().st_size,
                   "sha256": file_sha256(folder / "records.jsonl")}
            (folder / "manifest.json").write_text(json.dumps(old), encoding="utf-8")
            workspace = Workspace(directory)
            dataset_id = register_download(workspace, list_downloads(directory)[0])
            self.assertEqual(workspace.count(dataset_id), 2)
            fetches = fetch.call_count

            self.assertEqual(refresh_downloads(workspace), ["oasst1"])
            self.assertEqual(list_downloads(directory)[0]["format_version"], FORMAT_VERSION)
            self.assertEqual(workspace.count(dataset_id), 1)
            self.assertEqual(fetch.call_count, fetches)  # rebuilt from the cached source, nothing downloaded

            self.assertEqual(refresh_downloads(workspace), [])  # idempotent: every manifest is current now
            self.assertEqual(workspace.count(dataset_id), 1)

    def test_refresh_skips_downloads_without_a_cached_source_and_does_not_register_new_ones(self):
        with tempfile.TemporaryDirectory() as directory, patch("studio_catalog._fetch", side_effect=self.fake_fetch):
            result = download_dataset("oasst1", root=directory)
            folder = Path(directory) / "data" / "downloads" / "oasst1"
            (folder / "manifest.json").write_text(json.dumps({**result, "format_version": 2}), encoding="utf-8")
            workspace = Workspace(directory)
            self.assertEqual(refresh_downloads(workspace), ["oasst1"])  # rebuilt, but never imported by the person
            self.assertIsNone(workspace._dataset("catalog_oasst1"))
            (folder / "manifest.json").write_text(json.dumps({**result, "format_version": 2}), encoding="utf-8")
            (folder / Path(entry("oasst1")["filename"]).name).unlink()
            self.assertEqual(refresh_downloads(workspace), [])  # no source on disk: left alone, never fetched

    def test_refresh_downloads_is_a_noop_on_the_service_protocol(self):
        from studio_ia.services import FakeServices
        self.assertEqual(FakeServices().refresh_downloads(), [])


class TokenBudgets(unittest.TestCase):
    def test_9_budget_comes_from_the_rule(self):
        for rule in ("copy", "reverse", "sort", "count", "small_addition", "final_number", "choice"):
            self.assertEqual(studio_rules.max_tokens(rule), 32, rule)
        for rule in ("ends_cleanly", "max_length", "something_new", None):
            self.assertEqual(studio_rules.max_tokens(rule), studio_rules.DEFAULT_MAX_TOKENS, rule)
        self.assertEqual(studio_rules.DEFAULT_MAX_TOKENS, 96)

    def test_9_rounds_and_measurement_use_the_same_budgets(self):
        tiny = dict(device="cpu", width=32, layers=1, heads=2, context=32, expansion=2, dropout=0, batch_size=4, steps=2,
                    learning_rate=.004, warmup=0, validation=0, eval_every=10, objective="assistant")
        with tempfile.TemporaryDirectory() as directory:
            train(tiny, [{"user": "Copy: yes", "assistant": "yes"}] * 8, Path(directory) / "base")
            checkpoint = Path(directory) / "base" / "last.npz"
            seen = []
            real = LocalModel.generate

            def spy(self, messages, max_tokens=32, **kw):
                seen.append(max_tokens)
                return real(self, messages, max_tokens=min(max_tokens, 2), **kw)  # the budget is recorded; answers stay short and fast
            with patch.object(LocalModel, "generate", spy):
                for rule, prompt in (("copy", "Copy: yes"), ("ends_cleanly", "Write one sentence about tides.")):
                    seen.clear()
                    studio_rl.practice_with_checks(checkpoint, [{"prompt": prompt, "rule": rule}], Path(directory) / rule,
                                                   group=2, steps=1, prompts_per_step=1)
                    self.assertEqual(set(seen), {studio_rules.max_tokens(rule)}, rule)
                    seen.clear()
                    studio_rl.measure(checkpoint, [], [{"prompt": prompt, "rule": rule}])
                    self.assertEqual(seen, [studio_rules.max_tokens(rule)], rule)


if __name__ == "__main__":
    unittest.main()
