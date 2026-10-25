"""The tokenized text of a run is cached per data split and tokenizer, so the next run on them tokenizes nothing."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import studio_diloco
import studio_model
import studio_shards
import studio_spec
from tests import diloco_support as support

BYTE = {"tokenizer": "byte"}
CHARACTERS = {"tokenizer": "character"}


class TokenizedCache(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cache = self.root / "cache"

    def build(self, name, config=None, examples=None, progress=None):
        coordinator = studio_diloco.Coordinator.build(name, self.root / name, {**support.CONFIG, **(config or {})}, examples or support.EXAMPLES,
                                                      support.settings(), shard_dir=self.cache, progress=progress)
        self.addCleanup(coordinator.release)
        return coordinator

    def calls(self, name, **options):
        """(fits, windows cut) a build makes, and the progress phases it reports."""
        seen = []
        with mock.patch.object(studio_spec, "module_of", wraps=studio_spec.module_of) as fit, \
                mock.patch.object(studio_model, "cut_windows", wraps=studio_model.cut_windows) as cut:
            coordinator = self.build(name, progress=lambda *step: seen.append(step), **options)
        return coordinator, fit.call_count, cut.call_count, [step[0] for step in seen]

    def test_the_second_build_of_the_same_data_and_tokenizer_tokenizes_nothing(self):
        first, fits, cuts, phases = self.calls("first", config=BYTE)
        self.assertEqual((fits, cuts > 0), (1, True))
        self.assertIn("tokenizing", phases)
        second, fits, cuts, phases = self.calls("second", config=BYTE)
        self.assertEqual((fits, cuts), (0, 0))
        self.assertIn("cached", phases)
        self.assertNotIn("tokenizing", phases)
        self.assertEqual(len(first.train), len(second.train))
        self.assertEqual(first.tokenizer.serialize(), second.tokenizer.serialize())

    def test_another_tokenizer_on_the_same_data_gets_its_own_cache_and_keeps_the_first(self):
        self.calls("a", config=BYTE)
        _, fits, cuts, _ = self.calls("b", config=CHARACTERS)
        self.assertEqual((fits, cuts > 0), (1, True))
        self.assertEqual(len(list(self.cache.glob("tok-*"))), 2)
        _, fits, cuts, _ = self.calls("c", config=BYTE)
        self.assertEqual((fits, cuts), (0, 0))

    def test_changed_data_misses_the_cache(self):
        self.calls("a", config=BYTE)
        changed = support.EXAMPLES + [{"text": "A new document the cache has not seen."}]
        _, fits, cuts, _ = self.calls("b", config=BYTE, examples=changed)
        self.assertEqual((fits, cuts > 0), (1, True))

    def test_a_corrupt_or_partial_cache_is_rebuilt(self):
        first, *_ = self.calls("a", config=BYTE)
        first.train.close()  # a Windows file cannot be replaced while the first run still maps it
        first.val.close()
        folder = next(self.cache.glob("tok-*"))
        (folder / "tokenizer.json").write_text("{not json", encoding="utf-8")
        for index in folder.glob("*/index.json"):
            index.write_text("{", encoding="utf-8")
        _, fits, cuts, phases = self.calls("b", config=BYTE)
        self.assertEqual((fits, cuts > 0), (1, True))
        self.assertIn("tokenizing", phases)
        _, fits, cuts, _ = self.calls("c", config=BYTE)
        self.assertEqual((fits, cuts), (0, 0))

    def test_nothing_half_written_is_left_behind(self):
        self.calls("a", config=BYTE)
        self.assertEqual([p for p in self.cache.rglob("*") if p.suffix == ".part"], [])
        self.assertTrue(studio_shards.load_tokenizer(next(self.cache.glob("tok-*"))))

    def test_runs_under_a_runs_folder_share_data_shards(self):
        self.assertEqual(studio_diloco.tokenized_root(self.root / "runs" / "queue_1"), self.root / "data" / "shards")
        self.assertEqual(studio_diloco.tokenized_root(self.root / "elsewhere"), self.root / "elsewhere" / "windows")
