"""Token shards: windows on disk read through np.memmap give the batches the in-memory windows give, and are cached by data and tokenizer."""
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import studio_cleanup
import studio_model
import studio_shards
from studio_model import MappedWindows, Tokenizer, batch_arrays, samples_from, shard_windows, train, validate_config

TRAINING = ([{"text": f"Document {n}: " + "the quick brown fox jumps over the lazy dog. " * (n % 7 + 1)} for n in range(30)]
            + [{"messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": f"What is {n} plus {n}?"},
                             {"role": "assistant", "content": f"{n + n}."}, {"role": "user", "content": "Again?"},
                             {"role": "assistant", "content": f"It is still {n + n}, " * (n % 5 + 1)}]} for n in range(30)])
VALIDATION = [{"user": f"Say {w}", "assistant": w} for w in ("yes", "no", "cat")]
TINY = dict(device="cpu", width=16, layers=1, heads=2, context=32, expansion=2, dropout=0, batch_size=2, steps=6, learning_rate=.004,
            warmup=0, validation=.2, eval_every=3, seed=7, sample_every=0)


def config(**changes):
    return validate_config({"context": 48, **changes})


def window_folders(root):
    """The folders of finished windows under a shard root: the token cache (tokens-*) sits beside them and is not one."""
    return [p for p in Path(root).iterdir() if p.is_dir() and not p.name.startswith("tokens-")]


def same_pairs(test, left, right):
    test.assertEqual(len(left), len(right))
    for i in range(len(left)):
        for a, b in zip(left[i], right[i]):
            np.testing.assert_array_equal(a, b)


class Shards(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)  # a Windows memory map keeps its file until it is collected
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.tokenizer = Tokenizer("byte")

    def test_round_trip_keeps_tokens_and_masks(self):
        for windowing in ("anchored", "sequential"):
            for objective in ("assistant", "all"):
                with self.subTest(windowing=windowing, objective=objective):
                    cfg = config(windowing=windowing, objective=objective)
                    expected = samples_from(TRAINING, self.tokenizer, cfg)
                    train_windows, val_windows, reused = shard_windows(TRAINING, VALIDATION, self.tokenizer, cfg, "split", self.root)
                    self.assertFalse(reused)
                    self.assertIsInstance(train_windows, MappedWindows)
                    same_pairs(self, expected, train_windows)
                    same_pairs(self, samples_from(VALIDATION, self.tokenizer, cfg), val_windows)
                    self.assertEqual({k: v for k, v in expected.describe().items() if k != "megabytes"},
                                     {k: v for k, v in train_windows.describe().items() if k != "megabytes"})
                    self.assertEqual(train_windows.offsets.dtype, np.int64)
                    self.assertEqual(train_windows.tokens.dtype, np.uint16)

    def test_a_mask_is_stored_only_when_some_positions_are_not_targets(self):
        masked = shard_windows(TRAINING, VALIDATION, self.tokenizer, config(objective="assistant"), "a", self.root)[0]
        everything = shard_windows(TRAINING, VALIDATION, self.tokenizer, config(objective="all", windowing="sequential"), "b", self.root)[0]
        self.assertLess(masked.describe()["targets"], len(masked.flags))
        self.assertEqual(everything.describe()["targets"], len(everything.flags))
        self.assertIsInstance(masked.flags, np.memmap)
        self.assertIsInstance(everything.flags, studio_shards.Ones)
        self.assertEqual(sorted(p.name for p in self.root.glob("*/*.flags.bin")), ["train.flags.bin", "val.flags.bin"])

    def test_a_large_vocabulary_is_stored_in_32_bits(self):
        directory = self.root / "wide"
        directory.mkdir()
        writer = studio_shards.ShardWriter(directory, "train", 70000)
        window = np.array([studio_model.BOS, 65536, 69999, 3])
        writer.add(window, [True, False, True])
        entry = writer.close()
        tokens, flags, offsets = studio_shards.open_split(directory, "train", 70000, entry)
        self.assertEqual(tokens.dtype, np.uint32)
        store = MappedWindows(tokens, flags, offsets, entry)
        inputs, targets = store[0]
        self.assertEqual((inputs.tolist(), targets.tolist()), ([studio_model.BOS, 65536, 69999], [65536, -100, 3]))

    def test_an_empty_validation_split_is_an_empty_store(self):
        train_windows, val_windows, _ = shard_windows(TRAINING, [], self.tokenizer, config(), "none", self.root)
        self.assertEqual((len(val_windows), val_windows.describe()["windows"]), (0, 0))
        self.assertGreater(len(train_windows), 0)

    def test_memmapped_windows_give_the_same_batches_for_a_seed(self):
        for windowing in ("anchored", "sequential"):
            cfg = config(windowing=windowing)
            memory = samples_from(TRAINING, self.tokenizer, cfg)
            disk = shard_windows(TRAINING, VALIDATION, self.tokenizer, cfg, windowing, self.root)[0]
            first, second = random.Random(11), random.Random(11)
            for _ in range(25):  # train() draws a batch like this each step
                x, y = batch_arrays([memory[first.randrange(len(memory))] for _ in range(4)])
                u, v = batch_arrays([disk[second.randrange(len(disk))] for _ in range(4)])
                np.testing.assert_array_equal(x, u)
                np.testing.assert_array_equal(y, v)
            self.assertEqual(first.getstate(), second.getstate())

    def test_shards_are_reused_and_rebuilt_when_the_data_or_the_tokenizer_change(self):
        cfg = config()
        with mock.patch.object(studio_model, "cut_windows", wraps=studio_model.cut_windows) as cut:
            *_, reused = shard_windows(TRAINING, VALIDATION, self.tokenizer, cfg, "one", self.root)
            self.assertEqual((reused, cut.call_count), (False, 2))
            *_, reused = shard_windows(TRAINING, VALIDATION, self.tokenizer, cfg, "one", self.root)
            self.assertEqual((reused, cut.call_count), (True, 2))  # nothing was tokenized again
            *_, reused = shard_windows(TRAINING, VALIDATION, self.tokenizer, cfg, "two", self.root)
            self.assertEqual((reused, cut.call_count), (False, 4))  # other data
            characters = Tokenizer.fit("character", TRAINING)
            *_, reused = shard_windows(TRAINING, VALIDATION, characters, cfg, "one", self.root)
            self.assertEqual((reused, cut.call_count), (False, 6))  # other tokenizer
            *_, reused = shard_windows(TRAINING, VALIDATION, characters, cfg, "one", self.root)
            self.assertEqual((reused, cut.call_count), (True, 6))
            *_, reused = shard_windows(TRAINING, VALIDATION, self.tokenizer, config(context=64), "one", self.root)
            self.assertEqual((reused, cut.call_count), (False, 8))  # other cut
        self.assertEqual(len(window_folders(self.root)), 4)
        self.assertEqual([p for p in self.root.glob("*.part")], [])
        self.assertEqual(len(list(self.root.glob("tokens-*"))), 2)  # one cache per tokenizer, whatever the data split or the cut

    def test_a_damaged_folder_is_rebuilt(self):
        cfg = config()
        first = shard_windows(TRAINING, VALIDATION, self.tokenizer, cfg, "x", self.root)[0]
        expected = [pair for pair in first]
        del first
        for folder in window_folders(self.root):
            path = folder / "train.tokens.bin"
            path.write_bytes(path.read_bytes()[:-2])
        rebuilt, _, reused = shard_windows(TRAINING, VALIDATION, self.tokenizer, cfg, "x", self.root)
        self.assertFalse(reused)
        same_pairs(self, expected, rebuilt)

    def test_a_failed_build_leaves_no_folder(self):
        def fail(writers):
            writers["train"].add(np.array([1, 2, 3]), [True, True])
            raise RuntimeError("disk gone")
        with self.assertRaises(RuntimeError):
            studio_shards.build(self.root, "k", 263, fail)
        self.assertEqual(list(self.root.iterdir()), [])
        self.assertIsNone(studio_shards.load(self.root, "k", 263))

    def test_storage_counts_and_cleans_the_shards(self):
        shard_windows(TRAINING, VALIDATION, self.tokenizer, config(), "x", self.root / "data" / "shards")
        paths = studio_cleanup.plan(self.root)
        self.assertEqual({p.parent for p in paths}, {self.root / "data" / "shards"})
        self.assertEqual(len(paths), 2)  # the windows and the token cache beside them
        self.assertGreater(studio_cleanup.size(paths), 0)
        self.assertEqual(studio_cleanup.plan(self.root, days=30), [])  # used just now


class TokenCache(unittest.TestCase):
    """A run encodes only what no earlier run did: examples are cached under the tokenizer by their text, not by the split they fall in."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.tokenizer = Tokenizer("byte")
        self.cfg = config()

    def windows(self, training, validation, split="s", **kwargs):
        """(train, val) windows through shard_windows, and how many examples it tokenized."""
        with mock.patch.object(studio_model, "_encode_arrays", wraps=studio_model._encode_arrays) as encode:
            train_windows, val_windows, _ = shard_windows(training, validation, self.tokenizer, self.cfg, split, self.root, **kwargs)
        return train_windows, val_windows, encode.call_count

    def test_a_second_run_tokenizes_only_the_set_that_changed(self):
        first_set, second_set = TRAINING[:30], TRAINING[30:]
        _, _, encoded = self.windows(first_set + second_set, VALIDATION, "one")
        self.assertEqual(encoded, len(TRAINING) + len(VALIDATION))
        changed = second_set[:-1] + [{"text": "A row the collector added after the first run."}]
        _, _, encoded = self.windows(first_set + changed, VALIDATION, "two")  # another split_hash: the window shards are cut again
        self.assertEqual(encoded, 1)  # but only the new row was tokenized
        _, _, encoded = self.windows(list(reversed(first_set + changed)), VALIDATION, "three")  # the same text in another order
        self.assertEqual(encoded, 0)

    def test_windows_cut_from_the_cache_are_the_ones_samples_from_cuts(self):
        for windowing in ("anchored", "sequential"):
            for objective in ("assistant", "all"):
                with self.subTest(windowing=windowing, objective=objective):
                    cfg = config(windowing=windowing, objective=objective)
                    root = self.root / f"{windowing}-{objective}"
                    shard_windows(TRAINING[10:], VALIDATION, self.tokenizer, cfg, "warm", root)  # most of the text is cached already
                    shuffled = random.Random(3).sample(TRAINING, len(TRAINING))
                    train_windows, val_windows, _ = shard_windows(shuffled, VALIDATION, self.tokenizer, cfg, "cold", root)
                    same_pairs(self, samples_from(shuffled, self.tokenizer, cfg), train_windows)
                    same_pairs(self, samples_from(VALIDATION, self.tokenizer, cfg), val_windows)

    def test_a_different_tokenizer_or_target_keeps_its_own_cache(self):
        self.windows(TRAINING, VALIDATION)
        characters = Tokenizer.fit("character", TRAINING)
        with mock.patch.object(studio_model, "_encode_arrays", wraps=studio_model._encode_arrays) as encode:
            shard_windows(TRAINING, VALIDATION, characters, self.cfg, "s", self.root)
        self.assertEqual(encode.call_count, len(TRAINING) + len(VALIDATION))
        with mock.patch.object(studio_model, "_encode_arrays", wraps=studio_model._encode_arrays) as encode:
            shard_windows(TRAINING, VALIDATION, self.tokenizer, config(objective="all"), "s", self.root)
        self.assertEqual(encode.call_count, len(TRAINING) + len(VALIDATION))  # the mask depends on the objective

    def test_a_damaged_segment_is_encoded_again(self):
        self.windows(TRAINING, VALIDATION)
        for path in self.root.glob("tokens-*/seg-*/tokens.bin"):
            path.write_bytes(path.read_bytes()[:-2])
        train_windows, _, encoded = self.windows(TRAINING, VALIDATION, "another")
        self.assertEqual(encoded, len(TRAINING) + len(VALIDATION))
        same_pairs(self, samples_from(TRAINING, self.tokenizer, self.cfg), train_windows)
        self.assertEqual(list(self.root.glob("tokens-*/*.part")), [])

    def test_progress_rises_to_the_end_whether_text_is_cached_or_not(self):
        for expect_cached in (False, True):
            calls = []
            self.windows(TRAINING, VALIDATION, f"p{expect_cached}", progress=lambda done, total: calls.append((done, total)))
            self.assertEqual({total for _, total in calls}, {calls[0][1]})
            self.assertEqual([done for done, _ in calls], sorted(done for done, _ in calls))
            self.assertEqual((calls[0][0], calls[-1][0]), (0, calls[0][1]))

    def test_the_cache_is_listed_in_storage_and_ages_by_its_last_run(self):
        self.windows(TRAINING, VALIDATION)
        self.assertTrue(next(self.root.glob("tokens-*")).joinpath("used").is_file())


class TrainFromShards(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def run_training(self, name, **options):
        events = []
        train(TINY, TRAINING + VALIDATION, self.root / name, events.append, telemetry_every=0, **options)
        return events, [(e["step"], e["loss"], e["val_loss"]) for e in events if e["type"] == "metric"]

    def test_a_run_trains_from_shards_and_matches_the_in_memory_run(self):
        shards = self.root / "shards"
        memory_events, memory = self.run_training("memory", shard_dir=shards)  # under the threshold: no shards
        self.assertFalse(shards.exists())
        first_events, first = self.run_training("first", shard_dir=shards, stream_chars=0)
        self.assertEqual(first, memory)
        self.assertEqual(first[-1][0], TINY["steps"])
        self.assertTrue(any("on disk" in e["text"] and "reused" not in e["text"] for e in first_events if e["type"] == "info"))
        second_events, second = self.run_training("second", shard_dir=shards, stream_chars=0)
        self.assertEqual(second, memory)
        self.assertTrue(any("on disk, reused" in e["text"] for e in second_events if e["type"] == "info"))
        self.assertEqual(len(window_folders(shards)), 1)
        self.assertTrue((self.root / "second" / "last.npz").is_file())

    def test_preparing_the_text_is_reported_up_to_100_percent_before_the_first_step(self):
        for name in ("cold", "warm"):  # the second run finds every window shard: still ends at 100
            events, _ = self.run_training(name, shard_dir=self.root / "shards", stream_chars=0)
            kinds = [e["type"] for e in events]
            prepared = [e["percent"] for e in events if e["type"] == "prepare"]
            self.assertEqual(prepared[-1], 100)
            self.assertEqual(prepared, sorted(set(prepared)))
            self.assertLess(kinds.index("prepare"), kinds.index("metric"))
            self.assertLess(max(i for i, k in enumerate(kinds) if k == "prepare"), kinds.index("metric"))
        small, _ = self.run_training("small")
        self.assertNotIn("prepare", [e["type"] for e in small])  # under the threshold nothing is sharded

    def test_resuming_a_run_reads_the_same_batches_from_shards_as_from_memory(self):
        shards = self.root / "shards"
        half = self.root / "half"
        train(dict(TINY, steps=3), TRAINING + VALIDATION, half, telemetry_every=0)
        losses = {}
        for name, options in (("memory", {}), ("disk", dict(shard_dir=shards, stream_chars=0))):
            events = []
            train(TINY, TRAINING + VALIDATION, self.root / name, events.append, resume=half / "last.npz", telemetry_every=0, **options)
            losses[name] = [(e["step"], e["loss"], e["val_loss"]) for e in events if e["type"] == "metric"]
        self.assertEqual(losses["disk"], losses["memory"])
        self.assertEqual(max(step for step, *_ in losses["disk"]), TINY["steps"])


if __name__ == "__main__":
    unittest.main()
