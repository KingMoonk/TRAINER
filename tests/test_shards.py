"""Token shards: windows on disk read through np.memmap give the batches the in-memory windows give, and are cached by data and tokenizer."""
import random
import tempfile
import threading
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

    def test_a_folder_rename_refused_for_a_moment_is_retried(self):
        source, target = self.root / "a.part", self.root / "a"
        source.mkdir()
        real, calls = Path.rename, []

        def flaky(path, destination):
            calls.append(path)
            if len(calls) < 3:
                raise PermissionError(13, "Access is denied")  # WinError 5: a scanner holds a file inside
            return real(path, destination)
        with mock.patch.object(Path, "rename", flaky), mock.patch("studio_shards.time.sleep"):
            studio_shards._move(source, target)
        self.assertTrue(target.is_dir())
        self.assertEqual(len(calls), 3)

    def test_a_folder_rename_that_stays_refused_raises(self):
        source = self.root / "b.part"
        source.mkdir()
        with mock.patch.object(Path, "rename", side_effect=PermissionError(13, "Access is denied")), mock.patch("studio_shards.time.sleep"):
            with self.assertRaises(PermissionError):
                studio_shards._move(source, self.root / "b", tries=3)

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

    def test_a_stop_while_the_text_is_prepared_ends_the_run_early_and_leaves_no_unfinished_folder(self):
        stop, shards, seen = threading.Event(), self.root / "shards", []

        def emit(event):
            if event["type"] == "prepare":
                seen.append(event["percent"])
                if event["percent"] > 0:
                    stop.set()  # pressed partway through the preparation
        with self.assertRaises(InterruptedError):
            train(TINY, TRAINING + VALIDATION, self.root / "stopped", emit, stop=stop, telemetry_every=0, shard_dir=shards, stream_chars=0)
        self.assertLess(max(seen), 100)
        self.assertEqual(window_folders(shards), [])  # no finished windows and no .part folder
        self.assertEqual([p for p in shards.glob("tokens-*/*") if p.name.endswith(".part")], [])
        self.assertFalse((self.root / "stopped" / "last.npz").exists())

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

    def test_the_preflight_lookup_counts_cached_tokens_without_tokenizing(self):
        from studio_core import fingerprint
        cfg, tokenizer = config(), Tokenizer("byte")
        shard_windows(TRAINING, VALIDATION, tokenizer, cfg, "one", self.root)
        digest, vocab = studio_model._tokenizer_hash(tokenizer), tokenizer.vocab_size
        examples = TRAINING + VALIDATION
        expected = sum(len(studio_model._encode_arrays(e, tokenizer, True)[0]) for e in examples)
        with mock.patch.object(studio_model, "_encode_arrays") as encode, mock.patch.object(Tokenizer, "encode") as encoded:
            found = studio_shards.cached_tokens(self.root, digest, vocab, cfg["objective"] == "assistant", [fingerprint(e) for e in examples])
            half = studio_shards.cached_tokens(self.root, digest, vocab, cfg["objective"] == "assistant", [fingerprint(e) for e in TRAINING[:5]])
        self.assertFalse(encode.called or encoded.called)
        self.assertEqual(found, expected)
        self.assertGreater(half, 0)
        self.assertLess(half, found)
        unseen = [fingerprint({"text": "never encoded"}), fingerprint(TRAINING[0]), fingerprint(TRAINING[0])]
        self.assertEqual(studio_shards.cached_tokens(self.root, digest, vocab, cfg["objective"] == "assistant", unseen),
                         studio_shards.cached_tokens(self.root, digest, vocab, cfg["objective"] == "assistant", [fingerprint(TRAINING[0])]))

    def test_the_lookup_is_zero_for_another_tokenizer_or_no_cache(self):
        from studio_core import fingerprint
        cfg, tokenizer = config(), Tokenizer("byte")
        shard_windows(TRAINING, VALIDATION, tokenizer, cfg, "one", self.root)
        prints = [fingerprint(e) for e in TRAINING]
        self.assertEqual(studio_shards.cached_tokens(self.root, "another tokenizer", 256, False, prints), 0)
        self.assertEqual(studio_shards.cached_tokens(self.root / "nothing", studio_model._tokenizer_hash(tokenizer), 256, False, prints), 0)
        self.assertFalse((self.root / "nothing").exists())  # looking writes nothing


class PreflightCachedTokens(unittest.TestCase):
    """WorkspaceServices.cached_tokens: the tokens a warm start would find cached, from the checkpoint's tokenizer and the chosen sets' fingerprints."""

    def test_a_warm_start_counts_its_cache_and_a_fresh_run_counts_nothing(self):
        from types import SimpleNamespace
        import studio_core
        from studio_ia.services import WorkspaceServices
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            first = root / "first"
            train(dict(TINY, steps=2), TRAINING + VALIDATION, first, telemetry_every=0, shard_dir=root / "data" / "shards", stream_chars=0)
            texts = [e["text"] for e in TRAINING if e.get("text")]
            stub = SimpleNamespace(_abs=lambda p: p, _preview_ids=lambda ids: list(ids), _heldout_set=lambda: frozenset(),
                                   workspace=object())
            with mock.patch("studio_data_kinds.text_fingerprints", lambda ws, ids, exclude=(): (studio_core.fingerprint({"text": t}) for t in texts)),                     mock.patch.object(studio_core, "ROOT", root):
                warm = WorkspaceServices.cached_tokens(stub, ["x"], dict(TINY), str(first / "last.npz"))
                fresh = WorkspaceServices.cached_tokens(stub, ["x"], dict(TINY), None)
                refit = WorkspaceServices.cached_tokens(stub, ["x"], dict(TINY, refit_tokenizer=True), str(first / "last.npz"))
        self.assertGreater(warm, 0)
        self.assertEqual((fresh, refit), (0, 0))


class PreparedShardsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.tokenizer = Tokenizer("byte")
        self.cfg = config()

    def test_prepared_manifest_records_and_retrieves(self):
        folder = studio_shards.tokenizer_folder(self.root, "split123", {"id": "character", "config": {}})
        studio_shards.save_tokenizer(folder, self.tokenizer.serialize())
        train_windows, val_windows, _ = shard_windows(TRAINING, VALIDATION, self.tokenizer, self.cfg, "split123", folder)
        shard_key = studio_model.shard_key(self.tokenizer, self.cfg, "split123")
        mkey = studio_shards.manifest_key(["d1"], {"d1": 1}, 0.05, 42, 48, "sequential", "assistant", {"id": "character", "config": {}})
        entry = {
            "key": mkey,
            "dataset_ids": ["d1"],
            "split_hash": "split123",
            "digest": "dig123",
            "folder": folder.name,
            "shard_key": shard_key,
            "vocab_size": self.tokenizer.vocab_size,
            "train_windows": len(train_windows),
            "train_tokens": train_windows.entry["tokens"],
            "val_windows": len(val_windows),
            "val_tokens": val_windows.entry["tokens"],
        }
        self.assertIsNone(studio_shards.get_prepared(self.root, mkey))
        studio_shards.record_prepared(self.root, mkey, entry)
        got = studio_shards.get_prepared(self.root, mkey)
        self.assertIsNotNone(got)
        self.assertEqual(got["train_windows"], len(train_windows))
        self.assertEqual(got["train_tokens"], train_windows.entry["tokens"])

    def test_coordinator_build_with_prepared_shards_skips_reading_examples(self):
        import studio_diloco
        from tests import diloco_support
        folder = studio_shards.tokenizer_folder(self.root, "split123", {"id": "character", "config": {}})
        studio_shards.save_tokenizer(folder, self.tokenizer.serialize())
        train_windows, val_windows, _ = shard_windows(TRAINING, VALIDATION, self.tokenizer, self.cfg, "split123", folder)
        shard_key = studio_model.shard_key(self.tokenizer, self.cfg, "split123")
        mkey = studio_shards.manifest_key(["d1"], {"d1": 1}, 0.05, 42, 48, "sequential", "assistant", {"id": "character", "config": {}})
        entry = {
            "key": mkey,
            "dataset_ids": ["d1"],
            "split_hash": "split123",
            "digest": "dig123",
            "folder": folder.name,
            "shard_key": shard_key,
            "vocab_size": self.tokenizer.vocab_size,
            "train_windows": len(train_windows),
            "train_tokens": train_windows.entry["tokens"],
            "val_windows": len(val_windows),
            "val_tokens": val_windows.entry["tokens"],
        }
        coord = studio_diloco.Coordinator.build(
            "run_prepared", self.root / "run", diloco_support.CONFIG, None, diloco_support.settings(),
            shard_dir=self.root, prepared=entry
        )
        self.assertEqual(coord.split_hash, "split123")
        self.assertEqual(len(coord.train), len(train_windows))


if __name__ == "__main__":
    unittest.main()
