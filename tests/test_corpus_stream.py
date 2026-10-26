"""A big text is read as a stream: bounded memory, reading stops at the token budget, the queue sees the phases, resume still checks the data (#732)."""
import itertools
import json
import random
import tempfile
import threading
import tracemalloc
import unittest
from pathlib import Path
from types import SimpleNamespace

import studio_corpus
import studio_jobs
from studio_model import load_checkpoint, train

TINY = dict(device="cpu", width=16, layers=1, heads=2, context=32, expansion=2, dropout=0, batch_size=2, steps=6, learning_rate=.004,
            warmup=0, validation=.2, eval_every=3, seed=7, sample_every=0)


def documents(count, size=60, seed=1):
    """Generated text rows with ids; every 7th row repeats an earlier text under a new id and group (its lineage is merged into the kept row)."""
    sampler = random.Random(seed)
    words = ["alpha", "beta", "gamma", "delta", "kappa", "sigma", "omega", "theta"]
    rows = []
    for n in range(count):
        if n % 7 == 6:
            rows.append({"text": rows[n - 3]["text"], "id": f"copy_{n}", "groupId": f"g{n}", "provenance": {"source_ids": [f"origin_{n}"]}})
        else:
            rows.append({"text": f"Row {n}: " + " ".join(sampler.choice(words) for _ in range(size)), "id": f"row_{n}",
                         **({"groupId": f"g{n // 5}"} if n % 3 == 0 else {})})
    return rows


class Counted:
    """An iterator of rows that counts how many were taken."""

    def __init__(self, rows):
        self.rows, self.taken = rows, 0

    def __iter__(self):
        for row in self.rows:
            self.taken += 1
            yield row


class StreamedRuns(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def run_training(self, name, examples, **options):
        events = []
        train(TINY, examples, self.root / name, events.append, telemetry_every=0, shard_dir=self.root / "shards", **options)
        return events, [(e["step"], e["loss"], e["val_loss"]) for e in events if e["type"] == "metric"]

    def test_a_stream_trains_exactly_what_the_list_trains(self):
        rows = documents(80)
        _, from_list = self.run_training("list", rows, stream_chars=0)
        events, from_stream = self.run_training("stream", iter(rows), stream_chars=0)
        self.assertEqual(from_stream, from_list)
        listed, streamed = load_checkpoint(self.root / "list" / "last.npz"), load_checkpoint(self.root / "stream" / "last.npz")
        self.assertEqual(streamed["data_hash"], listed["data_hash"])
        self.assertEqual(streamed["split_hash"], listed["split_hash"])
        self.assertEqual((self.root / "stream" / "dataset.jsonl").read_bytes(), (self.root / "list" / "dataset.jsonl").read_bytes())
        merged = [json.loads(line) for line in (self.root / "stream" / "dataset.jsonl").read_text(encoding="utf-8").split("\n") if line.strip()]
        self.assertEqual(len(merged), 80 - len([n for n in range(80) if n % 7 == 6]))
        self.assertTrue(any("origin_" in json.dumps(row.get("provenance")) for row in merged))  # a duplicate's lineage joined the kept row
        self.assertEqual(list(self.root.glob("*/corpus-*.spool")), [])  # the spool is gone
        self.assertTrue(any(e["type"] == "info" and "on disk" in e["text"] for e in events))

    def test_a_small_stream_takes_the_in_memory_path(self):
        events, from_stream = self.run_training("small", iter(documents(20)))
        _, from_list = self.run_training("list", documents(20))
        self.assertEqual(from_stream, from_list)
        self.assertNotIn("prepare", [e["type"] for e in events])

    def test_the_queue_sees_reading_then_the_tokenizer_then_tokenizing(self):
        rows = documents(60)
        for name in ("cold", "warm"):  # the second run finds its windows already cut
            events, _ = self.run_training(name, iter(rows), stream_chars=0)
            prepared = [e for e in events if e["type"] == "prepare"]
            self.assertEqual([phase for phase, _ in itertools.groupby(e["phase"] for e in prepared)], ["reading", "tokenizer", "tokenizing"])
            percents = [e["percent"] for e in prepared]
            self.assertEqual(percents, sorted(percents))
            self.assertEqual(percents[-1], 100)
            kinds = [e["type"] for e in events]
            self.assertLess(max(i for i, k in enumerate(kinds) if k == "prepare"), kinds.index("metric"))

    def test_a_token_budget_below_one_pass_reads_only_the_rows_it_needs(self):
        rows = documents(600)
        stream = Counted(rows)
        self.run_training("budget", stream, stream_chars=0, token_budget=300)
        wanted = studio_corpus.budget_chars(300)
        characters = [len(r["text"]) for r in rows]
        needed = next(n for n in range(1, len(rows) + 1) if sum(characters[:n]) >= wanted)
        self.assertLessEqual(stream.taken, needed + 1)  # (the first row is read before the choice of path is made)
        self.assertLess(stream.taken, len(rows) // 4)
        kept = [line for line in (self.root / "budget" / "dataset.jsonl").read_text(encoding="utf-8").split("\n") if line.strip()]
        self.assertLessEqual(len(kept), needed)
        # a budget the whole text does not reach reads all of it
        everything = Counted(rows)
        self.run_training("whole", everything, stream_chars=0, token_budget=10 ** 9)
        self.assertEqual(everything.taken, len(rows))

    def test_a_small_budget_stops_the_read_even_when_the_text_would_fit_memory(self):
        stream = Counted(documents(600))
        self.run_training("tiny", stream, token_budget=300)  # default stream_chars: the in-memory path
        self.assertLess(stream.taken, 100)

    def test_a_stop_while_reading_ends_the_run_and_removes_the_spool(self):
        stop, seen = threading.Event(), []

        def emit(event):
            if event["type"] == "prepare":
                seen.append(event["phase"])
                stop.set()
        with self.assertRaises(InterruptedError):
            train(TINY, iter(documents(2000)), self.root / "stopped", emit, stop=stop, telemetry_every=0, stream_chars=0, shard_dir=self.root / "shards")
        self.assertEqual(set(seen), {"reading"})
        import gc
        gc.collect()
        self.assertEqual(list((self.root / "stopped").glob("corpus-*")), [])

    def test_resume_reads_the_snapshot_as_a_stream_and_still_checks_the_data(self):
        rows = documents(80)
        self.run_training("first", rows, stream_chars=0)
        snapshot = self.root / "first" / "dataset.jsonl"
        from studio_dataset_snapshot import read
        events = []
        train(dict(TINY, steps=9), read(snapshot), self.root / "second", events.append, resume=self.root / "first" / "last.npz", telemetry_every=0,
              stream_chars=0, shard_dir=self.root / "shards")
        self.assertEqual(max(e["step"] for e in events if e["type"] == "metric"), 9)
        with self.assertRaisesRegex(ValueError, "original dataset"):  # other text under the same checkpoint
            train(dict(TINY, steps=9), iter(documents(80, seed=5)), self.root / "third", resume=self.root / "first" / "last.npz", telemetry_every=0,
                  stream_chars=0, shard_dir=self.root / "shards")

    def test_resuming_an_unchanged_run_writes_no_new_spool_and_no_new_snapshot(self):
        from studio_dataset_snapshot import Snapshot, listing
        shared = self.root / "shared"
        self.run_training("first", documents(80), stream_chars=0, snapshot_dir=shared)
        before = self.root / "first" / "dataset.jsonl"
        stored = listing(shared)
        self.assertEqual(len(stored), 1)
        for name, chars in (("big", 0), ("small", 10 ** 9)):  # the on-disk path and the in-memory path
            created = []
            real = studio_corpus.Corpus.__init__

            def watching(corpus, folder, *args, **kwargs):
                created.append(folder)
                real(corpus, folder, *args, **kwargs)
            studio_corpus.Corpus.__init__ = watching
            try:
                events = []
                train(dict(TINY, steps=9), Snapshot(before), self.root / name, events.append, resume=self.root / "first" / "last.npz",
                      telemetry_every=0, stream_chars=chars, shard_dir=self.root / "shards", snapshot_dir=shared)
            finally:
                studio_corpus.Corpus.__init__ = real
            self.assertEqual(created, [])  # no spool was made
            self.assertEqual(max(e["step"] for e in events if e["type"] == "metric"), 9)
            self.assertEqual([path for path, _, _ in listing(shared)], [path for path, _, _ in stored])  # no second snapshot
            after = self.root / name / "dataset.jsonl"
            self.assertTrue(after.samefile(before))  # linked, not rewritten
            self.assertEqual(list((self.root / name).glob("corpus-*")), [])
            rows = lambda path: [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]
            self.assertEqual(rows(after), rows(before))  # the resumed rows are the original rows, in order

    def test_a_snapshot_with_a_repeated_row_is_read_the_long_way(self):
        from studio_dataset_snapshot import Snapshot
        path = self.root / "twice.jsonl"
        row = json.dumps({"text": "same"})
        path.write_text(row + "\n" + row + "\n", encoding="utf-8")
        self.assertIsNone(studio_corpus.Corpus.over(path))
        self.assertTrue(path.exists())
        rows, corpus = studio_corpus.collect(Snapshot(path), self.root, 0)
        self.assertEqual(len(corpus), 1)
        self.assertFalse(corpus.existing)
        corpus.discard()

    def test_a_job_streams_the_text_when_its_context_can(self):
        job = SimpleNamespace(params={"datasets": ["a"]})
        context = SimpleNamespace(example_stream=lambda params: iter([1]), examples=lambda params: [2])
        self.assertEqual(list(studio_jobs._example_source(job, context)), [1])
        job.params["examples"] = [3]
        self.assertEqual(studio_jobs._example_source(job, context), [3])
        self.assertEqual(studio_jobs._example_source(SimpleNamespace(params={}), SimpleNamespace(examples=lambda params: [2])), [2])


class Memory(unittest.TestCase):
    def test_reading_a_big_text_holds_a_batch_and_the_fingerprints_not_the_text(self):
        count, size = 4000, 5000  # 20 million characters on disk
        stream = ({"text": f"{n} " + "x" * size, "id": f"r{n}"} for n in range(count))
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            tracemalloc.start()
            try:
                rows, corpus = studio_corpus.collect(stream, folder, 1_000_000)
                training, validation, _, _ = corpus.partition(.1, random.Random(1), True)
                for _ in training:  # reading every row back one at a time holds none of them
                    pass
                _, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            self.assertIsNone(rows)
            self.assertEqual((len(training), len(validation)), (3600, 400))
            self.assertGreater(corpus.characters, 20_000_000)
            self.assertLess(peak, 5 * 1024 ** 2)  # a tenth of the text (list rows would hold 40+ MB of Python strings)
            corpus.discard()


if __name__ == "__main__":
    unittest.main()
