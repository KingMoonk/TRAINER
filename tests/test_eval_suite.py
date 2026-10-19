"""The frozen eval suite: fixed items stored with the workspace, per-version decontamination, cloze scoring, the stored per-version scores,
the trend with paired tests, and the gate an automated loop calls. Window-free."""
import json
import math
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

import studio_bench
import studio_decontam
import studio_eval
from studio_ia.services import WorkspaceServices
from studio_model import train
from studio_versions import VersionStore
from tests.test_bench import choice, pieces, problem
from tests.test_publish import EXAMPLES, TINY


def scores(bpc, n=30, chars=100):
    """per-document (nats, characters) rows with a given corpus bits per character"""
    return [[bpc * chars * studio_eval.LN2, chars] for _ in range(n)]


class Frozen(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_the_items_are_picked_once_and_kept_for_every_later_version(self):
        built = []

        def build():
            built.append(1)
            return studio_bench.build_suite(**pieces())
        first = studio_bench.frozen(self.root, build)
        self.assertTrue(studio_bench.suite_path(self.root).exists())
        later = studio_bench.frozen(self.root, lambda: self.fail("a complete frozen suite is never built again"))
        self.assertEqual((later["id"], later["docs"], later["choice"], later["problems"]), (first["id"], first["docs"], first["choice"], first["problems"]))
        # the workspace's data changes (new passages arrive): the frozen items do not
        moved = {**pieces(), "docs": [f"A different passage {i}. " + "new words here. " * 6 for i in range(60)]}
        self.assertNotEqual(studio_bench.build_suite(**moved)["id"], first["id"])
        self.assertEqual(studio_bench.frozen(self.root, lambda: studio_bench.build_suite(**moved))["id"], first["id"])
        self.assertEqual(len(built), 1)

    def test_a_kind_with_no_items_when_frozen_is_filled_when_it_has_some(self):
        empty = {**pieces(), "problems": []}
        first = studio_bench.frozen(self.root, lambda: studio_bench.build_suite(**empty))
        self.assertEqual(first["problems"], [])
        full = studio_bench.frozen(self.root, lambda: studio_bench.build_suite(**pieces()))
        self.assertEqual(full["docs"], first["docs"])  # what was frozen stays
        self.assertEqual(len(full["problems"]), studio_bench.PROBLEMS)
        self.assertEqual(studio_bench.frozen(self.root, lambda: self.fail("now complete"))["id"], full["id"])

    def test_nothing_is_stored_while_there_is_nothing_to_freeze(self):
        studio_bench.frozen(self.root, lambda: studio_bench.build_suite([], [], [], []))
        self.assertFalse(studio_bench.suite_path(self.root).exists())

    def test_a_suite_of_another_format_is_built_again(self):
        studio_bench.frozen(self.root, lambda: studio_bench.build_suite(**pieces()))
        path = studio_bench.suite_path(self.root)
        stored = json.loads(path.read_text(encoding="utf-8"))
        path.write_text(json.dumps({**stored, "format": studio_bench.SUITE_FORMAT - 1}), encoding="utf-8")
        again = studio_bench.frozen(self.root, lambda: studio_bench.build_suite(**pieces(), seed=3))
        self.assertEqual(again["seed"], 3)

    def test_the_id_follows_the_scoring_method_so_a_frozen_suite_goes_stale(self):
        suite = studio_bench.frozen(self.root, lambda: studio_bench.build_suite(**pieces()))
        with mock.patch.object(studio_eval, "SCORE_METHOD", "another-method"):
            self.assertNotEqual(studio_bench.frozen(self.root, lambda: self.fail("kept"))["id"], suite["id"])

    def test_passages_are_spread_over_the_sources(self):
        docs = [f"Long {i}. " + "alpha beta gamma. " * 4 for i in range(40)] + [f"Short {i}. " + "delta epsilon. " * 4 for i in range(3)]
        sources = {d: ("big" if d.startswith("Long") else "small") for d in docs}
        suite = studio_bench.build_suite(docs, [], [], [], sources=sources)
        self.assertEqual(len(suite["docs"]), studio_bench.DOCS)
        self.assertEqual(suite["sources"].count("small"), 3)  # every small source is in, though it is 7% of the text
        self.assertEqual(len(suite["sources"]), len(suite["docs"]))
        shuffled = studio_bench.build_suite(list(reversed(docs)), [], [], [], sources=sources)
        self.assertEqual(shuffled["id"], suite["id"])


class Decontaminated(unittest.TestCase):
    def test_an_item_sharing_a_13_gram_with_the_versions_training_text_is_dropped_and_counted(self):
        leaked = "the quick brown fox jumps over the lazy dog while seven small birds sing a song"  # 17 words
        suite = studio_bench.build_suite(**{**pieces(10), "docs": ["Clean passage one. " + "words go here. " * 5, leaked + " and more"]})
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            run.mkdir()
            (run / "dataset.jsonl").write_text(json.dumps({"text": "Some training. " + leaked}) + "\n" + json.dumps({"user": "u", "assistant": "a"}) + "\n",
                                               encoding="utf-8")
            index = studio_decontam.index_of_runs([run])
        skip = studio_bench.contaminated(suite, index)
        self.assertEqual([suite["docs"][n] for n in skip["docs"]], [leaked + " and more"])
        self.assertEqual((skip["choice"], skip["problems"]), ([], []))

    def test_the_score_leaves_them_out_and_the_result_counts_them(self):
        rows = [json.loads(line) for line in (Path(__file__).parent / "fixtures" / "sample.jsonl").read_text().splitlines()][:20]
        docs = [r["user"] + "\n" + r["assistant"] for r in rows[:6]]
        suite = studio_bench.build_suite(docs=docs, choice=[choice(i) for i in range(6)], pool=[choice(i) for i in range(12)],
                                         problems=[problem(i) for i in range(4)])
        with tempfile.TemporaryDirectory() as tmp:
            cfg = dict(width=16, layers=1, heads=2, context=64, steps=4, batch_size=4, eval_every=4, validation=.2, device="cpu")
            train(cfg, [{"text": d} for d in docs], Path(tmp) / "r")
            version = (1, Path(tmp) / "r" / "last.npz")
            gone = {"docs": [0, 1], "choice": [2], "problems": []}
            everything = studio_bench.run_suite(version, suite, skip={"docs": [], "choice": [], "problems": []})
            kept = studio_bench.run_suite(version, suite, skip=gone)
            unknown = studio_bench.run_suite(version, suite)
        self.assertEqual(kept["dropped"], {"docs": 2, "choice": 1, "problems": 0})
        self.assertEqual([r is None for r in kept["doc_scores"]].count(True), 2)
        self.assertEqual(kept["few_shot"]["items"], 5)
        self.assertEqual(kept["few_shot"]["rows"][-1]["hits"].count(None), 1)  # the lists keep their length: versions line up item by item
        self.assertEqual((everything["decontam"], kept["decontam"], unknown["decontam"]), ("checked", "checked", "unknown"))
        self.assertNotEqual(kept["bpc"], everything["bpc"])
        # the items both versions kept have the same score, whatever else was dropped
        self.assertEqual(kept["doc_scores"][2], everything["doc_scores"][2])
        self.assertEqual(kept["problems"]["each"], everything["problems"]["each"])


class Cloze(unittest.TestCase):
    ITEM = {"task": "", "question": "The sky is", "choices": ["blue", "a very long answer that nobody would give"], "answer": 0}

    def table(self, logprobs, norm):
        encode = lambda text: [ord(c) for c in text]  # noqa: E731  one id per character
        return studio_eval.few_shot_table([self.ITEM], encode, 500, lambda pairs: [logprobs(len(a)) for _, a in pairs], shots=(0,), norm=norm)

    def test_a_toy_model_that_prefers_the_right_option_picks_it(self):
        row = self.table(lambda n: -.5 * n if n < 10 else -.9 * n, "char")["rows"][0]  # the short, right option is likelier per character
        self.assertEqual((row["correct"], row["hits"], row["accuracy"]), (1, [1], 1.))
        wrong = self.table(lambda n: -.9 * n if n < 10 else -.5 * n, "char")["rows"][0]
        self.assertEqual((wrong["correct"], wrong["hits"]), (0, [0]))

    def test_the_score_is_per_character_so_a_long_right_option_is_not_punished(self):
        item = {**self.ITEM, "choices": ["no", "this is the correct and rather long answer"], "answer": 1}
        encode = lambda text: [ord(c) for c in text]  # noqa: E731  one id per character
        logprob = lambda pairs: [-.5 * len(a) if len(a) > 5 else -4. for _, a in pairs]  # noqa: E731  short: -4 in all; long: -0.5 each character
        short, long = (logprob([([], encode(" " + c))])[0] for c in item["choices"])
        self.assertGreater(short, long)  # the raw sum prefers the short option, because it has fewer pieces to pay for
        got = studio_eval.few_shot_table([item], encode, 500, logprob, shots=(0,), norm="char")["rows"][0]
        self.assertEqual(got["hits"], [1])  # per character the long one is likelier (-0.5 against -2)

    def test_the_char_norm_does_not_depend_on_how_the_tokenizer_cuts_the_option(self):
        item = {**self.ITEM, "choices": ["abcd", "wxyz"], "answer": 0}
        per_piece = lambda cut: (lambda text: [ord(c) for c in text] if cut == 1 else [ord(text[i]) for i in range(0, len(text), 2)])  # noqa: E731
        # the same likelihood per character of text under a tokenizer that cuts every character and one that cuts every second one
        for cut in (1, 2):
            encode = per_piece(cut)
            logprob = lambda pairs: [-1. * len("".join(map(chr, a))) * (.5 if "a" in "".join(map(chr, a)) else .6) for _, a in pairs]  # noqa: E731
            self.assertEqual(studio_eval.few_shot_table([item], encode, 500, logprob, shots=(0,), norm="char")["rows"][0]["hits"], [1])

    def test_the_worked_examples_of_a_question_do_not_depend_on_the_others(self):
        items = [{"task": "t", "question": f"Q{i}", "choices": ["a", "b"], "answer": i % 2} for i in range(12)]
        seen = {}

        def record(items_):
            def encode(text):
                return [ord(c) for c in text]
            prompts = []
            studio_eval.few_shot_table(items_, encode, 2000, lambda pairs: prompts.extend(bytes(p).decode() for p, _ in pairs) or [0.] * len(pairs),
                                       shots=(0, 4), seed=7, pool=items)
            return prompts
        for prompt in record(items):
            seen.setdefault(prompt.rpartition("\n\n")[2], set()).add(prompt)
        fewer = record(items[3:])
        self.assertTrue(all(prompt in seen[prompt.rpartition("\n\n")[2]] for prompt in fewer))

    def test_a_norm_must_be_known(self):
        with self.assertRaises(ValueError):
            self.table(lambda n: 0., "sum")


class PairedTests(unittest.TestCase):
    def test_a_real_gain_in_accuracy_is_better_and_a_wobble_is_not(self):
        parent = [1, 0] * 30
        better = [1, 1] * 30
        self.assertTrue(studio_eval.paired_hits(parent, better)["better"])
        same = studio_eval.paired_hits(parent, parent)
        self.assertEqual((same["diff"], same["better"]), (0., False))
        wobble = [1, 0] * 29 + [1, 1]
        self.assertFalse(studio_eval.paired_hits(parent, wobble)["better"])

    def test_few_items_are_never_better(self):
        got = studio_eval.paired_hits([0] * 5, [1] * 5)
        self.assertEqual((got["better"], got.get("reason"), got["diff"]), (False, "few_items", 1.))
        with self.assertRaises(ValueError):
            studio_eval.paired_hits([1, 0], [1])

    def test_the_accuracy_band_is_wide_for_few_questions(self):
        lo, hi = studio_eval.accuracy_band([1] * 20 + [0] * 20)
        self.assertLess(lo, .4)
        self.assertGreater(hi, .6)
        self.assertTrue(math.isnan(studio_eval.accuracy_band([])[0]))
        self.assertLess(studio_eval.accuracy_band([1] * 500 + [0] * 500)[1] - studio_eval.accuracy_band([1] * 500 + [0] * 500)[0], .1)


class Stored(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.checkpoint = self.root / "best.npz"
        self.checkpoint.write_bytes(b"weights")
        self.version = (3, self.checkpoint)
        self.calls = []

    def fake(self, version, suite, tokens, skip=None, resume=None, save=None):
        self.calls.append((suite["id"], skip))
        return {"suite": suite["id"], "bpc": 2.0}

    def test_a_version_records_its_scores_and_a_changed_method_scores_it_again(self):
        suite = studio_bench.build_suite(**pieces())
        self.assertFalse(studio_bench.stale(self.root, self.version, suite))  # never scored is not stale
        studio_bench.benchmark(self.root, self.version, suite, 1, run=self.fake)
        self.assertEqual(studio_bench.cached(self.root, self.version, suite)["bpc"], 2.0)
        self.assertFalse(studio_bench.stale(self.root, self.version, suite))
        with mock.patch.object(studio_eval, "SCORE_METHOD", "stride-half-3"):
            newer = studio_bench.build_suite(**pieces())
            self.assertNotEqual(newer["id"], suite["id"])
            self.assertIsNone(studio_bench.cached(self.root, self.version, newer))
            self.assertTrue(studio_bench.stale(self.root, self.version, newer))
            studio_bench.benchmark(self.root, self.version, newer, 1, run=self.fake)
            self.assertFalse(studio_bench.stale(self.root, self.version, newer))
        self.assertEqual([c[0] for c in self.calls], [suite["id"], newer["id"]])

    def test_a_retrained_checkpoint_is_not_stale_just_unscored(self):
        suite = studio_bench.build_suite(**pieces())
        studio_bench.benchmark(self.root, self.version, suite, 1, run=self.fake)
        self.checkpoint.write_bytes(b"other weights")
        self.assertFalse(studio_bench.stale(self.root, self.version, suite))
        self.assertIsNone(studio_bench.cached(self.root, self.version, suite))

    def test_the_skip_the_caller_computed_reaches_the_run(self):
        suite = studio_bench.build_suite(**pieces())
        studio_bench.benchmark(self.root, self.version, suite, 1, run=self.fake, skip={"docs": [1]})
        self.assertEqual(self.calls[0][1], {"docs": [1]})

    def test_an_interrupted_pass_resumes_after_the_stages_it_finished(self):
        rows = [json.loads(line) for line in (Path(__file__).parent / "fixtures" / "sample.jsonl").read_text().splitlines()][:20]
        docs = [r["user"] + "\n" + r["assistant"] for r in rows[:6]]
        suite = studio_bench.build_suite(docs=docs, choice=[choice(i) for i in range(6)], pool=[choice(i) for i in range(12)],
                                         problems=[problem(i) for i in range(4)])
        cfg = dict(width=16, layers=1, heads=2, context=64, steps=4, batch_size=4, eval_every=4, validation=.2, device="cpu")
        train(cfg, [{"text": d} for d in docs], self.root / "r")
        version = (1, self.root / "r" / "last.npz")
        real = studio_eval.few_shot
        with mock.patch.object(studio_eval, "few_shot", side_effect=RuntimeError("stopped")):
            with self.assertRaises(RuntimeError):
                studio_bench.benchmark(self.root, version, suite, 5, skip={})
        stored = json.loads(studio_bench.cache_path(self.root, 1).read_text(encoding="utf-8"))["suites"][suite["id"]]
        self.assertNotIn("result", stored)
        self.assertEqual(stored["partial"]["stages"], ["docs"])
        self.assertIsNone(studio_bench.cached(self.root, version, suite))
        with mock.patch.object(studio_eval, "score_docs", side_effect=AssertionError("the documents were scored again")), \
                mock.patch.object(studio_eval, "few_shot", wraps=real) as few:
            done = studio_bench.benchmark(self.root, version, suite, 5, skip={})
        self.assertEqual(few.call_count, 1)
        self.assertEqual(done["stages"], ["docs", "choice", "problems"])
        self.assertEqual(done["bpc"], stored["partial"]["bpc"])
        self.assertEqual(studio_bench.cached(self.root, version, suite), done)


def result(bpc, hits, rate_hits=(), docs=30, suite="s"):
    return {"suite": suite, "bpc": bpc, "doc_scores": scores(bpc, docs), "bpc_band": [bpc - .01, bpc + .01], "dropped": {},
            "few_shot": {"accuracy": sum(hits) / len(hits), "band": [0., 1.], "rows": [{"shots": 4, "hits": list(hits)}]},
            "problems": {"rate": sum(rate_hits) / len(rate_hits), "each": list(rate_hits)} if rate_hits else None}


class Trend(unittest.TestCase):
    def rows(self):
        a = result(2.6, [1, 0] * 30)
        b = result(2.2, [1, 1] * 30)
        c = result(2.2, [1, 1] * 30)
        c["doc_scores"] = [[n + ((i % 2) * .5 - .25), ch] for i, (n, ch) in enumerate(c["doc_scores"])]  # same total, a different spread
        return [{"number": 1, "lineage": "chain-1", "tokens": 100, "result": a}, {"number": 2, "lineage": "chain-1", "tokens": 300, "result": b},
                {"number": 3, "lineage": "chain-1", "tokens": 200, "result": c}, {"number": 4, "lineage": "other", "tokens": 50, "result": a},
                {"number": 5, "lineage": "chain-1", "tokens": 400, "result": None}]

    def test_points_are_in_tokens_order_with_a_paired_test_against_the_one_before(self):
        lines = {line["lineage"]: line for line in studio_bench.trend(self.rows())}
        chain = lines["chain-1"]
        self.assertEqual(chain["versions"], [1, 3, 2])
        self.assertIsNone(chain["points"][0]["gain"])
        step = chain["points"][1]["gain"]
        self.assertTrue(step["bpc"]["better"])  # 2.6 -> 2.2 on thirty documents each way
        self.assertTrue(step["few_shot"]["better"])
        self.assertIsNone(step["problems"])
        self.assertFalse(chain["points"][2]["gain"]["bpc"]["better"])  # 2.2 against 2.2: nothing gained
        self.assertEqual(len(lines["other"]["points"]), 1)

    def test_a_version_without_a_result_is_left_out(self):
        self.assertNotIn(5, [p["number"] for line in studio_bench.trend(self.rows()) for p in line["points"]])

    def test_only_the_items_both_versions_kept_are_paired(self):
        old, new = result(2.6, [1, 0] * 30), result(2.2, [1, 1] * 30)
        new["doc_scores"][0] = None
        new["few_shot"]["rows"][0]["hits"][0] = None
        got = studio_bench.compare_results(old, new)
        self.assertEqual((got["bpc"]["docs"], got["few_shot"]["items"]), (29, 59))

    def test_results_made_before_per_item_scores_are_not_compared(self):
        self.assertEqual(studio_bench.compare_results({"bpc": 2.0}, {"bpc": 1.0}), {"bpc": None, "few_shot": None, "problems": None})


class Gate(unittest.TestCase):
    def test_a_real_drop_fails_and_noise_passes(self):
        base = result(2.0, [1, 1, 0] * 20, [1] * 30)
        worse = result(2.4, [0, 1, 0] * 20, [0] * 30)
        got = studio_bench.gate_verdict(base, worse)
        self.assertFalse(got["ok"])
        self.assertEqual([c["ok"] for c in got["checks"]], [False, False, False])
        same = studio_bench.gate_verdict(base, dict(base))
        self.assertTrue(same["ok"])
        better = studio_bench.gate_verdict(base, result(1.8, [1, 1, 1] * 20, [1] * 30))
        self.assertTrue(better["ok"])

    def test_results_of_different_suites_cannot_be_compared(self):
        got = studio_bench.gate_verdict(result(2., [1] * 30, suite="a"), result(2., [1] * 30, suite="b"))
        self.assertFalse(got["ok"])
        self.assertEqual(got["checks"][0]["name"], "Same frozen suite")

    def test_an_unmeasured_side_does_not_fail_the_gate(self):
        base = {"suite": "s", "bpc": None}
        self.assertTrue(studio_bench.gate_verdict(base, dict(base))["ok"])

    def test_the_suite_texts_are_what_a_loop_keeps_its_prompts_away_from(self):
        suite = studio_bench.build_suite(**pieces(10))
        index = studio_bench.suite_index(suite)
        self.assertTrue(index.overlaps(suite["docs"][0]))
        self.assertTrue(index.overlaps(problem(0)["prompt"]) or not suite["problems"])
        self.assertFalse(index.overlaps("An unrelated sentence that is nowhere in the suite at all and has more than thirteen words in it"))


class Lineage(unittest.TestCase):
    """The services' view: lineages follow initialized_from, and a version that gets weights is scored by itself."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        services = object.__new__(WorkspaceServices)
        services.root, services.store = self.root, VersionStore(self.root)
        services._bench_lock, services._bench_wait, services._bench_busy = threading.Lock(), [], False
        services.emitted = []
        services.emit = lambda *args: services.emitted.append(args)
        self.services = services
        self.spec = {}

    def version(self, number, started_from=None):
        run = self.root / "runs" / f"v{number}"
        run.mkdir(parents=True)
        (run / "run.json").write_text(json.dumps({"initialized_from": str(started_from) if started_from else None}), encoding="utf-8")
        (run / "best.npz").write_bytes(b"w")
        return SimpleNamespace(number=number, runs=[str(run)], best_checkpoint=str(run / "best.npz"), spec=self.spec, status="trained",
                               name=f"Version {number}"), run

    def test_versions_that_continue_one_another_share_a_lineage(self):
        one, run1 = self.version(1)
        two, run2 = self.version(2, started_from=run1 / "best.npz")
        three, _ = self.version(3, started_from=run2 / "best.npz")
        loner, _ = self.version(4)
        out = self.services._lineages([one, two, three, loner])
        self.assertEqual(out[1], ("chain-1", None))
        self.assertEqual((out[2], out[3]), (("chain-1", 1), ("chain-1", 2)))
        self.assertNotIn("chain", out[4][0])

    def test_a_trained_version_is_scored_on_a_worker_once(self):
        scored = []
        started, release = threading.Event(), threading.Event()
        self.services.auto_bench = True
        self.services._bench_suite = lambda: {"docs": ["x"], "choice": [], "problems": []}
        self.services.run_benchmark = lambda number: scored.append(number) or started.set() or release.wait(10)
        self.services._bench_later(7)
        self.assertTrue(started.wait(10))
        self.services._bench_later(7)  # asked again while it is being scored: not queued twice
        release.set()
        for _ in range(100):
            if self.services.emitted:
                break
            threading.Event().wait(.05)
        self.assertEqual(scored, [7])
        self.assertEqual(self.services.emitted[-1], ("versions", None))

    def test_a_failure_is_logged_and_does_not_stop_the_next_version(self):
        scored, done = [], threading.Event()
        self.services.auto_bench = True
        self.services._bench_suite = lambda: {"docs": ["x"], "choice": [], "problems": []}

        def run(number):
            if number == 1:
                raise RuntimeError("broken checkpoint")
            scored.append(number)
            done.set()
        self.services.run_benchmark = run
        with self.assertLogs("studio_ia.services", "ERROR"):
            self.services._bench_later(1)
            self.services._bench_later(2)
            self.assertTrue(done.wait(10))
            for _ in range(100):
                if self.services.emitted:
                    break
                threading.Event().wait(.05)
        self.assertEqual(scored, [2])

    def test_nothing_runs_when_automatic_scoring_is_off(self):
        self.services.auto_bench = False
        self.services.run_benchmark = lambda number: self.fail("off")
        self.services._bench_later(1)
        self.assertEqual(self.services._bench_wait, [])


class GateEval(unittest.TestCase):
    def test_a_checkpoint_is_scored_against_a_baseline_on_the_one_frozen_suite(self):
        rows = [json.loads(line) for line in (Path(__file__).parent / "fixtures" / "sample.jsonl").read_text().splitlines()][:20]
        docs = [r["user"] + "\n" + r["assistant"] for r in rows[:8]]
        suite = studio_bench.build_suite(docs=docs, choice=[choice(i) for i in range(6)], pool=[choice(i) for i in range(12)],
                                         problems=[problem(i) for i in range(4)])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train(dict(TINY, steps=4, eval_every=4), EXAMPLES, root / "a", telemetry_every=0)
            train(dict(TINY, steps=8, eval_every=8), EXAMPLES, root / "b", telemetry_every=0)
            services = object.__new__(WorkspaceServices)
            services.root = root
            services._bench_suite = lambda: suite
            services._abs = lambda p: p
            got = services.gate_eval(str(root / "b" / "best.npz"), baseline=str(root / "a" / "best.npz"))
            again = services.gate_eval(str(root / "b" / "best.npz"))
        self.assertEqual(got["scores"]["suite"], suite["id"])
        self.assertEqual(got["scores"]["decontam"], "unknown")  # a file that is no version: no training text to check against
        self.assertEqual(set(got["verdict"]), {"ok", "checks", "gain"})
        self.assertEqual(len(got["verdict"]["checks"]), 3)
        self.assertIsNone(again["verdict"])
        self.assertEqual(again["scores"]["bpc"], got["scores"]["bpc"])
        self.assertEqual(np.shape(got["scores"]["doc_scores"]), (len(suite["docs"]), 2))


if __name__ == "__main__":
    unittest.main()
