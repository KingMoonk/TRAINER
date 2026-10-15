"""The benchmark track: the fixed suite, its per-version cache, the chart data and the page helpers, all window-free."""
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import studio_bench
import studio_catalog
import studio_eval
from studio_ia.copy import BENCH
from studio_ia.pages import benchmark as page
from studio_ia.services import FakeServices, FixtureServices
from studio_model import train

FIXTURE = Path(__file__).parent / "fixtures" / "sample.jsonl"
COLOURS = ["red", "blue", "green"]


def choice(i):
    return studio_bench.choice_item(studio_catalog._lettered("Answer the question. ", f"Question {i}: which is {COLOURS[i % 3]}?", COLOURS, i % 3, "choice"))


def problem(i, rule="copy"):
    return {"prompt": f"Copy: word{i}", "rule": rule, "params": {"text": f"word{i}"}}


def pieces(n=60):
    return dict(docs=[f"Document number {i}. " + "some text to read. " * 5 for i in range(n)], choice=[choice(i) for i in range(n)],
                pool=[choice(i) for i in range(n + 20)], problems=[problem(i) for i in range(n)])


class Suite(unittest.TestCase):
    def test_sizes_are_bounded_and_the_pick_does_not_depend_on_order(self):
        a = studio_bench.build_suite(**pieces())
        self.assertEqual((len(a["docs"]), len(a["choice"]), len(a["pool"]), len(a["problems"])),
                         (studio_bench.DOCS, studio_bench.CHOICE_ITEMS, 20, studio_bench.PROBLEMS))
        shuffled = {kind: list(reversed(items)) for kind, items in pieces().items()}
        self.assertEqual(studio_bench.build_suite(**shuffled)["id"], a["id"])
        self.assertEqual(studio_bench.build_suite(**pieces())["id"], a["id"])

    def test_the_id_changes_with_the_items_and_the_seed(self):
        base = studio_bench.build_suite(**pieces())["id"]
        self.assertNotEqual(studio_bench.build_suite(**pieces(), seed=1)["id"], base)
        other = pieces()
        other["docs"] = [d + "!" for d in other["docs"]]
        self.assertNotEqual(studio_bench.build_suite(**other)["id"], base)

    def test_lettered_problems_are_the_few_shot_checks_not_checkable_ones(self):
        suite = studio_bench.build_suite(**{**pieces(5), "problems": [problem(0), problem(1, "choice")]})
        self.assertEqual([p["rule"] for p in suite["problems"]], ["copy"])

    def test_the_filter_hook_drops_items_before_sampling_and_counts_them(self):
        seen = []

        def no_zero(kind, items):
            seen.append(kind)
            return [i for i in items if "0" not in json.dumps(i)] if kind == "docs" else items
        suite = studio_bench.build_suite(**pieces(10), filter_items=no_zero)
        self.assertEqual(sorted(seen), sorted(studio_bench.KINDS))
        self.assertTrue(all("0" not in d for d in suite["docs"]))
        self.assertGreater(suite["dropped"]["docs"], 0)
        self.assertEqual(suite["dropped"]["choice"], 0)
        self.assertEqual(studio_bench.build_suite(**pieces(10))["dropped"], {kind: 0 for kind in studio_bench.KINDS})  # keep_all drops nothing

    def test_the_id_changes_with_the_scoring_settings(self):
        base = studio_bench.build_suite(**pieces())["id"]
        with mock.patch.object(studio_eval, "SCORE_METHOD", "another-method"):
            self.assertNotEqual(studio_bench.build_suite(**pieces())["id"], base)
        with mock.patch.dict(studio_bench.studio_rules.MAX_TOKENS, {"copy": 99}):
            self.assertNotEqual(studio_bench.build_suite(**pieces())["id"], base)

    def test_held_back_questions_are_never_worked_examples(self):
        data = pieces(10)
        suite = studio_bench.build_suite(**data)
        asked = {studio_bench._key(i) for i in data["choice"]}
        self.assertTrue(data["pool"][10:])  # the pool had training-only items besides the held-back ones
        self.assertTrue(all(studio_bench._key(i) not in asked for i in suite["pool"]))
        self.assertEqual(len(suite["pool"]), 20)
        held, training = [choice(i) for i in range(5)], [choice(i) for i in range(5, 9)]
        rows = lambda items: [{"prompt": f"{i['question']}\n\n" + "\n".join(f"{'ABC'[n]}) {c}" for n, c in enumerate(i["choices"])),
                               "rule": "choice", "params": {"answer": "ABC"[i["answer"]]}} for i in items]
        built = studio_bench.suite_from_rows(["a document"], rows(held), rows(training))
        self.assertEqual((len(built["choice"]), len(built["pool"])), (5, 4))
        self.assertTrue({studio_bench._key(i) for i in built["pool"]}.isdisjoint({studio_bench._key(i) for i in built["choice"]}))

    def test_suite_from_rows_reads_post_rows(self):
        rows = [problem(0), {"prompt": "Q\n\nA) x\nB) y", "rule": "choice", "params": {"answer": "B"}}]
        suite = studio_bench.suite_from_rows(["a document"], rows, rows)
        self.assertEqual((len(suite["docs"]), len(suite["choice"]), len(suite["problems"])), (1, 1, 1))


class Tokens(unittest.TestCase):
    def test_best_checkpoint_counts_to_the_lowest_held_back_loss_others_the_whole_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            rows = [{"trained_tokens": 100, "val_loss": 3.0}, {"trained_tokens": 200, "val_loss": 2.0}, {"trained_tokens": 300, "val_loss": 2.5}]
            (run / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            self.assertEqual(studio_bench.tokens_seen([run], run / "best.npz"), 200)
            self.assertEqual(studio_bench.tokens_seen([run], run / "last.npz"), 300)
            self.assertIsNone(studio_bench.tokens_seen([run / "missing"]))
            self.assertIsNone(studio_bench.tokens_seen([]))


class Lineage(unittest.TestCase):
    """Tokens seen add up every run the weights descend from, not just the newest run's own count."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def run_dir(self, name, rows, **record):
        run = self.root / name
        run.mkdir()
        (run / "metrics.jsonl").write_text("\n".join(json.dumps({"trained_tokens": t, "val_loss": v}) for t, v in rows), encoding="utf-8")
        (run / "run.json").write_text(json.dumps(record), encoding="utf-8")
        return run

    def test_a_resumed_run_counts_its_parents_rows_and_then_its_own(self):
        # the log of a resumed run starts with its parent's rows and counts from 0 again
        run = self.run_dir("cool", [(100, 3.0), (200, 2.0), (300, 2.5), (50, 2.4), (100, 1.9)], resumed_from="gone/last.npz")
        self.assertEqual(studio_bench.tokens_seen([run], run / "best.npz"), 400)
        self.assertEqual(studio_bench.tokens_seen([run], run / "last.npz"), 400)
        kept = self.run_dir("kept", [(100, 3.0), (200, 2.0), (300, 2.5), (50, 2.4), (100, 2.3)], resumed_from="gone/last.npz")
        self.assertEqual(studio_bench.tokens_seen([kept], kept / "best.npz"), 200)  # the best weights are still the parent's

    def test_a_catch_up_run_adds_the_checkpoint_it_started_from(self):
        first = self.run_dir("first", [(100, 3.0), (200, 2.0), (300, 2.5)])
        second = self.run_dir("second", [(100, 2.5), (250, 2.6)], initialized_from=str(first / "best.npz"))
        self.assertEqual(studio_bench.tokens_seen([second], second / "last.npz"), 200 + 250)
        self.assertEqual(studio_bench.tokens_seen([second], second / "best.npz"), 200 + 100)
        third = self.run_dir("third", [(10, 1.0)], initialized_from=str(second / "last.npz"))
        self.assertEqual(studio_bench.tokens_seen([third], third / "last.npz"), 450 + 10)

    def test_an_unknown_starting_point_leaves_the_version_unplaced(self):
        orphan = self.run_dir("orphan", [(100, 2.0)], initialized_from=str(self.root / "nowhere" / "best.npz"))
        self.assertIsNone(studio_bench.tokens_seen([orphan], orphan / "last.npz"))
        first = self.run_dir("first", [(100, 2.0)])
        moved = self.run_dir("moved", [(10, 1.0)], initialized_from=str(self.root / "post" / "best.npz"))  # a post-training checkpoint
        self.assertIsNone(studio_bench.tokens_seen([moved], moved / "last.npz"))
        self.assertEqual(studio_bench.tokens_seen([moved], moved / "last.npz", owner=lambda path: [first]), 110)

    def test_the_run_holding_the_checkpoint_is_used_not_the_newest(self):
        old = self.run_dir("old", [(100, 2.0), (200, 1.0)])
        new = self.run_dir("new", [(900, 3.0)])
        self.assertEqual(studio_bench.tokens_seen([old, new], old / "best.npz"), 200)
        self.assertEqual(studio_bench.tokens_seen([old, new], self.root / "round" / "best.npz"), 900)  # elsewhere: the newest run


class Cache(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.checkpoint = self.root / "best.npz"
        self.checkpoint.write_bytes(b"weights")
        self.version = (3, self.checkpoint)
        self.calls = []

    def tearDown(self):
        self.tmp.cleanup()

    def run_fake(self, version, suite, tokens):
        self.calls.append(suite["id"])
        return {"suite": suite["id"], "version": version[0], "tokens": tokens, "bpc": 2.0}

    def test_a_result_is_stored_per_version_and_reused(self):
        suite = studio_bench.build_suite(**pieces())
        self.assertIsNone(studio_bench.cached(self.root, self.version, suite))
        first = studio_bench.benchmark(self.root, self.version, suite, 500, run=self.run_fake)
        self.assertTrue(studio_bench.cache_path(self.root, 3).exists())
        self.assertEqual(studio_bench.benchmark(self.root, self.version, suite, 500, run=self.run_fake), first)
        self.assertEqual(studio_bench.cached(self.root, self.version, suite), first)
        self.assertEqual(self.calls, [suite["id"]])

    def test_a_changed_suite_is_recomputed_and_the_old_result_is_kept(self):
        old = studio_bench.build_suite(**pieces())
        new = studio_bench.build_suite(**pieces(), seed=1)
        studio_bench.benchmark(self.root, self.version, old, 500, run=self.run_fake)
        self.assertIsNone(studio_bench.cached(self.root, self.version, new))
        studio_bench.benchmark(self.root, self.version, new, 500, run=self.run_fake)
        self.assertEqual(self.calls, [old["id"], new["id"]])
        self.assertEqual(studio_bench.cached(self.root, self.version, old)["suite"], old["id"])  # going back is free

    def test_a_retrained_checkpoint_is_recomputed(self):
        suite = studio_bench.build_suite(**pieces())
        studio_bench.benchmark(self.root, self.version, suite, 500, run=self.run_fake)
        self.checkpoint.write_bytes(b"different weights")
        self.assertIsNone(studio_bench.cached(self.root, self.version, suite))
        studio_bench.benchmark(self.root, self.version, suite, 500, run=self.run_fake)
        self.assertEqual(len(self.calls), 2)

    def test_the_stamp_is_taken_before_the_run_so_a_rewrite_mid_run_is_checked_again(self):
        suite = studio_bench.build_suite(**pieces())

        def rewrites(version, suite, tokens):
            self.calls.append(suite["id"])
            self.checkpoint.write_bytes(b"weights that arrived while the suite was running")
            return {"suite": suite["id"], "bpc": 2.0}
        studio_bench.benchmark(self.root, self.version, suite, 1, run=rewrites)
        self.assertIsNone(studio_bench.cached(self.root, self.version, suite))
        studio_bench.benchmark(self.root, self.version, suite, 1, run=self.run_fake)
        self.assertEqual(len(self.calls), 2)
        self.assertIsNotNone(studio_bench.cached(self.root, self.version, suite))
        self.assertEqual(studio_bench.cached(self.root, self.version, suite)["suite"], suite["id"])
        stored = json.loads(studio_bench.cache_path(self.root, 3).read_text(encoding="utf-8"))["suites"][suite["id"]]["checkpoint"]
        self.assertEqual(stored[0], str(self.checkpoint))  # the stamp names the file too
        elsewhere = self.root / "other.npz"
        elsewhere.write_bytes(self.checkpoint.read_bytes())
        self.assertIsNone(studio_bench.cached(self.root, (3, elsewhere), suite))

    def test_two_workers_do_not_run_the_same_version_at_once(self):
        suite = studio_bench.build_suite(**pieces())
        started, release = threading.Event(), threading.Event()

        def slow(version, suite, tokens):
            self.calls.append(1)
            started.set()
            release.wait(5)
            return {"suite": suite["id"], "bpc": 2.0}
        results = []
        first = threading.Thread(target=lambda: results.append(studio_bench.benchmark(self.root, self.version, suite, 1, run=slow)))
        first.start()
        self.assertTrue(started.wait(5))
        second = threading.Thread(target=lambda: results.append(studio_bench.benchmark(self.root, self.version, suite, 1, run=slow)))
        second.start()
        release.set()
        first.join(5)
        second.join(5)
        self.assertEqual((len(self.calls), len(results), results[0] == results[1]), (1, 2, True))

    def test_a_damaged_file_counts_as_empty(self):
        suite = studio_bench.build_suite(**pieces())
        path = studio_bench.cache_path(self.root, 3)
        path.parent.mkdir(parents=True)
        path.write_text("{not json", encoding="utf-8")
        self.assertIsNone(studio_bench.cached(self.root, self.version, suite))

    def test_only_the_latest_suites_are_remembered(self):
        for seed in range(studio_bench.KEPT_SUITES + 2):
            studio_bench.benchmark(self.root, self.version, studio_bench.build_suite(**pieces(), seed=seed), 1, run=self.run_fake)
        stored = json.loads(studio_bench.cache_path(self.root, 3).read_text(encoding="utf-8"))["suites"]
        self.assertEqual(len(stored), studio_bench.KEPT_SUITES)


def result(bpc=None, accuracy=None, rate=None, chance=.33):
    return {"bpc": bpc, "few_shot": None if accuracy is None else {"items": 9, "chance": chance, "accuracy": accuracy, "rows": []},
            "problems": None if rate is None else {"passed": 1, "total": 2, "rate": rate}}


def row(number, lineage, tokens, res):
    return {"number": number, "lineage": lineage, "tokens": tokens, "result": res}


class Chart(unittest.TestCase):
    rows = [row(3, "b", 500, result(2.4, .5, .2)), row(2, "a", 800, result(2.2, .4, .1)), row(1, "a", 200, result(2.6, .3, 0.)),
            row(4, "a", None, result(2.0, .6, .3)), row(5, "b", 900, None)]

    def test_points_are_ordered_by_tokens_and_linked_by_design(self):
        chart = studio_bench.chart_data(self.rows, "bpc")
        self.assertEqual({line["lineage"]: line["points"] for line in chart["lines"]},
                         {"a": [(200, 2.6, 1), (800, 2.2, 2)], "b": [(500, 2.4, 3)]})
        self.assertEqual((chart["unplaced"], chart["waiting"], chart["chance"]), ([4], [5], None))

    def test_each_metric_reads_its_own_number_and_chance_comes_with_few_shot(self):
        few = studio_bench.chart_data(self.rows, "few_shot")
        self.assertEqual([line["points"] for line in few["lines"] if line["lineage"] == "a"][0], [(200, .3, 1), (800, .4, 2)])
        self.assertEqual(few["chance"], .33)
        solved = studio_bench.chart_data(self.rows, "problems")
        self.assertEqual([p[1] for line in solved["lines"] if line["lineage"] == "a" for p in line["points"]], [0., .1])
        self.assertIsNone(solved["chance"])

    def test_a_metric_with_no_items_gives_no_points(self):
        chart = studio_bench.chart_data([row(1, "a", 100, result(2.0))], "few_shot")
        self.assertEqual((chart["lines"], chart["chance"]), ([], None))
        self.assertEqual(studio_bench.chart_data([], "bpc")["lines"], [])


class PageHelpers(unittest.TestCase):
    rows = Chart.rows

    def state(self, rows=None, docs=24, choice=40, problems=30):
        return {"suite": "x", "sizes": {"docs": docs, "choice": choice, "problems": problems}, "dropped": {}, "rows": self.rows if rows is None else rows}

    def test_numbers_read_plainly(self):
        self.assertEqual([page.tokens_text(n) for n in (950, 12_400, 3_200_000, 2_000_000_000)], ["950", "12K", "3.2M", "2.0B"])
        self.assertEqual((page.value_text("bpc", 2.345), page.value_text("few_shot", .4), page.value_text("problems", 0.)), ("2.35", "40%", "0%"))

    def test_designs_are_lettered_in_the_order_they_first_appear(self):
        self.assertEqual(page.designs(self.rows), {"a": ("A", [1, 2, 4]), "b": ("B", [3, 5])})

    def test_pending_lists_unchecked_versions_oldest_first(self):
        self.assertEqual(page.pending(self.state()), [5])
        self.assertEqual(page.pending(self.state([row(2, "a", 1, None), row(1, "a", 1, None)])), [1, 2])
        self.assertEqual(page.pending(self.state(docs=0, choice=0, problems=0, rows=[row(1, "a", 1, None)])), [])  # nothing to check with
        self.assertEqual(page.pending(None), [])

    def test_the_few_shot_chart_says_how_many_examples_were_really_used(self):
        def with_used(used):
            res = result(2.0, .4)
            res["few_shot"]["rows"] = [{"shots": 0, "used": 0, "accuracy": .3}, {"shots": 4, "used": used, "accuracy": .4}]
            return self.state([row(1, "a", 1, res)])
        self.assertEqual(page.shots_text(with_used(4.0)), "4")
        self.assertEqual(page.shots_text(with_used(3.5)), BENCH["up_to"].format(shots=4))
        self.assertEqual(page.shots_text(self.state()), "4")  # results with no rows: nothing to say otherwise

    def test_has_items_follows_the_suite_sizes(self):
        state = self.state(choice=0)
        self.assertEqual([page.has_items(state, m) for m in studio_bench.METRICS], [True, False, True])

    def test_plot_args_draw_a_line_per_design_and_a_dashed_chance_line(self):
        args, chart = page.plot_args(self.state(), "few_shot")
        *lines, chance = args["series"]
        self.assertEqual(len(lines), 2)
        self.assertTrue(all(s["dots"] for s in lines))
        self.assertNotEqual(lines[0]["color"], lines[1]["color"])
        self.assertEqual((chance["points"], chance["dash"]), ([(0, .33), (800, .33)], (4, 3)))
        self.assertEqual((args["fmt"](.5), args["x_fmt"](2_000_000)), ("50%", "2.0M"))
        plain, _ = page.plot_args(self.state(), "bpc")
        self.assertEqual(len(plain["series"]), 2)  # no chance line on the others
        self.assertEqual(page.plot_args(self.state([row(1, "a", 1, None)]), "bpc")[0], None)

    def test_status_line(self):
        self.assertEqual(page.status_text(self.state([])), BENCH["no_versions"])
        self.assertEqual(page.status_text(self.state(docs=0, choice=0, problems=0)), BENCH["no_data"])
        self.assertEqual(page.status_text(self.state()), BENCH["waiting"].format(n=1, total=5))
        self.assertEqual(page.status_text(self.state([row(1, "a", 1, result(2.))])), BENCH["all_done"])

    def test_checking_goes_through_each_version_and_a_failure_does_not_stop_the_rest(self):
        class Services:
            def __init__(self):
                self.ran, self.noted = [], []

            def note(self, text, fraction):
                self.noted.append(fraction)

            def run_benchmark(self, number):
                if number == 2:
                    raise ValueError("bad checkpoint")
                self.ran.append(number)
        svc = Services()
        failed = page.check_versions(svc, [1, 2, 3])
        self.assertEqual((svc.ran, list(failed)), ([1, 3], [2]))
        self.assertEqual(svc.noted, [0, 1 / 3, 2 / 3])


class FakePage(page.BenchmarkPage):
    """The page's logic without a window: quietly runs the work where it stands and keeps the callback to deliver later."""

    def __init__(self, services, state):
        self.services, self.state, self.running, self.failed = services, state, False, {}
        self.later, self.drawn, self.refreshed = [], 0, 0

    def draw(self):
        self.drawn += 1

    def refresh(self):
        self.refreshed += 1

    def winfo_exists(self):
        return True

    def quietly(self, work, then, lane="fast"):
        self.lane = lane
        self.later.append((work, then))


class Checking(unittest.TestCase):
    class Services:
        def __init__(self):
            self.ran = []

        def note(self, text, fraction):
            pass

        def run_benchmark(self, number):
            if number == 2:
                raise ValueError("bad checkpoint")
            self.ran.append(number)

    def state(self, numbers):
        return {"suite": "x", "sizes": {"docs": 1, "choice": 1, "problems": 1}, "dropped": {}, "rows": [row(n, "a", 1, None) for n in numbers]}

    def test_opening_the_page_only_reads_and_the_button_starts_one_run_on_the_slow_lane(self):
        svc = self.Services()
        fake = FakePage(svc, None)
        fake._show(self.state([1, 2, 3]))
        self.assertEqual((fake.later, svc.ran), ([], []))  # showing results starts nothing
        fake.on_event("versions", None)
        self.assertEqual((fake.refreshed, fake.later), (1, []))
        fake.check()
        fake.check()  # a second press while it runs
        fake.on_event("versions", None)
        self.assertEqual((len(fake.later), fake.lane, fake.running), (1, "slow", True))
        work, then = fake.later[0]
        then(work())
        self.assertEqual((svc.ran, fake.running, list(fake.failed)), ([1, 3], False, [2]))

    def test_failures_pile_up_and_a_version_that_now_works_clears_its_own(self):
        fake = FakePage(self.Services(), self.state([2, 4]))
        fake.failed = {7: "old trouble", 2: "older trouble"}
        fake.check()
        work, then = fake.later.pop()
        then(work())
        self.assertEqual(set(fake.failed), {7, 2})
        self.assertEqual(fake.failed[2], "bad checkpoint")
        fake.state = self.state([4])
        fake.check()
        work, then = fake.later.pop()
        then(work())
        self.assertEqual(set(fake.failed), {7, 2})  # 2 was not retried; 7 never was


class RealModel(unittest.TestCase):
    """A tiny trained model: the same suite gives the same numbers, and the numbers are in range."""

    def test_running_the_suite_twice_gives_the_same_numbers(self):
        rows = [json.loads(line) for line in FIXTURE.read_text().splitlines()][:20]
        suite = studio_bench.build_suite(docs=[r["user"] + "\n" + r["assistant"] for r in rows[:6]], choice=[choice(i) for i in range(6)],
                                         pool=[choice(i) for i in range(12)], problems=[problem(i) for i in range(4)])
        with tempfile.TemporaryDirectory() as tmp:
            cfg = dict(width=16, layers=1, heads=2, context=64, steps=4, batch_size=4, eval_every=4, validation=.2, device="cpu")
            train(cfg, [{"text": r["user"] + "\n" + r["assistant"]} for r in rows], Path(tmp) / "r")
            version = (1, Path(tmp) / "r" / "last.npz")
            first = studio_bench.run_suite(version, suite, tokens=123)
            second = studio_bench.run_suite(version, suite, tokens=123)
            stored = studio_bench.benchmark(tmp, version, suite, 123)
            again = studio_bench.benchmark(tmp, version, suite, 123, run=lambda *a: self.fail("should come from the cache"))
        for got in (first, second):
            got.pop("seconds")
        self.assertEqual(first, second)
        self.assertEqual((first["suite"], first["version"], first["tokens"]), (suite["id"], 1, 123))
        self.assertGreater(first["bpc"], 0)
        self.assertEqual(first["few_shot"]["items"], 6)
        self.assertAlmostEqual(first["few_shot"]["chance"], 1 / 3)
        self.assertEqual(first["problems"]["total"], 4)
        self.assertEqual(stored, again)


class Services(unittest.TestCase):
    def test_the_workspace_checks_each_trained_version_and_remembers(self):
        with tempfile.TemporaryDirectory() as tmp:
            svc = FixtureServices(tmp)
            try:
                before = svc.benchmark()
                self.assertEqual([r["number"] for r in before["rows"]], [1, 2])
                self.assertTrue(all(r["result"] is None and r["tokens"] for r in before["rows"]))
                done = svc.run_benchmark(1)
                after = svc.benchmark()
            finally:
                svc.close()
        self.assertEqual(done["suite"], before["suite"])
        self.assertEqual(done["version"], 1)
        self.assertEqual([r["result"] is not None for r in after["rows"]], [True, False])
        self.assertEqual(after["rows"][0]["result"], done)
        self.assertEqual(page.pending({**after}), [2])

    def test_one_bad_spec_does_not_break_the_whole_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            svc = FixtureServices(tmp)
            try:
                with mock.patch("studio_spec.structure_key", side_effect=KeyError("dims")):
                    rows = svc.benchmark()["rows"]
            finally:
                svc.close()
        self.assertEqual([r["lineage"] for r in rows], ["v1", "v2"])

    def test_a_held_back_document_also_in_training_is_left_out_of_the_suite(self):
        import studio_decontam

        def suite(leak):
            with tempfile.TemporaryDirectory() as tmp:
                svc = FixtureServices(tmp)
                try:
                    leaked = [d for d in svc._heldout() if len(d.split()) >= 13][:leak]
                    with mock.patch.object(svc, "_train_index", return_value=studio_decontam.Index().add_all(leaked)):
                        return svc.benchmark()
                finally:
                    svc.close()
        clean, dirty = suite(0), suite(1)
        self.assertEqual(clean["dropped"]["docs"], 0)
        self.assertEqual(dirty["dropped"]["docs"], 1)
        self.assertNotEqual(dirty["suite"], clean["suite"])

    def test_fake_services_return_the_same_shape(self):
        fake = FakeServices().benchmark()
        self.assertEqual(set(fake), {"suite", "sizes", "dropped", "rows"})
        self.assertTrue(studio_bench.chart_data(fake["rows"], "few_shot")["lines"])


if __name__ == "__main__":
    unittest.main()
