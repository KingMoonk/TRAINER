"""The benchmark track: the fixed suite, its per-version cache, the chart data and the page helpers, all window-free."""
import json
import tempfile
import unittest
from pathlib import Path

import studio_bench
import studio_catalog
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
                         (studio_bench.DOCS, studio_bench.CHOICE_ITEMS, 80, studio_bench.PROBLEMS))
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

    def test_pending_lists_unchecked_versions_oldest_first_and_skips_the_tried(self):
        self.assertEqual(page.pending(self.state()), [5])
        self.assertEqual(page.pending(self.state(), tried={5}), [])
        self.assertEqual(page.pending(self.state([row(2, "a", 1, None), row(1, "a", 1, None)])), [1, 2])
        self.assertEqual(page.pending(self.state(docs=0, choice=0, problems=0, rows=[row(1, "a", 1, None)])), [])  # nothing to check with
        self.assertEqual(page.pending(None), [])

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

    def test_fake_services_return_the_same_shape(self):
        fake = FakeServices().benchmark()
        self.assertEqual(set(fake), {"suite", "sizes", "dropped", "rows"})
        self.assertTrue(studio_bench.chart_data(fake["rows"], "few_shot")["lines"])


if __name__ == "__main__":
    unittest.main()
