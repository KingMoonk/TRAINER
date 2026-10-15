"""Build your model's "last time blocks changed": the signed move, what else differed, and when it counts as a fair comparison."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from studio_core import Workspace
from studio_ia import attribution as A
from studio_ia.pages import builder as B
from studio_ia.services import FakeServices, Version, WorkspaceServices
from studio_spec import default_spec

FACTS = dict(data="d1", steps=300, tokens=9000, lr=0.001, start=None, extra=False)


def spec(blocks, **train):
    s = B.write_spec(default_spec(), dict(B.read_spec(default_spec()), L=blocks))
    for path, value in train.items():
        slot, *keys = path.split("__")
        target = s["train"][slot]["config"] if slot != "run" else s["train"].setdefault("run", {})
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value
    return s


def edited(s, change):
    """A copy of spec `s` after change(copy)."""
    s = copy.deepcopy(s)
    change(s)
    return s


def version(number, blocks, bpc, **train):
    return Version(number, spec=spec(blocks, **train), scorecard={"bpc": bpc})


def compare(base, variant, delta, seeds=3, inside=False, mode="tokens", cancelled=False):
    return dict(base=base, variants=[variant], mode=mode,
                results=dict(cancelled=cancelled, variants=[dict(variant=0), dict(variant=1, delta=delta, seeds=list(range(seeds)),
                                                                                  inside_spread=inside)]))


def text(old, new, fa, fb, compares=()):
    return A.blocks_change_text(2, 4, 1, 2, A.attribute(old, new, fa, fb, compares))


class Direction(unittest.TestCase):
    def test_the_move_is_signed_and_lower_is_better(self):
        self.assertAlmostEqual(A.signed_move(3.0, 2.88), -0.12)
        self.assertAlmostEqual(A.signed_move(2.0, 2.05), 0.05)
        self.assertIsNone(A.signed_move(None, 2.0))
        self.assertIsNone(A.signed_move(2.0, "x"))
        self.assertEqual(A.move_text(-0.12), "error fell 0.12 bits per character (lower is better)")
        self.assertEqual(A.move_text(0.05), "error rose 0.05 bits per character (lower is better)")
        self.assertEqual(A.move_text(0.001), "error did not move")

    def test_a_rise_is_not_reported_as_a_fall(self):
        out = text(version(1, 2, 2.0), version(2, 4, 2.05), FACTS, FACTS)
        self.assertIn("error rose 0.05", out)
        self.assertNotIn("fell", out)
        out = text(version(1, 2, 2.0), version(2, 4, 1.88), FACTS, FACTS)
        self.assertIn("error fell 0.12", out)
        self.assertNotIn("rose", out)

    def test_a_missing_score_is_said_not_guessed(self):
        out = text(Version(1, spec=spec(2)), version(2, 4, 2.0), FACTS, FACTS)
        self.assertTrue(out.endswith("one of them has no held-back score yet."))

    def test_scores_from_two_scoring_methods_are_not_called_a_move(self):
        old = Version(1, spec=spec(2), scorecard={"bpc": 3.0, "bpc_method": "stride-half-1"})
        new = Version(2, spec=spec(4), scorecard={"bpc": 2.5, "bpc_method": "stride-half-2"})
        self.assertIsNone(A.attribute(old, new, FACTS, FACTS)["change"])
        new.scorecard["bpc_method"] = "stride-half-1"
        self.assertEqual(A.attribute(old, new, FACTS, FACTS)["change"], -0.5)


class Confounders(unittest.TestCase):
    def test_only_the_block_count_differing_is_clean(self):
        self.assertEqual(A.spec_changes(spec(2), spec(4)), ([], True))
        self.assertEqual(A.run_changes(FACTS, FACTS), ([], []))
        info = A.attribute(version(1, 2, 3.0), version(2, 4, 2.5), FACTS, FACTS)
        self.assertEqual((info["verdict"], info["also"], info["unchecked"]), ("clean", [], []))

    def test_spec_differences_are_named_in_plain_words(self):
        self.assertEqual(A.spec_changes(spec(2), spec(4, optimizer__lr=0.003)), (["learning rate"], False))
        self.assertEqual(A.spec_changes(spec(2), spec(4, run__steps=500)), (["steps"], False))
        self.assertEqual(A.spec_changes(spec(2), spec(4, run__steps=500, run__batch_size=16)), (["steps", "batch size"], False))
        roomy = B.write_spec(default_spec(), dict(B.read_spec(default_spec()), L=4, W=2))
        self.assertEqual(A.spec_changes(spec(2), roomy), (["thinking room"], False))

    def test_run_differences_are_named(self):
        changed, unchecked = A.run_changes(FACTS, dict(FACTS, data="d2", steps=600, start="/runs/v1/last.npz", extra=True))
        self.assertEqual((changed, unchecked), (["data", "steps", "starting weights", "extra training"], []))
        self.assertEqual(A.run_changes(FACTS, dict(FACTS, tokens=4000, lr=0.003)), (["tokens", "learning rate"], []))

    def test_what_cannot_be_checked_is_not_assumed_equal(self):
        self.assertEqual(A.run_changes(FACTS, {}), ([], ["data", "steps", "starting weights", "extra training", "tokens", "learning rate"]))
        info = A.attribute(version(1, 2, 3.0), version(2, 4, 2.5), {}, {})
        self.assertEqual(info["verdict"], "confounded")


class EveryField(unittest.TestCase):
    def test_dropout_changing_with_the_block_count_is_named_not_clean(self):
        dropout = edited(spec(4), lambda x: x["dims"].update(dropout=0.3))
        self.assertEqual(A.spec_changes(spec(2), dropout), (["dropout"], False))
        info = A.attribute(Version(1, spec=spec(2), scorecard={"bpc": 3.0}), Version(2, spec=dropout, scorecard={"bpc": 2.5}), FACTS, FACTS)
        self.assertEqual((info["verdict"], info["also"]), ("confounded", ["dropout"]))
        self.assertIn("also changed: dropout.", text(Version(1, spec=spec(2), scorecard={"bpc": 3.0}),
                                                      Version(2, spec=dropout, scorecard={"bpc": 2.5}), FACTS, FACTS))

    def test_a_field_with_no_name_falls_into_other_parts(self):
        eps = edited(spec(4), lambda x: x["block"]["norm"]["config"].update(eps=1e-6))
        self.assertEqual(A.spec_changes(spec(2), eps), (["other parts"], False))
        both = edited(eps, lambda x: x["dims"].update(dropout=0.3))
        self.assertEqual(A.spec_changes(spec(2), both), (["dropout", "other parts"], False))
        one_block = edited(spec(4), lambda x: x.update(blocks=[{"index": 3, "norm": {"id": "rmsnorm", "config": {}}}]))
        self.assertEqual(A.spec_changes(spec(2), one_block), (["other parts"], False))

    def test_a_name_alone_or_the_block_count_alone_is_still_clean(self):
        renamed = edited(spec(4), lambda x: x.update(name="Bigger"))
        self.assertEqual(A.spec_changes(spec(2), renamed), ([], True))
        self.assertEqual(A.spec_changes(spec(4), spec(2)), ([], True))
        past_the_end = edited(spec(4), lambda x: x.update(blocks=[{"index": 3, "norm": {"id": "rmsnorm", "config": {}}}]))
        self.assertEqual(A.spec_changes(past_the_end, spec(2)), ([], True))  # the override went with the block

    def test_a_design_that_cannot_be_read_is_unchecked_not_clean(self):
        broken = edited(spec(4), lambda x: x["tokenizer"].update(id="no_such_module"))
        info = A.attribute(Version(1, spec=spec(2), scorecard={"bpc": 3.0}), Version(2, spec=broken, scorecard={"bpc": 2.5}), FACTS, FACTS)
        self.assertEqual((info["verdict"], info["unchecked"]), ("confounded", ["the saved designs"]))


class Unknowns(unittest.TestCase):
    def test_unknown_is_never_equal_to_unknown(self):
        none = dict(FACTS, steps=None, tokens=None)
        changed, unchecked = A.run_changes(none, none)
        self.assertEqual((changed, unchecked), ([], ["steps", "tokens"]))
        self.assertEqual(A.run_changes(dict(FACTS, steps=None), FACTS)[1], ["steps"])
        self.assertEqual(A.run_changes(FACTS, dict(FACTS, tokens=None))[1], ["tokens"])
        info = A.attribute(version(1, 2, 3.0), version(2, 4, 2.5), none, none)
        self.assertEqual(info["verdict"], "confounded")
        self.assertIn("could not check: steps, tokens", text(version(1, 2, 3.0), version(2, 4, 2.5), none, none))

    def test_a_missing_start_is_unknown_and_none_is_from_scratch(self):
        missing = {k: v for k, v in FACTS.items() if k != "start"}
        self.assertEqual(A.run_changes(FACTS, missing), ([], ["starting weights"]))
        self.assertEqual(A.run_changes(missing, missing), ([], ["starting weights"]))
        self.assertEqual(A.run_changes(FACTS, dict(FACTS, start=None)), ([], []))  # both from scratch: equal
        self.assertEqual(A.run_changes(FACTS, dict(FACTS, start="/runs/a/last.npz")), (["starting weights"], []))

    def test_scores_that_are_not_numbers_are_no_score(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            self.assertIsNone(A.signed_move(bad, 2.0))
            self.assertIsNone(A.signed_move(2.0, bad))
            out = text(version(1, 2, 2.0), version(2, 4, bad), FACTS, FACTS)
            self.assertTrue(out.endswith("one of them has no held-back score yet."))
            self.assertNotIn("nan", out.lower())
            self.assertNotIn("inf", out.lower())
        self.assertIsNone(A.signed_move(True, 2.0))

    def test_a_compare_with_a_score_that_is_not_a_number_is_ignored(self):
        a, b = version(1, 2, 3.0), version(2, 4, 2.5)
        for bad in (float("nan"), float("inf"), "x"):
            self.assertIsNone(A.matched_compare(a.spec, b.spec, [compare(a.spec, b.spec, bad)]))

    def test_pending_facts_show_the_move_and_never_claim_the_effect(self):
        out = text(version(1, 2, 3.0), version(2, 4, 2.5), None, None)
        self.assertIn("error fell 0.50 bits per character (lower is better); checking what else changed…", out)
        self.assertNotIn("effect of the change", out)
        self.assertNotIn("fair comparison", out)
        self.assertEqual(A.attribute(version(1, 2, 3.0), version(2, 4, 2.5), None, FACTS)["verdict"], "pending")


class CompareChoice(unittest.TestCase):
    def setUp(self):
        self.a, self.b = version(1, 2, 3.0), version(2, 4, 2.5)

    def test_one_seed_is_not_seed_averaged_evidence(self):
        one = compare(self.a.spec, self.b.spec, -0.4, seeds=1)
        self.assertIsNone(A.matched_compare(self.a.spec, self.b.spec, [one]))
        out = text(self.a, self.b, FACTS, dict(FACTS, data="d2"), [one])
        self.assertNotIn("matched Compare", out)
        self.assertIn("not a fair comparison", out)
        two = compare(self.a.spec, self.b.spec, -0.4, seeds=2)
        self.assertEqual(A.matched_compare(self.a.spec, self.b.spec, [two])["seeds"], 2)
        self.assertIn("matched Compare run of 2 seeds, error fell 0.40", text(self.a, self.b, FACTS, FACTS, [two]))

    def test_the_wording_is_singular_for_one_seed(self):
        info = dict(change=-0.1, also=[], unchecked=[], verdict="matched", compare=dict(change=-0.1, seeds=1, inside=None))
        self.assertIn("run of 1 seed,", A.blocks_change_text(2, 4, 1, 2, info))

    def test_the_most_seeds_win_then_the_newest(self):
        def run(delta, seeds, finished):
            return dict(compare(self.a.spec, self.b.spec, delta, seeds=seeds), finished=finished)
        runs = [run(-0.1, 3, 5), run(-0.2, 5, 1), run(-0.3, 5, 2), run(-0.4, 2, 9)]
        self.assertAlmostEqual(A.matched_compare(self.a.spec, self.b.spec, runs)["change"], -0.3)
        self.assertAlmostEqual(A.matched_compare(self.a.spec, self.b.spec, runs[::-1])["change"], -0.3)
        self.assertAlmostEqual(A.matched_compare(self.a.spec, self.b.spec, runs[:2][::-1])["change"], -0.2)
        undated = [compare(self.a.spec, self.b.spec, -0.1, seeds=3), compare(self.a.spec, self.b.spec, -0.2, seeds=3)]
        self.assertAlmostEqual(A.matched_compare(self.a.spec, self.b.spec, undated)["change"], -0.2)  # later in the list

    def test_entries_that_are_not_the_expected_shape_are_skipped(self):
        good = compare(self.a.spec, self.b.spec, -0.4)
        gone = edited(self.a.spec, lambda x: x["tokenizer"].update(id="no_such_module"))
        row = dict(variant=1, delta=-0.1, seeds=[1, 2], inside_spread=False)
        bad = [None, "x", 7, [], {}, {"base": 5}, {"base": self.a.spec, "variants": 5, "mode": "tokens", "results": {}},
               {"base": self.a.spec, "variants": [self.b.spec], "mode": "tokens", "results": []},
               {"base": self.a.spec, "variants": [self.b.spec], "mode": "tokens", "results": {"variants": [None, 3]}},
               {"base": self.a.spec, "variants": [self.b.spec], "mode": "tokens", "results": {"variants": [dict(row, seeds=3)]}},
               {"base": self.a.spec, "variants": [None, 4], "mode": "tokens", "results": {"variants": [dict(row, variant=1)]}},
               dict(good, base=gone), dict(good, variants=[gone]), dict(good, base=None)]
        self.assertIsNone(A.matched_compare(self.a.spec, self.b.spec, bad))
        self.assertAlmostEqual(A.matched_compare(self.a.spec, self.b.spec, [*bad, good])["change"], -0.4)
        self.assertIsNone(A.matched_compare(gone, self.b.spec, [good]))


class Wording(unittest.TestCase):
    def test_a_clean_comparison_is_labelled_as_one(self):
        out = text(version(1, 2, 3.0), version(2, 4, 2.5), FACTS, FACTS)
        self.assertIn("error fell 0.50 bits per character (lower is better)", out)
        self.assertIn("Only the blocks differed, so this is the effect of the change", out)
        self.assertNotIn("not a fair comparison", out)

    def test_a_confounded_comparison_is_not_called_the_effect(self):
        out = text(version(1, 2, 3.0), version(2, 4, 2.5), FACTS, dict(FACTS, data="d2", steps=600))
        self.assertIn("error fell 0.50", out)
        self.assertIn("not a fair comparison: also changed: data, steps.", out)
        self.assertNotIn("effect of the change", out)

    def test_spec_and_run_differences_are_listed_together_once(self):
        out = text(version(1, 2, 3.0), version(2, 4, 2.5, run__steps=500), FACTS, dict(FACTS, steps=500, data="d2"))
        self.assertIn("also changed: steps, data.", out)

    def test_unchecked_facts_are_said(self):
        out = text(version(1, 2, 3.0), version(2, 4, 2.5), {}, {})
        self.assertIn("not a fair comparison: could not check: data, steps, starting weights, extra training, tokens, learning rate.", out)
        self.assertNotIn("effect of the change", out)

    def test_a_matched_compare_run_is_the_evidence_even_when_the_versions_differ(self):
        a, b = version(1, 2, 3.0), version(2, 4, 3.1, run__steps=500)
        out = text(a, b, FACTS, dict(FACTS, data="d2"), [compare(a.spec, b.spec, -0.12)])
        self.assertIn("in a matched Compare run of 3 seeds, error fell 0.12", out)
        self.assertNotIn("rose", out)  # the versions' own +0.10 is not the answer
        self.assertNotIn("within the spread", out)
        out = text(a, b, FACTS, FACTS, [compare(a.spec, b.spec, 0.02, inside=True)])
        self.assertIn("error rose 0.02", out)
        self.assertIn("within the spread across seeds, so it could be luck", out)

    def test_compare_orientation_and_eligibility(self):
        a, b = version(1, 2, 3.0), version(2, 4, 2.5)
        self.assertAlmostEqual(A.matched_compare(a.spec, b.spec, [compare(b.spec, a.spec, 0.4)])["change"], -0.4)
        for bad in (compare(a.spec, b.spec, -0.4, cancelled=True), compare(a.spec, b.spec, -0.4, mode="params"),
                    compare(a.spec, spec(6), -0.4), compare(a.spec, b.spec, None)):
            self.assertIsNone(A.matched_compare(a.spec, b.spec, [bad]))


class Services(unittest.TestCase):
    def test_the_fake_has_no_facts_so_nothing_is_claimed(self):
        svc = FakeServices()
        self.assertEqual((svc.run_facts(1), svc.compare_runs()), ({}, []))

    def test_the_page_passes_facts_and_compares_only_when_blocks_changed(self):
        calls = []
        vs = [version(1, 2, 3.0), version(2, 4, 2.5)]
        B.expectations(B.read_spec(default_spec()), B.read_spec(default_spec()), vs, facts=lambda n: calls.append(n) or FACTS)
        self.assertEqual(calls, [])
        out = B.expectations(dict(B.read_spec(default_spec()), L=4), dict(B.read_spec(default_spec()), L=6), vs,
                             facts=lambda n: calls.append(n) or FACTS)
        self.assertEqual(calls, [1, 2])
        self.assertTrue(any("effect of the change" in t for t in out))


class TkThread(unittest.TestCase):
    """The page never reads run files while painting: a worker does, and the line is painted again when it is done."""

    def page(self):
        queued, services = [], mock.Mock()
        services.run_facts.return_value = FACTS
        services.compare_runs.return_value = []
        page = SimpleNamespace(services=services, attr_cache={}, attr_fresh=set(), attr_busy=set(), painting_expect=False,
                               guide_label=None, quietly=lambda work, then, lane="fast": queued.append((work, then, lane)))
        return page, services, queued

    def test_painting_reads_nothing_and_asks_for_the_slow_lane_once(self):
        page, services, queued = self.page()
        vs = [version(1, 2, 3.0), version(2, 4, 2.5)]
        for _ in range(3):  # every redraw
            facts, compares = B.BuilderPage.attribution_inputs(page, vs)
            self.assertEqual((facts, compares), (None, None))
        services.run_facts.assert_not_called()
        services.compare_runs.assert_not_called()
        self.assertEqual([lane for _, _, lane in queued], ["slow"])
        out = B.expectations(dict(B.read_spec(default_spec()), L=4), dict(B.read_spec(default_spec()), L=6), vs, facts=facts, compares=compares)
        self.assertTrue(any("checking what else changed" in t for t in out))
        self.assertFalse(any("effect of the change" in t for t in out))

    def test_the_worker_result_is_kept_per_pair_and_used(self):
        page, services, queued = self.page()
        vs = [version(1, 2, 3.0), version(2, 4, 2.5)]
        B.BuilderPage.attribution_inputs(page, vs)
        work, then, _ = queued[0]
        then(work())
        self.assertEqual([c.args[0] for c in services.run_facts.call_args_list], [1, 2])
        facts, compares = B.BuilderPage.attribution_inputs(page, vs)
        self.assertEqual((facts(1), facts(2), compares()), (FACTS, FACTS, []))
        self.assertEqual(len(queued), 1)  # nothing asked again
        out = B.expectations(dict(B.read_spec(default_spec()), L=4), dict(B.read_spec(default_spec()), L=6), vs, facts=facts, compares=compares)
        self.assertTrue(any("effect of the change" in t for t in out))
        page.attr_fresh.clear()  # the page was refreshed: read again, showing the last result meanwhile
        self.assertIsNotNone(B.BuilderPage.attribution_inputs(page, vs)[0])
        self.assertEqual(len(queued), 2)

    def test_a_worker_that_cannot_read_says_so_instead_of_staying_pending(self):
        page, services, queued = self.page()
        services.run_facts.side_effect = OSError("gone")
        vs = [version(1, 2, 3.0), version(2, 4, 2.5)]
        B.BuilderPage.attribution_inputs(page, vs)
        work, then, _ = queued[0]
        then(work())
        facts, compares = B.BuilderPage.attribution_inputs(page, vs)
        self.assertEqual(facts(1), {})
        self.assertIn("could not check", B.expectations(dict(B.read_spec(default_spec()), L=4), dict(B.read_spec(default_spec()), L=6),
                                                         vs, facts=facts, compares=compares)[-1])

    def test_a_worker_that_fails_in_an_unexpected_way_still_frees_the_pair(self):
        page, services, queued = self.page()
        services.compare_runs.side_effect = RuntimeError("boom")
        vs = [version(1, 2, 3.0), version(2, 4, 2.5)]
        B.BuilderPage.attribution_inputs(page, vs)
        work, then, _ = queued[0]
        then(work())
        self.assertEqual(page.attr_busy, set())
        facts, compares = B.BuilderPage.attribution_inputs(page, vs)
        self.assertEqual(facts(1), {})
        self.assertEqual(len(queued), 1)  # shown as could not check, not asked again until the page refreshes
        self.assertIn("could not check", B.expectations(dict(B.read_spec(default_spec()), L=4), dict(B.read_spec(default_spec()), L=6),
                                                         vs, facts=facts, compares=compares)[-1])

    def test_without_a_block_change_nothing_is_gathered(self):
        page, services, queued = self.page()
        self.assertEqual(B.BuilderPage.attribution_inputs(page, [version(1, 2, 3.0), version(2, 2, 2.5)]), (None, None))
        self.assertEqual(queued, [])


class Records(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ws = Workspace(self.tmp.name)
        self.svc = WorkspaceServices(self.ws)
        self.addCleanup(self.svc.close)

    def run_dir(self, name, data, step=300, init=None):
        d = Path(self.ws.run_dir) / name
        d.mkdir(parents=True)
        (d / "run.json").write_text(json.dumps(dict(step=step, trained_tokens=step * 30, initialized_from=init)))
        (d / "config.json").write_text(json.dumps(dict(learning_rate=0.001)))
        (d / "dataset.jsonl").write_text(data)
        return d

    def test_facts_come_from_the_versions_run(self):
        v1, v2 = (self.svc.store.save_version(spec(n)) for n in (2, 4))
        self.svc.store.add_run(v1["number"], self.run_dir("a", '{"text": "x"}\n'))
        self.svc.store.add_run(v2["number"], self.run_dir("b", '{"text": "y"}\n', step=600, init="/runs/a/last.npz"))
        a, b = self.svc.run_facts(1), self.svc.run_facts(2)
        self.assertEqual((a["steps"], a["start"], a["extra"], b["steps"], b["start"]), (300, None, False, 600, "/runs/a/last.npz"))
        self.assertEqual(A.run_changes(a, b)[0], ["data", "steps", "starting weights"])
        self.assertEqual(self.svc.run_facts(3), {})  # no such version

    def test_same_data_gives_the_same_fingerprint(self):
        v1, v2 = (self.svc.store.save_version(spec(n)) for n in (2, 4))
        self.svc.store.add_run(v1["number"], self.run_dir("a", '{"text": "x"}\n'))
        self.svc.store.add_run(v2["number"], self.run_dir("b", '{"text": "x"}\n'))
        self.assertEqual(A.run_changes(self.svc.run_facts(1), self.svc.run_facts(2)), ([], []))

    def test_saved_compare_runs_are_read(self):
        a, b = spec(2), spec(4)
        folder = Path(self.ws.run_dir) / "compare_job1"
        folder.mkdir(parents=True)
        (folder / "experiment.json").write_text(json.dumps(dict(base=a, variants=[b], mode="tokens")))
        (folder / "results.json").write_text(json.dumps(compare(a, b, -0.2)["results"]))
        (Path(self.ws.run_dir) / "compare_job2").mkdir()  # unfinished: no files
        runs = self.svc.compare_runs()
        self.assertEqual(len(runs), 1)
        self.assertAlmostEqual(A.matched_compare(a, b, runs)["change"], -0.2)

    def test_unreadable_files_leave_a_fact_out_or_the_whole_run_unknown(self):
        v1 = self.svc.store.save_version(spec(2))
        run = self.run_dir("a", '{"text": "x"}\n')
        self.svc.store.add_run(v1["number"], run)
        (Path(self.ws.run_dir).parent / "data").mkdir(exist_ok=True)
        (run / "run.json").write_text(json.dumps(dict(step=300)))  # no tokens, no start keys
        facts = self.svc.run_facts(1)
        self.assertEqual((facts["steps"], "tokens" in facts, "start" in facts), (300, False, False))
        for name, body in (("run.json", "[1, 2]"), ("config.json", "7"), ("run.json", "{broken")):
            (run / name).write_text(body)
            self.assertEqual(self.svc.run_facts(1), {})
        (run / "run.json").write_text(json.dumps(dict(step=300, trained_tokens=9000, initialized_from=None)))
        (run / "config.json").write_text(json.dumps(dict(learning_rate=0.001)))
        self.assertEqual(self.svc.run_facts(1)["start"], None)  # a run from scratch says so
        self.assertFalse(self.svc.run_facts(1)["extra"])

    def test_a_post_training_file_in_the_wrong_shape_leaves_extra_unknown(self):
        v1 = self.svc.store.save_version(spec(2))
        self.svc.store.add_run(v1["number"], self.run_dir("a", '{"text": "x"}\n'))
        for body in ('{"1": {"3": "passed"}}', '{"1": [1]}', "[]", "{oops"):
            (self.svc.root / "data" / "post_training.json").write_text(body)
            facts = self.svc.run_facts(1)
            self.assertNotIn("extra", facts)
            self.assertEqual(facts["steps"], 300)
            self.assertEqual(A.run_changes(facts, facts)[1], ["extra training"])

    def test_facts_are_read_once_until_a_file_changes(self):
        v1 = self.svc.store.save_version(spec(2))
        run = self.run_dir("a", '{"text": "x"}\n')
        self.svc.store.add_run(v1["number"], run)
        real, seen = Path.read_text, []

        def counting(path, *args, **kwargs):
            seen.append(Path(path).name)
            return real(path, *args, **kwargs)

        with mock.patch.object(self.svc, "_file_hash", wraps=self.svc._file_hash) as hashed, \
                mock.patch.object(Path, "read_text", counting):
            first = self.svc.run_facts(1)
            self.assertIn("run.json", seen)
            seen.clear()
            self.assertEqual(self.svc.run_facts(1), first)
            self.assertEqual(([n for n in seen if n != "versions.json"], hashed.call_count), ([], 1))
            (run / "run.json").write_text(json.dumps(dict(step=999, trained_tokens=1, initialized_from=None)))
            self.assertEqual(self.svc.run_facts(1)["steps"], 999)

    def test_compare_runs_skip_files_in_the_wrong_shape_and_read_each_once(self):
        a, b = spec(2), spec(4)
        folders = {"compare_ok": (dict(base=a, variants=[b], mode="tokens"), compare(a, b, -0.2)["results"]),
                   "compare_list": ([1], {}), "compare_nobase": (dict(variants=[b]), {}), "compare_basestr": (dict(base="x"), {}),
                   "compare_results": (dict(base=a, variants=[b]), [1, 2]), "compare_variants": (dict(base=a, variants=3), {})}
        for name, (experiment, results) in folders.items():
            folder = Path(self.ws.run_dir) / name
            folder.mkdir(parents=True)
            (folder / "experiment.json").write_text(json.dumps(experiment))
            (folder / "results.json").write_text(json.dumps(results))
        broken = Path(self.ws.run_dir) / "compare_broken"
        broken.mkdir()
        (broken / "experiment.json").write_text("{nope")
        (broken / "results.json").write_text("{}")
        runs = self.svc.compare_runs()
        self.assertEqual([r["mode"] for r in runs], ["tokens"])
        self.assertIsInstance(runs[0]["finished"], int)
        with mock.patch.object(Path, "read_text", side_effect=AssertionError("read again")):
            self.assertEqual(self.svc.compare_runs(), runs)  # nothing changed: nothing re-read


if __name__ == "__main__":
    unittest.main()
