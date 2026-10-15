"""Build your model's "last time blocks changed": the signed move, what else differed, and when it counts as a fair comparison."""
import json
import tempfile
import unittest
from pathlib import Path

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
        self.assertEqual(A.run_changes(FACTS, {}), ([], ["data", "steps", "starting weights", "extra training"]))
        info = A.attribute(version(1, 2, 3.0), version(2, 4, 2.5), {}, {})
        self.assertEqual(info["verdict"], "confounded")


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
        self.assertIn("not a fair comparison: could not check: data, steps, starting weights, extra training.", out)
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


if __name__ == "__main__":
    unittest.main()
