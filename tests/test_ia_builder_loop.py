"""Build your model's learning loop (#83): a guess before saving, checked once the version has trained; one change per version;
and the lab notebook of every version. Plain functions and page methods on stand-ins: no window opens."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import studio_registry as registry
from studio_core import Workspace
from studio_ia.pages import builder as B
from studio_ia.services import FakeServices, Version, WorkspaceServices
from studio_spec import default_spec
from studio_versions import VersionStore

CUR = B.read_spec(default_spec())


def spec(**choices):
    return B.write_spec(default_spec(), dict(CUR, **choices), CUR)


def deeper(n):
    return spec(**B.resize(CUR, n))


def version(number, s, bpc=None, prediction=None, parent=None, method=None, trained=True):
    card = {} if bpc is None else {"bpc": bpc, **({"bpc_method": method} if method else {})}
    return Version(number, spec=s, scorecard=card, prediction=prediction or {}, parent=parent,
                   best_checkpoint=f"/runs/{number}/best.npz" if trained else None)


class Questions(unittest.TestCase):
    def test_error_is_always_asked_and_costs_only_when_the_estimate_moves(self):
        self.assertEqual(B.prediction_questions(deeper(2), deeper(4)), ["error", "step", "memory"])
        self.assertEqual(B.prediction_questions(deeper(2), deeper(2)), ["error"])  # nothing moves: only the error is a question

    def test_the_chips_become_directions_and_unpicked_ones_are_left_out(self):
        picked = {"error": "Lower", "step": None, "memory": "More"}
        self.assertEqual(B.guesses_from(picked), {"error": "lower", "memory": "more"})
        self.assertEqual(B.guesses_from({"step": "Faster"}), {"step": "faster"})
        self.assertEqual(B.guesses_from({"error": None}), {})


class Direction(unittest.TestCase):
    def test_error_moves_by_the_signed_change(self):
        self.assertEqual(B.direction("error", 2.31, 2.18), "lower")
        self.assertEqual(B.direction("error", 2.18, 2.31), "higher")
        self.assertEqual(B.direction("error", 2.31, 2.312), "same")
        self.assertIsNone(B.direction("error", None, 2.0))

    def test_step_and_memory_move_by_ratio_with_a_dead_band(self):
        self.assertEqual(B.direction("step", 1.0, 1.4), "slower")
        self.assertEqual(B.direction("step", 1.4, 1.0), "faster")
        self.assertEqual(B.direction("step", 1.0, 1.03), "same")
        self.assertEqual(B.direction("memory", 1.0, 2.0), "more")
        self.assertEqual(B.direction("memory", 2.0, 1.0), "less")
        self.assertIsNone(B.direction("memory", 0, 1.0))


class Check(unittest.TestCase):
    def test_a_hit_says_what_was_expected_and_what_happened(self):
        old, new = version(1, deeper(2), 2.31), version(2, deeper(4), 2.18, {"error": "lower"}, parent=1)
        check = B.guess_check(old, new)
        self.assertEqual(check["lines"], ["You expected lower error. It went from 2.31 to 2.18 bits per character. A hit."])
        self.assertEqual((check["hits"], check["misses"], check["state"]), (1, 0, "live"))

    def test_a_miss_is_a_miss_and_turns_the_tray_to_attention(self):
        old, new = version(1, deeper(2), 2.18), version(2, deeper(4), 2.31, {"error": "lower", "memory": "less"})
        check = B.guess_check(old, new)
        self.assertEqual(check["misses"], 2)
        self.assertEqual(check["state"], "attention")
        self.assertTrue(check["lines"][0].endswith("A miss."))
        self.assertIn("estimated from the design", check["lines"][1])  # memory is never measured

    def test_no_change_in_error_counts_as_a_miss(self):
        old, new = version(1, deeper(2), 2.2), version(2, deeper(4), 2.2, {"error": "lower"})
        line = B.guess_check(old, new)["lines"][0]
        self.assertIn("It stayed at 2.20 bits per character.", line)
        self.assertTrue(line.endswith("A miss."))

    def test_before_training_it_says_when_it_will_be_checked(self):
        old, new = version(1, deeper(2), 2.31), version(2, deeper(4), None, {"error": "lower"}, trained=False)
        check = B.guess_check(old, new)
        self.assertEqual(check["lines"], ["You expected lower error. It is checked once version 2 has trained."])
        self.assertEqual(check["state"], "neutral")

    def test_scores_from_two_methods_are_not_checked(self):
        old = version(1, deeper(2), 2.31, method="a")
        new = version(2, deeper(4), 2.18, {"error": "lower"}, method="b")
        self.assertIn("no held-back score to check it against", B.guess_check(old, new)["lines"][0])
        self.assertEqual(B.guess_check(old, new)["state"], "neutral")

    def test_step_time_is_measured_when_both_runs_logged_it(self):
        old, new = version(1, deeper(2), 2.3), version(2, deeper(4), 2.2, {"step": "slower"})
        line = B.guess_check(old, new, {1: 0.120, 2: 0.250})["lines"][0]
        self.assertEqual(line, "You expected slower steps. A step went from 120 ms to 250 ms. A hit.")
        line = B.guess_check(old, new, {1: 0.120})["lines"][0]  # one run has no speed: the design's estimate, said as such
        self.assertIn("estimated from the design", line)
        self.assertTrue(line.endswith("A hit."))

    def test_no_guess_no_check(self):
        self.assertIsNone(B.guess_check(version(1, deeper(2), 2.3), version(2, deeper(4), 2.2)))
        self.assertIsNone(B.guess_check(None, version(2, deeper(4), 2.2, {"error": "lower"})))
        self.assertIsNone(B.guess_check(version(1, deeper(2)), version(2, deeper(4), prediction={"error": "sideways"})))

    def test_the_tray_also_says_what_the_page_said_before_training(self):
        old, new = version(1, deeper(2), 2.31), version(2, deeper(4), 2.18, {"error": "lower"})
        state, kicker, text = B.check_text(old, new)
        self.assertEqual((state, kicker), ("live", "YOUR GUESS · VERSION 2"))
        self.assertIn("A hit. Before training, the page said: More blocks:", text)


def heads_free_mixer():
    """A look-back module without heads, so a heads change means nothing to it."""
    return next(m.short for m in registry.for_slot("mixer") if "heads" not in B.fields_of("mixer", m.short)
                and not B.why_not(default_spec(), CUR, CUR, "mixer", m.short))


class Split(unittest.TestCase):
    def test_each_version_has_one_change_and_the_last_is_the_draft(self):
        draft = dict(B.resize(CUR, 4), W=2)
        plan = B.split_plan(default_spec(), CUR, CUR, draft)
        self.assertEqual([chips for _, _, chips in plan], [["more blocks (2 → 4)"], ["wide thinking room"]])
        self.assertEqual(plan[-1][1], B.write_spec(default_spec(), draft, CUR))
        self.assertEqual(plan[0][1], deeper(4))

    def test_one_change_is_not_split(self):
        self.assertEqual(B.split_plan(default_spec(), CUR, CUR, B.resize(CUR, 4)), [])
        self.assertEqual(B.split_plan(default_spec(), CUR, CUR, CUR), [])

    def test_heads_the_new_look_back_has_no_use_for_are_no_version(self):
        mixer = heads_free_mixer()
        draft = dict(CUR, M=dict(CUR["M"], mixer=mixer), H=8, L=4)  # heads were set, then the look-back swapped for one without
        self.assertEqual(len(B.split_steps(CUR, draft)), 2)  # blocks, look-back
        plan = B.split_plan(default_spec(), CUR, CUR, draft)
        self.assertEqual(len(plan), 2)
        self.assertEqual(plan[-1][1], B.write_spec(default_spec(), draft, CUR))
        self.assertFalse(any("head" in chip for _, _, chips in plan for chip in chips if "look-back" not in chip))

    def test_a_step_that_changes_nothing_is_folded_away(self):
        mixer = heads_free_mixer()
        base = B.write_spec(default_spec(), dict(CUR, M=dict(CUR["M"], mixer=mixer)), CUR)
        cur = B.read_spec(base)
        draft = dict(B.resize(cur, 4), W=2, H=8)  # heads mean nothing to this look-back
        plan = B.split_plan(base, cur, cur, draft)
        self.assertEqual([chips for _, _, chips in plan], [["more blocks (2 → 4)"], ["wide thinking room"]])
        self.assertEqual(plan[-1][1], B.write_spec(base, draft, cur))
        specs = [s for _, s, _ in plan]
        self.assertEqual(len(specs), len({json.dumps(s, sort_keys=True) for s in specs}))

    def test_per_block_picks_are_a_step_each(self):
        mixer = heads_free_mixer()
        draft = dict(CUR, O={(0, "mixer"): mixer, (1, "mixer"): mixer})
        plan = B.split_plan(default_spec(), CUR, CUR, draft)
        self.assertEqual(len(plan), 2)
        self.assertEqual(plan[-1][1], B.write_spec(default_spec(), draft, CUR))

    def test_the_jobs_chain_each_from_the_one_before(self):
        draft = dict(B.resize(CUR, 4), W=2)
        plan = B.split_plan(default_spec(), CUR, CUR, draft)
        ids = iter(("a1", "b2"))
        jobs = B.split_jobs(5, plan, "catch_up", "/runs/v4/best.npz", ["t"], "/runs", new_id=lambda: next(ids))
        p = [j.params for j in jobs]
        self.assertEqual([x["version"] for x in p], [5, 6])
        self.assertEqual([x["mode"] for x in p], ["catch_up", "catch_up"])
        self.assertEqual(p[0]["init_from"], "/runs/v4/best.npz")
        self.assertEqual(p[0]["output_dir"], str(Path("/runs") / "queue_a1"))
        self.assertEqual(p[1]["init_from"], str(Path("/runs") / "queue_a1" / "best.npz"))
        self.assertEqual(p[1]["output_dir"], str(Path("/runs") / "queue_b2"))

    def test_a_word_list_step_trains_from_scratch(self):
        draft = dict(B.resize(CUR, 4), V=(CUR["V"] + 1) % 4)
        plan = B.split_plan(default_spec(), CUR, CUR, draft)
        self.assertEqual(plan[0][0]["V"], draft["V"])  # the word list goes first
        jobs = B.split_jobs(2, plan, "from_scratch", None, ["t"], "/runs")
        self.assertEqual([j.params["mode"] for j in jobs], ["from_scratch", "catch_up"])
        self.assertIsNone(jobs[0].params["init_from"])


class Notebook(unittest.TestCase):
    def test_one_line_per_version_with_effects_guesses_and_fairness(self):
        vs = [version(1, deeper(2), 2.40),
              version(2, deeper(4), 2.31, {"error": "lower", "step": "faster"}, parent=1),
              version(3, spec(**dict(B.resize(CUR, 4), W=2)), 2.35, parent=2)]
        vs[0].change, vs[1].change, vs[2].change = "first design", "more blocks (2 → 4)", "wide thinking room"
        facts = {1: dict(data="d", steps=300, tokens=1, lr=.001, start=None, extra=False),
                 2: dict(data="d", steps=300, tokens=1, lr=.001, start="/runs/1/best.npz", extra=False),
                 3: dict(data="e", steps=300, tokens=1, lr=.001, start="/runs/2/best.npz", extra=False)}
        rows = B.notebook(vs, {1: .1, 2: .2}, facts)
        self.assertEqual([r["number"] for r in rows], [1, 2, 3])
        self.assertEqual(rows[0]["effects"], "error 2.40")
        self.assertTrue(rows[1]["effects"].startswith("error 2.40 → 2.31, steps about 2.0× slower, memory "))
        self.assertNotIn("(est.)", rows[1]["effects"].split(", memory")[0])  # both runs measured their speed
        self.assertTrue(rows[1]["effects"].endswith("(est.)"))  # memory is estimated
        self.assertEqual(rows[1]["guesses"], "you guessed lower error ✓, faster steps ✗")
        self.assertEqual(rows[1]["state"], "attention")
        self.assertEqual(rows[1]["note"], "")  # one change, same data and steps: a fair pair
        self.assertIn("(est.)", rows[2]["effects"].split(", memory")[0])  # version 3 has no measured speed
        self.assertEqual(rows[2]["guesses"], "no guess")
        self.assertEqual(rows[2]["note"], "also differed: data")
        line = B.notebook_line(rows[1])
        self.assertTrue(line.startswith("Version 2 · more blocks (2 → 4) → error 2.40 → 2.31"))
        self.assertIn(" · you guessed lower error ✓", line)

    def test_several_changes_at_once_are_flagged_and_untrained_said(self):
        vs = [version(1, deeper(2), 2.4), version(2, spec(**dict(B.resize(CUR, 4), W=2)), trained=False)]
        rows = B.notebook(vs)
        self.assertTrue(rows[1]["effects"].startswith("not trained yet"))
        self.assertEqual(rows[1]["note"], "2 changes at once")

    def test_the_card_says_why_it_is_empty(self):
        self.assertEqual(B.notebook_lines(None), ["Reading the runs…"])
        self.assertEqual(B.notebook_lines([]), ["Save and train a version to start the notebook."])


class Stored(unittest.TestCase):
    def test_the_guess_is_kept_with_the_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = VersionStore(Path(tmp) / "versions.json")
            store.save_version(deeper(2))
            row = store.save_version(deeper(4), "more blocks", {"error": "lower"})
            self.assertEqual(store.get(row["number"])["prediction"], {"error": "lower"})
            self.assertEqual(store.save_version(deeper(2))["prediction"], {})

    def test_services_hand_it_back_on_the_version(self):
        fake = FakeServices()
        self.assertEqual(fake.save_version(deeper(4), "x", {"step": "slower"}).prediction, {"step": "slower"})
        with tempfile.TemporaryDirectory() as tmp:
            svc = WorkspaceServices(Workspace(tmp))
            self.addCleanup(svc.close)
            svc.save_version(deeper(2), "first")
            svc.save_version(deeper(4), "more blocks", {"memory": "more"})
            self.assertEqual([v.prediction for v in svc.versions()], [{}, {"memory": "more"}])


class Speed(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.ws = Workspace(tmp.name)
        self.svc = WorkspaceServices(self.ws)
        self.addCleanup(self.svc.close)

    def speed_of(self, rows):
        d = Path(self.ws.run_dir) / "a"
        d.mkdir(parents=True)
        (d / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        self.svc.store.save_version(deeper(2))
        self.svc.store.add_run(1, d)
        return self.svc.run_speed(1)

    def test_seconds_per_step_from_the_logged_speed(self):
        rows = [dict(step=0, tokens_per_second=0.)] + \
            [dict(step=s, tokens_per_second=t, trained_tokens=s * 1000) for s, t in ((5, 9000.), (10, 10000.), (15, 11000.))]
        self.assertAlmostEqual(self.speed_of(rows), 1000 / 10000)  # 1000 pieces a step at the median 10,000 a second

    def test_a_resumed_run_counts_only_its_own_steps(self):
        rows = [dict(step=s, tokens_per_second=10000., trained_tokens=(s - 100) * 500) for s in (105, 110, 120)]
        self.assertAlmostEqual(self.speed_of(rows), 500 / 10000)

    def test_no_run_no_speed(self):
        self.assertIsNone(self.svc.run_speed(1))
        self.assertIsNone(FakeServices().run_speed(1))


class TkThread(unittest.TestCase):
    """The notebook and the guess check read no file on the Tk thread: a slow-lane worker does."""

    def test_the_notebook_is_read_on_the_slow_lane(self):
        queued, services = [], mock.Mock()
        services.versions.return_value = [version(1, deeper(2), 2.4), version(2, deeper(4), 2.3, {"error": "lower"})]
        services.run_speed.return_value = .1
        services.run_facts.return_value = {}
        page = SimpleNamespace(services=services, tray=None, speeds={}, notebook_rows=None, _notebook_token=None,
                               _paint_notebook=mock.Mock(), quietly=lambda work, then, lane="fast": queued.append((work, then, lane)))
        B.BuilderPage.load_notebook(page)
        services.run_speed.assert_not_called()
        services.run_facts.assert_not_called()
        self.assertEqual([lane for _, _, lane in queued], ["slow"])
        work, then, _ = queued[0]
        then(work())
        self.assertEqual(page.speeds, {1: .1, 2: .1})
        self.assertEqual([r["number"] for r in page.notebook_rows], [1, 2])
        page._paint_notebook.assert_called_once()

    def test_a_superseded_read_is_dropped(self):
        queued, services = [], mock.Mock()
        services.versions.return_value = []
        page = SimpleNamespace(services=services, tray=None, speeds={}, notebook_rows=None, _notebook_token=None,
                               _paint_notebook=mock.Mock(), quietly=lambda work, then, lane="fast": queued.append((work, then, lane)))
        B.BuilderPage.load_notebook(page)
        B.BuilderPage.load_notebook(page)
        queued[0][1](queued[0][0]())
        self.assertIsNone(page.notebook_rows)
        queued[1][1](queued[1][0]())
        self.assertEqual(page.notebook_rows, [])

    def test_the_guess_tray_reads_nothing(self):
        vs = {1: version(1, deeper(2), 2.4), 2: version(2, deeper(4), 2.3, {"error": "lower"}, parent=1)}
        services = mock.Mock()
        page = SimpleNamespace(phase="edit", base=vs[2], n=2, speeds={}, services=services, check_tray=mock.Mock(),
                               version=vs.get)
        B.BuilderPage._paint_check(page)
        services.run_speed.assert_not_called()
        services.run_facts.assert_not_called()
        state, kicker, text = page.check_tray.set.call_args.args
        self.assertEqual(state, "live")
        self.assertIn("It went from 2.40 to 2.30", text)


class Saving(unittest.TestCase):
    def page(self, kind):
        chip = lambda value: SimpleNamespace(value=value)  # noqa: E731
        return SimpleNamespace(asking=(kind, {}), save=mock.Mock(), split=mock.Mock(), _paint_tray=mock.Mock(),
                               guesses=[{"error": chip("Lower"), "step": chip(None)}, {"memory": chip("More")}])

    def test_save_with_guesses(self):
        page = self.page("one")
        B.BuilderPage.save_guessed(page)
        page.save.assert_called_once_with({"error": "lower"})
        self.assertIsNone(page.asking)

    def test_save_without_guessing(self):
        page = self.page("one")
        B.BuilderPage.save_guessed(page, True)
        page.save.assert_called_once_with({})

    def test_a_split_saves_a_guess_per_version(self):
        page = self.page("split")
        B.BuilderPage.save_guessed(page)
        page.split.assert_called_once_with([{"error": "lower"}, {"memory": "more"}])
        page.save.assert_not_called()
        page._paint_tray.assert_called_once()  # the guess buttons are gone before the slow read: a second click cannot save again


if __name__ == "__main__":
    unittest.main()
