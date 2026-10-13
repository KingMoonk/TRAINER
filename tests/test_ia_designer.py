"""Logic of the three Network-designer pages (Build your model, Quick test, Compare runs). No Tk window is created."""
import copy
import unittest

from studio_ia.pages import builder as B
from studio_ia.pages import compare as K
from studio_ia.pages import quicktest as Q
from studio_ia.services import FakeServices, Job, QueueState, Version
from studio_ia.theme import C
from studio_ia.widgets import plot_series
from studio_model import validate_config
from studio_spec import default_spec, validate_spec


def choices(**patch):
    return dict(B.read_spec(default_spec()), **patch)


class Builder(unittest.TestCase):
    def test_write_spec_touches_only_what_changed(self):
        base = validate_spec(default_spec())
        base["tokenizer"] = {"id": "bpe_custom", "config": {"vocab_size": 700}}
        same = B.write_spec(base, B.read_spec(base))
        self.assertEqual(same, validate_spec(base))  # no change: nothing rewritten, tokenizer config kept
        more = B.write_spec(base, dict(B.read_spec(base), L=base["dims"]["n_layer"] + 2))
        self.assertEqual(more["tokenizer"]["config"], {"vocab_size": 700})
        self.assertEqual(more["dims"]["n_layer"], base["dims"]["n_layer"] + 2)
        words = B.write_spec(base, dict(B.read_spec(base), V=2))
        self.assertEqual(words["tokenizer"]["id"], "bpe_gpt2")

    def test_anchor_keeps_parts_the_picture_cannot_show(self):
        anchor = validate_spec(default_spec())
        anchor["train"]["optimizer"]["config"]["lr"] = 3e-4
        was = B.read_spec(anchor)
        out = B.write_spec(anchor, dict(was, H=8), was)
        self.assertEqual(out["train"]["optimizer"]["config"]["lr"], 3e-4)
        self.assertEqual(out["block"]["mixer"]["config"]["heads"], 8)

    def test_warm_start_outlook_follows_the_warmstart_rules(self):
        cur = choices()
        self.assertIsNone(B.warm_outlook(cur, dict(cur)))
        more = B.warm_outlook(cur, dict(cur, L=cur["L"] + 2))
        self.assertEqual((more["kind"], more["carries"], more["preserving"]), ("same", True, True))
        wide = B.warm_outlook(cur, dict(cur, W=min(2, cur["W"] + 1)))
        self.assertTrue(wide["preserving"] or cur["W"] == 2)
        heads = B.warm_outlook(cur, dict(cur, H=8 if cur["H"] != 8 else 4))
        self.assertEqual((heads["kind"], heads["carries"], heads["preserving"]), ("heads", True, False))
        fewer = B.warm_outlook(cur, dict(cur, L=max(1, cur["L"] - 1)))
        self.assertEqual((fewer["kind"], fewer["preserving"]), ("removed", False))
        scratch = B.warm_outlook(cur, dict(cur, V=(cur["V"] + 1) % 3, H=8))
        self.assertEqual((scratch["kind"], scratch["carries"]), ("scratch", False))  # word list beats every other change

    def test_cost_formulas_and_words(self):
        a, b = B.cfg(dict(choices(), L=2)), B.cfg(dict(choices(), L=4))
        self.assertGreater(B.P(b), B.P(a))
        self.assertEqual(B.TM(a), 16 * B.P(a))
        self.assertEqual(B.say(B.P(a), B.P(a), "bigger", "smaller"), "about the same")
        self.assertEqual(B.say(100, 150, "bigger", "smaller"), "about 1.5× bigger")
        self.assertEqual(B.say(150, 100, "bigger", "smaller"), "about 1.5× smaller")
        self.assertEqual(B.bar(50, 100), (50, 100))
        # time per step: C = 6*(L*(4d^2+2d*ff) + V*d) + 6*L*T*d
        c = dict(V=10, d=4, T=8, L=2, H=1, ff=16)
        self.assertEqual(B.C_(c), 6 * (2 * (64 + 128) + 40) + 6 * 2 * 8 * 4)

    def test_memory_fit_is_a_comparison_with_ram(self):
        c = B.cfg(choices())
        need = B.TM(c) + B.ROWS * B.activations(c)
        self.assertTrue(B.fits(c, ram=need))
        self.assertFalse(B.fits(c, ram=need - 1))

    def test_last_change_comes_from_history_never_a_guess(self):
        s2, s4 = B.write_spec(default_spec(), choices(L=2)), B.write_spec(default_spec(), choices(L=4))
        vs = [Version(1, spec=s2, scorecard={"bpc": 3.0}), Version(2, spec=s4, scorecard={"bpc": 2.5})]
        self.assertEqual(B.last_blocks_change(vs), (2, 4, 1, 2, "0.50"))
        self.assertEqual(B.last_blocks_change([Version(1, spec=s2), Version(2, spec=s4)])[4], "[N]")
        self.assertIsNone(B.last_blocks_change(vs[:1]))
        out = B.expectations(choices(L=4), choices(L=6), vs)
        self.assertTrue(any("Last time blocks changed (2 → 4, version 1 → 2), error moved by 0.50" in t for t in out))
        self.assertEqual(B.expectations(choices(), choices()), [B.T["change_something"]])

    def test_pieces_are_counted_not_invented(self):
        chips, count = B.pieces_for(0)
        self.assertEqual((len(chips), count), (len(B.PROMPT), f"{len(B.PROMPT)} pieces, one per character"))
        self.assertEqual(B.pieces_for(2)[1], "8 pieces")
        self.assertIn("your data", B.pieces_for(1)[1] + B.pieces_for(1)[0][0])

    def test_chart_continues_from_the_old_end_and_never_invents(self):
        self.assertEqual(B.chart_args(None, None, None)["smooth"], [])
        old = {"smooth": [(0, 5.), (100, 3.)], "heldout": [(100, 3.1)], "total": 100}
        new = {"steps": [(0, 3.), (50, 2.9)], "total": 200}
        a = B.chart_args(old, new, {"uniform_bpc": 8., "unigram_bpc": None, "best_so_far_bpc": 3., "noise_band": (2.9, 3.2)})
        self.assertEqual(a["steps"], [(100, 3.), (150, 2.9)])
        self.assertEqual(a["total"], 300)
        self.assertEqual(a["refs"], [("uniform guess", 8.), ("best so far", 3.)])
        self.assertEqual(a["band"], (2.9, 3.2))

    def test_pill_and_job_shape(self):
        v = Version(4, status="trained")
        self.assertEqual(B.pill_text(None, v), "Version 4 is trained")
        self.assertEqual(B.pill_text(None, None), "No saved version yet")
        job = Job("a", "pretrain", params={"version": 5, "mode": "catch_up"})
        self.assertEqual(B.pill_text(job, v), "Version 5 is catching up")
        job.params["mode"] = "from_scratch"
        self.assertEqual(B.pill_text(job, v), "Version 5 is training")

    def test_train_button_enqueues_a_runnable_pretrain_job(self):
        spec = validate_spec(default_spec())
        job = B.catchup_job("j1", 5, spec, "catch_up", "/ckpt/best.npz", ["d1"])
        self.assertEqual((job.kind, job.params["version"], job.params["init_from"], job.params["datasets"]), ("pretrain", 5, "/ckpt/best.npz", ["d1"]))
        validate_config(copy.deepcopy(job.params["config"]))  # the engine accepts it
        scratch = B.catchup_job("j2", 5, spec, "from_scratch", "/ckpt/best.npz", ["d1"])
        self.assertIsNone(scratch.params["init_from"])
        self.assertEqual(B.text_sets({"sets": [{"id": "a"}]}), ["a"])
        self.assertEqual(B.text_sets({"sets": []}), [])


class QuickTest(unittest.TestCase):
    def test_descriptions_and_costs_use_the_exact_formulas(self):
        spec = B.write_spec(default_spec(), choices(L=4, H=4, W=1))
        self.assertEqual(Q.describe(spec), "4 blocks · 4 look-back heads · normal thinking room")
        same = B.write_spec(spec, dict(B.read_spec(spec), H=8))
        self.assertEqual(Q.cost_sentence(spec, same), "same size · same work per step")
        bigger = B.write_spec(spec, dict(B.read_spec(spec), L=8))
        self.assertIn("bigger", Q.cost_sentence(spec, bigger))

    def test_time_left_and_marks(self):
        self.assertIsNone(Q.seconds_left(0, 0, 100, 10))
        self.assertEqual(Q.seconds_left(0, 10, 100, 10), 90.)
        self.assertEqual([Q.check_mark(x) for x in (True, None, False)], ["✓", "…", "✗"])

    def test_signal_sentence_repeats_the_guess_and_a_failed_check_names_the_part(self):
        ok = {"minutes": 3, "verdict": "lower"}
        self.assertEqual(Q.signal_text(ok, "Version 4", "Lower"), "After the same 3 minutes, your idea's error is lower than Version 4's. You guessed lower.")
        self.assertIn("about the same as Version 4's. You guessed about the same", Q.signal_text(dict(ok, verdict="about the same"), "Version 4", "Same"))
        broken = Q.signal_text({"minutes": 3, "broken": "finite"}, "Version 4", "Lower")
        self.assertIn("Numbers stay finite", broken)
        self.assertIn(Q.LOOK_AT["finite"], broken)
        self.assertEqual(set(Q.LOOK_AT), {k for k, _n, _w in Q.CHECKS})


class Compare(unittest.TestCase):
    def test_cells_show_spread_and_only_clear_differences(self):
        self.assertEqual(K.final_cell(None), "[N] · spread [N]")
        row = {"bpc": [3.0, 3.1], "mean": 3.05, "spread": .1, "pieces_per_s": 1200., "peak_mb": 80., "delta": .02, "inside_spread": True}
        self.assertEqual(K.final_cell(row), "3.050 · spread 0.100\n1,200 pieces/s · 80 MB peak")
        self.assertEqual(K.final_cell(dict(row, bpc=[3.0])), "3.050 · one seed\n1,200 pieces/s · 80 MB peak")
        self.assertEqual(K.delta_cell(row), "+0.020 · inside the spread")
        self.assertEqual(K.delta_cell(dict(row, delta=-.5, inside_spread=False)), "-0.500 · beyond the spread")
        self.assertEqual(K.delta_cell(dict(row, inside_spread=None)), "+0.020 · one seed: no spread")
        self.assertEqual(K.delta_cell(None), "[N]")

    def test_guess_is_shown_next_to_the_result(self):
        beyond = {"delta": -.5, "inside_spread": False}
        self.assertEqual(K.guess_line("Better", "B", beyond), "You guessed better; B came out better (beyond the spread).")
        self.assertIn("about the same (inside the spread)", K.guess_line("Worse", "B", {"delta": .02, "inside_spread": True}))
        self.assertIn("at least two seeds", K.guess_line("Better", "B", {"delta": .5, "inside_spread": None}))
        self.assertIn("no result yet", K.guess_line("Better", "B", None))

    def test_axis_converts_to_tokens_or_minutes_and_falls_back_to_steps(self):
        full = [[1, 4., 100, 30.], [2, 3., 200, 90.]]
        self.assertEqual(K.axis_points(full, "steps"), [(1, 4.), (2, 3.)])
        self.assertEqual(K.axis_points(full, "tokens"), [(100, 4.), (200, 3.)])
        self.assertEqual(K.axis_points(full, "time"), [(.5, 4.), (1.5, 3.)])
        self.assertIsNone(K.axis_points([[1, 4.], [2, 3.]], "tokens"))
        self.assertIsNone(K.axis_points([[1, 4., None, None]], "time"))
        self.assertEqual(K.axis_for([full, full], "time"), ("time", None))
        self.assertEqual(K.axis_for([full, [[1, 4.]]], "tokens"), ("steps", "tokens"))
        self.assertEqual(K.axis_for([[[1, 4.]]], "steps"), ("steps", None))

    def test_series_are_coloured_per_arm_and_running_seeds_are_dashed(self):
        done, running = K.arm_series(1, [(1, 4.)], False), K.arm_series(2, [(1, 4.)], True)
        self.assertIsNone(done["dash"])
        self.assertIsNotNone(running["dash"])
        self.assertNotEqual(done["color"], running["color"])
        lines = plot_series({"smooth": [(0, 1), (1, 2)], "series": [dict(done, points=[(0, 1), (1, 2)]), {"points": [(0, 1)]}, running | {"points": [(0, 1), (2, 2)]}]})
        self.assertEqual([(len(pts), dash) for pts, _c, _w, dash in lines], [(2, None), (2, None), (2, (6, 4))])
        self.assertEqual(lines[1][1], C["violet"])  # arm B
        self.assertEqual(lines[2][1], C["amber"])
        self.assertEqual(plot_series({"series": [{"points": [(0, 1), (1, 1)], "color": "#123456"}]})[0][1:], ("#123456", 2, None))

    def test_the_designer_hands_its_draft_over_through_services(self):
        fake = FakeServices()
        self.assertIsNone(fake.draft_spec())
        spec = validate_spec(default_spec())
        fake.stage_draft(spec)
        self.assertEqual(fake.draft_spec(), spec)
        fake.stage_draft(None)
        self.assertIsNone(fake.draft_spec())
        fake.set_queue(QueueState())


if __name__ == "__main__":
    unittest.main()
