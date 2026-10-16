"""Build your model's learning aids (#83): click a part to read what it does, where the parameters go, starting questions
that set up one change, and module menus that open once a few versions have trained. Plain functions and page methods on
stand-ins: no window opens."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from studio_core import Workspace
from studio_estimate import estimate
from studio_ia import pieces as cut
from studio_ia import design
from studio_ia.copy import BUILDER as T
from studio_ia.pages import builder as B
from studio_ia.services import FakeServices, Version, WorkspaceServices
from studio_spec import default_spec

CUR = B.read_spec(default_spec())
LISTS = {"tokenizers": (cut.Tokenizer("byte"), None, None, None), "sentence": "The cat sat."}


def version(number, s=None, trained=True, prediction=None):
    return Version(number, spec=s or default_spec(), scorecard={"bpc": 2.0} if trained else {}, prediction=prediction or {},
                   best_checkpoint=f"/runs/{number}/best.npz" if trained else None)


class Explain(unittest.TestCase):
    def test_every_part_says_what_it_does_its_module_and_its_example(self):
        for part in B.PART_LINKS:
            block = 0 if part in ("mixer", "channel", "norm") else None
            e = B.explain(part, CUR, block, LISTS)
            self.assertTrue(e["what"] and e["module"] and e["module_text"] and e["example"], part)
            self.assertTrue(e["links"], part)

    def test_the_words_come_from_the_glossary_and_the_registry(self):
        e = B.explain("mixer", CUR, 1)
        self.assertEqual(e["what"], B.part_tip("mixer"))
        self.assertEqual(e["module_text"], B.module_tip("mixer", CUR["M"]["mixer"]))
        self.assertEqual(e["module"], T["explain_now"].format(name=B.module_name("mixer", CUR["M"]["mixer"])))
        self.assertIn("2", e["title"])  # block 2
        final = B.explain("final_norm", CUR)
        self.assertIn(T["final_norm_tip"], final["what"])

    def test_a_block_explains_its_own_module(self):
        other = next(m.short for m in B.registry.for_slot("mixer") if m.short != CUR["M"]["mixer"])
        s = B.choose(CUR, "mixer", other, 1)
        self.assertEqual(B.explain("mixer", s, 1)["module_text"], B.module_tip("mixer", other))
        self.assertEqual(B.explain("mixer", s, 0)["module_text"], B.module_tip("mixer", CUR["M"]["mixer"]))

    def test_the_head_pages_open_on_the_clicked_block_and_only_for_a_look_back_with_heads(self):
        routes = [(route, at) for route, _label, at in B.explain("mixer", CUR, 1)["links"]]
        self.assertEqual(routes, [("head_zoom", 1), ("value_map", 1), ("head_lab", 1), ("inside3d", 1)])
        headless = next((m.short for m in B.registry.for_slot("mixer") if "heads" not in B.fields_of("mixer", m.short)), None)
        if headless:
            s = B.choose(CUR, "mixer", headless)
            self.assertFalse({r for r, _, _ in B.explain("mixer", s, 0)["links"]} & set(B.HEAD_PAGES))

    def test_every_look_inside_page_hangs_off_a_part(self):
        linked = {route for routes in B.PART_LINKS.values() for route in routes}
        self.assertEqual(linked, {"trace", "head_zoom", "value_map", "loss", "head_lab", "inside3d"})
        self.assertEqual([r for r, _ in B.LINKS], ["compare", "quicktest"])

    def test_the_example_uses_the_sentence_cut_by_the_draft_word_list(self):
        pieces = B.example_pieces(0, LISTS)
        self.assertEqual("".join(pieces).replace("␣", " "), "The cat sat.")  # chips show a space as ␣
        table = B.explain("embedding", CUR, None, LISTS)["example"]
        self.assertIn(f"{len(pieces)} pieces", table)
        self.assertIn("256", table)  # letters: one row per byte
        self.assertIn(str(len(pieces) - 1), B.explain("mixer", CUR, 0, LISTS)["example"])  # pieces before the last one

    def test_without_word_lists_the_example_still_reads(self):
        self.assertIsNone(B.example_pieces(1, None))
        self.assertIsNone(B.example_pieces(1, LISTS))  # that list is not built yet
        s = dict(CUR, V=1)
        self.assertEqual(B.explain("embedding", s)["example"],
                         T["example_table"].format(d=s["d"], upto=T["up_to"], rows=f"{B.VS[1]:,}"))
        self.assertIn(f"{s['H']} heads", B.explain("mixer", s, 0)["example"])

    def test_thinking_room_shows_its_width(self):
        s = dict(CUR, W=2)
        self.assertIn(str(B.FF[2] * s["d"]), B.explain("channel", s, 0)["example"])

    def test_a_skipped_norm_says_so(self):
        skip = next((m.short for m in B.registry.for_slot("norm") if m.skip), None)
        if skip is None:
            self.skipTest("no skipping norm registered")
        self.assertEqual(B.explain("norm", B.choose(CUR, "norm", skip), 0)["example"], T["example_skipped"])


class PanelOnStandIns(unittest.TestCase):
    def test_a_head_link_opens_inside_a_head_at_the_block(self):
        zoom = mock.Mock()
        page = SimpleNamespace(navigate=mock.Mock(return_value=zoom))
        B.BuilderPage.follow(page, "head_zoom", 2)
        page.navigate.assert_called_once_with("head_zoom")
        zoom.select.assert_called_once_with(2, 0)
        value = mock.Mock(spec=["pick"])
        page.navigate = mock.Mock(return_value=value)
        B.BuilderPage.follow(page, "value_map", 1)
        value.pick.assert_called_once_with(1, 0)

    def test_other_links_only_navigate(self):
        trace = mock.Mock(spec=[])
        page = SimpleNamespace(navigate=mock.Mock(return_value=trace))
        B.BuilderPage.follow(page, "trace", 0)
        page.navigate.assert_called_once_with("trace")


class Params(unittest.TestCase):
    def test_the_parts_add_up_to_the_size(self):
        for s in (CUR, dict(CUR, V=2), B.resize(CUR, 5)):
            spec = B.write_spec(default_spec(), s, CUR)
            c = B.sizes(spec, s["V"])
            self.assertEqual(sum(c["parts"].values()), c["params"])
            self.assertEqual(c["parts"]["table"] + c["parts"]["position"], c["table"])

    def test_blocks_hold_every_block_row(self):
        spec = default_spec()
        rows = estimate(spec, 256)["rows"]
        parts = design.param_parts(rows)
        self.assertEqual(parts["blocks"], sum(r["params"] for r in rows if r["block"] is not None))
        self.assertEqual(parts["head"], sum(r["params"] for r in rows if r["block"] is None and r["slot"] in ("norm", "head")))

    def test_the_bars_share_one_scale(self):
        a = dict(table=100, position=0, blocks=100, head=0)
        b = dict(table=100, position=0, blocks=300, head=0)
        now, new = B.param_bars(a, b, width=400)
        self.assertEqual(dict(new), dict(table=100, position=0, blocks=300, head=0))
        self.assertEqual(dict(now), dict(table=100, position=0, blocks=100, head=0))
        self.assertEqual([k for k, _ in now], [k for k, _ in B.PARAM_PARTS])

    def test_shares_and_the_word_table_note(self):
        parts = dict(table=60, position=0, blocks=40, head=0)
        self.assertEqual(dict(B.param_shares(parts))["table"], T["params_share"].format(name=T["params_table"], share="60%"))
        self.assertEqual(B.param_note(parts), T["params_table_heavy"])
        self.assertEqual(B.param_note(dict(parts, table=10)), "")

    def test_a_big_word_list_moves_the_size_into_the_table(self):
        small, large = (B.sizes(B.write_spec(default_spec(), dict(CUR, V=v), CUR), v)["parts"] for v in (0, 2))
        self.assertGreater(large["table"], small["table"])
        self.assertEqual(large["blocks"], small["blocks"])


class Experiments(unittest.TestCase):
    def test_each_question_is_one_change(self):
        for key, _q in B.EXPERIMENTS:
            draft, why = B.experiment_draft(CUR, key)
            self.assertEqual(why, "", key)
            self.assertEqual(len(B.change_chips(default_spec(), B.write_spec(default_spec(), draft, CUR), CUR, draft)), 1, key)

    def test_the_change_each_question_makes(self):
        self.assertEqual(B.experiment_draft(dict(CUR, H=4), "heads")[0]["H"], 8)
        self.assertEqual(B.experiment_draft(dict(CUR, H=8), "heads")[0]["H"], 4)  # already the most: fewer answers it too
        self.assertEqual(B.experiment_draft(dict(CUR, V=0), "words")[0]["V"], 1)
        self.assertEqual(B.experiment_draft(dict(CUR, V=2), "words")[0]["V"], 0)
        self.assertEqual(B.experiment_draft(CUR, "depth")[0]["L"], CUR["L"] + 1)
        self.assertEqual(B.experiment_draft(B.resize(CUR, 12), "depth")[0]["L"], 11)
        self.assertEqual(B.experiment_draft(dict(CUR, W=1), "width")[0]["W"], 2)
        self.assertEqual(B.experiment_draft(dict(CUR, W=2), "width")[0]["W"], 1)

    def test_a_look_back_without_heads_cannot_ask_about_heads(self):
        headless = next((m.short for m in B.registry.for_slot("mixer") if "heads" not in B.fields_of("mixer", m.short)), None)
        if headless is None:
            self.skipTest("every look-back has heads")
        draft, why = B.experiment_draft(B.choose(CUR, "mixer", headless), "heads")
        self.assertIsNone(draft)
        self.assertEqual(why, T["exp_no_heads"])

    def test_depth_and_width_ask_one_question(self):
        rows = B.experiment_rows(CUR)
        self.assertEqual([q for q, _ in rows], [T["exp_heads"], T["exp_words"], T["exp_depth"]])
        self.assertEqual([k for k, _, _ in rows[2][1]], ["depth", "width"])

    def test_the_question_holds_only_while_the_draft_is_that_change(self):
        draft = B.experiment_draft(CUR, "depth")[0]
        held = (T["exp_depth"], copy.deepcopy(draft))
        self.assertEqual(B.held_question(held, draft), T["exp_depth"])
        self.assertIsNone(B.held_question(held, dict(draft, H=8)))
        self.assertIsNone(B.held_question(None, draft))

    def test_try_question_sets_the_draft_and_save_stores_the_question(self):
        services = FakeServices()
        v1 = services.save_version(default_spec(), "first design")
        page = SimpleNamespace(phase="edit", cur=B.read_spec(v1.spec), draft=None, anchor=("x", {}), experiment=None, note=None,
                               stage=mock.Mock(), _paint_design=mock.Mock(), _paint_tray=mock.Mock())
        B.BuilderPage.try_question(page, "depth")
        self.assertEqual(page.draft["L"], page.cur["L"] + 1)
        self.assertIsNone(page.anchor)
        self.assertEqual(page.experiment[0], T["exp_depth"])
        page.draft.update(H=8)  # the page edits the draft in place (edit()): the held copy must not move with it
        self.assertIsNone(B.held_question(page.experiment, page.draft))

    def test_save_merges_the_question_with_the_guesses(self):
        services = FakeServices()
        base = services.save_version(default_spec(), "first design")
        cur = B.read_spec(base.spec)
        draft = B.experiment_draft(cur, "words")[0]
        page = SimpleNamespace(services=services, base=base, cur=cur, draft=draft, anchor=None, note=None,
                               experiment=(T["exp_words"], copy.deepcopy(draft)), phase="edit", _staged=None,
                               load_chart=mock.Mock(), render=mock.Mock())
        page.spec_for = lambda s: B.write_spec(base.spec, s, cur)
        page.base_spec = lambda: B.validate_spec(base.spec)
        page.changes = lambda: B.BuilderPage.changes(page)
        B.BuilderPage.save(page, {"error": "lower"})
        self.assertEqual(page.saved.prediction, {"error": "lower", "question": T["exp_words"]})
        self.assertIsNone(page.experiment)

    def test_save_without_a_question_stores_only_the_guesses(self):
        services = FakeServices()
        base = services.save_version(default_spec(), "first design")
        cur = B.read_spec(base.spec)
        page = SimpleNamespace(services=services, base=base, cur=cur, draft=B.resize(cur, cur["L"] + 1), anchor=None, note=None,
                               experiment=None, phase="edit", _staged=None, load_chart=mock.Mock(), render=mock.Mock())
        page.spec_for = lambda s: B.write_spec(base.spec, s, cur)
        page.base_spec = lambda: B.validate_spec(base.spec)
        page.changes = lambda: B.BuilderPage.changes(page)
        B.BuilderPage.save(page)
        self.assertEqual(page.saved.prediction, {})

    def test_the_notebook_and_the_check_show_the_question(self):
        old = version(1, trained=True)
        new = version(2, B.write_spec(default_spec(), B.resize(CUR, CUR["L"] + 1), CUR), trained=False,
                      prediction={"error": "lower", "question": T["exp_depth"]})
        rows = B.notebook([old, new])
        self.assertEqual(rows[1]["question"], T["exp_depth"])
        self.assertIn(T["notebook_asked"].format(question=T["exp_depth"]), B.notebook_line(rows[1]))
        self.assertNotIn("asked", B.notebook_line(rows[0]))
        _state, _kicker, text = B.check_text(old, new)
        self.assertTrue(text.startswith(T["check_question"].format(question=T["exp_depth"])))

    def test_the_question_is_not_a_guess(self):
        old, new = version(1), version(2, B.write_spec(default_spec(), B.resize(CUR, CUR["L"] + 1), CUR),
                                      prediction={"question": T["exp_depth"]})
        self.assertIsNone(B.guess_check(old, new))
        self.assertEqual(B.notebook([old, new])[1]["guesses"], T["notebook_no_guess"])


class Disclosure(unittest.TestCase):
    def test_menus_open_after_a_few_trained_versions(self):
        few = [version(n) for n in range(1, B.UNLOCK)]
        self.assertFalse(B.advanced_view(few))
        self.assertTrue(B.advanced_view(few + [version(B.UNLOCK)]))
        self.assertFalse(B.advanced_view(few + [version(B.UNLOCK, trained=False)]))  # saved, not trained

    def test_the_toggle_wins_either_way(self):
        many = [version(n) for n in range(1, B.UNLOCK + 2)]
        self.assertFalse(B.advanced_view(many, False))
        self.assertTrue(B.advanced_view([], True))

    def test_the_toggle_is_written_off_the_tk_thread_and_shown_at_once(self):
        queued, services = [], mock.Mock()
        page = SimpleNamespace(services=services, advanced_set=None, render=mock.Mock(),
                               quietly=lambda work, then, lane="fast": queued.append((work, then, lane)))
        B.BuilderPage.set_advanced(page, True)
        self.assertTrue(page.advanced_set)
        page.render.assert_called_once()
        services.set_setting.assert_not_called()  # nothing is written until the worker runs
        work, then, lane = queued[0]
        self.assertEqual(lane, "slow")
        then(work())
        services.set_setting.assert_called_once_with(B.ADVANCED, True)

    def test_services_remember_the_setting(self):
        fake = FakeServices()
        self.assertIsNone(fake.setting(B.ADVANCED))
        fake.set_setting(B.ADVANCED, True)
        self.assertTrue(fake.setting(B.ADVANCED))

    def test_the_workspace_keeps_the_setting_on_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            svc = WorkspaceServices(Workspace(tmp))
            self.assertEqual(svc.setting(B.ADVANCED, "unset"), "unset")
            svc.set_setting(B.ADVANCED, False)
            saved = json.loads((Path(tmp) / "data" / "workspace.json").read_text(encoding="utf-8"))
            self.assertIs(saved["studioConfig"][B.ADVANCED], False)
            self.assertIs(WorkspaceServices(Workspace(tmp)).setting(B.ADVANCED), False)


if __name__ == "__main__":
    unittest.main()
