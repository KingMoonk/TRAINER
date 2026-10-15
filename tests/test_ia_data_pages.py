"""Your data / Make extra data: decision logic and the data methods of the services (no Tk window)."""
import random
import tempfile
import unittest

from studio_core import Workspace
from studio_ia import copy
from studio_ia.pages import make_data as md
from studio_ia.pages import your_data as yd
from studio_ia.services import FakeServices, WorkspaceServices, problem_items, problem_report
from studio_ia.shell import ROUTES


class RoutesTest(unittest.TestCase):
    def test_titles_match_routes(self):
        for route, page in (("your_data", yd.YourDataPage), ("make_data", md.MakeDataPage)):
            self.assertEqual((page.title, page.lead), (ROUTES[route].title, ROUTES[route].lead))


class YourDataLogicTest(unittest.TestCase):
    def test_back_link_follows_the_stage(self):
        self.assertEqual(yd.back_link("text")[1], "pretraining")
        self.assertEqual(yd.back_link("conversations")[1], "post_training")
        self.assertEqual(yd.back_link("problems")[1], "post_training")

    def test_visible_rows_and_labels(self):
        rows = [{"status": "In use"}, {"status": "Near-duplicate"}]
        self.assertEqual(yd.visible_rows(rows, True), [rows[1]])
        self.assertEqual(yd.visible_rows(rows, False), rows)
        self.assertEqual(yd.dup_label(3, False), "Near-duplicates (3 found)")
        self.assertTrue(yd.dup_label(3, True).endswith("showing"))

    def test_parse_rule_takes_key_or_label(self):
        self.assertEqual(yd.parse_rule("small addition"), "small_addition")
        self.assertEqual(yd.parse_rule(" Reverse "), "reverse")
        self.assertEqual(yd.parse_rule("stay under N characters"), "max_length")
        self.assertIsNone(yd.parse_rule("poetry"))
        self.assertIsNone(yd.parse_rule(None))

    def test_collect_fields(self):
        said = []
        answers = iter(["Reverse: abc", "reverse"])
        self.assertEqual(yd.collect_fields("problems", lambda _t, _i='': next(answers), said.append), {"prompt": "Reverse: abc", "rule": "reverse"})
        answers = iter(["Reverse: abc", "poetry"])
        self.assertIsNone(yd.collect_fields("problems", lambda _t, _i='': next(answers), said.append))
        self.assertEqual(len(said), 1)
        self.assertIsNone(yd.collect_fields("text", lambda _t, _i='': "   ", said.append))
        self.assertIsNone(yd.collect_fields("conversations", lambda _t, _i='': None, said.append))
        answers = iter(["Hi", "Hello"])
        self.assertEqual(yd.collect_fields("conversations", lambda _t, _i='': next(answers), said.append), {"user": "Hi", "assistant": "Hello"})
        shown = []
        yd.collect_fields("problems", lambda t, i='': shown.append(i) or "reverse", said.append, {"prompt": "Reverse: abc", "rule": "reverse"})
        self.assertEqual(shown, ["Reverse: abc", "reverse"])

    def test_every_tab_has_its_board_copy(self):
        for k in yd.KINDS:
            for suffix in ("tab", "note", "add", "make", "col_a", "col_b", "search"):
                self.assertTrue(copy.YOURDATA[f"{k}_{suffix}"])
        for status in yd.STATUS_COLOR:
            self.assertIn(status, ("In use", "Held back", "Near-duplicate", "Good", "Review", "From Chat", "Ready"))


class MakeDataLogicTest(unittest.TestCase):
    def test_only_rule_made_problems_skip_review(self):
        self.assertEqual([md.stores_directly(c) for c in md.CARDS], [True, False, False])
        self.assertEqual([md.card_for(k) for k in ("problems", "conversations", "text", "nope")], [0, 1, 2, 0])

    def test_parse_count(self):
        self.assertEqual(md.parse_count(" 25 "), 25)
        for bad in ("0", "-3", "x", "", "100001", "2.5"):
            self.assertIsNone(md.parse_count(bad), bad)

    def test_problem_items_spread_over_every_rule(self):
        items = problem_items(None, 10, random.Random(1))
        self.assertEqual(len(items), 10)
        self.assertEqual({i["rule"] for i in items}, set(md.rules.RULES))
        self.assertEqual({i["rule"] for i in problem_items("sort", 4, random.Random(1))}, {"sort"})

    def test_report_counts_k_of_n_checked_answers(self):
        report = problem_report(problem_items(None, 14, random.Random(2)))
        self.assertEqual(report["checked"], 10)  # max_length and ends_cleanly have no canonical answer
        self.assertEqual(report["passed"], 10)
        lines = md.preview_lines(report)
        self.assertEqual(len(lines), 7)
        self.assertEqual(lines[-1], "10 of 10 sample answers pass their own rule.")
        self.assertIn("(rule decides)", "".join(md.preview_lines(problem_report(problem_items("ends_cleanly", 2, random.Random(0))))))

    def test_summaries(self):
        self.assertEqual(md.problems_summary({"added": 5, "duplicates": 0}), "Added 5 problems to Your data → Problems.")
        self.assertIn("(2 duplicates skipped)", md.problems_summary({"added": 3, "duplicates": 2}))
        self.assertIn("Nothing is used until you mark it Good", md.drafts_summary("text", 4))


class FakeServicesDataTest(unittest.TestCase):
    def setUp(self):
        self.svc = FakeServices()

    def test_canned_data_renders_every_kind(self):
        for kind in yd.KINDS:
            s = self.svc.data(kind)
            self.assertTrue(s.rows and s.count_line and s.used_by)
        self.assertEqual([r["a"] for r in self.svc.data("problems", "reverse").rows], ["Reverse: groundwork"])

    def test_generate_preview_then_store_and_dedupe(self):
        before = self.svc.data("problems").count
        self.assertEqual(self.svc.generate_problems("sort", 3, store=False, rng=random.Random(0))["added"], 0)
        self.assertEqual(self.svc.data("problems").count, before)
        self.assertEqual(self.svc.generate_problems("sort", 3, rng=random.Random(0))["added"], 3)
        again = self.svc.generate_problems("sort", 3, rng=random.Random(0))
        self.assertEqual((again["added"], again["duplicates"]), (0, 3))

    def test_drafts_need_a_model_then_wait_for_review(self):
        with self.assertRaises(RuntimeError):
            self.svc.make_drafts("conversations", 2)
        self.svc.draft_source = lambda kind, n: [{"user": f"q{i}", "assistant": f"a{i}"} for i in range(n)]
        self.assertEqual(self.svc.make_drafts("conversations", 2), 2)
        self.assertEqual(self.svc.draft_count(), 2)
        draft = next(r for r in self.svc.data("conversations").rows if r["status"] == "Review" and r["a"] == "q0")
        self.svc.review_data(draft["id"], True)
        self.assertEqual(self.svc.draft_count(), 1)
        other = next(r for r in self.svc.data("conversations").rows if r["a"] == "q1")
        self.svc.review_data(other["id"], False)
        self.assertEqual((self.svc.draft_count(), [r["a"] for r in self.svc.data("conversations").rows if r["a"] == "q1"]), (0, []))

    def test_add_data_rejects_empty_and_duplicates(self):
        self.assertTrue(self.svc.add_data("conversations", {"user": "x", "assistant": "y"}))
        self.assertFalse(self.svc.add_data("conversations", {"user": "x", "assistant": "y"}))
        self.assertFalse(self.svc.add_data("text", {"text": " "}))

    def test_edit_data_updates_the_row_and_refuses_empty_or_duplicate(self):
        self.assertEqual(self.svc.data_fields("c1"), {"user": "Name a prime.", "assistant": "7 is a prime number."})
        self.svc.edit_data("c1", {"user": "Name a prime.", "assistant": "11 is a prime number."})
        self.assertEqual(self.svc.edits[-1][0], "c1")
        self.assertEqual(self.svc.data("conversations", "11 is").rows[0]["id"], "c1")
        with self.assertRaises(ValueError):
            self.svc.edit_data("c1", {"user": "Say hello.", "assistant": "Hello there."})
        with self.assertRaises(ValueError):
            self.svc.edit_data("c1", {"user": " ", "assistant": "x"})
        self.svc.edit_data("c2", {"user": "Say hello.", "assistant": "Hi."})
        self.assertEqual(self.svc.data("conversations", "Hi.").rows[0]["status"], "Review")


class WorkspaceServicesDataTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.svc = WorkspaceServices(Workspace(self.tmp.name))
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(self.svc.close)

    def test_kinds_stay_apart_and_problems_keep_their_rule(self):
        self.assertTrue(self.svc.add_data("text", {"text": "The tide came in."}))
        self.assertTrue(self.svc.add_data("conversations", {"user": "Hi", "assistant": "Hello"}))
        self.assertTrue(self.svc.add_data("problems", {"prompt": "Reverse: abc", "rule": "reverse"}))
        self.assertFalse(self.svc.add_data("problems", {"prompt": "Reverse: abc", "rule": "reverse"}))
        self.assertEqual({k: self.svc.data(k).count for k in yd.KINDS}, {"text": 1, "conversations": 1, "problems": 1})
        self.assertEqual(self.svc.data("problems").rows[0]["b"], "reverse")
        self.assertEqual(self.svc.data("conversations").rows[0]["status"], "Good")

    def test_generate_stores_rule_problems_and_skips_duplicates(self):
        first = self.svc.generate_problems("small_addition", 5, rng=random.Random(3))
        self.assertEqual((first["added"] + first["duplicates"], first["passed"], first["checked"]), (5, 5, 5))
        again = self.svc.generate_problems("small_addition", 5, rng=random.Random(3))
        self.assertEqual((again["added"], again["duplicates"]), (0, 5))
        self.assertEqual(self.svc.data("problems").count, first["added"])
        self.assertEqual(self.svc.generate_problems(None, 7, store=False)["added"], 0)

    def test_review_keep_and_discard(self):
        ws = self.svc.workspace
        ds = ws.create_dataset("Rewritten conversations")
        ws.add(ds, [{"user": "a", "assistant": "b", "quality": "Review", "synthetic": True}, {"user": "c", "assistant": "d", "quality": "Review", "synthetic": True}], "synthetic")
        self.assertEqual(self.svc.draft_count(), 2)
        rows = self.svc.data("conversations").rows
        self.svc.review_data(rows[0]["id"], True)
        self.svc.review_data(rows[1]["id"], False)
        self.assertEqual(self.svc.draft_count(), 0)
        self.assertEqual([r["status"] for r in self.svc.data("conversations").rows], ["Good"])

    def test_unreviewed_prose_drafts_read_as_review(self):
        ws = self.svc.workspace
        ds = ws.create_dataset("Generated prose")
        ws.add(ds, [{"text": "A paragraph about tides.", "quality": "Review", "synthetic": True}], "synthetic")
        self.assertEqual(self.svc.data("text").rows[0]["status"], "Review")

    def test_edit_data_per_kind(self):
        self.svc.add_data("text", {"text": "The tide came in."})
        self.svc.add_data("conversations", {"user": "Hi", "assistant": "Hello"})
        self.svc.add_data("problems", {"prompt": "Reverse: abc", "rule": "reverse"})
        ids = {k: self.svc.data(k).rows[0]["id"] for k in yd.KINDS}
        self.assertEqual(self.svc.data_fields(ids["text"]), {"text": "The tide came in."})
        self.svc.edit_data(ids["text"], {"text": "The tide went out."})
        self.svc.edit_data(ids["conversations"], {"user": "Hey", "assistant": "Hello there"})
        self.assertEqual(self.svc.data_fields(ids["text"]), {"text": "The tide went out."})
        self.assertEqual(self.svc.data_fields(ids["conversations"]), {"user": "Hey", "assistant": "Hello there"})
        self.assertEqual(self.svc.data_fields(ids["problems"]), {"prompt": "Reverse: abc", "rule": "reverse"})
        self.assertEqual({k: self.svc.data(k).count for k in yd.KINDS}, {"text": 1, "conversations": 1, "problems": 1})
        with self.assertRaises(ValueError):
            self.svc.edit_data(ids["conversations"], {"user": " ", "assistant": "x"})

    def test_edit_problem_must_still_pass_its_rule(self):
        self.svc.add_data("problems", {"prompt": "Reverse: abc", "rule": "reverse"})
        self.svc.generate_problems("reverse", 1, rng=random.Random(1))
        row = next(r for r in self.svc.data("problems").rows if r["a"] != "Reverse: abc")
        word = row["a"].split(": ")[1]
        self.svc.edit_data(row["id"], {"prompt": f"Reverse: {word}", "rule": "reverse"})  # unchanged prompt still passes
        with self.assertRaises(ValueError) as caught:
            self.svc.edit_data(row["id"], {"prompt": "Reverse: zzzzq", "rule": "reverse"})
        self.assertIn("reverse", str(caught.exception))
        self.assertEqual(self.svc.data_fields(row["id"])["prompt"], f"Reverse: {word}")

    def test_edited_draft_stays_a_draft_until_marked_good(self):
        ws = self.svc.workspace
        ws.add(ws.create_dataset("Rewritten conversations"), [{"user": "a", "assistant": "b", "quality": "Review", "synthetic": True}], "synthetic")
        row = self.svc.data("conversations").rows[0]
        self.svc.edit_data(row["id"], {"user": "a2", "assistant": "b2"})
        edited = self.svc.data("conversations").rows[0]
        self.assertEqual((edited["a"], edited["status"], self.svc.draft_count()), ("a2", "Review", 1))
        self.svc.review_data(edited["id"], True)
        self.assertEqual((self.svc.data("conversations").rows[0]["status"], self.svc.draft_count()), ("Good", 0))


if __name__ == "__main__":
    unittest.main()
