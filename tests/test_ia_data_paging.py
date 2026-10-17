"""Your data pages through storage: the SQL page of a kind, the search debounce and its stale-result token, the progress throttle.
No Tk window: the page's own logic is plain classes that take the clock from the caller."""
import tempfile
import unittest

import studio_data_kinds as dk
from studio_core import Workspace
from studio_ia.services import FakeServices, WorkspaceServices


def talk(n, prefix="Q"):
    return [{"user": f"{prefix} {i}: tell me about tide {i}", "assistant": f"Answer {i} about the moon", "quality": "Good"} for i in range(n)]


def pages(workspace, kind, query="", sets=None, limit=50, dups=()):
    """Every page of a search, following the cursor: (list of pages, flat rows)."""
    sets = sets or [(d["id"], d["name"]) for d in workspace.state["datasets"]]
    out, cursor = [], None
    while True:
        rows, cursor = dk.page(workspace, kind, query, sets, cursor, limit, dups)
        out.append(rows)
        if cursor is None:
            return out, [r for page in out for r in page]


class PageTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = Workspace(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.disk = self.ws.create_dataset("Talk", storage="sqlite")
        self.json = self.ws.create_dataset("Notes")
        self.ws.add(self.disk, talk(120), "manual")
        self.ws.add(self.json, talk(30, "N"), "manual")

    def test_pages_cover_every_row_once_across_datasets(self):
        out, rows = pages(self.ws, "conversations")
        self.assertEqual([len(p) for p in out], [50, 50, 50])
        self.assertEqual(len(rows), 150)
        self.assertEqual(len({r["id"] for r in rows}), 150)
        self.assertEqual([r["dataset"] for r in rows], ["Talk"] * 120 + ["Notes"] * 30)
        self.assertEqual(rows[0]["a"], "Q 0: tell me about tide 0")

    def test_last_full_page_has_no_next(self):
        rows, cursor = dk.page(self.ws, "conversations", "", [(self.disk, "Talk")], None, 120)
        self.assertEqual((len(rows), cursor), (120, None))
        rows, cursor = dk.page(self.ws, "conversations", "", [(self.disk, "Talk")], None, 119)
        self.assertEqual((len(rows), cursor is None), (119, False))

    def test_search_reaches_rows_the_summary_sample_never_holds(self):
        summary = dk.summarize(self.ws, "conversations", "tide 119", scan_limit=40)
        self.assertEqual(summary.rows, [])  # the sample stops at 40 rows, so the old search could not see it
        _, rows = pages(self.ws, "conversations", "tide 119")
        self.assertEqual([r["a"] for r in rows], ["Q 119: tell me about tide 119"])

    def test_search_pages_and_matches_the_answer_column(self):
        out, rows = pages(self.ws, "conversations", "MOON", limit=40)  # ASCII case folds in SQL
        self.assertEqual((len(rows), [len(p) for p in out]), (150, [40, 40, 40, 30]))
        self.assertEqual(len(pages(self.ws, "conversations", "tide 7")[1]), 12)  # "tide 7" and "tide 70-79" in Talk, "tide 7" in Notes

    def test_query_with_like_characters_is_literal(self):
        self.ws.add(self.disk, [{"user": "100% sure", "assistant": "a_b", "quality": "Good"}], "manual")
        self.assertEqual(len(pages(self.ws, "conversations", "100%")[1]), 1)
        self.assertEqual(len(pages(self.ws, "conversations", "a_b")[1]), 1)

    def test_other_kinds_stay_out(self):
        self.ws.add(self.disk, [{"text": "The heron waited by the tide.", "quality": "Good"}], "manual")
        self.assertEqual(len(pages(self.ws, "text", "tide")[1]), 1)
        self.assertEqual(len(pages(self.ws, "conversations", "heron")[1]), 0)

    def test_problems_match_their_rule_label(self):
        dk.add_problems(self.ws, self.disk, [{"prompt": "Reverse: abc", "rule": "reverse", "params": {}},
                                             {"prompt": "17 + 25 =", "rule": "small_addition", "params": {}}])
        _, rows = pages(self.ws, "problems", "small addition")
        self.assertEqual([r["a"] for r in rows], ["17 + 25 ="])
        _, rows = pages(self.ws, "problems", "reverse")
        self.assertEqual([r["a"] for r in rows], ["Reverse: abc"])

    def test_json_datasets_match_in_python_with_full_casefold(self):
        self.ws.add(self.json, [{"user": "Straße am Meer", "assistant": "x", "quality": "Good"}], "manual")
        self.assertEqual(len(pages(self.ws, "conversations", "STRASSE")[1]), 1)  # casefold("ß") is "ss"
        self.assertEqual(len(pages(self.ws, "conversations", "STRAßE")[1]), 1)

    def test_known_duplicates_are_flagged(self):
        self.ws.add(self.disk, [{"text": "The heron waited by the tide.", "quality": "Good"}], "manual")
        row = pages(self.ws, "text", "heron", dups={"nope"})[1][0]
        self.assertEqual(row["status"], "In use")
        self.assertEqual(pages(self.ws, "text", "heron", dups={row["id"]})[1][0]["status"], "Near-duplicate")


    def test_review_filter_precedes_pagination_on_disk_and_in_memory(self):
        for dataset in (self.disk, self.json):
            self.ws.add(dataset, [
                {"user": f"Draft {dataset} {i}", "assistant": "Needs review", "quality": "Review", "synthetic": True}
                for i in range(2)
            ], "synthetic")
            self.ws.add(dataset, [{"user": f"Ordinary {dataset}", "assistant": "Not a model draft", "quality": "Review"}], "manual")
        sets = [(self.disk, "Talk"), (self.json, "Notes")]
        found, cursor = [], None
        while True:
            rows, cursor = dk.page(self.ws, "conversations", sets=sets, cursor=cursor, limit=1, review_only=True)
            found.extend(rows)
            if cursor is None:
                break
        self.assertEqual(len(found), 4)
        self.assertTrue(all(row["a"].startswith("Draft ") and row["status"] == "Review" for row in found))
        self.ws.mark_reviewed([found[0]["id"]])
        rows, _ = dk.page(self.ws, "conversations", sets=sets, limit=10, review_only=True)
        self.assertEqual(len(rows), 3)


class WorkspaceServicesPageTest(unittest.TestCase):
    def test_data_page_reads_the_sets_of_the_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            svc = WorkspaceServices(Workspace(tmp))
            try:
                svc.add_data("conversations", {"user": "Hi there", "assistant": "Hello"})
                svc.add_data("conversations", {"user": "Bye now", "assistant": "Later"})
                rows, cursor = svc.data_page("conversations", "bye")
                self.assertEqual(([r["a"] for r in rows], cursor), (["Bye now"], None))
                rows, cursor = svc.data_page("conversations", "", None, 1)
                self.assertEqual((len(rows), cursor is not None), (1, True))
                self.assertEqual(len(svc.data_page("conversations", "", cursor, 1)[0]), 1)
                self.assertEqual(svc.data_page("text")[0], [])
            finally:
                svc.close()

    def test_fake_pages_the_same_way(self):
        fake = FakeServices()
        rows, cursor = fake.data_page("conversations", "", None, 2)
        self.assertEqual((len(rows), cursor), (2, 2))
        rows, cursor = fake.data_page("conversations", "", cursor, 2)
        self.assertEqual((len(rows), cursor), (1, None))
        self.assertEqual([r["a"] for r in fake.data_page("conversations", "hello")[0]], ["Say hello."])


class Clock:
    """after/after_cancel stand-in: timers run only when the test says so."""

    def __init__(self):
        self.timers, self.next = {}, 0

    def after(self, ms, fn):
        self.next += 1
        self.timers[self.next] = (ms, fn)
        return self.next

    def cancel(self, handle):
        self.timers.pop(handle, None)

    def fire(self):
        for handle in sorted(self.timers):
            self.timers.pop(handle)[1]()


if __name__ == "__main__":
    unittest.main()
