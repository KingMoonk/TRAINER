"""Your data and Make extra data on FakeServices: board copy, tabs, review, generate/preview, navigation with a kind."""
from tests import windows  # noqa: F401  (CI-only: skips this module locally)
import tkinter as tk
import unittest
from tkinter import ttk

from studio_ia.copy import MAKEDATA as M
from studio_ia.copy import YOURDATA as Y
from studio_ia.services import FakeServices
from studio_ia.shell import Studio


def labels(widget):
    own = [str(widget.cget("text"))] if isinstance(widget, (tk.Label, ttk.Label)) else []
    return own + [t for child in widget.winfo_children() for t in labels(child)]


def find(widget, text, kinds=(tk.Label, ttk.Label)):
    if isinstance(widget, kinds) and str(widget.cget("text")) == text:
        return widget
    return next((w for child in widget.winfo_children() if (w := find(child, text, kinds))), None)


class DataPagesTest(unittest.TestCase):
    def setUp(self):
        self.fake = FakeServices()
        try:
            self.app = Studio(self.fake)
        except tk.TclError as error:
            self.skipTest(f"no display: {error}")
        self.addCleanup(self.app.destroy)

    def click(self, widget):
        self.app.update()  # an unmapped widget drops generated events
        widget.event_generate("<Button-1>")
        self.app.update()

    def invoke(self, button):
        button.invoke()
        while self.app._waiting:  # the work runs on the shell's worker; its result arrives through the event loop
            self.app.update()

    def test_your_data_board_copy_and_tabs(self):
        page = self.app.show("your_data")
        self.app.update()
        text = labels(page)
        for want in (Y["h1"], Y["lead"], Y["text_tab"], Y["conversations_tab"], Y["problems_tab"], Y["conversations_lead"],
                     Y["conversations_note"], "‹ " + Y["back_post"], Y["search"]):
            self.assertIn(want, text)
        self.click(find(page, Y["problems_tab"]))
        text = labels(page)
        for want in (Y["problems_lead"], Y["problems_note"], "‹ " + Y["back_post"]):
            self.assertIn(want, text)
        self.assertEqual(page.add_button.cget("text"), Y["problems_add"])
        self.assertEqual(page.make_button.cget("text"), Y["problems_make"])
        self.click(find(page, Y["text_tab"]))
        self.assertIn("‹ " + Y["back_pretraining"], labels(page))
        self.assertEqual(page.table.heading("a")["text"], Y["text_col_a"])

    def test_problems_tab_offers_its_recommended_collection(self):
        self.fake._collections = {"problems": [{"id": "gsm8k_check", "name": "GSM8K", "size": "4 MB download",
                                                "records": "7,473 problems", "state": "not downloaded"}]}
        page = self.app.show("your_data")
        self.click(find(page, Y["problems_tab"]))
        self.assertIn(Y["problems_collection_desc"], labels(page))
        find(page, Y["download_add_all"], (ttk.Button,)).invoke()
        find(page, Y["download_add"], (ttk.Button,)).invoke()
        self.assertEqual(self.fake.calls, [("download_sources", None, "problems", True),
                                           ("download_sources", ["gsm8k_check"], "problems", True)])

    def test_review_row_can_be_marked_good(self):
        page = self.app.show("your_data")
        self.app.update()
        page.table.selection_set("c2")
        self.app.update()
        find(page, Y["mark_good"], (ttk.Button,)).invoke()
        self.assertEqual(next(r["status"] for r in self.fake.data("conversations").rows if r["id"] == "c2"), "Good")

    def test_make_data_cards_preview_and_generate(self):
        page = self.app.show("make_data")
        self.app.update()
        text = labels(page)
        for want in (M["h1"], M["lead"], M["problems_rule"], M["rewrite_model"], M["prose_model"], M["kicker"], M["problems_example"],
                     M["problems_where"], M["how_many"]):
            self.assertIn(want, text)
        self.invoke(page.try5)
        self.assertTrue(page.status.cget("text").startswith(M["preview"]))
        before = self.fake.data("problems").count
        self.invoke(page.go)
        self.assertTrue(page.status.cget("text").startswith("Added "))
        self.assertGreater(self.fake.data("problems").count, before)
        self.click(find(page, M["rewrite_model"]))
        self.assertIn(M["rewrite_example"], labels(page))
        self.invoke(page.go)
        self.assertIn("No local model", page.status.cget("text"))

    def test_edit_button_shows_for_a_selected_row_and_saves(self):
        page = self.app.show("your_data")
        self.assertFalse(page.edit_button.winfo_ismapped())
        page.table.selection_set("c1")
        page.table.event_generate("<<TreeviewSelect>>")
        self.app.update()
        self.assertEqual(page.edit_button.cget("text"), Y["edit"])
        self.assertTrue(page.edit_button.winfo_ismapped())
        answers = iter(["Name a prime.", "11 is a prime number."])
        page._ask = lambda _prompt, initial="": next(answers)
        page.edit_button.invoke()
        self.assertEqual(self.fake.edits[-1], ("c1", {"user": "Name a prime.", "assistant": "11 is a prime number."}))

    def test_navigation_carries_the_kind(self):
        page = self.app.show("your_data")
        self.click(find(page, Y["text_tab"]))
        page.make_button.invoke()
        self.assertEqual(self.app.route, "make_data")
        self.assertEqual(self.app.pages["make_data"].k, 2)
        self.app.pages["make_data"].open_data("problems")
        self.assertEqual((self.app.route, self.app.pages["your_data"].kind), ("your_data", "problems"))


if __name__ == "__main__":
    unittest.main()
