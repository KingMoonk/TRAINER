"""Your data's table and recommended collection update in place on a refresh instead of being rebuilt (no window)."""
import unittest
from types import SimpleNamespace

from studio_ia.copy import YOURDATA as T
from studio_ia.pages import your_data as yd
from studio_ia.pages.pretraining import fill_collection_row
from studio_ia.widgets import KeyedRows
from tests.fake_widgets import Parent, Tree, Widget
from tests.test_ia_pretraining_rows import fake_line, source


def row(i, status="Good", b="answer"):
    return {"id": i, "a": f"prompt {i}", "b": b, "status": status}


class TableItemsTest(unittest.TestCase):
    def test_items_fit_each_cell_and_tag_the_status(self):
        cells = []
        items = yd.table_items([row("c1")], False, False, lambda text, column: cells.append(column) or text[:3])
        self.assertEqual(items, [("c1", {"values": ("pro", "ans", "Good"), "tags": ("Good",)})])
        self.assertEqual(cells, [0, 1])

    def test_problems_are_mono_and_the_duplicate_filter_applies(self):
        rows = [row("p1", "Ready"), row("p2", "Near-duplicate")]
        items = yd.table_items(rows, True, True, lambda text, column: text)
        self.assertEqual([(k, o["tags"]) for k, o in items], [("p2", ("Near-duplicate", "mono"))])


class Table(Tree):
    def column(self, _name, _option):
        return 300

    def exists(self, iid):
        return iid in self.children

    def selection_set(self, iid):
        self.selected = iid


def table_page(rows):
    font = SimpleNamespace(measure=len)
    return SimpleNamespace(kind="conversations", rows=rows, only_dups=False, cell_fonts=(font, font), table=Table(), selected=None,
                           fitted=None, summaries={}, applied="", cursor=None, shown=Widget(), more_button=Widget(Parent()),
                           _detail=lambda: None)


class PaintRowsTest(unittest.TestCase):
    def test_a_reload_updates_the_rows_that_stay_in_place(self):
        page = table_page([row("c1"), row("c2"), row("c3")])
        yd.YourDataPage._paint_rows(page)
        page.table.calls.clear()
        page.rows = [row("c1"), row("c3", "Review"), row("c4")]  # c2 discarded, c3 changed, c4 new
        yd.YourDataPage._paint_rows(page)
        self.assertEqual(page.table.get_children(), ("c1", "c3", "c4"))
        self.assertEqual(page.table.items["c3"]["values"][2], "Review")
        self.assertEqual(page.table.calls, [("delete", "c2"), ("item", "c1"), ("item", "c3"), ("insert", "c4")])

    def test_load_more_only_appends(self):
        page = table_page([row("c1")])
        yd.YourDataPage._paint_rows(page)
        page.table.calls.clear()
        page.rows += [row("c2")]
        yd.YourDataPage._paint_rows(page, 1)
        self.assertEqual(page.table.calls, [("insert", "c2")])

    def test_the_selection_survives_a_reload(self):
        page = table_page([row("c1"), row("c2")])
        page.selected = "c2"
        yd.YourDataPage._paint_rows(page)
        page.rows = [row("c2", "Review")]
        yd.YourDataPage._paint_rows(page)
        self.assertEqual((page.selected, page.table.selected), ("c2", "c2"))


class CollectionCardTest(unittest.TestCase):
    def page(self, rows, notes=None):
        card_parent = Parent()
        buttons = Widget(card_parent)
        buttons.pack()
        card = Parent()
        return SimpleNamespace(kind="text", summaries={}, collections={"text": rows}, notes=notes or {}, buttons=buttons,
                               collection_card=Widget(card_parent), download_all=Widget(card),
                               collection_list=KeyedRows(Parent(), fake_line, lambda line, data: fill_collection_row(line, *data)))

    def test_a_refresh_keeps_the_lines_and_the_download_all_button_follows_the_rows(self):
        page = self.page([source("not downloaded")])
        yd.YourDataPage.paint_collection(page)
        line = page.collection_list["tinystories"]
        self.assertTrue(page.download_all.packed)
        self.assertEqual(page.download_all.cget("text"), T["download_add_all"])
        page.collections["text"] = [source("downloading", completed=5)]
        page.notes["tinystories"] = "Downloading · 5 MB"
        yd.YourDataPage.paint_collection(page)
        self.assertIs(page.collection_list["tinystories"], line)
        self.assertEqual(line.status.cget("text"), "Downloading · 5 MB")
        self.assertFalse(page.download_all.packed)
        self.assertEqual(page.download_all.cget("text"), "")

    def test_switching_kind_replaces_only_the_lines(self):
        page = self.page([source("not downloaded")])
        yd.YourDataPage.paint_collection(page)
        old = page.collection_list["tinystories"]
        page.kind, page.collections["problems"] = "problems", [{**source("not downloaded"), "id": "gsm8k_check"}]
        yd.YourDataPage.paint_collection(page)
        self.assertTrue(old.destroyed)
        self.assertEqual(len(page.collection_list), 1)


if __name__ == "__main__":
    unittest.main()
