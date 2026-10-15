"""Pretraining's set tiles and recommended collection update in place, and the collection is read off the Tk thread (no window)."""
import unittest
from types import SimpleNamespace

import studio_catalog as cat
from studio_ia.copy import PRETRAINING as T
from studio_ia.pages.pretraining import PretrainingPage, collection_items, fill_collection_row, tile_items
from studio_ia.theme import C
from studio_ia.widgets import KeyedRows
from tests.fake_widgets import Parent, Widget


def fake_line(parent, _data=None):
    """The parts collection_row builds, as stand-ins."""
    line = Widget(parent)
    line.source, line.name_label, line.size_label, line.status = None, Widget(), Widget(), Widget()
    line.action = Widget(Parent())
    return line


def source(state="downloading", error="", completed=0):
    return {"id": "tinystories", "name": "TinyStories", "size": "64 MB", "records": "77,941 stories", "state": state,
            "completed": completed, "total": 100, "error": error}


class CollectionRowTest(unittest.TestCase):
    def test_items_pair_each_row_with_its_progress_line(self):
        row = source()
        self.assertEqual(collection_items([row], {"tinystories": "Downloading · 1 MB"}), [("tinystories", (row, "Downloading · 1 MB"))])
        self.assertEqual(collection_items([row], {}), [("tinystories", (row, ""))])

    def test_the_button_follows_the_next_step(self):
        line = fake_line(Parent())
        fill_collection_row(line, source("not downloaded"))
        self.assertTrue(line.action.packed)
        self.assertEqual(line.action.cget("text"), T["download"])
        fill_collection_row(line, source("downloaded"))
        self.assertEqual(line.action.cget("text"), T["import"])
        fill_collection_row(line, source("downloading"), "Downloading · 3 MB")
        self.assertFalse(line.action.packed)
        self.assertEqual(line.action.cget("text"), "")  # a hidden button says nothing either
        self.assertEqual((line.status.cget("text"), line.status.cget("fg")), ("Downloading · 3 MB", C["muted"]))

    def test_a_failure_shows_in_coral(self):
        line = fake_line(Parent())
        fill_collection_row(line, source("not downloaded", error="no network"))
        self.assertIn("no network", line.status.cget("text"))
        self.assertEqual(line.status.cget("fg"), C["coral"])

    def test_a_row_changed_in_place_by_the_services_still_repaints(self):
        rows, row = KeyedRows(Parent(), fake_line, lambda line, data: fill_collection_row(line, *data)), source("not downloaded")
        rows.set(collection_items([row], {}))
        row["error"] = "no network"
        rows.set(collection_items([row], {}))
        self.assertIn("no network", rows["tinystories"].status.cget("text"))

    def test_a_progress_line_drawn_directly_is_replaced_by_the_next_read(self):
        rows = KeyedRows(Parent(), fake_line, lambda line, data: fill_collection_row(line, *data))
        items = collection_items([source("imported")], {})
        rows.set(items)
        rows["tinystories"].status.configure(text="Downloading · 9 MB")  # a late throttled flush
        rows.touched("tinystories")
        rows.set(items)
        self.assertEqual(rows["tinystories"].status.cget("text"), T["state_imported"])

    def test_progress_and_state_changes_keep_the_same_line(self):
        rows = KeyedRows(Parent(), fake_line, lambda line, data: fill_collection_row(line, *data))
        rows.set(collection_items([source(completed=1)], {"tinystories": "1 MB"}))
        line = rows["tinystories"]
        rows.set(collection_items([source(completed=50)], {"tinystories": "50 MB"}))
        rows.set(collection_items([source("imported")], {}))
        self.assertIs(rows["tinystories"], line)
        self.assertFalse(line.destroyed)
        self.assertEqual(line.status.cget("text"), T["state_imported"])


class TilesTest(unittest.TestCase):
    def test_tile_items_carry_the_choice_and_held_out_sets(self):
        split = next(e["id"] for e in cat.CATALOG if e.get("split") in ("validation", "test"))
        sets = [{"id": "d1", "name": "Mine", "count": 1200}, {"id": f"catalog_{split}", "name": "Check", "count": 5}]
        items = tile_items(sets, {"d1": True})
        self.assertEqual([k for k, _ in items], ["d1", f"catalog_{split}"])
        self.assertEqual([(d["on"], d["held"]) for _, d in items], [(True, False), (False, True)])

    def test_a_toggle_repaints_the_same_tile(self):
        def make(parent, _data):
            tile = Widget(parent)
            tile.mark, tile.name_label, tile.note = Widget(), Widget(), Widget()
            return tile
        tiles = KeyedRows(Parent(), make, PretrainingPage._fill_tile)
        sets = [{"id": "d1", "name": "Mine", "count": 1200}]
        tiles.set(tile_items(sets, {"d1": True}))
        tile = tiles["d1"]
        self.assertEqual((tile.mark.cget("text"), tile.cget("highlightbackground")), ("✓", C["violet"]))
        self.assertEqual(tile.note.cget("text"), "1,200 examples")
        tiles.set(tile_items(sets, {"d1": False}))
        self.assertIs(tiles["d1"], tile)  # the tile keeps its focus: it was not rebuilt
        self.assertEqual((tile.mark.cget("text"), tile.cget("highlightbackground")), (" ", C["border"]))


class CollectionReadTest(unittest.TestCase):
    def test_the_collection_is_read_on_a_lane_not_the_tk_thread(self):
        read, lanes = [], []
        page = SimpleNamespace(services=SimpleNamespace(collection=lambda: read.append(1) or [source()]),
                               quietly=lambda work, then, lane="fast": lanes.append((work, then)), _collected=None)
        PretrainingPage.paint_collection(page)
        self.assertEqual(read, [])  # handed to the shell's lane, not read here
        work, _then = lanes[0]
        self.assertEqual(work(), [source()])

    def test_a_service_without_a_collection_shows_none(self):
        def missing():
            raise NotImplementedError
        got = []
        page = SimpleNamespace(services=SimpleNamespace(collection=missing), quietly=lambda work, then, lane="fast": got.append(work()),
                               _collected=None)
        PretrainingPage.paint_collection(page)
        self.assertEqual(got, [[]])


if __name__ == "__main__":
    unittest.main()
