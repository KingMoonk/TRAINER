"""Keyed rows (studio_ia.widgets): a refresh updates rows in place and destroys only the rows whose keys are gone (no window)."""
import unittest

from studio_ia.widgets import KeyedRows, keyed_plan, sync_tree
from tests.fake_widgets import Parent, Tree, Widget


class KeyedPlanTest(unittest.TestCase):
    def test_same_keys_change_nothing_but_the_contents(self):
        self.assertEqual(keyed_plan(["a", "b"], ["a", "b"]), ([], [], False))

    def test_gone_keys_are_dropped_and_the_rest_stay_in_order(self):
        self.assertEqual(keyed_plan(["a", "b", "c"], ["a", "c"]), (["b"], [], False))

    def test_new_keys_are_added_and_the_rest_need_not_move(self):
        self.assertEqual(keyed_plan(["a", "c"], ["a", "b", "c"]), ([], ["b"], False))
        self.assertEqual(keyed_plan([], ["a"]), ([], ["a"], False))

    def test_a_move_is_seen_without_dropping(self):
        self.assertEqual(keyed_plan(["a", "b"], ["b", "a"]), ([], [], True))
        self.assertEqual(keyed_plan(["a", "b", "c"], ["c", "x", "a"]), (["b"], ["x"], True))

    def test_everything_gone(self):
        self.assertEqual(keyed_plan(["a", "b"], []), (["a", "b"], [], False))


class KeyedRowsTest(unittest.TestCase):
    def setUp(self):
        self.parent, self.made, self.filled = Parent(), [], []
        self.rows = KeyedRows(self.parent, self.make, self.fill, fill="x")

    def make(self, parent, data):
        widget = Widget(parent)
        self.made.append(data["id"])
        return widget

    def fill(self, widget, data):
        widget.configure(text=data["text"])
        self.filled.append(data["id"])

    @staticmethod
    def items(*pairs):
        return [(k, {"id": k, "text": t}) for k, t in pairs]

    def shown(self):
        return [w.cget("text") for w in self.parent.slaves]

    def test_first_set_makes_and_packs_every_row_in_order(self):
        self.rows.set(self.items(("a", "A"), ("b", "B")))
        self.assertEqual(self.made, ["a", "b"])
        self.assertEqual(self.shown(), ["A", "B"])
        self.assertEqual(len(self.rows), 2)

    def test_a_changed_row_is_updated_in_place_and_nothing_is_rebuilt(self):
        self.rows.set(self.items(("a", "A"), ("b", "B")))
        a, b = self.rows["a"], self.rows["b"]
        self.made.clear(), self.filled.clear()
        self.rows.set(self.items(("a", "A"), ("b", "B 50%")))
        self.assertEqual(self.made, [])
        self.assertEqual(self.filled, ["b"])  # an unchanged row is not touched at all
        self.assertIs(self.rows["a"], a)
        self.assertIs(self.rows["b"], b)
        self.assertEqual(self.shown(), ["A", "B 50%"])

    def test_only_gone_rows_are_destroyed(self):
        self.rows.set(self.items(("a", "A"), ("b", "B"), ("c", "C")))
        a, b, c = self.rows["a"], self.rows["b"], self.rows["c"]
        self.rows.set(self.items(("a", "A"), ("c", "C")))
        self.assertTrue(b.destroyed)
        self.assertFalse(a.destroyed or c.destroyed)
        self.assertIsNone(self.rows.get("b"))
        self.assertEqual(self.shown(), ["A", "C"])

    def test_new_rows_go_in_their_place(self):
        self.rows.set(self.items(("b", "B"), ("d", "D")))
        self.rows.set(self.items(("a", "A"), ("b", "B"), ("c", "C"), ("d", "D"), ("e", "E")))
        self.assertEqual(self.made, ["b", "d", "a", "c", "e"])
        self.assertEqual(self.shown(), ["A", "B", "C", "D", "E"])

    def test_moved_rows_are_repacked_not_rebuilt(self):
        self.rows.set(self.items(("a", "A"), ("b", "B"), ("c", "C")))
        widgets = dict(self.rows.rows)
        self.rows.set(self.items(("c", "C"), ("a", "A"), ("b", "B")))
        self.assertEqual(self.shown(), ["C", "A", "B"])
        self.assertEqual(self.rows.rows, widgets)
        self.assertEqual(self.made, ["a", "b", "c"])

    def test_a_key_that_comes_back_gets_a_fresh_row(self):
        self.rows.set(self.items(("a", "A")))
        old = self.rows["a"]
        self.rows.set([])
        self.rows.set(self.items(("a", "A")))
        self.assertIsNot(self.rows["a"], old)
        self.assertEqual(self.filled, ["a", "a"])  # the new row is filled even though its data matches the old one's


class SyncTreeTest(unittest.TestCase):
    @staticmethod
    def rows(*keys):
        return [(k, {"values": (k.upper(),)}) for k in keys]

    def test_rows_that_stay_are_configured_in_place(self):
        tree = Tree(self.rows("a", "b", "c"))
        sync_tree(tree, [("a", {"values": ("A2",)}), ("c", {"values": ("C",)})])
        self.assertEqual(tree.get_children(), ("a", "c"))
        self.assertEqual(tree.items["a"]["values"], ("A2",))
        self.assertEqual([c[0] for c in tree.calls], ["delete", "item", "item"])  # no row is deleted and inserted again

    def test_new_rows_are_inserted_where_they_belong(self):
        tree = Tree(self.rows("b", "d"))
        sync_tree(tree, self.rows("a", "b", "c", "d", "e"))
        self.assertEqual(tree.get_children(), ("a", "b", "c", "d", "e"))
        self.assertEqual([c for c in tree.calls if c[0] != "item"], [("insert", "a"), ("insert", "c"), ("insert", "e")])

    def test_a_new_order_is_kept_by_moving_rows(self):
        tree = Tree(self.rows("a", "b", "c"))
        sync_tree(tree, self.rows("c", "a", "b"))
        self.assertEqual(tree.get_children(), ("c", "a", "b"))
        self.assertNotIn("insert", [c[0] for c in tree.calls])

    def test_empty(self):
        tree = Tree(self.rows("a"))
        sync_tree(tree, [])
        self.assertEqual(tree.get_children(), ())


if __name__ == "__main__":
    unittest.main()
