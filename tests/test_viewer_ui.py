"""The 3D viewer's control decisions (studio_viewer_ui): colours, which controls show, labels, layout and label placement."""
import unittest

import studio_viewer_ui as ui
from studio_ia.theme import C
from studio_scene import HEAD_STAGES, LEVELS, MODES


class ViewerControls(unittest.TestCase):
    def test_colours_come_from_the_theme(self):
        self.assertEqual(ui.rgba("violet"), (0x9f / 255, 0x8c / 255, 1., 1.))
        self.assertEqual(ui.rgba("#000000", .5), (0., 0., 0., .5))
        self.assertEqual(ui.button_colors(True)[0], ui.rgba("violet"))
        self.assertEqual(ui.button_colors(True)[3], ui.rgba("panel2"))  # disabled primary looks like the app's
        self.assertEqual(ui.button_colors()[0], ui.rgba("panel2"))
        self.assertEqual(ui.text_colors(True)[0], ui.rgba("bg"))
        self.assertEqual(ui.text_colors()[3], ui.rgba("muted"))
        self.assertTrue(all(len(c) == 4 for c in ui.button_colors() + ui.text_colors()))
        self.assertIn("bg", C)

    def test_levels_are_ones_the_scene_draws(self):
        self.assertTrue(set(ui.LEVELS) <= set(LEVELS))
        self.assertNotIn("Run", ui.LEVELS)

    def test_controls_are_the_same_width(self):
        left, right, *_ = ui.menu_frame()
        self.assertAlmostEqual((right - left + ui.MARKER) * ui.TEXT, ui.CONTROL_W)  # the marker is added by the menu
        left, right, *_ = ui.button_frame()
        self.assertAlmostEqual((right - left) * ui.TEXT, ui.CONTROL_W)
        self.assertAlmostEqual(-left, right)  # centred on its position
        self.assertAlmostEqual(ui.CONTROL_W + 2 * ui.PAD, ui.PANEL_W)

    def test_rows_show_only_where_they_work(self):
        self.assertEqual(ui.view_rows("Model", "Train", []), ["level", "mode"])
        self.assertEqual(ui.view_rows("Model", "Build", []), ["level", "mode", "mixer"])
        self.assertEqual(ui.view_rows("Head", "Train", ["x"]), ["level", "mode", "stage"])
        self.assertEqual(ui.view_rows("Weights", "Train", []), ["level", "mode", "tile"])
        self.assertEqual(ui.view_rows("Weights", "Ride a token", ["x"]), ["level", "mode", "tile", "matrix"])
        for level in ui.LEVELS:
            for mode in MODES:
                rows = ui.view_rows(level, mode, ["x"])
                self.assertTrue(set(rows) <= set(ui.captions(0)))

    def test_captions_name_the_block(self):
        text = ui.captions(2)
        self.assertEqual(text["mixer"], "BLOCK 3 TOKEN MIXER")
        self.assertEqual(text["matrix"], "MATRIX (BLOCK 3)")

    def test_view_layout_fits_its_panel(self):
        for count in range(1, 6):
            rows, bottom = ui.view_layout(count)
            self.assertEqual(len(rows), count)
            for i, (caption, control) in enumerate(rows):
                self.assertLess(caption, -ui.PAD + 1e-9 - ui.SMALL * .5)  # the caption's top is inside the panel
                self.assertLess(control + 1.1 * ui.TEXT, caption - ui.SMALL * .2)  # the control sits below its caption
                if i:
                    previous_control = rows[i - 1][1]
                    self.assertLess(caption + ui.SMALL, previous_control - .5 * ui.TEXT)  # no overlap with the row above
            self.assertAlmostEqual(rows[-1][1] - .5 * ui.TEXT - ui.PAD, bottom)  # padding below the last control

    def test_stack_layout_fits_its_panel(self):
        self.assertEqual(ui.stack_layout(0), ([], -2 * ui.PAD))
        rows, bottom = ui.stack_layout(3)
        self.assertAlmostEqual(rows[0] + 1.1 * ui.TEXT, -ui.PAD)
        for above, below in zip(rows, rows[1:]):
            self.assertAlmostEqual(above - below, ui.CONTROL_H + ui.GAP)
        self.assertAlmostEqual(rows[-1] - .5 * ui.TEXT - ui.PAD, bottom)

    def test_matrix_menu_offers_the_selected_block(self):
        names = ["block0.f1.w", "block0.qkv.w", "block1.f1.w", "block1.qkv.w", "block10.f1.w", "position", "token"]
        self.assertEqual(ui.block_matrices(names, 1), ["block1.f1.w", "block1.qkv.w", "position", "token"])
        self.assertEqual(ui.block_matrices(names, 10), ["block10.f1.w", "position", "token"])
        self.assertEqual(ui.pick_matrix(names, 1, "block1.qkv.w"), "block1.qkv.w")
        self.assertEqual(ui.pick_matrix(names, 1, "token"), "token")
        self.assertEqual(ui.pick_matrix(names, 1, "block0.qkv.w"), "block1.f1.w")  # the block changed: its first matrix
        self.assertEqual(ui.pick_matrix(names, 1, None), "block1.f1.w")
        self.assertEqual(ui.pick_matrix(["token"], 3, None), "token")
        self.assertIsNone(ui.pick_matrix([], 0, None))

    def test_playback(self):
        empty = ui.playback(0, -1, False)
        self.assertEqual((empty["show"], empty["position"]), (False, "No snapshots loaded."))
        first = ui.playback(3, 0, False)
        self.assertEqual((first["play"], first["position"], first["previous"], first["next"]), ("Play", "Snapshot 1 of 3", False, True))
        last = ui.playback(3, 2, True)
        self.assertEqual((last["play"], last["previous"], last["next"]), ("Pause", True, False))

    def test_snapshot_buttons(self):
        demo = ui.snapshot_buttons(False, False)
        self.assertFalse(demo["show"])
        self.assertIn("Open the viewer from the studio", demo["note"])
        ready = ui.snapshot_buttons(True, False)
        self.assertEqual((ready["show"], ready["probe"], ready["probe_enabled"], ready["note"]), (True, "Run one real step", True, ""))
        busy = ui.snapshot_buttons(True, True)
        self.assertEqual((busy["probe"], busy["probe_enabled"]), ("Running one step...", False))

    def test_toggles_show_where_they_change_something(self):
        self.assertEqual(ui.toggles("Model", [], False), [])
        self.assertEqual(ui.toggles("Model", ["a"], False), ["motion"])
        self.assertEqual(ui.toggles("Model", [], True), ["motion"])
        self.assertEqual(ui.toggles("Weights", [], False), ["numbers"])
        self.assertEqual(ui.toggle_text("Show numbers", True), "Show numbers: on")
        self.assertEqual(ui.toggle_text("Reduce motion", False), "Reduce motion: off")

    def test_block_labels_stay_on_their_block_and_never_overlap(self):
        height, gap = .05, .01
        # a column of blocks seen end on: every label lands on the same spot, so only the first one shows
        items = [(0., .1, .3)] * 6 + [None, (.5, .1, .2), (.05, .3, .3)]
        shown = ui.place_labels(items, height, gap)
        self.assertEqual(shown, [True] + [False] * 5 + [False, True, True])
        boxes = [(x - w / 2, x + w / 2, y, y + height) for item, on in zip(items, shown) if on for x, y, w in [item]]
        for i, a in enumerate(boxes):
            for b in boxes[i + 1:]:
                self.assertTrue(a[1] + gap <= b[0] + 1e-9 or b[1] + gap <= a[0] + 1e-9 or a[3] + gap <= b[2] + 1e-9 or b[3] + gap <= a[2] + 1e-9, (a, b))
        self.assertEqual(ui.place_labels([], height), [])

    def test_scene_colours_tell_marks_apart(self):
        colours = [ui.SCENE[name][:3] for name in ('activation', 'mixer', 'channel', 'token', 'lane')]
        self.assertEqual(len(set(colours)), len(colours))

    def test_play_shows_each_snapshot_for_a_full_pass(self):
        self.assertEqual(ui.pass_seconds(12), 12)
        self.assertEqual(ui.pass_seconds(0), 1)

    def test_matrix_legend_explains_the_colours(self):
        self.assertIn('amber = positive', ui.matrix_legend('Weights'))
        self.assertIn('rows = queries', ui.matrix_legend('Head'))
        self.assertNotEqual(ui.SCENE['positive'][:3], ui.SCENE['negative'][:3])

    def test_empty_matrix_note_says_why(self):
        self.assertIn('Load snapshots', ui.empty_matrix_note('Head', False, 0, 'gru', False))
        self.assertIn('no attention heads', ui.empty_matrix_note('Head', True, 0, 'gru', False))
        self.assertIn('Block 1', ui.empty_matrix_note('Head', True, 0, 'gru', False))
        self.assertIn('no weights values', ui.empty_matrix_note('Weights', True, 2, 'attention', True))

    def test_camera_frames_each_level(self):
        target, distance = ui.camera_view("Model", 6)
        self.assertEqual(target, (0., -7.5, 1.))  # the middle of the column (gates every 3 units)
        self.assertGreaterEqual(distance, 22.)
        self.assertGreater(ui.camera_view("Window", 24)[1], ui.camera_view("Window", 6)[1])
        target, distance = ui.camera_view("Head", 6, rows=16)
        self.assertEqual(target, (0., -3., 0.))  # the middle of 16 rows .4 apart
        self.assertEqual(ui.camera_view("Weights", 6, rows=0)[0], (0., 0., 0.))
        self.assertGreater(distance, 0)

    def test_head_stages_fit_the_menu(self):
        self.assertIn("pattern", HEAD_STAGES)


if __name__ == "__main__":
    unittest.main()
