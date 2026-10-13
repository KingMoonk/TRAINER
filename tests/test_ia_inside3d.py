"""Inside the network (3D): the decision logic and the Services viewer calls, no Tk, no process spawned."""
import unittest

from studio_ia import copy
from studio_ia.pages import inside3d as p
from studio_ia.services import FakeServices
from studio_ia.shell import ROUTES


class Inside3DLogic(unittest.TestCase):
    def test_page_matches_route(self):
        self.assertEqual(p.Inside3DPage.title, ROUTES["inside3d"].title)
        self.assertEqual(p.Inside3DPage.lead, ROUTES["inside3d"].lead)
        self.assertEqual(p.Inside3DPage.crumb, ROUTES["inside3d"].crumb)
        self.assertEqual(p.Inside3DPage.title, copy.INSIDE3D["h1"])

    def test_modes_map_to_viewer_names(self):
        from studio_scene import LEVELS, MODES
        for mode_id, label, viewer in p.MODES:
            self.assertIn(viewer, MODES)
            self.assertEqual(p.SceneState(mode=mode_id).message()["mode"], viewer)
        for level_id, label, viewer, _ in p.LEVELS:
            self.assertIn(viewer, LEVELS)
            self.assertEqual(p.SceneState(level=level_id).message()["level"], viewer)
        self.assertEqual(p.label_to_id(p.MODES, "Ride a position"), "ride")

    def test_text_follows_mode_and_level(self):
        s = p.SceneState(mode="ride", block=3)
        self.assertEqual(s.hud()[1], 'Riding the stream at "is" through block 3')
        self.assertEqual(s.tip()[3], "trace")
        self.assertEqual(len(s.legend()), 5)
        s.level = "weights"  # zoom text replaces the mode text
        self.assertTrue(s.hud()[0].startswith("ZOOM 5"))
        self.assertEqual((s.tip()[3], len(s.legend())), ("value_map", 2))
        s.level = "head"
        self.assertEqual(s.tip()[3], "head_zoom")
        for mode in ("build", "train", "ride"):
            self.assertTrue(all(p.SceneState(mode=mode).hud()))

    def test_every_template_formats(self):
        for level in ("window", "head", "weights"):
            for mode in ("build", "train", "ride"):
                p.SceneState(level=level, mode=mode).hud()

    def test_viewer_view_states(self):
        gone = p.viewer_view({"available": False}, True)
        self.assertEqual((gone["enabled"], gone["scene_enabled"]), (False, False))
        self.assertIn("not installed", gone["note"])
        off = p.viewer_view({"available": True, "running": False}, False)
        self.assertEqual((off["button"], off["enabled"], off["scene_enabled"]), (copy.INSIDE3D["open_viewer"], True, False))
        self.assertIn(copy.INSIDE3D["viewer_demo"], off["notes"])
        on = p.viewer_view({"available": True, "running": True}, True)
        self.assertEqual((on["button"], on["scene_enabled"]), (copy.INSIDE3D["close_viewer"], True))
        self.assertNotIn(copy.INSIDE3D["viewer_demo"], on["notes"])
        self.assertEqual(p.viewer_view(None, False)["enabled"], False)

    def test_fake_services_viewer_spawns_nothing(self):
        svc = FakeServices()
        svc.viewer_send({"type": "scene"})  # not running: dropped
        self.assertEqual(svc.viewer_log, [])
        scene = p.SceneState(mode="train").message()
        svc.viewer_start(scene)
        self.assertTrue(svc.viewer_status()["running"])
        svc.viewer_send({"type": "scene", **scene})
        svc.viewer_stop()
        self.assertEqual([a for a, _ in svc.viewer_log], ["start", "send", "stop"])
        self.assertEqual(svc.viewer_log[0][1], {"level": "Window", "mode": "Train"})
        self.assertFalse(svc.viewer_status()["running"])

    def _version(self, svc):
        from studio_spec import default_spec, validate_spec
        spec = default_spec()
        spec["dims"].update(d_model=16, n_layer=3, n_ctx=64, dropout=0.)
        return svc.save_version(validate_spec(spec), "first")

    def test_module_swap_saves_a_new_version(self):
        svc = FakeServices()
        text, spec = p.save_swap(svc, {"type": "module_swap", "block_id": "blk_000001", "slot": "mixer", "entry": {"id": "gru"}})
        self.assertIn("version", text.lower())
        self.assertEqual(spec, None)  # nothing to swap in: no version yet
        first = self._version(svc)
        text, spec = p.save_swap(svc, {"type": "module_swap", "block_id": "blk_000001", "slot": "mixer", "entry": {"id": "gru"}})
        self.assertEqual(len(svc.versions()), 2)
        saved = svc.current_version()
        self.assertNotEqual(saved.spec, first.spec)
        self.assertEqual(saved.spec["blocks"][0]["mixer"]["id"], "gru")
        self.assertEqual(saved.spec["blocks"][0]["index"], 1)
        self.assertEqual(spec, saved.spec)
        self.assertIn("block 2", saved.change)
        self.assertIn("Saved as version 2", text)

    def test_bad_swap_saves_nothing(self):
        svc = FakeServices()
        self._version(svc)
        for message in ({"block_id": "blk_000063", "slot": "mixer", "entry": {"id": "gru"}},
                        {"block_id": "blk_000000", "slot": "mixer", "entry": {"id": "nope"}},
                        {"block_id": "oops", "slot": "mixer", "entry": {"id": "gru"}}, {}):
            text, spec = p.save_swap(svc, message)
            self.assertIsNone(spec)
            self.assertIn("not saved", text)
        self.assertEqual(len(svc.versions()), 1)

    def test_train_events_become_compact_stats(self):
        self.assertEqual(p.stats_message({"job": "j", "step": 5, "total": 9, "loss": 1.5, "lr": 1e-3, "tokens_per_second": 900, "extra": 1}),
                         {"type": "stats", "step": 5, "total": 9, "loss": 1.5, "lr": 1e-3, "tokens_per_second": 900})
        self.assertIsNone(p.stats_message({"type": "heat"}))
        self.assertEqual(p.stats_message({"job": "j", "type": "viz_snapshot", "step": 50, "path": "s.npz", "markers": []}),
                         {"type": "viz_snapshot", "path": "s.npz"})
        self.assertIsNone(p.stats_message(None))


if __name__ == "__main__":
    unittest.main()
