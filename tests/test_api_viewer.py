"""studio_api.routers.viewer over FakeServices and real snapshot files: the scene in its three encodings, the controls a browser
keeps, the snapshots a client may name, the probe job and the Panda3D process controls."""
import copy
import math
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from fastapi.testclient import TestClient

from studio_api.app import create_app
from studio_api.hosts import ANY
from studio_compile import CompiledModel
from studio_ia.copy import INSIDE3D as T
from studio_ia.services import FakeServices
from studio_snapshot import one_step, save
from tests.api_tap import EventTap
from tests.api_work import finished, trained

ENVELOPE = {"schema", "full", "groups", "cursor", "removed_groups", "up", "palette"}


class ViewerApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = trained(FakeServices()).spec
        cls.snapshot = one_step(CompiledModel(spec, 9, device="cpu"), [[1, 2, 3, 4, 5, 6]], [[2, 3, 4, 5, 6, 7]])

    def setUp(self):
        self.services = FakeServices()
        self.version = trained(self.services)
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.paths = [self.save(1), self.save(2)]
        self.services.snapshot_paths = [str(p) for p in self.paths]
        self.app = create_app(self.services, open=True, hosts=ANY)
        self.client = self.enterContext(TestClient(self.app))

    def save(self, step):
        snapshot = copy.deepcopy(self.snapshot)
        snapshot["step"] = step
        return save(self.folder, snapshot)

    def sections(self, scene):
        return set(scene) - ENVELOPE

    def test_scene_is_the_whole_payload_for_the_viewers_controls(self):
        response = self.client.get("/viewer/scene", params={"level": "Head", "headStage": "scores", "snapshot": 1})
        self.assertEqual(response.status_code, 200, response.text)
        scene = response.json()
        self.assertTrue(scene["full"])
        self.assertEqual(scene["stats"]["step"], 2)  # snapshot 1 is the second file
        self.assertIn("Head", scene["stats"]["info"])
        self.assertEqual(self.client.get("/viewer/scene", params={"snapshot": 0}).json()["stats"]["step"], 1)
        self.assertEqual(self.client.get("/viewer/scene", params={"snapshot": 9}).json()["stats"]["step"], 2)  # past the end is the last
        self.assertEqual(self.client.get("/viewer/scene", params={"level": "Roof"}).status_code, 422)

    def test_the_thermal_look_is_a_control_the_server_keeps_and_gives_the_tiles_token_text(self):
        plain = self.client.get("/viewer/scene", params={"level": "Head"}).json()
        self.assertFalse(plain["view"]["thermal"])
        self.assertNotIn("tiles", plain["groups"]["matrix"])
        words = ["a", " b", "c", "d", "e", "f"]
        with patch.object(self.services, "viewer_token_text", create=True, side_effect=lambda number, ids: words[: len(ids)]) as text:
            scene = self.client.get("/viewer/scene", params={"level": "Head", "thermal": "true"}).json()
        text.assert_called_once()
        self.assertEqual(text.call_args.args[1], [1, 2, 3, 4, 5, 6])
        self.assertTrue(scene["view"]["thermal"])
        self.assertIn("pattern", {tile["id"] for tile in scene["groups"]["matrix"]["tiles"]})
        axis = [label["text"] for label in scene["groups"]["matrix"]["labels"] if label["kind"] == "tile_axis"]
        self.assertIn("\u00b7b", axis)
        out = self.client.post("/viewer/action", json={"thermal": True, "action": {"type": "level", "value": "Head"}}).json()
        self.assertTrue(out["state"]["thermal"])  # the controls come back with it, so the next call keeps the look

    def test_scene_without_a_snapshot_still_draws_the_design(self):
        self.services.snapshot_paths = []
        scene = self.client.get("/viewer/scene").json()
        self.assertTrue(scene["groups"])
        self.assertIsNone(scene["stats"]["step"])

    def test_scene_of_a_version_that_does_not_exist_is_a_bad_request(self):
        self.assertEqual(self.client.get("/viewer/scene", params={"version": 9}).json()["detail"], "There is no version 9.")
        self.assertEqual(self.client.get("/viewer/scene", params={"version": self.version.number}).status_code, 200)
        self.services._versions.clear()
        self.assertEqual(self.client.get("/viewer/scene").status_code, 400)

    def test_a_frame_carries_only_what_changed_since_the_cursor(self):
        scene = self.client.get("/viewer/scene").json()
        same = self.client.post("/viewer/frame", json={"since": scene["cursor"]}).json()
        self.assertFalse(same["full"])
        self.assertEqual(self.sections(same), set())
        changed = self.client.post("/viewer/frame", json={"since": scene["cursor"], "level": "Window"}).json()
        self.assertFalse(changed["full"])
        self.assertIn("view", self.sections(changed))
        self.assertTrue(self.client.post("/viewer/frame", json={}).json()["full"])
        self.assertEqual(self.client.post("/viewer/frame", json={"since": "yesterday"}).status_code, 422)

    def test_binary_and_manifest_describe_the_same_bytes_as_the_payload(self):
        body = {"arrays": "binary", "level": "Weights", "tileKind": "weight"}
        scene = self.client.get("/viewer/scene", params={"arrays": "binary", "level": "Weights"}).json()
        data = self.client.post("/viewer/binary", json=body)
        self.assertEqual(data.headers["content-type"], "application/octet-stream")
        manifest = self.client.post("/viewer/manifest", json=body).json()
        self.assertTrue(manifest)
        self.assertEqual(len(data.content), sum(4 * math.prod(entry["shape"]) for entry in manifest.values()))
        self.assertEqual({entry["dtype"] for entry in manifest.values()}, {"float32"})
        named = []

        def walk(node):
            if isinstance(node, dict):
                if "name" in node and "shape" in node:
                    named.append(node["name"])
                for value in node.values():
                    walk(value)
            elif isinstance(node, list):
                for value in node:
                    walk(value)
        walk(scene)
        self.assertTrue(named)
        self.assertTrue(set(named) <= set(manifest), set(named) - set(manifest))
        nothing_new = self.client.post("/viewer/binary", json={**body, "since": scene["cursor"]}).content
        self.assertEqual(nothing_new, b"")  # the client holds every array already

    def test_an_action_answers_with_the_new_controls_and_the_frame(self):
        scene = self.client.get("/viewer/scene").json()
        response = self.client.post("/viewer/action", json={"action": {"type": "level", "value": "Window"}, "since": scene["cursor"]})
        self.assertEqual(response.status_code, 200, response.text)
        out = response.json()
        self.assertEqual((out["state"]["level"], out["state"]["mode"]), ("Window", "Build"))
        self.assertFalse(out["scene"]["full"])
        self.assertIn("view", self.sections(out["scene"]))
        again = self.client.post("/viewer/action", json={**out["state"], "action": {"type": "mode", "value": "Train"}})
        self.assertEqual((again.json()["state"]["level"], again.json()["state"]["mode"]), ("Window", "Train"))
        self.assertTrue(again.json()["scene"]["full"])  # no cursor, so the whole scene

    def test_actions_walk_through_the_snapshots_and_keep_the_state_in_the_client(self):
        out = self.client.post("/viewer/action", json={"action": {"type": "scrub", "index": 1}}).json()
        self.assertEqual((out["state"]["snapshot"], out["scene"]["stats"]["step"]), (1, 2))
        out = self.client.post("/viewer/action", json={**out["state"], "action": {"type": "scrub", "delta": -1}}).json()
        self.assertEqual((out["state"]["snapshot"], out["scene"]["stats"]["step"]), (0, 1))
        out = self.client.post("/viewer/action", json={**out["state"], "action": {"type": "play"}}).json()
        self.assertTrue(out["state"]["playing"])

    def test_the_probe_action_asks_the_client_to_run_the_probe_job(self):
        out = self.client.post("/viewer/action", json={"action": {"type": "probe_step"}}).json()
        self.assertEqual((out["answer"], out["state"]["probing"]), (True, True))
        busy = self.client.post("/viewer/action", json={**out["state"], "action": {"type": "probe_step"}}).json()
        self.assertFalse(busy["answer"])  # one is already running

    def test_a_browser_may_send_only_its_own_controls(self):
        for kind in ("viz_snapshot", "snapshot", "replay", "spec", "summary", "stats", "notice", "error", "nonsense"):
            response = self.client.post("/viewer/action", json={"action": {"type": kind, "path": "/etc/passwd", "paths": ["/etc/passwd"]}})
            self.assertEqual(response.status_code, 400, kind)
            self.assertIn("scene, level, mode", response.json()["detail"])
        self.assertEqual(self.client.post("/viewer/action", json={"action": {"type": "level"}}).status_code, 400)  # a missing value
        self.assertEqual(self.client.post("/viewer/action", json={"action": {}}).status_code, 400)

    def test_a_scene_message_cannot_name_a_snapshot_file(self):
        outside = self.save(3)
        self.services.snapshot_paths = [str(self.paths[0])]
        self.client.post("/viewer/action", json={"action": {"type": "scene", "paths": [str(outside)]}})
        stats = self.client.get("/viewer/scene", params={"snapshot": 5}).json()["stats"]
        self.assertEqual(stats["step"], 1)  # the file the server did not list was never opened

    def test_snapshots_are_named_by_index_without_a_path(self):
        listed = self.client.get("/viewer/snapshots").json()
        self.assertEqual(listed, [{"index": 0, "name": "step_00000001.npz"}, {"index": 1, "name": "step_00000002.npz"}])
        self.services.snapshot_paths = []
        self.assertEqual(self.client.get("/viewer/snapshots").json(), [])

    def test_probe_is_a_job_that_adds_a_snapshot(self):
        tap = self.enterContext(EventTap(self.app))
        self.services.probe_path = str(self.save(3))
        job, steps = finished(tap, self.client.post("/viewer/probe", json={"text": "hello there"}))
        self.assertEqual(job["status"], "done", job)
        self.assertEqual(job["result"], {"snapshot": 2, "note": T["probe_done"].format(n=self.version.number)})
        self.assertEqual(steps[0]["status"], "queued")  # every state change reached /events
        self.assertEqual(self.services.viewer_log, [("probe", "hello there")])
        self.assertEqual([item["name"] for item in self.client.get("/viewer/snapshots").json()][-1], "step_00000003.npz")
        self.assertEqual(self.client.get("/viewer/scene", params={"snapshot": 2}).json()["stats"]["step"], 3)
        again, _ = finished(tap, self.client.post("/viewer/probe"))  # the same file is not listed twice
        self.assertEqual(again["result"]["snapshot"], 2)
        self.assertEqual(len(self.client.get("/viewer/snapshots").json()), 3)
        self.assertEqual(self.client.post("/viewer/probe", json={"text": "x"}).status_code, 422)

    def test_selected_version_uses_its_own_snapshots_and_probe(self):
        older = self.version.number
        newer = trained(self.services).number
        paths = {older: [str(self.paths[0])], newer: [str(self.paths[1])]}
        probe_path = str(self.save(3))
        tap = self.enterContext(EventTap(self.app))
        with patch.object(self.services, "viewer_snapshots", side_effect=lambda number: paths[number]), \
                patch.object(self.services, "viewer_probe_step", return_value=probe_path) as probe:
            before = self.client.get("/viewer/scene", params={"version": older}).json()
            self.assertEqual(before["stats"]["step"], 1)
            job, _ = finished(tap, self.client.post("/viewer/probe", json={"version": older, "text": "hello there"}))
            self.assertEqual(job["status"], "done", job)
            probe.assert_called_once_with("hello there", older)
            self.assertEqual(job["result"]["snapshot"], 1)
            own = self.client.get("/viewer/snapshots", params={"version": older}).json()
            other = self.client.get("/viewer/snapshots", params={"version": newer}).json()
            self.assertEqual([row["name"] for row in own], ["step_00000001.npz", "step_00000003.npz"])
            self.assertEqual([row["name"] for row in other], ["step_00000002.npz"])
            after = self.client.get("/viewer/scene", params={"version": older, "snapshot": 1}).json()
            self.assertEqual(after["stats"]["step"], 3)

    def test_unknown_version_never_probes_the_current_version(self):
        with patch.object(self.services, "viewer_probe_step") as probe:
            self.assertEqual(self.client.post("/viewer/probe", json={"version": 999}).status_code, 400)
            self.assertEqual(self.client.get("/viewer/snapshots", params={"version": 999}).status_code, 400)
            probe.assert_not_called()

    def test_a_probe_with_no_checkpoint_fails_with_a_sentence(self):
        tap = self.enterContext(EventTap(self.app))
        job, _ = finished(tap, self.client.post("/viewer/probe", json={}))
        self.assertEqual(job["status"], "failed")
        self.assertIn(T["probe_no_checkpoint"], job["error"])

    def test_landscape_is_a_job_with_a_grid_a_minimum_and_a_path(self):
        tap = self.enterContext(EventTap(self.app))
        job, _ = finished(tap, self.client.post("/viewer/landscape", json={"grid": 9, "span": 2}))
        self.assertEqual(job["status"], "done", job)
        out = job["result"]
        self.assertEqual((out["grid"], len(out["axis"]), len(out["loss"]), len(out["loss"][0])), (9, 9, 9, 9))
        self.assertEqual(out["axis"][4], 0)  # the current weights are a cell
        self.assertEqual(out["current"]["loss"], out["loss"][4][4])
        self.assertLessEqual(out["minimum"]["loss"], out["current"]["loss"])
        self.assertTrue(out["trajectory"])
        self.assertEqual(set(out["cost"]), {"seconds", "evaluations"})
        self.assertEqual(out["mode"], "random")

    def test_landscape_live_frames_are_small_and_the_path_grows(self):
        tap = self.enterContext(EventTap(self.app))
        first, _ = finished(tap, self.client.post("/viewer/landscape", json={"live": True, "grid": 41}))
        later, _ = finished(tap, self.client.post("/viewer/landscape", json={"live": True, "grid": 41}))
        self.assertLessEqual(first["result"]["grid"], 15)
        self.assertGreater(len(later["result"]["trajectory"]), len(first["result"]["trajectory"]))

    def test_landscape_rejects_a_grid_outside_its_bounds_and_reports_a_sentence_when_it_cannot_run(self):
        self.assertEqual(self.client.post("/viewer/landscape", json={"grid": 3}).status_code, 422)
        self.assertEqual(self.client.post("/viewer/landscape", json={"mode": "tsne"}).status_code, 422)
        tap = self.enterContext(EventTap(self.app))
        with patch.object(self.services, "inspect_landscape", side_effect=ValueError("No held-back text yet.")):
            job, _ = finished(tap, self.client.post("/viewer/landscape", json={}))
        self.assertEqual(job["status"], "failed")
        self.assertIn("No held-back text yet.", job["error"])

    def test_the_viewer_process_is_started_stopped_and_spoken_to(self):
        self.assertEqual(self.client.get("/viewer/status").json(), {"available": True, "running": False, "connected": False})
        self.assertEqual(self.client.post("/viewer/start", json={"level": "Window", "mode": "Train"}).json(),
                         {"available": True, "running": True, "connected": True})
        self.assertEqual(self.client.post("/viewer/send", json={"type": "scene", "level": "Head"}).status_code, 204)
        self.assertEqual(self.client.post("/viewer/poll").json(), [])
        self.assertEqual(self.client.post("/viewer/stop").json()["running"], False)
        self.assertEqual(self.client.post("/viewer/start").status_code, 200)
        self.assertEqual(self.services.viewer_log, [("start", {"level": "Window", "mode": "Train"}), ("send", {"type": "scene", "level": "Head"}),
                                                    ("stop", None), ("start", None)])

    def test_send_needs_a_json_object(self):
        self.assertEqual(self.client.post("/viewer/send", json=["scene"]).status_code, 422)


if __name__ == "__main__":
    unittest.main()
