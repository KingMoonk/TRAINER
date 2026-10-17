"""The whole-context summary behind the Model level: computed on a tiny real checkpoint (studio_summary.whole_context), saved beside
the run's snapshots, loaded into the scene by studio_api.routers.viewer, and missing or stale without any invented numbers."""
import copy
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient

import studio_summary as summary
from studio_api.app import create_app
from studio_api.hosts import ANY
from studio_api.routers.viewer import ViewControls, build
from studio_ia.copy import INSIDE3D as T
from studio_ia.services import FakeServices
from studio_modules.tokenizer.byte import Tokenizer
from tests.api_tap import EventTap
from tests.api_work import finished, small_spec, trained
from studio_spec import validate_spec
from tests.test_export import write_checkpoint

TEXT = ["The cat sat on the mat because it was tired. " * 8, "A second held-back document. " * 8]


def spec_with_gru_second_block():
    spec = small_spec()
    spec["blocks"] = [{"index": 1, "mixer": {"id": "gru"}}]  # blocks count from 0
    return validate_spec(spec)


class WholeContext(unittest.TestCase):
    def setUp(self):
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.checkpoint = self.folder / "best.npz"

    def test_a_window_of_held_back_text_gives_attention_and_stream_sizes_per_block(self):
        spec = small_spec()
        write_checkpoint(self.checkpoint, spec, Tokenizer("byte"))
        made = summary.whole_context(self.checkpoint, TEXT)
        self.assertEqual((made["edges"][0], made["edges"][-1], made["n_ctx"], made["step"]), (0, 64, 64, 7))
        self.assertEqual(len(made["edges"]), 33)  # 32 bins
        self.assertEqual(set(made["attention"]), {"block0.attn.pattern", "block1.attn.pattern"})
        self.assertEqual(set(made["stream_norms"]), {"block0.resid.norm", "block1.resid.norm"})
        for mass in made["attention"].values():
            self.assertEqual(mass.shape, (4, 32, 32))  # heads x bins x bins
            self.assertTrue(np.isfinite(mass).all() and mass.sum() > 0)
        for norms in made["stream_norms"].values():
            self.assertEqual(norms.shape, (32,))
            self.assertTrue(np.isfinite(norms).all() and (norms > 0).all())

    def test_a_block_without_attention_has_stream_sizes_and_no_attention(self):
        write_checkpoint(self.checkpoint, spec_with_gru_second_block(), Tokenizer("byte"))
        made = summary.whole_context(self.checkpoint, TEXT)
        self.assertEqual(set(made["attention"]), {"block0.attn.pattern"})
        self.assertEqual(set(made["stream_norms"]), {"block0.resid.norm", "block1.resid.norm"})

    def test_text_shorter_than_the_context_is_said_not_padded(self):
        write_checkpoint(self.checkpoint, small_spec(), Tokenizer("byte"))
        with self.assertRaisesRegex(ValueError, "shorter than the model's 64 positions"):
            summary.whole_context(self.checkpoint, ["too short"])
        with self.assertRaisesRegex(ValueError, "no held-back text"):
            summary.whole_context(self.checkpoint, [])

    def test_a_saved_summary_records_its_checkpoint_and_goes_stale_when_that_changes(self):
        write_checkpoint(self.checkpoint, small_spec(), Tokenizer("byte"))
        self.assertEqual(summary.summary_state(self.checkpoint), ("none", None))
        made = summary.whole_context(self.checkpoint, TEXT)
        path = summary.save_summary(self.checkpoint, made)
        self.assertEqual(path, self.folder / "viz" / "summary.npz")  # beside the run's viz/step_*.npz
        state, saved = summary.summary_state(self.checkpoint)
        self.assertEqual(state, "ready")
        self.assertEqual((saved["step"], saved["n_ctx"], saved["edges"]), (7, 64, made["edges"]))
        self.assertEqual(saved["checkpoint"]["file"], "best.npz")
        np.testing.assert_array_equal(saved["attention"]["block1.attn.pattern"], made["attention"]["block1.attn.pattern"])
        np.testing.assert_array_equal(saved["stream_norms"]["block0.resid.norm"], made["stream_norms"]["block0.resid.norm"])
        write_checkpoint(self.checkpoint, small_spec(), Tokenizer("byte"), seed=1)  # training moved on: a new file
        os.utime(self.checkpoint, ns=(1, 1))
        self.assertEqual(summary.summary_state(self.checkpoint)[0], "stale")
        self.checkpoint.unlink()
        self.assertEqual(summary.summary_state(self.checkpoint), ("no_checkpoint", None))
        self.assertEqual(summary.summary_state(None), ("no_checkpoint", None))

    def test_an_unreadable_summary_file_is_none_not_a_crash(self):
        write_checkpoint(self.checkpoint, small_spec(), Tokenizer("byte"))
        (self.folder / "viz").mkdir()
        (self.folder / "viz" / "summary.npz").write_bytes(b"not an archive")
        self.assertEqual(summary.summary_state(self.checkpoint), ("none", None))

    def test_the_summary_file_does_not_disturb_the_snapshot_listing_or_run_paths(self):
        write_checkpoint(self.checkpoint, small_spec(), Tokenizer("byte"))
        summary.save_summary(self.checkpoint, summary.whole_context(self.checkpoint, TEXT))
        self.assertEqual(list((self.folder / "viz").glob("step_*.npz")), [])
        self.assertEqual(summary.run_paths([self.folder])[0]["markers"], [])


class ModelLevel(unittest.TestCase):
    def setUp(self):
        self.services = FakeServices()
        self.version = trained(self.services)
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.checkpoint = self.folder / "best.npz"
        write_checkpoint(self.checkpoint, self.version.spec, Tokenizer("byte"))
        self.version.best_checkpoint = str(self.checkpoint)
        self.services.heldout_docs = TEXT
        self.app = create_app(self.services, open=True, hosts=ANY)
        self.client = self.enterContext(TestClient(self.app))
        self.tap = self.enterContext(EventTap(self.app))

    def make(self):
        return finished(self.tap, self.client.post("/viewer/summary", json={}))[0]

    def model(self, **controls):
        """The SceneModel the viewer routes build for these controls."""
        return build(SimpleNamespace(app=self.app), ViewControls(**controls))[0]

    def frame(self):
        return self.model().frame()

    def arcs(self, scene):
        """How many attention arcs the scene draws, over all blocks."""
        return sum(line["count"] for group in scene["groups"].values() for line in group["lines"] if line["kind"] == "mixer")

    def test_a_job_makes_the_summary_and_the_scene_at_the_model_level_carries_it(self):
        before = self.client.get("/viewer/scene", params={"level": "Model"}).json()
        self.assertEqual((self.frame()["attention"], self.frame()["stream_norms"]), ({}, {}))  # nothing invented
        self.assertEqual(self.arcs(before), 0)
        self.assertEqual(before["stats"]["note"], T["summary_none"])
        job = self.make()
        self.assertEqual(job["status"], "done", job)
        self.assertEqual(job["result"], {"step": 7, "bins": 32, "n_ctx": 64, "note": T["summary_done"].format(n=self.version.number)})
        self.assertTrue((self.folder / "viz" / "summary.npz").exists())
        frame = self.frame()
        self.assertEqual(set(frame["attention"]), {"block0.attn.pattern", "block1.attn.pattern"})
        self.assertEqual(set(frame["stream_norms"]), {"block0.resid.norm", "block1.resid.norm"})
        self.assertEqual((frame["edges"][0], frame["edges"][-1], len(frame["edges"])), (0, 64, 33))
        self.assertTrue(all(np.asarray(v).shape == (4, 32, 32) for v in frame["attention"].values()))
        scene = self.client.get("/viewer/scene", params={"level": "Model"}).json()
        self.assertGreater(self.arcs(scene), 0)
        self.assertEqual(scene["stats"]["note"], "")
        self.assertEqual(self.client.post("/viewer/frame", json={"level": "Model"}).json()["stats"]["note"], "")
        self.assertGreater(len(self.client.post("/viewer/binary", json={"level": "Model", "arrays": "binary"}).content), 0)

    def bins(self, scene, gate):
        """(sizes [n, 3], values or None) of the activation boxes of one block in a Model scene."""
        (entry,) = [box for box in scene["groups"][f"gate:{gate}"]["boxes"] if box["kind"] == "activation"]
        sizes = np.asarray(entry["size"]["data"]).reshape(-1, 3)
        return sizes, (np.asarray(entry["value"]["data"]) if "value" in entry else None)

    def test_the_activation_boxes_follow_the_streams_size_when_there_is_a_summary(self):
        self.make()
        scene = self.client.get("/viewer/scene", params={"level": "Model"}).json()
        saved = summary.summary_state(self.checkpoint)[1]["stream_norms"]
        peak = max(norms.max() for norms in saved.values())
        heights = []
        for gate in (0, 1):
            sizes, values = self.bins(scene, gate)
            np.testing.assert_allclose(values, saved[f"block{gate}.resid.norm"], rtol=1e-4)  # the real number, not the drawn size
            np.testing.assert_allclose(sizes[:, 2], np.clip(2. * values / peak, .1, 2.), rtol=1e-5)
            self.assertTrue(np.all(sizes[:, :2] == [.18, .6]))
            heights.append(sizes[:, 2])
        self.assertAlmostEqual(max(h.max() for h in heights), 2., places=5)  # one scale for both blocks
        self.assertGreater(np.ptp(np.concatenate(heights)), 0)  # not all the same size
        framed = self.client.post("/viewer/frame", json={"level": "Model"}).json()
        self.assertEqual(self.bins(framed, 0)[1].tolist(), self.bins(scene, 0)[1].tolist())

    def test_the_activation_boxes_are_fixed_without_a_summary(self):
        for _ in range(2):  # none, then stale
            scene = self.client.get("/viewer/scene", params={"level": "Model"}).json()
            for gate in (0, 1):
                sizes, values = self.bins(scene, gate)
                self.assertIsNone(values)
                self.assertTrue(np.all(sizes == [.18, .6, .8]))
            self.make()
            write_checkpoint(self.checkpoint, self.version.spec, Tokenizer("byte"), seed=5)
            os.utime(self.checkpoint, ns=(1, 1))

    def test_a_gru_block_still_gets_boxes_sized_by_its_stream(self):
        self.version.spec = spec_with_gru_second_block()
        write_checkpoint(self.checkpoint, self.version.spec, Tokenizer("byte"))
        self.make()
        scene = self.client.get("/viewer/scene", params={"level": "Model"}).json()
        sizes, values = self.bins(scene, 1)
        saved = summary.summary_state(self.checkpoint)[1]["stream_norms"]["block1.resid.norm"]
        np.testing.assert_allclose(values, saved, rtol=1e-4)
        self.assertEqual(sizes.shape, (32, 3))
        self.assertFalse(np.all(sizes[:, 2] == .8))

    def test_other_levels_do_not_read_the_summary(self):
        self.make()
        self.assertEqual(self.model(level="Window").summaries, {})
        self.assertEqual(self.model(level="Model").summaries.keys(), self.frame()["attention"].keys())
        scene = self.client.get("/viewer/scene", params={"level": "Window"}).json()
        self.assertNotIn("summary", scene["stats"]["note"].lower())

    def test_a_block_without_attention_gets_stream_sizes_and_no_arcs(self):
        self.version.spec = spec_with_gru_second_block()
        write_checkpoint(self.checkpoint, self.version.spec, Tokenizer("byte"))
        self.assertEqual(self.make()["status"], "done")
        frame = self.frame()
        self.assertEqual(set(frame["attention"]), {"block0.attn.pattern"})
        self.assertEqual(set(frame["stream_norms"]), {"block0.resid.norm", "block1.resid.norm"})
        scene = self.client.get("/viewer/scene", params={"level": "Model"}).json()
        second = scene["groups"]["gate:1"]
        self.assertEqual([line for line in second["lines"] if line["kind"] == "mixer"], [])
        self.assertGreater(sum(line["count"] for line in scene["groups"]["gate:0"]["lines"] if line["kind"] == "mixer"), 0)

    def test_a_summary_of_an_older_checkpoint_is_hidden_and_the_note_says_how_to_remake_it(self):
        self.make()
        write_checkpoint(self.checkpoint, self.version.spec, Tokenizer("byte"), seed=3)
        os.utime(self.checkpoint, ns=(1, 1))
        scene = self.client.get("/viewer/scene", params={"level": "Model"}).json()
        self.assertEqual(scene["stats"]["note"], T["summary_stale"])
        self.assertEqual(self.arcs(scene), 0)
        self.assertEqual(self.frame()["attention"], {})
        self.assertEqual(self.client.get("/viewer/summary").json()["state"], "stale")
        self.assertEqual(self.make()["status"], "done")  # run again: current
        self.assertEqual(self.client.get("/viewer/summary").json()["state"], "ready")
        self.assertEqual(self.client.get("/viewer/scene", params={"level": "Model"}).json()["stats"]["note"], "")

    def test_a_summary_for_another_context_length_is_stale(self):
        self.make()
        spec = copy.deepcopy(self.version.spec)
        spec["dims"]["n_ctx"] = 32
        self.version.spec = spec
        self.assertEqual(self.client.get("/viewer/summary").json()["state"], "stale")
        self.assertEqual(self.frame()["attention"], {})

    def test_the_advice_in_a_clients_note_goes_once_the_summary_is_there(self):
        advice = self.client.get("/viewer/scene", params={"level": "Model"}).json()["stats"]["note"]
        self.make()
        scene = self.client.post("/viewer/frame", json={"level": "Model", "note": advice}).json()
        self.assertEqual(scene["stats"]["note"], "")
        own = self.client.post("/viewer/frame", json={"level": "Model", "note": "Could not load the snapshot: gone"}).json()
        self.assertEqual(own["stats"]["note"], "Could not load the snapshot: gone")  # a note that is not ours stays

    def test_a_version_without_a_checkpoint_still_draws_and_says_so(self):
        self.version.best_checkpoint = None
        response = self.client.get("/viewer/scene", params={"level": "Model"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["stats"]["note"], T["summary_no_checkpoint"])
        self.assertTrue(response.json()["groups"])
        status = self.client.get("/viewer/summary").json()
        self.assertEqual((status["state"], status["step"], status["bins"]), ("no_checkpoint", None, None))
        job = self.make()
        self.assertEqual((job["status"], T["summary_no_checkpoint"] in job["error"]), ("failed", True))

    def test_without_held_back_text_the_job_fails_with_a_sentence_and_no_summary_is_written(self):
        self.services.heldout_docs = []
        job = self.make()
        self.assertEqual(job["status"], "failed")
        self.assertIn("no held-back text", job["error"])
        self.assertFalse((self.folder / "viz").exists())

    def test_status_says_what_the_saved_summary_is(self):
        self.assertEqual(self.client.get("/viewer/summary").json()["state"], "none")
        self.make()
        status = self.client.get("/viewer/summary").json()
        self.assertEqual((status["state"], status["step"], status["bins"], status["n_ctx"]), ("ready", 7, 32, 64))
        self.assertEqual(status["note"], T["summary_done"].format(n=self.version.number))
        self.assertEqual(self.client.get("/viewer/summary", params={"version": 9}).status_code, 400)

    def test_the_probe_step_makes_a_missing_summary_but_not_a_current_one(self):
        self.services.probe_path = str(self.folder / "probe.npz")
        finished(self.tap, self.client.post("/viewer/probe", json={}))
        self.assertEqual(self.client.get("/viewer/summary").json()["state"], "ready")
        saved = summary.summary_path(self.checkpoint)
        made = saved.stat().st_mtime_ns
        finished(self.tap, self.client.post("/viewer/probe", json={}))
        self.assertEqual(saved.stat().st_mtime_ns, made)  # fresh already: not computed again

    def test_a_probe_still_works_when_the_summary_cannot_be_made(self):
        self.services.heldout_docs = []
        self.services.probe_path = str(self.folder / "probe.npz")
        job, _ = finished(self.tap, self.client.post("/viewer/probe", json={}))
        self.assertEqual(job["status"], "done", job)
        self.assertEqual(self.client.get("/viewer/summary").json()["state"], "none")


if __name__ == "__main__":
    unittest.main()
