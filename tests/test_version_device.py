"""A version records the device kind that trained it and the one that scored it, and a checkpoint resumes on whatever device is present."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import studio_eval
import studio_publish
from studio_backend import get_backend
from studio_model import checkpoint_device, load_checkpoint, train
from studio_spec import default_spec
from studio_versions import VersionStore
from tests.test_publish import EXAMPLES, TINY


def present():
    """The device `auto` picks here: cuda or mps when torch can use one, else cpu."""
    return get_backend("auto").device


class Recorded(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def checkpoint(self, device="cpu", steps=4):
        out = self.root / "run"
        train(dict(TINY, device=device, steps=steps, eval_every=steps), EXAMPLES, out, telemetry_every=0)
        return out / "last.npz"

    def test_a_checkpoint_says_which_device_trained_it(self):
        self.assertEqual(checkpoint_device(self.checkpoint("cpu")), "cpu")
        self.assertIsNone(checkpoint_device(self.root / "missing.npz"))
        (self.root / "junk.npz").write_bytes(b"not an archive")
        self.assertIsNone(checkpoint_device(self.root / "junk.npz"))

    def test_a_checkpoint_that_names_no_known_device_has_none(self):
        arrays = dict(np.load(self.checkpoint("cpu"), allow_pickle=False))
        inner = json.loads(str(arrays["metadata"].item()))
        for bad in (None, "tpu", "auto"):
            inner["device"] = bad
            np.savez(self.root / "odd.npz", **{**arrays, "metadata": np.array(json.dumps(inner))})
            self.assertIsNone(checkpoint_device(self.root / "odd.npz"))

    def test_the_version_row_keeps_both_devices(self):
        store = VersionStore(self.root)
        number = store.save_version(default_spec())["number"]
        self.assertNotIn("train_device", store.get(number))
        store.set_devices(number, train="cuda")
        self.assertEqual((store.get(number)["train_device"], store.get(number).get("score_device")), ("cuda", None))
        store.set_devices(number, score=studio_eval.SCORE_DEVICE)
        self.assertEqual((store.get(number)["train_device"], store.get(number)["score_device"]), ("cuda", "cpu"))
        store.set_devices(number)  # nothing to say leaves both as they were
        self.assertEqual((store.get(number)["train_device"], store.get(number)["score_device"]), ("cuda", "cpu"))

    def test_a_pulled_version_keeps_the_device_that_trained_it(self):
        row = VersionStore(self.root).add_pulled(default_spec(), "from mac", "pulled", "x.npz", "mac/v1-abc", "mps")
        self.assertEqual((row["train_device"], row.get("score_device")), ("mps", None))

    def test_the_published_metadata_names_both_devices(self):
        path = self.checkpoint("cpu")
        scratch = self.root / "scratch"
        scratch.mkdir()
        version = {"number": 1, "name": "v", "change": "", "parent": None, "spec": {}, "scorecard": {"bpc": 2.0}, "score_device": "cpu"}
        _, meta = studio_publish.build_artifact(version, path, scratch, "box")
        self.assertEqual((meta["train_device"], meta["score_device"], meta["device"]), ("cpu", "cpu", "cpu"))
        _, unscored = studio_publish.build_artifact({**version, "score_device": None}, path, scratch, "box")
        self.assertIsNone(unscored["score_device"])

    def test_a_finished_job_records_who_trained_and_who_scored_the_version(self):
        from types import SimpleNamespace

        from studio_ia.services import Job, WorkspaceServices
        path = self.checkpoint("cpu")
        store = VersionStore(self.root)
        number = store.save_version(default_spec())["number"]
        services = object.__new__(WorkspaceServices)
        services.store, services.root = store, self.root
        job = Job("job1", "pretrain", params={"version": number, "output_dir": str(path.parent)})
        services.runner = SimpleNamespace(jobs=[job])
        services._held_scores = lambda checkpoint: np.array([[3., 10.]])
        services._to = lambda *args: None
        services.emit = lambda *args: None
        services._record_job("job", {"job": job.id, "status": "done", "error": ""})
        row = store.get(number)
        self.assertEqual((row["train_device"], row["score_device"]), ("cpu", studio_eval.SCORE_DEVICE))


class Resume(unittest.TestCase):
    def test_a_saved_checkpoint_loads_and_resumes_on_the_device_present(self):
        device = present()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = dict(TINY, device=device, eval_every=5)
            train(dict(config, steps=20), EXAMPLES, root / "first", telemetry_every=0)
            saved = load_checkpoint(root / "first" / "last.npz")
            self.assertEqual((saved["step"], saved["device"]), (20, device))
            train(dict(config, steps=25), EXAMPLES, root / "second", resume=root / "first" / "last.npz", telemetry_every=0)
            resumed = load_checkpoint(root / "second" / "last.npz")
            self.assertEqual((resumed["step"], resumed["optimizer_steps"], resumed["device"]), (25, 25, device))
            self.assertFalse(np.array_equal(saved["model"]["token"], resumed["model"]["token"]))
            # training continued rather than restarting: the loss after resuming is below an untrained model's, near where it left off
            before, after = saved["history"][-1]["loss"], resumed["history"][-1]["loss"]
            self.assertLess(after, saved["history"][0]["loss"])
            self.assertLess(abs(after - before), max(.5, before))


if __name__ == "__main__":
    unittest.main()
