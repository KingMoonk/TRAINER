"""WorkspaceServices' parts of the plan's sizing: the kept step rate, and the earlier run's validation points."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

RATE = {"seconds_per_step": .25, "tokens_per_second": 1000., "device": "cpu", "device_name": "PyTorch / CPU", "note": "", "rows": 4, "context": 256}


class StepRates(unittest.TestCase):
    def setUp(self):
        from studio_ia.services import FixtureServices
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.svc = FixtureServices(self.tmp.name)
        self.addCleanup(self.svc.close)
        self.root = Path(self.tmp.name)

    def test_a_measured_rate_is_kept_for_that_design_and_those_rows_across_restarts(self):
        from studio_ia.services import FixtureServices
        spec = self.svc.current_version().spec
        self.assertIsNone(self.svc.cached_step_rate(spec, 4))
        with mock.patch("studio_scaling.measure_step_rate", return_value=dict(RATE)) as measure:
            found = self.svc.measure_step_rate(spec, 4, 263)
        measure.assert_called_once()
        self.assertEqual(found["seconds_per_step"], .25)
        self.assertEqual(self.svc.cached_step_rate(spec, 4)["seconds_per_step"], .25)
        self.assertIsNone(self.svc.cached_step_rate(spec, 8))  # another batch costs another time
        changed = json.loads(json.dumps(spec))
        changed["name"] = "a wider design"
        self.assertIsNone(self.svc.cached_step_rate(changed, 4))
        again = FixtureServices(self.tmp.name)
        self.addCleanup(again.close)
        self.assertEqual(again.cached_step_rate(spec, 4)["seconds_per_step"], .25)

    def test_a_stopped_measure_keeps_nothing(self):
        spec = self.svc.current_version().spec
        with mock.patch("studio_scaling.measure_step_rate", return_value=None):
            self.assertIsNone(self.svc.measure_step_rate(spec, 4, 263))
        self.assertIsNone(self.svc.cached_step_rate(spec, 4))


class LossEvidence(unittest.TestCase):
    def setUp(self):
        from studio_ia.services import FixtureServices
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.svc = FixtureServices(self.tmp.name)
        self.addCleanup(self.svc.close)
        self.version = self.svc.versions()[0]

    def run_folder(self, rows, record=None, name="run1"):
        folder = Path(self.tmp.name) / "runs" / name
        folder.mkdir(parents=True)
        (folder / "config.json").write_text(json.dumps({"batch_size": 8, "accumulation": 2}), encoding="utf-8")
        (folder / "spec.json").write_text(json.dumps({"dims": {"n_ctx": 64}}), encoding="utf-8")
        (folder / "run.json").write_text(json.dumps(record or {}), encoding="utf-8")
        (folder / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        self.svc.store._update(self.version.number, runs=[str(folder)])
        return folder

    def test_the_validation_points_are_in_word_pieces_read(self):
        rows = [{"step": 0, "loss": 5., "val_loss": 5.5}, {"step": 10, "loss": 3.}, {"step": 20, "loss": 2.9, "val_loss": 3.1},
                {"step": 40, "loss": 2.5, "val_loss": 2.7}]
        folder = self.run_folder(rows)
        found = self.svc.loss_evidence(self.version)
        per_step = 8 * 2 * 64
        self.assertEqual(found["points"], [(20 * per_step, 3.1), (40 * per_step, 2.7)])  # step 0 is the untrained model: not on the curve
        self.assertEqual(found["run"], str(folder))

    def test_a_run_that_began_from_other_weights_is_not_this_designs_curve(self):
        rows = [{"step": 20, "loss": 2.9, "val_loss": 3.1}]
        self.run_folder(rows, {"initialized_from": "runs/before/best.npz"})
        self.assertIsNone(self.svc.loss_evidence(self.version))

    def test_no_run_or_no_validation_points_is_no_evidence(self):
        self.svc.store._update(self.version.number, runs=[])
        self.assertIsNone(self.svc.loss_evidence(self.version))
        self.run_folder([{"step": 20, "loss": 2.9}], name="run2")
        self.assertIsNone(self.svc.loss_evidence(self.version))

    def test_a_run_folder_without_its_files_is_no_evidence(self):
        self.svc.store._update(self.version.number, runs=[str(Path(self.tmp.name) / "runs" / "gone")])
        self.assertIsNone(self.svc.loss_evidence(self.version))


if __name__ == "__main__":
    unittest.main()
