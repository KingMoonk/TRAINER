"""Evidence-based autopilot parameter choices; no training or application window."""
import math
import json
import tempfile
import unittest
from pathlib import Path

from studio_autopilot_control import choose


def phase(**config):
    return {"kind": "pretrain", "datasets": ["text"], "config": {"learning_rate": .004, "steps": 1000,
                                                                "batch_size": 8, "width": 64, "layers": 2, **config}}


class ControlTests(unittest.TestCase):
    def test_divergence_halves_learning_rate_and_keeps_original_spec(self):
        incoming = phase()
        choice = choose(incoming, [{"index": 0, "kind": "pretrain", "reason": "validation diverged", "accepted": False}],
                        {"learning_rate": (.0001, .005)})
        self.assertEqual(choice["spec"]["config"]["learning_rate"], .002)
        self.assertEqual(incoming["config"]["learning_rate"], .004)
        self.assertIn("diverged", choice["reason"])
        self.assertEqual(choice["spec"]["datasets"], ["text"])

    def test_measured_lr_probe_picks_best_rate_within_bounds(self):
        choice = choose(phase(), [{"index": 0, "kind": "pretrain", "metrics": {
            "lr_probe": [[.0001, 2.1], [.001, 1.6], [.004, 2.0], [.01, 1.0]]}}],
            {"learning_rate": (.0001, .005)})
        self.assertEqual(choice["spec"]["config"]["learning_rate"], .001)
        self.assertEqual(len(choice["evidence"]["lr_probe"]["points"]), 3)

    def test_falling_heldout_curve_lengthens_same_phase_but_not_different_phase(self):
        points = [[step, 1. + 100 / math.sqrt(step)] for step in (100, 150, 225, 337, 506, 759, 1000)]
        previous = {"index": 0, "kind": "pretrain", "metrics": {"loss_curve": points}}
        chosen = choose(phase(), [previous], {"steps": (100, 1800)})
        self.assertGreater(chosen["spec"]["config"]["steps"], 1000)
        self.assertLessEqual(chosen["spec"]["config"]["steps"], 1800)
        self.assertIn("loss_fit", chosen["evidence"])
        other = choose({**phase(), "kind": "sft"}, [previous], {"steps": (100, 1800)})
        self.assertEqual(other["spec"]["config"]["steps"], 1000)

    def test_too_small_plateau_proposes_bounded_growth_without_mutating_architecture(self):
        previous = {"index": 0, "kind": "pretrain", "reason": "model too small on new data"}
        chosen = choose(phase(), [previous], {"width": (32, 72), "layers": (1, 2)})
        self.assertEqual(chosen["evidence"]["design_proposal"], {"width": 72})
        self.assertEqual(chosen["spec"]["config"]["width"], 64)
        self.assertIn("larger design", chosen["reason"])

    def test_checkpoint_metrics_are_used_when_no_metrics_are_attached(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            rows = [{"step": step, "val_loss": 1. + 100 / math.sqrt(step)}
                    for step in (100, 150, 225, 337, 506, 759, 1000)]
            (path / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            chosen = choose(phase(), [{"index": 0, "kind": "pretrain", "checkpoint": str(path / "best.npz")}],
                            {"steps": (100, 2000)})
        self.assertGreater(chosen["spec"]["config"]["steps"], 1000)

    def test_all_selected_knobs_respect_bounds(self):
        incoming = phase(steps=10000, batch_size=32, width=400, layers=8, context=1024,
                         beta=.4, sft_weight=4., kl_coefficient=2.)
        chosen = choose(incoming, [{"kind": "pretrain", "metrics": {"diverged": True,
                         "max_batch_size": 4, "too_small": True}}],
                        {"learning_rate": (.003, .005), "steps": (10, 2000), "batch_size": (2, 8),
                         "width": (32, 256), "layers": (1, 6), "context": (128, 512),
                         "beta": (.01, .2), "sft_weight": (0., 2.), "kl_coefficient": (0., 1.)})
        config = chosen["spec"]["config"]
        self.assertEqual((config["learning_rate"], config["steps"], config["batch_size"]), (.003, 2000, 4))
        self.assertEqual((config["width"], config["layers"], config["context"]), (256, 6, 512))
        self.assertEqual((config["beta"], config["sft_weight"], config["kl_coefficient"]), (.2, 2., 1.))

    def test_invalid_bounds_fail_closed(self):
        for bounds in ({"steps": (0, 100)}, {"learning_rate": (.01, .001)},
                       {"batch_size": (1.5, 4)}, {"width": (16, float("inf"))}):
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                choose(phase(), [], bounds)
        with self.assertRaisesRegex(ValueError, "memory fit"):
            choose(phase(), [{"kind": "pretrain", "metrics": {"max_batch_size": 1}}], {"batch_size": (2, 8)})


if __name__ == "__main__":
    unittest.main()
