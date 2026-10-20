"""Issue #484: the optimizer step, gradient zeroing and accumulation use foreach kernels. The arithmetic is the per-parameter
loop's, so a seeded run ends on the same weights, optimizer state and loss (within float rounding)."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from studio_model import load_checkpoint, train
from studio_modules.optimizer.adamw import AdamW
from tests.test_optimizer_offload import CONFIG, EXAMPLES


def run(root, name, foreach):
    out = Path(root) / name
    losses = []
    patch = mock.patch.object(AdamW, "apply_all", AdamW.apply_all if foreach else None)
    with patch:
        train(dict(CONFIG, steps=6, accumulation=1), EXAMPLES, out, emit=lambda e: losses.append(e["loss"]) if e.get("type") == "metric" else None)
    return load_checkpoint(out / "last.npz"), losses


class ForeachChecks(unittest.TestCase):
    def test_foreach_adamw_matches_the_per_parameter_loop_for_the_same_seed(self):
        with tempfile.TemporaryDirectory() as root:
            loop, loop_losses = run(root, "loop", False)
            fused, fused_losses = run(root, "foreach", True)
        np.testing.assert_allclose(fused_losses, loop_losses, rtol=1e-5, atol=1e-6)
        for key, value in loop["model"].items():
            np.testing.assert_allclose(fused["model"][key], value, rtol=1e-5, atol=1e-6, err_msg=key)
        for state in ("m", "v"):
            for key, value in loop["optimizer_state"][state].items():
                np.testing.assert_allclose(fused["optimizer_state"][state][key], value, rtol=1e-5, atol=1e-6, err_msg=f"{state}::{key}")

    def test_the_foreach_path_is_the_one_a_plain_run_takes(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.object(AdamW, "apply_all", autospec=True, side_effect=AdamW.apply_all) as used:
            train(dict(CONFIG, steps=2, accumulation=1), EXAMPLES, Path(root) / "r")
        self.assertEqual(used.call_count, 2)
