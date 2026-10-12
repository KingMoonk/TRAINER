"""Tests for studio_widgets and studio_circuit bug fixes."""
import unittest
from unittest.mock import Mock
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import studio_widgets
import studio_circuit


class WheelDeltaTests(unittest.TestCase):
    """Test wheel delta conversion for cross-platform compatibility."""

    def test_normalize_scroll_delta_windows_full_units(self):
        """Test that Windows 120-unit deltas give 1 scroll unit."""
        # Assume we're testing on a system where _wheel_notches works correctly
        self.assertEqual(studio_widgets.normalize_scroll_delta(120), 1)
        self.assertEqual(studio_widgets.normalize_scroll_delta(-120), -1)

    def test_normalize_scroll_delta_symmetry(self):
        """Test that scroll delta rounding is symmetric (+180 vs -180)."""
        # Same absolute value should give same absolute magnitude
        pos = studio_widgets.normalize_scroll_delta(180)
        neg = studio_widgets.normalize_scroll_delta(-180)
        self.assertEqual(abs(pos), abs(neg))
        self.assertEqual(pos, -neg)

    def test_normalize_scroll_delta_never_zero_for_nonzero_input(self):
        """Test that non-zero delta never returns 0."""
        for delta in [1, -1, 30, -30, 60, -60]:
            result = studio_widgets.normalize_scroll_delta(delta)
            if delta != 0:
                self.assertNotEqual(result, 0, f"normalize_scroll_delta({delta}) returned 0")

    def test_normalize_scroll_delta_zero_stays_zero(self):
        """Test that zero delta returns zero."""
        self.assertEqual(studio_widgets.normalize_scroll_delta(0), 0)

    def test_wheel_notches_returns_float(self):
        """Test that _wheel_notches returns a float for zoom calculations."""
        # Should return a float that can be used with ** operator
        notches = studio_widgets._wheel_notches(120)
        self.assertIsInstance(notches, float)
        # Should be usable in zoom calculation
        zoom_factor = 1.12 ** notches
        self.assertIsInstance(zoom_factor, float)
        self.assertGreater(zoom_factor, 1.0)

    def test_wheel_notches_sub_120_windows_delta(self):
        """Test that sub-120 Windows delta gives fractional notches."""
        # Precision mice can send deltas like 30 (quarter scroll)
        notches = studio_widgets._wheel_notches(30)
        # Should be a fraction like 0.25 (30/120)
        self.assertGreater(notches, 0)
        self.assertLess(abs(notches), 1.0)

    def test_zoom_base_constant_exists(self):
        """Test that zoom base constant is defined."""
        self.assertTrue(hasattr(studio_widgets, '_ZOOM_BASE'))
        self.assertGreater(studio_widgets._ZOOM_BASE, 1.0)
        self.assertLess(studio_widgets._ZOOM_BASE, 1.15)


class CircuitDescribeTests(unittest.TestCase):
    """Test the describe() method handles all pad types correctly."""

    def test_describe_output_pad(self):
        """Test that output pads produce correct descriptions."""
        # Create a mock circuit with necessary attributes
        circuit = Mock(spec=studio_circuit.CircuitCanvas)
        circuit.s = {"blocks": [{"id": "block1"}, {"id": "block2"}]}
        circuit.tele = {"blocks": []}
        circuit._unit_count = Mock(return_value=6)

        # Use the actual describe method
        describe = studio_circuit.CircuitCanvas.describe

        # Test output pad tag p:e:out:0
        result = describe(circuit, "p:e:out:0")
        self.assertIn("output", result.lower())
        self.assertNotEqual(result, "Embedding dimension sample")

        # Test output pad tag p:e:out:5
        result = describe(circuit, "p:e:out:5")
        self.assertIn("output", result.lower())

        # Test embedding pad tag p:e:emb:0 (should work as before)
        result = describe(circuit, "p:e:emb:0")
        self.assertIn("embedding", result.lower())


if __name__ == "__main__":
    unittest.main()
