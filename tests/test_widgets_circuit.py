"""Tests for studio_widgets and studio_circuit bug fixes."""
import unittest
from unittest.mock import Mock
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import studio_widgets
import studio_circuit


class ScrollWheelDeltaTests(unittest.TestCase):
    """Test scroll wheel delta normalization for cross-platform compatibility."""

    def test_normalize_scroll_delta(self):
        """Test that normalize_scroll_delta handles both Windows (120) and macOS (1/-1) deltas."""
        # Windows uses multiples of 120
        self.assertEqual(studio_widgets.normalize_scroll_delta(120), 1)
        self.assertEqual(studio_widgets.normalize_scroll_delta(-120), -1)
        self.assertEqual(studio_widgets.normalize_scroll_delta(240), 2)
        self.assertEqual(studio_widgets.normalize_scroll_delta(-240), -2)

        # macOS uses small numbers like 1/-1
        self.assertEqual(studio_widgets.normalize_scroll_delta(1), 1)
        self.assertEqual(studio_widgets.normalize_scroll_delta(-1), -1)

        # Zero should stay zero
        self.assertEqual(studio_widgets.normalize_scroll_delta(0), 0)


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
