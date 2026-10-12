"""Tests for studio_arch bugs: BLOCK_ID regex, _int, _float."""
import unittest
from studio_arch import _int, _float, validate_spec, default_spec


class BlockIDValidation(unittest.TestCase):
    def test_block_id_rejects_newline_suffix(self):
        """Bug 1: $ in regex matches before trailing newline, so 'blk_abc123\n' is accepted."""
        # This should fail with current code but pass after fix
        with self.assertRaises(ValueError):
            spec = default_spec()
            spec["blocks"][0]["id"] = "blk_abc123\n"
            validate_spec(spec)

    def test_block_id_accepts_valid_id(self):
        """Valid block ID should pass."""
        spec = default_spec()
        spec["blocks"][0]["id"] = "blk_abc123"
        result = validate_spec(spec)
        self.assertEqual(result["blocks"][0]["id"], "blk_abc123")


class IntValidation(unittest.TestCase):
    def test_int_rejects_inf(self):
        """Bug 2: int(inf) raises OverflowError instead of ValueError."""
        with self.assertRaises(ValueError):
            _int(float('inf'), "test", 0, 100)

    def test_int_rejects_neg_inf(self):
        """Bug 2: int(-inf) raises OverflowError instead of ValueError."""
        with self.assertRaises(ValueError):
            _int(float('-inf'), "test", 0, 100)

    def test_int_rejects_nan(self):
        """Bug 2: int(nan) may raise or slip through."""
        with self.assertRaises(ValueError):
            _int(float('nan'), "test", 0, 100)

    def test_int_accepts_valid_int(self):
        """Valid int should pass."""
        result = _int(42, "test", 0, 100)
        self.assertEqual(result, 42)


class FloatValidation(unittest.TestCase):
    def test_float_rejects_inf(self):
        """Bug 2: float('inf') passes the try/except but fails range check silently."""
        with self.assertRaises(ValueError):
            _float(float('inf'), "test", 0, 1)

    def test_float_rejects_neg_inf(self):
        """Bug 2: float('-inf') passes the try/except but fails range check silently."""
        with self.assertRaises(ValueError):
            _float(float('-inf'), "test", 0, 1)

    def test_float_rejects_nan(self):
        """Bug 2: float('nan') passes the try/except but fails range check (nan <= x is always False)."""
        with self.assertRaises(ValueError):
            _float(float('nan'), "test", 0, 1)

    def test_float_accepts_valid_float(self):
        """Valid float should pass."""
        result = _float(0.5, "test", 0, 1)
        self.assertEqual(result, 0.5)


if __name__ == "__main__":
    unittest.main()
