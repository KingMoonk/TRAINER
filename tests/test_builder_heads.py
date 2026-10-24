"""Twelve attention heads (#657): the Builder offers them, and a width they do not split rounds to one they do."""
import unittest

from studio_ia import design as D
from studio_spec import default_spec
from studio_validate import validate


def wide(d):
    spec = default_spec()
    spec["dims"]["d_model"] = d
    return spec


class TwelveHeads(unittest.TestCase):
    def test_twelve_is_the_largest_choice(self):
        self.assertEqual(max(D.HEADS), 12)

    def test_widths_round_to_the_nearest_that_splits_into_even_heads(self):
        self.assertEqual([D.width_for(d, 12, even=True) for d in (64, 256, 384, 512, 1024)], [72, 264, 384, 504, 1008])
        self.assertEqual((D.width_for(256, 8, even=True), D.width_for(100, 4), D.width_for(100, 4, even=True)), (256, 100, 104))

    def test_twelve_heads_on_a_256_wide_design_write_a_valid_264_wide_spec(self):
        base = wide(256)
        cur = D.read_spec(base)
        spec = D.write_spec(base, dict(cur, H=12), cur)
        mixer = spec["block"]["mixer"]["config"]
        self.assertEqual((spec["dims"]["d_model"], mixer["heads"], 12 % mixer.get("kv_heads", 1)), (264, 12, 0))
        self.assertEqual([i.text for i in validate(spec) if i.level == "ERROR"], [])
        self.assertIn("each word's width goes from 256 to 264: a little bigger and slower.", " ".join(D.expectations(cur, dict(cur, H=12))))

    def test_a_width_that_already_fits_is_kept(self):
        base = wide(256)
        cur = D.read_spec(base)
        spec = D.write_spec(base, dict(cur, H=1 if cur["H"] != 1 else 4), cur)
        self.assertEqual(spec["dims"]["d_model"], 256)
        self.assertFalse(any("width goes from" in line for line in D.expectations(cur, dict(cur, H=8))))


if __name__ == "__main__":
    unittest.main()
