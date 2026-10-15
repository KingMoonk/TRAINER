import unittest

from studio_widgets import touchpad_dy


class TouchpadDeltaTests(unittest.TestCase):
    def test_vertical_pixels_come_from_the_signed_low_half(self):
        self.assertEqual(touchpad_dy(7), 7)
        self.assertEqual(touchpad_dy(0x10000 - 7), -7)

    def test_horizontal_pixels_in_the_high_half_are_ignored(self):
        self.assertEqual(touchpad_dy((3 << 16) | 5), 5)
        self.assertEqual(touchpad_dy((-3 << 16) | (0x10000 - 5)), -5)


if __name__ == "__main__":
    unittest.main()
