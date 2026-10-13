"""Tests for the IA copy module."""

import unittest
import re
from studio_ia import copy


class TestIACopy(unittest.TestCase):
    """Gate tests for studio_ia.copy module."""

    # Expected h1 values per page (from the boards)
    EXPECTED_H1_VALUES = {
        "BUILDER": "Build your model",
        "CHAT": "Chat",
        "COMPARE": "Compare runs",
        "QUEUE": "This computer",
        "QUICKTEST": "Quick test",
        "TRACE": "Trace a position",
        "TRAINING": "Pretraining · version 5 catching up",
        "HEAD_ZOOM": "Inside a head",
        "VALUE_MAP": "Value map",
        "LOSS": "Loss",
        "HEAD_LAB": "Head lab",
        "INSIDE3D": "Inside the network",
        "PRETRAINING": "Pretraining",
        "FEEDBACK": "Post-training",
        "MAKEDATA": "Make extra data",
        "YOURDATA": "Your data",
        "NAV": "Network designer",  # First item in NAV
    }

    def test_all_pages_have_h1_and_lead(self):
        """Every page dict must have 'h1' and 'lead' keys."""
        page_dicts = [
            copy.BUILDER,
            copy.CHAT,
            copy.COMPARE,
            copy.QUEUE,
            copy.QUICKTEST,
            copy.TRACE,
            copy.TRAINING,
            copy.HEAD_ZOOM,
            copy.VALUE_MAP,
            copy.LOSS,
            copy.HEAD_LAB,
            copy.INSIDE3D,
            copy.PRETRAINING,
            copy.FEEDBACK,
            copy.MAKEDATA,
            copy.YOURDATA,
        ]

        for page_dict in page_dicts:
            self.assertIn("h1", page_dict, f"Missing 'h1' in {page_dict}")
            self.assertIn("lead", page_dict, f"Missing 'lead' in {page_dict}")

    def test_h1_values_match_boards(self):
        """H1 values must match the boards exactly."""
        test_cases = [
            (copy.BUILDER, self.EXPECTED_H1_VALUES["BUILDER"]),
            (copy.CHAT, self.EXPECTED_H1_VALUES["CHAT"]),
            (copy.COMPARE, self.EXPECTED_H1_VALUES["COMPARE"]),
            (copy.QUEUE, self.EXPECTED_H1_VALUES["QUEUE"]),
            (copy.QUICKTEST, self.EXPECTED_H1_VALUES["QUICKTEST"]),
            (copy.TRACE, self.EXPECTED_H1_VALUES["TRACE"]),
            (copy.TRAINING, self.EXPECTED_H1_VALUES["TRAINING"]),
            (copy.HEAD_ZOOM, self.EXPECTED_H1_VALUES["HEAD_ZOOM"]),
            (copy.VALUE_MAP, self.EXPECTED_H1_VALUES["VALUE_MAP"]),
            (copy.LOSS, self.EXPECTED_H1_VALUES["LOSS"]),
            (copy.HEAD_LAB, self.EXPECTED_H1_VALUES["HEAD_LAB"]),
            (copy.INSIDE3D, self.EXPECTED_H1_VALUES["INSIDE3D"]),
            (copy.PRETRAINING, self.EXPECTED_H1_VALUES["PRETRAINING"]),
            (copy.FEEDBACK, self.EXPECTED_H1_VALUES["FEEDBACK"]),
            (copy.MAKEDATA, self.EXPECTED_H1_VALUES["MAKEDATA"]),
            (copy.YOURDATA, self.EXPECTED_H1_VALUES["YOURDATA"]),
        ]

        for page_dict, expected_h1 in test_cases:
            self.assertEqual(
                page_dict["h1"],
                expected_h1,
                f"H1 mismatch: got '{page_dict['h1']}', expected '{expected_h1}'",
            )

    def test_no_bare_percentages_for_small_samples(self):
        r"""No string should contain bare percentage templates for small samples.

        Allowed: percentages with 'about' or '[N]' nearby.
        Not allowed: bare r"\b\d+%" (like "50%") without qualifying context.
        """
        # Collect all string values from all dicts
        all_dicts = [
            copy.BUILDER,
            copy.CHAT,
            copy.COMPARE,
            copy.QUEUE,
            copy.QUICKTEST,
            copy.TRACE,
            copy.TRAINING,
            copy.HEAD_ZOOM,
            copy.VALUE_MAP,
            copy.LOSS,
            copy.HEAD_LAB,
            copy.INSIDE3D,
            copy.PRETRAINING,
            copy.FEEDBACK,
            copy.MAKEDATA,
            copy.YOURDATA,
            copy.NAV,
            copy.MACHINE_STRIP,
        ]

        bare_percentage_pattern = re.compile(r"\b\d+%\b")
        qualifying_pattern = re.compile(r"\b\d+%.*\b(about|[N])\b|\b(about|[N])\b.*\b\d+%")

        violations = []

        for page_dict in all_dicts:
            for key, value in page_dict.items():
                if not isinstance(value, str):
                    continue

                # Check for bare percentages
                if bare_percentage_pattern.search(value):
                    # Verify it has qualifying context
                    if not qualifying_pattern.search(value):
                        violations.append(
                            f"Bare percentage in {key}: '{value}'"
                        )

        self.assertEqual(
            violations,
            [],
            "Found bare percentages without 'about' or '[N]':\n" + "\n".join(violations),
        )


if __name__ == "__main__":
    unittest.main()
