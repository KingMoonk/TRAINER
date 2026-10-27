"""studio_gate.passes: the Round 1 chat rule, the drift budget and what an unmeasured check does."""
import unittest

import studio_gate


def measured(bpc=2.0, conv_loss=1.5, eos_rate=0.5, **more):
    return {"bpc": bpc, "conv_loss": conv_loss, "eos_rate": eos_rate, **more}


def round_one(after, before=None, **kw):
    return studio_gate.passes(before or measured(), after, number=1, **kw)


class RoundOneGate(unittest.TestCase):
    def test_passes_when_conversation_loss_fell_and_stops_did_not_and_text_stayed_in_budget(self):
        ok, checks = round_one(measured(bpc=2.08, conv_loss=1.2, eos_rate=0.5))
        self.assertTrue(ok)
        self.assertEqual([c["ok"] for c in checks], [True, True, True, True])
        self.assertEqual(checks[0]["detail"], "1.200 per answer token, was 1.500")

    def test_each_condition_alone_fails_the_round(self):
        for name, after in (("conv loss up", measured(conv_loss=1.5)), ("conv loss equal", measured(conv_loss=1.5)),
                            ("stops fell", measured(conv_loss=1.0, eos_rate=0.4)), ("text drifted", measured(bpc=2.2, conv_loss=1.0))):
            with self.subTest(name):
                self.assertFalse(round_one(after)[0])

    def test_the_bootstrap_band_does_not_decide_round_one(self):
        after = measured(bpc=2.09, conv_loss=1.0)   # inside the budget, beyond a tight band
        self.assertTrue(round_one(after, band=(1.99, 2.01))[0])

    def test_the_drift_budget_is_configurable(self):
        after = measured(bpc=2.08, conv_loss=1.0)
        self.assertFalse(round_one(after, drift=0.02)[0])
        self.assertTrue(round_one(after, drift=0.05)[0])

    def test_no_held_back_text_is_not_measured_and_the_round_can_still_pass(self):
        ok, checks = round_one(measured(bpc=None, conv_loss=1.0), before=measured(bpc=None))
        self.assertTrue(ok)
        bpc = next(c for c in checks if c["name"].startswith("Raw text"))
        self.assertIsNone(bpc["ok"])
        self.assertIn("not measured", bpc["detail"])

    def test_no_held_back_conversations_is_no_evidence_so_the_round_fails(self):
        ok, checks = round_one(measured(conv_loss=None, eos_rate=None), before=measured(conv_loss=None, eos_rate=None))
        self.assertFalse(ok)
        self.assertEqual([c["ok"] for c in checks[:2]], [False, False])

    def test_the_problem_pass_rate_still_gates(self):
        before, after = measured(pass_k=4, pass_n=5), measured(conv_loss=1.0, pass_k=3, pass_n=5)
        self.assertFalse(round_one(after, before=before)[0])


class OtherRounds(unittest.TestCase):
    def test_unmeasured_text_does_not_fail_and_a_rise_still_does(self):
        ok, checks = studio_gate.passes({"bpc": None}, {"bpc": None}, number=3, record={"kl_before": 1., "kl_after": .5})
        self.assertTrue(ok)
        self.assertEqual([c["ok"] for c in checks], [None, True, True])
        self.assertFalse(studio_gate.passes({"bpc": 2.0}, {"bpc": 2.1}, number=3)[0])

    def test_without_a_round_number_the_old_rule_holds(self):
        self.assertEqual(len(studio_gate.passes(measured(), measured())[1]), 2)
        self.assertTrue(studio_gate.passes({"bpc": 2.0}, {"bpc": 2.0})[0])


def problems(k, n=100, bpc=2.0):
    return {"bpc": bpc, "pass_k": k, "pass_n": n}


class ImprovementRounds(unittest.TestCase):
    def test_round_two_needs_the_pass_rate_to_rise_by_more_than_one_over_root_n(self):
        before = problems(40)
        self.assertFalse(studio_gate.passes(before, problems(40), number=2)[0])      # unchanged: a no-op round
        self.assertFalse(studio_gate.passes(before, problems(50), number=2)[0])      # +10 is not more than 1/sqrt(100)
        ok, checks = studio_gate.passes(before, problems(51), number=2)
        self.assertTrue(ok)
        self.assertEqual([c["ok"] for c in checks], [True, True])

    def test_round_two_with_few_problems_needs_one_more_solved(self):
        self.assertFalse(studio_gate.passes(problems(1, 3), problems(1, 3), number=2)[0])
        self.assertTrue(studio_gate.passes(problems(1, 3), problems(2, 3), number=2)[0])

    def test_round_two_without_problems_is_no_evidence_and_fails(self):
        ok, checks = studio_gate.passes(problems(0, 0), problems(0, 0), number=2)
        self.assertFalse(ok)
        self.assertIn("not measured", checks[-1]["detail"])

    def test_round_two_still_fails_when_text_error_rose(self):
        self.assertFalse(studio_gate.passes(problems(40), problems(60, bpc=2.5), number=2)[0])

    def test_round_three_needs_the_teacher_kl_down_and_the_pass_rate_not_down(self):
        record = {"kl_before": 1.0, "kl_after": 0.8}
        ok, checks = studio_gate.passes(problems(40), problems(40), number=3, record=record)
        self.assertTrue(ok)
        self.assertEqual(checks[-1]["name"], "Teacher gap: KL to the teacher fell")
        self.assertFalse(studio_gate.passes(problems(40), problems(40), number=3, record={"kl_before": 1.0, "kl_after": 1.0})[0])
        self.assertFalse(studio_gate.passes(problems(40), problems(39), number=3, record=record)[0])

    def test_round_three_without_a_kl_record_fails(self):
        for record in (None, {}, {"kl_before": 1.0, "kl_after": float("nan")}):
            with self.subTest(record=record):
                ok, checks = studio_gate.passes(problems(40), problems(40), number=3, record=record)
                self.assertFalse(ok)
                self.assertIs(checks[-1]["ok"], False)

    def test_round_four_needs_held_back_accuracy_above_55_percent(self):
        def run(accuracy, pairs=20):
            return studio_gate.passes(problems(40), problems(40), number=4, record={"validation_pairs": pairs, "final_validation": {"reward_accuracy": accuracy}})
        self.assertTrue(run(0.6)[0])
        self.assertFalse(run(0.55)[0])
        self.assertFalse(run(0.5)[0])
        self.assertIn("not above 55%", run(0.5)[1][1]["detail"])

    def test_round_four_with_no_held_back_pairs_fails(self):
        for record in (None, {"validation_pairs": 0, "final_validation": None}):
            with self.subTest(record=record):
                ok, checks = studio_gate.passes(problems(40), problems(40), number=4, record=record)
                self.assertFalse(ok)
                self.assertIn("not measured", checks[1]["detail"])


if __name__ == "__main__":
    unittest.main()
