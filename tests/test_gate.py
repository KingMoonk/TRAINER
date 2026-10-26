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
        ok, checks = studio_gate.passes({"bpc": None}, {"bpc": None}, number=2)
        self.assertTrue(ok)
        self.assertEqual([c["ok"] for c in checks], [None, True])
        self.assertFalse(studio_gate.passes({"bpc": 2.0}, {"bpc": 2.1}, number=3)[0])

    def test_without_a_round_number_the_old_rule_holds(self):
        self.assertEqual(len(studio_gate.passes(measured(), measured())[1]), 2)
        self.assertTrue(studio_gate.passes({"bpc": 2.0}, {"bpc": 2.0})[0])


if __name__ == "__main__":
    unittest.main()
