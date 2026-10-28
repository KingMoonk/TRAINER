import unittest

import studio_autotune as tune

B = tune.bounds_of({})


def falling():     # loss still falling steeply with tokens
    return [(x, 0.1 + 8 * x ** -0.5) for x in (100, 200, 400, 800, 1600, 3200, 6400)]


def flat():        # loss at its floor
    return [(x, 2 + 0.0001 * x ** -0.5) for x in (100, 200, 400, 800, 1600, 3200, 6400)]


class OutcomeTests(unittest.TestCase):
    def test_a_non_finite_loss_or_a_flag_or_the_words_say_diverged(self):
        self.assertTrue(tune.outcome("sft", 1, "done", {"after": {"bpc": float("nan")}}, {})["diverged"])
        self.assertTrue(tune.outcome("sft", 1, "failed", {"diverged": True}, {})["diverged"])
        self.assertTrue(tune.outcome("sft", 1, "failed", {"error": "Loss diverged at step 4"}, {})["diverged"])
        self.assertFalse(tune.outcome("sft", 1, "done", {"after": {"bpc": 1.2}, "ok": True}, {})["diverged"])

    def test_a_curve_gives_the_gain_per_doubling_and_a_flat_pretrain_is_too_small(self):
        self.assertGreater(tune.outcome("pretrain", 1, "done", {"curve": falling()}, {})["gain_per_doubling"], 0.02)
        out = tune.outcome("pretrain", 1, "done", {"curve": flat()}, {})
        self.assertLess(out["gain_per_doubling"], 0.02)
        self.assertTrue(out["too_small"])
        self.assertTrue(tune.outcome("midtrain", 1, "done", {"too_small": True}, {})["too_small"])


class DecideTests(unittest.TestCase):
    def history(self, kind, result, used, status="done", attempt=1):
        return [tune.outcome(kind, attempt, status, result, used)]

    def test_a_divergence_halves_the_learning_rate_that_diverged(self):
        out = tune.decide("rlvr", {"learning_rate": 1e-4}, B, self.history("sft", {"diverged": True}, {"learning_rate": 4e-4}, "failed"))
        self.assertEqual(out["settings"]["learning_rate"], 2e-4)
        self.assertEqual(out["choices"][0]["param"], "learning_rate")
        self.assertIn("diverged", out["choices"][0]["why"])
        self.assertTrue(out["evidence"]["diverged"])

    def test_a_second_divergence_halves_again(self):
        first = tune.decide("sft", {"learning_rate": 1e-4}, B, self.history("sft", {"diverged": True}, {"learning_rate": 1e-4}, "failed"))
        again = tune.decide("sft", {"learning_rate": 1e-4}, B, self.history("sft", {"diverged": True}, first["settings"], "failed", 2))
        self.assertEqual(again["settings"]["learning_rate"], 2.5e-5)

    def test_a_falling_curve_lengthens_the_next_phase_and_a_flat_one_shortens_it(self):
        longer = tune.decide("sft", {"steps": 200}, B, self.history("midtrain", {"curve": falling()}, {"steps": 200}))
        self.assertEqual(longer["settings"]["steps"], 300)
        self.assertIn("still falling", longer["choices"][0]["why"])
        shorter = tune.decide("sft", {"steps": 200}, B, self.history("midtrain", {"curve": flat()}, {"steps": 200}))
        self.assertEqual(shorter["settings"]["steps"], 150)

    def test_a_too_small_plateau_proposes_a_bigger_design_and_changes_no_setting(self):
        out = tune.decide("sft", {"steps": 200}, B, self.history("pretrain", {"too_small": True}, {}))
        self.assertEqual(out["proposal"]["action"], "grow")
        self.assertEqual((out["settings"], out["choices"]), ({"steps": 200}, []))

    def test_dpo_beta_follows_the_gate_that_failed(self):
        rose = {"ok": False, "checks": [{"name": "Held-back text: error did not rise", "ok": False}]}
        out = tune.decide("dpo", {"beta": 0.1}, B, self.history("dpo", rose, {"beta": 0.1}, "done"))
        self.assertAlmostEqual(out["settings"]["beta"], 0.15)
        acc = {"ok": False, "checks": [{"name": "Held-back pairs: preference accuracy above 55%", "ok": False}]}
        out = tune.decide("dpo", {"beta": 0.1}, B, self.history("dpo", acc, {"beta": 0.1}, "done"))
        self.assertAlmostEqual(out["settings"]["beta"], 0.07)

    def test_no_evidence_changes_nothing(self):
        out = tune.decide("sft", {"learning_rate": 1e-4, "steps": 200}, B, [])
        self.assertEqual((out["settings"], out["choices"], out["proposal"]), ({"learning_rate": 1e-4, "steps": 200}, [], None))

    def test_no_choice_leaves_the_bounds(self):
        tight = tune.bounds_of({"bounds": {"learning_rate": {"min": 9e-5, "max": 1e-4}, "steps": {"min": 100, "max": 220}}})
        low = tune.decide("sft", {"learning_rate": 1e-4, "steps": 200}, tight,
                          self.history("sft", {"diverged": True}, {"learning_rate": 1e-4, "steps": 200}, "failed"))
        self.assertEqual(low["settings"]["learning_rate"], 9e-5)
        high = tune.decide("sft", {"steps": 200}, tight, self.history("midtrain", {"curve": falling()}, {"steps": 200}))
        self.assertEqual(high["settings"]["steps"], 220)
        self.assertIn("bound", high["choices"][0]["why"])
        out = tune.decide("sft", {"learning_rate": 5e-3, "steps": 5}, tight, [])      # the plan's own values are held too
        self.assertEqual((out["settings"]["learning_rate"], out["settings"]["steps"]), (1e-4, 100))

    def test_bounds_are_validated(self):
        self.assertEqual(tune.validate_bounds({}), [])
        self.assertEqual(len(tune.validate_bounds({"steps": {"min": 5, "max": 1}})), 1)
        self.assertEqual(len(tune.validate_bounds({"nope": {"min": 1, "max": 2}})), 1)
        self.assertEqual(len(tune.validate_bounds({"beta": {"min": 0, "max": 2}})), 1)


if __name__ == "__main__":
    unittest.main()
