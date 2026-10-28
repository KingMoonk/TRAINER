"""The Auto Run parameter controller (studio_autotune): scripted phase results in, choices out. Pure; nothing trains."""
import copy
import unittest

import numpy as np

import studio_autorun as ar
import studio_autotune as tune

T0 = "2026-01-01T00:00:00+00:00"
FALLING = [(float(x), .1 + 5 * x ** -.2) for x in np.geomspace(1e3, 1e6, 14)]      # still dropping ~10% per doubling at the end
FLAT = [(float(x), 3 + 500 * x ** -1.) for x in np.geomspace(1e2, 1e5, 14)]        # flat from the first few points on


def plan_of(kinds=("pretrain", "midtrain", "sft", "rlvr", "dpo"), **settings):
    plan = ar.DEFAULT_PLAN(1)
    plan["id"] = "run1"
    plan["phases"] = [{"kind": k, "enabled": True, "settings": dict(settings.get(k, {})), "on_pass": "continue", "on_fail": "pause"} for k in kinds]
    return plan


def retry(plan, state, phase):
    """The state after `phase` failed once and was resumed: attempt 2, the phase's own result gone (as ar.resume leaves it)."""
    state = copy.deepcopy(state)
    entry = state["phases"][ar._index(plan, phase)]
    entry.update(status="pending", attempt=entry["attempt"] + 1, error="", result=None)
    return state


def failed(plan, phase, result, state=None):
    state = tune.record_attempt(plan, state or ar.new_state(plan), phase, "failed", result, T0)
    return retry(plan, state, phase)


def passed_pretrain(plan):
    state = ar.new_state(plan)
    state["phases"][ar._index(plan, "pretrain")]["status"] = "passed"
    return state


class ControllerTests(unittest.TestCase):
    def test_a_divergence_lowers_the_learning_rate_and_says_why(self):
        plan = plan_of(sft={"learning_rate": 1e-4})
        state = failed(plan, "sft", {"error": "Non-finite loss. Reduce the learning rate."})
        [choice] = tune.decide(plan, state, "sft")
        self.assertEqual((choice["key"], choice["was"], choice["value"], choice["rule"]), ("learning_rate", 1e-4, 5e-5, "divergence"))
        self.assertIn("not a number", choice["why"])
        self.assertIn("Non-finite", choice["evidence"]["error"])
        self.assertEqual(tune.settings_with({"steps": 5}, [choice]), {"steps": 5, "learning_rate": 5e-5})

    def test_a_round_without_a_rate_in_the_plan_starts_from_its_own_default(self):
        plan = plan_of()
        state = failed(plan, "rlvr", {"error": "Non-finite loss. Reduce the learning rate."})
        self.assertEqual(tune.decide(plan, state, "rlvr")[0]["value"], tune.LR_START["rlvr"] * tune.LR_CUT)

    def test_an_ordinary_failure_or_a_first_attempt_changes_no_rate(self):
        plan = plan_of()
        self.assertEqual(tune.decide(plan, ar.new_state(plan), "sft"), [])
        self.assertEqual(tune.decide(plan, failed(plan, "sft", {"error": "Out of disk."}), "sft"), [])

    def test_each_further_divergence_halves_again(self):
        plan = plan_of(sft={"learning_rate": 1e-4})
        state = failed(plan, "sft", {"error": "Non-finite loss."})
        state = tune.add_choices(state, tune.decide(plan, state, "sft"), T0)
        state = failed(plan, "sft", {"error": "Non-finite loss."}, state)
        [choice] = tune.decide(plan, state, "sft")
        self.assertEqual((choice["attempt"], choice["was"], choice["value"]), (3, 5e-5, 2.5e-5))

    def test_a_phase_whose_curve_still_falls_is_lengthened(self):
        plan = plan_of(midtrain={"steps": 1000})
        [choice] = tune.decide(plan, passed_pretrain(plan), "midtrain", FALLING)
        self.assertEqual((choice["key"], choice["was"], choice["value"], choice["rule"]), ("steps", 1000, 1500, "still_falling"))
        self.assertTrue(choice["evidence"]["falling"])
        self.assertIn("still falls", choice["why"])

    def test_midtrain_with_no_steps_in_the_plan_starts_from_the_stage_default(self):
        plan = plan_of()
        [choice] = tune.decide(plan, passed_pretrain(plan), "midtrain", FALLING, width=256)
        self.assertEqual((choice["was"], choice["value"]), (2000, 3000))
        self.assertEqual(tune.decide(plan, passed_pretrain(plan), "midtrain", FALLING), [])      # width unknown: no base to grow

    def test_a_flat_curve_is_not_lengthened_but_proposes_a_bigger_design(self):
        plan = plan_of(midtrain={"steps": 1000})
        choices = tune.decide(plan, passed_pretrain(plan), "midtrain", FLAT)
        [proposal] = choices
        self.assertEqual((proposal["key"], proposal["rule"], proposal["applied"], proposal["value"]), ("design", "bigger_design", False, None))
        self.assertIn(tune.studio_stages.TOO_SMALL, proposal["why"])
        self.assertLess(proposal["evidence"]["flat_share"], tune.TOO_SMALL_SHARE)
        self.assertEqual(tune.settings_with({"steps": 1000}, choices), {"steps": 1000})          # a proposal changes no setting

    def test_the_proposal_is_made_once(self):
        plan = plan_of()
        state = passed_pretrain(plan)
        state = tune.add_choices(state, tune.decide(plan, state, "midtrain", FLAT), T0)
        self.assertEqual(tune.decide(plan, state, "sft", FLAT), [])

    def test_no_curve_or_an_unfinished_pretrain_means_no_choice(self):
        plan = plan_of(midtrain={"steps": 1000})
        self.assertEqual(tune.decide(plan, passed_pretrain(plan), "midtrain", None), [])
        self.assertEqual(tune.decide(plan, passed_pretrain(plan), "midtrain", FALLING[:3]), [])
        self.assertEqual(tune.decide(plan, ar.new_state(plan), "midtrain", FALLING), [])

    def test_a_failed_gate_raises_the_dpo_beta_and_the_kl(self):
        plan = plan_of()
        gate = {"ok": False, "verdict": "It got worse.", "mean_kl": .3}
        [beta] = tune.decide(plan, failed(plan, "dpo", gate), "dpo")
        self.assertEqual((beta["key"], beta["was"], beta["rule"]), ("beta", tune.BETA_START, "gate_history"))
        self.assertAlmostEqual(beta["value"], tune.BETA_START * tune.PULL)
        self.assertEqual(beta["evidence"]["verdict"], "It got worse.")
        [kl] = tune.decide(plan, failed(plan, "rlvr", gate), "rlvr")
        self.assertEqual((kl["key"], kl["evidence"]["mean_kl"]), ("kl", .3))
        self.assertGreater(kl["value"], kl["was"])

    def test_a_gate_that_passed_changes_nothing(self):
        plan = plan_of()
        state = tune.record_attempt(plan, ar.new_state(plan), "dpo", "done", {"ok": True, "verdict": "Better."}, T0)
        self.assertEqual(tune.decide(plan, retry(plan, state, "dpo"), "dpo"), [])

    def test_no_choice_leaves_the_bounds(self):
        plan = plan_of(sft={"learning_rate": 1e-4}, midtrain={"steps": 1000}, dpo={"beta": .4})
        plan["controller"]["bounds"].update(learning_rate={"min": 8e-5, "max": 1e-3}, steps={"min": 10, "max": 1200}, beta={"min": .02, "max": .5})
        [lr] = tune.decide(plan, failed(plan, "sft", {"error": "Non-finite loss."}), "sft")
        self.assertEqual((lr["value"], lr["clamped"]), (8e-5, True))
        self.assertIn("limit", lr["why"])
        [steps] = tune.decide(plan, passed_pretrain(plan), "midtrain", FALLING)
        self.assertEqual((steps["value"], steps["clamped"]), (1200, True))
        [beta] = tune.decide(plan, failed(plan, "dpo", {"ok": False}), "dpo")
        self.assertEqual((beta["value"], beta["clamped"]), (.5, True))
        for choice in (lr, steps, beta):
            bounds = plan["controller"]["bounds"][choice["key"]]
            self.assertTrue(bounds["min"] <= choice["value"] <= bounds["max"])

    def test_a_disabled_controller_or_a_plan_without_one_chooses_nothing(self):
        plan = plan_of(midtrain={"steps": 1000})
        state = passed_pretrain(plan)
        plan["controller"]["enabled"] = False
        self.assertEqual(tune.decide(plan, state, "midtrain", FALLING), [])
        del plan["controller"]
        self.assertEqual(tune.decide(plan, state, "midtrain", FALLING), [])
        self.assertEqual(tune.record_attempt(plan, state, "sft", "failed", {"error": "x"}, T0), state)

    def test_asking_again_for_the_same_attempt_returns_the_same_choice(self):
        plan = plan_of(sft={"learning_rate": 1e-4})
        state = failed(plan, "sft", {"error": "Non-finite loss."})
        first = tune.decide(plan, state, "sft")
        state = tune.add_choices(state, first, T0)
        self.assertEqual(tune.decide(plan, state, "sft"), first)
        self.assertEqual(tune.add_choices(state, first, T0)["choices"], state["choices"])

    def test_every_choice_is_journalled_with_its_evidence(self):
        plan = plan_of(sft={"learning_rate": 1e-4})
        state = failed(plan, "sft", {"error": "Non-finite loss."})
        state = tune.add_choices(state, tune.decide(plan, state, "sft"), T0)
        [event] = [e for e in state["events"] if e["event"] == "tuned"]
        self.assertEqual((event["phase"], event["key"], event["was"], event["value"], event["applied"]), ("sft", "learning_rate", 1e-4, 5e-5, True))
        self.assertIn("Non-finite", event["evidence"]["error"])


class ControllerPlanTests(unittest.TestCase):
    def test_the_default_plan_carries_the_default_bounds_and_is_valid(self):
        plan = ar.DEFAULT_PLAN(1)
        plan["id"] = "x"
        self.assertEqual(plan["controller"], {"enabled": True, "bounds": tune.DEFAULT_BOUNDS})
        self.assertEqual(ar.validate(plan), [])

    def test_bad_bounds_are_plain_sentences(self):
        plan = plan_of()
        plan["controller"]["bounds"]["steps"] = {"min": 50, "max": 10}
        plan["controller"]["bounds"]["beta"] = {"min": "a", "max": 1}
        del plan["controller"]["bounds"]["kl"]
        problems = " ".join(ar.validate(plan))
        for part in ("steps bounds need", "beta bounds must be numbers", "bounds for kl pull-back"):
            self.assertIn(part, problems)
        plan["controller"] = {"enabled": "yes", "bounds": {}}
        self.assertIn("enabled set to true or false", " ".join(ar.validate(plan)))

    def test_a_plan_from_before_the_controller_is_still_valid(self):
        plan = plan_of()
        del plan["controller"]
        self.assertEqual(ar.validate(plan), [])


if __name__ == "__main__":
    unittest.main()
