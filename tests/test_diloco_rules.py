"""studio_diloco_rules: the stop rules, the plateau and pivot rules and the Settings checks, as pure functions on an injected clock."""
import unittest

import studio_diloco_rules as rules
from tests import diloco_support as support


def settings(**changes):
    return support.settings(rounds=40, inner_steps=3, **changes)


def play(s, losses, sources=1):
    """Feed the val_losses to a fresh State one round each; returns (state, verdicts), stopping after a stop."""
    state, verdicts = rules.State(), []
    for number, loss in enumerate(losses):
        verdict = rules.observe(state, s, number, (number + 1) * s.inner_steps, loss, sources)
        verdicts.append(verdict)
        if verdict["action"] == "stop":
            break
    return state, verdicts


class Clock:
    def __init__(self):
        self.now = 0.

    def __call__(self):
        return self.now


class StopRules(unittest.TestCase):
    def test_no_rule_set_never_stops_however_the_loss_goes(self):
        state, verdicts = play(settings(), [5., 6., 7., 8., 9., 10.] * 5)
        self.assertEqual({v["action"] for v in verdicts}, {"continue"})
        self.assertEqual((state.stop_rule, state.worse_streak, state.lr_scale, state.pivot_index), ("", 29, 1., 0))

    def test_patience_stops_after_that_many_rounds_without_a_better_loss(self):
        state, verdicts = play(settings(patience_rounds=3), [5., 4., 4.5, 4.6, 4.7, 3.])
        self.assertEqual(len(verdicts), 5)
        self.assertEqual((verdicts[-1]["action"], verdicts[-1]["rule"]), ("stop", "patience"))
        self.assertIn("3 rounds in a row without a better validation loss", verdicts[-1]["text"])
        self.assertEqual((state.best_round, state.best_loss, state.stop_rule), (1, 4., "patience"))

    def test_a_better_loss_resets_the_patience(self):
        _, verdicts = play(settings(patience_rounds=3), [5., 5.1, 5.2, 4., 4.1, 4.2, 3.])
        self.assertEqual({v["action"] for v in verdicts}, {"continue"})

    def test_target_val_loss_stops_at_the_first_round_at_or_under_it(self):
        state, verdicts = play(settings(target_val_loss=3.5), [5., 4., 3.5, 3.])
        self.assertEqual(len(verdicts), 3)
        self.assertEqual((verdicts[-1]["rule"], state.best_loss), ("target_val_loss", 3.5))
        self.assertIn("target 3.5", verdicts[-1]["text"])

    def test_a_missing_loss_before_the_first_finite_one_is_not_a_worse_round(self):
        s = settings(worse_limit=1, pivots=("halve_lr",))
        state, verdicts = play(s, [None, float("nan"), None, float("nan")])
        self.assertEqual({v["action"] for v in verdicts}, {"continue"})
        self.assertEqual((state.best_round, state.best_loss, state.rounds_since_best, state.worse_streak, state.lr_scale), (None, None, 0, 0, 1.))
        s = settings(patience_rounds=1)
        self.assertEqual({v["action"] for v in play(s, [None, None, None])[1]}, {"continue"})
        state, verdicts = play(s, [None, 5., None])  # once there is a best, a missing loss is a worse round
        self.assertEqual([v["action"] for v in verdicts], ["continue", "continue", "stop"])
        self.assertEqual(verdicts[-1]["rule"], "patience")

    def test_the_last_round_only_updates_the_best(self):
        s = settings(worse_limit=1, pivots=("halve_lr",))
        state = rules.State()
        rules.observe(state, s, 0, 3, 5.)
        verdict = rules.observe(state, s, 1, 6, 6., final=True)
        self.assertEqual((verdict["action"], state.stop_rule, state.lr_scale, state.pivot_index), ("continue", "", 1., 0))
        self.assertTrue(rules.observe(state, s, 2, 9, 4., final=True)["improved"])
        self.assertEqual(state.best_round, 2)

    def test_max_seconds_counts_only_the_time_with_enough_members(self):
        s, state, clock = settings(max_seconds=100.), rules.State(), Clock()
        rules.accrue(state, clock(), True)
        clock.now = 60.
        rules.accrue(state, clock(), False)  # 60 s with members: counted; the stretch that starts now has too few
        clock.now = 500.
        rules.accrue(state, clock(), True)   # 440 s below min_members: not counted
        self.assertEqual(state.active_seconds, 60.)
        self.assertIsNone(rules.stop_reason(state, s))
        clock.now = 540.
        rules.accrue(state, clock(), True)
        self.assertEqual(state.active_seconds, 100.)
        self.assertEqual(rules.stop_reason(state, s)[0], "max_seconds")
        verdict = rules.observe(state, s, 0, 3, 5.)
        self.assertEqual((verdict["action"], verdict["rule"]), ("stop", "max_seconds"))

    def test_the_first_rule_that_holds_is_the_one_named(self):
        state, verdicts = play(settings(patience_rounds=2, target_val_loss=10.), [5., 6., 7.])
        self.assertEqual(verdicts[0]["rule"], "target_val_loss")
        self.assertEqual(len(verdicts), 1)


class Pivots(unittest.TestCase):
    def test_a_worsening_loss_goes_back_to_the_best_and_applies_the_pivots_in_order_then_ends(self):
        s = settings(worse_limit=2, pivots=("halve_lr", "next_dataset"))
        state, verdicts = play(s, [5., 4., 4.5, 4.6, 4.4, 4.3, 4.2, 4.1], sources=2)
        names = [(v["action"], v.get("name")) for v in verdicts]
        self.assertEqual(names[:4], [("continue", None), ("continue", None), ("continue", None), ("pivot", "halve_lr")])
        self.assertEqual((verdicts[3]["before"], verdicts[3]["best"], verdicts[3]["best_round"], verdicts[3]["why"]), (4.6, 4., 1, "worse"))
        self.assertEqual(names[4:6], [("continue", None), ("pivot", "next_dataset")])  # 4.4 and 4.3 are worse than the best 4.0
        self.assertEqual((verdicts[-1]["action"], verdicts[-1]["rule"]), ("stop", "pivots"))
        self.assertEqual((state.lr_scale, state.source, state.pivot_index, state.stop_rule), (.5, 1, 2, "pivots"))
        self.assertIn("no change left to try", state.stop_text)

    def test_a_pivot_starts_a_new_stage_with_clean_streaks(self):
        s = settings(worse_limit=2, pivots=("halve_lr",))
        state = rules.State()
        for number, loss in enumerate([5., 6., 7.]):
            verdict = rules.observe(state, s, number, (number + 1) * 3, loss)
        self.assertEqual(verdict["action"], "pivot")
        self.assertEqual((state.worse_streak, state.rounds_since_best, state.points, state.best_loss), (0, 0, [], 5.))

    def test_next_dataset_without_a_later_source_is_skipped(self):
        s = settings(worse_limit=1, pivots=("next_dataset", "halve_lr"))
        state = rules.State()
        self.assertEqual(rules.next_pivot(state, s, sources=1), (1, "halve_lr"))
        self.assertEqual(rules.next_pivot(state, s, sources=2), (0, "next_dataset"))
        state.pivot_index = 2
        self.assertIsNone(rules.next_pivot(state, s))

    def test_a_plateau_triggers_the_pivot_and_without_pivots_it_stops(self):
        flat = [5., 4., 3.995, 3.99, 3.99, 3.99, 3.99]
        s = settings(plateau_checks=3, plateau_min_improvement=.05, pivots=("halve_lr",))
        state, verdicts = play(s, flat)
        pivots = [v for v in verdicts if v["action"] == "pivot"]
        self.assertEqual((len(pivots), pivots[0]["why"], state.lr_scale), (1, "plateau", .5))
        state, verdicts = play(settings(plateau_checks=3, plateau_min_improvement=.05), flat)
        self.assertEqual((verdicts[-1]["action"], verdicts[-1]["rule"]), ("stop", "plateau"))

    def test_a_loss_that_keeps_improving_is_never_a_plateau(self):
        _, verdicts = play(settings(plateau_checks=3, plateau_min_improvement=.05, pivots=("halve_lr",)), [10., 8., 6., 4., 3., 2.])
        self.assertEqual({v["action"] for v in verdicts}, {"continue"})

    def test_a_nonfinite_loss_counts_as_not_better(self):
        state, verdicts = play(settings(patience_rounds=2), [5., float("nan"), float("inf")])
        self.assertEqual((verdicts[-1]["rule"], state.best_loss, state.points), ("patience", 5., [[3, 5.]]))


class State(unittest.TestCase):
    def test_the_state_round_trips_through_json_without_the_clock_fields(self):
        s = settings(worse_limit=1, pivots=("halve_lr",))
        state = rules.State()
        rules.accrue(state, 1., True)
        rules.accrue(state, 4., True)
        rules.observe(state, s, 0, 3, 5.)
        rules.observe(state, s, 1, 6, 6.)
        saved = state.to_json()
        self.assertNotIn("tick", saved)
        self.assertEqual(rules.State.from_json(saved).to_json(), saved)
        self.assertEqual((saved["active_seconds"], saved["lr_scale"], saved["pivot_index"]), (3., .5, 1))

    def test_unknown_and_missing_keys_are_tolerated(self):
        self.assertEqual(rules.State.from_json(None).to_json(), rules.State().to_json())
        self.assertEqual(rules.State.from_json({"best_round": 2, "from_the_future": 1}).best_round, 2)


class Checks(unittest.TestCase):
    def rejects(self, **changes):
        with self.assertRaises(ValueError, msg=str(changes)):
            support.settings(**changes)

    def test_bad_values_are_rejected_with_a_sentence(self):
        for changes in ({"max_seconds": 0}, {"max_seconds": -5.}, {"max_seconds": float("nan")}, {"patience_rounds": -1}, {"patience_rounds": 1.5},
                        {"target_val_loss": 0}, {"target_val_loss": -1.}, {"target_win_rate": .5, "teacher": "t"},
                        {"target_win_rate": .5}, {"pivots": ("drop_lessons",), "worse_limit": 1}, {"plateau_checks": 1}, {"plateau_checks": -2}, {"plateau_min_improvement": 0.},
                        {"plateau_min_improvement": 1.}, {"worse_limit": -1}, {"keep_checkpoints": -1}, {"pivots": ("warm_restart",), "worse_limit": 1},
                        {"pivots": ("halve_lr",)}, {"worse_limit": 3, "pivots": ("halve_lr",), "patience_rounds": 3}, {"teacher": 5}, {"max_seconds": True}):
            self.rejects(**changes)

    def test_the_message_names_the_option(self):
        with self.assertRaisesRegex(ValueError, "worse_limit"):
            support.settings(worse_limit=-1)
        with self.assertRaisesRegex(ValueError, "Each pivot must be one of"):
            support.settings(pivots=("nope",), worse_limit=1)

    def test_target_win_rate_and_drop_lessons_say_they_arrive_with_508(self):
        with self.assertRaisesRegex(ValueError, "target_win_rate.*#508"):
            support.settings(target_win_rate=.6, teacher="big")
        with self.assertRaisesRegex(ValueError, "drop_lessons.*#508"):
            support.settings(pivots=("drop_lessons",), worse_limit=1)

    def test_good_values_are_accepted_and_the_defaults_set_nothing(self):
        s = support.settings(max_seconds=60, patience_rounds=3, target_val_loss=2.5, plateau_checks=3,
                             plateau_min_improvement=.02, worse_limit=2, pivots=["halve_lr", "next_dataset"], keep_checkpoints=2)
        self.assertEqual(s.pivots, ("halve_lr", "next_dataset"))  # a list from JSON becomes a tuple
        plain = support.settings()
        self.assertEqual((plain.max_seconds, plain.patience_rounds, plain.worse_limit, plain.pivots, plain.keep_checkpoints), (None, 0, 0, (), 0))
        self.assertIsNone(rules.stop_reason(rules.State(), plain))

    def test_settings_come_from_job_params_as_plain_json(self):
        import studio_diloco
        s = studio_diloco.Settings.from_params({"inner_steps": 4, "rounds": 6, "patience_rounds": 2, "pivots": ["halve_lr"], "worse_limit": 1})
        self.assertEqual((s.patience_rounds, s.pivots), (2, ("halve_lr",)))


if __name__ == "__main__":
    unittest.main()
