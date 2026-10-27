"""The Auto Run plan and state machine (studio_autorun): validation, every transition and policy, the budget, attempt ids,
reconcile after a restart, and the files. Nothing here touches the queue or a model."""
import copy
import tempfile
import unittest
from pathlib import Path

import studio_autorun as ar

T0 = "2026-01-01T00:00:00+00:00"
T1 = "2026-01-01T01:00:00+00:00"
T2 = "2026-01-01T02:00:00+00:00"


def plan_of(kinds=("data", "pretrain", "sft"), hours=10, **policy):
    plan = ar.DEFAULT_PLAN(3)
    plan["id"] = "run1"
    plan["budget"]["hours"] = hours
    plan["phases"] = [{"kind": k, "enabled": True, "settings": {}, "on_pass": "continue", "on_fail": "pause"} for k in kinds]
    for phase in plan["phases"]:
        phase.update(policy)
    return plan


def running(plan, now=T0):
    return ar.start(plan, ar.new_state(plan), now)


def advance(plan, state, now=T0):
    """Take the next action and begin it; returns (action, state)."""
    action = ar.next_action(plan, state, now)
    return action, ar.begin(plan, state, action, now)


class ValidateTests(unittest.TestCase):
    def test_the_default_plan_is_valid_and_ordered(self):
        plan = ar.DEFAULT_PLAN(1)
        plan["id"] = "x"
        self.assertEqual(ar.validate(plan), [])
        self.assertEqual([p["kind"] for p in plan["phases"]], list(ar.KINDS))

    def test_duplicate_and_out_of_order_kinds_are_plain_sentences(self):
        plan = plan_of(("data", "sft", "pretrain", "sft"))
        problems = ar.validate(plan)
        self.assertIn("The pretrain phase is out of order; the order is " + ", ".join(ar.KINDS) + ".", problems)
        self.assertIn("The sft phase appears more than once.", problems)

    def test_bad_policies_and_fields(self):
        plan = plan_of()
        plan["phases"][0].update(on_pass="maybe", on_fail="explode", enabled="yes")
        plan["phases"][1]["kind"] = "bogus"
        plan["budget"]["hours"] = 0
        plan["id"] = "a/b"
        problems = " ".join(ar.validate(plan))
        for part in ("bad on_pass 'maybe'", "bad on_fail 'explode'", "enabled set to true or false", "unknown kind 'bogus'",
                     "budget hours must be more than 0", "cannot hold slashes"):
            self.assertIn(part, problems)

    def test_nothing_enabled_or_empty_or_not_an_object(self):
        plan = plan_of()
        for phase in plan["phases"]:
            phase["enabled"] = False
        self.assertEqual(ar.validate(plan), ["At least one phase must be enabled."])
        self.assertTrue(ar.validate({**plan_of(), "phases": []}))
        self.assertEqual(ar.validate([]), ["The plan must be an object."])

    def test_no_budget_limit_is_valid(self):
        self.assertEqual(ar.validate(plan_of(hours=None)), [])

    def test_job_kinds_cover_every_phase_but_data(self):
        self.assertEqual(set(ar.JOB_KIND), set(ar.KINDS) - {"data"})


class FlowTests(unittest.TestCase):
    def test_a_draft_waits_and_start_sets_the_deadline(self):
        plan = plan_of()
        state = ar.new_state(plan)
        self.assertEqual((state["status"], ar.next_action(plan, state, T0)), ("draft", ("wait",)))
        state = ar.start(plan, state, T0)
        self.assertEqual(state["status"], "running")
        self.assertEqual(state["deadline"], "2026-01-01T10:00:00+00:00")
        self.assertEqual(state["current"], "data")
        self.assertEqual(ar.remaining_hours(state, T1), 9)
        with self.assertRaises(ar.AutorunError):
            ar.start(plan, state, T0)

    def test_disabled_phases_start_skipped_and_are_passed_over(self):
        plan = plan_of()
        plan["phases"][0]["enabled"] = False
        state = running(plan)
        self.assertEqual(state["phases"][0]["status"], "skipped")
        self.assertEqual(ar.next_action(plan, state, T0), ("enqueue", "pretrain", "run1-1-1"))

    def test_a_clean_run_walks_every_phase_to_done(self):
        plan = plan_of()
        state = running(plan)
        action, state = advance(plan, state)
        self.assertEqual(action, ("data", "data"))
        self.assertEqual(state["phases"][0]["status"], "running")
        self.assertEqual(ar.next_action(plan, state, T0), ("wait",))
        state = ar.apply_outcome(plan, state, "data", "done", {"rows": 5}, T1)
        for kind, jid in (("pretrain", "run1-1-1"), ("sft", "run1-2-1")):
            action, state = advance(plan, state, T1)
            self.assertEqual(action, ("enqueue", kind, jid))
            self.assertEqual(ar.next_action(plan, state, T1), ("wait",))
            state = ar.apply_outcome(plan, state, kind, "done", {"checkpoint": f"{kind}.npz"}, T1)
        self.assertEqual(state["status"], "done")
        self.assertEqual(state["phases"][1]["checkpoint"], "pretrain.npz")
        self.assertEqual(ar.next_action(plan, state, T1)[:2], ("finish", "done"))

    def test_enqueue_is_idempotent_until_begin(self):
        plan = plan_of(("pretrain",))
        state = running(plan)
        self.assertEqual(ar.next_action(plan, state, T0), ar.next_action(plan, state, T1))

    def test_transitions_do_not_change_their_input(self):
        plan = plan_of()
        state = running(plan)
        before = copy.deepcopy(state)
        advance(plan, state)
        ar.apply_outcome(plan, state, "data", "done", {}, T1)
        self.assertEqual(state, before)

    def test_mark_running_moves_queued_to_running(self):
        plan = plan_of(("pretrain",))
        _, state = advance(plan, running(plan))
        self.assertEqual(state["phases"][0]["status"], "queued")
        state = ar.mark_running(plan, state, "pretrain", T1)
        self.assertEqual(state["phases"][0]["status"], "running")

    def test_a_late_report_for_an_ended_phase_changes_nothing(self):
        plan = plan_of(("pretrain", "sft"))
        _, state = advance(plan, running(plan))
        state = ar.apply_outcome(plan, state, "pretrain", "done", {}, T1)
        self.assertIs(ar.apply_outcome(plan, state, "pretrain", "failed", {}, T2), state)

    def test_unknown_outcome_is_refused(self):
        plan = plan_of(("pretrain",))
        _, state = advance(plan, running(plan))
        with self.assertRaises(ar.AutorunError):
            ar.apply_outcome(plan, state, "pretrain", "exploded", {}, T1)


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.plan = plan_of(("pretrain", "sft"))

    def fail_first(self, plan, job_status="failed", result=None):
        _, state = advance(plan, running(plan))
        return ar.apply_outcome(plan, state, "pretrain", job_status, {"error": "out of memory"} if result is None else result, T1)

    def test_on_pass_pause_pauses_after_the_phase(self):
        plan = plan_of(("pretrain", "sft"), on_pass="pause")
        _, state = advance(plan, running(plan))
        state = ar.apply_outcome(plan, state, "pretrain", "done", {}, T1)
        self.assertEqual(state["status"], "paused")
        self.assertIn("paused as planned", state["reason"])
        self.assertEqual(ar.next_action(plan, state, T1), ("wait",))
        state = ar.resume(plan, state, T1)
        self.assertEqual(ar.next_action(plan, state, T1), ("enqueue", "sft", "run1-1-1"))

    def test_on_fail_pause_then_resume_retries_with_the_next_attempt(self):
        state = self.fail_first(self.plan)
        self.assertEqual((state["status"], state["phases"][0]["status"]), ("paused", "failed"))
        self.assertIn("out of memory", state["reason"])
        state = ar.resume(self.plan, state, T1)
        self.assertEqual(state["phases"][0]["attempt"], 2)
        self.assertEqual(ar.next_action(self.plan, state, T1), ("enqueue", "pretrain", "run1-0-2"))
        self.assertEqual(state["phases"][0]["error"], "")

    def test_on_fail_stop_ends_the_run(self):
        plan = plan_of(("pretrain", "sft"), on_fail="stop")
        state = self.fail_first(plan)
        self.assertEqual((state["status"], state["phases"][0]["status"], state["phases"][1]["status"]), ("stopped", "failed", "pending"))
        self.assertEqual(ar.next_action(plan, state, T1)[:2], ("finish", "stopped"))

    def test_on_fail_skip_carries_on(self):
        plan = plan_of(("pretrain", "sft"), on_fail="skip")
        state = self.fail_first(plan)
        self.assertEqual((state["status"], state["phases"][0]["status"]), ("running", "skipped"))
        self.assertEqual(state["phases"][0]["error"], "out of memory")
        self.assertEqual(ar.next_action(plan, state, T1), ("enqueue", "sft", "run1-1-1"))

    def test_skipping_the_last_phase_finishes_done(self):
        plan = plan_of(("pretrain",), on_fail="skip")
        self.assertEqual(self.fail_first(plan)["status"], "done")

    def test_a_gate_that_said_no_is_a_failed_phase(self):
        plan = plan_of(("sft",), on_fail="stop")
        _, state = advance(plan, running(plan))
        state = ar.apply_outcome(plan, state, "sft", "done", {"ok": False, "checkpoint": "c.npz", "verdict": "Held-back checks failed."}, T1)
        self.assertEqual((state["status"], state["phases"][0]["status"]), ("stopped", "failed"))
        self.assertIn("Held-back checks failed.", state["reason"])
        self.assertEqual(state["phases"][0]["checkpoint"], "c.npz")

    def test_passed_false_wins_over_ok_true(self):
        plan = plan_of(("sft",))
        _, state = advance(plan, running(plan))
        state = ar.apply_outcome(plan, state, "sft", "done", {"passed": False, "ok": True}, T1)
        self.assertEqual(state["phases"][0]["status"], "failed")

    def test_a_stage_that_stopped_before_its_gate_is_stopped_not_failed(self):
        plan = plan_of(("sft",))
        _, state = advance(plan, running(plan))
        state = ar.apply_outcome(plan, state, "sft", "done", {"ok": False, "stopped": True}, T1)
        self.assertEqual((state["status"], state["phases"][0]["status"]), ("paused", "stopped"))

    def test_a_stopped_job_pauses_and_resume_runs_it_again(self):
        state = self.fail_first(self.plan, "stopped", {})
        self.assertEqual((state["status"], state["phases"][0]["status"]), ("paused", "stopped"))
        state = ar.resume(self.plan, state, T1)
        self.assertEqual(ar.next_action(self.plan, state, T1), ("enqueue", "pretrain", "run1-0-2"))

    def test_a_deleted_job_applies_on_fail(self):
        for policy, expect in (("pause", "paused"), ("stop", "stopped"), ("skip", "running")):
            plan = plan_of(("pretrain", "sft"), on_fail=policy)
            state = self.fail_first(plan, "missing", {})
            self.assertEqual(state["status"], expect, policy)
            self.assertIn("removed", state["phases"][0]["error"])

    def test_a_blocked_phase_waits_for_the_user_and_resume_keeps_the_attempt(self):
        plan = plan_of(("data", "pretrain"))
        _, state = advance(plan, running(plan))
        state = ar.apply_outcome(plan, state, "data", "blocked", {"reason": "Download the starter text first."}, T1)
        self.assertEqual((state["status"], state["reason"]), ("waiting_user", "Download the starter text first."))
        self.assertEqual(state["phases"][0]["status"], "blocked")
        self.assertEqual(ar.next_action(plan, state, T1), ("wait",))
        state = ar.resume(plan, state, T1)
        self.assertEqual((state["phases"][0]["attempt"], ar.next_action(plan, state, T1)), (1, ("data", "data")))

    def test_skip_phase_after_a_failure_lets_resume_carry_on(self):
        state = self.fail_first(self.plan)
        state = ar.skip_phase(self.plan, state, "pretrain", T1)
        state = ar.resume(self.plan, state, T1)
        self.assertEqual(ar.next_action(self.plan, state, T1), ("enqueue", "sft", "run1-1-1"))


class PauseStopTests(unittest.TestCase):
    def test_pause_when_idle_is_immediate(self):
        plan = plan_of()
        state = ar.request_pause(running(plan), T1)
        self.assertEqual(state["status"], "paused")

    def test_pausing_lets_the_running_phase_finish_then_pauses(self):
        plan = plan_of(("pretrain", "sft"))
        _, state = advance(plan, running(plan))
        state = ar.request_pause(state, T1)
        self.assertEqual(state["status"], "pausing")
        self.assertEqual(ar.next_action(plan, state, T1), ("wait",))
        done = ar.apply_outcome(plan, state, "pretrain", "done", {}, T1)
        self.assertEqual(done["status"], "paused")
        self.assertEqual(ar.next_action(plan, done, T1), ("wait",))

    def test_pausing_with_a_skipped_failure_pauses_and_a_stop_policy_stops(self):
        plan = plan_of(("pretrain", "sft"), on_fail="skip")
        _, state = advance(plan, running(plan))
        state = ar.apply_outcome(plan, ar.request_pause(state, T1), "pretrain", "failed", {}, T1)
        self.assertEqual(state["status"], "paused")
        plan = plan_of(("pretrain", "sft"), on_fail="stop")
        _, state = advance(plan, running(plan))
        state = ar.apply_outcome(plan, ar.request_pause(state, T1), "pretrain", "failed", {}, T1)
        self.assertEqual(state["status"], "stopped")

    def test_settle_pause_turns_an_idle_pausing_into_paused(self):
        plan = plan_of(("pretrain", "sft"))
        _, state = advance(plan, running(plan))
        state = ar.request_pause(state, T1)
        state["phases"][0]["status"] = "pending"  # the job vanished before the pause settled
        self.assertEqual(ar.next_action(plan, state, T1)[0], "pause")
        self.assertEqual(ar.settle_pause(state, T1)["status"], "paused")

    def test_only_a_running_run_pauses_and_only_a_paused_run_resumes(self):
        plan = plan_of()
        with self.assertRaises(ar.AutorunError):
            ar.request_pause(ar.new_state(plan), T0)
        with self.assertRaises(ar.AutorunError):
            ar.resume(plan, running(plan), T0)

    def test_stop_marks_live_phases_stopped_and_finishes(self):
        plan = plan_of(("pretrain", "sft"))
        _, state = advance(plan, running(plan))
        state = ar.stop(state, "", T1)
        self.assertEqual((state["status"], state["phases"][0]["status"], state["phases"][1]["status"]), ("stopped", "stopped", "pending"))
        self.assertEqual(state["reason"], "Stopped by you.")
        self.assertEqual(ar.next_action(plan, state, T1), ("finish", "stopped", "Stopped by you."))


class BudgetTests(unittest.TestCase):
    def test_no_phase_starts_past_the_deadline(self):
        plan = plan_of(("data", "pretrain"), hours=1)
        state = running(plan)
        _, state = advance(plan, state)
        state = ar.apply_outcome(plan, state, "data", "done", {}, T2)  # two hours later
        self.assertEqual(ar.next_action(plan, state, T2), ("finish", "stopped", "The time budget is spent."))
        applied = ar.finish(state, "stopped", "The time budget is spent.", T2)
        self.assertEqual((applied["status"], applied["reason"]), ("stopped", "The time budget is spent."))

    def test_a_running_phase_is_not_cut_off_by_the_deadline(self):
        plan = plan_of(("pretrain",), hours=1)
        _, state = advance(plan, running(plan))
        self.assertEqual(ar.next_action(plan, state, T2), ("wait",))

    def test_pretrain_hours_are_capped_to_what_remains(self):
        plan = plan_of(("data", "pretrain"), hours=4)
        plan["phases"][1]["settings"] = {"hours": 8, "width": 64}
        state = running(plan)
        self.assertEqual(ar.phase_settings(plan, state, "pretrain", T1), {"hours": 3, "width": 64})
        plan["phases"][1]["settings"]["hours"] = 1
        self.assertEqual(ar.phase_settings(plan, state, "pretrain", T1)["hours"], 1)
        plan["phases"][1]["settings"].pop("hours")
        self.assertEqual(ar.phase_settings(plan, state, "pretrain", T1)["hours"], 3)

    def test_other_phases_and_unlimited_budgets_are_left_alone(self):
        plan = plan_of(("pretrain", "sft"), hours=None)
        plan["phases"][0]["settings"] = {"hours": 8}
        plan["phases"][1]["settings"] = {"hours": 8}
        state = running(plan)
        self.assertEqual(ar.phase_settings(plan, state, "pretrain", T1)["hours"], 8)
        plan["budget"]["hours"] = 2
        state = running(plan)
        self.assertEqual(ar.phase_settings(plan, state, "sft", T1)["hours"], 8)
        self.assertIsNone(ar.remaining_hours(running(plan_of(hours=None)), T1))

    def test_the_settings_are_a_copy(self):
        plan = plan_of(("pretrain",), hours=None)
        plan["phases"][0]["settings"] = {"a": [1]}
        ar.phase_settings(plan, running(plan), "pretrain", T0)["a"].append(2)
        self.assertEqual(plan["phases"][0]["settings"], {"a": [1]})


class AttemptTests(unittest.TestCase):
    def test_job_ids_follow_index_and_attempt(self):
        plan = plan_of(("data", "pretrain", "sft"))
        self.assertEqual(ar.job_id({"id": "run1"}, 2, 3), "run1-2-3")
        state = running(plan)
        state["phases"][0]["status"] = "passed"
        self.assertEqual(ar.next_action(plan, state, T0)[2], "run1-1-1")

    def test_every_retry_gets_a_new_id(self):
        plan = plan_of(("pretrain",))
        state = running(plan)
        ids = []
        for _ in range(3):
            action, state = advance(plan, state)
            ids.append(action[2])
            state = ar.apply_outcome(plan, state, "pretrain", "failed", {}, T1)
            state = ar.resume(plan, state, T1)
        self.assertEqual(ids, ["run1-0-1", "run1-0-2", "run1-0-3"])


class ReconcileTests(unittest.TestCase):
    def setUp(self):
        self.plan = plan_of(("data", "pretrain", "sft"))
        state = running(self.plan)
        _, state = advance(self.plan, state)
        state = ar.apply_outcome(self.plan, state, "data", "done", {}, T0)
        _, self.state = advance(self.plan, state)

    def lookup(self, table):
        return lambda jid: table.get(jid)

    def test_live_jobs_wait_in_any_live_status(self):
        for status in ("waiting", "held", "running"):
            actions = ar.reconcile(self.state, self.lookup({"run1-1-1": {"status": status}}), self.lookup({}))
            self.assertEqual(actions, [("wait", "pretrain")], status)

    def test_a_finished_job_in_history_becomes_an_outcome(self):
        history = self.lookup({"run1-1-1": {"status": "done", "result": {"checkpoint": "a.npz"}}})
        self.assertEqual(ar.reconcile(self.state, self.lookup({}), history), [("outcome", "pretrain", "done", {"checkpoint": "a.npz"})])

    def test_a_finished_job_in_the_queue_and_its_error(self):
        queue = self.lookup({"run1-1-1": {"status": "failed", "error": "boom"}})
        self.assertEqual(ar.reconcile(self.state, queue, self.lookup({})), [("outcome", "pretrain", "failed", {"error": "boom"})])

    def test_objects_work_as_well_as_dicts(self):
        class Job:
            status, result, error = "stopped", None, ""
        self.assertEqual(ar.reconcile(self.state, self.lookup({"run1-1-1": Job()}), self.lookup({})),
                         [("outcome", "pretrain", "stopped", {})])

    def test_a_missing_job_is_enqueued_again_with_the_same_id(self):
        self.assertEqual(ar.reconcile(self.state, self.lookup({}), self.lookup({})), [("enqueue", "pretrain", "run1-1-1")])

    def test_a_data_phase_that_was_running_runs_again(self):
        plan = plan_of(("data",))
        _, state = advance(plan, running(plan))
        self.assertEqual(ar.reconcile(state, self.lookup({}), self.lookup({})), [("data", "data")])

    def test_phases_that_are_not_in_flight_and_runs_that_are_not_running_give_nothing(self):
        self.assertEqual(ar.reconcile(ar.new_state(self.plan), self.lookup({}), self.lookup({})), [])
        paused = ar.request_pause(self.state, T1)
        paused = ar.settle_pause(paused, T1) if paused["status"] == "pausing" else paused
        paused["status"] = "paused"
        self.assertEqual(ar.reconcile(paused, self.lookup({}), self.lookup({})), [])

    def test_a_pausing_run_is_reconciled_too(self):
        pausing = ar.request_pause(self.state, T1)
        self.assertEqual(pausing["status"], "pausing")
        self.assertEqual(ar.reconcile(pausing, self.lookup({}), self.lookup({})), [("enqueue", "pretrain", "run1-1-1")])

    def test_the_outcome_feeds_apply_outcome(self):
        history = self.lookup({"run1-1-1": {"status": "done", "result": {"ok": True, "checkpoint": "a.npz"}}})
        (_, kind, status, result), = ar.reconcile(self.state, self.lookup({}), history)
        state = ar.apply_outcome(self.plan, self.state, kind, status, result, T1)
        self.assertEqual(state["phases"][1]["status"], "passed")


class FilesTests(unittest.TestCase):
    def test_plan_and_state_round_trip_and_the_journal_has_a_line_per_transition(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "run1"
            plan = plan_of(("pretrain",))
            ar.save_plan(directory, plan)
            self.assertEqual(ar.load_plan(directory), plan)
            state = running(plan)
            ar.save_state(directory, state)
            self.assertEqual(state["events"], [])
            action, state = advance(plan, state)
            state = ar.apply_outcome(plan, state, "pretrain", "done", {}, T1)
            ar.save_state(directory, state)
            loaded = ar.load_state(directory)
            self.assertEqual((loaded["status"], loaded["events"]), ("done", []))
            events = [row["event"] for row in ar.read_journal(directory)]
            self.assertEqual(events, ["start", "begin", "passed", "done"])
            self.assertEqual(ar.read_journal(directory)[-1]["status"], "done")

    def test_a_missing_or_broken_file_loads_as_none_and_a_cut_journal_line_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(ar.load_plan(tmp))
            self.assertIsNone(ar.load_state(tmp))
            self.assertEqual(ar.read_journal(tmp), [])
            ar.paths(tmp)["state"].write_text("{broken", encoding="utf-8")
            ar.paths(tmp)["journal"].write_text('{"event": "start"}\n{"ev', encoding="utf-8")
            self.assertIsNone(ar.load_state(tmp))
            self.assertEqual(ar.read_journal(tmp), [{"event": "start"}])

    def test_an_invalid_plan_is_not_saved(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ar.AutorunError):
                ar.save_plan(tmp, {"id": ""})
            self.assertIsNone(ar.load_plan(tmp))


if __name__ == "__main__":
    unittest.main()
