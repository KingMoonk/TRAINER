"""The Auto Run orchestrator (studio_ia.autorun) over FakeServices' queue: enqueue on start, a done event leading to the next phase,
a failed gate per policy, pause/stop/skip/continue, the pretrain phase waiting for its restore_answers round, the budget, and a
restart in the middle of a phase. No runner, no worker, no window: tests end a job by hand with FakeServices.finish_job."""
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy

import studio_autorun as ar
from studio_ia.autorun import Orchestrator
from studio_ia.services import FakeServices, Job, Version

GOOD = {"ok": True, "checkpoint": "ck-good", "checks": {}}
BAD = {"ok": False, "checkpoint": "ck-kept", "verdict": "It got worse."}


def plan_of(kinds=("sft", "rlvr"), hours=10, version=1, run="run1", **policy):
    plan = ar.DEFAULT_PLAN(version)
    plan["id"], plan["budget"]["hours"] = run, hours
    plan["phases"] = [{"kind": k, "enabled": True, "settings": {}, "on_pass": "continue", "on_fail": "pause"} for k in kinds]
    for phase in plan["phases"]:
        phase.update(policy)
    return plan


def builder(settings, version):
    return Job(id="", kind="pretrain", label="Pretrain", params={"version": version, "settings": settings})


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.services = FakeServices()
        self.services._versions.append(Version(1, "v1", "first", status="trained", best_checkpoint="ck0"))
        self.events = []
        self.services.watch(lambda kind, payload: self.events.append(payload) if kind == "auto_run" else None)
        self.auto = Orchestrator(self.services, self.tmp.name, build_pretrain=builder)
        self.services.watch(self.auto.on_event)

    def go(self, plan):
        run_id = self.auto.create(plan)
        self.auto.start(run_id)
        return run_id

    def state(self, run_id="run1"):
        return self.auto.view(run_id)["state"]

    def queued(self):
        return [j.id for j in self.services.queue().up_next]

    def phase(self, kind, run_id="run1"):
        return next(p for p in self.state(run_id)["phases"] if p["kind"] == kind)

    def end(self, job_id, status="done", result=None, error=""):
        self.services.finish_job(job_id, status, result, error)


class FlowTests(Base):
    def test_start_enqueues_the_first_phase_as_a_round_job_with_a_stable_id(self):
        self.go(plan_of())
        job = self.services.queue().up_next[0]
        self.assertEqual((job.id, job.kind, job.params["round"], job.params["auto_run"]), ("run1-0-1", "post_round", 1, "run1"))
        self.assertEqual((job.params["version"], job.params["checkpoint"]), (1, "ck0"))
        self.assertEqual((self.state()["status"], self.phase("sft")["status"], self.phase("sft")["job_id"]), ("running", "queued", "run1-0-1"))

    def test_a_done_event_leads_to_the_next_phase_and_the_last_one_finishes_the_run(self):
        self.go(plan_of())
        self.end("run1-0-1", result=GOOD)
        self.assertEqual((self.phase("sft")["status"], self.phase("sft")["checkpoint"]), ("passed", "ck-good"))
        self.assertEqual(self.queued(), ["run1-1-1"])
        self.assertEqual(self.services.queue().up_next[0].params["round"], 2)
        self.end("run1-1-1", result=GOOD)
        self.assertEqual((self.state()["status"], self.state()["current"]), ("done", None))

    def test_every_transition_emits_an_auto_run_event(self):
        self.go(plan_of(("sft",)))
        self.end("run1-0-1", result=GOOD)
        self.assertEqual([e["status"] for e in self.events], ["draft", "running", "running", "done"])
        self.assertEqual(self.events[-1]["id"], "run1")
        journal = self.auto.view("run1")["journal"]
        self.assertEqual([row["event"] for row in journal][:2], ["start", "begin"])

    def test_the_phases_map_to_rounds_1_to_4(self):
        self.go(plan_of(("sft", "rlvr", "distill", "dpo")))
        seen = []
        for _ in range(4):
            job = self.services.queue().up_next[0]
            seen.append(job.params["round"])
            self.end(job.id, result=GOOD)
        self.assertEqual(seen, [1, 2, 3, 4])
        self.assertEqual(self.state()["status"], "done")

    def test_a_pretrain_phase_uses_the_injected_builder_with_the_capped_hours(self):
        plan = plan_of(("pretrain",), hours=2)
        plan["phases"][0]["settings"] = {"hours": 5}
        self.go(plan)
        job = self.services.queue().up_next[0]
        self.assertEqual((job.id, job.kind, job.params["auto_run"]), ("run1-0-1", "pretrain", "run1"))
        self.assertLessEqual(job.params["settings"]["hours"], 2)
        self.services._versions[0].best_checkpoint = "ck-new"
        self.end("run1-0-1")
        self.assertEqual((self.state()["status"], self.phase("pretrain")["checkpoint"]), ("done", "ck-new"))

    def test_a_pretrain_phase_is_not_done_while_its_restore_answers_round_waits(self):
        self.go(plan_of(("pretrain", "sft")))
        restore = Job(id="restore1", kind="restore_answers", params={"round": 3})
        self.services.enqueue(restore)
        self.end("run1-0-1")
        self.assertEqual(self.phase("pretrain")["status"], "queued")
        self.assertEqual(self.queued(), ["restore1"])
        self.end("restore1")
        self.assertEqual(self.phase("pretrain")["status"], "passed")
        self.assertEqual(self.queued(), ["run1-1-1"])

    def test_a_missing_builder_or_version_or_weights_waits_for_the_person(self):
        auto = Orchestrator(self.services, tempfile.mkdtemp(dir=self.tmp.name))
        plan = plan_of(("pretrain",), run="r2")
        auto.start(auto.create(plan))
        self.assertEqual(auto.view("r2")["state"]["status"], "waiting_user")
        self.assertIn("cannot start a pretraining run", auto.view("r2")["state"]["reason"])
        self.services._versions[0].best_checkpoint = None
        self.go(plan_of(("sft",), run="r3", version=1))
        self.assertEqual(self.state("r3")["status"], "waiting_user")
        self.assertIn("no trained weights", self.state("r3")["reason"])
        self.assertEqual(self.queued(), [])

    def test_a_builder_refusal_is_the_reason_shown(self):
        def refuse(settings, version):
            raise ValueError("There is no text to train on yet.")
        self.auto.build_pretrain = refuse
        self.go(plan_of(("pretrain",)))
        self.assertEqual((self.state()["status"], self.state()["reason"]), ("waiting_user", "There is no text to train on yet."))

    def test_midtrain_enqueues_the_stage_job_with_the_steps_and_a_stable_id(self):
        plan = plan_of(("midtrain", "sft"))
        plan["phases"][0]["settings"] = {"steps": 30}
        self.go(plan)
        job = self.services.queue().up_next[0]
        self.assertEqual((job.id, job.kind, job.params["auto_run"], job.params["auto_phase"]), ("run1-0-1", "midtrain", "run1", "midtrain"))
        self.assertEqual((job.params["version"], job.params["config"]), (1, {"steps": 30}))
        self.assertEqual(self.phase("midtrain")["status"], "queued")

    def test_midtrain_without_steps_sends_no_config(self):
        self.go(plan_of(("midtrain",)))
        self.assertNotIn("config", self.services.queue().up_next[0].params)

    def test_a_passed_midtrain_stage_moves_on_to_sft(self):
        self.go(plan_of(("midtrain", "sft")))
        self.end("run1-0-1", result={"stage": "midtrain", "ok": True, "checkpoint": "ck-mid", "checks": []})
        self.assertEqual((self.phase("midtrain")["status"], self.phase("midtrain")["checkpoint"]), ("passed", "ck-mid"))
        self.assertEqual(self.queued(), ["run1-1-1"])

    def test_a_midtrain_stage_that_fails_its_gate_follows_the_plans_policy(self):
        self.go(plan_of(("midtrain", "sft")))
        self.end("run1-0-1", result={"stage": "midtrain", "ok": False, "checkpoint": "ck-kept", "verdict": "It got worse."})
        self.assertEqual((self.phase("midtrain")["status"], self.state()["status"]), ("failed", "paused"))
        self.assertIn("It got worse.", self.phase("midtrain")["error"])
        self.auto.continue_run("run1", "next")
        self.assertEqual((self.phase("midtrain")["status"], self.queued()), ("skipped", ["run1-1-1"]))

    def test_a_midtrain_stage_that_leaves_no_result_fails(self):
        self.go(plan_of(("midtrain",)))
        self.end("run1-0-1")
        self.assertEqual(self.phase("midtrain")["status"], "failed")
        self.assertIn("left no result", self.phase("midtrain")["error"])

    def test_midtrain_needs_trained_weights(self):
        self.services._versions[0].best_checkpoint = None
        self.go(plan_of(("midtrain",)))
        self.assertEqual(self.state()["status"], "waiting_user")
        self.assertIn("no trained weights", self.state()["reason"])
        self.assertEqual(self.queued(), [])

    def test_a_midtrain_phase_cannot_adopt_a_checkpoint(self):
        self.go(plan_of(("midtrain",), on_fail="pause"))
        self.end("run1-0-1", result={"ok": False, "checkpoint": "ck-kept"})
        with self.assertRaises(ar.AutorunError):
            self.auto.continue_run("run1", "adopt")

    def test_the_data_phase_checks_for_text_when_a_phase_trains_on_it(self):
        self.services.reviewed_text = False
        self.go(plan_of(("data", "midtrain")))
        self.assertEqual(self.state()["status"], "waiting_user")
        self.assertIn("no text to train on yet for midtrain", self.state()["reason"])
        self.services.reviewed_text = True
        self.auto.resume("run1")
        self.assertEqual(self.phase("data")["status"], "passed")
        self.assertEqual(self.queued(), ["run1-1-1"])

    def test_the_data_phase_checks_for_conversations_when_sft_is_planned(self):
        self.services.conversations_in_text = 0
        self.go(plan_of(("data", "sft")))
        self.assertEqual(self.state()["status"], "waiting_user")
        self.assertIn("no conversations marked Good yet for sft", self.state()["reason"])
        self.assertNotIn("no text", self.state()["reason"])
        self.services.conversations_in_text = 5
        self.auto.resume("run1")
        self.assertEqual(self.phase("data")["status"], "passed")
        self.assertEqual(self.queued(), ["run1-1-1"])

    def test_the_data_phase_names_every_missing_piece(self):
        self.services.reviewed_text = False
        self.services.conversations_in_text = 0
        self.go(plan_of(("data", "pretrain", "sft")))
        reason = self.state()["reason"]
        self.assertIn("text to train on yet for pretrain", reason)
        self.assertIn("conversations marked Good yet for sft", reason)

    def test_the_data_phase_asks_only_for_what_the_enabled_phases_use(self):
        self.services.reviewed_text = False
        plan = plan_of(("data", "pretrain", "sft"))
        plan["phases"][1]["enabled"] = False
        self.go(plan)
        self.assertEqual(self.phase("data")["status"], "passed")

    def test_stop_removes_the_job_and_leaves_the_checkpoints(self):
        self.go(plan_of())
        self.auto.stop("run1")
        self.assertEqual((self.state()["status"], self.phase("sft")["status"]), ("stopped", "stopped"))
        self.assertEqual(self.queued(), [])
        self.assertEqual(self.services._versions[0].best_checkpoint, "ck0")

    def test_skip_a_blocked_phase_then_resume(self):
        self.services._versions[0].best_checkpoint = None
        self.go(plan_of(("midtrain", "sft")))
        self.services._versions[0].best_checkpoint = "ck0"
        self.auto.skip("run1")
        self.assertEqual(self.phase("midtrain")["status"], "skipped")
        self.auto.resume("run1")
        self.assertEqual(self.queued(), ["run1-1-1"])

    def test_skipping_a_running_phase_is_refused(self):
        self.go(plan_of())
        with self.assertRaises(ar.AutorunError):
            self.auto.skip("run1")

    def test_one_run_per_version_and_no_manual_round_in_the_way(self):
        self.go(plan_of())
        other = self.auto.create(plan_of(run="run2"))
        with self.assertRaises(ar.AutorunError):
            self.auto.start(other)
        self.auto.stop("run1")
        self.services.enqueue(Job(id="manual", kind="post_round", params={"round": 2, "version": 1}))
        with self.assertRaises(ar.AutorunError):
            self.auto.start(other)
        self.services.remove("manual")
        self.auto.start(other)
        self.assertEqual(self.state("run2")["status"], "running")

    def test_a_bad_plan_is_refused_with_its_sentences(self):
        bad = plan_of(("rlvr", "sft"))
        with self.assertRaises(ar.AutorunError) as caught:
            self.auto.create(bad)
        self.assertIn("out of order", str(caught.exception))


class PretrainBuilderTests(unittest.TestCase):
    def test_the_api_builder_makes_the_job_the_start_route_makes_or_says_why_not(self):
        from studio_api.routers.pretraining import auto_run_pretrain
        services = FakeServices()
        services._versions.append(Version(1, "v1", "x", status="saved", spec={}))
        job = auto_run_pretrain(services, {"hours": 1, "passes": 1}, 1)
        self.assertEqual((job.kind, job.params["version"]), ("pretrain", 1))
        with self.assertRaises(ValueError):   # a setting the plan refuses is a sentence for the person, not a crash
            auto_run_pretrain(services, {"hours": -1}, 1)


class RestartTests(Base):
    def restarted(self):
        """The app starts again: a new orchestrator over the same folder, the old one no longer listening."""
        self.services._raw.clear()
        fresh = Orchestrator(self.services, self.tmp.name, build_pretrain=builder)
        self.services.watch(fresh.on_event)
        return fresh

    def test_a_restart_in_the_middle_of_a_phase_waits_for_the_job_it_has(self):
        self.go(plan_of())
        fresh = self.restarted()
        fresh.reconcile()
        self.assertEqual(self.queued(), ["run1-0-1"])
        self.end("run1-0-1", result=GOOD)
        self.assertEqual(fresh.view("run1")["state"]["phases"][0]["status"], "passed")
        self.assertEqual(self.queued(), ["run1-1-1"])

    def test_a_job_that_ended_while_the_app_was_away_is_recorded_then_the_next_one_asked_for(self):
        self.go(plan_of())
        self.services._raw.clear()
        self.end("run1-0-1", result=GOOD)
        fresh = self.restarted()
        fresh.reconcile()
        self.assertEqual(fresh.view("run1")["state"]["phases"][0]["status"], "passed")
        self.assertEqual(self.queued(), ["run1-1-1"])

    def test_a_job_lost_with_the_queue_is_asked_for_again_with_the_same_id(self):
        self.go(plan_of())
        self.services._queue.up_next.clear()
        fresh = self.restarted()
        fresh.reconcile()
        self.assertEqual(self.queued(), ["run1-0-1"])
        fresh.reconcile()
        self.assertEqual(self.queued(), ["run1-0-1"])

    def test_paused_and_finished_runs_stay_as_they_were(self):
        self.go(plan_of())
        self.auto.pause("run1")
        fresh = self.restarted()
        fresh.reconcile()
        self.assertEqual(fresh.view("run1")["state"]["status"], "paused")
        self.assertEqual(self.queued(), ["run1-0-1"])
        self.assertEqual(self.services.queue().up_next[0].status, "held")
        fresh.resume("run1")
        self.assertEqual(self.services.queue().up_next[0].status, "waiting")

    def test_the_files_are_where_the_design_says(self):
        self.go(plan_of())
        folder = Path(self.tmp.name) / "run1"
        self.assertTrue((folder / "plan.json").is_file() and (folder / "state.json").is_file() and (folder / "journal.jsonl").is_file())


class ControllerTests(Base):
    """The parameter controller (studio_autotune) over the queue: the job a retry or the next phase asks for carries the choice, and the
    choice is in the state and the journal with its evidence."""

    def test_a_retry_after_a_divergence_runs_at_half_the_rate_and_says_so(self):
        plan = plan_of(("sft",))
        plan["phases"][0]["settings"]["learning_rate"] = 1e-4
        self.go(plan)
        self.assertEqual(self.services.find_job("run1-0-1").params["learning_rate"], 1e-4)
        self.end("run1-0-1", "failed", error="Non-finite loss. Reduce the learning rate.")
        self.assertEqual(self.state()["status"], "paused")
        self.auto.continue_run("run1", "retry")
        self.assertEqual(self.services.find_job("run1-0-2").params["learning_rate"], 5e-5)
        [choice] = self.state()["choices"]
        self.assertEqual((choice["phase"], choice["attempt"], choice["key"], choice["was"], choice["value"]), ("sft", 2, "learning_rate", 1e-4, 5e-5))
        self.assertIn("Non-finite", choice["evidence"]["error"])
        tuned = [line for line in self.auto.view("run1")["journal"] if line["event"] == "tuned"]
        self.assertEqual([(line["key"], line["value"]) for line in tuned], [("learning_rate", 5e-5)])

    def test_a_pretrain_curve_that_still_falls_lengthens_midtrain(self):
        points = [(float(x), .1 + 5 * x ** -.2) for x in numpy.geomspace(1e3, 1e6, 14)]
        self.services.loss_evidence = lambda version: {"points": points, "run": "r"}
        plan = plan_of(("pretrain", "midtrain"))
        plan["phases"][1]["settings"]["steps"] = 1000
        self.go(plan)
        self.end("run1-0-1", "done")
        self.assertEqual(self.services.find_job("run1-1-1").params["config"]["steps"], 1500)
        [choice] = self.state()["choices"]
        self.assertEqual((choice["rule"], choice["applied"]), ("still_falling", True))

    def test_a_controller_that_is_off_leaves_the_settings_alone(self):
        plan = plan_of(("sft",))
        plan["phases"][0]["settings"]["learning_rate"] = 1e-4
        plan["controller"]["enabled"] = False
        self.go(plan)
        self.end("run1-0-1", "failed", error="Non-finite loss.")
        self.auto.continue_run("run1", "retry")
        self.assertEqual(self.services.find_job("run1-0-2").params["learning_rate"], 1e-4)
        self.assertEqual(self.state().get("choices", []), [])


class RlhfPhaseTests(Base):
    """The RLHF loop as the last phase: rule-checked practice from the sft phase's best weights, gated by the report card."""

    def loop_job(self, plan_kinds=("rlhf",)):
        self.go(plan_of(plan_kinds))
        return self.services.queue().up_next[0]

    def test_the_phase_queues_the_loop_job_with_the_report_card_on_and_no_judge(self):
        plan = plan_of(("rlhf",))
        plan["phases"][0]["settings"] = {"max_rounds": 6, "card_patience": 2, "ignored": 9}
        self.go(plan)
        job = self.services.queue().up_next[0]
        self.assertEqual((job.id, job.kind, job.params["auto_run"], job.params["auto_phase"]), ("run1-0-1", "rlhf_loop", "run1", "rlhf"))
        self.assertEqual(job.params["config"], {"max_rounds": 6, "card_patience": 2})
        self.assertEqual((job.params["no_judge"], job.params["report_card"], job.params["loop_id"]), (True, True, "run1-0-1"))   # its loop id is the job id
        self.assertEqual(self.phase("rlhf")["status"], "queued")

    def test_it_starts_from_the_sft_phases_best_weights(self):
        self.go(plan_of(("sft", "rlhf")))
        self.services._versions[0].best_checkpoint = "ck-sft"      # what Services.record_round does for a round that passed its gate
        self.end("run1-0-1", result=GOOD)
        job = self.services.queue().up_next[0]
        self.assertEqual((job.kind, job.params["checkpoint"]), ("rlhf_loop", "ck-sft"))

    def test_a_refused_setting_waits_for_the_person_with_the_sentence(self):
        plan = plan_of(("rlhf",))
        plan["phases"][0]["settings"] = {"card_patience": 0}
        self.go(plan)
        self.assertEqual(self.state()["status"], "waiting_user")
        self.assertIn("card_patience", self.state()["reason"])

    def test_the_data_phase_wants_checked_problems_and_held_back_data(self):
        self.go(plan_of(("data", "rlhf")))
        self.assertEqual(self.state()["status"], "waiting_user")
        self.assertIn("no problems a rule can check yet for rlhf", self.state()["reason"])
        self.assertIn("no held-back text or problems yet for rlhf", self.state()["reason"])
        rows = {"problems": [{"prompt": "Copy: a", "rule": "copy", "params": {"text": "a"}}], "heldout_docs": ["a held-back text"]}
        self.services.post_rows = lambda kind: rows.get(kind, [])
        self.auto.resume("run1")
        self.assertEqual(self.phase("data")["status"], "passed")
        self.assertEqual(self.queued(), ["run1-1-1"])

    def test_a_loop_that_leaves_no_result_fails_the_phase(self):
        self.loop_job()
        self.end("run1-0-1")
        self.assertEqual(self.phase("rlhf")["status"], "failed")
        self.assertIn("left no result", self.phase("rlhf")["error"])

    def test_a_best_that_fails_the_full_held_back_check_fails_the_phase_by_policy(self):
        self.loop_job()
        self.end("run1-0-1", result={"ok": False, "round": 5, "checkpoint": "ck-best", "verdict": "The loop's best is worse on the held-back data.", "rounds": []})
        self.assertEqual((self.phase("rlhf")["status"], self.state()["status"]), ("failed", "paused"))
        self.assertIn("worse on the held-back data", self.phase("rlhf")["error"])
        with self.assertRaises(ar.AutorunError):
            self.auto.continue_run("run1", "adopt")        # only a post-training round keeps a checkpoint to adopt

    def test_the_loop_keeps_and_rolls_back_rounds_by_the_gate_and_stops_when_the_card_flattens(self):
        """The whole phase in a tiny Auto Run: studio_loop.run_job runs as the queue would, the model, the DPO step and the measurements are
        scripted (no judge: the phase uses rule-checked problems only), and the run ends with the phase's result."""
        import studio_jobs
        import studio_loop
        import tests.test_loop as scripted
        job = self.loop_job()
        table = {0: {"bpc": 2.0, "pass_k": 5}, 1: {"bpc": 1.9, "pass_k": 7},    # round 1: better on both
                 2: {"bpc": 2.5, "pass_k": 7},                                   # round 2: held-back error rose
                 3: {"bpc": 1.85, "pass_k": 8},                                  # round 3: better numbers, but the card says an entry got worse
                 4: {"bpc": 1.9, "pass_k": 7}}                                   # round 4: nothing moved
        card = scripted.scripted_card({1: {"improved": True}, 2: {}, 3: {"ok": False}, 4: {}})
        real = studio_loop.collect
        with tempfile.TemporaryDirectory() as tmp:
            start = Path(tmp) / "start.npz"
            start.write_text("weights", encoding="utf-8")
            job.params.update(output_dir=str(Path(tmp) / "loop"), checkpoint=str(start), rows=scripted.ROWS)
            context = SimpleNamespace(root=tmp, run_dir=tmp, post_rows=lambda kind: [])
            with patch("studio_loop.report_card", lambda ctx: card), \
                 patch("studio_loop.collect", lambda *a, **k: real(*a, **{**k, "model": scripted.PassingModel()})), \
                 patch("studio_rl.measure", scripted.scripted_measure(table)), \
                 patch("studio_eval.noise_band", lambda ckpt, docs: None), \
                 patch("studio_rlhf.dpo_train", scripted.fake_trainer()):
                studio_jobs.HANDLERS["rlhf_loop"](job, context, lambda event: None, threading.Event(), threading.Event())
            result = job.params["result"]
            self.end("run1-0-1", result=result)
        rounds = result["rounds"]
        self.assertEqual([r["status"] for r in rounds], ["kept", "rolled_back", "rolled_back", "rolled_back"])
        self.assertIn("report card", rounds[2]["reason"].lower())                  # round 3 passed the numbers and failed the card
        self.assertIn("report card has not improved", result["stop_reason"])      # three rounds in a row without the card improving
        entry = self.phase("rlhf")
        self.assertEqual(entry["status"], "passed")
        self.assertTrue(entry["checkpoint"].replace("\\", "/").endswith("round_0001/dpo/last.npz"))
        self.assertEqual((result["round"], result["ok"]), (5, True))              # the best was checked on all the held-back data
        self.assertEqual(self.state()["status"], "done")


if __name__ == "__main__":
    unittest.main()
