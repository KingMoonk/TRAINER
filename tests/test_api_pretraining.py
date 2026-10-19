import copy
import math
import threading
import unittest
from pathlib import Path

from studio_spec import default_spec

from tests.sse_harness import ApiCase


class PretrainingRouterTests(ApiCase):
    prefix = "/pretraining"

    def test_every_route_answers(self):
        cached = self.call("GET", "/pretraining/preview", params={"dataset_ids": ["a", "b"], "rows": 4})
        self.assertEqual((cached["pieces"], len(cached["first_rows"])), (2_000_000, 4))

        preview = self.job("POST", "/pretraining/preview", json={"dataset_ids": ["a"], "rows": 3})
        self.assertEqual(preview["status"], "done")
        self.assertEqual((preview["result"]["pieces"], preview["result"]["context"], len(preview["result"]["first_rows"])),
                         (2_000_000, 256, 3))

        lists = self.job("POST", "/pretraining/word_lists")
        self.assertEqual(lists["status"], "done", lists)
        result = lists["result"]
        self.assertTrue(result["sentence"])
        self.assertEqual(len(result["lists"]), 4)
        letters = result["lists"][0]
        self.assertEqual((letters["id"], letters["count"] == len(letters["chips"])), ("letters", True))
        self.assertIsNone(result["lists"][2])  # GPT-2's list is not fetched by the fake
        self.assertEqual(set(letters["chips"][0]), {"label", "known"})

        found = self.job("POST", "/pretraining/lr_range_test", json={"spec": {"dims": {"n_ctx": 64}}})
        self.assertEqual(found["status"], "done")
        self.assertEqual(len(found["result"]["rates"]), 100)
        self.assertEqual(found["result"]["steepest"], 70)
        self.assertFalse(found["result"]["diverged"])
        self.run_page_reads()
        self.save_versions()
        rate = self.job("POST", "/pretraining/step_rate", json={"dataset_ids": ["a"], "rows": 8})
        self.assertEqual(rate["status"], "done", rate)
        self.assertEqual((rate["result"]["device"], rate["result"]["rows"]), ("cpu", 8))
        self.call("GET", "/pretraining/text_sets")
        self.call("POST", "/pretraining/plan", json={"dataset_ids": ["a"]})
        self.assertAllRoutesHit()

    def save_versions(self, parent_trained=True, tokenizer_changes=False):
        """Version 1 (trained or not), then version 2 continuing from it."""
        first, second = default_spec(), default_spec()
        if tokenizer_changes:
            second["tokenizer"] = {"id": "character", "config": {}}
        v1 = self.services.save_version(first, "first")
        v2 = self.services.save_version(second, "second")
        v2.parent = 1
        if parent_trained:
            v1.best_checkpoint = "/work/runs/v1/best.npz"
        return v1, v2

    def plan(self, **body):
        return self.call("POST", "/pretraining/plan", json={"dataset_ids": ["Your files"], **body})

    def test_the_text_sets_start_ticked_unless_held_out(self):
        sets = self.call("GET", "/pretraining/text_sets")
        self.assertTrue(sets)
        self.assertEqual({s["id"] for s in sets}, {s["id"] for s in self.services.data("text").sets})
        self.assertTrue(all(s["on"] and not s["held"] for s in sets))

    def test_a_version_that_needs_all_steps_gets_them_and_a_job_the_queue_lists(self):
        self.save_versions()
        plan = self.plan(rows=16, passes=2, hold=0.1)
        self.assertEqual((plan["version"], plan["parent"], plan["mode"], plan["can_continue"], plan["continue_blocked"]),
                         (2, 1, "catch_up", True, None))
        self.assertFalse(plan["estimating"])
        self.assertEqual(plan["pieces_per_step"], 16 * 256)
        self.assertEqual(plan["steps"], math.ceil(2_000_000 * 2 / (16 * 256)))
        self.assertEqual(plan["enough"]["verdict"], "little")
        self.assertTrue(plan["enough"]["headline"])
        self.assertTrue(plan["expectation"]["headline"])
        self.assertEqual((plan["config"]["steps"], plan["config"]["batch_size"], plan["config"]["validation"], plan["config"]["objective"]),
                         (plan["steps"], 16, 0.1, "all"))
        self.assertTrue(plan["can_add"])
        self.assertEqual(plan["reason"], "")
        self.assertTrue(plan["outlook"].startswith("If you add it:"))
        job = plan["job"]
        self.assertEqual((job["kind"], job["label"]), ("pretrain", "Version 2 catch-up"))
        self.assertEqual({k: job["params"][k] for k in ("datasets", "mode", "init_from", "version", "heldout_fraction", "pieces_per_pass", "previous_best")},
                         {"datasets": ["Your files"], "mode": "catch_up", "init_from": "/work/runs/v1/best.npz", "version": 2,
                          "heldout_fraction": 0.1, "pieces_per_pass": 2_000_000, "previous_best": "/work/runs/v1/best.npz"})
        self.assertIs(job["params"]["user_text"], True)
        self.assertIsNone(job["params"]["output_dir"])
        queued = self.call("POST", "/queue/jobs", 201, json=job)
        listed = self.call("GET", "/queue/jobs")
        self.assertIn(queued["id"], [j["id"] for j in listed])
        self.assertEqual(next(j for j in listed if j["id"] == queued["id"])["params"], job["params"])

    def test_the_params_match_the_ones_the_tk_page_built(self):
        from studio_ia.pretraining_plan import pretrain_config
        v1, v2 = self.save_versions()
        job = self.plan(rows=8, passes=4, hold=0.02, mode="from_scratch")["job"]
        steps = math.ceil(2_000_000 * 4 / (8 * 256))
        self.assertEqual(job["params"]["config"], pretrain_config(copy.deepcopy(v2.spec), steps, 8, 0.02))
        self.assertEqual((job["label"], job["params"]["mode"], job["params"]["init_from"]), ("Version 2 from scratch", "from_scratch", None))

    def test_a_mode_is_forced_when_continuing_is_not_possible(self):
        self.save_versions(parent_trained=False)
        plan = self.plan(mode="catch_up")
        self.assertEqual((plan["mode"], plan["can_continue"], plan["continue_blocked"]), ("from_scratch", False, "no_trained_parent"))
        self.assertEqual((plan["job"]["label"], plan["job"]["params"]["init_from"]), ("Version 2 from scratch", None))

    def test_a_changed_word_list_forces_a_fresh_start(self):
        self.save_versions(tokenizer_changes=True)
        plan = self.plan(mode="catch_up")
        self.assertEqual((plan["mode"], plan["continue_blocked"]), ("from_scratch", "word_list_changed"))

    def test_too_little_text_says_so_and_a_missing_choice_blocks_adding(self):
        self.save_versions()
        self.services.cached_preview = lambda ids, rows=16: {"pieces": 100_000, "params": 1_000_000, "context": 256, "first_rows": []}
        plan = self.plan(passes=1)
        self.assertEqual((plan["enough"]["verdict"], plan["steps"]), ("little", math.ceil(100_000 / (16 * 256))))
        self.assertIn("more text", plan["enough"]["headline"])
        none = self.call("POST", "/pretraining/plan", json={"dataset_ids": []})
        self.assertEqual((none["can_add"], none["job"], none["steps"], none["enough"]), (False, None, None, None))
        self.assertEqual(none["reason"], "Choose some text first.")

    def test_an_unknown_size_is_estimating_and_adds_nothing(self):
        self.save_versions()
        self.services.cached_preview = lambda ids, rows=16: None
        plan = self.plan()
        self.assertEqual((plan["estimating"], plan["can_add"], plan["steps"], plan["job"]), (True, False, None, None))
        self.assertIn("still being worked out", plan["reason"])

    def test_no_design_blocks_adding_and_a_named_version_is_used(self):
        plan = self.plan()
        self.assertEqual((plan["version"], plan["can_add"], plan["job"]), (None, False, None))
        self.assertIn("Network designer", plan["reason"])
        self.save_versions()
        self.assertEqual(self.plan(version=1)["version"], 1)
        self.call("POST", "/pretraining/plan", 404, json={"version": 9})
        self.call("POST", "/pretraining/plan", 422, json={"rows": 0})
        self.call("POST", "/pretraining/plan", 422, json={"mode": "sideways"})

    def test_an_unknown_preview_is_null_not_computed(self):
        self.services.cached_preview = lambda ids, rows=16: None
        self.assertIsNone(self.call("GET", "/pretraining/preview"))

    def test_the_preview_without_a_design_is_empty(self):
        self.services.pretrain_preview = lambda ids, rows=16: {}
        done = self.job("POST", "/pretraining/preview", json={})
        self.assertEqual(done["result"], {})

    def test_the_range_test_gets_the_spec_and_a_stop_that_the_job_sets(self):
        seen, running = {}, threading.Event()

        def slow(spec, stop=None):
            seen["spec"] = spec
            running.set()
            stop.wait(10)
            return {"rates": [1e-4], "losses": [3.0], "cancelled": stop.is_set()}
        self.services.lr_range_test = slow
        started = self.call("POST", "/pretraining/lr_range_test", 202, json={"spec": {"x": 1}})
        self.assertTrue(running.wait(10))
        self.call("DELETE", "/jobs/{job_id}", path={"job_id": started["id"]})
        self.events.wait(lambda ev: any(k == "job" and p["id"] == started["id"] and p["status"] == "cancelled" for k, p in ev), "the cancel")
        self.assertEqual(seen["spec"], {"x": 1})
        self.assertTrue(self.client.get(f"/jobs/{started['id']}").json()["result"]["cancelled"])

    def test_a_failing_read_fails_its_job_with_the_sentence(self):
        def refuse(spec, stop=None):
            raise ValueError("There is no plain text to try it on. Add some in Your data.")
        self.services.lr_range_test = refuse
        failed = self.job("POST", "/pretraining/lr_range_test", json={"spec": {}})
        self.assertEqual(failed["status"], "failed")
        self.assertIn("There is no plain text to try it on.", failed["error"])

    def test_bad_requests(self):
        self.call("POST", "/pretraining/lr_range_test", 422, json={})
        self.call("POST", "/pretraining/preview", 422, json={"rows": 0})
        self.call("GET", "/pretraining/preview", 422, params={"rows": 1000})

    def run_page_reads(self):
        """The training-run page's reads find their run from a version or a job, never from a path."""
        from studio_ia.services import Job, QueueState

        asked = []
        self.services.run_curve = lambda run_dir: asked.append(("curve", run_dir)) or {
            "steps": [(10, 5.5)], "smooth": [(10, 5.4)], "heldout": [(10, 5.6)], "lr": [(10, 3e-4)],
            "phases": [(0, 5, "warm-up"), (5, 100, "steady")], "total": 100,
            "samples": [{"step": 10, "text": "once"}], "sample_every": 50, "prompt": "Once"}
        self.services.run_phases = lambda run_dir: asked.append(("phases", run_dir)) or [(0, 5, "warm-up"), (5, 100, "steady")]
        self.services.save_version({"dims": {"n_ctx": 64}}, "first")
        self.services.versions()[0].best_checkpoint = "/work/runs/v1/best.npz"
        running = Job("j1", "pretrain", params={"output_dir": "/work/runs/live"}, status="running")
        self.services.set_queue(QueueState(job=running))

        curve = self.call("GET", "/pretraining/versions/{number}/curve", path={"number": 1})
        self.assertEqual((curve["steps"], curve["total"], curve["samples"], curve["sample_every"]), ([[10, 5.5]], 100, [{"step": 10, "text": "once"}], 50))
        self.assertEqual(curve["phases"][1], [5, 100, "steady"])
        phases = self.call("GET", "/pretraining/versions/{number}/phases", path={"number": 1})
        self.assertEqual(phases[0], {"from_step": 0, "to_step": 5, "name": "warm-up"})
        self.call("GET", "/pretraining/jobs/{job_id}/curve", path={"job_id": "j1"})
        self.call("GET", "/pretraining/jobs/{job_id}/phases", path={"job_id": "j1"})
        self.assertEqual(sorted(asked), sorted([("curve", str(Path("/work/runs/v1"))), ("phases", str(Path("/work/runs/v1"))),
                                                ("curve", "/work/runs/live"), ("phases", "/work/runs/live")]))

        refs = self.call("GET", "/pretraining/versions/{number}/references", path={"number": 1})
        self.assertEqual(set(refs), {"uniform_bpc", "unigram_bpc", "best_so_far_bpc"})
        self.assertIsNone(self.call("GET", "/pretraining/versions/{number}/gate_result", path={"number": 1}))
        self.services.gate_result = lambda version: {"better": True, "diff": -0.1, "upper": -0.02, "vs": 0}
        self.assertEqual(self.call("GET", "/pretraining/versions/{number}/gate_result", path={"number": 1}),
                         {"better": True, "diff": -0.1, "upper": -0.02, "vs": 0})

    def test_a_run_that_does_not_exist_is_a_404_and_no_path_is_taken(self):
        self.services.save_version({}, "no run yet")
        for path in ("/pretraining/versions/9/curve", "/pretraining/versions/9/phases", "/pretraining/versions/9/references",
                     "/pretraining/versions/9/gate_result", "/pretraining/versions/1/curve", "/pretraining/jobs/nope/curve",
                     "/pretraining/jobs/nope/phases"):
            self.assertEqual(self.client.get(path).status_code, 404, path)
        self.assertEqual(self.client.get("/pretraining/versions/1/curve", params={"run_dir": "/etc"}).status_code, 404)

    def test_every_event_reaches_events(self):
        self.job("POST", "/pretraining/preview", json={"dataset_ids": [], "rows": 2})
        self.job("POST", "/pretraining/word_lists")
        done = self.job("POST", "/pretraining/lr_range_test", json={"spec": {}})
        self.assertDelivered()
        finished = [p for p in self.events.of("job") if p["id"] == done["id"]]
        self.assertEqual([p["status"] for p in finished], ["queued", "running", "done"])
        self.assertEqual(finished[-1]["result"]["suggestion"], done["result"]["suggestion"])  # the result rides the last event
        self.assertEqual({p["kind"] for p in self.events.of("job")}, {"pretrain_preview", "word_lists", "lr_range_test"})


class PlanSizingTests(ApiCase):
    """The plan's size section: the verdict, the time from a measured step rate, and the design that fits the hours given."""

    prefix = "/pretraining"
    PIECES, PARAMS = 545_000_000, 133_312  # version 1 on the five catalog text sets
    RATE = {"seconds_per_step": .2275, "tokens_per_second": 18_000., "device": "cpu", "device_name": "PyTorch / CPU", "note": "",
            "rows": 16, "context": 256}

    def plan_for(self, rate=RATE, **body):
        self.services.save_version(default_spec(), "first")
        self.services.cached_preview = lambda ids, rows=16: {"pieces": self.PIECES, "params": self.PARAMS, "vocab": 263, "context": 256,
                                                             "first_rows": []}
        self.services.cached_step_rate = lambda spec, rows: rate
        return self.call("POST", "/pretraining/plan", json={"dataset_ids": ["Your files"], **{"passes": 1, **body}})

    def test_a_model_far_too_small_for_the_text_is_warned_and_given_a_design_that_fits_the_hours(self):
        plan = self.plan_for(hours=4)
        size = plan["sizing"]
        self.assertEqual(size["verdict"], "model_too_small")
        self.assertIn("too small", size["warning"])
        self.assertGreater(size["tokens_per_param"], 1000)
        suggestion = size["suggestion"]
        self.assertFalse(suggestion["same_design"])
        self.assertGreater(suggestion["params"], self.PARAMS)
        self.assertLessEqual(suggestion["hours"], 4.01)  # the steps it suggests fit the stated time at the measured rate
        self.assertGreaterEqual(suggestion["tokens_per_param"], 20)
        self.assertLess(suggestion["steps"], plan["steps"])
        self.assertIn("4.0 h", suggestion["text"])

    def test_the_time_comes_from_the_measured_step_rate(self):
        plan = self.plan_for()
        self.assertTrue(plan["sizing"]["measured"])
        self.assertAlmostEqual(plan["sizing"]["run_hours"], plan["steps"] * .2275 / 3600)
        self.assertEqual(plan["minutes"], round(plan["steps"] * .2275 / 60))
        base = self.call("POST", "/pretraining/plan", json={"dataset_ids": ["Your files"], "passes": 1})
        self.services.cached_step_rate = lambda spec, rows: dict(self.RATE, seconds_per_step=.455)
        again = self.call("POST", "/pretraining/plan", json={"dataset_ids": ["Your files"], "passes": 1})
        self.assertAlmostEqual(again["sizing"]["run_hours"], 2 * base["sizing"]["run_hours"])

    def test_a_chosen_step_count_replaces_the_passes_in_the_job(self):
        plan = self.plan_for(steps=14_007)
        self.assertEqual((plan["steps"], plan["config"]["steps"], plan["job"]["params"]["config"]["steps"]), (14_007, 14_007, 14_007))
        self.assertAlmostEqual(plan["sizing"]["run_hours"], 14_007 * .2275 / 3600)
        self.call("POST", "/pretraining/plan", 422, json={"dataset_ids": ["a"], "steps": 0})
        self.assertIsNone(self.call("POST", "/pretraining/plan", json={"dataset_ids": [], "steps": 5})["steps"])

    def test_a_long_run_gets_the_plateau_rule_and_a_short_one_does_not(self):
        plan = self.plan_for()
        rule = plan["job"]["params"]["plateau"]
        self.assertEqual(rule, plan["plateau"])
        self.assertEqual((rule["window_steps"], rule["min_gain"]), (int(plan["steps"] * .05), .01))
        self.assertIn("Stops by itself", plan["plateau_line"])
        short = self.plan_for(steps=100)  # 100 steps at .2275 s: well under an hour
        self.assertEqual((short["plateau"], short["plateau_line"], "plateau" in short["job"]["params"]), (None, "", False))
        forced = self.plan_for(steps=100, plateau=True)
        self.assertIsNotNone(forced["job"]["params"]["plateau"])
        self.assertNotIn("plateau", self.plan_for(plateau=False)["job"]["params"])

    def test_without_a_measure_the_verdict_stands_and_the_time_waits(self):
        plan = self.plan_for(rate=None)
        size = plan["sizing"]
        self.assertEqual((size["measured"], size["seconds_per_step"], size["run_hours"], size["suggestion"], plan["minutes"]),
                         (False, None, None, None, None))
        self.assertEqual(size["verdict"], "model_too_small")

    def test_an_earlier_run_of_the_design_shows_where_its_error_went_flat(self):
        points = [(s * 4096, v) for s, v in [(1000, 2.515), (5000, 2.006), (10000, 1.843), (20000, 1.764), (40000, 1.678), (80000, 1.622),
                                             (120000, 1.608), (153000, 1.598)]]
        self.services.loss_evidence = lambda version: {"points": points, "run": "runs/queue_49cb71ff"}
        size = self.plan_for(passes=6)["sizing"]  # 6 passes over the five sets: about 800,000 steps, as the live plan asked for
        self.assertIn("its own last run", size["warning"])
        expected = size["expected"]
        self.assertEqual(expected["basis"], "fit")
        self.assertTrue(100_000 < expected["flat_steps"] < 180_000, expected["flat_steps"])
        self.assertLess(expected["loss_planned"], expected["loss_flat"])
        self.assertGreater(len(expected["curve"]), 10)
        self.assertEqual(sorted(expected["curve"]), expected["curve"])

    def test_the_earlier_run_is_not_read_for_a_catch_up_that_starts_from_other_weights(self):
        asked = []
        self.services.loss_evidence = lambda version: asked.append(version.number)
        v1 = self.services.save_version(default_spec(), "first")
        v2 = self.services.save_version(default_spec(), "second")
        v2.parent, v1.best_checkpoint = 1, "/work/runs/v1/best.npz"
        self.services.cached_preview = lambda ids, rows=16: {"pieces": self.PIECES, "params": self.PARAMS, "vocab": 263, "context": 256}
        self.call("POST", "/pretraining/plan", json={"dataset_ids": ["a"], "mode": "catch_up"})
        self.assertEqual(asked, [])
        self.call("POST", "/pretraining/plan", json={"dataset_ids": ["a"], "mode": "from_scratch"})
        self.assertEqual(asked, [2])

    def test_a_plan_that_fits_the_text_says_so(self):
        self.services.save_version(default_spec(), "first")
        self.services.cached_preview = lambda ids, rows=16: {"pieces": 3_000_000, "params": self.PARAMS, "vocab": 263, "context": 256}
        self.services.cached_step_rate = lambda spec, rows: self.RATE
        size = self.call("POST", "/pretraining/plan", json={"dataset_ids": ["a"], "passes": 1})["sizing"]
        self.assertEqual(size["verdict"], "balanced")
        self.assertIn("suit each other", size["warning"])

    def test_the_step_rate_job_needs_the_text_size_and_keeps_its_answer(self):
        self.services.save_version(default_spec(), "first")
        self.services.cached_preview = lambda ids, rows=16: None
        self.call("POST", "/pretraining/step_rate", 409, json={"dataset_ids": ["a"]})
        self.services.cached_preview = lambda ids, rows=16: {"vocab": 300}
        seen = []
        self.services.measure_step_rate = lambda spec, rows, vocab, stop=None: seen.append((rows, vocab)) or self.RATE
        done = self.job("POST", "/pretraining/step_rate", json={"dataset_ids": ["a"], "rows": 16})
        self.assertEqual((done["status"], seen), ("done", [(16, 300)]))
        self.assertEqual(done["result"]["seconds_per_step"], .2275)

    def test_a_device_that_cannot_hold_the_design_fails_the_job_with_the_sentence(self):
        self.services.save_version(default_spec(), "first")

        def refuse(spec, rows, vocab, stop=None):
            raise RuntimeError("CUDA ran out of memory: use fewer rows.")
        self.services.measure_step_rate = refuse
        failed = self.job("POST", "/pretraining/step_rate", json={"dataset_ids": []})
        self.assertEqual(failed["status"], "failed")
        self.assertIn("fewer rows", failed["error"])


if __name__ == "__main__":
    unittest.main()
