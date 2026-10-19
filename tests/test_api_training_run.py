from studio_ia.services import FakeServices, Job, QueueState, Version
from tests.sse_harness import ApiCase

CURVE = {"steps": [(1, 3.0), (2, 2.8), (3, 2.7), (4, 2.6)], "smooth": [(1, 3.0), (4, 2.7)], "heldout": [(2, 2.9), (4, 2.7)], "lr": [],
         "phases": [(0, 1, "warm-up"), (1, 3, "steady"), (3, 4, "cool-down")], "total": 4,
         "samples": [{"step": 2, "text": "The cat"}], "sample_every": 2, "prompt": "Once"}


class RunServices(FakeServices):
    def __init__(self):
        super().__init__()
        self.curve_of = []
        self.refs = {"uniform_bpc": 8.0, "unigram_bpc": 5.0, "best_so_far_bpc": 2.9}
        self.gate = None

    def run_curve(self, run_dir):
        self.curve_of.append(run_dir)
        return CURVE

    def references(self, version):
        return self.refs

    def gate_result(self, version):
        return self.gate


class TrainingRunRouterTests(ApiCase):
    prefix = "/training-run"

    def make_services(self):
        return RunServices()

    def add_versions(self):
        self.services._versions = [Version(1, "version 1", "First design", best_checkpoint="out/v1/best.npz", status="trained"),
                                   Version(2, "version 2", "6 blocks", parent=1, status="training")]

    def test_the_fake_with_nothing_running_is_empty(self):
        page = self.call("GET", "/training-run")
        self.assertEqual((page["version"], page["running"], page["status"], page["phase"]), (None, False, None, ""))
        self.assertEqual(page["switch"], {"key": "no_band_yet", "gap": None, "docs": None, "need": None})
        self.assertAllRoutesHit()

    def test_a_running_job_reads_its_own_run_and_says_the_phase_and_status(self):
        self.add_versions()
        job = Job("j1", "pretrain", "v2", status="running", params={"output_dir": "out/v2"})
        self.services.set_queue(QueueState(job=job, step=2, total=4))
        page = self.call("GET", "/training-run")
        self.assertEqual(self.services.curve_of, ["out/v2"])
        self.assertEqual((page["version"], page["parent"], page["change"], page["running"], page["job_id"]), (2, 1, "6 blocks", True, "j1"))
        self.assertEqual((page["step"], page["total"], page["phase"], page["status"]), (2, 4, "steady", "learning"))
        self.assertEqual(page["curve"]["samples"], [{"step": 2, "text": "The cat"}])
        self.assertEqual((page["chat_version"], page["gate_vs"]), (1, None))
        self.assertEqual(page["switch"]["key"], "held_back_below")  # 2.7 held back against 2.9 so far
        self.assertEqual(page["references"]["uniform_bpc"], 8.0)

    def test_the_manager_phase_wins_over_the_schedule(self):
        self.add_versions()
        job = Job("j1", "pretrain", "v2", status="running", params={"output_dir": "out/v2"})
        self.services.set_queue(QueueState(job=job, step=2, total=4, phase="cool-down"))
        self.assertEqual(self.call("GET", "/training-run")["phase"], "cool-down")

    def test_after_the_run_ended_the_versions_latest_run_and_the_gate_verdict_show(self):
        self.add_versions()
        self.services.gate = {"better": True, "diff": -0.12, "upper": -0.05, "vs": 1}
        self.services._versions[1].runs = ["out/v2"]
        page = self.call("GET", "/training-run")
        self.assertEqual(self.services.curve_of, ["out/v2"])
        self.assertEqual((page["running"], page["job_id"], page["step"], page["total"], page["phase"]), (False, None, 4, 4, "cool-down"))
        self.assertEqual((page["gate_vs"], page["switch"]), (1, {"key": "gate_better", "gap": "0.120", "docs": None, "need": None}))

    def test_a_version_with_no_run_has_an_empty_curve(self):
        self.add_versions()
        page = self.call("GET", "/training-run")
        self.assertEqual((page["curve"]["steps"], page["status"], page["running"]), ([], None, False))
        self.assertEqual(self.services.curve_of, [])
        self.assertIsNone(page["ended"])

    def test_a_run_that_ended_itself_says_why(self):
        import json
        import tempfile
        from pathlib import Path
        self.add_versions()
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            self.services._versions[1].runs = [str(run)]
            self.assertIsNone(self.call("GET", "/training-run")["ended"])  # no run.json yet
            ended = {"reason": "plateau", "text": "Stopped at step 30 because the loss went flat.", "step": 30, "best": 1.5}
            (run / "run.json").write_text(json.dumps({"status": "completed", "ended": ended}), encoding="utf-8")
            page = self.call("GET", "/training-run")
            self.assertEqual(page["ended"], {"reason": "plateau", "text": ended["text"], "step": 30})
            self.services.set_queue(QueueState(job=Job("j1", "pretrain", "v2", status="running", params={"output_dir": "out/v3"}), step=1, total=4))
            self.assertIsNone(self.call("GET", "/training-run")["ended"])  # a run in progress has not ended
