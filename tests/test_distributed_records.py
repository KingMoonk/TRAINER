"""The Training tab's run records are kept in the workspace (data/distributed_runs.json) and read back when the API starts again."""
import tempfile
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

import studio_diloco
import studio_distributed as runs
import studio_queue
from studio_api.app import create_app
from studio_api.hosts import ANY
from studio_core import Workspace, read_json
from studio_ia.services import WorkspaceServices
from tests import diloco_support as support

MACHINES = [{"node": "mac", "address": "local", "device": "mps", "device_name": "M3"},
            {"node": "box", "address": "box:8787", "device": "cuda", "device_name": "RTX"}]


class Session:
    """The API over a workspace, started afresh each time as after a restart: this process forgets its runs first."""

    def __init__(self, test, root):
        runs.RUNS.clear()
        studio_diloco.COORDINATORS.clear()
        self.services = WorkspaceServices(Workspace(root))
        test.addCleanup(self.services.close)
        self.client = test.enterContext(TestClient(create_app(self.services, open=True, hosts=ANY)))

    def runs(self):
        return self.client.get("/distributed/runs").json()


class RunRecords(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(runs.RUNS.clear)
        self.addCleanup(studio_diloco.COORDINATORS.clear)
        self.root = Path(self.tmp.name) / "ws"
        self.file = self.root / "data" / "distributed_runs.json"

    def start(self, session, rounds=2):
        """A run the owner has: its job in the queue and its coordinator, with both machines in."""
        job = studio_queue.Job(id="run1", kind="pretrain_distributed", label="Version 1 across 2 machines", status="running", created_at=time.time(), params={
            "version": 1, "machines": MACHINES, "owner": "mac", "settings": {"rounds": rounds, "inner_steps": 2}})
        session.services.runner.jobs.append(job)
        runs.open_run(job)
        coordinator = support.build(Path(self.tmp.name) / "run1", run="run1", rounds=rounds, inner_steps=2, outer_lr=1.)
        studio_diloco.register("run1", coordinator)
        for node in ("mac", "box"):
            coordinator.join(node)
        return coordinator

    def round(self, coordinator):
        number = coordinator.current.number
        for node in list(coordinator.current.members):
            coordinator.submit(number, node, coordinator.current.generation, 0, 10, 2, support.constant_delta(coordinator), 1., 3.)

    def kept(self, state, rounds):
        """Wait for the record on disk to hold the run in `state` with `rounds` finished (it is written just after the round closes)."""
        end = time.monotonic() + 10
        while time.monotonic() < end:
            try:
                final = read_json(self.file)["runs"][0]["final"]
                if final and final["state"] == state and final["round"] == rounds:
                    return
            except (OSError, ValueError, KeyError, IndexError):
                pass
            time.sleep(.02)
        self.fail(f"the record never held a {state} run at round {rounds}")

    def test_a_finished_run_is_still_listed_with_its_rounds_and_members_after_a_restart(self):
        first = Session(self, self.root)
        coordinator = self.start(first)
        self.round(coordinator)
        self.round(coordinator)
        self.assertTrue(coordinator.finished)
        self.kept("finished", 2)
        again = Session(self, self.root)
        (run,) = again.runs()
        self.assertEqual((run["id"], run["state"], run["round"], run["rounds"], run["version"]), ("run1", "finished", 2, 2, 1))
        self.assertEqual([row["round"] for row in run["history"]], [0, 1])
        self.assertEqual({member["node"] for member in run["members"]}, {"mac", "box"})
        self.assertEqual(again.client.get("/distributed/runs/run1").json()["id"], "run1")

    def test_a_run_going_when_the_app_stopped_loads_as_stopped_with_the_restart_note(self):
        first = Session(self, self.root)
        coordinator = self.start(first, rounds=5)
        self.round(coordinator)
        self.kept("running", 1)
        again = Session(self, self.root)
        (run,) = again.runs()
        self.assertEqual((run["state"], run["round"], run["can_stop"]), ("stopped", 1, False))
        self.assertIn("restarted", run["note"])
        self.assertEqual(len(run["history"]), 1)

    def test_a_missing_or_corrupt_file_loads_as_no_runs_and_does_not_stop_the_start(self):
        self.assertEqual(Session(self, self.root).runs(), [])
        for text in ("", "{not json", '{"runs": 3}', '{"runs": [7, {"id": "x"}]}', "[]"):
            with self.subTest(text=text):
                self.file.write_text(text, encoding="utf-8")
                self.assertEqual(Session(self, self.root).runs(), [])

    def test_only_the_newest_runs_are_kept(self):
        Session(self, self.root)
        for n in range(runs.KEPT + 3):
            runs.open_run(studio_queue.Job(id=f"r{n}", kind="pretrain_distributed", created_at=n + 1, params={"machines": MACHINES}))
        self.assertEqual(len(read_json(self.file)["runs"]), runs.KEPT)
        again = Session(self, self.root)
        self.assertEqual(len(again.runs()), runs.KEPT)
        self.assertNotIn("r0", {run["id"] for run in again.runs()})

    def test_a_run_whose_job_the_queue_still_holds_waits_to_be_resumed_instead_of_stopping(self):
        first = Session(self, self.root)
        coordinator = self.start(first, rounds=5)
        self.round(coordinator)
        self.kept("running", 1)
        again = Session(self, self.root)
        again.services.runner.jobs.append(studio_queue.Job(id="run1", kind="pretrain_distributed", status="waiting", params={"machines": MACHINES}))
        (run,) = again.runs()
        self.assertEqual((run["state"], run["note"], run["round"]), ("queued", "", 1))


if __name__ == "__main__":
    unittest.main()
