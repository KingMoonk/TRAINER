"""Starting a passive session from the page and resuming a failed or stopped one: the store, the services and the trainer's restart.
A fake spawn and a fake trainer stand in for every process; nothing here starts a watchdog."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from studio_ia.services import Version, WorkspaceServices
from studio_passive import AlreadyActive, PassiveStore, merge_plan
from studio_spec import default_spec
from tests.test_passive import Rig


class Services(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.spawned, self.installed = [], []
        self.hooks = dict(spawn=lambda command, log, cwd: self.spawned.append(command) or 777, install=self.installed.append,
                          alive=lambda pid: False, ours=lambda *a: False)
        self.checkpoint = self.root / "v2.npz"
        self.checkpoint.write_bytes(b"x")
        sets = [SimpleNamespace(id="mine", name="Mine"), SimpleNamespace(id="more", name="More")]
        self.versions = [Version(1, spec=default_spec()), Version(2, spec=default_spec(), best_checkpoint=str(self.checkpoint))]
        self.services = SimpleNamespace(root=self.root, versions=lambda: self.versions, data=lambda kind: {"sets": [{"id": s.id, "name": s.name} for s in sets]},
                                        _abs=lambda p: WorkspaceServices._abs(self.services, p))

    def start(self, plan=None, version=None):
        return WorkspaceServices.start_passive(self.services, plan or {}, version, **self.hooks)

    def test_a_session_is_created_with_the_merged_plan_and_its_watchdog_started(self):
        sid = self.start({"stop": {"duration": 2, "unit": "days"}, "teacher": {"polish": False, "lessons": False}}, version=2)
        row = PassiveStore(self.root).get(sid)
        self.assertEqual((row["status"], row["watchdog_pid"]), ("queued", 777))
        plan = row["plan"]
        self.assertEqual((plan["stop"]["duration"], plan["stop"]["unit"], plan["stop"]["patience_rounds"], plan["teacher"]["polish"]), (2, "days", 3, False))
        self.assertEqual((plan["data"]["sources"], plan["base_checkpoint"], plan["config"]["name"]), (["mine", "more"], str(self.checkpoint), "My network"))
        self.assertEqual((len(self.spawned), self.installed), (1, [self.root]))

    def test_the_current_version_and_chosen_sources_are_used(self):
        sid = self.start({"data": {"sources": ["more"]}})
        plan = PassiveStore(self.root).get(sid)["plan"]
        self.assertEqual((plan["data"]["sources"], plan["base_checkpoint"]), (["more"], str(self.checkpoint)))
        PassiveStore(self.root).update(sid, status="stopped")
        plan = PassiveStore(self.root).get(self.start(version=1))["plan"]
        self.assertIsNone(plan["base_checkpoint"])  # version 1 has not trained

    def test_a_bad_plan_or_version_is_a_sentence_and_creates_nothing(self):
        for plan, version, words in (({"stop": {"duration": 0}}, None, "time limit"), ({"limits": {"memory_fraction": 3}}, None, "memory limit"),
                                     ({}, 9, "no version 9")):
            with self.subTest(words), self.assertRaisesRegex(ValueError, words):
                self.start(plan, version)
        self.versions.clear()
        with self.assertRaisesRegex(ValueError, "Design a version"):
            self.start()
        self.assertEqual((PassiveStore(self.root).list(), self.spawned), ([], []))

    def test_a_second_active_session_is_refused(self):
        self.start()
        with self.assertRaises(RuntimeError) as refused:
            self.start()
        self.assertIsInstance(refused.exception, AlreadyActive)
        self.assertEqual((len(PassiveStore(self.root).list()), len(self.spawned)), (1, 1))

    def test_a_watchdog_that_cannot_start_leaves_a_failed_session_with_the_reason(self):
        def refuse(command, log, cwd):
            raise OSError("no process for you")
        with self.assertRaisesRegex(RuntimeError, "no process for you"):
            WorkspaceServices.start_passive(self.services, {}, None, **{**self.hooks, "spawn": refuse})
        row = PassiveStore(self.root).list()[0]
        self.assertEqual(row["status"], "failed")
        self.assertIn("no process for you", row["error"])


class Resume(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.services = SimpleNamespace(root=self.root)
        self.spawned = []
        self.hooks = dict(spawn=lambda command, log, cwd: self.spawned.append(command) or 888, install=lambda root: None,
                          alive=lambda pid: False, ours=lambda *a: False)
        self.store = PassiveStore(self.root)
        self.id = self.store.create(merge_plan({}))

    def resume(self, session_id=None):
        return WorkspaceServices.command_passive(self.services, session_id or self.id, "resume", **self.hooks)

    def test_failed_and_stopped_go_back_to_queued_with_a_watchdog(self):
        for status in ("failed", "stopped"):
            with self.subTest(status):
                self.spawned.clear()
                self.store.update(self.id, status=status, command="stop", error="ran out of disk", trainer_pid=4001, watchdog_pid=4002, heartbeat=5.0,
                                  restarts=[1.0, 2.0], state={"started": 1.0, "round": 2, "stop_reason": "user", "paused": True, "checkpoints": [{"n": 1}]})
                (self.store.directory(self.id) / "watchdog.json").write_text('{"fruitless": 5, "spawned": 1}')
                self.assertEqual(self.resume(), "queued")
                row = self.store.get(self.id)
                self.assertEqual((row["status"], row["command"], row["error"], row["trainer_pid"], row["watchdog_pid"], row["restarts"]), ("queued", "", "", None, 888, []))
                self.assertEqual((row["state"]["started"], row["state"]["checkpoints"], row["state"]["stop_reason"], row["state"]["paused"]), (1.0, [{"n": 1}], None, False))
                self.assertFalse((self.store.directory(self.id) / "watchdog.json").exists())
                self.assertEqual(len(self.spawned), 1)
                self.assertIn("resume", [e["type"] for e in self.store.journal(self.id)])

    def test_the_trainer_carries_on_from_its_last_checkpoint(self):
        rig = Rig(self, [3.0, 2.0, 1.0, 0.5], stop={"target_error": 0.6}, plateau={"checks": 9}, fail_train_on=2)
        with self.assertRaises(RuntimeError):
            rig.run()
        first = rig.row()["state"]["checkpoints"][0]["path"]
        rig.store.update(rig.id, status="failed")  # what the watchdog writes when it gives up
        services = SimpleNamespace(root=rig.root)
        self.assertEqual(WorkspaceServices.command_passive(services, rig.id, "resume", **self.hooks), "queued")
        state = rig.run()
        self.assertEqual((rig.row()["status"], state["stop_reason"]), ("finished", "target"))
        self.assertEqual(state["checkpoints"][0]["path"], first)  # round 1 was kept, not trained again
        self.assertEqual(rig.types().count("start"), 1)
        self.assertIn("restart", rig.types())

    def test_only_failed_and_stopped_can_be_resumed(self):
        for status in ("finished",):
            self.store.update(self.id, status=status)
            with self.assertRaises(RuntimeError):
                self.resume()
        self.assertEqual(self.spawned, [])
        self.assertIsNone(self.resume("nope"))

    def test_another_active_session_blocks_it_and_nothing_changes(self):
        self.store.update(self.id, status="failed", error="x")
        other = self.store.create(merge_plan({}))
        with self.assertRaises(RuntimeError):
            self.resume()
        self.assertEqual((self.store.get(self.id)["status"], self.store.get(self.id)["error"], self.spawned), ("failed", "x", []))
        self.store.update(other, status="finished")
        self.assertEqual(self.resume(), "queued")

    def test_pause_still_refuses_an_ended_session(self):
        self.store.update(self.id, status="failed")
        with self.assertRaises(RuntimeError):
            WorkspaceServices.command_passive(self.services, self.id, "pause", **self.hooks)


if __name__ == "__main__":
    unittest.main()
