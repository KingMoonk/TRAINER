"""A DiLoCo member lives by its own node's queue plan: its memory share, keep-awake and battery pause (issue #488). The members here are
real (studio_diloco.Member training a tiny model against a coordinator behind the real routes); the node's gate is a real queue Runner
whose battery reading is faked, and its pause event is the one the member job is given."""
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

import studio_cluster_queue
import studio_diloco
import studio_distributed
import studio_facts
import studio_jobs
import studio_platform
from studio_queue import Job, Plan, Runner
from tests import diloco_support as support
from tests.test_diloco import Clock


def wait(check, seconds=60.):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if check():
            return True
        time.sleep(.01)
    return check()


class BatteryPause(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.client = self.enterContext(support.TermClient(support.app()))
        self.addCleanup(studio_diloco.COORDINATORS.clear)

    def node(self, name, plan):
        """A node's queue runner (not started: its gate is read by hand) and the battery reading it will see."""
        (self.dir / name / "data").mkdir(parents=True)
        runner = Runner(self.dir / name / "data" / "queue.json", {}, tick=0.)
        runner.set_plan(plan)
        self.addCleanup(runner.shutdown)
        return runner

    def test_a_member_paused_on_battery_is_dropped_at_the_deadline_while_the_run_continues_and_rejoins_when_resumed(self):
        clock = Clock()
        coordinator = support.build(self.dir / "run", "run", clock=clock, inner_steps=3, rounds=4, min_members=2, outer_lr=1.)
        studio_diloco.COORDINATORS["run"] = coordinator
        plugged = Plan(hours="any", idle_only=False, pause_on_battery=False)
        laptop = self.node("laptop", Plan(hours="any", idle_only=False, pause_on_battery=True))
        desktop = self.node("desktop", plugged)
        battery, paused_at = {"on": False}, []

        def gate(runner):
            runner._gate_stale = True
            runner._watch()

        def emit_for(name, runner):
            def emit(event):
                if name == "laptop" and event.get("type") == "metric" and event.get("step") == 1 and not paused_at:
                    battery["on"] = True  # unplugged mid-round: the node's gate pauses its job before the next step
                    gate(runner)
                    paused_at.append(event["step"])
            return emit
        errors, results = [], {}

        def member(name, runner):
            try:
                results[name] = studio_diloco.run_member(support.Peer(self.client), "run", name, 0, self.dir / name, self.dir / f"{name}-out",
                                                         emit_for(name, runner), threading.Event(), wait=.2, pause=runner.pause_event)
            except Exception as exc:  # noqa: BLE001 - reported below
                errors.append(exc)
        with mock.patch.object(studio_platform, "on_battery", lambda: battery["on"]):
            threads = [threading.Thread(target=member, args=(name, runner)) for name, runner in (("desktop", desktop), ("laptop", laptop))]
            for thread in threads:
                thread.start()
            self.assertTrue(wait(lambda: paused_at), "the laptop never started training")
            self.assertEqual(laptop.blocked, "on battery")
            self.assertTrue(laptop.pause_event.is_set())
            self.assertFalse(desktop.pause_event.is_set())  # only the machine on battery pauses
            self.assertTrue(wait(lambda: coordinator.current is not None and "desktop" in coordinator.current.contributions),
                            "the desktop did not finish its round alone")
            clock.now += 5000.  # the round's deadline and grace pass with the laptop still paused
            self.assertTrue(wait(lambda: coordinator.history), "the round did not close")
            self.assertEqual(coordinator.history[0]["dropped"], ["laptop"])
            self.assertEqual(coordinator.history[0]["contributed"], ["desktop"])  # the run went on with the other member
            self.assertEqual(laptop.pause_event.is_set(), True)
            battery["on"] = False  # plugged in again
            gate(laptop)
            self.assertFalse(laptop.pause_event.is_set())
            for thread in threads:
                thread.join(240)
        self.assertEqual(errors, [])
        self.assertTrue(coordinator.finished)
        later = [row for row in coordinator.history[1:] if "laptop" in row["contributed"]]
        self.assertTrue(later, f"the laptop never rejoined: {coordinator.history}")
        self.assertEqual(sorted(results), ["desktop", "laptop"])


class MemberJob(unittest.TestCase):
    """studio_jobs hands the member job the node's plan: its pause event and its memory share."""

    def test_the_member_is_given_the_jobs_pause_and_memory_share(self):
        seen = {}

        def run_member(client, run, node, epoch, root, folder, emit, stop, **options):
            seen.update(options, run=run, node=node)
        job = Job("diloco_member", "member", params={"coordinator_url": "http://owner:8787", "run": "r1", "node": "laptop", "memory_share": .5,
                                                      "output_dir": "out"})
        pause = threading.Event()
        with mock.patch.object(studio_diloco, "run_member", run_member):
            studio_jobs.diloco_member(job, types.SimpleNamespace(root=Path("."), run_dir=Path(".")), lambda e: None, threading.Event(), pause)
        self.assertIs(seen["pause"], pause)
        self.assertEqual(seen["memory_share"], .5)
        self.assertEqual((seen["run"], seen["node"]), ("r1", "laptop"))

    def test_a_member_trains_with_a_batch_that_fits_its_share(self):
        calls = {}

        def capped(config, share, **options):
            calls["share"] = share
            return 1
        member = studio_diloco.Member(None, "r", "n", 0, "shards", threading.Event(), memory_share=.5, pause=threading.Event())
        self.assertEqual(member.memory_share, .5)
        import studio_memory
        with mock.patch.object(studio_memory, "batch_cap", capped):
            self.assertEqual(studio_memory.batch_cap({}, member.memory_share), 1)
        self.assertEqual(calls["share"], .5)


def run_of(node="laptop", **machine):
    return studio_distributed.Run("r1", "x", 1, 0., {}, "owner", [studio_distributed.Machine(node=node, address=node, **machine)])


class Reasons(unittest.TestCase):
    """The run view says why a member is not training, in the shared member vocabulary, from the limits its node publishes."""

    def row(self, state, held, running=True, **machine):
        run = run_of(**machine)
        snap = {"nodes": {"laptop": {"state": state, "note": "", "missed": 1, "sent": False}} if state else {}, "history": []}
        peers = {"laptop": {"down": False, "member_running": running, "blocking": None, "limits": held}}
        return studio_distributed.member_rows(run, snap, "", peers)[0]

    def test_a_member_whose_node_is_on_battery_says_so_and_what_ends_it(self):
        row = self.row("behind", {"blocked": "on battery", "paused": False})
        self.assertEqual(row["state"], "behind")
        self.assertEqual(row["reason"], "laptop is on battery and its plan pauses training then. It continues when it is plugged in.")
        self.assertEqual(row["note"], row["reason"])

    def test_every_gate_reason_and_a_paused_queue_has_its_own_text(self):
        said = {why: studio_distributed.limit_reason("laptop", {"blocked": why, "paused": False})
                for why in ("on battery", "outside hours", "computer in use", "passive training is running (session ab)", "something new")}
        self.assertEqual(len(set(said.values())), 5)
        self.assertIn("outside the hours", said["outside hours"])
        self.assertIn("passive training", said["passive training is running (session ab)"])
        self.assertIn("something new", said["something new"])
        self.assertIn("paused", studio_distributed.limit_reason("laptop", {"blocked": None, "paused": True}))

    def test_a_machine_waiting_for_its_plan_is_waiting_not_just_invited(self):
        row = self.row(None, {"blocked": "outside hours", "paused": False}, running=False)
        self.assertEqual(row["state"], "waiting")
        self.assertIn("outside the hours", row["reason"])

    def test_nothing_is_said_when_the_plan_holds_nothing(self):
        row = self.row("training", None)
        self.assertEqual(row["state"], "training")
        self.assertNotIn("battery", row["reason"])

    def test_a_finished_run_keeps_its_own_account(self):
        run = run_of()
        snap = {"nodes": {"laptop": {"state": "training", "note": "", "missed": 0, "sent": False}}, "history": []}
        peers = {"laptop": {"down": False, "member_running": True, "blocking": None, "limits": {"blocked": "on battery", "paused": False}}}
        self.assertEqual(studio_distributed.member_rows(run, snap, "finished", peers)[0]["state"], "finished")


class Published(unittest.TestCase):
    def test_a_node_publishes_its_plan_and_what_holds_it(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            (Path(tmp) / "data").mkdir()
            runner = Runner(Path(tmp) / "data" / "queue.json", {}, tick=0.)
            self.addCleanup(runner.shutdown)
            runner.set_plan(Plan(hours="any", idle_only=False, pause_on_battery=True, keep_awake=True, memory="half"))
            runner.blocked = "on battery"
            found = studio_facts.queue_limits(runner)
        self.assertEqual(found["blocked"], "on battery")
        self.assertEqual((found["plan"]["memory"], found["plan"]["keep_awake"], found["plan"]["pause_on_battery"]), ("half", True, True))
        self.assertFalse(found["paused"])

    def test_limits_of_reads_a_queue_answer_and_ignores_anything_else(self):
        self.assertIsNone(studio_cluster_queue.limits_of({"job": None}))
        self.assertIsNone(studio_cluster_queue.limits_of("nope"))
        found = studio_cluster_queue.limits_of({"plan": {"memory": "all"}, "blocked": "", "paused": True})
        self.assertEqual(found, {"plan": {"memory": "all"}, "blocked": None, "paused": True})

    def test_peer_facts_carry_the_limits_of_a_machine_the_store_holds_a_queue_of(self):
        class Replica:
            def __init__(self, entries):
                self.entries = entries

            def get(self, origin, kind, key):
                return self.entries.get((origin, kind, key))
        entries = {("laptop", "queue", "jobs"): {"value": [], "age": 1.},
                   ("laptop", "queue", "limits"): {"value": {"plan": {}, "blocked": "on battery", "paused": False}, "age": 1.}}
        cluster = types.SimpleNamespace(state=Replica(entries), is_down=lambda node: False)
        run = run_of()
        self.assertEqual(studio_distributed.peer_facts(run, cluster)["laptop"]["limits"], {"blocked": "on battery", "paused": False})
        entries[("laptop", "queue", "limits")]["value"]["blocked"] = None
        self.assertIsNone(studio_distributed.peer_facts(run, cluster)["laptop"]["limits"])


if __name__ == "__main__":
    unittest.main()
