"""The cluster self-test's verdicts (studio_selftest) with a fake probe, its manager, the peer answers and the routes over FakeServices.
The real thing across two API processes is tests/e2e_cluster.py."""
import tempfile
import threading
import types
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

import studio_selftest as st
from studio_api.app import create_app
from studio_api.hosts import ANY
from studio_ia.services import FakeServices
from studio_peer import PeerError

PEER = "box:8787"


def info(online=True, error="", version="v1", busy=False):
    return types.SimpleNamespace(online=online, error=error, version=version, busy=busy, hostname="box" if online else "", device="cuda", device_name="RTX")


class Client:
    def __init__(self, probe):
        self.probe = probe

    def get_json(self, path, params=None, timeout=None):
        if self.probe.auth is not None:
            raise self.probe.auth
        return {}

    def post_json(self, path, body=None, params=None, timeout=None):
        if path.endswith("/ping"):
            return self.probe.ping
        return {"sha256": self.probe.theirs, "bytes": 1, "seconds": 0.}


def history(first=3.2, last=2.5, contributed=2):
    names = ["me", "box"]
    return [{"round": n, "members": names, "contributed": names[:contributed], "delta_bytes": 30000, "theta_bytes": 50000, "seconds": .5,
             "val_loss": first + (last - first) * n / 2} for n in range(3)]


class Probe:
    """What the app side gives the test, scripted: `running` is the snapshot while the run is open at round 1, `final` once it ends."""

    def __init__(self, peers=(info(),), final_history=None, error="", **changes):
        self.peers, self.auth, self.theirs, self.error = list(peers), None, "aa", error
        self.ping = {"ok": True, "error": "", "seconds": .01}
        self.mine, self.launched, self.stopped = "aa", False, False
        self.lanes = {"me": {"theta_taken": .01}, "box": {"theta_taken": .02}}
        nodes = {"me": {"shards": 4}, "box": {"shards": 4}}
        base = {"nodes": nodes, "timeline": [{"round": 0, "lanes": self.lanes}], "theta_bytes": 100000}
        self.snaps = [{**base, "state": "running", "round": 1, "history": (final_history or history())[:1]},
                      {**base, "state": "finished", "round": 3, "history": final_history or history()}]
        self.run_ = types.SimpleNamespace(machines=[types.SimpleNamespace(local=True, node="me", address="local", state="invited", reason="", peer_job=""),
                                                    types.SimpleNamespace(local=False, node="box", address=PEER, state="invited", reason="", peer_job="j1")])
        for key, value in changes.items():
            setattr(self, key, value)

    def survey(self, addresses):
        return self.peers

    def client(self, address):
        return Client(self)

    def version(self):
        return "v1"

    def launch(self):
        self.launched = True

    def run(self):
        return self.run_

    def run_id(self):
        return "r1"

    def snapshot(self):
        return self.snaps.pop(0) if len(self.snaps) > 1 else self.snaps[0]

    def rows(self):
        return [{"node": "me", "note": ""}, {"node": "box", "note": "box is in use and its plan trains only while it is idle."}]

    def job_error(self):
        return self.error

    def theta_digest(self, number):
        return self.mine

    def term(self):
        return 1

    def stop(self):
        self.stopped = True


def test(probe):
    return st.SelfTest(probe, [PEER], "http://me:8787", "me", sleep=lambda s: None)


class Verdicts(unittest.TestCase):
    def statuses(self, result):
        return {s.id: s.status for s in result.steps}

    def test_a_peer_that_does_not_answer_fails_at_reach_with_its_fix_and_nothing_after_it_runs(self):
        probe = Probe(peers=[info(online=False, error="box:8787 did not answer within 2 seconds.")])
        result = test(probe).run()
        step = next(s for s in result.steps if s.id == f"reach:{PEER}")
        self.assertEqual((result.state, step.status), ("failed", "fail"))
        self.assertIn("did not answer within 2 seconds", step.detail)
        self.assertIn("same Tailscale network", step.fix)
        self.assertFalse(probe.launched)
        rest = [s.status for s in result.steps if s.id != step.id]
        self.assertEqual(set(rest), {"skipped"})
        self.assertIsNotNone(result.finished_at)

    def test_a_working_peer_passes_every_step(self):
        probe = Probe()
        result = test(probe).run()
        self.assertEqual(result.state, "passed", [(s.id, s.status, s.detail) for s in result.steps if s.status != "pass"])
        self.assertTrue(probe.launched)
        self.assertTrue(all(s.status == "pass" for s in result.steps))
        loss = next(s for s in result.steps if s.id == "loss")
        self.assertIn("fell from 3.200 to 2.500", loss.detail)

    def test_a_different_app_version_fails_before_anything_is_started(self):
        result = test(Probe(peers=[info(version="v0")])).run()
        self.assertEqual(self.statuses(result)[f"version:{PEER}"], "fail")
        self.assertEqual(result.state, "failed")

    def test_a_refused_caller_fails_the_auth_step_naming_what_to_allow(self):
        probe = Probe()
        probe.auth = PeerError("answered 403", status=403, detail="not allowed")
        result = test(probe).run()
        step = next(s for s in result.steps if s.id == f"auth:{PEER}")
        self.assertEqual(step.status, "fail")
        self.assertIn("Tailscale login", step.detail)
        self.assertIn("Access", step.fix)

    def test_a_busy_peer_fails_the_free_step(self):
        result = test(Probe(peers=[info(busy=True)])).run()
        self.assertEqual(self.statuses(result)[f"free:{PEER}"], "fail")

    def test_a_peer_that_cannot_reach_this_computer_fails_with_the_address_fix(self):
        probe = Probe()
        probe.ping = {"ok": False, "error": "502 bad gateway", "seconds": 1.}
        result = test(probe).run()
        step = next(s for s in result.steps if s.id == f"back:{PEER}")
        self.assertEqual(step.status, "fail")
        self.assertIn("http://me:8787", step.detail)
        self.assertIn("TRAINER_NODE_ADDRESS", step.fix)
        self.assertFalse(probe.launched)

    def test_weights_that_differ_between_machines_fail_the_checksum(self):
        probe = Probe()
        probe.theirs = "bb"
        result = test(probe).run()
        step = next(s for s in result.steps if s.id == "checksum")
        self.assertEqual((step.status, result.state), ("fail", "failed"))
        self.assertIn("checksum", step.detail)

    def test_a_round_a_member_missed_fails_that_round_and_names_the_machine(self):
        result = test(Probe(final_history=history(contributed=1))).run()
        step = next(s for s in result.steps if s.id == "round:0")
        self.assertEqual(step.status, "fail")
        self.assertIn("box", step.detail)
        self.assertEqual(result.state, "failed")

    def test_a_loss_that_does_not_fall_fails_the_last_step(self):
        result = test(Probe(final_history=history(first=2.0, last=2.4))).run()
        self.assertEqual(self.statuses(result)["loss"], "fail")

    def test_a_peer_that_never_joins_says_why_from_its_row_after_the_wait(self):
        probe = Probe()
        probe.snaps = [{"nodes": {"me": {"shards": 1}}, "timeline": [], "history": [], "state": "running", "round": 0}]
        clock = iter(range(0, 10_000, 100))
        selftest = st.SelfTest(probe, [PEER], "http://me:8787", "me", clock=lambda: next(clock), sleep=lambda s: None)
        result = selftest.run()
        step = next(s for s in result.steps if s.id == f"joined:{PEER}")
        self.assertEqual(step.status, "fail")
        self.assertIn("plan trains only while it is idle", step.detail)
        self.assertTrue(probe.stopped)

    def test_a_job_that_fails_on_this_computer_is_the_verdict(self):
        probe = Probe(error="Out of memory.")
        probe.snaps = [{"nodes": {}, "timeline": [], "history": [], "state": "waiting", "round": 0}]
        result = test(probe).run()
        self.assertEqual(result.state, "failed")
        self.assertIn("Out of memory.", " ".join(s.detail for s in result.steps))


class Peers(unittest.TestCase):
    def test_ping_answers_whether_the_url_is_reachable(self):
        class Good:
            def __init__(self, url):
                pass

            def get_json(self, path):
                return {}

        class Bad(Good):
            def get_json(self, path):
                raise PeerError("refused")
        self.assertTrue(st.peer_ping("http://x", Good)["ok"])
        answer = st.peer_ping("http://x", Bad)
        self.assertEqual((answer["ok"], answer["error"]), (False, "refused"))

    def test_digest_is_the_checksum_of_the_downloaded_weights(self):
        calls = []

        class Coordinator:
            def __init__(self, url):
                pass

            def get_bytes(self, path, params=None):
                calls.append((path, params))
                return b"weights"
        answer = st.peer_digest("http://x", "r1", 2, 1, Coordinator)
        self.assertEqual(answer["sha256"], st.checksum(b"weights"))
        self.assertEqual(calls, [("/collective/r1/params", {"round": 2, "term": 1})])


class Keeping(unittest.TestCase):
    def test_the_last_result_is_kept_and_read_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data" / "cluster_selftest.json"
            path.parent.mkdir()
            manager = st.Manager(path)
            self.assertEqual(manager.state()["state"], "never")
            manager.start(test(Probe()))
            manager.thread.join(20)
            self.assertEqual(manager.state()["state"], "passed")
            self.assertEqual(st.Manager(path).state()["state"], "passed")  # a new process reads it from the file

    def test_a_second_test_cannot_start_while_one_runs(self):
        gate = threading.Event()

        class Slow(Probe):
            def survey(self, addresses):
                gate.wait(10)
                return self.peers
        with tempfile.TemporaryDirectory() as tmp:
            manager = st.Manager(Path(tmp) / "s.json")
            manager.start(test(Slow()))
            self.assertEqual(manager.state()["state"], "running")
            with self.assertRaises(ValueError):
                manager.start(test(Probe()))
            gate.set()
            manager.thread.join(20)


class Routes(unittest.TestCase):
    def setUp(self):
        from unittest import mock

        from studio_api import app as api_app
        with mock.patch.object(api_app, "WEB_OUT", Path(tempfile.gettempdir()) / "no-such-web-out"):
            self.client = TestClient(create_app(FakeServices(), open=True, hosts=ANY))

    def test_nothing_ran_yet(self):
        self.assertEqual(self.client.get("/cluster/selftest").json()["state"], "never")

    def test_starting_it_shows_the_steps_and_the_last_result(self):
        started = self.client.post("/cluster/selftest", json={"machines": ["box"]})
        self.assertEqual(started.status_code, 202)
        shown = self.client.get("/cluster/selftest").json()
        self.assertEqual(shown["state"], "passed")
        self.assertEqual(shown["steps"][0]["title"], "Reach box")

    def test_a_peer_answers_the_ping_and_refuses_a_digest_it_cannot_take(self):
        self.assertFalse(self.client.post("/cluster/selftest/ping", json={"url": "http://127.0.0.1:1"}).json()["ok"])
        self.assertEqual(self.client.post("/cluster/selftest/digest", json={"url": "http://127.0.0.1:1", "run": "r1", "round": 0, "term": 1}).status_code, 409)


class Starting(unittest.TestCase):
    def test_the_real_services_refuse_a_machine_that_is_not_listed_and_an_empty_choice(self):
        from studio_core import Workspace
        from studio_ia.services import WorkspaceServices
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            services = WorkspaceServices(Workspace(Path(tmp) / "ws"))
            self.addCleanup(services.close)
            with self.assertRaisesRegex(ValueError, "not on the Machines tab"):
                services.cluster_selftest_start(["box"], "http://me:8787", 8787)
            with self.assertRaisesRegex(ValueError, "at least one other machine"):
                services.cluster_selftest_start([], "http://me:8787", 8787)
            self.assertEqual(services.cluster_selftest()["state"], "never")


class JobParams(unittest.TestCase):
    def test_the_run_needs_every_machine_and_carries_its_own_text(self):
        machines = [{"node": "me", "address": "local"}, {"node": "box", "address": PEER}]
        p = st.job_params(machines, "http://me:8787", 8787, "me")
        self.assertEqual((p["min_members"], p["max_members"], p["rounds"], p["inner_steps"]), (2, 2, st.ROUNDS, st.INNER_STEPS))
        self.assertEqual(p["config"]["steps"], st.ROUNDS * st.INNER_STEPS)
        self.assertTrue(p["examples"] and p["selftest"] and "version" not in p)


if __name__ == "__main__":
    unittest.main()
