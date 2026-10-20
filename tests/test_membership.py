"""One member vocabulary for both computers: the owner's member rows (studio_distributed.member_rows with peer_facts), this computer's
invites (studio_membership), and the routes to pause a local job and join, leave a run and ask a peer to pause."""
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

import studio_distributed as runs
import studio_membership as membership
from studio_api import app as api_app
from studio_api.app import create_app
from studio_api.hosts import ANY
from studio_ia.services import FakeServices, Job
from studio_peer import PeerError
from tests.test_api_distributed import MACHINES, FakeClient, clients, owner_job


def peer(down=False, member_running=False, blocking=None, slow=False):
    return {"down": down, "slow": slow, "member_running": member_running, "blocking": blocking}


class OwnerSide(unittest.TestCase):
    """The state, reason and blocking job of each machine on the owner's page."""

    def setUp(self):
        self.addCleanup(runs.RUNS.clear)
        self.job = owner_job(MACHINES[:2])
        self.run = runs.run_of(self.job)
        self.box = self.run.machines[1]
        self.box.peer_job, self.box.state = "job-box", "invited"

    def row(self, peers=None, snap=None, ended=""):
        return {m["node"]: m for m in runs.member_rows(self.run, snap, ended, peers)}["box"]

    def snap(self, state):
        return {"state": "running", "round": 2, "rounds": 4, "nodes": {"box": {"state": state, "missed": 1 if state == "behind" else 0, "sent": False,
                                                                              "note": "Missed 1 round in a row." if state == "behind" else ""}}, "history": []}

    def test_a_machine_that_has_not_been_heard_from_is_invited(self):
        row = self.row()
        self.assertEqual((row["state"], row["blocking_job"]), ("invited", None))
        self.assertIn("joins by itself", row["reason"])

    def test_a_machine_busy_with_another_job_is_waiting_and_names_that_job_and_its_time_left(self):
        row = self.row({"box": peer(blocking={"label": "Version 1", "eta_s": 720.})})
        self.assertEqual((row["state"], row["blocking_job"]), ("waiting", {"label": "Version 1", "eta_s": 720.}))
        self.assertEqual(row["reason"], "Busy with 'Version 1', about 12 min left. It joins when that finishes.")
        self.assertTrue(row["can_stop"])

    def test_a_busy_machine_with_no_time_left_to_give_says_so_without_one(self):
        row = self.row({"box": peer(blocking={"label": "Version 1", "eta_s": None})})
        self.assertEqual(row["reason"], "Busy with 'Version 1'. It joins when that finishes.")

    def test_a_machine_the_scheduler_gave_no_slot_is_waiting_not_unplaced(self):
        self.box.peer_job, self.box.state, self.box.reason = "", "unplaced", runs.UNPLACED
        row = self.row()
        self.assertEqual((row["state"], row["reason"], row["can_stop"]), ("waiting", runs.UNPLACED, False))

    def test_a_machine_that_is_not_answering_is_unreachable_with_why(self):
        row = self.row({"box": peer(down=True)})
        self.assertEqual(row["state"], "unreachable")
        self.assertIn("box did not answer", row["reason"])

    def test_a_machine_down_because_it_refused_or_is_unknown_is_told_to_be_checked(self):
        row = self.row({"box": peer(down=True)})
        self.assertEqual(row["reason"], "box did not answer. Check that it is on and that Tailscale is running on it.")

    def test_a_machine_down_only_because_it_answers_slowly_says_so_without_the_advice_to_turn_it_on(self):
        row = self.row({"box": peer(down=True, slow=True)})
        self.assertEqual((row["state"], row["reason"]), ("unreachable", "box is answering slowly: it did not answer within 2 seconds. It is asked again shortly."))

    def test_an_invite_that_timed_out_says_the_machine_is_slow_and_one_refused_does_not(self):
        slow = runs.refusal(self.box, PeerError(f"{self.box.address} did not answer POST /queue/jobs: timed out"))
        self.assertEqual(slow, "box is answering slowly: it did not answer within 30 seconds. It is invited again at the next round.")
        refused = runs.refusal(self.box, ConnectionRefusedError())
        self.assertNotIn("slowly", refused)
        self.assertIn("refused the connection", refused)

    def test_peer_facts_say_slow_only_for_a_down_machine_whose_last_poll_timed_out(self):
        import studio_facts
        queue = [{"id": "job-box", "status": "waiting", "label": "m"}]
        for down, slowly, expected in ((True, True, True), (True, False, False), (False, True, False)):
            cluster = types.SimpleNamespace(is_down=lambda node, d=down: d, answers_slowly=lambda node, s=slowly: s)
            with mock.patch.object(studio_facts, "queue_facts", return_value=(queue, 3.)), mock.patch.object(studio_facts, "limits_facts", return_value=None), \
                mock.patch.object(studio_facts, "member_failures", return_value=[]):
                self.assertEqual(runs.peer_facts(self.run, cluster)["box"]["slow"], expected)

    def test_a_machine_that_refused_the_invite_is_unreachable_with_the_refusal(self):
        self.box.state, self.box.reason = "unreachable", "box refused the connection."
        row = self.row()
        self.assertEqual((row["state"], row["reason"]), ("unreachable", "box refused the connection."))

    def test_a_machine_whose_member_job_failed_there_is_not_invited_any_more_and_says_why(self):  # #543
        error = "https://beast.tailb22e91.ts.net answered 502 to POST /collective/run1/join. Set TRAINER_NODE_ADDRESS to an address box can reach."
        row = self.row({"box": {**peer(), "member_failed": error}})
        self.assertEqual((row["state"], row["can_stop"]), ("unreachable", False))
        self.assertEqual(row["reason"], f"The member job on box failed: {error}")

    def test_peer_facts_carry_the_error_of_the_member_job_that_failed_on_the_machine(self):  # #543
        import studio_facts
        cluster = types.SimpleNamespace(is_down=lambda node: False, answers_slowly=lambda node: False)
        failures = [{"id": "other", "run": "x", "error": "not this one"}, {"id": "job-box", "run": self.run.id, "error": "It could not reach the owner."}]
        with mock.patch.object(studio_facts, "queue_facts", return_value=([], 3.)), mock.patch.object(studio_facts, "limits_facts", return_value=None), \
                mock.patch.object(studio_facts, "member_failures", return_value=failures):
            facts = runs.peer_facts(self.run, cluster)
        self.assertEqual(facts["box"]["member_failed"], "It could not reach the owner.")
        row = self.row(facts)
        self.assertEqual((row["state"], row["reason"]), ("unreachable", "The member job on box failed: It could not reach the owner."))

    def test_a_machine_publishes_its_failed_member_jobs_and_only_those(self):  # #543
        import studio_facts
        from studio_queue import Job
        jobs = [Job(id="ok", created_at=1., kind="diloco_member", status="running", params={"run": "r"}),
                Job(id="bad", created_at=1., kind="diloco_member", status="failed", error="Could not reach the owner.", params={"run": "r"}),
                Job(id="other", created_at=1., kind="pretrain", status="failed", error="x")]
        runner = types.SimpleNamespace(lock=__import__("threading").RLock(), jobs=jobs)
        self.assertEqual(studio_facts._member_failures(runner), [{"id": "bad", "run": "r", "error": "Could not reach the owner."}])

    def test_a_machine_whose_member_job_is_starting_is_joining(self):
        self.assertEqual(self.row({"box": peer(member_running=True)})["state"], "joining")

    def test_the_coordinator_s_own_word_wins_once_it_has_heard_from_the_machine(self):
        self.assertEqual(self.row({"box": peer(blocking={"label": "x", "eta_s": 1.})}, self.snap("training"))["state"], "training")
        behind = self.row(None, self.snap("behind"))
        self.assertEqual((behind["state"], behind["reason"]), ("behind", "Missed 1 round in a row."))

    def test_every_row_carries_a_reason_and_a_blocking_job_field(self):
        for row in runs.member_rows(self.run, None, ""):
            self.assertIn("reason", row)
            self.assertIn("blocking_job", row)

    def test_a_machine_that_never_joined_a_finished_run_keeps_why(self):
        self.box.peer_job, self.box.state, self.box.reason = "", "unplaced", runs.UNPLACED
        row = self.row(ended="finished")
        self.assertEqual((row["state"], row["reason"]), ("never_joined", runs.UNPLACED))

    def test_peer_facts_read_the_machine_s_queue_in_the_observation_store(self):
        import studio_facts
        cluster = types.SimpleNamespace(is_down=lambda node: False)
        queue = [{"id": "other", "status": "running", "label": "Version 1", "eta_s": 600.}, {"id": "job-box", "status": "waiting", "label": "m"}]
        with mock.patch.object(studio_facts, "queue_facts", return_value=(queue, 3.)), mock.patch.object(studio_facts, "limits_facts", return_value=None), \
                mock.patch.object(studio_facts, "member_failures", return_value=[]):
            facts = runs.peer_facts(self.run, cluster)
        self.assertEqual(facts, {"box": {"down": False, "slow": False, "member_running": False, "blocking": {"label": "Version 1", "eta_s": 600.}, "limits": None}})
        with mock.patch.object(studio_facts, "queue_facts", return_value=([], None)), mock.patch.object(studio_facts, "limits_facts", return_value=None):
            self.assertEqual(runs.peer_facts(self.run, cluster), {})


def member_job(status="waiting", **params):
    return Job(id="m1", created_at=1., kind="diloco_member", label="Version 7 across 2 machines (member)", status=status,
               params={"coordinator_url": "https://mikes-mac.tail.ts.net", "run": "run7", "node": "beast", "epoch": 0, **params})


def running_job(label="Version 1"):
    return types.SimpleNamespace(id="j1", kind="pretrain", label=label, status="running")


RUN = {"id": "run7", "label": "Version 7", "owner": "Michael's Mac", "round": 4, "rounds": 50,
       "members": [{"node": "beast", "state": "training", "reason": "", "blocking_job": None, "live_tokens_per_second": 1200., "tokens_per_second": None}]}


class MemberSide(unittest.TestCase):
    """What this computer shows for a run it was added to."""

    def test_an_idle_computer_is_invited_and_joins_by_itself(self):
        row = membership.invite_row(member_job(), None, None, None)
        self.assertEqual((row["state"], row["run_label"], row["can_pause_and_join"]), ("invited", "Version 7 across 2 machines", False))
        self.assertIn("joins by itself", row["reason"])
        self.assertEqual(row["owner"], "mikes-mac")

    def test_a_computer_busy_with_its_own_job_is_waiting_and_can_pause_it_and_join(self):
        row = membership.invite_row(member_job(), running_job(), 720., RUN)
        self.assertEqual((row["state"], row["blocking_job"], row["can_pause_and_join"]), ("waiting", {"label": "Version 1", "eta_s": 720.}, True))
        self.assertEqual(row["reason"], "Busy with 'Version 1', about 12 min left. It joins when that finishes.")
        self.assertEqual(row["owner"], "Michael's Mac")

    def test_the_running_member_job_is_joining_until_the_owner_reports_it_training(self):
        job = member_job("running")
        self.assertEqual(membership.invite_row(job, job, None, None)["state"], "joining")
        row = membership.invite_row(job, job, None, RUN)
        self.assertEqual((row["state"], row["round"], row["rounds"], row["tokens_per_second"]), ("training", 4, 50, 1200.))
        self.assertFalse(row["can_pause_and_join"])

    def test_only_live_member_jobs_are_invites(self):
        jobs = [member_job(), member_job("done"), member_job("running"), running_job()]
        self.assertEqual([j.status for j in membership.member_jobs(jobs)], ["waiting", "running"])

    def test_a_pause_request_is_kept_once_per_computer_and_run_and_goes_when_nothing_runs(self):
        self.addCleanup(membership.REQUESTS.clear)
        first = membership.ask_here("beast", "Version 7")
        again = membership.ask_here("beast", "Version 7")
        self.assertEqual(first["id"], again["id"])
        shown = membership.pending_requests(running_job(), 60.)
        self.assertEqual([(r["from_node"], r["blocking_job"]) for r in shown], [("beast", {"label": "Version 1", "eta_s": 60.})])
        self.assertEqual(membership.pending_requests(None, None), [])

    def test_leaving_tells_the_owner_and_survives_one_that_does_not_answer(self):
        FakeClient.calls = []
        self.assertTrue(membership.tell_owner_left(member_job(), clients()))
        self.assertEqual(FakeClient.calls, [("https://mikes-mac.tail.ts.net", "POST", "/distributed/runs/run7/members/beast/stop", {})])
        self.assertFalse(membership.tell_owner_left(member_job(), clients(**{"https://mikes-mac.tail.ts.net": PeerError("down")})))


class Routes(unittest.TestCase):
    """The invite, pause-request and leave routes over FakeServices, with a fake runner and fake peers."""

    def setUp(self):
        self.services = FakeServices(step_seconds=.005)
        self.yielded = []
        self.services.runner = types.SimpleNamespace(yield_running=lambda to=None: self.yielded.append(to) or True)
        with mock.patch.object(api_app, "WEB_OUT", Path(tempfile.gettempdir()) / "no-such-web-out"):
            self.client = self.enterContext(TestClient(create_app(self.services, open=True, hosts=ANY)))
        self.addCleanup(membership.REQUESTS.clear)
        FakeClient.calls = []

    def queue(self, *, running=None, up_next=()):
        state = self.services.queue()
        state.job, state.up_next = running, list(up_next)
        state.eta_s = 720. if running is not None else None

    def invites(self):
        return self.client.get("/distributed/invites").json()

    def test_nothing_is_listed_when_no_run_added_this_computer(self):
        self.queue()
        self.assertEqual(self.invites(), {"invites": [], "pause_requests": []})

    def test_an_invite_behind_a_local_job_is_waiting_with_that_job_and_its_time_left(self):
        self.queue(running=running_job(), up_next=[member_job()])
        invite = self.invites()["invites"][0]
        self.assertEqual((invite["job_id"], invite["state"], invite["blocking_job"], invite["can_pause_and_join"]),
                         ("m1", "waiting", {"label": "Version 1", "eta_s": 720.}, True))

    def test_pause_and_join_steps_the_running_job_aside_for_the_member_job(self):
        self.queue(running=running_job(), up_next=[member_job()])
        self.assertEqual(self.client.post("/distributed/invites/m1/pause-and-join").status_code, 200)
        self.assertEqual(self.yielded, ["m1"])
        self.assertEqual(self.client.post("/distributed/invites/nope/pause-and-join").status_code, 404)

    def test_pause_and_join_is_refused_when_the_member_job_already_runs_or_the_queue_cannot_yield(self):
        self.queue(running=member_job("running"))
        self.assertEqual(self.client.post("/distributed/invites/m1/pause-and-join").json()["detail"], "It is already running.")
        self.queue(running=running_job(), up_next=[member_job()])
        self.services.runner = types.SimpleNamespace(yield_running=lambda to=None: False)
        self.assertEqual(self.client.post("/distributed/invites/m1/pause-and-join").status_code, 409)
        del self.services.runner
        self.assertEqual(self.client.post("/distributed/invites/m1/pause-and-join").status_code, 409)

    def test_leave_tells_the_owner_and_takes_the_member_job_out_of_the_queue(self):
        self.queue(up_next=[member_job()])
        removed = []
        with mock.patch.object(membership, "PeerClient", clients()), mock.patch.object(self.services, "remove", side_effect=removed.append):
            answer = self.client.post("/distributed/invites/m1/leave")
        self.assertEqual(answer.status_code, 200)
        self.assertEqual(FakeClient.calls, [("https://mikes-mac.tail.ts.net", "POST", "/distributed/runs/run7/members/beast/stop", {})])
        self.assertEqual(removed, ["m1"])
        self.assertEqual(self.client.post("/distributed/invites/nope/leave").status_code, 404)

    def test_leave_still_works_when_the_owner_does_not_answer(self):
        self.queue(up_next=[member_job()])
        removed = []
        with mock.patch.object(membership, "PeerClient", clients(**{"https://mikes-mac.tail.ts.net": ConnectionRefusedError()})), \
                mock.patch.object(self.services, "remove", side_effect=removed.append):
            self.assertEqual(self.client.post("/distributed/invites/m1/leave").status_code, 200)
        self.assertEqual(removed, ["m1"])

    def test_a_pause_request_is_shown_and_never_acted_on_until_accepted(self):
        self.queue(running=running_job())
        sent = self.client.post("/distributed/pause-requests", json={"from_node": "Michael's Mac", "run_label": "Version 7"})
        self.assertEqual(sent.status_code, 201)
        self.assertEqual(self.yielded, [])
        request = self.invites()["pause_requests"][0]
        self.assertEqual((request["from_node"], request["run_label"], request["blocking_job"]["label"]), ("Michael's Mac", "Version 7", "Version 1"))
        self.assertEqual(self.yielded, [])
        self.assertEqual(self.client.post(f"/distributed/pause-requests/{request['id']}/accept").status_code, 200)
        self.assertEqual(self.yielded, [None])
        self.assertEqual(self.invites()["pause_requests"], [])

    def test_a_pause_request_can_be_declined_and_an_unknown_one_is_404(self):
        self.queue(running=running_job())
        sent = self.client.post("/distributed/pause-requests", json={"from_node": "mac", "run_label": ""}).json()
        self.assertEqual(self.client.delete(f"/distributed/pause-requests/{sent['id']}").status_code, 200)
        self.assertEqual(self.yielded, [])
        self.assertEqual(self.client.delete(f"/distributed/pause-requests/{sent['id']}").status_code, 404)
        self.assertEqual(self.client.post("/distributed/pause-requests/nope/accept").status_code, 404)

    def test_asking_a_peer_to_pause_posts_the_request_to_it(self):
        self.queue()
        answer = {"id": "r1", "from_node": "mac", "run_label": "Version 7", "blocking_job": None, "at": time.time()}
        calls = []

        def post_json(path, body=None, **kw):
            calls.append((path, body))
            return answer
        fake = lambda address, **kw: types.SimpleNamespace(post_json=post_json)  # noqa: E731
        with mock.patch.object(membership, "PeerClient", fake):
            sent = self.client.post("/distributed/peers/pause-request", json={"address": "https://beast.tail.ts.net", "run_label": "Version 7"})
        self.assertEqual(sent.status_code, 201)
        self.assertEqual(calls[0][0], "/distributed/pause-requests")
        self.assertEqual(calls[0][1]["run_label"], "Version 7")
        refused = lambda address, **kw: types.SimpleNamespace(post_json=mock.Mock(side_effect=PeerError("beast did not answer.")))  # noqa: E731
        with mock.patch.object(membership, "PeerClient", refused):
            self.assertEqual(self.client.post("/distributed/peers/pause-request", json={"address": "x"}).status_code, 502)
