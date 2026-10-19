"""studio_api.routers.collective: the DiLoCo wire protocol over FastAPI's TestClient, and two members training through it."""
import socket
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import uvicorn

import studio_diloco
import studio_distributed
import studio_jobs
import studio_model
from studio_queue import Job
from tests import diloco_support as support

# Final val loss, int8 vs float16. Two members race in one process (and share its random number generator), so a run is not bit-exact: even
# with the same steps dealt, float16 alone ends anywhere from 3.75 to 3.84 and the gap to int8 measured -0.05 to +0.09 over 25 runs (-0.14
# to +0.02 before the steps were pinned). .25 is about twice that spread, and a broken int8 does not train at all: it stays near the
# starting loss of 4.7, 0.9 away.
MARGIN = .25


def equal_steps():
    """deal_steps hands steps out by each member's measured wall-clock speed, so two threads that differ by a few percent are dealt 5 and 3
    steps in some rounds and the losses move from run to run. The tests that compare losses are about the same steps per member (the
    balancing has its own tests, test_diloco_speed_share)."""
    return mock.patch.object(studio_diloco, "deal_steps", lambda settings, members, step_times: {node: settings.inner_steps for node in members})


class Collective(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.client = self.enterContext(support.TermClient(support.app()))
        self.addCleanup(studio_diloco.COORDINATORS.clear)

    def start(self, run="r1", **changes):
        coordinator = support.build(self.dir / run, run, **changes)
        studio_diloco.COORDINATORS[run] = coordinator
        return coordinator

    def delta_body(self, coordinator, dtype="float32", value=.01):
        return studio_diloco.encode_arrays(support.constant_delta(coordinator, value), dtype)

    def post_delta(self, run, number, generation, epoch=0, node="a", body=b"", **fields):
        params = {"node": node, "generation": generation, "epoch": epoch, "tokens": 10, "steps": 2, **fields}
        return self.client.post(f"/collective/{run}/rounds/{number}/delta", params=params, content=body,
                                headers={"content-type": "application/octet-stream"})


class WireProtocol(Collective):
    def test_a_member_joins_downloads_and_contributes_through_every_route(self):
        coordinator = self.start(inner_steps=2, rounds=2, outer_lr=1.)
        joined = self.client.post("/collective/r1/join", json={"node": "a", "epoch": 3})
        self.assertEqual(joined.status_code, 200, joined.text)
        info = joined.json()
        self.assertEqual((info["node"], info["round"], info["inner_steps"], info["rounds"], info["total_steps"], info["delta_dtype"]),
                         ("a", 0, 2, 2, 4, "float32"))
        self.assertEqual(info["config"]["context"], 32)
        view = self.client.get("/collective/r1/rounds/0", params={"node": "a"}).json()
        self.assertEqual((view["state"], view["member"], view["rank"], view["generation"]), ("open", True, 0, 1))
        self.assertEqual(view["shards"], list(range(coordinator.shard_count)))
        theta = self.client.get("/collective/r1/params", params={"round": 0})
        self.assertEqual(theta.headers["content-type"], "application/octet-stream")
        name = next(iter(coordinator.theta))
        np.testing.assert_array_equal(studio_diloco.decode_arrays(theta.content)[name], coordinator.theta[name])
        shard = self.client.get("/collective/r1/shards/0")
        self.assertEqual(shard.content, coordinator.shard_bytes(0))
        self.assertEqual(self.client.get(f"/collective/r1/shards/{coordinator.shard_count}").status_code, 404)
        before = coordinator.theta[name].copy()
        accepted = self.post_delta("r1", 0, view["generation"], epoch=3, body=self.delta_body(coordinator), loss=2.5)
        self.assertEqual(accepted.status_code, 202, accepted.text)
        np.testing.assert_allclose(coordinator.theta[name], before - .01, rtol=1e-5)
        after = self.client.get("/collective/r1/rounds/0", params={"node": "a", "wait": 1}).json()
        self.assertEqual((after["state"], after["current"]), ("closed", 1))
        self.assertEqual(self.client.get("/collective/r1/rounds/1", params={"node": "a"}).json()["state"], "open")
        self.assertEqual(self.client.post("/collective/r1/leave", json={"node": "a", "generation": 1, "epoch": 3}).status_code, 204)

    def test_half_precision_deltas_are_accepted(self):
        coordinator = self.start(inner_steps=2, delta_dtype="float16", outer_lr=1.)
        self.client.post("/collective/r1/join", json={"node": "a"})
        view = self.client.get("/collective/r1/rounds/0", params={"node": "a"}).json()
        name = next(iter(coordinator.theta))
        before = coordinator.theta[name].copy()
        body = self.delta_body(coordinator, "float16")
        self.assertLess(len(body), len(self.delta_body(coordinator)))
        self.assertEqual(self.post_delta("r1", 0, view["generation"], body=body).status_code, 202)
        np.testing.assert_allclose(coordinator.theta[name], before - .01, atol=1e-5)

    def test_a_stale_generation_round_or_epoch_is_409_and_not_applied(self):
        coordinator = self.start(inner_steps=2, outer_lr=1.)
        self.client.post("/collective/r1/join", json={"node": "a", "epoch": 5})
        generation = self.client.get("/collective/r1/rounds/0", params={"node": "a"}).json()["generation"]
        before = {k: v.copy() for k, v in coordinator.theta.items()}
        body = self.delta_body(coordinator)
        for why, answer in {"generation": self.post_delta("r1", 0, generation + 1, epoch=5, body=body),
                            "round": self.post_delta("r1", 1, generation, epoch=5, body=body),
                            "epoch": self.post_delta("r1", 0, generation, epoch=4, body=body),
                            "stranger": self.post_delta("r1", 0, generation, epoch=5, node="b", body=body)}.items():
            with self.subTest(why):
                self.assertEqual(answer.status_code, 409, answer.text)
                self.assertTrue(answer.json()["detail"])
        self.assertEqual(self.client.post("/collective/r1/join", json={"node": "a", "epoch": 4}).status_code, 409)
        self.assertEqual(coordinator.current.contributions, {})
        for name, value in before.items():
            np.testing.assert_array_equal(coordinator.theta[name], value)
        self.assertEqual(self.client.get("/collective/r1/params", params={"round": 1}).status_code, 409)
        self.assertEqual(self.post_delta("r1", 0, generation, epoch=5, body=body).status_code, 202)
        self.assertEqual(self.post_delta("r1", 0, generation, epoch=5, body=body).status_code, 409)  # a round takes one contribution each

    def test_progress_is_taken_without_weights_and_a_stale_one_is_409(self):
        coordinator = self.start(inner_steps=2, outer_lr=1.)
        self.client.post("/collective/r1/join", json={"node": "a", "epoch": 5})
        generation = self.client.get("/collective/r1/rounds/0", params={"node": "a"}).json()["generation"]
        body = {"node": "a", "generation": generation, "epoch": 5, "round": 0, "phase": "training", "step": 1, "loss": 3.5, "tokens_per_second": 800.}
        taken = self.client.post("/collective/r1/progress", json=body)
        self.assertEqual((taken.status_code, taken.json()), (200, {"accepted": True, "term": 1}))
        self.assertEqual((coordinator.reports["a"]["step"], coordinator.reports["a"]["loss"]), (1, 3.5))
        for why, change in {"generation": {"generation": generation + 1}, "epoch": {"epoch": 4}, "round": {"round": 1},
                            "stranger": {"node": "b"}}.items():
            with self.subTest(why):
                answer = self.client.post("/collective/r1/progress", json={**body, **change})
                self.assertEqual(answer.status_code, 409, answer.text)
        self.assertEqual(self.client.post("/collective/r1/progress", json={**body, "phase": "sleeping"}).status_code, 422)
        self.assertEqual(self.client.post("/collective/nope/progress", json=body).status_code, 404)
        self.assertEqual(coordinator.reports["a"]["step"], 1)  # the refused ones changed nothing

    def test_a_progress_report_for_a_closed_round_is_409_but_a_waiting_one_is_taken(self):
        coordinator = self.start(inner_steps=2, outer_lr=1.)
        self.client.post("/collective/r1/join", json={"node": "a", "epoch": 0})
        generation = self.client.get("/collective/r1/rounds/0", params={"node": "a"}).json()["generation"]
        self.assertEqual(self.post_delta("r1", 0, generation, body=self.delta_body(coordinator)).status_code, 202)
        body = {"node": "a", "generation": generation, "epoch": 0, "round": 0, "phase": "uploading", "step": 2}
        self.assertEqual(self.client.post("/collective/r1/progress", json=body).status_code, 409)
        self.assertEqual(self.client.post("/collective/r1/progress", json={**body, "phase": "waiting", "round": 1}).status_code, 200)

    def test_the_params_route_notes_who_took_theta_on_the_timeline(self):
        coordinator = self.start(inner_steps=2)
        self.client.post("/collective/r1/join", json={"node": "a"})
        self.client.get("/collective/r1/params", params={"round": 0, "node": "a"})
        self.assertIn("theta_taken", coordinator.current.marks["a"])

    def test_a_malformed_delta_is_400(self):
        coordinator = self.start(inner_steps=2)
        self.client.post("/collective/r1/join", json={"node": "a"})
        generation = self.client.get("/collective/r1/rounds/0", params={"node": "a"}).json()["generation"]
        self.assertEqual(self.post_delta("r1", 0, generation, body=b"junk").status_code, 400)
        wrong = studio_diloco.encode_arrays({"nope": np.zeros(2)}, "float32")
        self.assertEqual(self.post_delta("r1", 0, generation, body=wrong).status_code, 400)
        self.assertEqual(coordinator.current.contributions, {})

    def test_an_unknown_or_malformed_run_is_refused(self):
        self.assertEqual(self.client.post("/collective/nope/join", json={"node": "a"}).status_code, 404)
        self.assertEqual(self.client.get("/collective/nope/rounds/0").status_code, 404)
        self.assertEqual(self.client.post("/collective/bad run/join", json={"node": "a"}).status_code, 422)


class Training(Collective):
    def train(self, members, run, wrap=lambda peer: peer, **changes):
        """Run `members` member threads against the coordinator over the routes; return the finished coordinator."""
        coordinator = self.start(run, **changes)
        errors = []

        def member(index):
            try:
                studio_diloco.run_member(wrap(support.Peer(self.client)), run, f"node{index}", 0, self.dir / f"{run}-m{index}", self.dir / f"{run}-out{index}",
                                         lambda event: None, threading.Event(), wait=.5)
            except Exception as exc:  # noqa: BLE001 - reported below
                errors.append(exc)
        threads = [threading.Thread(target=member, args=(i,)) for i in range(members)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(240)
        self.assertEqual(errors, [])
        self.assertTrue(coordinator.finished)
        return coordinator

    def test_members_report_their_progress_and_the_timeline_fills_in(self):
        coordinator = self.train(1, "live", inner_steps=3, rounds=2)
        report = coordinator.reports["node0"]
        self.assertIn(report["phase"], studio_diloco.PHASES)
        timeline = coordinator.snapshot()["timeline"]
        self.assertEqual([line["round"] for line in timeline], [0, 1])
        for line in timeline:
            lane = line["lanes"]["node0"]
            self.assertTrue(lane["counted"])
            marks = [lane[name] for name in studio_diloco.MARKS if lane[name] is not None]  # a report is best effort: one may be missing
            self.assertEqual(marks, sorted(marks))
            self.assertIsNotNone(lane["delta_received"])
        self.assertEqual(coordinator.snapshot()["nodes"]["node0"]["progress"]["steps"], 3)

    def test_a_failing_progress_report_never_stops_a_member(self):
        from studio_peer import PeerError

        class Refusing:
            """The routes, except that every progress report fails: refused, then unreachable."""

            def __init__(self, peer):
                self.peer, self.reports = peer, 0

            def __getattr__(self, name):
                return getattr(self.peer, name)

            def post_json(self, path, body=None, params=None, timeout=None):
                if path.endswith("/progress"):
                    self.reports += 1
                    raise PeerError("409 " + path, status=409, detail="refused") if self.reports % 2 else OSError("unreachable")
                return self.peer.post_json(path, body, params, timeout)
        peers = []
        coordinator = self.train(1, "refused", wrap=lambda peer: peers.append(Refusing(peer)) or peers[-1], inner_steps=3, rounds=2)
        self.assertTrue(coordinator.finished)
        self.assertEqual(len(coordinator.history), 2)
        self.assertGreater(peers[0].reports, 0)  # it did try
        self.assertEqual(coordinator.reports, {})

    def test_report_never_raises(self):
        member = studio_diloco.Member(types.SimpleNamespace(post_json=lambda *a, **k: 1 / 0), "r", "n", 0, self.dir / "s", threading.Event())
        member.info, member.report_round = {"generation": 1}, 0
        self.assertFalse(member.report())

    def test_two_members_over_the_routes_beat_one_member_after_the_same_steps_per_member(self):
        shape = dict(inner_steps=4, rounds=6, outer_lr=.7, outer_momentum=.9, min_members=1, delta_dtype="float16")
        with equal_steps():
            one = self.train(1, "one", **shape)
            two = self.train(2, "two", **{**shape, "min_members": 2})
        self.assertEqual([len(r["contributed"]) for r in two.history], [2] * 6)
        self.assertEqual([r["steps"] for r in two.history][0], {"node0": 4, "node1": 4})
        losses = (one.history[-1]["val_loss"], two.history[-1]["val_loss"])
        print(f"final validation loss after 6 rounds x 4 steps per member: 1 member {losses[0]:.4f}, 2 members {losses[1]:.4f}")
        self.assertLess(losses[1], losses[0])
        self.assertLess(two.history[-1]["val_loss"], two.history[0]["val_loss"])
        first = two.history[0]["generation"]
        self.assertTrue(all(r["generation"] == first for r in two.history))
        self.assertEqual(sorted(p.name for p in (self.dir / "two").glob("last.npz")), ["last.npz"])
        self.assertFalse((self.dir / "two-m0" / "data" / "remote_shards" / "two").exists())  # members delete their shard cache when the run ends

    def test_int8_deltas_end_within_a_margin_of_the_float16_run_and_cost_fewer_bytes(self):
        shape = dict(inner_steps=4, rounds=6, outer_lr=.7, outer_momentum=.9, min_members=2)
        with equal_steps():
            half = self.train(2, "half", delta_dtype="float16", **shape)
            quarter = self.train(2, "quarter", delta_dtype="int8", **shape)
        losses = (half.history[-1]["val_loss"], quarter.history[-1]["val_loss"])
        sent = (sum(r["delta_bytes"] for r in half.history), sum(r["delta_bytes"] for r in quarter.history))
        print(f"2 members, 6 rounds x 4 steps: final val loss float16 {losses[0]:.4f}, int8 {losses[1]:.4f} (gap {losses[1] - losses[0]:+.4f}); "
              f"delta bytes float16 {sent[0]}, int8 {sent[1]} ({sent[1] / sent[0]:.0%}); "
              f"round seconds float16 {sum(r['seconds'] for r in half.history):.2f}, int8 {sum(r['seconds'] for r in quarter.history):.2f}")
        self.assertEqual({r["delta_dtype"] for r in quarter.history}, {"int8"})
        self.assertLess(abs(losses[1] - losses[0]), MARGIN)
        self.assertLess(quarter.history[-1]["val_loss"], quarter.history[0]["val_loss"])
        self.assertGreater(sent[0], 0)
        self.assertLess(sent[1], sent[0])


class OwnerJob(unittest.TestCase):
    """The owner's pretrain_distributed job end to end: prepare() starts the coordinator in the app, the handler is its local
    member and talks to it over a real socket with studio_peer.PeerClient."""

    def test_prepare_then_the_handler_trains_a_run_over_http_and_leaves_a_loadable_checkpoint(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            sock = socket.socket()
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
            sock.close()
            server = uvicorn.Server(uvicorn.Config(support.app(), host="127.0.0.1", port=port, log_level="error"))
            thread = threading.Thread(target=server.run, daemon=True)
            thread.start()
            self.addCleanup(thread.join, 10)
            self.addCleanup(setattr, server, "should_exit", True)
            self.addCleanup(studio_diloco.COORDINATORS.clear)
            self.addCleanup(studio_distributed.RUNS.clear)  # prepare() lists the run for the Cluster tab; other tests expect none
            services = types.SimpleNamespace(root=Path(tmp), workspace=types.SimpleNamespace(run_dir=Path(tmp) / "runs"), examples=lambda p: support.EXAMPLES)
            job = Job("pretrain_distributed", params={"config": support.CONFIG, "inner_steps": 3, "rounds": 2, "shards": 4, "api_port": port,
                                                      "outer_lr": .7, "outer_momentum": .9})
            context = studio_jobs.prepare(job, services)
            self.assertIn(job.id, studio_diloco.COORDINATORS)
            self.assertEqual(job.params["coordinator_url"], f"http://127.0.0.1:{port}")
            events = []
            studio_jobs.HANDLERS["pretrain_distributed"](job, types.SimpleNamespace(root=context["root"], run_dir=context["run_dir"]),
                                                         events.append, threading.Event(), threading.Event())
            coordinator = studio_diloco.COORDINATORS[job.id]
            self.assertTrue(coordinator.finished)
            self.assertEqual(len(coordinator.history), 2)
            self.assertTrue(any(e.get("type") == "metric" for e in events))  # the local member's training shows on the job
            checkpoint = studio_model.load_checkpoint(Path(job.params["output_dir"]) / "last.npz")
            self.assertEqual(checkpoint["step"], 6)
            self.assertFalse((Path(tmp) / "data" / "remote_shards" / job.id).exists())
            coordinator.release()


if __name__ == "__main__":
    unittest.main()
