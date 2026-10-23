"""Parallel transfers on the collective routes (#584): Range on the big bodies, a change posted in parts, the rates in the round's health."""
import hashlib
import socket
import threading
import unittest

import numpy as np
import uvicorn

import studio_diloco
import studio_run_health
from studio_peer import PeerClient
from studio_upload_admission import UploadAdmission
from tests import diloco_support as support
from tests.test_api_collective import Collective


class Ranged(Collective):
    def test_weights_and_state_answer_a_range_with_206_and_the_whole_without(self):
        coordinator = self.start(inner_steps=2, rounds=3, outer_lr=1.)
        self.client.post("/collective/r1/join", json={"node": "a"})
        whole = self.client.get("/collective/r1/params", params={"round": 0})
        self.assertEqual((whole.status_code, whole.headers["accept-ranges"]), (200, "bytes"))
        size = len(whole.content)
        part = self.client.get("/collective/r1/params", params={"round": 0}, headers={"range": "bytes=10-4999"})
        self.assertEqual(part.status_code, 206)
        self.assertEqual(part.headers["content-range"], f"bytes 10-4999/{size}")
        self.assertEqual(part.content, whole.content[10:5000])
        self.assertEqual(part.headers[studio_diloco.TERM_HEADER], "1")
        tail = self.client.get("/collective/r1/params", params={"round": 0}, headers={"range": f"bytes={size - 7}-"})
        self.assertEqual(tail.content, whole.content[-7:])
        self.assertEqual(self.client.get("/collective/r1/params", params={"round": 0}, headers={"range": f"bytes={size}-"}).status_code, 416)
        self.assertEqual(self.client.get("/collective/r1/params", params={"round": 5}, headers={"range": "bytes=0-9"}).status_code, 409)
        generation = self.client.get("/collective/r1/rounds/0", params={"node": "a"}).json()["generation"]
        self.assertEqual(self.post_delta("r1", 0, generation, body=self.delta_body(coordinator)).status_code, 202)
        state = self.client.get("/collective/r1/state", params={"round": 1})
        again = self.client.get("/collective/r1/state", params={"round": 1}, headers={"range": "bytes=0-99"})
        self.assertEqual((again.status_code, again.content), (206, state.content[:100]))

    def test_a_shard_file_answers_a_range(self):
        coordinator = self.start()
        self.client.post("/collective/r1/join", json={"node": "a"})
        whole = self.client.get("/collective/r1/shards/0").content
        self.assertEqual(whole, coordinator.shard_bytes(0))
        part = self.client.get("/collective/r1/shards/0", headers={"range": "bytes=5-99"})
        self.assertEqual((part.status_code, part.content), (206, whole[5:100]))
        self.assertEqual(part.headers["content-range"], f"bytes 5-99/{len(whole)}")
        self.assertEqual(part.headers[studio_diloco.TERM_HEADER], "1")


class Parts(Collective):
    def open_round(self, **changes):
        coordinator = self.start(inner_steps=2, rounds=3, outer_lr=1., **changes)
        self.client.post("/collective/r1/join", json={"node": "a", "epoch": 2})
        generation = self.client.get("/collective/r1/rounds/0", params={"node": "a"}).json()["generation"]
        return coordinator, generation

    def post_part(self, body, index, count, total, whole, generation, number=0, part_hash=None, **fields):
        params = {"node": "a", "generation": generation, "epoch": 2, "tokens": 10, "steps": 2, "count": count, "part": index, "total": total,
                  "sha256": whole, "part_sha256": part_hash or hashlib.sha256(body).hexdigest(), **fields}
        return self.client.post(f"/collective/r1/rounds/{number}/delta/parts", params=params, content=body,
                                headers={"content-type": "application/octet-stream"})

    def cut(self, body, size):
        return [body[i:i + size] for i in range(0, len(body), size)]

    def test_parts_arriving_in_any_order_are_one_contribution_once_the_hash_matches(self):
        coordinator, generation = self.open_round()
        name = next(iter(coordinator.theta))
        before = coordinator.theta[name].copy()
        body = self.delta_body(coordinator)
        parts, whole = self.cut(body, 4000), hashlib.sha256(body).hexdigest()
        self.assertGreater(len(parts), 2)
        for index in reversed(range(len(parts))):
            answer = self.post_part(parts[index], index, len(parts), len(body), whole, generation, down_bps=3 * 2 ** 20)
            self.assertEqual(answer.status_code, 202, answer.text)
            if index:
                self.assertEqual(coordinator.current.contributions, {} if coordinator.current else {})
        self.assertEqual(len(coordinator.history), 1)
        np.testing.assert_allclose(coordinator.theta[name], before - .01, rtol=1e-5)
        member = coordinator.history[0]["health"]["members"]["a"]
        self.assertAlmostEqual(member["down_mb_s"], 3.)
        self.assertGreater(member["up_mb_s"], 0.)
        self.assertEqual(coordinator.history[0]["node_bytes"], {"a": len(body)})

    def test_shards_are_still_served_once_a_change_came_in_parts(self):
        coordinator, generation = self.open_round()
        body = self.delta_body(coordinator)
        parts = self.cut(body, 4000)
        self.assertEqual(self.post_part(parts[0], 0, len(parts), len(body), hashlib.sha256(body).hexdigest(), generation).status_code, 202)
        answer = self.client.get("/collective/r1/shards/0/hash", params={"term": 1})
        self.assertEqual(answer.status_code, 200, answer.text)
        self.assertEqual(answer.json()["hash"], hashlib.sha256(coordinator.shard_bytes(0)).hexdigest())
        self.assertEqual(self.client.get("/collective/r1/shards/0").content, coordinator.shard_bytes(0))

    def test_a_part_that_does_not_match_its_hash_is_rejected_and_sent_again(self):
        coordinator, generation = self.open_round()
        body = self.delta_body(coordinator)
        parts, whole = self.cut(body, 4000), hashlib.sha256(body).hexdigest()
        wrong = bytes([parts[0][0] ^ 1]) + parts[0][1:]
        answer = self.post_part(wrong, 0, len(parts), len(body), whole, generation, part_hash=hashlib.sha256(parts[0]).hexdigest())
        self.assertEqual(answer.status_code, 400, answer.text)
        for index, part in enumerate(parts):
            self.assertEqual(self.post_part(part, index, len(parts), len(body), whole, generation).status_code, 202)
        self.assertEqual(len(coordinator.history), 1)

    def test_a_body_that_fails_the_whole_hash_is_not_accepted(self):
        coordinator, generation = self.open_round()
        body = self.delta_body(coordinator)
        parts, wrong = self.cut(body, 4000), hashlib.sha256(b"something else").hexdigest()
        answers = [self.post_part(part, index, len(parts), len(body), wrong, generation).status_code for index, part in enumerate(parts)]
        self.assertEqual(answers[:-1], [202] * (len(parts) - 1))
        self.assertEqual(answers[-1], 400)
        self.assertEqual(coordinator.history, [])
        self.assertEqual(coordinator.current.contributions, {})

    def test_a_stale_change_in_parts_is_409_and_changes_nothing(self):
        coordinator, generation = self.open_round()
        body = self.delta_body(coordinator)
        parts, whole = self.cut(body, 6000), hashlib.sha256(body).hexdigest()
        answers = [self.post_part(part, index, len(parts), len(body), whole, generation + 1).status_code for index, part in enumerate(parts)]
        self.assertEqual(answers[-1], 409)
        self.assertEqual(coordinator.history, [])

    def test_count_zero_only_asks_whether_the_route_exists(self):
        coordinator, generation = self.open_round()
        answer = self.client.post("/collective/r1/rounds/0/delta/parts", params={
            "node": "a", "generation": generation, "epoch": 2, "tokens": 10, "steps": 2, "count": 0, "total": 5, "sha256": "x"}, content=b"")
        self.assertEqual((answer.status_code, answer.json()), (202, {"accepted": True, "term": 1}))
        self.assertEqual(coordinator.history, [])

    def test_the_single_post_reports_its_rates_too(self):
        coordinator, generation = self.open_round()
        self.assertEqual(self.post_delta("r1", 0, generation, epoch=2, body=self.delta_body(coordinator), down_bps=2 ** 20).status_code, 202)
        member = coordinator.history[0]["health"]["members"]["a"]
        self.assertEqual(member["down_mb_s"], 1.)
        self.assertIsNotNone(member["up_mb_s"])


class OverHttp(Collective):
    """A real socket, studio_peer.PeerClient with small parts: the ranged downloads and the upload in parts through the real stack."""

    def serve(self, admission=None):
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        app = support.app()
        if admission is not None:
            app.state.upload_admission = admission
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 10)
        self.addCleanup(setattr, server, "should_exit", True)
        return f"http://127.0.0.1:{port}"

    def test_a_member_downloads_in_ranges_and_uploads_in_parts(self):
        coordinator = self.start(inner_steps=2, rounds=3, outer_lr=1.)
        peer = PeerClient(self.serve(), part=3000, streams=4)
        peer.post_json("/collective/r1/join", {"node": "a", "epoch": 1}, {"term": 0})
        view = peer.get_json("/collective/r1/rounds/0", {"node": "a", "term": 1})
        theta = peer.get_bytes("/collective/r1/params", {"round": 0, "node": "a", "term": 1})
        self.assertGreater(len(theta), 3 * 3000)
        self.assertEqual(theta, coordinator.params_bytes(0))
        self.assertEqual(peer.get_bytes("/collective/r1/shards/0", {"term": 1}), coordinator.shard_bytes(0))
        body = self.delta_body(coordinator)
        peer.post_bytes("/collective/r1/rounds/0/delta", body, {"node": "a", "generation": view["generation"], "epoch": 1, "tokens": 10, "steps": 2,
                                                                 "term": 1, "down_bps": 1000})
        self.assertIs(peer.parts_route, True)
        self.assertEqual(len(coordinator.history), 1)
        self.assertEqual(coordinator.history[0]["node_bytes"], {"a": len(body)})
        self.assertIsNotNone(studio_run_health.summarize(coordinator.history)["rounds"][0]["members"]["a"]["up_mb_s"])

    def test_receiver_grants_are_opt_in_and_submit_the_delta_once(self):
        coordinator = self.start(inner_steps=2, rounds=3, outer_lr=1.)
        peer = PeerClient(self.serve(), part=3000, streams=4, receiver_grants=True)
        peer.post_json("/collective/r1/join", {"node": "a", "epoch": 1}, {"term": 0})
        view = peer.get_json("/collective/r1/rounds/0", {"node": "a", "term": 1})
        body = self.delta_body(coordinator)
        peer.post_bytes("/collective/r1/rounds/0/delta", body, {"node": "a", "generation": view["generation"],
                                                                  "epoch": 1, "tokens": 10, "steps": 2, "term": 1})
        self.assertEqual(len(coordinator.history), 1)
        self.assertEqual(coordinator.history[0]["node_bytes"], {"a": len(body)})
        self.assertEqual(coordinator.history[0]["contributed"], ["a"])
        peer.post_bytes("/collective/r1/rounds/0/delta", body, {"node": "a", "generation": view["generation"],
                                                                  "epoch": 1, "tokens": 10, "steps": 2, "term": 1})
        self.assertEqual(len(coordinator.history), 1)  # a retry after a lost final acknowledgement cannot submit twice

    def test_sender_uses_the_part_size_advertised_by_the_receiver(self):
        coordinator = self.start(inner_steps=2, rounds=3, outer_lr=1.)
        admission = UploadAdmission(max_assembly=1 << 20, max_outstanding=2000, grant_bytes=1000)
        peer = PeerClient(self.serve(admission), part=3000, streams=4, receiver_grants=True)
        peer.post_json("/collective/r1/join", {"node": "a", "epoch": 1}, {"term": 0})
        view = peer.get_json("/collective/r1/rounds/0", {"node": "a", "term": 1})
        body = self.delta_body(coordinator)
        peer.post_bytes("/collective/r1/rounds/0/delta", body, {"node": "a", "generation": view["generation"],
                                                                  "epoch": 1, "tokens": 10, "steps": 2, "term": 1})
        self.assertEqual(coordinator.history[0]["node_bytes"], {"a": len(body)})
        self.assertEqual(admission.outstanding_bytes, 0)
        self.assertEqual(admission.assembly_bytes, 0)


if __name__ == "__main__":
    unittest.main()
