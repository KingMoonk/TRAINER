"""Replicas of a DiLoCo run (docs/distributed.md, "Terms and coordinator failover"): who they are across membership changes, what they
keep (theta, v, diloco.json of the latest round) over three members on the real routes, what they fetch once, and that they clean up."""
import io
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

import studio_diloco
from tests import diloco_support as support


class Selection(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)

    def test_the_lowest_ranked_members_are_the_replicas(self):
        self.assertEqual(studio_diloco.replica_nodes(["a", "b", "c", "d"], 2), ["a", "b"])
        self.assertEqual(studio_diloco.replica_nodes(["a"], 2), ["a"])
        self.assertEqual(studio_diloco.replica_nodes(["a", "b"], 0), [])

    def flags(self, coordinator, nodes, number):
        views = {node: coordinator.round_view(number, node) for node in nodes}
        return {node: (view["rank"], view["replica"]) for node, view in views.items() if view["state"] == "open"}

    def close(self, coordinator, number, nodes):
        generation = coordinator.round_view(number, nodes[0])["generation"]
        for node in nodes:
            coordinator.submit(number, node, generation, 0, 10, 2, support.constant_delta(coordinator))

    def test_replicas_follow_rank_as_members_join_and_leave(self):
        coordinator = support.build(Path(self.tmp.name) / "r", inner_steps=2, rounds=6, max_members=4)
        coordinator.join("a")
        self.assertEqual(self.flags(coordinator, "a", 0), {"a": (0, True)})  # fewer members than R: all of them
        for node in "bcd":
            coordinator.join(node)
        self.close(coordinator, 0, ["a"])
        self.assertEqual(self.flags(coordinator, "abcd", 1), {"a": (0, True), "b": (1, True), "c": (2, False), "d": (3, False)})
        generation = coordinator.round_view(1, "a")["generation"]
        coordinator.leave("a", generation, 0)  # leaves at the boundary: b and c are the lowest ranked then
        self.close(coordinator, 1, ["b", "c", "d"])
        self.assertEqual(self.flags(coordinator, "bcd", 2), {"b": (0, True), "c": (1, True), "d": (2, False)})
        coordinator.drop("b")
        self.close(coordinator, 2, ["c", "d"])
        self.assertEqual(self.flags(coordinator, "cd", 3), {"c": (0, True), "d": (1, True)})

    def test_no_replicas_when_the_run_asks_for_none(self):
        coordinator = support.build(Path(self.tmp.name) / "r", replicas=0)
        coordinator.join("a")
        self.assertFalse(coordinator.round_view(0, "a")["replica"])
        with self.assertRaises(ValueError):
            support.settings(replicas=-1)


class Replication(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.client = self.enterContext(TestClient(support.app()))
        self.addCleanup(studio_diloco.COORDINATORS.clear)

    def run_three(self, rounds=3):
        """Three members over the routes. Each member's calls are logged; a replica's folder is compared with the coordinator's at the
        moment it says it holds a round (the coordinator cannot close that round before this member's delta)."""
        coordinator = support.build(self.dir / "run", "run1", inner_steps=2, rounds=rounds, outer_lr=.7, outer_momentum=.9, min_members=3,
                                    delta_dtype="float16")
        studio_diloco.COORDINATORS["run1"] = coordinator
        calls, dealt, held, errors = {}, {}, [], []

        class Logged(support.Peer):
            def __init__(self, client, node):
                super().__init__(client)
                self.node = node

            def get_json(self, path, params=None, timeout=None):
                found = super().get_json(path, params, timeout)
                if found.get("state") == "open":
                    dealt.setdefault(self.node, set()).update(found["shards"])
                return found

            def get_bytes(self, path, params=None, timeout=None):
                last = path.rsplit("/", 1)[1]
                calls.setdefault(self.node, []).append(int(last) if "/shards/" in path else last)
                return super().get_bytes(path, params, timeout)

        def member(index):
            node, folder = f"node{index}", self.dir / f"m{index}"

            def emit(event):
                text = event.get("text", "")
                if text.startswith("Holding round"):
                    held.append((node, int(text.split()[2]), self.snapshot(coordinator, folder)))
                elif text.startswith("Replica holds"):
                    held.append((node, "prefetched", sorted(p.name for p in (folder / "data" / "replicas" / "run1").iterdir())))
            try:
                studio_diloco.run_member(Logged(self.client, node), "run1", node, 0, folder, self.dir / f"out{index}", emit, threading.Event(), wait=.5)
            except Exception as exc:  # noqa: BLE001 - reported below
                errors.append(exc)
        threads = [threading.Thread(target=member, args=(i,)) for i in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(240)
        self.assertEqual(errors, [])
        self.assertTrue(coordinator.finished)
        return coordinator, calls, dealt, held

    @staticmethod
    def snapshot(coordinator, folder):
        """The replica's three files read now, with the coordinator's own (theta of the open round, v, the diloco.json on its disk)."""
        replica = folder / "data" / "replicas" / "run1"
        return {"theta": studio_diloco.decode_arrays((replica / "theta.npz").read_bytes()),
                "momentum": studio_diloco.decode_arrays((replica / "momentum.npz").read_bytes()),
                "record": (replica / "diloco.json").read_bytes(),
                "files": sorted(p.name for p in replica.iterdir() if not p.name.startswith("validation")),
                "round": coordinator.next_round, "own_theta": {k: v.copy() for k, v in coordinator.theta.items()},
                "own_momentum": {k: v.copy() for k, v in coordinator.momentum.items()},
                "own_record": (coordinator.directory / "diloco.json").read_bytes()}

    def test_the_replicas_copy_equals_the_coordinators_each_round_and_a_non_replica_downloads_nothing_extra(self):
        coordinator, calls, dealt, held = self.run_three(rounds=3)
        rounds = [(node, number) for node, number, _ in held if number != "prefetched"]
        replicas = {node for node, _ in rounds}
        self.assertEqual(len(replicas), 2)  # R = 2 of the 3 members
        for node in replicas:
            self.assertEqual(sorted(number for n, number in rounds if n == node), [1, 2])  # round 0 has nothing closed to keep
        for node, number, found in held:
            if number == "prefetched":
                continue
            with self.subTest(node=node, round=number):
                self.assertEqual(found["round"], number)
                self.assertEqual(found["files"], ["diloco.json", "momentum.npz", "theta.npz"])  # the latest round only
                for name, value in found["own_theta"].items():
                    np.testing.assert_array_equal(found["theta"][name], value)
                for name, value in found["own_momentum"].items():
                    np.testing.assert_array_equal(found["momentum"][name], value)
                self.assertEqual(found["record"], found["own_record"])
        for node in replicas:
            self.assertEqual(calls[node].count("validation"), 1)  # once per run
            self.assertEqual(sorted(s for s in calls[node] if isinstance(s, int)), list(range(coordinator.shard_count)))  # every shard, once
            self.assertEqual(calls[node].count("state"), 2)
        (plain,) = {"node0", "node1", "node2"} - replicas
        self.assertNotIn("state", calls[plain])
        self.assertNotIn("validation", calls[plain])
        self.assertEqual(sorted(s for s in calls[plain] if isinstance(s, int)), sorted(dealt[plain]))  # only what it was dealt

    def test_replicas_hold_the_validation_windows_and_delete_everything_at_the_end(self):
        _, _, _, held = self.run_three(rounds=2)
        done = [files for _, number, files in held if number == "prefetched"]
        self.assertEqual(len(done), 2)
        for files in done:
            self.assertIn("validation.npz", files)
        for index in range(3):
            self.assertFalse((self.dir / f"m{index}" / "data" / "replicas" / "run1").exists())
            self.assertFalse((self.dir / f"m{index}" / "data" / "remote_shards" / "run1").exists())

    def test_validation_is_served_as_windows(self):
        coordinator = support.build(self.dir / "run", "run1", inner_steps=2, rounds=2)
        studio_diloco.COORDINATORS["run1"] = coordinator
        body = self.client.get("/collective/run1/validation")
        self.assertEqual(body.status_code, 200)
        self.assertGreater(len(coordinator.val), 0)
        with np.load(io.BytesIO(body.content)) as archive:
            self.assertEqual(len(archive["offsets"]) - 1, len(coordinator.val))

    def test_the_state_route_serves_the_latest_round_only(self):
        coordinator = support.build(self.dir / "run", "run1", inner_steps=2, rounds=3, outer_lr=.7, outer_momentum=.9)
        studio_diloco.COORDINATORS["run1"] = coordinator
        self.assertEqual(self.client.get("/collective/run1/state", params={"round": 0}).status_code, 404)  # nothing has closed
        self.client.post("/collective/run1/join", json={"node": "a"})
        view = self.client.get("/collective/run1/rounds/0", params={"node": "a"}).json()
        self.assertTrue(view["replica"])
        body = studio_diloco.encode_arrays(support.constant_delta(coordinator), "float32")
        params = {"node": "a", "generation": view["generation"], "epoch": 0, "tokens": 10, "steps": 2}
        self.client.post("/collective/run1/rounds/0/delta", params=params, content=body, headers={"content-type": "application/octet-stream"})
        state = self.client.get("/collective/run1/state", params={"round": 1})
        self.assertEqual(state.status_code, 200)
        momentum, record = studio_diloco.decode_state(state.content)
        name = next(iter(momentum))
        np.testing.assert_array_equal(momentum[name], coordinator.momentum[name])
        self.assertEqual(record, (coordinator.directory / "diloco.json").read_bytes())
        self.assertEqual(self.client.get("/collective/run1/state", params={"round": 0}).status_code, 409)
        self.assertEqual(self.client.get("/collective/run1/state", params={"round": 2}).status_code, 409)
        self.assertEqual(self.client.get("/collective/nope/state", params={"round": 1}).status_code, 404)


class LateWrites(unittest.TestCase):
    def test_a_fetch_that_finishes_after_demotion_or_the_end_of_the_run_does_not_bring_the_folder_back(self):
        for why in ("demoted", "ended"):
            with self.subTest(why), tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
                inside, release = threading.Event(), threading.Event()

                class Slow:
                    def get_bytes(self, path, params=None, timeout=None):
                        inside.set()
                        release.wait(10)
                        return b"validation"
                replica = Path(tmp) / "replicas" / "run1"
                member = studio_diloco.Member(Slow(), "run1", "n", 0, Path(tmp) / "shards", threading.Event(), replica_dir=replica)
                member.info = {"shard_count": 0}
                fetch = threading.Thread(target=member._prefetch, args=(member.replica_epoch,))
                fetch.start()
                self.assertTrue(inside.wait(10))
                member._drop_replica()  # what a demotion does, and what the end of a run does
                release.set()
                fetch.join(10)
                self.assertFalse(replica.exists())
                self.assertFalse(member.prefetched)
                self.assertTrue(member._write_replica(member.replica_epoch, "theta.npz", b"x"))  # a later epoch writes again
                self.assertTrue(replica.exists())


if __name__ == "__main__":
    unittest.main()
