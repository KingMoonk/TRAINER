"""What a member downloads and how fast it can be stopped (#568): a round's shards follow its steps, a shard is built once and
streamed, a finished run refuses weights and shards (410), and a stop ends a download within a block."""
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

import studio_diloco
import studio_peer
from tests import diloco_support as support


class Counting(support.Peer):
    def __init__(self, client):
        super().__init__(client)
        self.shards = []

    def get_bytes(self, path, params=None, timeout=None):
        if "/shards/" in path:
            self.shards.append(int(path.rsplit("/", 1)[1]))
        return super().get_bytes(path, params, timeout)


class Directory(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)


class Volume(Directory):
    def test_a_member_trains_on_a_few_of_its_dealt_shards_not_all_of_them(self):
        coordinator = support.build(self.dir / "c", "vol", shards=8, inner_steps=1, rounds=2, replicas=0)
        studio_diloco.COORDINATORS["vol"] = coordinator
        self.addCleanup(studio_diloco.COORDINATORS.pop, "vol", None)
        with TestClient(support.app()) as client:
            peer = Counting(client)
            studio_diloco.run_member(peer, "vol", "solo", 0, self.dir, self.dir / "m", lambda event: None, threading.Event(), wait=.2)
        self.assertTrue(coordinator.finished)
        self.assertEqual(coordinator.shard_count, 8)
        self.assertLessEqual(len(peer.shards), 2)  # a step draws 2 windows: one shard covers a round; it used to fetch all 8 before step 1

    def test_a_second_run_of_the_same_data_downloads_no_shard_again(self):
        fetched = []
        for run in ("first", "second"):
            coordinator = support.build(self.dir / run, run, shards=8, inner_steps=1, rounds=2, replicas=0)
            studio_diloco.COORDINATORS[run] = coordinator
            self.addCleanup(studio_diloco.COORDINATORS.pop, run, None)
            with TestClient(support.app()) as client:
                peer = Counting(client)
                studio_diloco.run_member(peer, run, "solo", 0, self.dir, self.dir / f"m{run}", lambda event: None, threading.Event(), wait=.2)
            fetched.append(len(peer.shards))
        self.assertGreater(fetched[0], 0)
        self.assertEqual(fetched[1], 0)  # the same shards, held by hash under data/shard_cache from the first run

    def test_the_shards_of_a_round_are_the_same_on_every_run_of_the_deal(self):
        self.assertEqual(studio_diloco.round_order(1, 0, 3, [5, 2, 9, 1]), studio_diloco.round_order(1, 0, 3, [1, 2, 5, 9]))


class Serving(Directory):
    def test_a_shard_is_built_once_and_every_request_gets_the_same_bytes_from_its_file(self):
        coordinator = support.build(self.dir / "c", "srv", shards=4)
        studio_diloco.COORDINATORS["srv"] = coordinator
        self.addCleanup(studio_diloco.COORDINATORS.pop, "srv", None)
        with TestClient(support.app()) as client, mock.patch.object(studio_diloco, "shard_blob", wraps=studio_diloco.shard_blob) as built:
            first = client.get("/collective/srv/shards/1", params={"term": 1})
            second = client.get("/collective/srv/shards/1", params={"term": 1})
        self.assertEqual((first.status_code, second.status_code), (200, 200))
        self.assertEqual(first.content, second.content)
        self.assertEqual(built.call_count, 1)
        self.assertTrue(coordinator.shard_path(1).is_file())

    def test_a_stopped_run_refuses_weights_and_shards_with_410(self):
        coordinator = support.build(self.dir / "c", "gone", shards=4)
        studio_diloco.COORDINATORS["gone"] = coordinator
        self.addCleanup(studio_diloco.COORDINATORS.pop, "gone", None)
        coordinator.stop()
        with TestClient(support.app()) as client:
            self.assertEqual(client.get("/collective/gone/shards/0", params={"term": 1}).status_code, 410)
            self.assertEqual(client.get("/collective/gone/params", params={"term": 1, "round": 0}).status_code, 410)


class Client:
    interruptible = True

    def __init__(self, body):
        self.body = body

    def get_json(self, path, params=None, timeout=None):
        return {"state": "open", "member": True, "generation": 0, "rank": 0, "deadline_s": 10., "shards": [0, 1, 2], "steps": 2}

    def get_bytes(self, path, params=None, timeout=None, cancel=None):
        return self.body(path, cancel)


class Member(Directory):
    def member(self, client, stop=None):
        member = studio_diloco.Member(client, "r", "n", 0, self.dir / "s", stop or threading.Event(), progress=False, wait=.1, patience=1.)
        member.info = {"seed": 1, "rounds": 3, "inner_steps": 2, "delta_dtype": "float32", "round": 0, "vocab_size": 300,
                       "config": {"batch_size": 2}, "shard_count": 4, "term": 1}
        member.term = 1
        return member

    def test_a_410_for_the_weights_means_the_run_is_over_and_the_member_leaves(self):
        def gone(path, cancel):
            raise studio_peer.PeerError("gone", status=410, detail="Run r was stopped.")
        member = self.member(Client(gone))
        self.assertIsNone(member._take(0))
        self.assertTrue(member.finished)

    def test_a_stop_during_a_download_ends_it_within_a_block(self):
        stop, chunks = threading.Event(), []

        def slow(path, cancel):
            while not cancel():  # a body that never ends, a block at a time
                chunks.append(1)
                time.sleep(.02)
            raise studio_peer.Cancelled("cancelled")
        member = self.member(Client(slow), stop)
        timer = threading.Timer(.2, stop.set)
        timer.start()
        self.addCleanup(timer.cancel)
        began = time.monotonic()
        self.assertIsNone(member._take(0))
        self.assertLess(time.monotonic() - began, 2.)
        self.assertGreater(len(chunks), 0)

    def test_the_client_stops_reading_a_body_when_cancelled(self):
        reads = []

        class Body:
            headers = {}
            status = 200

            def read(self, size=-1):
                reads.append(size)
                return b"x" * 8

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        client = studio_peer.PeerClient("http://node:1", opener=lambda request, timeout: Body())
        with self.assertRaises(studio_peer.Cancelled):
            client.get_bytes("x", cancel=lambda: len(reads) >= 3)
        self.assertEqual(reads, [studio_peer.BLOCK] * 3)


if __name__ == "__main__":
    unittest.main()
