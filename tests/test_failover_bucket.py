"""The bucket as the failover's tie-breaker (#362): create-if-absent on the stores, and ClusterHost.tie_break over it. FolderStore stands in
for the bucket, and a store that always fails for an unreachable one. Nothing contacts a real bucket or opens a window."""
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

import studio_publish
from studio_failover import Claim, ClusterHost

KEY = "runs/r1/term/2"


class Unreachable:
    """A store whose every call fails, as a bucket nobody can reach."""
    def create_if_absent(self, key, data):
        raise OSError("unreachable")

    def read(self, key):
        raise OSError("unreachable")


class ClientError(Exception):
    def __init__(self, code, status):
        super().__init__(code)
        self.response = {"Error": {"Code": code}, "ResponseMetadata": {"HTTPStatusCode": status}}


class FakeClient:
    def __init__(self, error=None):
        self.error, self.calls = error, []

    def put_object(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error


def host(store, name="n1", down=()):
    cluster = SimpleNamespace(hostname=name, lease_s=30., store=lambda: store)
    found = ClusterHost(cluster)
    found.is_down = lambda node: node in down
    return found


class CreateIfAbsent(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.store = studio_publish.FolderStore(self._dir.name)

    def test_the_first_writer_wins_and_the_marker_keeps_its_bytes(self):
        self.assertTrue(self.store.create_if_absent(KEY, b"a"))
        self.assertFalse(self.store.create_if_absent(KEY, b"b"))
        self.assertEqual(self.store.read(KEY), b"a")
        self.assertIsNone(self.store.read("runs/r1/term/3"))
        self.assertEqual(self.store.list(), [])
        self.assertEqual([p.name for p in Path(self._dir.name, "runs/r1/term").iterdir()], ["2"])

    def test_racing_threads_give_exactly_one_winner(self):
        results, gate = [], threading.Barrier(8)

        def go(i):
            gate.wait()
            results.append(self.store.create_if_absent(KEY, str(i).encode()))

        threads = [threading.Thread(target=go, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(results.count(True), 1)

    def test_a_key_that_is_not_a_path_of_plain_segments_is_refused(self):
        with self.assertRaises(ValueError):
            self.store.create_if_absent("../x", b"a")

    def test_the_bucket_store_sends_if_none_match_and_reads_a_412_as_losing(self):
        config = {"endpoint": "http://x", "bucket": "b", "key_id": "k", "secret": "s", "prefix": "p"}
        client = FakeClient()
        self.assertTrue(studio_publish.S3Store(config, client=client).create_if_absent(KEY, b"a"))
        self.assertEqual((client.calls[0]["IfNoneMatch"], client.calls[0]["Key"]), ("*", "p/" + KEY))
        for error in (ClientError("PreconditionFailed", 412), ClientError("ConditionalRequestConflict", 409)):
            self.assertFalse(studio_publish.S3Store(config, client=FakeClient(error)).create_if_absent(KEY, b"a"))
        with self.assertRaises(ClientError):
            studio_publish.S3Store(config, client=FakeClient(ClientError("AccessDenied", 403))).create_if_absent(KEY, b"a")


class TieBreak(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.store = studio_publish.FolderStore(self._dir.name)

    def test_two_claimants_racing_on_a_term_one_wins_and_the_other_follows(self):
        first, second = host(self.store, "n1"), host(self.store, "n2")
        self.assertTrue(first.tie_break(Claim("r1", "n1", 2, 5, 0)))
        self.assertFalse(second.tie_break(Claim("r1", "n2", 2, 5, 1)))
        self.assertTrue(first.tie_break(Claim("r1", "n1", 2, 5, 0)))  # a retry after a failed takeover finds its own marker
        self.assertTrue(second.tie_break(Claim("r1", "n2", 3, 5, 1)))  # the next term is a new race

    def test_an_unreachable_store_leaves_the_rule_to_decide(self):
        with self.assertLogs("studio_failover", "WARNING"):
            self.assertTrue(host(Unreachable()).tie_break(Claim("r1", "n1", 2, 5)))

    def test_no_store_leaves_the_rule_to_decide(self):
        self.assertTrue(host(None).tie_break(Claim("r1", "n1", 2, 5)))

    def test_a_winner_that_is_down_or_unreadable_does_not_hold_the_run(self):
        self.store.create_if_absent(KEY, b'{"node": "n0"}')
        self.assertTrue(host(self.store, "n2", down={"n0"}).tie_break(Claim("r1", "n2", 2, 5)))
        self.store.create_if_absent("runs/r1/term/4", b"not json")
        self.assertTrue(host(self.store, "n2").tie_break(Claim("r1", "n2", 4, 5)))


if __name__ == "__main__":
    unittest.main()
