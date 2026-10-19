"""A peer that answers again is shown as answering within a few seconds, and the node says when it was last heard and will be asked next
(studio_remote.Cluster.contact, GET /cluster/nodes). Two in-process nodes, a fake clock and a fake network; no sockets, no sleeping."""
import unittest
from unittest import mock

import studio_cluster
import studio_cluster_queue as cq
import studio_remote
from studio_peer import PeerError
from tests.test_api_cluster import FakePeers, node_payload
from tests import test_cluster_facts


class Recovery(test_cluster_facts.TwoNodes):
    def nodes(self, answering=True):
        """B's GET /cluster/nodes, where A's own answer to that request is `answering` (the poll's network is self.net)."""
        peers = FakePeers(A=node_payload("A") if answering else PeerError("A did not answer"))
        with mock.patch.object(studio_cluster, "fetch_json", peers):
            return {n["address"]: n for n in self.b.client.get("/cluster/nodes").json()}

    def test_a_peer_that_answers_again_is_not_down_after_two_polls_however_long_it_was_gone(self):
        self.sync()
        self.assertFalse(self.b.cluster.is_down("A"))
        self.net.down.add("A")
        for _ in range(12):  # a long outage: the failures in a row reach the longest wait between polls
            self.clock.advance(studio_remote.POLL_EVERY)
            self.b.cluster.poll_once()
        self.assertTrue(self.b.cluster.is_down("A"))
        self.net.down.discard("A")  # A is back
        for _ in range(2):  # the longest wait is two polls
            self.clock.advance(studio_remote.POLL_EVERY)
            self.sync()
        self.assertFalse(self.b.cluster.is_down("A"))
        self.assertTrue(self.b.cluster.nodes()[1].online)
        self.assertEqual(self.b.cluster.health.state("A"), "up")

    def test_the_wait_between_polls_of_a_silent_peer_never_passes_the_ceiling(self):
        self.sync()
        self.net.down.add("A")
        for _ in range(10):
            self.clock.advance(studio_remote.POLL_BACKOFF_MAX)
            self.b.cluster.poll_once()
        _, check = self.b.cluster.contact("A")
        self.assertLessEqual(check, studio_remote.POLL_BACKOFF_MAX)
        before = len(self.net.to("A"))
        self.clock.advance(check)
        self.b.cluster.poll_once()
        self.assertGreater(len(self.net.to("A")), before)  # asked again as soon as the wait was over

    def test_the_machine_row_says_when_the_peer_was_last_heard_and_when_it_is_asked_next(self):
        self.assertEqual(self.b.cluster.contact("A"), (None, None))  # never asked
        self.sync()
        self.clock.advance(2)
        self.assertEqual(self.b.cluster.contact("A"), (2., studio_remote.POLL_EVERY - 2))
        self.net.down.add("A")
        self.clock.advance(30)
        self.b.cluster.poll_once()  # fails: asked again after the wait
        self.clock.advance(3)
        self.assertEqual(self.b.cluster.contact("A"), (35., studio_remote.POLL_BACKOFF_MAX - 3))
        row = self.nodes(answering=False)["A"]
        self.assertFalse(row["online"])
        self.assertEqual((row["last_heard_s"], row["next_check_s"]), (35., 7.))
        self.net.down.discard("A")
        self.clock.advance(studio_remote.POLL_EVERY)
        self.sync()
        row = self.nodes()["A"]
        self.assertEqual((row["online"], row["last_heard_s"]), (True, 0.))

    def test_this_machine_has_no_contact_times(self):
        self.sync()
        local = self.nodes()["local"]
        self.assertEqual((local["last_heard_s"], local["next_check_s"]), (None, None))


class QueueSurvey(unittest.TestCase):
    def test_a_peer_the_merged_queue_marked_offline_is_asked_again_within_the_ceiling_and_then_online(self):
        answers = {"node": {"hostname": "mac", "device": "cpu", "busy": False, "dtypes": ["float32"]}, "queue/jobs": [], "queue": {}}
        up, clock = [False], [100.]

        def ask(address, path, timeout):
            if not up[0]:
                raise ConnectionRefusedError("refused")
            return answers[path]

        peers = cq.Peers(clock=lambda: clock[0])
        for _ in range(10):  # the Mac's API is restarting for a long while
            clock[0] += cq.BACKOFF_MAX
            self.assertFalse(peers.survey(["mac"], ask, 1)[0].info.online)
        up[0] = True
        clock[0] += cq.BACKOFF_MAX
        [back] = peers.survey(["mac"], ask, 1)
        self.assertTrue(back.info.online)
        self.assertFalse(back.stale)


if __name__ == "__main__":
    unittest.main()
