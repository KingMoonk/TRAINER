"""How long a member waits for the coordinator that took over a run, and what it does when the coordinator it was sent to has already
finished it (studio_diloco.Member._call, _relocated_within, _rebind, work). A fake client, a fake clock and a fake `locate`: no network,
no coordinator, no training."""
import tempfile
import threading
import unittest
from pathlib import Path

import studio_diloco
from studio_peer import PeerError

GONE = PeerError("404", status=404, detail=f"{studio_diloco.GONE_RUN} r1")


class Clock:
    """A clock the member's sleep moves: waiting costs no time."""

    def __init__(self):
        self.now, self.slept = 0., []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


class GoneClient:
    """Every call finds the coordinator without the run (its app restarted, or the run moved to another node)."""

    def __init__(self):
        self.calls = 0

    def get_json(self, path, params=None, timeout=None):
        self.calls += 1
        raise GONE

    post_json = get_json


class FinishedClient:
    """A coordinator that has finished the run: a join is a 409 'The run is finished.'."""

    def __init__(self):
        self.paths = []

    def post_json(self, path, body=None, params=None, timeout=None):
        self.paths.append(path)
        raise PeerError("409", status=409, detail=studio_diloco.FINISHED)

    get_json = post_json


class Fixtures(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.clock = Clock()

    def member(self, locate, client=None, **changes):
        member = studio_diloco.Member(client or GoneClient(), "r1", "a", 0, Path(self.tmp.name) / "shards", threading.Event(),
                                      clock=self.clock, sleep=self.clock.sleep, locate=locate, **changes)
        member.term, member.where = 2, ("n1", 2)
        return member

    def call(self, member):
        return member._call(lambda: member.client.get_json("/collective/r1/params"))


class Waiting(Fixtures):
    def test_a_coordinator_that_lost_the_run_and_names_nobody_is_left_after_the_short_window_not_the_patience(self):
        member = self.member(lambda run: None)
        with self.assertRaises(PeerError) as raised:
            self.call(member)
        self.assertIn("no longer has run r1", str(raised.exception))
        self.assertEqual(raised.exception.status, 404)
        self.assertGreaterEqual(self.clock.now, member.relocate)
        self.assertLessEqual(self.clock.now, member.relocate + 2.)
        self.assertLess(self.clock.now, member.patience / 10)

    def test_a_claim_that_names_the_coordinator_it_already_talks_to_is_no_move_and_ends_the_same_way(self):
        member = self.member(lambda run: ("n1", 2, object()))  # the stuck case: the same (node, term) as the one that just said it has no run
        with self.assertRaises(PeerError) as raised:
            self.call(member)
        self.assertIn("no longer has run r1", str(raised.exception))
        self.assertLessEqual(self.clock.now, member.relocate + 2.)

    def test_a_locator_that_fails_is_no_answer(self):
        def broken(run):
            raise OSError("no store")
        with self.assertRaises(PeerError):
            self.call(self.member(broken))
        self.assertLessEqual(self.clock.now, studio_diloco.RELOCATE_SECONDS + 2.)

    def test_the_claim_arriving_inside_the_window_rebinds_the_member(self):
        successor = ("n2", 3, object())
        member = self.member(lambda run: successor if self.clock.now >= 5. else None)
        with self.assertRaises(studio_diloco.Rebound) as raised:
            self.call(member)
        self.assertIs(raised.exception.found, successor)
        self.assertLess(self.clock.now, member.relocate)

    def test_the_wait_shares_the_calls_own_deadline(self):
        member = self.member(lambda run: None, patience=4.)  # patience is shorter than the window: the call ends at its own deadline
        with self.assertRaises(PeerError):
            self.call(member)
        self.assertLessEqual(self.clock.now, 4. + 2.)

    def test_without_a_locator_the_member_leaves_at_once(self):
        member = self.member(None)
        with self.assertRaises(PeerError):
            self.call(member)
        self.assertEqual(self.clock.now, 0.)
        self.assertEqual(member.client.calls, 1)


class Finished(Fixtures):
    def test_a_rebind_to_a_run_that_is_finished_ends_the_member_instead_of_failing(self):
        member = self.member(lambda run: None, client=FinishedClient())
        member._rebind(("n2", 3, FinishedClient()))
        self.assertTrue(member.finished)
        self.assertEqual(member.where, ("n2", 3))

    def test_any_other_refusal_of_the_join_still_fails(self):
        class Banned(FinishedClient):
            def post_json(self, path, body=None, params=None, timeout=None):
                raise PeerError("409", status=409, detail="n1 was stopped by the run's owner.")
        member = self.member(lambda run: None)
        with self.assertRaises(PeerError):
            member._rebind(("n2", 3, Banned()))
        self.assertFalse(member.finished)

    def test_work_returns_cleanly_when_the_first_join_is_sent_to_a_run_that_is_already_finished(self):
        successor = ("n2", 3, FinishedClient())
        member = self.member(lambda run: successor)  # the first join finds the run gone from here and in the hands of a coordinator that finished it
        self.assertIsNone(member.work(Path(self.tmp.name) / "out"))  # no AttributeError on `info`, no training
        self.assertTrue(member.finished)


if __name__ == "__main__":
    unittest.main()
