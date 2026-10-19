"""The rules of coordinator failover (studio_failover, docs/distributed.md "Terms and coordinator failover"), without a network, a store or a
coordinator: claim ordering, who the successor is, when the coordinator counts as lost, and the driver over a fake host with a fake clock."""
import unittest
from types import SimpleNamespace

import studio_failover as failover
from studio_failover import Claim, Held, Seen
from tests.fake_cluster import Clock

LEASE = 30.


class Ordering(unittest.TestCase):
    def test_a_higher_term_wins_whatever_the_round(self):
        self.assertTrue(failover.beats(Claim("r", "b", 3, 1, 5), Claim("r", "a", 2, 9, 0)))
        self.assertFalse(failover.beats(Claim("r", "a", 2, 9, 0), Claim("r", "b", 3, 1, 5)))

    def test_in_the_same_term_the_higher_round_wins_then_the_lower_rank_then_the_lower_name(self):
        self.assertTrue(failover.beats(Claim("r", "b", 2, 6, 1), Claim("r", "a", 2, 5, 0)))
        self.assertTrue(failover.beats(Claim("r", "b", 2, 5, 0), Claim("r", "a", 2, 5, 1)))
        self.assertTrue(failover.beats(Claim("r", "a", 2, 5, 0), Claim("r", "b", 2, 5, 0)))
        self.assertFalse(failover.beats(Claim("r", "b", 2, 5, 0), Claim("r", "a", 2, 5, 0)))

    def test_no_claim_beats_itself_and_the_order_is_the_same_whichever_way_it_is_asked(self):
        claims = [Claim("r", n, t, rnd, rank) for n in "ab" for t in (1, 2) for rnd in (3, 4) for rank in (0, 1)]
        for a in claims:
            self.assertFalse(failover.beats(a, a))
            for b in claims:
                if a != b:
                    self.assertNotEqual(failover.beats(a, b), failover.beats(b, a), (a, b))

    def test_standing_is_the_claim_that_outranks_the_others(self):
        old, new = Seen(Claim("r", "a", 1, 7), 40.), Seen(Claim("r", "b", 2, 5, 1), 1.)
        self.assertIs(failover.standing([old, new]), new)
        self.assertIs(failover.standing([new, old]), new)
        self.assertIsNone(failover.standing([]))


class Successor(unittest.TestCase):
    live = staticmethod(lambda node: True)

    def test_the_highest_completed_round_wins(self):
        held = [Held("a", 4, 0), Held("b", 5, 1), Held("c", 3, 2)]
        self.assertEqual(failover.successor(held, self.live).node, "b")

    def test_ties_go_to_the_lowest_rank(self):
        held = [Held("c", 5, 2), Held("b", 5, 1), Held("a", 5, 3)]
        self.assertEqual(failover.successor(held, self.live).node, "b")

    def test_a_node_that_is_not_live_or_not_whole_is_not_a_candidate(self):
        held = [Held("a", 6, 0), Held("b", 5, 1, ready=False), Held("c", 5, 2)]
        self.assertEqual(failover.successor(held, lambda node: node != "a").node, "c")
        self.assertIsNone(failover.successor(held, lambda node: False))
        self.assertIsNone(failover.successor([], self.live))

    def test_every_node_picks_the_same_one_whatever_order_its_store_lists_them_in(self):
        held = [Held("a", 5, 1), Held("b", 5, 0), Held("c", 4, 0)]
        for order in (held, held[::-1], held[1:] + held[:1]):
            self.assertEqual(failover.successor(order, self.live).node, "b")


class Lost(unittest.TestCase):
    def test_the_coordinator_is_lost_when_it_is_down_and_the_leases_have_lapsed(self):
        self.assertTrue(failover.coordinator_lost(True, LEASE, LEASE))
        self.assertTrue(failover.coordinator_lost(True, LEASE + 5, LEASE))

    def test_down_alone_or_silence_alone_is_not_enough(self):
        self.assertFalse(failover.coordinator_lost(True, LEASE - .1, LEASE))  # the members' leases still hold
        self.assertFalse(failover.coordinator_lost(False, 10 * LEASE, LEASE))  # the detector still hears it


class Notes(unittest.TestCase):
    def test_the_sentence_a_failover_wrote_is_replaced_or_removed_and_the_rest_kept(self):
        note = failover.noted("Waiting for B.", "Coordinated on B (term 2).")
        self.assertEqual(note, "Waiting for B. Coordinated on B (term 2).")
        self.assertEqual(failover.noted(note, "Coordinated on C (term 3)."), "Waiting for B. Coordinated on C (term 3).")
        self.assertEqual(failover.noted(note, ""), "Waiting for B.")
        self.assertEqual(failover.noted("", "Coordinated on B (term 2)."), "Coordinated on B (term 2).")


class FakeHost(failover.Host):
    """A host over plain dicts, with a clock the test moves. Records what the driver did."""

    def __init__(self, name, clock):
        self.name, self.clock, self.lease_s = name, clock, LEASE
        self.seen, self.held, self.down, self.mine, self.is_superseded = {}, {}, set(), None, False
        self.log, self.takes, self.allow = [], True, True

    def runs(self):
        return {run: list(seen) for run, seen in self.seen.items()}

    def replicas(self, run):
        return list(self.held.get(run, []))

    def is_down(self, node):
        return node in self.down

    def local(self, run):
        return self.mine

    def superseded(self, run):
        return self.is_superseded

    def tie_break(self, claim):
        self.log.append(("tie_break", claim))
        return self.allow

    def claim(self, claim):
        self.log.append(("claim", claim))
        self.mine = claim

    def withdraw(self, run):
        self.log.append(("withdraw", run))
        self.mine = None

    def take_over(self, claim):
        self.log.append(("take_over", claim))
        return self.takes

    def step_down(self, run, claim):
        self.log.append(("step_down", claim))
        self.mine = None

    def point(self, run, claim, node_down):
        self.log.append(("point", claim.node, claim.term, node_down))

    def note(self, run, claim):
        self.log.append(("note", claim.node, claim.term))

    def finished_under_successor(self, run):
        self.log.append(("finished_under_successor", run))

    def did(self, what):
        return [entry for entry in self.log if entry[0] == what]


class Driver(unittest.TestCase):
    """Three nodes a, b, c; a coordinates run r at term 1 round 5 and went silent at clock 1000."""

    def setUp(self):
        self.clock = Clock(1000.)
        self.host = FakeHost("b", self.clock)
        self.host.held["r"] = [Held("b", 5, 0), Held("c", 5, 1), Held("a", 5, 2)]

    def lost(self, silent=LEASE + 1, state="running", down=("a",)):
        self.host.seen["r"] = [Seen(Claim("r", "a", 1, 5), silent, state)]
        self.host.down = set(down)

    def step(self):
        failover.Failover(self.host).step()

    def test_the_lowest_ranked_replica_claims_the_next_term_at_its_round_and_takes_over(self):
        self.lost()
        self.step()
        self.assertEqual(self.host.did("claim"), [("claim", Claim("r", "b", 2, 5, 0))])
        self.assertEqual(len(self.host.did("take_over")), 1)
        self.assertEqual(self.host.did("withdraw"), [])
        self.assertIn(("point", "b", 2, False), self.host.log)

    def chosen(self, round=5, ready=True, hands_to="b", silent=0., down=()):
        """a is up and coordinating at term 1 round `round`; the person chose `hands_to` to coordinate the run."""
        self.host.seen["r"] = [Seen(Claim("r", "a", 1, round), silent, "running", hands_to)]
        self.host.held["r"] = [Held("b", round, 1, ready), Held("a", round, 0)]
        self.host.down = set(down)

    def test_the_chosen_node_takes_over_from_a_coordinator_that_is_up_once_it_holds_the_last_round(self):
        self.chosen()
        self.step()
        self.assertEqual(self.host.did("claim"), [("claim", Claim("r", "b", 2, 5, 1))])
        self.assertEqual(len(self.host.did("take_over")), 1)

    def test_the_chosen_node_waits_until_its_replica_is_whole_and_current(self):
        for round, ready in ((5, False), (0, True)):
            self.chosen(round=round, ready=ready)
            self.step()
        self.chosen()
        self.host.held["r"] = [Held("b", 4, 1), Held("a", 5, 0)]  # a round behind the coordinator's last
        self.step()
        self.chosen(round=0)  # nothing to continue from before a round has closed
        self.host.held["r"] = [Held("b", 0, 1)]
        self.step()
        self.assertEqual(self.host.did("claim"), [])

    def test_only_the_chosen_node_hands_over_and_a_run_with_no_choice_waits_for_a_loss(self):
        self.chosen(hands_to="c")
        self.step()
        self.chosen(hands_to="")
        self.step()
        self.assertEqual(self.host.did("claim"), [])

    def test_a_run_that_ended_is_not_handed_over(self):
        for state in failover.ENDED:
            self.chosen()
            self.host.seen["r"] = [Seen(Claim("r", "a", 1, 5), 0., state, "b")]
            self.step()
        self.assertEqual(self.host.did("claim"), [])

    def test_the_chosen_node_that_already_coordinates_does_not_claim_again(self):
        self.chosen()
        self.host.mine = Claim("r", "b", 2, 5, 1)
        self.host.seen["r"] = [Seen(Claim("r", "a", 1, 5), 0., "running", "b"), Seen(Claim("r", "b", 2, 5, 1), 0.)]
        self.step()
        self.assertEqual(self.host.did("claim"), [])

    def test_a_run_that_finished_under_a_successor_is_reported_at_every_poll(self):
        self.host.seen["r"] = [Seen(Claim("r", "a", 2, 5), 0., "finished")]
        self.step()
        self.assertEqual(self.host.did("finished_under_successor"), [("finished_under_successor", "r")])
        self.assertEqual(self.host.did("claim"), [])

    def test_a_run_that_ended_otherwise_or_at_term_one_is_not_reported(self):
        for claim, state in ((Claim("r", "a", 1, 5), "finished"), (Claim("r", "a", 2, 5), "stopped"), (Claim("r", "a", 2, 5), "failed")):
            self.host.seen["r"] = [Seen(claim, 0., state)]
            self.step()
        self.assertEqual(self.host.did("finished_under_successor"), [])

    def test_another_replica_that_is_not_the_successor_does_nothing(self):
        self.host.name = "c"
        self.lost()
        self.step()
        self.assertEqual(self.host.did("claim"), [])
        self.assertEqual(self.host.did("take_over"), [])

    def test_nothing_happens_while_the_coordinator_is_up_or_its_leases_hold(self):
        self.lost(down=())
        self.step()
        self.lost(silent=LEASE - 1)
        self.step()
        self.assertEqual(self.host.did("claim"), [])

    def test_a_finished_or_stopped_run_is_not_taken_over(self):
        for state in failover.ENDED:
            self.lost(state=state)
            self.step()
        self.assertEqual(self.host.did("claim"), [])

    def test_the_next_replica_takes_over_when_the_successor_is_down_too(self):
        self.lost(down=("a", "b"))
        self.host.name = "c"
        self.step()
        self.assertEqual(self.host.did("claim"), [("claim", Claim("r", "c", 2, 5, 1))])

    def test_a_claim_that_could_not_be_served_is_withdrawn(self):
        self.lost()
        self.host.takes = False
        self.step()
        self.assertEqual(self.host.did("withdraw"), [("withdraw", "r")])
        self.assertIsNone(self.host.mine)

    def test_the_bucket_seam_can_refuse_a_claim_and_then_nothing_is_written(self):
        self.lost()
        self.host.allow = False
        self.step()
        self.assertEqual(len(self.host.did("tie_break")), 1)
        self.assertEqual(self.host.did("claim"), [])
        self.assertEqual(self.host.did("take_over"), [])

    def test_a_dead_successors_claim_is_replaced_by_the_next_term_once_it_is_lost_too(self):
        self.host.seen["r"] = [Seen(Claim("r", "a", 1, 5), 90.), Seen(Claim("r", "c", 2, 5, 1), LEASE + 1)]
        self.host.down = {"a", "c"}
        self.step()
        self.assertEqual(self.host.did("claim"), [("claim", Claim("r", "b", 3, 5, 0))])

    def test_a_coordinator_whose_claim_is_outranked_steps_down(self):
        self.host.mine = Claim("r", "b", 2, 5, 1)
        self.host.seen["r"] = [Seen(self.host.mine, 0.), Seen(Claim("r", "c", 2, 5, 0), 1.)]
        self.step()
        self.assertEqual(self.host.did("step_down"), [("step_down", Claim("r", "c", 2, 5, 0))])

    def test_a_coordinator_that_outranks_the_other_claim_keeps_the_run(self):
        self.host.mine = Claim("r", "b", 2, 5, 0)
        self.host.seen["r"] = [Seen(self.host.mine, 0.), Seen(Claim("r", "c", 2, 5, 1), 1.)]
        self.step()
        self.assertEqual(self.host.did("step_down"), [])

    def test_the_old_coordinator_that_returns_steps_down_at_its_first_step(self):
        self.host.name = "a"
        self.host.mine = Claim("r", "a", 1, 6)
        self.host.seen["r"] = [Seen(self.host.mine, 0.), Seen(Claim("r", "b", 2, 5, 0), 1.)]
        self.step()
        self.assertEqual(self.host.did("step_down"), [("step_down", Claim("r", "b", 2, 5, 0))])

    def test_a_coordinator_told_of_a_higher_term_by_a_call_steps_down_without_waiting_for_the_store(self):
        self.host.mine = Claim("r", "b", 1, 6)
        self.host.seen["r"] = [Seen(self.host.mine, 0.)]
        self.host.is_superseded = True
        self.step()
        self.assertEqual(self.host.did("step_down"), [("step_down", None)])

    def test_one_run_failing_does_not_stop_the_others(self):
        self.lost()
        self.host.seen["bad"] = [Seen(Claim("bad", "a", 1, 5), 99.)]
        original = self.host.local

        def local(run):
            if run == "bad":
                raise RuntimeError("boom")
            return original(run)
        self.host.local = local
        with self.assertLogs("studio_failover", "WARNING"):
            self.step()
        self.assertEqual(len(self.host.did("claim")), 1)


class WhereFileAddress(unittest.TestCase):
    """ClusterHost.url_of: the address the where-file gives a member here for the coordinator on `node`."""

    def host(self, published):
        claims = {node: {"value": {"term": 2, "round": 3, "rank": 0, **({"url": url} if url else {})}} for node, url in published.items()}
        cluster = SimpleNamespace(hostname="here", lease_s=LEASE, api_port=9123, address_of=lambda node: f"{node}-as-listed",
                                  state=SimpleNamespace(get=lambda origin, kind, key: claims.get(origin) if kind == failover.CLAIM else None))
        return failover.ClusterHost(cluster)

    def test_this_node_is_reached_on_loopback_at_the_port_its_api_listens_on(self):
        self.assertEqual(self.host({}).url_of("here", "r"), "http://127.0.0.1:9123")

    def test_a_peer_is_reached_at_the_url_it_published_with_its_claim(self):
        self.assertEqual(self.host({"box": "https://box.tail.ts.net"}).url_of("box", "r"), "https://box.tail.ts.net")

    def test_without_a_usable_published_url_a_peer_is_reached_at_its_listed_address(self):
        host = self.host({"old": None, "lone": "http://127.0.0.1:8787"})
        self.assertEqual(host.url_of("old", "r"), "old-as-listed")  # a claim from before urls were published
        self.assertEqual(host.url_of("lone", "r"), "lone-as-listed")  # a loopback url is of no use from another machine
        self.assertEqual(host.url_of("unknown", "r"), "unknown-as-listed")


class FinishedUnderSuccessor(unittest.TestCase):
    """ClusterHost.finished_under_successor: replays the end of the owner's job so the services fetch the successor's weights."""

    def host(self, status="failed", version=3, owner=""):
        import studio_distributed
        from threading import RLock
        self.addCleanup(studio_distributed.RUNS.clear)
        studio_distributed.RUNS["r"] = studio_distributed.Run("r", "", version, 0., {}, "here")
        self.run = studio_distributed.RUNS["r"]
        self.events = []
        job = SimpleNamespace(id="r", kind="pretrain_distributed", owner=owner, status=status, params={"version": version})
        runner = SimpleNamespace(lock=RLock(), jobs=[job], _emit=lambda kind, payload: self.events.append((kind, payload)))
        return failover.ClusterHost(SimpleNamespace(hostname="here", lease_s=LEASE, runner=runner))

    def test_an_owner_job_that_failed_or_stopped_is_ended_again_for_the_services(self):
        for status in ("failed", "stopped"):
            self.host(status).finished_under_successor("r")
            self.assertEqual(self.events, [("job", {"job": "r", "status": status, "error": ""})])

    def test_nothing_happens_for_a_running_job_a_stopped_run_a_saved_run_a_job_of_another_owner_or_no_version(self):
        self.host("running").finished_under_successor("r")
        self.host(owner="there").finished_under_successor("r")
        self.host(version=None).finished_under_successor("r")
        host = self.host()
        self.run.stopped = True
        host.finished_under_successor("r")
        self.run.stopped, self.run.saved = False, 3
        host.finished_under_successor("r")
        self.assertEqual(self.events, [])

    def test_a_fetch_that_keeps_failing_is_asked_for_a_few_times_only(self):
        host = self.host()
        for _ in range(failover.SAVE_TRIES + 4):
            host.finished_under_successor("r")
        self.assertEqual(len(self.events), failover.SAVE_TRIES)


if __name__ == "__main__":
    unittest.main()
