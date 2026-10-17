"""Run terms (docs/distributed.md, "Terms and coordinator failover"): every /collective call carries one, a lower one is 409, a higher
one makes the coordinator step down, a member that sees a higher one stops sending, and diloco.json keeps it."""
import threading
import unittest

import studio_diloco
from studio_peer import PeerError
from tests import diloco_support as support
from tests.test_api_collective import Collective


class Clock:
    def __init__(self):
        self.now = 0.

    def __call__(self):
        return self.now


class Calls(Collective):
    """The same fixtures as the wire protocol tests: a coordinator in the registry behind the real routes."""

    def call(self, method, path, term, **options):
        return self.client.request(method, f"/collective/r1/{path}", **{**options, "params": {**options.get("params", {}), "term": term}})

    def routes(self, coordinator, term):
        """Every route of the wire protocol, called with `term`, in an order that does not matter for the answer's status."""
        body = self.delta_body(coordinator)
        node = {"node": "a", "generation": 1, "epoch": 0}
        return {
            "join": self.call("POST", "join", term, json={"node": "a", "epoch": 0}),
            "params": self.call("GET", "params", term, params={"round": 0}),
            "shards": self.call("GET", "shards/0", term),
            "round": self.call("GET", "rounds/0", term, params={"node": "a"}),
            "progress": self.call("POST", "progress", term, json={**node, "round": 0, "phase": "training"}),
            "delta": self.call("POST", "rounds/0/delta", term, content=body, headers={"content-type": "application/octet-stream"},
                               params={"node": "a", "generation": 1, "epoch": 0, "tokens": 10, "steps": 2}),
            "leave": self.call("POST", "leave", term, json=node),
        }


class TermChecks(Calls):
    def test_a_lower_term_is_409_on_every_route_and_changes_nothing(self):
        coordinator = self.start(inner_steps=2)
        coordinator.term = 2
        for route, answer in self.routes(coordinator, 1).items():
            with self.subTest(route):
                self.assertEqual(answer.status_code, 409, answer.text)
                self.assertTrue(answer.json()["detail"].startswith(studio_diloco.TERM_STALE))
                self.assertEqual(answer.headers[studio_diloco.TERM_HEADER], "2")
        self.assertEqual((coordinator.members, coordinator.pending, coordinator.superseded), ({}, {}, 0))

    def test_a_member_that_does_not_know_the_term_may_only_join(self):
        coordinator = self.start(inner_steps=2)
        coordinator.term = 3
        answers = self.routes(coordinator, 0)
        self.assertEqual(answers["join"].status_code, 200, answers["join"].text)
        self.assertEqual(answers["join"].json()["term"], 3)
        for route in set(answers) - {"join"}:
            with self.subTest(route):
                self.assertEqual(answers[route].status_code, 409)

    def test_every_answer_carries_the_coordinators_term(self):
        coordinator = self.start(inner_steps=2)
        coordinator.term = 2
        answers = self.routes(coordinator, 2)
        for route, answer in answers.items():
            with self.subTest(route):
                self.assertIn(answer.status_code, (200, 202, 204), answer.text)
                self.assertEqual(answer.headers[studio_diloco.TERM_HEADER], "2")
        for route in ("join", "round", "progress", "delta"):
            self.assertEqual(answers[route].json()["term"], 2, route)

    def test_a_higher_term_makes_the_coordinator_step_down_and_refuse_everything(self):
        told = []
        coordinator = self.start(inner_steps=2)
        coordinator.on_step_down = told.append
        self.assertEqual(self.call("POST", "join", 1, json={"node": "a", "epoch": 0}).status_code, 200)
        before = {name: value.copy() for name, value in coordinator.theta.items()}
        refused = self.call("GET", "rounds/0", 2, params={"node": "a"})
        self.assertEqual(refused.status_code, 409)
        self.assertEqual((told, coordinator.superseded, coordinator.term), ([2], 2, 1))
        for route, answer in self.routes(coordinator, 1).items():  # even its own term is refused now
            with self.subTest(route):
                self.assertEqual(answer.status_code, 409)
        self.assertEqual(coordinator.history, [])
        for name, value in before.items():
            self.assertTrue((coordinator.theta[name] == value).all())

    def test_the_step_down_hook_is_told_once_and_only_of_a_higher_term(self):
        told = []
        coordinator = self.start()
        coordinator.on_step_down = told.append
        coordinator.step_down(1)
        coordinator.step_down(3)
        coordinator.step_down(3)
        coordinator.step_down(2)
        self.assertEqual((told, coordinator.superseded), ([3], 3))

    def test_a_coordinator_that_stepped_down_closes_no_round(self):
        clock = Clock()
        coordinator = support.build(self.dir / "late", "late", clock=clock, inner_steps=2)
        studio_diloco.COORDINATORS["late"] = coordinator
        coordinator.join("a", 0)
        self.assertEqual(coordinator.current.number, 0)
        coordinator.step_down(2)
        clock.now += 10_000.  # long past the deadline and its grace
        coordinator.round_view(0, "a")
        coordinator.progress(0, "a", 1, 0, "training")
        self.assertEqual((coordinator.next_round, coordinator.history, coordinator.current.number), (0, [], 0))
        with self.assertRaises(studio_diloco.Stale):
            coordinator.check_term(1)

    def test_the_lease_fence_compares_term_then_epoch(self):
        coordinator = self.start()
        coordinator.join("a", 5)
        self.assertEqual(coordinator.members["a"]["term"], 1)
        coordinator.join("a", 6)  # a newer epoch under the same term is taken
        with self.assertRaises(studio_diloco.Stale):
            coordinator.join("a", 6, term=0)  # an earlier term loses even to an equal epoch ...
        with self.assertRaises(studio_diloco.Stale):
            coordinator.join("a", 99, term=0)  # ... and to a larger one: the term is compared first
        with self.assertRaises(studio_diloco.Stale):
            coordinator.join("a", 5)
        coordinator.join("a", 7, term=1)
        self.assertEqual(coordinator.members["a"]["epoch"], 7)


class Persistence(Calls):
    def close_round(self, coordinator):
        coordinator.join("a", 0)
        coordinator.submit(0, "a", coordinator.generation, 0, 10, 2, support.constant_delta(coordinator))
        self.assertEqual(coordinator.next_round, 1)

    def test_a_new_run_is_at_term_one_and_diloco_json_keeps_the_term(self):
        coordinator = support.build(self.dir / "t", "t", inner_steps=2, rounds=3)
        self.assertEqual(coordinator.term, 1)
        coordinator.term = 4
        self.close_round(coordinator)
        self.assertEqual(support.read_json(self.dir / "t" / "diloco.json")["term"], 4)
        again = support.build(self.dir / "t", "t", inner_steps=2, rounds=3)
        self.assertEqual((again.term, again.next_round), (4, 1))

    def test_a_diloco_json_without_a_term_continues_at_term_one(self):
        coordinator = support.build(self.dir / "old", "old", inner_steps=2, rounds=3)
        self.close_round(coordinator)
        path = self.dir / "old" / "diloco.json"
        saved = support.read_json(path)
        del saved["term"]
        path.write_text(studio_diloco.json.dumps(saved), encoding="utf-8")
        self.assertEqual(support.build(self.dir / "old", "old", inner_steps=2, rounds=3).term, 1)


class MemberTerms(Calls):
    def member(self, **changes):
        calls = []

        class Spy(support.Peer):
            def _check(self, response, path):
                calls.append(path)
                return super()._check(response, path)
        member = studio_diloco.Member(Spy(self.client), "r1", "a", 0, self.dir / "shards", threading.Event(), **changes)
        member.events = []
        member.emit = member.events.append
        return member, calls

    def test_a_member_learns_the_term_at_join_and_sends_it_after(self):
        coordinator = self.start(inner_steps=2)
        coordinator.term = 3
        member, calls = self.member()
        self.assertEqual(member.term, 0)
        member.join()
        self.assertEqual(member.term, 3)
        self.assertEqual(member.client.client.get("/collective/r1/rounds/0", params=member._q({"node": "a"})).json()["term"], 3)

    def test_a_member_that_sees_a_higher_term_sends_nothing_more_to_the_old_coordinator(self):
        self.start(inner_steps=2)
        member, calls = self.member()
        member.join()
        member.report_round = 0
        member.phase, member.generation = "waiting", 1
        sent = len(calls)
        member.follow(2)
        self.assertEqual((member.superseded, member.fenced), (2, True))
        self.assertTrue(any("term 2 exists" in event["text"].lower() for event in member.events))
        self.assertFalse(member.report())
        with self.assertRaises(PeerError) as raised:
            member._call(lambda: member.client.get_json("/collective/r1/rounds/0", {"node": "a"}))
        self.assertEqual(raised.exception.status, 409)
        member.leave()
        self.assertEqual(len(calls), sent)  # not one request left the member
        member.follow(1)  # a lower term changes nothing
        self.assertEqual(member.superseded, 2)

    def test_an_answer_with_a_higher_term_is_followed(self):
        self.start(inner_steps=2)
        member, _ = self.member()
        member.join()
        member._seen({"term": member.term})
        self.assertEqual(member.superseded, 0)
        member._seen({"term": member.term + 1})
        self.assertEqual((member.superseded, member.fenced), (member.term + 1, True))

    def test_a_409_that_names_a_higher_term_is_followed(self):
        coordinator = self.start(inner_steps=2)
        member, _ = self.member()
        member.join()
        coordinator.term = 5  # the run moved on while this member held term 1
        with self.assertRaises(PeerError):
            member._call(lambda: member.client.get_json("/collective/r1/rounds/0", member._q({"node": "a"})))
        self.assertEqual((member.superseded, member.fenced), (5, True))


if __name__ == "__main__":
    unittest.main()
