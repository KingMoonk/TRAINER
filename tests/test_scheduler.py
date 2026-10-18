"""studio_scheduler: the failure detector, leases, self-fencing, placement and the two ends of a dispatch, on fake clocks and peers."""
import unittest

import studio_cluster as sc
import studio_scheduler as sched
from studio_cluster import NodeInfo
from studio_scheduler import DOWN, SUSPECT, UP, Guest, Health, Lease, LeaseTable, Owner, SelfFence, StaleEpoch
from tests.fake_cluster import Clock, Net, refuse

GB = 2 ** 30


def node(address, device="cpu", memory=16 * GB, busy=False, hostname=None):
    return NodeInfo(address=address, hostname=hostname or address, device=device, memory_bytes=memory, busy=busy, dtypes=sc.DTYPES[device])


class FailureDetector(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.health = Health(self.clock, down_after=30.)

    def test_a_node_that_answered_is_up(self):
        self.health.observe("box", True)
        self.assertEqual(self.health.state("box"), UP)

    def test_a_node_never_heard_from_is_not_known_to_be_up(self):
        self.assertEqual(self.health.state("box"), SUSPECT)

    def test_one_missed_poll_makes_a_node_suspect_not_down(self):
        self.health.observe("box", True)
        self.clock.advance(5)
        self.health.observe("box", False)
        self.assertEqual(self.health.state("box"), SUSPECT)

    def test_down_only_after_the_timeout_since_the_last_answer(self):
        self.health.observe("box", True)
        for _ in range(5):
            self.clock.advance(5)
            self.health.observe("box", False)
            self.assertEqual(self.health.state("box"), SUSPECT)
        self.clock.advance(5)  # 30 s since the last answer
        self.health.observe("box", False)
        self.assertEqual(self.health.state("box"), DOWN)

    def test_a_node_that_never_answered_is_down_30_seconds_after_the_first_failure(self):
        self.health.observe("box", False)
        self.clock.advance(29)
        self.health.observe("box", False)
        self.assertEqual(self.health.state("box"), SUSPECT)
        self.clock.advance(1)
        self.assertEqual(self.health.state("box"), DOWN)

    def test_an_answer_brings_a_down_node_back(self):
        self.health.observe("box", False)
        self.clock.advance(60)
        self.assertEqual(self.health.state("box"), DOWN)
        self.health.observe("box", True)
        self.assertEqual(self.health.state("box"), UP)


class Leases(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.table = LeaseTable(self.clock)

    def test_every_grant_increments_the_epoch(self):
        first = self.table.grant("j", "a", 30.)
        second = self.table.grant("j", "b", 30.)
        self.assertEqual((first.epoch, second.epoch, second.node), (1, 2, "b"))
        self.assertEqual(self.table.grant("other", "a").epoch, 1)  # epochs are per job

    def test_a_write_with_the_current_epoch_is_accepted_and_any_other_is_refused(self):
        lease = self.table.grant("j", "a", 30.)
        self.assertEqual(self.table.renew("j", lease.epoch), lease)
        self.table.grant("j", "b", 30.)
        with self.assertRaises(StaleEpoch):
            self.table.renew("j", lease.epoch)
        with self.assertRaises(StaleEpoch):
            self.table.check("j", 3)

    def test_a_released_lease_refuses_everyone_and_the_next_grant_goes_on_counting(self):
        lease = self.table.grant("j", "a", 30.)
        self.table.release("j")
        with self.assertRaises(StaleEpoch):
            self.table.renew("j", lease.epoch)
        self.assertEqual(self.table.grant("j", "a", 30.).epoch, 2)

    def test_seeded_epochs_survive_a_restart(self):
        self.table.seed("j", 4)
        self.assertEqual(self.table.grant("j", "a").epoch, 5)

    def test_a_lease_lapses_after_its_duration_and_expires_after_two(self):
        lease = self.table.grant("j", "a", 30.)
        self.clock.advance(30)
        self.assertEqual((self.table.lapsed(), self.table.expired()), ([], []))
        self.clock.advance(1)
        self.assertEqual((self.table.lapsed(), self.table.expired()), (["j"], []))
        self.clock.advance(29)
        self.assertEqual(self.table.expired(), [])
        self.clock.advance(1)
        self.assertEqual(self.table.expired(), ["j"])
        self.assertEqual(lease.renew_every, 10.)

    def test_a_renewal_restarts_the_count(self):
        lease = self.table.grant("j", "a", 30.)
        self.clock.advance(25)
        self.table.renew("j", lease.epoch)
        self.clock.advance(25)
        self.assertEqual(self.table.lapsed(), [])


class SelfFencing(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.fence = SelfFence(Lease("j", "a", 1, 30.), self.clock)

    def test_the_lease_runs_from_the_dispatch_and_a_renewal_is_due_every_third(self):
        self.assertFalse(self.fence.due())
        self.clock.advance(10)
        self.assertTrue(self.fence.due())
        self.assertEqual(self.fence.remaining(), 20)

    def test_a_renewal_counts_from_when_it_was_sent_not_when_it_was_answered(self):
        self.clock.advance(10)
        sent = self.fence.sending()
        self.clock.advance(8)  # a slow answer
        self.fence.acked(sent)
        self.assertEqual(self.fence.remaining(), 30 - 8)

    def test_a_late_answer_never_moves_the_deadline_back(self):
        self.clock.advance(10)
        early = self.fence.sending()
        self.clock.advance(10)
        later = self.fence.sending()
        self.fence.acked(later)
        self.fence.acked(early)
        self.assertEqual(self.fence.remaining(), 30)

    def test_it_expires_when_no_renewal_was_acknowledged_in_time(self):
        self.clock.advance(29)
        self.assertFalse(self.fence.expired())
        self.clock.advance(1)
        self.assertTrue(self.fence.expired())

    def test_an_unanswered_renewal_does_not_extend_it(self):
        self.clock.advance(10)
        self.fence.sending()
        self.clock.advance(20)
        self.assertTrue(self.fence.expired())


class Placement(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.health = Health(self.clock)
        self.cpu = node(sc.LOCAL, hostname="laptop")
        self.gpu = node("gpu-box", device="cuda", hostname="gpu-box")
        self.health.observe("gpu-box", True)

    def decide(self, kind="pretrain", nodes=None, **kwargs):
        return sched.decide(kind, nodes or [self.cpu, self.gpu], self.health, **kwargs)

    def test_a_pretraining_job_goes_to_the_free_cuda_peer(self):
        self.assertEqual(self.decide(), sched.Placement("remote", "gpu-box"))

    def test_it_stays_here_when_the_peer_is_busy_or_not_answering(self):
        self.assertEqual(self.decide(nodes=[self.cpu, node("gpu-box", "cuda", busy=True)]).where, "local")
        self.health.observe("gpu-box", False)
        self.assertEqual(self.decide().where, "local")

    def test_a_node_that_just_refused_the_job_is_skipped(self):
        self.assertEqual(self.decide(exclude={"gpu-box"}).where, "local")

    def test_a_kind_that_cannot_leave_its_owner_stays_here(self):
        for kind in ("quicktest", "import_pretrained", "checkpoint_now"):
            self.assertEqual(self.decide(kind).where, "local", kind)

    def test_the_other_kinds_go_to_a_free_gpu(self):
        for kind in ("pretrain", "post_round", "restore_answers"):
            self.assertEqual(self.decide(kind), sched.Placement("remote", "gpu-box"), kind)
        self.assertEqual(self.decide("compare", pinned="gpu-box"), sched.Placement("remote", "gpu-box"))

    def test_without_a_store_nothing_leaves_this_machine(self):
        self.assertEqual(self.decide(store_ready=False).where, "local")

    def test_a_pinned_job_goes_to_its_node(self):
        self.assertEqual(self.decide(pinned="gpu-box"), sched.Placement("remote", "gpu-box"))
        self.assertEqual(self.decide(pinned="laptop").where, "local")

    def test_a_pinned_job_waits_when_its_node_cannot_take_it_and_never_moves_to_a_better_one(self):
        self.health.observe("gpu-box", False)
        placed = self.decide(pinned="gpu-box")
        self.assertEqual(placed.where, "wait")
        self.assertIn("not answering", placed.reason)
        self.assertEqual(self.decide(nodes=[node(sc.LOCAL, hostname="laptop", busy=True), self.gpu], pinned="laptop").where, "wait")

    def test_the_wait_says_why(self):
        self.assertIn("not one of this cluster's machines", self.decide(pinned="ghost").reason)
        self.assertIn("only runs on the machine that queued it", self.decide("quicktest", pinned="gpu-box").reason)
        self.assertIn("version store", self.decide(pinned="gpu-box", store_ready=False).reason)
        self.assertIn("just refused", self.decide(pinned="gpu-box", exclude={"gpu-box"}).reason)
        self.assertIn("busy", self.decide(nodes=[self.cpu, node("gpu-box", "cuda", busy=True)], pinned="gpu-box").reason)

    def test_a_pin_to_a_hostname_two_machines_share_waits_and_names_them(self):
        twin = node("100.64.0.3:8787", "cuda", hostname="GPU-Box")
        first = node("100.64.0.2:8787", "cuda", hostname="gpu-box")
        for address in (first.address, twin.address):
            self.health.observe(address, True)
        nodes = [self.cpu, first, twin]
        placed = self.decide(nodes=nodes, pinned="gpu-box")
        self.assertEqual(placed.where, "wait")
        for text in ("gpu-box (100.64.0.2:8787)", "GPU-Box (100.64.0.3:8787)", "by its address"):
            self.assertIn(text, placed.reason)
        self.assertEqual(self.decide(nodes=nodes, pinned="GPU-BOX").where, "wait")  # however it is typed
        self.assertEqual(self.decide(nodes=nodes, pinned=first.address), sched.Placement("remote", first.address))
        self.assertEqual(self.decide(nodes=nodes, pinned=twin.address), sched.Placement("remote", twin.address))
        self.assertEqual(self.decide(nodes=nodes, pinned="laptop").where, "local")  # a unique hostname is unchanged

    def test_a_pin_to_a_hostname_this_machine_shares_can_still_name_this_machine_by_its_address(self):
        twin = node("100.64.0.3:8787", "cuda", hostname="laptop")
        self.health.observe(twin.address, True)
        nodes = [self.cpu, twin]
        self.assertIn("laptop (local)", self.decide(nodes=nodes, pinned="laptop").reason)
        self.assertEqual(self.decide(nodes=nodes, pinned=sc.LOCAL).where, "local")
        self.assertEqual(self.decide(nodes=nodes, pinned=twin.address).where, "remote")

    def test_a_job_with_no_fitting_node_waits(self):
        busy = [node(sc.LOCAL, busy=True), node("gpu-box", "cuda", busy=True)]
        self.assertEqual(self.decide(nodes=busy).where, "wait")

    def test_a_job_that_needs_more_memory_than_any_node_has_waits(self):
        small = [node(sc.LOCAL, memory=GB), node("gpu-box", "cuda", memory=GB)]
        needs = sc.Requirements(min_memory_bytes=8 * GB)
        self.assertEqual(self.decide(nodes=small, requirements=needs).where, "wait")


class Dispatching(unittest.TestCase):
    def setUp(self):
        self.clock, self.net = Clock(), Net()
        self.net.handlers["gpu"] = lambda method, path, body: {"job_id": body["lease"]["job_id"]}
        self.owner = Owner(self.net.client_for, self.clock, lease_s=30.)

    def test_the_peer_gets_the_job_and_a_lease_at_epoch_1(self):
        result = self.owner.dispatch("j", "gpu", {"job": {"id": "j"}})
        self.assertTrue(result.confirmed)
        self.assertEqual(result.lease, Lease("j", "gpu", 1, 30.))
        method, path, body = self.net.to("gpu")[0]
        self.assertEqual((method, path, body["lease"], body["job"]), ("POST", "cluster/jobs", result.lease.as_dict(), {"id": "j"}))

    def test_the_lease_is_counted_from_the_answer_not_from_the_grant(self):
        def slow(method, path, body):
            self.clock.advance(20)  # the peer took 20 s to answer
            return {}
        self.net.handlers["gpu"] = slow
        self.owner.dispatch("j", "gpu", {})
        self.clock.advance(31)
        self.assertEqual(self.owner.leases.lapsed(), ["j"])
        self.clock.advance(-1)
        self.assertEqual(self.owner.leases.lapsed(), [])

    def test_a_refusal_revokes_the_lease_and_keeps_the_peer_from_being_asked_again_for_a_while(self):
        def no(method, path, body):
            raise refuse(409, "This machine has no version store.")
        self.net.handlers["gpu"] = no
        with self.assertRaisesRegex(sched.Refused, "no version store"):
            self.owner.dispatch("j", "gpu", {})
        self.assertIsNone(self.owner.leases.get("j"))
        self.assertEqual(self.owner.refused_by("j"), {"gpu"})
        self.clock.advance(sched.REFUSAL_COOLDOWN + 1)
        self.assertEqual(self.owner.refused_by("j"), set())
        self.net.handlers["gpu"] = lambda method, path, body: {}
        self.assertEqual(self.owner.dispatch("j", "gpu", {}).lease.epoch, 2)  # offered again, at the next epoch

    def test_a_peer_that_does_not_answer_may_have_started_the_job_so_the_lease_stays(self):
        self.net.down.add("gpu")
        result = self.owner.dispatch("j", "gpu", {})
        self.assertFalse(result.confirmed)
        self.assertIsNotNone(self.owner.leases.get("j"))
        self.clock.advance(61)  # nothing renews it: two durations later the job may be placed again
        self.assertEqual(self.owner.tick(), [("j", "reassign")])

    def test_lapse_is_reported_once_and_reassignment_waits_a_second_duration(self):
        self.owner.dispatch("j", "gpu", {})
        self.clock.advance(31)
        self.assertEqual(self.owner.tick(), [("j", "lapsed")])
        self.clock.advance(10)
        self.assertEqual(self.owner.tick(), [])
        self.clock.advance(20)
        self.assertEqual(self.owner.tick(), [("j", "reassign")])
        self.assertEqual(self.owner.tick(), [])

    def test_a_renewal_after_a_lapse_recovers_the_lease(self):
        lease = self.owner.dispatch("j", "gpu", {}).lease
        self.clock.advance(31)
        self.owner.tick()
        _, recovered = self.owner.renew("j", lease.epoch)
        self.assertTrue(recovered)
        self.clock.advance(31)
        self.assertEqual(self.owner.tick(), [("j", "lapsed")])  # and can lapse again

    def test_after_reassignment_the_old_epoch_is_refused_and_the_new_grant_is_epoch_2(self):
        old = self.owner.dispatch("j", "gpu", {}).lease
        self.clock.advance(61)
        self.owner.tick()
        with self.assertRaises(StaleEpoch):
            self.owner.renew("j", old.epoch)
        self.assertEqual(self.owner.dispatch("j", "gpu", {}).lease.epoch, 2)
        with self.assertRaises(StaleEpoch):
            self.owner.renew("j", old.epoch)

    def test_cancel_tells_the_peer_and_fences_the_copy(self):
        lease = self.owner.dispatch("j", "gpu", {}).lease
        self.net.handlers["gpu"] = lambda m, p, b: None
        self.owner.cancel("j")
        self.assertEqual(self.net.to("gpu")[-1], ("POST", "cluster/jobs/j/stop", {"epoch": 1}))
        with self.assertRaises(StaleEpoch):
            self.owner.renew("j", lease.epoch)

    def test_cancel_still_fences_when_the_peer_cannot_be_told(self):
        lease = self.owner.dispatch("j", "gpu", {}).lease
        self.net.down.add("gpu")
        self.owner.cancel("j")
        with self.assertRaises(StaleEpoch):
            self.owner.renew("j", lease.epoch)

    def test_finishing_needs_the_current_epoch(self):
        lease = self.owner.dispatch("j", "gpu", {}).lease
        with self.assertRaises(StaleEpoch):
            self.owner.end("j", lease.epoch + 1)
        self.owner.end("j", lease.epoch)
        self.assertIsNone(self.owner.leases.get("j"))


class TheRunnersEnd(unittest.TestCase):
    def setUp(self):
        self.clock, self.net = Clock(), Net()
        self.answers = []
        self.net.handlers["owner"] = self.owner_says
        self.guest = Guest(Lease("j", "gpu", 3, 30.), self.net.client_for("owner", retries=0), self.clock)

    def owner_says(self, method, path, body):
        answer = self.answers.pop(0) if self.answers else {}
        if isinstance(answer, Exception):
            raise answer
        return answer

    def renewals(self):
        return [(p, b) for _, p, b in self.net.to("owner")]

    def test_nothing_is_sent_before_a_third_of_the_lease_has_passed(self):
        self.assertIsNone(self.guest.renew(dict))
        self.clock.advance(9)
        self.assertIsNone(self.guest.renew(dict))
        self.assertEqual(self.renewals(), [])

    def test_a_renewal_carries_the_epoch_and_the_progress_and_repeats_every_third(self):
        self.clock.advance(10)
        taken = []
        self.assertIsNone(self.guest.renew(lambda: {"step": 7}, lambda: taken.append(1)))
        self.assertEqual(self.renewals(), [("cluster/jobs/j/renew", {"epoch": 3, "step": 7})])
        self.assertEqual(taken, [1])
        self.assertIsNone(self.guest.renew(lambda: {"step": 8}))  # not due yet
        self.clock.advance(10)
        self.guest.renew(lambda: {"step": 9})
        self.assertEqual(len(self.renewals()), 2)

    def test_a_409_means_the_job_has_moved_on(self):
        self.clock.advance(10)
        self.answers.append(refuse(409, "placed again"))
        self.assertEqual(self.guest.renew(dict), "stale")

    def test_renewals_that_fail_fence_the_job_once_the_lease_runs_out(self):
        self.net.down.add("owner")
        verdicts = []
        for _ in range(4):
            self.clock.advance(10)
            verdicts.append(self.guest.renew(dict))
        self.assertEqual(verdicts, [None, None, "fenced", "fenced"])  # 10 s, 20 s: still leased; 30 s: the lease is gone

    def test_one_missed_renewal_is_survived_when_the_next_gets_through(self):
        self.answers = [refuse(503, "busy"), {}, {}]
        verdicts = []
        for _ in range(5):
            self.clock.advance(10)
            verdicts.append(self.guest.renew(dict))
        self.assertEqual(verdicts, [None] * 5)

    def test_the_renewal_that_extends_the_lease_counts_from_when_it_was_sent(self):
        def slow(method, path, body):
            self.clock.advance(5)
            return {}
        self.net.handlers["owner"] = slow
        self.clock.advance(10)
        self.guest.renew(dict)
        self.assertEqual(self.guest.fence.remaining(), 25)

    def test_the_end_is_reported_with_the_epoch(self):
        self.assertEqual(self.guest.finish({"status": "done", "key": "k"}), "ok")
        self.assertEqual(self.renewals(), [("cluster/jobs/j/finished", {"epoch": 3, "status": "done", "key": "k"})])

    def test_the_end_of_a_job_that_moved_on_is_refused_and_a_dead_link_is_retried(self):
        self.answers = [refuse(409, "placed again")]
        self.assertEqual(self.guest.finish({"status": "done"}), "stale")
        self.net.down.add("owner")
        self.assertEqual(self.guest.finish({"status": "done"}), "retry")


class OwnerAndRunnerTogether(unittest.TestCase):
    """The invariant: a job never runs on two nodes at once. Both ends talk through the fake network, on one clock."""

    def setUp(self):
        self.clock, self.net = Clock(), Net()
        self.owner = Owner(self.net.client_for, self.clock, lease_s=30.)
        self.running = {}  # node -> the Guest whose job is running there
        self.net.handlers["owner"] = self.owner_handler
        for name in ("a", "b"):
            self.net.handlers[name] = self.peer_handler(name)

    def owner_handler(self, method, path, body):
        try:
            self.owner.renew("j", body["epoch"])
        except StaleEpoch as exc:
            raise refuse(409, str(exc)) from None
        return {}

    def peer_handler(self, name):
        def handle(method, path, body):
            lease = Lease.from_dict(body["lease"])
            self.running[name] = Guest(lease, self.net.client_for("owner", retries=0), self.clock)
            return {}
        return handle

    def step(self, seconds=1.):
        """Time passes; each runner renews when due and stops its job when told to or when its lease ran out; the owner ticks."""
        self.clock.advance(seconds)
        for name, guest in list(self.running.items()):
            if guest.renew(dict):
                del self.running[name]  # the job stopped on that node
        return self.owner.tick()

    def test_a_lost_node_is_fenced_before_the_owner_places_the_job_again(self):
        self.owner.dispatch("j", "a", {})
        for _ in range(12):
            self.step()
        self.assertIn("a", self.running)  # renewing happily
        self.net.down.update({"a", "owner"})  # a's link to the owner is cut (the owner cannot reach a either)
        stopped_at = reassigned_at = None
        for second in range(1, 90):
            actions = self.step()
            if "a" not in self.running and stopped_at is None:
                stopped_at = second
            if ("j", "reassign") in actions:
                reassigned_at = second
                break
        self.assertIsNotNone(stopped_at)
        self.assertLessEqual(stopped_at, 30)  # a gave up within one lease
        self.assertGreater(reassigned_at, 2 * 30 - 12)  # the owner waited for two
        self.assertLess(stopped_at, reassigned_at)  # so the old copy was gone first
        self.net.down.clear()
        self.assertEqual(self.owner.dispatch("j", "b", {}).lease.epoch, 2)
        self.assertEqual(set(self.running), {"b"})

    def test_a_node_that_comes_back_with_the_old_epoch_is_refused_and_stops(self):
        self.owner.dispatch("j", "a", {})
        a = self.running["a"]
        self.net.down.add("a")  # asleep: its clock does not move for the owner, who gives up on it
        self.clock.advance(61)
        self.owner.tick()
        self.owner.dispatch("j", "b", {})
        # a wakes up without having noticed time pass (its fence has not fired yet), and renews at epoch 1
        a.fence._deadline = self.clock() + 20
        a.fence._last_sent = self.clock() - 11
        self.assertEqual(a.renew(dict), "stale")
        self.assertEqual(self.running["b"].renew(dict), None)


class Resuming(unittest.TestCase):
    def params(self, **cluster):
        return {"config": {"steps": 1000, "batch_size": 8}, "examples": [{"text": "x"}], "cluster": {"steps": 1000, "offset": 0, **cluster}}

    def test_with_a_checkpoint_the_job_continues_from_it_with_the_steps_that_are_left(self):
        params, note, finished = sched.resume_params(self.params(checkpoint={"key": "box/v0-abc", "step": 400}), "gpu-box")
        self.assertFalse(finished)
        self.assertEqual((params["mode"], params["fetch"], params["config"]["steps"]), ("catch_up", {"init_from": "box/v0-abc"}, 600))
        self.assertEqual(params["cluster"]["offset"], 400)
        self.assertEqual(params["config"]["batch_size"], 8)
        self.assertIn("step 400", note)

    def test_only_the_node_that_wrote_the_checkpoint_can_continue_from_its_own_copy(self):
        notes = {"steps": 1000, "checkpoint": {"key": "k", "step": 400, "node": "gpu-box", "epoch": 2, "base": 100}}
        self.assertEqual(sched.resume_here(notes, "gpu-box"), {"epoch": 2, "steps": 900, "base": 100})
        self.assertIsNone(sched.resume_here(notes, "other-box"))
        self.assertIsNone(sched.resume_here({"steps": 1000, "checkpoint": {"key": "k", "step": 400}}, "gpu-box"))  # no record of where
        self.assertIsNone(sched.resume_here({"steps": 1000}, "gpu-box"))

    def test_a_second_interruption_before_a_new_checkpoint_resumes_from_the_same_one(self):
        first, _, _ = sched.resume_params(self.params(checkpoint={"key": "k", "step": 400}), "a")
        again, _, _ = sched.resume_params(first, "b")
        self.assertEqual((again["config"]["steps"], again["fetch"]), (600, {"init_from": "k"}))

    def test_a_later_checkpoint_counts_from_the_steps_already_done(self):
        first, _, _ = sched.resume_params(self.params(checkpoint={"key": "k", "step": 400}), "a")
        first["cluster"]["checkpoint"] = {"key": "k2", "step": 700}  # the second attempt's own, in absolute steps
        again, _, _ = sched.resume_params(first, "b")
        self.assertEqual((again["config"]["steps"], again["fetch"]), (300, {"init_from": "k2"}))

    def test_without_a_checkpoint_it_starts_again_and_says_so(self):
        params, note, finished = sched.resume_params(self.params(), "gpu-box")
        self.assertFalse(finished)
        self.assertNotIn("fetch", params)
        self.assertEqual(params["config"]["steps"], 1000)
        self.assertIn("before its first checkpoint", note)

    def test_a_checkpoint_at_the_last_step_means_the_job_had_finished(self):
        _, _, finished = sched.resume_params(self.params(checkpoint={"key": "k", "step": 1000}), "a")
        self.assertTrue(finished)

    def test_the_input_params_are_not_changed(self):
        original = self.params(checkpoint={"key": "k", "step": 400})
        sched.resume_params(original, "a")
        self.assertEqual(original["config"]["steps"], 1000)
        self.assertNotIn("mode", original)


class Relaying(unittest.TestCase):
    def test_only_events_that_make_sense_on_the_owner_are_passed_on(self):
        self.assertIsNotNone(sched.relayed({"type": "metric", "step": 5, "total": 10}))
        self.assertIsNone(sched.relayed({"type": "viz_snapshot", "step": 5, "path": "/runner/disk/x.npz"}))
        self.assertIsNone(sched.relayed({"type": "done", "directory": "/runner/disk"}))
        self.assertIsNone(sched.relayed("not an event"))

    def test_steps_continue_after_a_resume(self):
        self.assertEqual(sched.relayed({"type": "metric", "step": 5, "total": 600, "loss": 1.5}, 400),
                         {"type": "metric", "step": 405, "total": 1000, "loss": 1.5})
        self.assertEqual(sched.relayed({"type": "info", "text": "hello"}, 400), {"type": "info", "text": "hello"})


if __name__ == "__main__":
    unittest.main()
