"""Shared byte accounting and transfer fencing for receiver granted uploads."""
import hashlib
import unittest

from studio_upload_admission import AdmissionError, UploadAdmission
from studio_peer import SenderEgressBudget


class UploadAdmissionTests(unittest.TestCase):
    def test_grants_are_exact_and_shared_outstanding_budget_is_not_exceeded(self):
        manager = UploadAdmission(max_assembly=100, max_outstanding=6, grant_bytes=4, ttl=10)
        payload = b"abcdefghij"
        identity = ("run", 1, 0, "worker", 1, 0, 3, 2, 0., None, 0.)
        announced = manager.announce(identity, len(payload), hashlib.sha256(payload).hexdigest(), 3, 4)
        batch = manager.grant(announced["transfer_id"], announced["incarnation"], identity)
        self.assertEqual([(g["start"], g["end"]) for g in batch["grants"]], [(0, 3)])
        self.assertEqual(manager.outstanding_bytes, 4)
        self.assertEqual(len(manager.grant(announced["transfer_id"], announced["incarnation"], identity)["grants"]), 1)
        manager.begin_part(announced["transfer_id"], announced["incarnation"], identity,
                           batch["grants"][0]["grant_id"], 0, 4)
        manager.finish_part(announced["transfer_id"], announced["incarnation"], identity,
                            batch["grants"][0]["grant_id"], 0, payload[:4], hashlib.sha256(payload[:4]).hexdigest())
        manager.end_part(batch["grants"][0]["grant_id"])
        self.assertEqual(manager.outstanding_bytes, 0)

    def test_expired_unstarted_grant_returns_capacity_and_can_be_reissued(self):
        now = [0.]
        manager = UploadAdmission(clock=lambda: now[0], max_assembly=100, max_outstanding=4, grant_bytes=4, ttl=2)
        identity = ("run", 1, 0, "worker", 1, 0, 3, 2, 0., None, 0.)
        announced = manager.announce(identity, 4, hashlib.sha256(b"data").hexdigest(), 1, 4)
        old = manager.grant(announced["transfer_id"], announced["incarnation"], identity)["grants"][0]
        now[0] = 3.
        new = manager.grant(announced["transfer_id"], announced["incarnation"], identity)["grants"][0]
        self.assertNotEqual(old["grant_id"], new["grant_id"])
        self.assertEqual(manager.outstanding_bytes, 4)

    def setup_transfer(self, manager, node="a", body=b"abcdefgh", part=4, number=0):
        identity = ("run", 1, number, node, 1, 0, 3, 2, 0., None, 0.)
        transfer = manager.announce(identity, len(body), hashlib.sha256(body).hexdigest(), -(-len(body) // part), part)
        return identity, transfer["transfer_id"], transfer["incarnation"]

    def test_poll_replays_lost_grants_without_charging_twice(self):
        manager = UploadAdmission(max_assembly=1000, max_outstanding=16, grant_bytes=4)
        identity, ident, incarnation = self.setup_transfer(manager)
        first = manager.grant(ident, incarnation, identity)
        again = manager.grant(ident, incarnation, identity)
        self.assertEqual(first["grants"][0]["grant_id"], again["grants"][0]["grant_id"])
        self.assertEqual(manager.outstanding_bytes, 4)

    def test_idle_sender_does_not_block_another_senders_window(self):
        manager = UploadAdmission(max_assembly=1000, max_outstanding=8, grant_bytes=4)
        a, aid, inc = self.setup_transfer(manager)
        b, bid, _ = self.setup_transfer(manager, "b")
        self.assertEqual(len(manager.grant(aid, inc, a)["grants"]), 1)
        self.assertEqual(len(manager.grant(bid, inc, b)["grants"]), 1)
        self.assertEqual(manager.outstanding_bytes, 8)

    def test_oldest_demand_gets_released_capacity_without_polling_first(self):
        manager = UploadAdmission(max_assembly=1000, max_outstanding=4, grant_bytes=4)
        a, aid, inc = self.setup_transfer(manager, body=b"abcd")
        b, bid, _ = self.setup_transfer(manager, "b")
        c, cid, _ = self.setup_transfer(manager, "c")
        grant = manager.grant(aid, inc, a)["grants"][0]
        manager.grant(bid, inc, b)
        manager.grant(cid, inc, c)
        manager.begin_part(aid, inc, a, grant["grant_id"], 0, 4)
        manager.finish_part(aid, inc, a, grant["grant_id"], 0, b"abcd", hashlib.sha256(b"abcd").hexdigest())
        manager.mark_complete(aid, inc, a)
        manager.end_apply(aid)
        manager.end_part(grant["grant_id"])
        self.assertEqual(manager.grant(cid, inc, c)["grants"], [])
        self.assertEqual(len(manager.grant(bid, inc, b)["grants"]), 1)

    def test_cancelled_or_expired_receiving_request_keeps_credit_until_it_drains(self):
        for expire in (False, True):
            now = [0.]
            manager = UploadAdmission(clock=lambda: now[0], max_assembly=1000, max_outstanding=4, grant_bytes=4)
            identity, ident, inc = self.setup_transfer(manager)
            grant = manager.grant(ident, inc, identity)["grants"][0]
            manager.begin_part(ident, inc, identity, grant["grant_id"], 0, 4)
            reserved = manager.assembly_bytes
            if expire:
                now[0] = 601.
                manager._expire(now[0])
            else:
                manager.cancel_transfer(ident, inc)
            self.assertEqual((manager.outstanding_bytes, manager.assembly_bytes), (4, reserved))
            with self.assertRaises(AdmissionError):
                manager.finish_part(ident, inc, identity, grant["grant_id"], 0, b"abcd", hashlib.sha256(b"abcd").hexdigest())
            manager.end_part(grant["grant_id"])
            manager.end_part(grant["grant_id"])
            self.assertEqual((manager.outstanding_bytes, manager.assembly_bytes), (0, 0))

    def test_completed_receipt_drops_body_only_after_applying_and_receiving_drain(self):
        manager = UploadAdmission(max_assembly=1000, max_outstanding=4, grant_bytes=4)
        identity, ident, inc = self.setup_transfer(manager, body=b"data")
        grant = manager.grant(ident, inc, identity)["grants"][0]
        manager.begin_part(ident, inc, identity, grant["grant_id"], 0, 4)
        self.assertEqual(manager.finish_part(ident, inc, identity, grant["grant_id"], 0, b"data",
                                            hashlib.sha256(b"data").hexdigest())[0], b"data")
        self.assertEqual(manager.pending(ident, inc, identity)[0], b"data")
        self.assertIsNone(manager.pending(ident, inc, identity))
        manager.mark_complete(ident, inc, identity)
        manager.end_apply(ident)
        self.assertGreater(manager.assembly_bytes, 0)
        manager.end_part(grant["grant_id"])
        self.assertEqual(manager.assembly_bytes, 0)
        self.assertEqual(manager.transfers[ident]["parts"], {})
        self.assertIsNone(manager.transfers[ident]["body"])
        status = manager.grant(ident, inc, identity)
        self.assertTrue(status["complete"])
        self.assertEqual(status["accepted_parts"], [0])

    def test_part_limit_fits_turn_and_manifest_cannot_change_under_same_contribution(self):
        manager = UploadAdmission(grant_bytes=8, turn_bytes=4)
        self.assertEqual(manager.capabilities()["max_part_size"], 4)
        identity, _, _ = self.setup_transfer(manager)
        with self.assertRaises(AdmissionError):
            manager.announce(identity, 8, hashlib.sha256(b"different").hexdigest(), 2, 4)

    def test_required_upload_gets_next_turn_and_older_background_then_progresses(self):
        manager = UploadAdmission(max_assembly=1000, max_outstanding=4, grant_bytes=4)
        a, aid, inc = self.setup_transfer(manager, body=b'abcdefgh')
        b, bid, _ = self.setup_transfer(manager, 'b', body=b'ABCDEFGH')
        first = manager.grant(aid, inc, a)['grants'][0]
        self.assertEqual(manager.grant(bid, inc, b, priority=True)['grants'], [])

        def accept(identity, ident, grant, body):
            manager.begin_part(ident, inc, identity, grant['grant_id'], grant['part'], len(body))
            manager.finish_part(ident, inc, identity, grant['grant_id'], grant['part'], body, hashlib.sha256(body).hexdigest())
            manager.end_part(grant['grant_id'])

        accept(a, aid, first, b'abcd')
        required = manager.grant(bid, inc, b, priority=True)['grants'][0]
        accept(b, bid, required, b'ABCD')
        older = manager.grant(aid, inc, a)['grants'][0]
        self.assertEqual(older['part'], 1)

    def test_restart_incarnation_fences_old_grants_and_manifest_identity(self):
        manager = UploadAdmission()
        identity = ("run", 1, 0, "worker", 1, 0, 3, 2, 0., None, 0.)
        announced = manager.announce(identity, 1, hashlib.sha256(b"x").hexdigest(), 1, 1)
        with self.assertRaises(AdmissionError):
            manager.grant(announced["transfer_id"], "old-incarnation", identity)
        with self.assertRaises(AdmissionError):
            manager.grant(announced["transfer_id"], announced["incarnation"], (*identity[:-1], 1.))

    def test_serviced_transfer_yields_next_window_to_waiting_peer(self):
        manager = UploadAdmission(max_assembly=1000, max_outstanding=4, grant_bytes=4)
        a, aid, inc = self.setup_transfer(manager)
        b, bid, _ = self.setup_transfer(manager, "b")
        grant = manager.grant(aid, inc, a)["grants"][0]
        self.assertEqual(manager.grant(bid, inc, b)["grants"], [])
        manager.begin_part(aid, inc, a, grant["grant_id"], 0, 4)
        manager.finish_part(aid, inc, a, grant["grant_id"], 0, b"abcd", hashlib.sha256(b"abcd").hexdigest())
        manager.end_part(grant["grant_id"])
        # Even when A polls first, B's older queued demand receives the window.
        self.assertEqual(manager.grant(aid, inc, a)["grants"], [])
        self.assertEqual(len(manager.grant(bid, inc, b)["grants"]), 1)

    def assembled(self):
        manager = UploadAdmission(max_assembly=1000, max_outstanding=4, grant_bytes=4)
        identity, ident, inc = self.setup_transfer(manager, body=b"data")
        grant = manager.grant(ident, inc, identity)["grants"][0]
        manager.begin_part(ident, inc, identity, grant["grant_id"], 0, 4)
        manager.finish_part(ident, inc, identity, grant["grant_id"], 0, b"data", hashlib.sha256(b"data").hexdigest())
        manager.end_part(grant["grant_id"])
        return manager, identity, ident, inc

    def test_assembled_body_without_request_has_no_application_claim_and_can_resume(self):
        manager, identity, ident, inc = self.assembled()
        self.assertFalse(manager.transfers[ident]["applying"])
        self.assertEqual(manager.pending(ident, inc, identity)[0], b"data")
        self.assertIsNone(manager.pending(ident, inc, identity))
        manager.end_apply(ident, failed=True)
        self.assertEqual(manager.assembly_bytes, 0)

    def test_cancel_after_application_claim_cannot_destroy_commit_receipt(self):
        manager, identity, ident, inc = self.assembled()
        manager.pending(ident, inc, identity)
        manager.cancel_transfer(ident, inc)
        self.assertFalse(manager.transfers[ident]["cancelled"])
        manager.mark_complete(ident, inc, identity)
        manager.end_apply(ident)
        self.assertTrue(manager.completed(ident, inc, identity))
        self.assertEqual(manager.assembly_bytes, 0)

    def test_existing_manifest_resume_is_subject_to_announcement_limit(self):
        manager = UploadAdmission(clock=lambda: 0.)
        identity, ident, inc = self.setup_transfer(manager)
        transfer = manager.transfers[ident]
        with self.assertRaises(AdmissionError):
            for _ in range(100):
                manager.resume(identity, 8, transfer["digest"], 2, 4)

    def test_upload_and_local_download_demand_share_ingress_and_retry_reacquires(self):
        budget = SenderEgressBudget(4)
        manager = UploadAdmission(max_assembly=1000, max_outstanding=16, grant_bytes=8, ingress_budget=budget)
        self.assertEqual(manager.capabilities()["max_part_size"], 4)
        identity, ident, inc = self.setup_transfer(manager)
        download = object()
        self.assertTrue(budget.acquire(download, 4))
        self.assertEqual(manager.grant(ident, inc, identity)["grants"], [])
        budget.release(4)
        grant = manager.grant(ident, inc, identity)["grants"][0]
        self.assertEqual(budget.outstanding, 4)
        self.assertFalse(budget.acquire(download, 4))
        manager.begin_part(ident, inc, identity, grant["grant_id"], 0, 4)
        manager.end_part(grant["grant_id"])  # failed/disconnected read
        self.assertEqual((manager.outstanding_bytes, budget.outstanding), (0, 0))
        self.assertTrue(budget.acquire(download, 4))
        self.assertEqual(manager.grant(ident, inc, identity)["grants"], [])
        budget.release(4)
        retry = manager.grant(ident, inc, identity)["grants"][0]
        self.assertNotEqual(grant["grant_id"], retry["grant_id"])
        self.assertEqual(budget.outstanding, 4)
        manager.cancel_transfer(ident, inc)
        self.assertEqual(budget.outstanding, 0)
        self.assertEqual(list(budget.queue), [])

    def test_shared_ingress_remains_reserved_until_cancelled_request_drains(self):
        budget = SenderEgressBudget(4)
        manager = UploadAdmission(max_assembly=1000, max_outstanding=4, grant_bytes=4, ingress_budget=budget)
        identity, ident, inc = self.setup_transfer(manager)
        grant = manager.grant(ident, inc, identity)["grants"][0]
        manager.begin_part(ident, inc, identity, grant["grant_id"], 0, 4)
        manager.cancel_transfer(ident, inc)
        self.assertEqual(budget.outstanding, 4)
        manager.end_part(grant["grant_id"])
        manager.end_part(grant["grant_id"])
        self.assertEqual(budget.outstanding, 0)

    def test_unstarted_expired_grant_releases_external_credit_before_reissue(self):
        now = [0.]
        budget = SenderEgressBudget(4, clock=lambda: now[0])
        manager = UploadAdmission(clock=lambda: now[0], max_assembly=1000, max_outstanding=4,
                                  grant_bytes=4, ttl=2, ingress_budget=budget)
        identity, ident, inc = self.setup_transfer(manager)
        old = manager.grant(ident, inc, identity)["grants"][0]
        now[0] = 3.
        manager._expire(now[0])
        self.assertEqual((manager.outstanding_bytes, budget.outstanding), (0, 0))
        new = manager.grant(ident, inc, identity)["grants"][0]
        self.assertNotEqual(old["grant_id"], new["grant_id"])
        self.assertEqual(budget.outstanding, 4)
        manager.cancel_transfer(ident, inc)
        self.assertEqual(budget.outstanding, 0)


if __name__ == "__main__":
    unittest.main()
