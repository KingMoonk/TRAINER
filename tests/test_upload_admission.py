"""Shared byte accounting and transfer fencing for receiver granted uploads."""
import hashlib
import unittest

from studio_upload_admission import AdmissionError, UploadAdmission


class UploadAdmissionTests(unittest.TestCase):
    def test_grants_are_exact_and_shared_outstanding_budget_is_not_exceeded(self):
        manager = UploadAdmission(max_assembly=100, max_outstanding=6, grant_bytes=4, ttl=10)
        payload = b"abcdefghij"
        identity = ("run", 1, 0, "worker", 1, 0, 3, 2, 0., None, 0.)
        announced = manager.announce(identity, len(payload), hashlib.sha256(payload).hexdigest(), 3, 4)
        batch = manager.grant(announced["transfer_id"], announced["incarnation"], identity)
        self.assertEqual([(g["start"], g["end"]) for g in batch["grants"]], [(0, 3)])
        self.assertEqual(manager.outstanding_bytes, 4)
        self.assertEqual(manager.grant(announced["transfer_id"], announced["incarnation"], identity)["grants"], [])
        manager.begin_part(announced["transfer_id"], announced["incarnation"], identity,
                           batch["grants"][0]["grant_id"], 0, 4)
        manager.finish_part(announced["transfer_id"], announced["incarnation"], identity,
                            batch["grants"][0]["grant_id"], 0, payload[:4], hashlib.sha256(payload[:4]).hexdigest())
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

    def test_restart_incarnation_fences_old_grants_and_manifest_identity(self):
        manager = UploadAdmission()
        identity = ("run", 1, 0, "worker", 1, 0, 3, 2, 0., None, 0.)
        announced = manager.announce(identity, 1, hashlib.sha256(b"x").hexdigest(), 1, 1)
        with self.assertRaises(AdmissionError):
            manager.grant(announced["transfer_id"], "old-incarnation", identity)
        with self.assertRaises(AdmissionError):
            manager.grant(announced["transfer_id"], announced["incarnation"], (*identity[:-1], 1.))


if __name__ == "__main__":
    unittest.main()
