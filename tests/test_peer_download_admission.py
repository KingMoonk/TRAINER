"""Real HTTP download identity, shared ingress and drain guarantees for #651."""
import hashlib
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from studio_peer import Cancelled, PeerClient, PeerError, SenderEgressBudget, _Job, _read_exact
from tests.transfer_support import Peer, blob


class AdmittedPeer(Peer):
    def __init__(self, payload, **options):
        super().__init__(payload, **options)
        self.matches, self.priorities = [], []
        self.changed, self.corrupt = False, False

    def handler(self):
        base, peer = super().handler(), self

        class Handler(base):
            def reply(self, status, body, kind='application/octet-stream', headers=None):
                digest = hashlib.sha256(peer.blob).hexdigest()
                header = self.headers.get('Range', '')
                peer.matches.append(self.headers.get('If-Match'))
                peer.priorities.append(self.headers.get('X-Transfer-Priority'))
                etag = '"' + ('0' * 64 if peer.changed and not header.startswith('bytes=0-') else digest) + '"'
                if peer.corrupt and body:
                    body = bytes([body[0] ^ 1]) + body[1:]
                super().reply(status, body, kind, {**(headers or {}), 'X-Transfer-Admission': '1',
                                                  'ETag': etag, 'X-Content-SHA256': digest})

        return Handler


class AdmittedDownloadTests(unittest.TestCase):
    def test_memory_and_file_single_and_multiple_streams_pin_and_hash(self):
        payload = blob(17003)
        for streams in (1, 8):
            with self.subTest(streams=streams), AdmittedPeer(payload) as peer:
                client = PeerClient(peer.address, receiver_grants=True, streams=streams, part=4096)
                self.assertEqual(client.get_bytes('/blob', priority=True), payload)
                self.assertEqual(client.download_admission_mode, 'admitted')
                self.assertIsNone(peer.matches[0])
                self.assertTrue(all(value == '"' + hashlib.sha256(payload).hexdigest() + '"' for value in peer.matches[1:]))
                self.assertEqual(set(peer.priorities), {'required'})
                with tempfile.TemporaryDirectory() as directory:
                    target = Path(directory) / 'weights'
                    client.get_file('/blob', target)
                    self.assertEqual(target.read_bytes(), payload)

    def test_changed_identity_and_incorrect_whole_digest_are_rejected(self):
        for setting in ('changed', 'corrupt'):
            with self.subTest(setting=setting), AdmittedPeer(blob(11000)) as peer:
                setattr(peer, setting, True)
                client = PeerClient(peer.address, receiver_grants=True, part=4096, retries=0)
                with self.assertRaises(PeerError):
                    client.get_bytes('/blob')
                self.assertEqual(client.receiver_ingress.outstanding, 0)

    def test_first_part_retry_is_pinned_after_its_headers_arrive(self):
        # Inject a body read failure after identity headers, without changing
        # the representation. Every subsequent request must carry If-Match.
        with AdmittedPeer(blob(11000)) as peer:
            client = PeerClient(peer.address, receiver_grants=True, part=4096, retries=0)
            original = client._open
            failed = []

            def open_response(request, **kwargs):
                response = original(request, **kwargs)
                if not failed:
                    failed.append(True)
                    def read(*args):
                        raise OSError('injected interrupted probe')
                    response.read = read
                return response

            client._open = open_response
            self.assertEqual(client.get_bytes('/blob'), peer.blob)
            self.assertIsNone(peer.matches[0])
            self.assertTrue(all(value for value in peer.matches[1:]))

    def test_clients_share_ingress_even_with_single_stream_and_part_is_capped(self):
        budget = SenderEgressBudget(4096)
        with AdmittedPeer(blob(19000), latency=.01) as peer:
            clients = [PeerClient(peer.address, receiver_grants=True, streams=streams, part=10000,
                                  receiver_ingress=budget) for streams in (1, 8)]
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(lambda client: client.get_bytes('/blob'), clients))
            self.assertEqual(results, [peer.blob, peer.blob])
            self.assertEqual(peer.peak, 1)
            self.assertEqual(budget.outstanding, 0)
            self.assertEqual(list(budget.queue), [])

    def test_cancelled_file_drains_readers_before_credit_and_descriptor_release(self):
        budget = SenderEgressBudget(8192)
        with AdmittedPeer(blob(50000), rate=50000) as peer, tempfile.TemporaryDirectory() as directory:
            client = PeerClient(peer.address, receiver_grants=True, part=4096, receiver_ingress=budget)
            stopped = threading.Event()
            original = client._open
            calls = []

            def open_response(*args, **kwargs):
                response = original(*args, **kwargs)
                calls.append(1)
                if len(calls) >= 3:
                    stopped.set()
                return response

            client._open = open_response
            with self.assertRaises(Cancelled):
                client.get_file('/blob', Path(directory) / 'weights', cancel=stopped.is_set)
            self.assertEqual(budget.outstanding, 0)
            self.assertEqual(list(budget.queue), [])

    def test_legacy_fallback_is_explicit_and_upload_fallback_keeps_download_opt_in(self):
        with Peer(blob(15000)) as peer:
            client = PeerClient(peer.address, receiver_grants=True, part=4096)
            client.post_bytes('/up', b'x')  # capabilities 404 disables upload admission
            self.assertFalse(client.receiver_grants)
            self.assertTrue(client.receiver_downloads)
            self.assertEqual(client.get_bytes('/blob'), peer.blob)
            self.assertEqual(client.download_admission_mode, 'legacy-fallback')


class PriorityBudgetTests(unittest.TestCase):
    def test_required_gets_next_turn_then_older_background_progresses(self):
        budget = SenderEgressBudget(1)
        holder, background, required, another = [object() for _ in range(4)]
        self.assertTrue(budget.acquire(holder, 1))
        self.assertFalse(budget.acquire(background, 1))
        self.assertFalse(budget.acquire(required, 1, True))
        self.assertFalse(budget.acquire(another, 1, True))
        budget.release(1)
        self.assertTrue(budget.acquire(required, 1, True))
        budget.release(1)
        self.assertFalse(budget.acquire(another, 1, True))
        self.assertTrue(budget.acquire(background, 1))
        budget.release(1)
        self.assertTrue(budget.acquire(another, 1, True))
        budget.release(1)

    def test_queued_background_can_become_required_via_callback(self):
        budget = SenderEgressBudget(1)
        holder, background, promoted = [object() for _ in range(3)]
        required = threading.Event()
        self.assertTrue(budget.acquire(holder, 1))
        self.assertFalse(budget.acquire(background, 1))
        self.assertFalse(budget.acquire(promoted, 1, required.is_set))
        required.set()
        budget.release(1)
        self.assertTrue(budget.acquire(promoted, 1, required.is_set))
        budget.release(1)
        self.assertTrue(budget.acquire(background, 1))
        budget.release(1)


class DemandExpiryTests(unittest.TestCase):
    def test_dormant_waiter_expires_without_expiring_active_bytes(self):
        now = [0.]
        budget = SenderEgressBudget(4, demand_ttl=5., clock=lambda: now[0])
        active, dormant, live = object(), object(), object()
        self.assertTrue(budget.acquire(active, 4))
        self.assertFalse(budget.acquire(dormant, 4))
        now[0] = 4.
        self.assertFalse(budget.acquire(live, 4))
        now[0] = 6.
        self.assertFalse(budget.acquire(live, 4))
        self.assertEqual(budget.outstanding, 4)
        budget.release(4)
        self.assertTrue(budget.acquire(live, 4))
        budget.release(4)
        self.assertEqual(list(budget.queue), [])

    def test_aborted_range_job_cannot_read_or_retry_another_body(self):
        class Response:
            def read(self, *args):
                raise AssertionError('aborted job must not consume more bytes')

        job = _Job(None)
        job.abort.set()
        with self.assertRaises(Cancelled):
            job.check()
        with self.assertRaises(Cancelled):
            _read_exact(Response(), 1, job)
