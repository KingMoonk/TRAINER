"""Sender admission state machine: cancellation, replay and ambiguous acknowledgements."""
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

from studio_peer import Cancelled, PeerClient, PeerError, SenderEgressBudget


class ScriptedPeer(PeerClient):
    def __init__(self, statuses):
        super().__init__('127.0.0.1', receiver_grants=True, streams=2)
        self.statuses = iter(statuses)
        self.sent, self.deleted = [], []

    def post_json(self, *args, **kwargs):
        return {'transfer_id': 'test', 'incarnation': 'one'}

    def get_json(self, *args, **kwargs):
        return next(self.statuses)

    def _send(self, method, path, params, *args, **kwargs):
        if method == 'DELETE':
            self.deleted.append(path)
            return b''
        self.sent.append(params['part'])
        return b'{}'

    def run(self, check=lambda: None):
        return self._run_granted_transfer('/up/transfers', {}, {}, 2., b'ab', 1, 2, check)


def grant(index):
    return {'part': index, 'grant_id': str(index)}


class SenderGrantTests(unittest.TestCase):
    def test_resume_uses_receiver_accepted_parts_and_waits_for_complete(self):
        peer = ScriptedPeer([{'accepted_parts': [0], 'grants': [grant(0), grant(1), grant(1)]},
                             {'accepted_parts': [0, 1], 'complete': True}])
        self.assertTrue(peer.run()['accepted'])
        self.assertEqual(peer.sent, [1])
        self.assertEqual(peer.deleted, [])

    def test_lost_part_ack_is_reconciled_without_blind_post_retry(self):
        peer = ScriptedPeer([{'grants': [grant(0)]}, {'accepted_parts': [0], 'grants': [grant(1)]},
                             {'complete': True}])
        original = peer._send

        def send(method, path, params, *args, **kwargs):
            answer = original(method, path, params, *args, **kwargs)
            if method == 'POST' and params['part'] == 0:
                raise PeerError('response lost')
            return answer

        peer._send = send
        peer.run()
        self.assertEqual(peer.sent, [0, 1])

    def test_cancel_drains_running_part_before_receiver_cancel_and_return(self):
        peer = ScriptedPeer([{'grants': [grant(0)]}])
        active, stopped = threading.Event(), threading.Event()
        original = peer._send

        def send(method, *args, **kwargs):
            if method == 'POST':
                active.set()
                time.sleep(.03)
                stopped.set()
            if method == 'DELETE':
                self.assertTrue(stopped.is_set())
            return original(method, *args, **kwargs)

        def check():
            if active.is_set():
                raise Cancelled('test stop')

        peer._send = send
        with self.assertRaises(Cancelled):
            peer.run(check)
        self.assertTrue(stopped.is_set())
        self.assertEqual(len(peer.deleted), 1)

    def test_empty_grants_remain_cancellable(self):
        peer = ScriptedPeer([{'grants': []}])
        checks = []

        def check():
            checks.append(1)
            if len(checks) > 2:
                raise Cancelled('test stop')

        with self.assertRaises(Cancelled):
            peer.run(check)
        self.assertEqual(peer.sent, [])
        self.assertEqual(len(peer.deleted), 1)

    def test_pre_cancelled_upload_never_announces_or_falls_back(self):
        peer = ScriptedPeer([])
        with self.assertRaises(Cancelled):
            peer.post_bytes('/up', b'a', cancel=lambda: True)
        self.assertEqual(peer.sent, [])

    def test_full_receiver_admission_retries_the_same_manifest(self):
        peer = ScriptedPeer([{'complete': True}])
        attempts = []
        manifest = {'sha256': 'same immutable hash'}

        def announce(route, body, **kwargs):
            attempts.append(body)
            if len(attempts) == 1:
                raise PeerError('temporarily full', status=429)
            return {'transfer_id': 'test', 'incarnation': 'one'}

        peer.post_json = announce
        peer._sleep = lambda _: None
        result = peer._run_granted_transfer('/up/transfers', manifest, {}, 2., b'ab', 1, 2)
        self.assertTrue(result['accepted'])
        self.assertEqual(attempts, [manifest, manifest])
        self.assertIs(attempts[0], attempts[1])

    def test_bulk_preserves_caller_socket_timeout_independent_of_total_deadline(self):
        peer = ScriptedPeer([{'grants': [grant(0), grant(1)]}, {'complete': True}])
        timeouts = []
        original = peer._send

        def send(method, path, params, *args, **kwargs):
            if method == 'POST':
                timeouts.append(args[2])
            return original(method, path, params, *args, **kwargs)

        peer._send = send
        peer._run_granted_transfer('/up/transfers', {}, {}, 120., b'ab', 1, 2)
        self.assertEqual(timeouts, [120., 120.])
        self.assertEqual(PeerClient('localhost', grant_deadline=900.).grant_deadline, 900.)


class SenderEgressTests(unittest.TestCase):
    def test_fifo_queue_and_cancelled_waiter_release_the_next_turn(self):
        budget = SenderEgressBudget(4)
        first, second, third = object(), object(), object()
        self.assertTrue(budget.acquire(first, 4))
        self.assertFalse(budget.acquire(second, 3))
        self.assertFalse(budget.acquire(third, 1))
        budget.release(4)
        self.assertFalse(budget.acquire(third, 1))
        budget.cancel(second)
        self.assertTrue(budget.acquire(third, 1))
        budget.release(1)
        self.assertEqual(budget.outstanding, 0)

    def test_concurrent_clients_share_credit_and_poll_control_while_waiting(self):
        budget = SenderEgressBudget(1)
        lock, barrier = threading.Lock(), threading.Barrier(2)
        state = {'active': 0, 'peak': 0, 'waiting_polls': 0}

        class Peer(ScriptedPeer):
            def __init__(self):
                super().__init__([])
                self.sender_egress = budget
                self.accepted = set()

            def get_json(self, *args, **kwargs):
                with lock:
                    if state['active'] and not self.accepted:
                        state['waiting_polls'] += 1
                    return {'accepted_parts': list(self.accepted), 'complete': len(self.accepted) == 2,
                            'grants': [grant(i) for i in range(2) if i not in self.accepted], 'retry_after': .02}

            def _send(self, method, path, params, *args, **kwargs):
                if method == 'DELETE':
                    return b''
                with lock:
                    state['active'] += 1
                    state['peak'] = max(state['peak'], state['active'])
                time.sleep(.04)
                with lock:
                    self.accepted.add(params['part'])
                    state['active'] -= 1
                return b'{}'

        def run(peer):
            barrier.wait()
            return peer.run()

        with ThreadPoolExecutor(max_workers=2) as pool:
            answers = list(pool.map(run, [Peer(), Peer()]))
        self.assertTrue(all(answer['accepted'] for answer in answers))
        self.assertEqual(state['peak'], 1)
        self.assertGreater(state['waiting_polls'], 0)
        self.assertEqual(budget.outstanding, 0)
        self.assertEqual(list(budget.queue), [])
