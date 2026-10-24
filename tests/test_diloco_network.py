"""Actual exchange waits, dynamic required-work priority, and replica cancellation."""
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from studio_diloco import Background, Member
from studio_peer import Cancelled


class MemberNetworkTests(unittest.TestCase):
    def member(self, client, **kwargs):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        return Member(client, 'network', 'node', 0, Path(directory), threading.Event(), progress=False, **kwargs)

    def test_exchange_priority_rises_only_when_the_training_thread_joins(self):
        begun = threading.Event()
        events, priorities = [], []

        class Client:
            interruptible = supports_admission_priority = True

            def get_bytes(self, path, params, cancel, priority):
                priorities.append(priority())
                begun.set()
                for _ in range(100):
                    if priority():
                        priorities.append(True)
                        return b'weights'
                    threading.Event().wait(.01)
                raise AssertionError('Joining did not promote the actual exchange wait')

        member = self.member(Client(), emit=events.append)
        member.round, member.pending_round = 5, 4
        member.pending = Background(lambda: member._download('/params', {'round': 5}, priority=member.exchange_waiting.is_set))
        self.assertTrue(begun.wait(2))
        self.assertEqual(member._join(), b'weights')
        self.assertEqual(priorities, [False, True])
        self.assertFalse(member.exchange_waiting.is_set())
        event = next(event for event in events if event.get('stage') == 'exchange_join')
        self.assertEqual((event['compute_round'], event['exchange_round']), (5, 4))
        self.assertGreaterEqual(event['exposed_wait_s'], 0)

    def test_replica_demotion_cancels_state_without_retrying_or_failing_training(self):
        calls = []

        class Client:
            interruptible = supports_admission_priority = True

            def get_bytes(self, path, params, cancel, priority):
                calls.append((path, priority))
                self.member.replica_epoch += 1
                self.assert_cancelled = cancel()
                raise Cancelled('replica demoted')

        client = Client()
        member = self.member(client)
        client.member = member
        member._replicate(2, b'unused-theta')
        self.assertEqual(calls, [('/collective/network/state', False)])
        self.assertTrue(client.assert_cancelled)
        self.assertFalse(member.stop.is_set())
        self.assertIsNone(member.holding)

    def test_required_shard_promotes_an_existing_background_prefetch(self):
        begun = threading.Event()
        priorities = []

        class Client:
            interruptible = supports_admission_priority = True

            def get_bytes(self, path, params, cancel, priority):
                priorities.append(priority())
                begun.set()
                for _ in range(100):
                    if priority():
                        priorities.append(True)
                        return b'shard'
                    threading.Event().wait(.01)
                raise AssertionError('Required shard did not promote prefetch')

        member = self.member(Client())
        member.info = {'vocab_size': 128}
        with patch('studio_diloco.install_shard', return_value=[]):
            prefetch = Background(lambda: member._shard(0, priority=False))
            self.assertTrue(begun.wait(2))
            member._shard(0, priority=True)
            prefetch.result()
        self.assertEqual(priorities, [False, True])
        self.assertEqual(member.shard_demands, {})

    def test_exchange_join_records_wait_on_failure_and_always_clears_urgency(self):
        events = []
        member = self.member(types.SimpleNamespace(), emit=events.append)
        member.pending_round = 2

        def fail():
            raise RuntimeError('exchange failed')

        member.pending = Background(fail)
        with self.assertRaisesRegex(RuntimeError, 'exchange failed'):
            member._join()
        self.assertFalse(member.exchange_waiting.is_set())
        self.assertEqual(events[-1]['stage'], 'exchange_join')
