import asyncio
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
import weakref
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException
from starlette.requests import Request

import studio_download_admission as admission
import studio_peer
from studio_api.routers.collective import binary


def request(**headers):
    return Request({'type': 'http', 'headers': [(k.encode(), v.encode()) for k, v in headers.items()]})


class Versions(unittest.TestCase):
    def test_same_size_version_change_rejected_and_legacy_unchanged(self):
        found = type('Coordinator', (), {'term': 1})()
        first = binary(found, b'abcd', request(**{'x-transfer-admission': '1', 'range': 'bytes=0-1'}))
        self.assertEqual(first.headers['etag'], '"' + hashlib.sha256(b'abcd').hexdigest() + '"')
        with self.assertRaises(HTTPException) as error:
            binary(found, b'ABCD', request(**{'x-transfer-admission': '1', 'range': 'bytes=2-3', 'if-match': first.headers['etag']}))
        self.assertEqual(error.exception.status_code, 412)
        legacy = binary(found, b'ABCD', request(range='bytes=2-3'))
        self.assertEqual(legacy.body, b'CD')
        self.assertNotIn('x-transfer-admission', legacy.headers)

    def test_shard_snapshot_survives_in_place_mutation(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'shard'
            path.write_bytes(b'abcd')
            first = admission.snapshot(path)
            path.write_bytes(b'ABCD')
            second = admission.snapshot(path)
            self.assertEqual(first.read(0, 4), b'abcd')
            self.assertEqual(second.read(0, 4), b'ABCD')
            self.assertNotEqual(first.sha256, second.sha256)

    def test_active_snapshots_stay_charged_after_cache_eviction(self):
        with tempfile.TemporaryDirectory() as root, patch.object(admission, '_FILE_BYTES', 4), \
                patch.object(admission, '_FILES', OrderedDict()), patch.object(admission, '_LIVE_FILES', weakref.WeakValueDictionary()):
            a, b = Path(root) / 'a', Path(root) / 'b'
            a.write_bytes(b'abcd')
            b.write_bytes(b'ABCD')
            first = admission.snapshot(a)
            with self.assertRaises(HTTPException) as error:
                admission.snapshot(b)
            self.assertEqual(error.exception.status_code, 429)
            self.assertIs(admission.snapshot(a), first)
            del first
            self.assertEqual(admission.snapshot(b).read(0, 4), b'ABCD')

    def test_concurrent_shard_requests_share_one_snapshot(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'shard'
            path.write_bytes(b'payload' * 4096)
            with ThreadPoolExecutor(max_workers=4) as pool:
                copies = list(pool.map(lambda _: admission.snapshot(path), range(4)))
            self.assertTrue(all(copy is copies[0] for copy in copies))


class Sending(unittest.IsolatedAsyncioTestCase):
    async def test_credit_held_until_cancelled_send_drains(self):
        budget = studio_peer.SenderEgressBudget(2)
        entered, drain = asyncio.Event(), asyncio.Event()
        async def send(message):
            if message['type'] == 'http.response.body' and message.get('body'):
                self.assertEqual(budget.outstanding, 2)
                entered.set()
                await drain.wait()
        async def receive():
            await asyncio.Future()
        response = admission.AdmittedResponse(b'abcd', 0, 3, headers={})
        with patch.object(studio_peer, '_SENDER_EGRESS', budget):
            task = asyncio.create_task(response({}, receive, send))
            await asyncio.wait_for(entered.wait(), 1)
            task.cancel()
            await asyncio.sleep(.02)
            self.assertEqual(budget.outstanding, 2)
            self.assertFalse(task.done())
            drain.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(budget.outstanding, 0)
            self.assertEqual(len(budget.queue), 0)

    async def test_disconnect_drains_active_send_before_releasing(self):
        budget = studio_peer.SenderEgressBudget(2)
        entered, disconnected, drain = asyncio.Event(), asyncio.Event(), asyncio.Event()
        bodies = []
        async def send(message):
            if message['type'] == 'http.response.body' and message.get('body'):
                bodies.append(message['body'])
                entered.set()
                await drain.wait()
        async def receive():
            await disconnected.wait()
            return {'type': 'http.disconnect'}
        with patch.object(studio_peer, '_SENDER_EGRESS', budget):
            task = asyncio.create_task(admission.AdmittedResponse(b'abcd', 0, 3, headers={})({}, receive, send))
            await asyncio.wait_for(entered.wait(), 1)
            disconnected.set()
            await asyncio.sleep(.02)
            self.assertEqual(budget.outstanding, 2)
            self.assertFalse(task.done())
            drain.set()
            await asyncio.wait_for(task, 1)
            self.assertEqual(bodies, [b'ab'])
            self.assertEqual(budget.outstanding, 0)

    async def test_waiting_cancel_removes_fifo_entry(self):
        budget = studio_peer.SenderEgressBudget(2)
        blocker = object()
        self.assertTrue(budget.acquire(blocker, 2))
        async def send(message):
            pass
        async def receive():
            await asyncio.Future()
        with patch.object(studio_peer, '_SENDER_EGRESS', budget):
            task = asyncio.create_task(admission.AdmittedResponse(b'ab', 0, 1, headers={})({}, receive, send))
            await asyncio.sleep(.02)
            self.assertEqual(len(budget.queue), 1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(len(budget.queue), 0)
            self.assertEqual(budget.outstanding, 2)
            budget.release(2)
