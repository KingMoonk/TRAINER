"""Opt-in immutable download versions and process-shared ASGI egress admission.

Credits bound application writes, not physical link rate or kernel buffering.
"""
import asyncio
from collections import OrderedDict
import hashlib
import os
import tempfile
import threading
import weakref

import anyio
from fastapi import HTTPException, Response

import studio_peer

# Retain immutable byte objects to prevent id reuse. Bound both count and bytes.
_LOCK = threading.Lock()
_BUILD_LOCK = threading.Lock()
_BYTES = OrderedDict()
_FILES = OrderedDict()
_LIVE_FILES = weakref.WeakValueDictionary()
_CACHE_BYTES = 64 << 20
_FILE_BYTES = 512 << 20


def digest(content):
    with _LOCK:
        cached = _BYTES.get(id(content))
        if cached is not None and cached[0] is content:
            _BYTES.move_to_end(id(content))
            return cached[1]
    value = hashlib.sha256(content).hexdigest()
    if len(content) <= _CACHE_BYTES:
        with _LOCK:
            _BYTES[id(content)] = (content, value)
            while len(_BYTES) > 16 or sum(len(v[0]) for v in _BYTES.values()) > _CACHE_BYTES:
                _BYTES.popitem(last=False)
    return value


def _identity(stat):
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


class Snapshot:
    def __init__(self, path, expected=None):
        self.file = tempfile.TemporaryFile()
        hashed = hashlib.sha256()
        with open(path, 'rb') as source:
            before = _identity(os.fstat(source.fileno()))
            if expected is not None and before != expected:
                raise HTTPException(412, 'Shard changed before creating its snapshot.')
            if before[2] > _FILE_BYTES:
                raise HTTPException(413, 'Shard exceeds admitted snapshot capacity.')
            while chunk := source.read(1 << 20):
                if self.file.tell() + len(chunk) > before[2]:
                    raise HTTPException(412, 'Shard grew while creating its snapshot.')
                self.file.write(chunk)
                hashed.update(chunk)
            if before != _identity(os.fstat(source.fileno())) or before != _identity(os.stat(path)):
                raise HTTPException(412, 'Shard changed while creating its immutable snapshot.')
        self.file.flush()
        self.size, self.sha256 = before[2], hashed.hexdigest()

    def __del__(self):
        self.file.close()

    def read(self, offset, size):
        return os.pread(self.file.fileno(), size, offset)


def snapshot(path):
    # Serialize lookup/build so simultaneous readers share one immutable copy.
    # API callers wait for their build limiter asynchronously before entering here.
    with _BUILD_LOCK:
        key = (os.fspath(path), _identity(os.stat(path)))
        with _LOCK:
            found = _LIVE_FILES.get(key)
            if found is not None:
                if key in _FILES:
                    _FILES.move_to_end(key)
                return found
            size = key[1][2]
            if size > _FILE_BYTES:
                raise HTTPException(413, 'Shard exceeds admitted snapshot capacity.')
            while _FILES and sum(item.size for item in _LIVE_FILES.values()) + size > _FILE_BYTES:
                _FILES.popitem(last=False)
            if sum(item.size for item in _LIVE_FILES.values()) + size > _FILE_BYTES:
                raise HTTPException(429, 'Admitted shard snapshot storage is full; retry later.')
        found = Snapshot(path, expected=key[1])
        with _LOCK:
            _FILES[key] = found
            _LIVE_FILES[key] = found
            while len(_FILES) > 8 or sum(v.size for v in _FILES.values()) > _FILE_BYTES:
                _FILES.popitem(last=False)
        return found


async def _drained_send(send, message):
    # Cancellation cannot make already dispatched bytes disappear. Drain the send
    # before its credit is returned, even with raw asyncio task cancellation.
    with anyio.CancelScope(shield=True):
        task = asyncio.ensure_future(send(message))
        cancelled = False
        while True:
            try:
                await asyncio.shield(task)
                break
            except asyncio.CancelledError:
                if task.done():
                    raise
                cancelled = True
        if cancelled:
            raise asyncio.CancelledError


class AdmittedResponse(Response):
    def __init__(self, source, first, last, *, headers, status_code=206, priority=False):
        self.source, self.first, self.last = source, first, last
        self.priority = priority
        super().__init__(b'', status_code=status_code, media_type='application/octet-stream', headers=headers)
        self.headers['content-length'] = str(max(0, last - first + 1))

    async def __call__(self, scope, receive, send):
        budget = studio_peer._SENDER_EGRESS
        owner = object()

        async def transmit():
            await send({'type': 'http.response.start', 'status': self.status_code, 'headers': self.raw_headers})
            offset = self.first
            while offset <= self.last:
                await anyio.lowlevel.checkpoint()
                size = min(1 << 20, budget.capacity, self.last - offset + 1)
                while not budget.acquire(owner, size, priority=self.priority):
                    await anyio.sleep(.01)
                try:
                    if isinstance(self.source, bytes):
                        chunk = self.source[offset:offset + size]
                    else:
                        chunk = await anyio.to_thread.run_sync(self.source.read, offset, size)
                    if len(chunk) != size:
                        raise RuntimeError('Immutable download snapshot truncated.')
                    await _drained_send(send, {'type': 'http.response.body', 'body': chunk, 'more_body': True})
                finally:
                    budget.release(size)
                offset += size
            await send({'type': 'http.response.body', 'body': b'', 'more_body': False})

        async def disconnect(group):
            while True:
                if (await receive())['type'] == 'http.disconnect':
                    group.cancel_scope.cancel()
                    return

        try:
            async with anyio.create_task_group() as group:
                group.start_soon(disconnect, group)
                await transmit()
                group.cancel_scope.cancel()
        finally:
            budget.cancel(owner)


def response(source, request, headers):
    import studio_parts
    size = len(source) if isinstance(source, bytes) else source.size
    sha256 = digest(source) if isinstance(source, bytes) else source.sha256
    etag = '"' + sha256 + '"'
    headers = {**headers, 'ETag': etag, 'X-Content-SHA256': sha256,
               'X-Transfer-Admission': '1', 'Accept-Ranges': 'bytes'}
    condition = request.headers.get('if-match')
    if condition is not None and condition != '*' and etag not in [v.strip() for v in condition.split(',')]:
        raise HTTPException(412, 'Download version changed.', headers=headers)
    try:
        span = studio_parts.byte_range(request.headers.get('range'), size)
    except studio_parts.BadPart as exc:
        raise HTTPException(416, str(exc), headers={**headers, 'Content-Range': f'bytes */{size}'}) from None
    priority = request.headers.get("x-transfer-priority") == "required"
    if span is None:
        return AdmittedResponse(source, 0, size - 1, headers=headers, status_code=200, priority=priority)
    first, last = span
    return AdmittedResponse(source, first, last, headers={**headers, 'Content-Range': f'bytes {first}-{last}/{size}'}, priority=priority)
