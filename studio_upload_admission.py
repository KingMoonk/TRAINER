"""Process-wide admission for opt-in collective delta uploads.

The API currently runs one Uvicorn worker. Instances are attached to the app and
shared by every request in that process; deployments with multiple workers must
partition traffic or provide a shared admission service before enabling grants.
"""
from __future__ import annotations

import hashlib
import secrets
import threading
import time
from collections import deque

MAX_TRANSFER = 256 << 20
MAX_ASSEMBLY = 512 << 20
MAX_OUTSTANDING = 32 << 20
GRANT_BYTES = 4 << 20
GRANT_TTL = 30.
KEEP = 600.
MAX_TRANSFERS = 256
MAX_TRANSFERS_PER_NODE = 8
MAX_POLLS_PER_SECOND = 80
MAX_NODE_POLLS_PER_SECOND = 12


class AdmissionError(ValueError):
    pass


class UploadAdmission:
    def __init__(self, clock=time.monotonic, max_assembly=MAX_ASSEMBLY, max_outstanding=MAX_OUTSTANDING,
                 grant_bytes=GRANT_BYTES, ttl=GRANT_TTL, on_event=None):
        self.clock, self.max_assembly, self.max_outstanding = clock, max_assembly, max_outstanding
        self.grant_bytes, self.ttl = grant_bytes, ttl
        self.on_event = on_event
        self.lock = threading.RLock()
        self.incarnation = secrets.token_hex(12)
        self.transfers, self.grants = {}, {}
        self.by_identity, self.polls, self.node_polls = {}, deque(), {}
        self.assembly_bytes = self.outstanding_bytes = 0

    def _expire(self, now):
        for ident, transfer in list(self.transfers.items()):
            if now - transfer["touched"] <= KEEP:
                continue
            for grant_id in transfer["grants"]:
                grant = self.grants.pop(grant_id, None)
                if grant and grant["state"] in ("issued", "receiving"):
                    self.outstanding_bytes -= grant["size"]
            if not transfer["complete"]:
                self.assembly_bytes -= transfer["memory_reservation"]
            self.by_identity.pop(transfer["manifest_key"], None)
            del self.transfers[ident]
        for grant_id, grant in list(self.grants.items()):
            if grant["state"] == "issued" and grant["expires"] <= now:
                # No body has begun at the receiver, so this reservation can be
                # safely returned. A request arriving later is rejected pre-read.
                self.outstanding_bytes -= grant["size"]
                grant["state"] = "expired"

    def announce(self, identity, total, digest, count, part_size):
        if (not 0 < total <= MAX_TRANSFER or not 0 < count <= 4096 or
                not 0 < part_size <= min(MAX_TRANSFER, self.grant_bytes, self.max_outstanding)):
            raise AdmissionError("Transfer size or part geometry is outside the admitted bounds.")
        if count != (total + part_size - 1) // part_size or len(digest) != 64:
            raise AdmissionError("Transfer manifest has inconsistent part geometry or hash.")
        try:
            int(digest, 16)
        except ValueError:
            raise AdmissionError("Transfer hash must be SHA-256 hex.") from None
        now = self.clock()
        with self.lock:
            self._expire(now)
            manifest_key = (tuple(identity), total, digest.lower(), count, part_size)
            previous = self.by_identity.get(manifest_key)
            if previous in self.transfers:
                transfer = self.transfers[previous]
                transfer["touched"] = now
                return {"transfer_id": previous, "incarnation": self.incarnation, "part_size": part_size,
                        "complete": transfer["complete"], "assembled": transfer["assembled"]}
            node = identity[3]
            if (len(self.transfers) >= MAX_TRANSFERS or
                    sum(item["identity"][3] == node for item in self.transfers.values()) >= MAX_TRANSFERS_PER_NODE):
                raise AdmissionError("Receiver transfer admission is full; retry the announcement later.")
            ident = secrets.token_urlsafe(18)
            reservation = 2 * total + 3 * part_size  # parts, assembled body and bounded final-request buffers
            if self.assembly_bytes + reservation > self.max_assembly:
                raise AdmissionError("Receiver assembly-memory budget is full; retry the announcement later.")
            self.assembly_bytes += reservation
            self.transfers[ident] = {"identity": tuple(identity), "total": total, "digest": digest.lower(), "count": count,
                                     "part_size": part_size, "parts": {}, "grants": set(), "touched": now, "began": now,
                                     "assembled": False, "complete": False, "memory_reservation": reservation,
                                     "manifest_key": manifest_key,
                                     "last_poll": -float("inf")}
            self.by_identity[manifest_key] = ident
            self._emit("announced", transfer=self._public(ident, self.transfers[ident]))
            return {"transfer_id": ident, "incarnation": self.incarnation, "part_size": part_size}

    def resume(self, identity, total, digest, count, part_size):
        """Return an existing manifest after an ambiguous announcement/final acknowledgement."""
        key = (tuple(identity), total, digest.lower(), count, part_size)
        now = self.clock()
        with self.lock:
            self._expire(now)
            ident = self.by_identity.get(key)
            transfer = self.transfers.get(ident)
            if transfer is None:
                return None
            transfer["touched"] = now
            return {"transfer_id": ident, "incarnation": self.incarnation, "part_size": part_size,
                    "complete": transfer["complete"], "assembled": transfer["assembled"]}

    def grant(self, ident, incarnation, identity, limit=8):
        now = self.clock()
        with self.lock:
            self._expire(now)
            transfer = self._get(ident, incarnation, identity)
            transfer["touched"] = now
            node = transfer["identity"][3]
            while self.polls and now - self.polls[0] >= 1.:
                self.polls.popleft()
            node_history = self.node_polls.setdefault(node, deque())
            while node_history and now - node_history[0] >= 1.:
                node_history.popleft()
            if (now - transfer["last_poll"] < .05 or len(self.polls) >= MAX_POLLS_PER_SECOND or
                    len(node_history) >= MAX_NODE_POLLS_PER_SECOND):
                return {"grants": [], "outstanding_bytes": self.outstanding_bytes,
                        "assembly_bytes": self.assembly_bytes, "retry_after": .1}
            transfer["last_poll"] = now
            self.polls.append(now)
            node_history.append(now)
            grants, available = [], self.max_outstanding - self.outstanding_bytes
            for part in range(transfer["count"]):
                if part in transfer["parts"] or any(self.grants[g]["part"] == part and
                        self.grants[g]["state"] in ("issued", "receiving", "accepted")
                        for g in transfer["grants"] if g in self.grants):
                    continue
                start = part * transfer["part_size"]
                size = min(transfer["part_size"], transfer["total"] - start)
                if size > min(available, self.grant_bytes) or len(grants) >= limit:
                    break
                grant_id = secrets.token_urlsafe(18)
                grant = {"transfer": ident, "part": part, "start": start, "end": start + size - 1, "size": size,
                         "expires": now + self.ttl, "state": "issued", "digest": None}
                self.grants[grant_id] = grant
                transfer["grants"].add(grant_id)
                self.outstanding_bytes += size
                available -= size
                grants.append({"grant_id": grant_id, "part": part, "start": start, "end": start + size - 1, "expires_in": self.ttl})
            if grants:
                self._emit("granted", transfer=self._public(ident, transfer), ranges=grants,
                           outstanding_bytes=self.outstanding_bytes, assembly_bytes=self.assembly_bytes)
            return {"grants": grants, "outstanding_bytes": self.outstanding_bytes,
                    "assembly_bytes": self.assembly_bytes, "retry_after": .1 if not grants else 0.}

    def begin_part(self, ident, incarnation, identity, grant_id, part, size):
        now = self.clock()
        with self.lock:
            self._expire(now)
            transfer = self._get(ident, incarnation, identity)
            grant = self.grants.get(grant_id)
            if grant is None or grant["transfer"] != ident or grant["part"] != part:
                raise AdmissionError("Part has no matching receiver grant.")
            if grant["state"] == "accepted":
                return "duplicate"
            if grant["state"] != "issued" or grant["expires"] <= now:
                raise AdmissionError("Receiver grant expired or is already in use.")
            if size != grant["size"]:
                raise AdmissionError("Part size does not match its exact granted range.")
            grant["state"], transfer["touched"] = "receiving", now
            return "receive"

    def finish_part(self, ident, incarnation, identity, grant_id, part, data, part_digest):
        now = self.clock()
        with self.lock:
            transfer = self._get(ident, incarnation, identity)
            grant = self.grants[grant_id]
            if hashlib.sha256(data).hexdigest() != part_digest:
                grant["state"] = "issued"
                raise AdmissionError("Part hash does not match; retry the same grant.")
            grant["state"], grant["digest"] = "accepted", part_digest
            self.outstanding_bytes -= grant["size"]
            transfer["parts"][part] = bytes(data)
            transfer["touched"] = now
            if len(transfer["parts"]) != transfer["count"]:
                return None
            body = b"".join(transfer["parts"][index] for index in range(transfer["count"]))
            if len(body) != transfer["total"] or hashlib.sha256(body).hexdigest() != transfer["digest"]:
                self.assembly_bytes -= transfer["memory_reservation"]
                for issued_id in transfer["grants"]:
                    issued = self.grants.pop(issued_id, None)
                    if issued and issued["state"] in ("issued", "receiving"):
                        self.outstanding_bytes -= issued["size"]
                del self.transfers[ident]
                self.by_identity.pop(transfer["manifest_key"], None)
                raise AdmissionError("Assembled transfer does not match its announced size and hash.")
            transfer["assembled"] = True
            return body, max(1e-6, now - transfer["began"])

    def pending(self, ident, incarnation, identity):
        with self.lock:
            transfer = self._get(ident, incarnation, identity)
            if not transfer["assembled"] or transfer["complete"]:
                return None
            body = b"".join(transfer["parts"][index] for index in range(transfer["count"]))
            return body, max(1e-6, self.clock() - transfer["began"])

    def mark_complete(self, ident, incarnation, identity):
        with self.lock:
            transfer = self._get(ident, incarnation, identity)
            if transfer["complete"]:
                return
            transfer["complete"], transfer["touched"] = True, self.clock()
            self.assembly_bytes -= transfer["memory_reservation"]
            self._emit("accepted", transfer=self._public(ident, transfer),
                       duration_s=max(1e-6, transfer["touched"] - transfer["began"]))

    def cancel_part(self, grant_id):
        with self.lock:
            grant = self.grants.get(grant_id)
            if grant and grant["state"] == "receiving":
                grant["state"] = "issued"

    def cancel_transfer(self, ident, incarnation):
        with self.lock:
            if incarnation != self.incarnation:
                return
            transfer = self.transfers.pop(ident, None)
            if transfer is None:
                return
            for grant_id in transfer["grants"]:
                grant = self.grants.pop(grant_id, None)
                if grant and grant["state"] in ("issued", "receiving"):
                    self.outstanding_bytes -= grant["size"]
            if not transfer["complete"]:
                self.assembly_bytes -= transfer["memory_reservation"]
            self.by_identity.pop(transfer["manifest_key"], None)
            self._emit("fenced", transfer=self._public(ident, transfer))

    def fence_member(self, run, node, generation, epoch, term):
        """Discard reservations from an older run term or member lease after a new valid identity arrives."""
        with self.lock:
            stale = [ident for ident, transfer in self.transfers.items()
                     if transfer["identity"][0] == run and transfer["identity"][3] == node and
                     (transfer["identity"][1] != term or transfer["identity"][4] != generation or
                      transfer["identity"][5] != epoch)]
        for ident in stale:
            self.cancel_transfer(ident, self.incarnation)

    def _get(self, ident, incarnation, identity):
        if incarnation != self.incarnation:
            raise AdmissionError("Receiver restarted; announce a new transfer.")
        transfer = self.transfers.get(ident)
        if transfer is None or transfer["identity"] != tuple(identity):
            raise AdmissionError("Unknown or fenced transfer identity.")
        return transfer

    def capabilities(self):
        part = min(self.grant_bytes, self.max_outstanding)
        return {"protocol": 1, "max_transfer": min(MAX_TRANSFER, max(0, (self.max_assembly - 3 * part) // 2)),
                "max_part_size": part}

    def _public(self, ident, transfer):
        identity = transfer["identity"]
        return {"id": ident, "run": identity[0], "term": identity[1], "round": identity[2], "node": identity[3],
                "generation": identity[4], "epoch": identity[5], "bytes": transfer["total"]}

    def _emit(self, kind, **event):
        if self.on_event is not None:
            try:
                self.on_event(kind, event)
            except Exception:  # telemetry must not affect admission or contribution semantics
                pass
