"""Process-wide credit for opt-in delta uploads; one owner per API process.

A reservation outlives cancellation while a request holds its body. Multi-worker
servers must partition budgets or supply shared ownership before enabling this.
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
RECEIVE_TIMEOUT = 120.  # separate from unused-credit expiry; active requests retain their reservations
KEEP = 600.
MAX_TRANSFERS = 256
MAX_TRANSFERS_PER_NODE = 8
MAX_POLLS_PER_SECOND = 400
MAX_NODE_POLLS_PER_SECOND = 80


class AdmissionError(ValueError):
    pass


class UploadAdmission:
    def __init__(self, clock=time.monotonic, max_assembly=MAX_ASSEMBLY, max_outstanding=MAX_OUTSTANDING,
                 grant_bytes=GRANT_BYTES, turn_bytes=None, ttl=GRANT_TTL, on_event=None):
        if min(max_assembly, max_outstanding, grant_bytes, ttl, grant_bytes if turn_bytes is None else turn_bytes) <= 0:
            raise ValueError("Admission budgets and lifetimes must be positive.")
        self.clock, self.max_assembly, self.max_outstanding = clock, max_assembly, max_outstanding
        self.grant_bytes, self.ttl = grant_bytes, ttl
        self.turn_bytes = min(max_outstanding, grant_bytes if turn_bytes is None else turn_bytes)
        self.on_event = on_event
        self.lock = threading.RLock()
        self.incarnation = secrets.token_hex(12)
        self.transfers, self.grants, self.by_identity = {}, {}, {}
        self.polls, self.node_polls = deque(), {}
        self.announcements, self.node_announcements = deque(), {}
        self.grant_turns = deque()
        self.assembly_bytes = self.outstanding_bytes = 0

    def _limited(self, now, node, global_history, histories):
        for key, history in list(histories.items()):
            while history and now - history[0] >= 1.:
                history.popleft()
            if not history:
                del histories[key]
        while global_history and now - global_history[0] >= 1.:
            global_history.popleft()
        history = histories.setdefault(node, deque())
        if len(global_history) >= MAX_POLLS_PER_SECOND or len(history) >= MAX_NODE_POLLS_PER_SECOND:
            return True
        global_history.append(now)
        history.append(now)
        return False

    def _active(self, transfer):
        return any(self.grants[g]["active"] for g in transfer["grants"] if g in self.grants)

    def _release_memory(self, transfer):
        if transfer["reserved"] and not self._active(transfer) and not transfer["applying"]:
            self.assembly_bytes -= transfer["memory_reservation"]
            transfer["reserved"] = False
            transfer["parts"].clear()
            transfer["body"] = None

    def _forget(self, ident):
        transfer = self.transfers.pop(ident)
        for grant_id in transfer["grants"]:
            self.grants.pop(grant_id, None)
        if self.by_identity.get(transfer["manifest_key"]) == ident:
            self.by_identity.pop(transfer["manifest_key"], None)
        self._drop_turn(ident)

    def _expire(self, now):
        for ident, transfer in list(self.transfers.items()):
            if now - transfer["touched"] > KEEP:
                self.cancel_transfer(ident, self.incarnation, reason="idle_expiry")
                if not transfer["reserved"]:
                    self._forget(ident)
        for grant_id, grant in list(self.grants.items()):
            if grant["state"] == "issued" and grant["expires"] <= now:
                self.outstanding_bytes -= grant["size"]
                self.transfers[grant["transfer"]]["grants"].discard(grant_id)
                del self.grants[grant_id]
                self._emit("grant_expired", transfer_id=grant["transfer"], bytes=grant["size"])

    def announce(self, identity, total, digest, count, part_size):
        if (not 0 < total <= MAX_TRANSFER or not 0 < count <= 4096 or
                not 0 < part_size <= min(self.grant_bytes, self.max_outstanding, self.turn_bytes)):
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
            node = identity[3]
            if self._limited(now, node, self.announcements, self.node_announcements):
                raise AdmissionError("Receiver announcement admission is full; retry later.")
            existing = self.resume(identity, total, digest, count, part_size, limited=False)
            if existing is not None:
                return existing
            if any(t["identity"][:6] == tuple(identity)[:6] and not t["cancelled"] for t in self.transfers.values()):
                raise AdmissionError("An immutable manifest already exists for this contribution.")
            live = [t for t in self.transfers.values() if t["reserved"]]
            if len(live) >= MAX_TRANSFERS or sum(t["identity"][3] == node for t in live) >= MAX_TRANSFERS_PER_NODE:
                raise AdmissionError("Receiver transfer admission is full; retry the announcement later.")
            while len(self.transfers) >= MAX_TRANSFERS:
                disposable = next((key for key, t in self.transfers.items() if not t["reserved"]), None)
                if disposable is None:
                    raise AdmissionError("Receiver transfer admission is full; retry later.")
                self._forget(disposable)
            ident = secrets.token_urlsafe(18)
            reservation = 3 * total + 3 * part_size  # concurrent read buffers, parts, one assembled body
            if self.assembly_bytes + reservation > self.max_assembly:
                raise AdmissionError("Receiver assembly-memory budget is full; retry the announcement later.")
            manifest_key = (tuple(identity), total, digest.lower(), count, part_size)
            self.assembly_bytes += reservation
            self.transfers[ident] = {"identity": tuple(identity), "total": total, "digest": digest.lower(), "count": count,
                                     "part_size": part_size, "parts": {}, "grants": set(), "touched": now, "began": now,
                                     "assembled": False, "complete": False, "cancelled": False, "reserved": True,
                                     "body": None, "applying": False, "memory_reservation": reservation,
                                     "manifest_key": manifest_key, "last_poll": -float("inf"), "first_grant": None}
            self.by_identity[manifest_key] = ident
            self._emit("announced", transfer=self._public(ident, self.transfers[ident]))
            return {"transfer_id": ident, "incarnation": self.incarnation, "part_size": part_size}

    def resume(self, identity, total, digest, count, part_size, limited=True):
        key = (tuple(identity), total, digest.lower(), count, part_size)
        with self.lock:
            self._expire(self.clock())
            if limited and self._limited(self.clock(), identity[3], self.announcements, self.node_announcements):
                raise AdmissionError("Receiver announcement admission is full; retry later.")
            ident = self.by_identity.get(key)
            transfer = self.transfers.get(ident)
            if transfer is None or transfer["cancelled"]:
                return None
            transfer["touched"] = self.clock()
            return {"transfer_id": ident, "incarnation": self.incarnation, "part_size": part_size,
                    "complete": transfer["complete"], "assembled": transfer["assembled"]}

    def _schedule(self, now):
        # Allocate for queued peers even if they are not the current caller. A
        # dormant caller can hold only its own bounded window until expiry.
        for _ in range(len(self.grant_turns)):
            ident = self.grant_turns.popleft()
            transfer = self.transfers.get(ident)
            if (transfer is None or transfer["complete"] or transfer["cancelled"] or
                    now - transfer["last_poll"] > min(1., self.ttl)):
                continue
            live = [self.grants[g] for g in transfer["grants"] if g in self.grants]
            covered = set(transfer["parts"]) | {g["part"] for g in live}
            used = sum(g["size"] for g in live if g["state"] == "issued" or g["active"])
            available = min(self.turn_bytes - used, self.max_outstanding - self.outstanding_bytes)
            issued = []
            for part in range(transfer["count"]):
                if part in covered:
                    continue
                start = part * transfer["part_size"]
                size = min(transfer["part_size"], transfer["total"] - start)
                if size > available:
                    break
                grant_id = secrets.token_urlsafe(18)
                grant = {"transfer": ident, "part": part, "start": start, "end": start + size - 1, "size": size,
                         "expires": now + self.ttl, "state": "issued", "active": False, "digest": None}
                self.grants[grant_id] = grant
                transfer["grants"].add(grant_id)
                self.outstanding_bytes += size
                available -= size
                covered.add(part)
                issued.append(self._grant_public(grant_id, grant, now))
            if issued:
                if transfer["first_grant"] is None:
                    transfer["first_grant"] = now
                self._emit("granted", transfer=self._public(ident, transfer), ranges=issued,
                           admission_wait_s=transfer["first_grant"] - transfer["began"])
            if len(covered) < transfer["count"]:
                self.grant_turns.append(ident)
                # Preserve this oldest demand when the global budget cannot fit
                # its next part; smaller new transfers cannot starve it forever.
                if (not issued and used < self.turn_bytes and
                        self.max_outstanding - self.outstanding_bytes < min(transfer["part_size"], transfer["total"])):
                    self.grant_turns.rotate(1)
                    break

    def _grant_public(self, ident, grant, now):
        return {"grant_id": ident, "part": grant["part"], "start": grant["start"], "end": grant["end"],
                "expires_in": max(0., grant["expires"] - now)}

    def grant(self, ident, incarnation, identity):
        now = self.clock()
        with self.lock:
            self._expire(now)
            transfer = self._get(ident, incarnation, identity)
            transfer["touched"] = now
            limited = self._limited(now, identity[3], self.polls, self.node_polls)
            if not limited and not transfer["complete"]:
                transfer["last_poll"] = now
                if ident not in self.grant_turns:
                    self.grant_turns.append(ident)
                self._schedule(now)
            grants = [self._grant_public(g, self.grants[g], now) for g in transfer["grants"]
                      if g in self.grants and self.grants[g]["state"] == "issued"] if not limited else []
            accepted = [self.grants[g]["part"] for g in transfer["grants"]
                        if g in self.grants and self.grants[g]["state"] == "accepted"]
            return {"grants": sorted(grants, key=lambda g: g["part"]), "accepted_parts": sorted(accepted),
                    "outstanding_bytes": self.outstanding_bytes, "assembly_bytes": self.assembly_bytes,
                    "retry_after": .1 if not grants else 0., "complete": transfer["complete"]}

    def begin_part(self, ident, incarnation, identity, grant_id, part, size):
        with self.lock:
            self._expire(self.clock())
            transfer = self._get(ident, incarnation, identity)
            grant = self.grants.get(grant_id)
            if grant is None or grant["transfer"] != ident or grant["part"] != part:
                raise AdmissionError("Receiver grant expired or has no matching part.")
            if size != grant["size"]:
                raise AdmissionError("Part size does not match its exact granted range.")
            if grant["state"] == "accepted":
                return "duplicate"
            if grant["state"] != "issued":
                raise AdmissionError("Receiver grant is already in use.")
            grant["state"], grant["active"], transfer["touched"] = "receiving", True, self.clock()
            return "receive"

    def finish_part(self, ident, incarnation, identity, grant_id, part, data, part_digest):
        with self.lock:
            transfer = self._get(ident, incarnation, identity)
            grant = self.grants.get(grant_id)
            if (grant is None or grant["transfer"] != ident or grant["part"] != part or
                    grant["state"] != "receiving" or len(data) != grant["size"]):
                raise AdmissionError("Part does not match a receiving grant.")
            if hashlib.sha256(data).hexdigest() != part_digest:
                raise AdmissionError("Part hash does not match; retry the same grant.")
            grant["state"], grant["digest"] = "accepted", part_digest
            transfer["parts"][part] = data
            transfer["touched"] = self.clock()
            if len(transfer["parts"]) != transfer["count"]:
                return None
            body = b"".join(transfer["parts"][index] for index in range(transfer["count"]))
            if len(body) != transfer["total"] or hashlib.sha256(body).hexdigest() != transfer["digest"]:
                self.cancel_transfer(ident, incarnation, reason="hash_mismatch")
                raise AdmissionError("Assembled transfer does not match its announced size and hash.")
            transfer["assembled"], transfer["body"] = True, body
            transfer["parts"].clear()
            # Assembly does not claim application: the request may disappear
            # before scheduling a worker. Status polling can recover this body.
            return body, max(1e-6, self.clock() - transfer["began"])

    def ready_to_apply(self, ident, incarnation, identity):
        """Inspect readiness without taking application ownership or a worker slot."""
        with self.lock:
            transfer = self._get(ident, incarnation, identity)
            return transfer["assembled"] and not transfer["complete"] and not transfer["applying"]

    def pending(self, ident, incarnation, identity):
        """Claim the one assembled body for application; concurrent retries only poll."""
        with self.lock:
            transfer = self._get(ident, incarnation, identity)
            if not transfer["assembled"] or transfer["complete"] or transfer["applying"]:
                return None
            transfer["applying"] = True
            return transfer["body"], max(1e-6, self.clock() - transfer["began"])

    def end_apply(self, ident, failed=False):
        with self.lock:
            transfer = self.transfers.get(ident)
            if transfer is not None:
                transfer["applying"] = False
                if failed:
                    self.cancel_transfer(ident, self.incarnation, reason="apply_failed")
                if transfer["complete"] or transfer["cancelled"]:
                    self._release_memory(transfer)

    def completed(self, ident, incarnation, identity):
        with self.lock:
            return self._get(ident, incarnation, identity)["complete"]

    def mark_complete(self, ident, incarnation, identity):
        with self.lock:
            transfer = self._get(ident, incarnation, identity)
            if transfer["complete"]:
                return
            if not transfer["assembled"]:
                raise AdmissionError("An incomplete body cannot be marked complete.")
            transfer["complete"], transfer["touched"] = True, self.clock()
            transfer["parts"].clear()
            transfer["body"] = None
            self._drop_turn(ident)
            self._release_memory(transfer)
            self._emit("accepted", transfer=self._public(ident, transfer),
                       duration_s=max(1e-6, transfer["touched"] - transfer["began"]))

    def end_part(self, grant_id):
        """Called only by the request that acquired begin_part, once its buffers drain."""
        with self.lock:
            grant = self.grants.get(grant_id)
            if grant is None or not grant["active"]:
                return
            grant["active"] = False
            transfer = self.transfers[grant["transfer"]]
            if grant["state"] == "accepted" or transfer["cancelled"]:
                self.outstanding_bytes -= grant["size"]
                if transfer["cancelled"]:
                    grant["state"] = "cancelled"
            else:
                grant["state"] = "issued"  # same exact range may be retried after a disconnect
                self._emit("part_retry", transfer_id=grant["transfer"], part=grant["part"])
            if transfer["complete"] or transfer["cancelled"]:
                self._release_memory(transfer)

    cancel_part = end_part

    def cancel_transfer(self, ident, incarnation, reason="cancelled"):
        with self.lock:
            if incarnation != self.incarnation:
                return
            transfer = self.transfers.get(ident)
            if transfer is None or transfer["cancelled"] or transfer["complete"] or transfer["applying"]:
                # A claimed application owns its outcome through the receipt.
                # Cancellation remains best effort once submission can begin.
                return
            transfer["cancelled"] = True
            for grant_id in transfer["grants"]:
                grant = self.grants[grant_id]
                if grant["state"] == "issued":
                    self.outstanding_bytes -= grant["size"]
                    grant["state"] = "cancelled"
            self.by_identity.pop(transfer["manifest_key"], None)
            self._drop_turn(ident)
            self._release_memory(transfer)
            self._emit("cancelled", transfer=self._public(ident, transfer), reason=reason)

    def fence_member(self, run, node, generation, epoch, term):
        with self.lock:
            for ident, transfer in list(self.transfers.items()):
                old = transfer["identity"]
                if old[0] == run and old[3] == node and (old[1] != term or old[4] != generation or old[5] != epoch):
                    self.cancel_transfer(ident, self.incarnation, reason="fenced")

    def _get(self, ident, incarnation, identity):
        if incarnation != self.incarnation:
            raise AdmissionError("Receiver restarted; announce a new transfer.")
        transfer = self.transfers.get(ident)
        if transfer is None or transfer["identity"] != tuple(identity) or transfer["cancelled"]:
            raise AdmissionError("Unknown or fenced transfer identity.")
        return transfer

    def _drop_turn(self, ident):
        self.grant_turns = deque(queued for queued in self.grant_turns if queued != ident)

    def capabilities(self):
        part = min(self.grant_bytes, self.max_outstanding, self.turn_bytes)
        return {"protocol": 1, "max_transfer": min(MAX_TRANSFER, max(0, (self.max_assembly - 3 * part) // 3)),
                "max_part_size": part, "max_turn_bytes": self.turn_bytes}

    def _public(self, ident, transfer):
        identity = transfer["identity"]
        return {"id": ident, "run": identity[0], "term": identity[1], "round": identity[2], "node": identity[3],
                "generation": identity[4], "epoch": identity[5], "bytes": transfer["total"]}

    def _emit(self, kind, **event):
        if self.on_event is not None:
            try:
                self.on_event(kind, {"at": self.clock(), "outstanding_bytes": self.outstanding_bytes,
                                     "assembly_bytes": self.assembly_bytes, **event})
            except Exception:  # telemetry must not change admission or contribution semantics
                pass
