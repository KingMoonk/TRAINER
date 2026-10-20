"""A made-up coordinator for `studio_api --fake`: the Training tab has a run to watch and stop without a model or a second machine.
It answers what studio_diloco.Coordinator answers to studio_distributed (snapshot, drop, stop), and the invitations to the other
machines are the real ones (studio_distributed.invite), so an unreachable or refusing machine reads the way it does for real.

A pause freezes the whole schedule at its beat (a round in flight does not finish, unlike a real run), and a resume carries on from there.
The run is simulated on a clock, not on a thread: nothing moves until it is asked, and then it is brought up to the clock. Every
member goes through a round the way a real one does: download theta, train H inner steps (its loss and speed reported as it goes),
upload its change, wait for the others; then the outer step and the check on the held-back text, and the next round. A CUDA machine
is faster than any other. `unit` is the length in seconds of one beat of that schedule (a round is about six of them)."""
import math
import threading
import time
from collections import deque

import numpy as np

import studio_run_health as run_health
from studio_diloco import EVENTS_KEPT, SAMPLES_KEPT, TIMELINES_KEPT, Stale

BYTES = 4_000_000  # made-up size of the weights
DELTA_SCALE = {"float32": 1., "float16": .5, "int8": .25}  # a change's size against the weights, by dtype
OUTER, CHECK = .4, .6  # beats the outer step and the check on the held-back text take
REPORT = .5  # beats between a member's progress reports
FAST = (.5, 2.2, .6, 9800.)  # beats to download, to train and to upload, and tokens per second: a CUDA machine
SLOW = (.8, 3.6, .6, 2400.)  # any other
SLACK = 1.25  # the round's deadline against its slowest member
TOKENS = 4096  # target tokens in a member's round


def loss_at(round_number, step, steps, index=0):
    """A made-up inner loss: it falls from round to round, and a little within one."""
    base = 2.4 + 2.8 * math.exp(-round_number / 12.)
    return round(base * (1. - .03 * step / max(1, steps)) + .02 * math.sin(step * 1.7 + index), 4)


class FakeCoordinator:
    def __init__(self, run, settings, clock=time.monotonic, unit=1.):
        self.run, self.settings, self.clock, self.unit = run, dict(settings), clock, max(unit, 1e-6)
        self.rounds = int(self.settings.get("rounds", 10))
        self.steps = int(self.settings.get("inner_steps", 100))
        self.dtype = self.settings.get("delta_dtype", "float16")
        self.finished = self.stopped = self.paused = False
        self.paused_at = None  # the clock reading when it was paused
        self.touched = time.monotonic()
        self.history, self.members, self.gone = [], [], {}
        self.events, self.timelines, self.seq = deque(maxlen=EVENTS_KEPT), deque(maxlen=TIMELINES_KEPT), 0
        self.begun, self.frozen = None, None  # the clock at start; the beat the run stopped or finished at
        self.opened, self.roster, self.drops, self.logged = 0., [], {}, set()
        self.lock = threading.Lock()

    def start(self):
        """The machines that took their invitation (and this one) are the members; the others are not."""
        with self.lock:
            self.members = [m.node for m in self.run.machines if m.local or m.state == "invited"]
            self.roster = list(self.members)
            self.begun = self.clock()
            for node in self.members:
                self._event(0., "joined", node, 0, f"{node} joined.")

    @property
    def round(self):
        return len(self.history)

    # -- the schedule
    def _beat(self):
        if self.frozen is not None:
            return self.frozen
        if self.paused_at is not None and self.begun is not None:
            return (self.paused_at - self.begun) / self.unit
        return 0. if self.begun is None else (self.clock() - self.begun) / self.unit

    def _profile(self, node):
        machine = next((m for m in self.run.machines if m.node == node), None)
        fast = machine is not None and machine.device == "cuda"
        download, train, upload, rate = FAST if fast else SLOW
        index = self.members.index(node) if node in self.members else 0
        return {"download": download, "train": train + .15 * index, "upload": upload, "tps": rate, "index": index}

    def _plan(self):
        """{node: (beat it started, started training, started uploading, delivered)} for the open round, from when it opened."""
        out = {}
        for node in self.roster:
            p = self._profile(node)
            out[node] = (0., p["download"], p["download"] + p["train"], p["download"] + p["train"] + p["upload"])
        return out

    def _counted(self, node, plan):
        """Whether the node's change reaches the round: it was not stopped before it had delivered."""
        stopped = self.drops.get(node)
        return stopped is None or stopped - self.opened >= plan[node][3]

    def _closes(self, plan):
        """Beats after the round opened at which its last awaited change is in (a stopped member is awaited until it is stopped);
        None when nobody's change will arrive."""
        counted = [plan[node][3] for node in plan if self._counted(node, plan)]
        if not counted:
            return None
        return max([*counted, *(t - self.opened for t in self.drops.values())])

    def _event(self, beat, kind, node, number, text):
        self.seq += 1
        self.events.append({"seq": self.seq, "t": beat * self.unit, "kind": kind, "node": node, "round": number, "text": text})

    def _catch_up(self, now):
        """Close every round that is over by `now`, opening the next."""
        while not self.finished and self.roster:
            plan = self._plan()
            closes = self._closes(plan)
            if closes is None:
                return
            if now < self.opened + closes + OUTER + CHECK:
                return
            self._close(plan, closes)
            if self.finished:
                self.frozen = self.opened
                return
            self.roster = [m for m in self.members if m not in self.gone]

    def _log_outer(self, closes, upto):
        number = self.round
        if ("outer", number) not in self.logged and upto >= self.opened + closes + OUTER:
            self.logged.add(("outer", number))
            counted = [m for m in self.roster if self._counted(m, self._plan())]
            self._event(self.opened + closes + OUTER, "outer_step", "", number,
                        f"Outer step applied for round {number}: size {round(.6 / (number + 3), 4)}, from {len(counted)} of {len(self.roster)} members.")

    def _close(self, plan, closes):
        number = self.round
        self._log_outer(closes, math.inf)
        counted = [m for m in self.roster if self._counted(m, plan)]
        dropped = [m for m in self.roster if m not in counted]
        profiles = {m: self._profile(m) for m in self.roster}
        node_tokens = {m: round(profiles[m]["tps"] * profiles[m]["train"] * self.unit) for m in counted}
        loss = loss_at(number, self.steps, self.steps)
        size = BYTES * DELTA_SCALE.get(self.dtype, .5)
        self.timelines.append(self._timeline(number, plan, closes, True, math.inf))
        done = closes + OUTER + CHECK
        row = {
            "round": number, "step": (number + 1) * self.steps, "generation": 1, "members": list(self.roster), "contributed": counted,
            "dropped": dropped, "tokens": sum(node_tokens.values()), "steps": {m: self.steps for m in counted},
            "loss": loss, "val_loss": round(loss + .08, 4), "seconds": done * self.unit,
            "delta_bytes": int(size) * len(counted), "theta_bytes": BYTES * len(self.roster), "delta_dtype": self.dtype,
            "node_tokens": node_tokens, "node_seconds": {m: profiles[m]["train"] * self.unit for m in counted},
            "node_bytes": {m: int(size) for m in counted}, "outer_step_norm": round(.6 / (number + 3), 4),
            "tokens_per_second": sum(node_tokens.values()) / max(1e-9, done * self.unit), "lr": self.settings.get("outer_lr", 0.)}
        arrived = {m: plan[m][3] * self.unit for m in counted}
        row["health"] = run_health.record(row, {m: loss_at(number, self.steps, self.steps, profiles[m]["index"]) for m in counted}, arrived, BYTES)
        self.history.append(row)
        self.opened += done
        self._keep_weights()
        self.logged.discard(("outer", number))
        self.drops = {}
        self._event(self.opened, "checkpoint", "", number, f"Round {number} checkpoint written.")
        self.finished = self.round >= self.rounds
        if self.finished:
            self._event(self.opened, "finished", "", number, f"The run finished after {self.round} rounds.")

    def _keep_weights(self):
        """last.npz and best.npz as stubs in the run's folder (when it has one): a restart looks for them."""
        folder = getattr(self.run, "folder", "")
        if folder:
            for name in ("last.npz", "best.npz"):
                with open(f"{folder}/{name}", "wb") as stream:
                    np.savez(stream, **{"weight::stub": np.zeros(1, dtype=np.float32)})

    def _timeline(self, number, plan, closes, closed, offset):
        """The swimlanes of the open round as of `offset` beats after it opened (every moment later than that is not known yet)."""
        lanes = {}
        for node, (taken, first, sent, received) in plan.items():
            stopped = self.drops.get(node)
            known = lambda beat: None if beat > offset or (stopped is not None and stopped - self.opened < beat) else beat * self.unit  # noqa: E731
            counted = self._counted(node, plan)
            profile = self._profile(node)
            delivered = counted and received <= offset
            lanes[node] = {"theta_taken": known(taken), "first_step": known(first), "delta_sent": known(sent), "delta_received": known(received),
                           "counted": (True if counted else False) if closed else (False if not counted else True if received <= offset else None),
                           "tokens": round(profile["tps"] * profile["train"] * self.unit) if delivered else 0, "steps": self.steps if delivered else 0,
                           "delta_bytes": int(BYTES * DELTA_SCALE.get(self.dtype, .5)) if delivered else 0}
        window = self._window(plan)
        return {"round": number, "open": not closed, "deadline": window * self.unit, "window": window * self.unit,
                "closed": closes * self.unit if closed or (closes is not None and offset >= closes) else None, "lanes": lanes}

    def _window(self, plan):
        return SLACK * max((p[3] for p in plan.values()), default=1.)

    def _member_state(self, node, plan, offset):
        """(phase, step, tps, loss, reported_ago, samples) of a live member `offset` beats into the round."""
        taken, first, sent, received = plan[node]
        profile = self._profile(node)
        number = self.round
        if offset < first:
            phase, step = "downloading", 0
        elif offset < sent:
            phase, step = "training", int(self.steps * (offset - first) / (sent - first))
        elif offset < received:
            phase, step = "uploading", self.steps
        else:
            phase, step = "waiting", self.steps
        every = max(1, self.steps // 12)
        samples = [{"step": s, "t": (first + (sent - first) * s / self.steps) * self.unit, "loss": loss_at(number, s, self.steps, profile["index"]),
                    "tokens_per_second": profile["tps"] + 40. * math.sin(s)} for s in range(0, step + 1, every)] if phase != "downloading" else []
        del samples[:-SAMPLES_KEPT]
        loss = loss_at(number, step, self.steps, profile["index"]) if phase != "downloading" else None
        tps = profile["tps"] + 40. * math.sin(step) if phase != "downloading" else None
        return phase, step, tps, loss, (offset % REPORT) * self.unit, samples

    def advance(self):
        """Bring the run up to the clock."""
        with self.lock:
            self._catch_up(self._beat())

    # -- what the owner's tab reads
    def snapshot(self):
        with self.lock:
            self.touched = time.monotonic()
            self._catch_up(self._beat())
            now = self._beat()  # a finished run's clock stopped when it finished
            plan, offset = self._plan(), now - self.opened
            closes = self._closes(plan)
            if closes is not None and not self.finished:
                self._log_outer(closes, now)
            live = [m for m in self.roster if m not in self.gone]
            states = {m: self._member_state(m, plan, offset) for m in plan} if not self.finished else {}
            nodes = {}
            for node in self.members:
                if node in self.gone:
                    continue
                phase, step, tps, loss, ago, samples = states.get(node, ("waiting", self.steps, None, None, None, []))
                sent = node in plan and self._counted(node, plan) and (self.finished or offset >= plan[node][3])
                nodes[node] = {"state": "training", "missed": 0, "sent": sent, "note": "",
                               "progress": {"phase": phase, "step": step, "steps": self.steps, "loss": loss, "tokens_per_second": tps,
                                            "reported_ago": ago, "samples": samples}, "shards": 4}
            for node, (state, note) in self.gone.items():
                nodes[node] = {"state": state, "missed": 0, "sent": False, "note": note,
                               "progress": {"phase": "waiting", "step": 0, "steps": self.steps, "loss": None, "tokens_per_second": None,
                                            "reported_ago": None, "samples": []}, "shards": 0}
            state = "stopped" if self.stopped else "finished" if self.finished else "running" if live else "waiting"
            phase = self._phase(plan, offset, closes, states, live)
            shown = [] if self.finished else [self._timeline(self.round, plan, closes, False, offset)] if live and closes is not None else []
            window = self._window(plan) * self.unit if live else None
            return {"state": state, "phase": phase, "round": self.round, "rounds": self.rounds, "generation": 1, "settings": self.settings,
                    "nodes": nodes, "history": [dict(row) for row in self.history],
                    "deadline_s": max(0., window - offset * self.unit) if window is not None else None,
                    "grace_s": 2. * self.unit, "window_s": window,
                    "contributed": sum(1 for n in nodes.values() if n["sent"]) if not self.finished else 0,
                    "expected": len(live) if not self.finished else 0, "theta_bytes": BYTES, "now_s": now * self.unit,
                    "timeline": [*self.timelines, *shown], "events": [dict(event) for event in self.events], "paused": self.paused,
                    "rules": {"best_round": self.history[-1]["round"] if self.history else None, "best_loss": self.history[-1]["val_loss"] if self.history else None,
                              "stop_rule": "", "stop_text": ""}}

    def _phase(self, plan, offset, closes, states, live):
        if self.stopped:
            return "stopped"
        if self.finished:
            return "finished"
        if self.paused:
            return "paused"
        if not live or closes is None:
            return "waiting"
        if offset >= closes + OUTER:
            return "validating"
        if offset >= closes:
            return "outer_step"
        phases = [s[0] for m, s in states.items() if m in live]
        if any(p == "uploading" for p in phases) or any(offset >= plan[m][3] for m in live):
            return "collecting"
        return "training" if "training" in phases else "next_round"

    def refusal(self, node):
        """As studio_diloco.Coordinator.refusal. A demo run's members are the ones it started with, so a machine added later waits."""
        with self.lock:
            if self.finished or self.stopped:
                return "The run is finished."
            if self.gone.get(node, ("",))[0] == "stopped":
                return f"{node} was stopped by the run's owner."
            if node in self.members and node not in self.gone:
                return f"{node} is already in this run."
            most = int(self.settings.get("max_members", 8))
            if len([m for m in self.members if m not in self.gone]) >= most:
                return f"The run already has its {most} members."
            return ""

    def drop(self, node):
        with self.lock:
            self._catch_up(self._beat())
            if node not in self.members or node in self.gone:
                raise Stale(f"{node} is not a member of this run.")
            now = self._beat()
            self.gone[node] = ("stopped", "Stopped by you.")
            self.drops[node] = now
            self._event(now, "left", node, self.round, f"{node} was stopped by you.")

    def pause(self):
        with self.lock:
            if self.finished:
                raise Stale("The run is over; there is nothing to pause.")
            if self.paused:
                return
            self._catch_up(self._beat())
            self.paused, self.paused_at = True, self.clock()
            self._event(self._beat(), "paused", "", self.round, "The run was paused by you. No new round opens until you resume it.")

    def resume(self):
        with self.lock:
            if self.finished:
                raise Stale("The run is over; there is nothing to resume.")
            if not self.paused:
                return
            if self.begun is not None:
                self.begun += self.clock() - self.paused_at  # the schedule carries on from the beat it stopped at
            self.paused, self.paused_at = False, None
            self._event(self._beat(), "resumed", "", self.round, "The run was resumed by you.")

    def stop(self):
        with self.lock:
            if self.finished:
                return
            self._catch_up(self._beat())
            self.frozen = self._beat()
            self.stopped = self.finished = True
            self._event(self.frozen, "finished", "", self.round, "The run was stopped by you.")

    def release(self):
        pass
