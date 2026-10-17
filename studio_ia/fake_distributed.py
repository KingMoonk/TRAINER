"""A made-up coordinator for `studio_api --fake`: the Training tab has a run to show and stop without a model or a second machine.
It answers what studio_diloco.Coordinator answers to studio_distributed (snapshot, drop, stop), and the invitations to the other
machines are the real ones (studio_distributed.invite), so an unreachable or refusing machine reads the way it does for real."""
import threading
import time

from studio_diloco import Stale

BYTES = 4_000_000  # made-up size of one member's change and of the weights


class FakeCoordinator:
    def __init__(self, run, settings):
        self.run, self.settings = run, dict(settings)
        self.rounds = int(self.settings.get("rounds", 10))
        self.finished = self.stopped = False
        self.touched = time.monotonic()
        self.history, self.members, self.gone = [], [], {}
        self.lock = threading.Lock()

    def start(self):
        """The machines that took their invitation (and this one) are the members; the others are not."""
        self.members = [m.node for m in self.run.machines if m.local or m.state == "invited"]

    @property
    def round(self):
        return len(self.history)

    def advance(self):
        """Close one round: every live member contributed."""
        with self.lock:
            if self.finished:
                return
            n, live = self.round, [m for m in self.members if m not in self.gone]
            steps = int(self.settings.get("inner_steps", 100))
            self.history.append({
                "round": n, "step": (n + 1) * steps, "generation": 1, "members": live, "contributed": live, "dropped": [],
                "tokens": 4096 * len(live), "steps": {m: steps for m in live},
                "loss": round(5.2 - 3. * (n + 1) / self.rounds, 4), "val_loss": round(5.3 - 2.8 * (n + 1) / self.rounds, 4), "seconds": 2.,
                "delta_bytes": BYTES * len(live), "theta_bytes": BYTES * len(live), "delta_dtype": self.settings.get("delta_dtype", "float16"),
                "node_tokens": {m: 4096 for m in live}, "node_seconds": {m: 2. + i / 2 for i, m in enumerate(live)},
                "node_bytes": {m: BYTES for m in live}})
            self.finished = self.round >= self.rounds

    def snapshot(self):
        with self.lock:
            nodes = {m: {"state": "training", "missed": 0, "sent": False, "note": ""} for m in self.members if m not in self.gone}
            nodes.update({m: {"state": state, "missed": 0, "sent": False, "note": note} for m, (state, note) in self.gone.items()})
            state = "stopped" if self.stopped else "finished" if self.finished else "running"
            return {"state": state, "round": self.round, "rounds": self.rounds, "generation": 1, "settings": self.settings,
                    "nodes": nodes, "history": [dict(row) for row in self.history]}

    def drop(self, node):
        with self.lock:
            if node not in self.members or node in self.gone:
                raise Stale(f"{node} is not a member of this run.")
            self.gone[node] = ("stopped", "Stopped by you.")

    def stop(self):
        with self.lock:
            self.stopped = self.finished = True

    def release(self):
        pass
