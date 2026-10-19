"""Coordinator failover (docs/distributed.md, "Terms and coordinator failover", #361). Pure rules first, so each is a window-free test with
fake clocks; then `Failover`, the driver every node runs at each poll against a `Host`; then `ClusterHost`, the Host over a
studio_remote.Cluster (its observation store, its coordinators, its files).

* A **claim** is `(run, node, term, round, rank)`: the node that coordinates the run at that term, continuing from that round, and the
  rank it held as a replica. Claims order by `(term, round, lower rank)`, then the lower node name, so every node orders the same pair
  the same way (`beats`).
* The **coordinator is lost** when the failure detector has its node down and the freshest thing it wrote about the run is older than
  the members' lease (`coordinator_lost`): the wait the scheduler takes before it reassigns a job.
* A **handoff** is a coordinator moving on purpose: the person chose another node to coordinate at the start (`Seen.hands_to`). That
  node, once it keeps a whole replica no older than the coordinator's last round, claims the next term at once instead of waiting
  for the coordinator to be lost; the coordinator then steps down like any that sees a claim that beats its own.
* The **successor** is, among the live replicas that hold the whole run (`ready`), the one with the highest completed round, ties to the
  lowest rank (`successor`). Every node evaluates the rule on its own replica of the store; only the successor acts.
* A node that coordinates a run and sees a claim that beats its own **steps down**: its rounds are discarded (Coordinator.defer), so
  the run has one theta. A coordinator that returns finds the claim of the higher term in the store and does the same on its first
  poll; a member that calls it with the higher term makes it step down sooner.
* **The seam for the bucket (#362).** `Host.tie_break(claim)` is asked before a claim is written: a host that has the bucket writes
  `runs/<run>/term/<term>` with a conditional PUT there and answers False when another claimant won. The default answers True, so
  the rule above decides alone, which is also what happens when the bucket cannot be reached (#362: `ClusterHost.tie_break`).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

_log = logging.getLogger(__name__)

ENDED = ("finished", "stopped", "failed")  # states of a run nobody takes over
CLAIM = "claim"  # store kind, key run id: {"term", "round", "rank", "released"}, written by the node that coordinates the run
HELD = "held"  # store kind, key run id: {"term", "round", "rank", "ready", "held"}, written by a node that keeps a replica of it
WHERE_PREFIX = "Coordinated on "
SAVE_TRIES = 3  # polls that ask the successor for a finished run's weights when the owner's own job ended without saving them


@dataclass(frozen=True)
class Claim:
    run: str
    node: str
    term: int
    round: int
    rank: int = 0


@dataclass(frozen=True)
class Held:
    """A replica of the run on `node`: the round it holds, its rank among the replicas, and whether it is whole (`ready`: it also has the
    validation windows and every shard, so it could coordinate)."""
    node: str
    round: int
    rank: int
    ready: bool = True


@dataclass(frozen=True)
class Seen:
    """What the store says about who coordinates a run: the claim, how long ago its node last wrote anything about the run, its state."""
    claim: Claim
    silent_s: float
    state: str = "running"
    hands_to: str = ""  # the node the person chose to coordinate the run (the run row's `handoff`): this coordinator hands the run to it


# ---- the rules ----------------------------------------------------------------------------------------------------------

def order(claim):
    """The key claims are ordered by: higher term, then higher round, then lower rank. Greater wins."""
    return claim.term, claim.round, -claim.rank


def beats(a, b):
    """Whether claim `a` outranks claim `b`. Total over distinct claims: the same (term, round, rank) from two nodes goes to the lower name."""
    if order(a) != order(b):
        return order(a) > order(b)
    return a.node < b.node


def standing(seen):
    """The Seen whose claim outranks the others', or None. Several nodes writing about a run at once is what a failover or a heal looks like."""
    best = None
    for found in seen:
        if best is None or beats(found.claim, best.claim):
            best = found
    return best


def coordinator_lost(down, silent_s, lease_s):
    """The coordinator counts as gone when the failure detector has it down (`down`) and the members' leases have lapsed: nothing it wrote
    about the run is younger than `lease_s`."""
    return bool(down) and silent_s >= lease_s


def successor(replicas, live):
    """The replica that takes the run over: among those that are ready and whose node `live(node)` accepts, the highest round, ties to the
    lowest rank (then the lower name). None when there is none."""
    candidates = [r for r in replicas if r.ready and live(r.node)]
    return min(candidates, key=lambda r: (-r.round, r.rank, r.node), default=None)


def noted(note, text):
    """`note` with the sentence a failover wrote (everything from WHERE_PREFIX on) replaced by `text` ('' removes it)."""
    at = note.find(WHERE_PREFIX)
    base = (note if at < 0 else note[:at]).strip()
    return f"{base} {text}".strip() if text else base


# ---- the driver ---------------------------------------------------------------------------------------------------------

class Host:
    """What `Failover` needs from a node. ClusterHost is the real one; tests may fake any of it."""
    name = ""
    lease_s = 30.

    def runs(self):
        """{run id: [Seen]}: every node's account of who coordinates each run (its run entry or claim), this node's own coordinator included."""
        raise NotImplementedError

    def replicas(self, run):
        """[Held] of every node that keeps a replica of `run` (this node's included)."""
        raise NotImplementedError

    def is_down(self, node):
        raise NotImplementedError

    def local(self, run):
        """The Claim of the coordinator this node runs for `run`, or None when it runs none."""
        raise NotImplementedError

    def superseded(self, run):
        """Whether this node's coordinator of `run` was told of a higher term by a call."""
        return False

    def tie_break(self, claim):
        """The bucket's conditional PUT (ClusterHost): False when another claimant of this term already won there."""
        return True

    def claim(self, claim):
        raise NotImplementedError

    def withdraw(self, run):
        raise NotImplementedError

    def take_over(self, claim):
        """Build the coordinator from the replica and serve it. True when it runs; False (or an exception) when it could not."""
        raise NotImplementedError

    def step_down(self, run, claim):
        """Stop coordinating `run`: `claim` outranks this node's (None: a call carried a higher term)."""
        raise NotImplementedError

    def point(self, run, claim, node_down):
        """Tell the members on this node where `run` is coordinated now."""

    def note(self, run, claim):
        """Say on the owner's queue record where `run` is coordinated now."""

    def finished_under_successor(self, run):
        """`run` finished under a coordinator of term 2 or more (asked at every poll while the store says so)."""


class Failover:
    """One decision per run per poll. No threads and no clock of its own: the Host brings both."""

    def __init__(self, host):
        self.host = host

    def step(self):
        host = self.host
        for run, seen in host.runs().items():
            try:
                self._run(run, seen)
            except Exception:  # noqa: BLE001 - one run's trouble must not stop the others, nor the poll
                _log.warning("Failover step for run %s failed", run, exc_info=True)

    def _run(self, run, seen):
        host = self.host
        best = standing(seen)
        if best is None:
            return
        mine = host.local(run)
        if mine is not None:
            if host.superseded(run):
                host.step_down(run, None)
            elif best.claim.node != host.name and beats(best.claim, mine):
                host.step_down(run, best.claim)
            return
        host.point(run, best.claim, host.is_down(best.claim.node))
        host.note(run, best.claim)
        if best.state in ENDED:
            if best.state == "finished" and best.claim.term >= 2:
                host.finished_under_successor(run)
            return
        if best.hands_to == host.name and best.claim.node != host.name:
            chosen = self._handed(run, best)  # the person chose this node: it takes the run over as soon as it holds the last round
        elif not coordinator_lost(host.is_down(best.claim.node), best.silent_s, host.lease_s):
            return
        else:
            chosen = successor(host.replicas(run), lambda node: node != best.claim.node and (node == host.name or not host.is_down(node)))
        if chosen is None or chosen.node != host.name:
            return
        claim = Claim(run, host.name, best.claim.term + 1, chosen.round, chosen.rank)
        if not host.tie_break(claim):
            return
        host.claim(claim)
        try:
            taken = host.take_over(claim)
        except Exception:  # noqa: BLE001 - the replica may not be whole after all; say so and let the next poll's successor try
            _log.warning("Taking over run %s failed", run, exc_info=True)
            taken = False
        if not taken:
            host.withdraw(run)
        else:
            host.point(run, claim, False)
            host.note(run, claim)


    def _handed(self, run, best):
        """This node's replica of `run` when it is whole and holds every round the coordinator closed (and at least one: before that
        there is nothing to continue from), else None: the coordinator keeps the run until then."""
        mine = next((r for r in self.host.replicas(run) if r.node == self.host.name), None)
        return mine if mine is not None and mine.ready and mine.round >= max(1, best.claim.round) else None


# ---- the Host over a Cluster ---------------------------------------------------------------------------------------------

class ClusterHost(Host):
    """A studio_remote.Cluster as the Failover's Host: the store is its replica, the coordinators are its registry (the process's
    studio_diloco.COORDINATORS unless the Cluster was given its own), the files are under the runner's workspace."""

    def __init__(self, cluster):
        self.cluster = cluster
        self.claims = {}  # run id -> the Claim this node made (its coordinator's place in the order)
        self.name = cluster.hostname
        self.lease_s = cluster.lease_s
        self.pointed = {}  # run id -> (node, term) written to the where-file
        self.saves = {}  # run id -> how many times the successor's weights were asked for after the owner's job had ended

    # -- reading the store
    def runs(self):
        cluster, out = self.cluster, {}
        for origin in sorted({e["origin"] for e in cluster.state.entries(kind="heartbeat")}):
            index = cluster.state.get(origin, "runs", "index")
            ids = set(index["value"].get("live", index["value"].get("ids", [])) if index else [])  # a kept account is not a coordinator
            claims = {e["key"]: e for e in cluster.state.entries(origin=origin, kind=CLAIM)}
            for run in sorted(ids | set(claims)):
                found = self._seen(origin, run, run in ids, claims.get(run))
                if found is not None:
                    out.setdefault(run, []).append(found)
        return out

    def _seen(self, origin, run, indexed, claim):
        row = self.cluster.state.get(origin, "run", run) if indexed else None
        value = row["value"] if row else None
        ages = [row["age"]] if row else []
        if claim is not None and not claim["value"].get("released"):
            held = claim["value"]
            if value is None or int(value.get("term", 0)) == int(held["term"]):
                ages.append(claim["age"])
            return Seen(Claim(run, origin, int(held["term"]), int(held["round"]), int(held.get("rank", 0))),
                        0. if origin == self.name else min(ages), (value or {}).get("state", "running"), str((value or {}).get("handoff") or ""))
        if value is None:
            return None
        if claim is not None and value.get("state", "running") not in ENDED:
            return None  # it released its claim (stepped down or withdrew): its run row, until the next publish, is not a coordinator to yield to
        return Seen(Claim(run, origin, int(value.get("term", 1)), int(value.get("round", 0)), 0), 0. if origin == self.name else row["age"],
                    value.get("state", "running"), str(value.get("handoff") or ""))

    def replicas(self, run):
        out = []
        for entry in self.cluster.state.entries(kind=HELD):
            value = entry["value"]
            if entry["key"] == run and value.get("held"):
                out.append(Held(entry["origin"], int(value["round"]), int(value.get("rank", 0)), bool(value.get("ready"))))
        return out

    def is_down(self, node):
        return self.cluster.is_down(node)

    # -- the bucket's tie-break
    def tie_break(self, claim):
        """Write `runs/<run>/term/<term>` in the version store, create-if-absent. The first writer wins; the others read the winner and
        follow (False). A store that is not set up or cannot be reached, or a winner that cannot be read, leaves the rule to decide (True):
        nothing is decided only here. A winner that is down is no reason to wait for it, so the rule decides then too."""
        try:
            store = self.cluster.store()
            if store is None or not hasattr(store, "create_if_absent"):
                return True
            key = f"runs/{claim.run}/term/{claim.term}"
            if store.create_if_absent(key, json.dumps({"node": claim.node, "round": claim.round, "rank": claim.rank}).encode()):
                return True
            try:
                winner = json.loads(store.read(key) or b"null")["node"]
            except (ValueError, KeyError, TypeError):
                return True
        except Exception:  # noqa: BLE001 - the bucket is a tie-breaker only; a failure of it never stops a run
            _log.warning("The store could not break the tie for run %s term %s; the rule decides", claim.run, claim.term, exc_info=True)
            return True
        if winner == self.name or self.is_down(winner):
            return True
        _log.info("Run %s term %s was claimed first by %s; following it", claim.run, claim.term, winner)
        return False

    # -- this node's coordinators
    def _coordinator(self, run):
        import studio_diloco
        found = self.cluster.coordinators.get(run)
        return found if isinstance(found, studio_diloco.Coordinator) else None  # the demo fleet's fake coordinator has no term to fail over

    def local(self, run):
        coordinator = self._coordinator(run)
        if coordinator is None:
            return None
        return self.claims.get(run) or Claim(run, self.name, coordinator.term, coordinator.next_round, 0)

    def superseded(self, run):
        coordinator = self._coordinator(run)
        return coordinator is not None and bool(coordinator.superseded)

    def claim(self, claim):
        self.claims[claim.run] = claim
        url = self.cluster.url_for_peers()  # where the others' members find this coordinator (url_of)
        self.cluster.claims[claim.run] = {"term": claim.term, "round": claim.round, "rank": claim.rank, **({"url": url} if url else {})}
        self.cluster.state.put(CLAIM, claim.run, self.cluster.claims[claim.run])

    def withdraw(self, run):
        self.claims.pop(run, None)
        self.cluster.claims[run] = {"released": True}
        self.cluster.state.put(CLAIM, run, {"released": True})

    def take_over(self, claim):
        import studio_diloco
        import studio_distributed
        cluster = self.cluster
        root = Path(cluster.runner.root)
        prior = standing([s for s in self.runs().get(claim.run, []) if s.claim.node != self.name])
        row = cluster.state.get(prior.claim.node, "run", claim.run) if prior else None
        coordinator = studio_diloco.Coordinator.from_replica(
            claim.run, root / "data" / "failover" / f"{claim.run}_term{claim.term}", root / "data" / "replicas" / claim.run,
            root / "data" / "remote_shards" / claim.run, claim.term)
        if row is not None:
            studio_distributed.adopt({"id": claim.run, **row["value"]})
        if cluster.coordinators is studio_diloco.COORDINATORS:
            studio_diloco.register(claim.run, coordinator)
        else:
            cluster.coordinators[claim.run] = coordinator
        cluster.runner._emit("cluster", {"job": claim.run, "state": "took_over", "node": self.name,
                                         "detail": f"Term {claim.term} began on {self.name}, continuing from round {coordinator.next_round}."})
        return True

    def step_down(self, run, claim):
        coordinator = self.cluster.coordinators.get(run)
        if coordinator is not None:  # the members here learn where the run went before the coordinator leaves the registry, so a call that lands in between finds the new one
            coordinator.defer(claim.term if claim is not None else coordinator.superseded or coordinator.term)
            if claim is not None:
                self.point(run, claim, self.is_down(claim.node))
        self.cluster.coordinators.pop(run, None)
        self.claims.pop(run, None)
        self.cluster.claims[run] = {"released": True}
        self.cluster.state.put(CLAIM, run, {"released": True})
        if coordinator is not None:
            import studio_distributed
            term = claim.term if claim is not None else coordinator.superseded or coordinator.term
            studio_distributed.archive(run, coordinator)  # the Training tab keeps what it showed of the term that ended
            self.cluster.runner._emit("cluster", {"job": run, "state": "stepped_down", "node": self.name,
                                                  "detail": f"Term {term} is coordinated on {claim.node if claim else 'another node'}; rounds closed here are discarded."})

    # -- telling the others on this node
    def url_of(self, node, run):
        """Where a member on this node reaches the coordinator of `run` on `node`: this node's API on loopback at the port it listens on,
        else the url `node` published with its claim, else (an old claim, or one naming a loopback address) the address this node lists
        `node` at."""
        if node == self.name:
            return f"http://127.0.0.1:{self.cluster.api_port}"
        import studio_distributed
        entry = self.cluster.state.get(node, CLAIM, run)
        url = str(((entry or {}).get("value") or {}).get("url") or "")
        return url if url and not studio_distributed.loopback(url) else self.cluster.address_of(node)

    def point(self, run, claim, node_down):
        """Write where the members here find the coordinator (studio_diloco.file_locator): only after a failover (term 2 or more), only
        for a coordinator that answers, and only when it changed."""
        if claim.term < 2 or node_down or self.pointed.get(run) == (claim.node, claim.term):
            return
        if run not in self.cluster.members and run not in self.cluster.replicas:
            return  # no member of the run here
        url = self.url_of(claim.node, run)
        if not url:
            return
        import studio_diloco
        from studio_core import atomic_json
        path = studio_diloco.where_file(self.cluster.runner.root, run)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(path, {"node": claim.node, "term": claim.term, "url": url})
        self.pointed[run] = (claim.node, claim.term)

    def finished_under_successor(self, run):
        """The owner's job ended without saving the run's weights (its member stopped or failed while the run went on, #453): replay the
        job's end, so the services fetch the successor's final weights (WorkspaceServices._record_distributed). A run the person stopped
        saves nothing; a run already saved is left alone; a fetch that failed is tried again at the next poll, up to SAVE_TRIES times."""
        import studio_distributed
        runner = self.cluster.runner
        with runner.lock:
            job = next((j for j in runner.jobs if j.id == run and j.kind == "pretrain_distributed" and not j.owner), None)
            status = job.status if job is not None else ""
            version = job.params.get("version") if job is not None else None
        kept = studio_distributed.RUNS.get(run)
        if status not in ("done", "failed", "stopped") or not version or (kept is not None and (kept.stopped or kept.saved is not None)):
            return
        if self.saves.get(run, 0) >= SAVE_TRIES:
            return
        self.saves[run] = self.saves.get(run, 0) + 1
        runner._emit("job", {"job": run, "status": status, "error": ""})

    def note(self, run, claim):
        """The owner's queue record says where its run is coordinated now ('' again once it is back here, or never moved)."""
        runner = self.cluster.runner
        with runner.lock:
            job = next((j for j in runner.jobs if j.id == run and j.kind == "pretrain_distributed" and not j.owner), None)
            if job is None:
                return
            text = f"{WHERE_PREFIX}{claim.node} (term {claim.term})." if claim.term > 1 and claim.node != self.name else ""
            now = noted(job.note or "", text)
            if now != (job.note or ""):
                job.note = now
                runner._save()
