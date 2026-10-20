"""The cluster self-test: does a DiLoCo run work across the machines the person picked? (docs/distributed.md, "Self-test")

It answers in about a minute, with a verdict per step. The checks before the run (the machine answers, runs this version, lets this
computer in, can reach this computer back) are made directly. Then a real, tiny `pretrain_distributed` run (3 rounds on a built-in text)
is queued and goes through the real launch, invite, member, sync and average path; every later step is read from what that run did:
the member job taken, the member joined, shards and weights delivered, each round's contributions, and the loss going down. The one
extra check is a checksum: while a round is open each machine downloads that round's weights from the coordinator, as a member does,
and says their checksum, which must equal the coordinator's own.

`SelfTest` is pure logic over a `Probe` (what it needs from the app), so every verdict is a window-free test with fakes. The app wires
a real `Probe` (WorkspaceServices.cluster_selftest_start); the peers answer `POST /cluster/selftest/ping` and `.../digest`
(studio_api/routers/cluster.py).
"""
from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import asdict, dataclass, field

ROUNDS = 3
INNER_STEPS = 20
JOIN_TIMEOUT_S = 120.  # a peer's worker has to start (it imports torch) before it joins
RUN_TIMEOUT_S = 240.
DIGEST_AFTER_ROUND = 1  # the checksum is taken on a round after the first outer step, so it covers an averaged theta
PENDING, RUNNING, PASS, FAIL, SKIPPED = "pending", "running", "pass", "fail", "skipped"
LABEL = "Cluster self-test"

TEXT = [{"text": f"Document {n}: the quick brown fox jumps over the lazy dog, and the dog sleeps. " * (n % 5 + 2)} for n in range(48)]
CONFIG = {"device": "cpu", "width": 16, "layers": 1, "heads": 2, "context": 32, "expansion": 2, "dropout": 0.1, "batch_size": 4,
          "learning_rate": 0.01, "warmup": 2, "validation": 0.2, "seed": 7, "sample_every": 0, "objective": "all", "name": LABEL}


@dataclass
class Step:
    id: str
    title: str
    machine: str = ""  # the machine it is about; "" for the whole run
    status: str = PENDING
    detail: str = ""  # what was measured, or what went wrong
    fix: str = ""  # on a failure: the plain thing to do about it
    seconds: float | None = None


@dataclass
class Result:
    id: str
    machines: list
    started_at: float
    finished_at: float | None = None
    state: str = "running"  # running | passed | failed
    steps: list = field(default_factory=list)

    def view(self):
        return asdict(self)


def job_params(machines, peer_url, api_port, owner):
    """The params of the self-test run's job: the built-in text and a tiny design, every machine required, nothing saved as a version."""
    from studio_diloco import Settings
    settings = Settings(rounds=ROUNDS, inner_steps=INNER_STEPS, outer_lr=.7, outer_momentum=.9, min_members=len(machines), max_members=len(machines),
                        shards=8, delta_dtype="float16")
    return {"config": {**CONFIG, "steps": ROUNDS * INNER_STEPS}, "examples": TEXT, "mode": "from_scratch", "selftest": True,
            **{k: getattr(settings, k) for k in ("rounds", "inner_steps", "outer_lr", "outer_momentum")}, "delta_dtype": settings.delta_dtype,
            "settings": asdict(settings), "min_members": settings.min_members, "max_members": settings.max_members,
            "machines": machines, "owner": owner, "peer_url": peer_url, "api_port": api_port}


def checksum(data):
    return hashlib.sha256(data).hexdigest()


def megabytes(count):
    return f"{count / 1e6:.2f} MB" if count >= 1e5 else f"{count / 1e3:.1f} KB"


class SelfTest:
    """One self-test. `probe` is the app's side (see Probe in the tests): survey(addresses), client(address), launch(), run(), snapshot(),
    rows(), job_error(), theta_digest(number), term(), stop(), plus `clock` and `sleep`. `peers` are the other machines' addresses."""

    def __init__(self, probe, peers, peer_url, hostname, clock=time.monotonic, sleep=time.sleep, now=time.time):
        self.probe, self.peers, self.peer_url, self.hostname = probe, list(peers), peer_url, hostname
        self.clock, self.sleep = clock, sleep
        self.lock = threading.Lock()
        self.result = Result(id=f"selftest-{int(now())}", machines=list(peers), started_at=now())
        self.now = now
        self.names = {}  # address -> the node name the run knows it by
        steps = self.result.steps
        for a in self.peers:
            steps += [Step(f"reach:{a}", f"Reach {a}", a), Step(f"version:{a}", f"{a} runs this app version", a),
                      Step(f"auth:{a}", f"{a} lets this computer in", a), Step(f"free:{a}", f"{a} is free to train", a),
                      Step(f"back:{a}", f"{a} can reach this computer", a)]
        steps += [Step("launch", "Start a tiny run on these machines"), Step("joined:local", "This computer's own member starts", self.hostname)]
        for a in self.peers:
            steps += [Step(f"accepted:{a}", f"{a} takes its member job", a), Step(f"joined:{a}", f"{a} joins the run", a),
                      Step(f"delivered:{a}", f"{a} receives the shards and weights", a)]
        steps += [Step(f"round:{n}", f"Round {n + 1}: every machine trains and the average is applied") for n in range(ROUNDS)]
        steps += [Step("checksum", "Every machine holds the same weights as this one"), Step("loss", "The loss falls")]
        self.index = {s.id: s for s in steps}

    # ---- steps
    def _set(self, step_id, status, detail="", fix="", started=None):
        with self.lock:
            step = self.index[step_id]
            step.status, step.detail, step.fix = status, detail, fix
            if started is not None:
                step.seconds = round(self.clock() - started, 2)

    def _ok(self, step_id, detail="", started=None):
        self._set(step_id, PASS, detail, "", started)

    def _bad(self, step_id, detail, fix, started=None):
        self._set(step_id, FAIL, detail, fix, started)
        return False

    def _pending(self):
        return [s for s in self.result.steps if s.status in (PENDING, RUNNING)]

    # ---- the whole test
    def run(self):
        try:
            ok = self._before() and self._launch() and self._watch()
        except Exception as exc:  # noqa: BLE001 - a bug here must still end in a verdict
            first = next(iter(self._pending()), None)
            if first is not None:
                self._bad(first.id, f"The test itself failed: {type(exc).__name__}: {exc}", "Try again; if it keeps failing, report it with this text.")
            ok = False
        finally:
            try:
                self.probe.stop()
            except Exception:  # noqa: BLE001
                pass
        with self.lock:
            for step in self._pending():
                step.status, step.detail = SKIPPED, step.detail or "Not reached: an earlier step failed."
            self.result.state = "passed" if ok and not any(s.status == FAIL for s in self.result.steps) else "failed"
            self.result.finished_at = self.now()
        return self.result

    def _before(self):
        peers = self.probe.survey(self.peers)
        nodes = dict(zip(self.peers, peers))
        for address in self.peers:
            info, started = nodes[address], self.clock()
            if not getattr(info, "online", False):
                return self._bad(f"reach:{address}", getattr(info, "error", "") or f"{address} did not answer.",
                                 f"Start the app on {address}, check its entry on the Machines tab, and that both machines are on the same Tailscale network.", started)
            self._ok(f"reach:{address}", f"Answered: {info.hostname}, {info.device_name or info.device}.", started)
            self.names[address] = info.hostname or address
            own = self.probe.version()
            if own and info.version and own != info.version:
                return self._bad(f"version:{address}", f"{address} runs {info.version}; this computer runs {own}.",
                                 f"Update the app on {address} (or here) so both run the same version.")
            self._ok(f"version:{address}", info.version or "")
            started = self.clock()
            try:
                self.probe.client(address).get_json("/queue")
            except Exception as exc:  # noqa: BLE001 - its text is the machine's answer
                import studio_distributed
                machine = type("M", (), {"node": address, "address": address})()
                return self._bad(f"auth:{address}", studio_distributed.refusal(machine, exc), "Allow this computer on that machine (Machines tab, Access), then test again.", started)
            self._ok(f"auth:{address}", "It accepts calls from this computer.", started)
            if info.busy:
                return self._bad(f"free:{address}", f"{address} is busy with a job of its own.", f"Wait for it to finish or stop it on {address}, then test again.")
            self._ok(f"free:{address}", "Nothing is running there.")
            started = self.clock()
            try:
                answer = self.probe.client(address).post_json("/cluster/selftest/ping", {"url": self.peer_url}, timeout=15) or {}
            except Exception as exc:  # noqa: BLE001
                answer = {"ok": False, "error": str(exc)}
            if not answer.get("ok"):
                return self._bad(f"back:{address}", f"{address} could not reach this computer at {self.peer_url}: {answer.get('error', 'no answer')}",
                                 "Set TRAINER_NODE_ADDRESS on this computer to an address the other machines can reach (its Tailscale name, https://<name>.ts.net), then restart the app.", started)
            self._ok(f"back:{address}", f"Reached {self.peer_url} in {answer.get('seconds', 0) * 1000:.0f} ms.", started)
        return True

    def _launch(self):
        started = self.clock()
        try:
            self.probe.launch()
        except Exception as exc:  # noqa: BLE001
            return self._bad("launch", str(exc), "Free this computer (stop or wait for its current job), then test again.", started)
        self._ok("launch", "Queued; the real invite and join path takes over.", started)
        self.began = self.clock()
        return True

    def _watch(self):
        digested = set()
        while True:
            snap, run, rows = self.probe.snapshot(), self.probe.run(), self.probe.rows()
            over = self.clock() - self.began
            ended = snap is not None and snap.get("state") in ("finished", "stopped")
            failed = self.probe.job_error()
            problem = self._machines(run, snap, rows, over) or self._rounds(snap)
            if problem:
                return False
            if snap is not None and snap.get("round", 0) >= DIGEST_AFTER_ROUND and not ended and not self._done("checksum"):
                self._checksum(snap, digested)
            if ended or failed:
                break
            if over > RUN_TIMEOUT_S:
                step = next(iter(self._pending()), None)
                return self._bad(step.id if step else "launch", f"The run had not finished after {RUN_TIMEOUT_S:.0f} seconds.", "Look at the Training tab for where it is stuck.")
            self.sleep(.05)
        if snap is not None:
            self._rounds(snap)
        if failed and not (snap and snap.get("state") == "finished"):
            step = next(iter(self._pending()), None)
            return self._bad(step.id if step else "loss", failed, "Look at the run on the Training tab.")
        if not self._done("checksum"):
            self._set("checksum", SKIPPED, "The rounds were too short for every machine to download the weights while one was open.")
        return self._loss(snap)

    def _done(self, step_id):
        return self.index[step_id].status in (PASS, FAIL, SKIPPED)

    def _machines(self, run, snap, rows, over):
        """Member job taken, joined, delivered; True when one failed."""
        nodes = (snap or {}).get("nodes", {})
        local = next((m for m in getattr(run, "machines", []) if m.local), None)
        if local is not None and not self._done("joined:local"):
            if local.node in nodes:
                self._ok("joined:local", f"As {local.node}.", self.began)
            elif over > JOIN_TIMEOUT_S:
                row = next((r for r in rows if r["node"] == local.node), None)
                return not self._bad("joined:local", (row or {}).get("note") or "Its member never started.",
                                     "If it says the plan holds the queue, change the plan on this computer (Queue page); otherwise look at the run on the Training tab.")
        for address in self.peers:
            machine = next((m for m in getattr(run, "machines", []) if m.address == address), None)
            row = next((r for r in rows if machine is not None and r["node"] == machine.node), None)
            if machine is None:
                continue
            if not self._done(f"accepted:{address}"):
                if machine.state == "unreachable":
                    return not self._bad(f"accepted:{address}", machine.reason, "Fix that on the machine, then test again.")
                if machine.peer_job:
                    self._ok(f"accepted:{address}", f"It is job {machine.peer_job} in its queue.")
            known = nodes.get(machine.node)
            if self.index[f"accepted:{address}"].status == PASS and not self._done(f"joined:{address}"):
                if known is not None:
                    self._ok(f"joined:{address}", f"As {machine.node}.", self.began)
                elif over > JOIN_TIMEOUT_S:
                    why = (row or {}).get("note") or "It never contacted this computer."
                    return not self._bad(f"joined:{address}", why, "If it says its plan holds the queue, change the plan on that machine; otherwise check that it can reach this computer.")
            if self.index[f"joined:{address}"].status == PASS and not self._done(f"delivered:{address}") and snap is not None:
                lane = next((line["lanes"].get(machine.node) for line in snap.get("timeline", []) if line["round"] == 0 and machine.node in line["lanes"]), None)
                taken = (lane or {}).get("theta_taken")
                if known is not None and known.get("shards") and taken is not None:
                    size = snap.get("theta_bytes", 0) // max(1, len(snap.get("nodes", {})) or 1)
                    self._ok(f"delivered:{address}", f"{known['shards']} shards, and the weights ({megabytes(size)}) taken {taken:.2f} s after round 1 opened.")
        return False

    def _rounds(self, snap):
        if snap is None:
            return False
        for row in snap.get("history", []):
            step_id = f"round:{row['round']}"
            if step_id not in self.index or self._done(step_id):
                continue
            members, sent = row.get("members", []), row.get("contributed", [])
            if len(sent) < len(members) or len(members) < len(self.peers) + 1:
                missing = [n for n in members if n not in sent] or ["a machine"]
                return not self._bad(step_id, f"{', '.join(missing)} sent nothing, or the machine was not in the round (members: {len(members)} of {len(self.peers) + 1}).",
                                     "Look at that machine on the Training tab: its note says why.")
            self._ok(step_id, f"{len(sent)} of {len(members)} sent their change ({megabytes(row.get('delta_bytes', 0))} up, {megabytes(row.get('theta_bytes', 0))} down); "
                              f"{row.get('seconds', 0):.2f} s; loss on the held-back text {row.get('val_loss', 0):.3f}.")
        return False

    def _checksum(self, snap, digested):
        number = snap["round"]
        if number in digested:
            return
        digested.add(number)
        mine = self.probe.theta_digest(number)
        if mine is None:
            return
        for address in self.peers:
            try:
                theirs = self.probe.client(address).post_json("/cluster/selftest/digest", {"url": self.peer_url, "run": self.probe.run_id(), "round": number,
                                                                                               "term": self.probe.term()}, timeout=30) or {}
            except Exception:  # noqa: BLE001 - the round closed under it: another round gets another try
                digested.discard(number)
                return
            if theirs.get("sha256") != mine:
                self._bad("checksum", f"{address} holds weights with checksum {str(theirs.get('sha256'))[:12]}, this computer {mine[:12]} (round {number + 1}).",
                          "The weights changed in transit: check the link (a proxy rewriting bodies, a flaky network) and that both run the same app version.")
                return
        self._ok("checksum", f"Round {number + 1}'s weights have checksum {mine[:12]} on this computer and on {len(self.peers)} other.")

    def _loss(self, snap):
        history = (snap or {}).get("history", [])
        if len(history) < ROUNDS:
            return False
        first, last = history[0].get("val_loss"), history[-1].get("val_loss")
        if first is None or last is None or not last < first:
            return self._bad("loss", f"The loss on the held-back text went from {first} to {last}.", "Training together is not improving the model: look at the Training tab, and try again.")
        self._ok("loss", f"The loss on the held-back text fell from {first:.3f} to {last:.3f}.")
        return True

class AppProbe:
    """The real side of a self-test, over one app's services. `make_job(infos)` builds the run's job from the peers' answers, `enqueue(job)`
    queues and starts it, `job_status(id)` is (status, error) of it in the queue and `remove(id)` takes it off. Everything else is read
    from this process's registries."""

    def __init__(self, make_job, enqueue, job_status, remove, version, client_for=None, survey=None):
        from studio_peer import PeerClient
        self._make, self._enqueue, self._status, self._remove, self._version = make_job, enqueue, job_status, remove, version
        self._client_for = client_for or (lambda address: PeerClient(address, timeout=10, retries=0))
        self._survey, self.infos, self.job = survey, [], None

    def survey(self, addresses):
        import studio_cluster
        self.infos = (self._survey or studio_cluster.poll)(addresses)
        return self.infos

    def client(self, address):
        return self._client_for(address)

    def version(self):
        return self._version

    def run_id(self):
        return self.job.id

    def launch(self):
        import studio_distributed
        self.job = self._make(self.infos)
        studio_distributed.open_run(self.job)
        self._enqueue(self.job)

    def run(self):
        import studio_distributed
        return studio_distributed.RUNS.get(self.job.id) if self.job else None

    def coordinator(self):
        import studio_diloco
        return studio_diloco.COORDINATORS.get(self.job.id) if self.job else None

    def snapshot(self):
        found, run = self.coordinator(), self.run()
        return found.snapshot() if found is not None else getattr(run, "final", None)

    def rows(self):
        import studio_distributed
        run = self.run()
        return studio_distributed.member_rows(run, self.snapshot(), "") if run is not None else []

    def job_error(self):
        if self.job is None:
            return ""
        status, error = self._status(self.job.id)
        return (error or "The run's job on this computer failed.") if status == "failed" else ""

    def term(self):
        return getattr(self.coordinator(), "term", 1)

    def theta_digest(self, number):
        import studio_diloco
        found = self.coordinator()
        if found is None:
            return None
        try:
            return checksum(found.params_bytes(number))
        except studio_diloco.Stale:
            return None

    def stop(self):
        import studio_distributed
        run, found = self.run(), self.coordinator()
        if run is not None and not (found is not None and found.finished):
            studio_distributed.stop_run(run, found)
            self._remove(self.job.id)


def peer_ping(url, client_for=None):
    """What a peer answers `POST /cluster/selftest/ping`: whether it can reach `url` (the owner, as the owner told it to be reached)."""
    from studio_peer import PeerClient
    started = time.monotonic()
    try:
        (client_for or (lambda u: PeerClient(u, timeout=10, retries=0)))(url).get_json("/node")
    except Exception as exc:  # noqa: BLE001 - the text is the answer
        return {"ok": False, "error": str(exc), "seconds": time.monotonic() - started}
    return {"ok": True, "error": "", "seconds": time.monotonic() - started}


def peer_digest(url, run, number, term, client_for=None):
    """What a peer answers `POST /cluster/selftest/digest`: the checksum of the weights it downloads for `number` from the coordinator
    at `url`, the way a member does. Raises PeerError (e.g. 409 when the round has closed)."""
    from studio_peer import PeerClient
    client = (client_for or (lambda u: PeerClient(u, timeout=30, retries=0)))(url)
    started = time.monotonic()
    data = client.get_bytes(f"/collective/{run}/params", {"round": number, "term": term})
    return {"sha256": checksum(data), "bytes": len(data), "seconds": time.monotonic() - started}


class Manager:
    """The one self-test an app runs at a time, and the last result kept in `path` (a JSON file in the workspace)."""

    def __init__(self, path):
        self.path, self.test, self.thread, self.lock = path, None, None, threading.Lock()

    def running(self):
        return self.thread is not None and self.thread.is_alive()

    def start(self, test):
        with self.lock:
            if self.running():
                raise ValueError("A cluster self-test is already running.")
            self.test = test
            self.thread = threading.Thread(target=self._go, args=(test,), daemon=True, name="cluster-selftest")
            self.thread.start()

    def _go(self, test):
        result = test.run()
        try:
            from studio_core import atomic_json
            atomic_json(self.path, result.view())
        except OSError:
            pass  # the verdict is still shown until the app restarts

    def state(self):
        """The running test's steps as they stand, else the last finished one (also across restarts), else {"state": "never"}."""
        test = self.test
        if test is not None:
            with test.lock:
                return test.result.view()
        try:
            from studio_core import read_json
            return read_json(self.path)
        except (OSError, ValueError):
            return {"state": "never", "id": "", "machines": [], "started_at": 0., "finished_at": None, "steps": []}
