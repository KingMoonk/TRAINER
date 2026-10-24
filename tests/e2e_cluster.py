"""A real DiLoCo run across two real API servers on this machine: two `python -m studio_api` processes with their own workspaces and
free ports, talking real HTTP to each other, each running its member in its own worker process. The run goes through the same
start, invite, queue, join, shard, round and average path as a run across two computers; only the network is loopback.

Not named test_*.py so `unittest discover` skips it; CI runs `python -m unittest tests.e2e_cluster -v` (tasks.py test:ci:e2e).
It never uses the live app's ports (8787, 3000) or data: ports are free ephemeral ones, workspaces are temporary folders.
"""
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ROUNDS, INNER = 3, 6
PLAN = {"hours": "any", "start_hour": 22, "end_hour": 7, "memory": "most", "idle_only": False, "idle_after_s": 60, "pause_on_battery": False,
        "keep_awake": False, "chat_follows_best": True}  # a CI runner and a laptop alike: train now, whoever is at the keyboard
START_S, RUN_S = 60, 150


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Node:
    """One app process: `python -m studio_api --workspace <temp folder> --port <free port>`."""

    def __init__(self, folder, name):
        self.port = free_port()
        self.address = f"127.0.0.1:{self.port}"
        self.root = Path(folder) / name
        (self.root / "data").mkdir(parents=True)
        self.log = open(Path(folder) / f"{name}.log", "w", encoding="utf-8")
        # the address the other node is told to join at: without it the app names itself by its tailnet name, which loopback does not have
        env = {**os.environ, "TRAINER_NODE_ADDRESS": f"http://{self.address}"}
        self.process = subprocess.Popen([sys.executable, "-m", "studio_api", "--workspace", str(self.root), "--port", str(self.port), "--allow-cpu"],
                                        cwd=REPO, env=env, stdout=self.log, stderr=subprocess.STDOUT)

    def call(self, method, path, body=None, timeout=60):
        request = urllib.request.Request(f"http://{self.address}{path}", method=method, headers={"content-type": "application/json"},
                                         data=None if body is None else json.dumps(body).encode())
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                text = response.read()
                return json.loads(text) if text else None
        except urllib.error.HTTPError as exc:
            raise AssertionError(f"{method} {path} on {self.address}: {exc.code} {exc.read().decode(errors='replace')[:300]}") from None

    def wait_ready(self):
        end = time.monotonic() + START_S
        while time.monotonic() < end:
            if self.process.poll() is not None:
                raise AssertionError(f"the app on {self.address} exited with {self.process.returncode}: {self.tail()}")
            try:
                return self.call("GET", "/node", timeout=2)
            except (OSError, AssertionError):
                time.sleep(.2)
        raise AssertionError(f"the app on {self.address} did not start in {START_S} s: {self.tail()}")

    def tail(self):
        self.log.flush()
        return Path(self.log.name).read_text(encoding="utf-8", errors="replace")[-1500:]

    def stop(self):
        self.process.terminate()
        try:
            self.process.wait(10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(10)
        self.log.close()

    def idle(self, seconds=30):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            queue = self.call("GET", "/queue")
            if queue["job"] is None and not queue["up_next"]:
                return True
            time.sleep(.5)
        return False

    def traffic_records(self):
        """Read the bounded SSE backfill and close before waiting for live traffic."""
        with urllib.request.urlopen(f"http://{self.address}/traffic?last=2000", timeout=10) as response:
            response.readline()  # connected comment
            response.readline()
            hello = json.loads(response.readline()[6:])
            response.readline()
            records = []
            for _ in range(hello["payload"]["count"]):
                records.append(json.loads(response.readline()[6:])["payload"])
                response.readline()
            return records


class TwoNodeCluster(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from studio_spec import default_spec
        cls.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.nodes = []
        try:
            cls.a, cls.b = Node(cls.tmp.name, "a"), Node(cls.tmp.name, "b")
            cls.nodes = [cls.a, cls.b]
            spec = default_spec("cluster e2e")
            spec["dims"].update(d_model=16, n_layer=1, n_ctx=32)
            for node in cls.nodes:
                node.wait_ready()
                node.call("PUT", "/queue/plan", PLAN)
            for n in range(60):
                cls.a.call("POST", "/data/text/rows", {"fields": {"text": f"Document {n}: the quick brown fox jumps over the lazy dog. " * (n % 4 + 2)}})
            cls.a.call("POST", "/versions", {"spec": spec, "change": "tiny"})
            cls.a.call("PUT", "/cluster/nodes", {"addresses": [cls.b.address]})
            cls.text = cls.a.call("GET", "/pretraining/text_sets")[0]["id"]
        except BaseException:
            cls.tearDownClass()
            raise

    @classmethod
    def tearDownClass(cls):
        for node in cls.nodes:
            node.stop()
        cls.tmp.cleanup()

    def wait_for(self, what, done, seconds):
        end, last = time.monotonic() + seconds, None
        while time.monotonic() < end:
            last = what()
            if done(last):
                return last
            time.sleep(.5)
        self.fail(f"timed out after {seconds} s: {json.dumps(last)[:800]}\n--- a ---\n{self.a.tail()}\n--- b ---\n{self.b.tail()}")

    def test_a_run_across_two_apps_finishes_with_every_member_in_every_round_and_saves_the_version(self):
        started = self.a.call("POST", "/distributed/runs", {"dataset_ids": [self.text], "machines": [self.b.address], "rounds": ROUNDS,
                                                             "inner_steps": INNER, "min_members": 2})
        run_id = started["id"]
        run = self.wait_for(lambda: next(r for r in self.a.call("GET", "/distributed/runs") if r["id"] == run_id),
                            lambda r: r["state"] in ("finished", "stopped", "failed"), RUN_S)
        self.assertEqual(run["state"], "finished", run)
        self.assertEqual(run["round"], ROUNDS)
        self.assertEqual(len(run["members"]), 2)
        self.assertEqual([m["state"] for m in run["members"]], ["finished", "finished"])
        self.assertEqual([m["rounds_done"] for m in run["members"]], [ROUNDS, ROUNDS])  # every member contributed every round
        self.assertEqual([(row["members"], row["contributed"]) for row in run["history"]], [(2, 2)] * ROUNDS)
        self.assertTrue(all(row["bytes_up"] > 0 and row["bytes_down"] > 0 for row in run["history"]))  # real bytes crossed the wire
        self.assertLess(run["history"][-1]["val_loss"], run["history"][0]["val_loss"])
        saved = self.wait_for(lambda: next(r for r in self.a.call("GET", "/distributed/runs") if r["id"] == run_id),
                              lambda r: r["saved_version"] is not None, 60)
        version = next(v for v in self.a.call("GET", "/versions") if v["number"] == saved["saved_version"])
        self.assertEqual(version["status"], "trained")
        self.assertTrue(self.a.idle(), "the owner's queue did not empty after the run")
        self.assertTrue(self.b.idle(), "the peer's queue did not empty after the run")

    def test_the_self_test_proves_every_step_across_the_two_apps(self):
        self.assertTrue(self.a.idle(), "node a is still busy")
        started = self.a.call("POST", "/cluster/selftest", {"machines": [self.b.address]})
        self.assertEqual(started["state"], "running")
        result = self.wait_for(lambda: self.a.call("GET", "/cluster/selftest"), lambda r: r["state"] != "running", RUN_S)
        failed = [(s["id"], s["detail"], s["fix"]) for s in result["steps"] if s["status"] == "fail"]
        self.assertEqual(result["state"], "passed", failed)
        statuses = {s["id"]: s["status"] for s in result["steps"]}
        self.assertEqual(statuses[f"reach:{self.b.address}"], "pass")
        self.assertEqual(statuses[f"back:{self.b.address}"], "pass")
        self.assertEqual([statuses[f"round:{n}"] for n in range(3)], ["pass"] * 3)
        self.assertEqual(statuses["loss"], "pass")
        for s in result["steps"]:
            print(f"{s['status']:5} {s['title']}: {s['detail']} ({s['seconds']} s)")

    def test_receiver_grants_complete_a_real_two_process_run(self):
        started = self.a.call("POST", "/distributed/runs", {"dataset_ids": [self.text], "machines": [self.b.address],
                                                             "rounds": 2, "inner_steps": 4, "min_members": 2,
                                                             "receiver_grants": True})
        run_id = started["id"]
        run = self.wait_for(lambda: next(r for r in self.a.call("GET", "/distributed/runs") if r["id"] == run_id),
                            lambda r: r["state"] in ("finished", "stopped", "failed"), RUN_S)
        self.assertEqual(run["state"], "finished", run)
        self.assertEqual([(row["members"], row["contributed"]) for row in run["history"]], [(2, 2)] * 2)
        records = self.a.traffic_records()
        paths = [row["path"] for row in records if row["direction"] == "in" and row["status"] and row["status"] < 400]
        self.assertTrue(any(path.endswith("/delta/transfers") for path in paths), "no admitted delta announcement")
        self.assertTrue(any("/delta/transfers/" in path and path.endswith("/parts") for path in paths),
                        "no admitted delta body")
        self.assertTrue(any(row["path"].endswith("/params") and
                            row["request"]["headers"].get("x-transfer-admission") == "1" and
                            row["response"]["headers"].get("x-transfer-admission") == "1"
                            for row in records if row["direction"] == "in"), "no admitted round-parameter download")


if __name__ == "__main__":
    unittest.main()
