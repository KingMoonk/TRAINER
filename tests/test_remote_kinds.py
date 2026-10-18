"""The job kinds besides pretrain that run on a peer (#184): compare, post_round and restore_answers. The owner ships their rows in the
dispatch and their checkpoints as store keys, the peer reports the result with the end, and the owner brings the finished weights
home before it says the job ended. A remote pretrain on new text then leaves its restore_answers round. Fake peers for the owner's
side; two real queues (worker processes, a folder for a store) for the whole trip."""
import json
import unittest
from pathlib import Path
from unittest import mock

import studio_remote
import studio_rl
import studio_scheduler as sched
import studio_worker
from studio_queue import Job
from tests import test_queue_cluster as qc  # (its classes are used as bases, not imported, so they do not run twice)
from tests.test_queue_cluster import FakeStore, wait, write_checkpoint

EXPERIMENT = {"question": "Does it help?", "base": {"L": 1, "d_model": 16}, "variants": [{"changes": {}}], "seeds": [1]}
RESULT = {"variants": [], "sweeps": [], "references": {}, "cancelled": False, "lr_retuned": False, "runs": []}


def compareish(job, context, emit, stop, pause):
    """A compare stand-in: the rows it was given, and a result."""
    Path(context.root, "ran.json").write_text(json.dumps({"kind": "compare", "examples": len(job.params.get("examples") or []),
                                                          "experiment": job.params.get("experiment")}))
    emit({"type": "compare", "variant": 0, "seed": 1, "step": 1, "total": 2, "bpc": 3.})
    job.params["result"] = dict(RESULT, runs=[{"variant": 0, "directory": str(Path(job.params["output_dir"]) / "variant_0")}])


def roundish(job, context, emit, stop, pause):
    """A round stand-in: it reads the checkpoint it was handed and the rows that came with it, writes a candidate and its gate result."""
    p = job.params
    out = Path(p["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    Path(context.root, "ran.json").write_text(json.dumps({"kind": job.kind, "checkpoint": p["checkpoint"], "teacher": p.get("teacher"),
                                                          "checkpoint_exists": Path(p["checkpoint"]).is_file(),
                                                          "rows": sorted(k for k in p if k in ("conversations", "problems", "heldout_docs", "heldout_problems",
                                                                                               "pairs", "corrections"))}))
    ok = not p.get("fail")
    write_checkpoint(out / "last.npz", 7)
    job.params["result"] = {"round": 3 if job.kind == "restore_answers" else int(p.get("round", 1)), "ok": ok, "checks": [], "verdict": "moved",
                            "before": {"bpc": 2.}, "after": {"bpc": 1.}, "band": None, "checkpoint": str(out / "last.npz")}


def rows(self, kind):
    return [f"{kind}-row"]


def own_tests_only(cls):
    """The base classes' tests are not run again under this one's name."""
    for base in cls.__mro__[1:]:
        for name in vars(base):
            if name.startswith("test_") and name not in vars(cls):
                setattr(cls, name, None)
    return cls


@own_tests_only
class OwnerDispatch(qc.OwnerSide):
    """The body that leaves for the peer, for each kind that moves."""

    def make_runner(self, store="default"):
        runner = super().make_runner(store)
        self.start = self.root / "start.npz"
        self.teacher = self.root / "teacher.npz"
        write_checkpoint(self.start, 5)
        write_checkpoint(self.teacher, 9)

        def prepare(job):  # what studio_jobs.prepare does for a round: it fills in the checkpoint (and the teacher) from the versions
            job.params.setdefault("checkpoint", str(self.start))
            if job.kind == "restore_answers" and not job.params.get("teacher"):
                job.params["teacher"] = str(self.teacher)
            job.params["output_dir"] = str(self.root / "runs" / f"queue_{job.id}")
            self.prepared.append(job.id)
            return {}

        self.prepared = []
        runner.prepare = prepare
        return runner

    def setUp(self):
        super().setUp()
        patch = mock.patch.object(studio_worker.JobContext, "post_rows", rows)
        patch.start()
        self.addCleanup(patch.stop)
        patch = mock.patch.object(studio_worker.JobContext, "examples", lambda self, params: [{"text": "ex"}])
        patch.start()
        self.addCleanup(patch.stop)

    def test_the_kinds_that_move_are_the_ones_the_scheduler_lists(self):
        self.assertEqual(sched.REMOTE_KINDS, ("pretrain", "compare", "post_round", "restore_answers"))

    def test_a_compare_goes_with_its_experiment_and_its_data(self):
        job = self.runner.enqueue(Job("compare", "Compare", params={"experiment": EXPERIMENT, "datasets": ["a"], "output_dir": "C:/here", "pinned": "gpu-box"}))
        self.place()
        self.assertEqual((job.status, job.node), ("running", "gpu-box"))
        body = self.peer.dispatched[0]
        self.assertEqual(body["job"]["kind"], "compare")
        self.assertEqual(body["job"]["params"]["experiment"], EXPERIMENT)
        self.assertEqual(body["job"]["params"]["examples"], [{"text": "ex"}])
        self.assertNotIn("output_dir", body["job"]["params"])
        self.assertEqual(body["fetch"], {})

    def test_a_round_one_goes_with_its_rows_and_its_checkpoint_as_a_key(self):
        job = self.runner.enqueue(Job("post_round", "Round 1", params={"round": 1, "version": 3, "steps": 10}))
        self.place()
        self.assertEqual((job.status, job.node, self.prepared), ("running", "gpu-box", [job.id]))
        body = self.peer.dispatched[0]
        params = body["job"]["params"]
        self.assertEqual({k: params[k] for k in ("conversations", "heldout_docs", "heldout_problems")},
                         {"conversations": ["conversations-row"], "heldout_docs": ["heldout_docs-row"], "heldout_problems": ["heldout_problems-row"]})
        self.assertNotIn("examples", params)  # (the pretraining text is no use to a round, and large)
        for local in ("version", "output_dir", "checkpoint", "teacher"):
            self.assertNotIn(local, params)
        self.assertEqual(params["steps"], 10)
        key = body["fetch"]["checkpoint"]
        self.assertEqual(list(body["fetch"]), ["checkpoint"])
        self.assertEqual(self.store.published[0][0], key)
        self.assertEqual(self.store.published[0][2], self.start)
        self.assertEqual(job.params["fetch"], body["fetch"])  # recorded, so a sweep leaves the key alone

    def test_a_restore_answers_goes_with_its_teacher_and_the_rows_of_round_three(self):
        job = self.runner.enqueue(Job("restore_answers", "Restore", params={"round": 3, "teacher": None}))
        self.place()
        body = self.peer.dispatched[0]
        self.assertEqual(sorted(body["fetch"]), ["checkpoint", "teacher"])
        for kind in ("conversations", "heldout_docs", "heldout_problems"):
            self.assertEqual(body["job"]["params"][kind], [f"{kind}-row"])
        self.assertEqual([p[2] for p in self.store.published], [self.start, self.teacher])
        self.assertEqual(job.status, "running")

    def test_what_each_round_reads_goes_and_no_more(self):
        self.assertEqual(studio_rl.rows_needed(2, {}), ["problems", "heldout_docs", "heldout_problems"])
        self.assertEqual(studio_rl.rows_needed(4, {}), ["heldout_docs", "heldout_problems", "pairs"])
        self.assertEqual(studio_rl.rows_needed(3, {"teacher_choice": "correction"}), ["heldout_docs", "heldout_problems", "corrections"])
        self.assertEqual(studio_rl.rows_needed(3, {"prompts": ["x"], "heldout_docs": []}), ["heldout_problems"])
        self.assertEqual(studio_rl.rows_needed(1, {"conversations": []}), ["heldout_docs", "heldout_problems"])

    def test_a_round_whose_checkpoint_cannot_be_found_waits_with_the_reason(self):
        job = self.runner.enqueue(Job("post_round", "Round 1", params={"round": 1, "checkpoint": str(self.root / "gone.npz")}))
        self.place()
        self.assertEqual((job.status, self.peer.dispatched), ("waiting", []))
        self.assertIn("gone.npz", self.runner.cluster.waiting[job.id])

    def finish(self, job, **body):
        self.place()
        self.runner.cluster.finished(job.id, {"epoch": 1, "status": "done", **body})

    def test_a_finished_compare_is_written_where_a_local_one_would_be(self):
        job = self.runner.enqueue(Job("compare", "Compare", params={"experiment": EXPERIMENT, "pinned": "gpu-box"}))
        self.finish(job, result=RESULT)
        folder = self.root / "runs" / f"compare_{job.id}"
        self.assertEqual(json.loads((folder / "results.json").read_text()), RESULT)
        self.assertEqual(json.loads((folder / "experiment.json").read_text())["question"], EXPERIMENT["question"])
        self.assertEqual((job.status, self.store.pulled), ("done", []))
        self.assertEqual(self.sent("job")[-1], {"job": job.id, "status": "done", "error": ""})

    def test_a_round_that_passed_comes_home_with_its_checkpoint_before_the_end_is_said(self):
        job = self.runner.enqueue(Job("post_round", "Round 1", params={"round": 1, "version": 3}))
        heard = []

        def hear(kind, event):  # what the app sees on the end: the file the result names is already here
            if kind == "job" and event["status"] == "done":
                heard.append(Path(job.params["result"]["checkpoint"] or "").is_file())
        self.runner.subscribe(hear)
        self.finish(job, key="gpu-box/v0-r", step=7, result={"round": 1, "ok": True, "checks": [], "checkpoint": "D:/peer/last.npz"})
        out = self.root / "runs" / f"queue_{job.id}"
        self.assertEqual(self.store.pulled, ["gpu-box/v0-r"])
        self.assertTrue((out / "best.npz").is_file())
        self.assertEqual(job.params["result"]["checkpoint"], str(out / "best.npz"))  # not the peer's path
        self.assertNotIn("landing", job.params)
        self.assertEqual(heard, [True])

    def test_a_round_that_failed_its_gate_brings_nothing_and_names_no_checkpoint(self):
        job = self.runner.enqueue(Job("post_round", "Round 1", params={"round": 1}))
        self.finish(job, result={"round": 1, "ok": False, "checks": [], "checkpoint": "D:/peer/last.npz"})
        self.assertEqual((job.status, self.store.pulled, job.params["result"]["ok"], job.params["result"]["checkpoint"]), ("done", [], False, None))

    def test_a_failed_round_keeps_the_peers_sentence_and_no_result(self):
        job = self.runner.enqueue(Job("restore_answers", "Restore", params={"round": 3}))
        self.finish(job, status="failed", error="Nothing to learn from.", result={"round": 3, "ok": True})
        self.assertEqual((job.status, job.error, "result" in job.params), ("failed", "Nothing to learn from.", False))

    def test_a_remote_pretrain_on_new_text_leads_to_restore_answers_once_its_version_is_here(self):
        job = self.runner.enqueue(self.job(user_text=True, previous_best=2))
        queued = []
        self.runner.subscribe(lambda kind, event: queued.append([j.kind for j in self.runner.jobs]) if kind == "job" and event["status"] == "done" else None)
        self.finish(job, key="gpu-box/v0-abc", step=1000)
        self.assertEqual(self.store.pulled, ["gpu-box/v0-abc"])
        self.assertTrue((Path(job.params["output_dir"]) / "best.npz").is_file())
        follow = [j for j in self.runner.jobs if j.kind == "restore_answers"]
        self.assertEqual(len(follow), 1)
        self.assertEqual((follow[0].params, follow[0].status), ({"round": 3, "teacher": 2}, "waiting"))
        self.assertEqual(queued, [["pretrain"]])  # (the version was recorded on the end, before the round could be placed)

    def test_a_remote_pretrain_without_new_text_leaves_nothing_behind(self):
        job = self.runner.enqueue(self.job())
        self.finish(job, key="gpu-box/v0-abc", step=1000)
        self.assertEqual([j.kind for j in self.runner.jobs], ["pretrain"])

    def test_a_pretrain_that_was_stopped_is_not_pulled_and_adds_nothing(self):
        job = self.runner.enqueue(self.job(user_text=True))
        self.finish(job, status="stopped", key="gpu-box/v0-abc", step=3)
        self.assertEqual((job.status, self.store.pulled, [j.kind for j in self.runner.jobs]), ("stopped", [], ["pretrain"]))

    def test_a_pull_that_fails_is_tried_again_and_then_fails_the_job_with_its_key(self):
        job = self.runner.enqueue(self.job(user_text=True))
        failing = mock.patch.object(FakeStore, "pull", side_effect=OSError("The bucket did not answer."))
        with failing:
            self.finish(job, key="gpu-box/v0-abc", step=1000)
            self.assertEqual((job.status, self.sent("job")[-1]), ("done", {"job": job.id, "status": "running"}))  # (nothing said yet)
            self.assertTrue(job.params["landing"])
            self.runner.cluster.tick()
            self.assertTrue(job.params["landing"])  # too soon to try again
            for _ in range(2):
                self.clock.advance(studio_remote.LAND_RETRY_S + 1)
                self.runner.cluster.tick()
        self.assertEqual(job.status, "failed")
        self.assertIn("gpu-box/v0-abc", job.error)
        self.assertIn("The bucket did not answer.", job.error)
        self.assertNotIn("landing", job.params)
        self.assertEqual([j.kind for j in self.runner.jobs], ["pretrain"])  # no follow-up for a version that is not here
        self.assertEqual(self.sent("job")[-1], {"job": job.id, "status": "failed", "error": job.error})

    def test_a_pull_the_app_was_interrupted_in_is_finished_when_it_comes_back(self):
        job = self.runner.enqueue(self.job(user_text=True))
        with mock.patch.object(FakeStore, "pull", side_effect=OSError("down")):
            self.finish(job, key="gpu-box/v0-abc", step=1000)
        self.assertTrue(job.params["landing"])
        self.clock.advance(studio_remote.LAND_RETRY_S + 1)
        self.runner.cluster.tick()  # (a fresh start finds the flag in the saved queue; the retry clock is the only memory lost)
        self.assertNotIn("landing", job.params)
        self.assertEqual(self.sent("job")[-1], {"job": job.id, "status": "done", "error": ""})
        self.assertEqual(len([j for j in self.runner.jobs if j.kind == "restore_answers"]), 1)
        self.runner.cluster.tick()
        self.assertEqual(len([j for j in self.runner.jobs if j.kind == "restore_answers"]), 1)  # once


class RoundGuard(unittest.TestCase):
    def test_a_comparisons_progress_reaches_the_owner_and_a_gate_does_not(self):
        event = {"type": "compare", "variant": 0, "seed": 1, "step": 1, "total": 2, "bpc": 3.}
        self.assertEqual(sched.relayed(event), event)
        self.assertIsNone(sched.relayed({"type": "gate", "checkpoint": "D:/peer/last.npz"}))

    def test_a_round_another_node_owns_is_never_resolved_against_this_nodes_versions(self):
        services = mock.Mock()
        job = Job("post_round", "Round", params={"checkpoint": "pulled.npz"}, owner="owner-a")
        studio_rl.resolve(1, job, services)
        self.assertEqual(job.params, {"checkpoint": "pulled.npz"})
        self.assertEqual(services.mock_calls, [])
        with self.assertRaises(ValueError):
            studio_rl.resolve(3, Job("restore_answers", "Round", params={}, owner="owner-a"), services)

    def test_a_local_round_is_still_resolved_here(self):
        services = mock.Mock()
        services.current_version.return_value = mock.Mock(best_checkpoint="cur.npz")
        services.best_version.return_value = mock.Mock(best_checkpoint="best.npz")
        job = Job("restore_answers", "Round", params={})
        studio_rl.resolve(3, job, services)
        self.assertEqual((job.params["checkpoint"], job.params["teacher"]), ("cur.npz", "best.npz"))


@own_tests_only
class RemoteTrip(qc.TwoNodes):
    """A job of each kind runs on the second queue's worker process and ends on the first."""

    def setUp(self):
        super().setUp()
        self.b.handlers.update({"compare": compareish, "post_round": roundish, "restore_answers": roundish})
        patch = mock.patch.object(studio_worker.JobContext, "post_rows", rows)
        patch.start()
        self.addCleanup(patch.stop)
        patch = mock.patch.object(studio_worker.JobContext, "examples", lambda self, params: [{"text": "ex"}])
        patch.start()
        self.addCleanup(patch.stop)

    def run_remote(self, job):
        self.a.enqueue(job)
        self.a.cluster.poll_once()
        self.a._place_waiting()
        self.assertEqual((job.status, job.node), ("running", "gpu-box"))
        self.assertTrue(wait(lambda: any(k == "job" and e.get("job") == job.id and e.get("status") in ("done", "failed", "stopped")
                                         for k, e in list(self.a_events))), "the owner did not say the job ended")
        return job

    def test_a_compare_runs_on_the_peer_and_its_result_reaches_the_owner(self):
        job = self.run_remote(Job("compare", "Compare", params={"experiment": EXPERIMENT, "pinned": "gpu-box"}))
        self.assertEqual(job.status, "done", job.error)
        ran = self.ran()
        self.assertEqual((ran["kind"], ran["examples"], ran["experiment"]), ("compare", 1, EXPERIMENT))
        folder = self.a_root / "runs" / f"compare_{job.id}"
        self.assertEqual(json.loads((folder / "results.json").read_text())["runs"][0]["variant"], 0)
        self.assertTrue((folder / "experiment.json").is_file())

    def test_a_round_runs_on_the_peer_from_the_owners_checkpoint_and_comes_home(self):
        start = self.a_root / "start.npz"
        write_checkpoint(start, 4)
        job = self.run_remote(Job("post_round", "Round 1", params={"round": 1, "checkpoint": str(start), "version": 2}))
        self.assertEqual(job.status, "done", job.error)
        ran = self.ran()
        self.assertTrue(ran["checkpoint_exists"])
        self.assertNotIn(str(self.a_root), ran["checkpoint"])  # (pulled from the store, not the owner's path)
        self.assertEqual(ran["rows"], ["conversations", "heldout_docs", "heldout_problems"])
        result = job.params["result"]
        self.assertTrue(result["ok"])
        self.assertTrue(result["checkpoint"].startswith(str(self.a_root)))
        self.assertTrue(Path(result["checkpoint"]).is_file())
        self.assertEqual(studio_remote.checkpoint_step(result["checkpoint"]), 7)

    def test_a_restore_answers_runs_on_the_peer_with_its_teacher(self):
        start, teacher = self.a_root / "start.npz", self.a_root / "teacher.npz"
        write_checkpoint(start, 4)
        write_checkpoint(teacher, 6)
        job = self.run_remote(Job("restore_answers", "Restore", params={"round": 3, "checkpoint": str(start), "teacher": str(teacher)}))
        self.assertEqual(job.status, "done", job.error)
        ran = self.ran()
        self.assertEqual(ran["kind"], "restore_answers")
        self.assertTrue(Path(ran["teacher"]).is_file())
        self.assertNotIn(str(self.a_root), ran["teacher"])
        self.assertTrue(Path(job.params["result"]["checkpoint"]).is_file())

    def test_a_round_that_fails_its_gate_sends_its_result_and_no_weights(self):
        start = self.a_root / "start.npz"
        write_checkpoint(start, 4)
        job = self.run_remote(Job("post_round", "Round 1", params={"round": 1, "checkpoint": str(start), "fail": True}))
        self.assertEqual((job.status, job.params["result"]["ok"], job.params["result"]["checkpoint"]), ("done", False, None))
        self.assertNotIn("remote_result", job.params)

    def test_a_remote_pretrain_on_new_text_leaves_restore_answers_on_the_owner(self):
        job = self.run_remote(self.job(6, user_text=True, previous_best=2))
        self.assertEqual(job.status, "done", job.error)
        self.assertTrue((Path(job.params["output_dir"]) / "best.npz").is_file())
        self.assertTrue(wait(lambda: any(j.kind == "restore_answers" for j in self.a.jobs)))
        follow = next(j for j in self.a.jobs if j.kind == "restore_answers")
        self.assertEqual(follow.params, {"round": 3, "teacher": 2})

    def test_the_peer_refuses_a_kind_that_stays_with_its_owner(self):
        for kind in ("quicktest", "import_pretrained", "checkpoint_now"):
            with self.assertRaises(sched.Refused) as raised:
                self.b.cluster.accept({"id": "x", "kind": kind, "params": {}}, sched.Lease("x", "gpu-box", 1, 30.), "owner-a", {})
            self.assertIn("only runs on the machine that queued it", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
