"""Passive training engine: plan, pure decisions, the SQLite record and the whole loop driven with fakes."""
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import studio_passive as passive
from studio_passive import PassiveStore, merge_plan, next_move, plateaued, run_session, stop_reason

EXAMPLES = [{"user": f"say {w}", "assistant": w} for w in ("yes", "no", "cat", "dog", "sun", "red", "sky", "big")] * 3


def plan_with(**groups):
    base = {"stop": {"duration": 1, "unit": "days", "patience_rounds": 0}, "plateau": {"checks": 2, "max_steps": 50},
            "worse": {"limit": 2, "pivots": ["halve_lr"]}, "teacher": {"polish": True, "lessons": True}}
    for key, value in groups.items():
        base[key] = {**base[key], **value} if isinstance(base.get(key), dict) else value
    return merge_plan(base)


class Rig:
    """A session in a temp folder with fake train / evaluate / teacher_round / build."""

    def __init__(self, test, errors, checks=(3.0, 2.0, 1.0), teacher=None, fail_train_on=None, **groups):
        self.dir = tempfile.TemporaryDirectory()
        test.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name)
        self.store = PassiveStore(self.root)
        self.id = self.store.create(plan_with(**groups))
        self.errors, self.checks, self.teacher, self.fail_on = list(errors), checks, teacher, fail_train_on
        self.calls, self.teacher_calls, self.emitted = [], [], 0
        self.time = [1000.0]
        self.on_train = None
        self.fail_with = RuntimeError("boom")

    def train(self, config, examples, output_dir, emit, stop, pause, init_from=None):
        self.calls.append({"config": config, "examples": list(examples), "init_from": init_from, "dir": Path(output_dir)})
        if self.fail_on == len(self.calls):
            self.fail_on = None
            raise self.fail_with
        Path(output_dir).mkdir(parents=True)
        (Path(output_dir) / "best.npz").write_bytes(b"x")
        if self.on_train:
            self.on_train(self, len(self.calls))
        for i, value in enumerate(self.checks):
            self.emitted += 1
            emit({"type": "metric", "step": i + 1, "loss": 1.0, "val_loss": value})
            if stop.is_set():
                break

    def evaluate(self, path, examples):
        return self.errors.pop(0)

    def teacher_round(self, directory, plan, settings, checkpoint, examples, stop, emit):
        self.teacher_calls.append(checkpoint)
        return self.teacher(len(self.teacher_calls), checkpoint)

    def run(self):
        return run_session(self.root, self.id, train=self.train, evaluate=self.evaluate, teacher_round=self.teacher_round,
                           build=lambda root, data: [dict(e) for e in EXAMPLES], clock=lambda: self.time[0])

    def row(self):
        return self.store.get(self.id)

    def types(self):
        return [e["type"] for e in self.store.journal(self.id)]

    def set_teacher(self, name="tiny-teacher"):
        from studio_rlhf import save_settings
        save_settings(self.root, {"teacher_model": name})


class PlanTests(unittest.TestCase):
    def test_defaults_and_overrides(self):
        plan = merge_plan(None)
        self.assertEqual(plan, passive.DEFAULT_PLAN)
        merged = merge_plan({"stop": {"duration": 2, "unit": "days"}, "worse": {"limit": 3}})
        self.assertEqual((merged["stop"]["duration"], merged["stop"]["unit"], merged["stop"]["patience_rounds"]), (2, "days", 3))
        self.assertEqual(merged["worse"]["pivots"], passive.DEFAULT_PLAN["worse"]["pivots"])
        merged["worse"]["pivots"].append("x")
        self.assertEqual(len(passive.DEFAULT_PLAN["worse"]["pivots"]), 3)  # defaults are never shared

    def test_rejections(self):
        for bad in ({"stop": {"unit": "weeks"}}, {"stop": {"duration": 0}}, {"stop": {"duration": -1}}, {"stop": {"duration": "8"}},
                    {"worse": {"pivots": ["explode"]}}, {"worse": {"limit": 0}}, {"plateau": {"checks": 1}}, {"limits": {"memory_fraction": 2}},
                    {"limits": {"keep_checkpoints": 0}}, {"stop": {"win_rate": 5}}, {"data": {"balance": "x"}}, {"mystery": 1}, {"stop": {"mystery": 1}}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                merge_plan(bad)

    def test_non_finite_numbers_and_config_are_rejected(self):
        for bad in ({"stop": {"duration": float("inf")}}, {"stop": {"duration": float("nan")}}, {"stop": {"target_error": float("nan")}},
                    {"stop": {"target_error": float("inf")}}, {"plateau": {"checks": float("inf")}}, {"limits": {"memory_fraction": float("nan")}},
                    {"plateau": {"min_improvement": float("inf")}}, {"config": "fast"}, {"config": [1]}):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, r"number|group of settings"):
                merge_plan(bad)

    def test_duration_seconds(self):
        self.assertEqual(passive.duration_seconds(merge_plan({"stop": {"duration": 30, "unit": "minutes"}})), 1800)
        self.assertEqual(passive.duration_seconds(merge_plan({})), 8 * 3600)
        self.assertEqual(passive.duration_seconds(merge_plan({"stop": {"duration": 2, "unit": "days"}})), 172800)

    def test_plan_summary_rows(self):
        summary = passive.plan_summary(merge_plan({"stop": {"target_error": 1.5}}))
        self.assertEqual(set(summary), {"stop", "limits", "plateau", "teacher", "worse"})
        self.assertIn("8 hours", summary["stop"])
        self.assertIn("3 rounds", summary["stop"])
        self.assertIn("1.5", summary["stop"])
        self.assertIn("halve the learning rate", summary["worse"])


class JournalTests(unittest.TestCase):
    def test_bad_lines_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "journal.jsonl"
            path.write_bytes(b'{"a": 1}\nnot json\n{"b": 2}\n{"c": "caf\xc3')  # killed in the middle of a two-byte character
            self.assertEqual(passive._read_jsonl(path), [{"a": 1}, {"b": 2}])


class DecisionTests(unittest.TestCase):
    plan = merge_plan({"plateau": {"checks": 3, "min_improvement": 0.01}})

    def test_plateaued(self):
        self.assertFalse(plateaued([1.0, 0.9, 0.8, 0.7], self.plan))  # still improving
        self.assertTrue(plateaued([1.0, 1.0, 1.0], self.plan))  # flat
        self.assertTrue(plateaued([2.0, 1.0, 1.0, 0.999, 1.0], self.plan))  # only the last 3 count
        self.assertFalse(plateaued([1.0, 1.0], self.plan))  # too few checks
        self.assertFalse(plateaued([], self.plan))

    @staticmethod
    def state(**kw):
        return {"active_seconds": 0.0, "tick": None, "rounds_since_best": 0, "worse_streak": 0, "pivot_index": 0, "best": 1,
                "checkpoints": [{"n": 1, "error": 2.0}], "win_rate": None, **kw}

    def test_stop_reason(self):
        plan = merge_plan({"stop": {"duration": 1, "unit": "hours", "patience_rounds": 2, "target_error": 1.0, "win_rate": 0.6}})
        self.assertIsNone(stop_reason(self.state(), plan, 0))
        self.assertEqual(stop_reason(self.state(active_seconds=3600), plan, 0), "duration")
        self.assertEqual(stop_reason(self.state(active_seconds=3000, tick=100.0), plan, 800.0), "duration")  # time since the last tick counts
        self.assertIsNone(stop_reason(self.state(active_seconds=3000, tick=100.0, paused=True), plan, 800.0))  # paused time does not
        self.assertEqual(stop_reason(self.state(rounds_since_best=2), plan, 0), "patience")
        self.assertEqual(stop_reason(self.state(checkpoints=[{"n": 1, "error": 0.9}]), plan, 0), "target")
        self.assertEqual(stop_reason(self.state(win_rate=0.7), plan, 0), "win_rate")
        self.assertEqual(stop_reason(self.state(worse_streak=2, pivot_index=3), plan, 0), "pivots")
        self.assertIsNone(stop_reason(self.state(worse_streak=2, pivot_index=0), plan, 0))
        off = merge_plan({"stop": {"patience_rounds": 0}})
        self.assertIsNone(stop_reason(self.state(rounds_since_best=50), off, 0))

    def test_next_move(self):
        plan = merge_plan({"data": {"sources": ["a", "b"]}})
        self.assertEqual(next_move(self.state(worse_streak=1), plan), {"action": "continue"})
        first = next_move(self.state(worse_streak=2), plan)
        self.assertEqual((first["action"], first["pivot"]), ("rollback", "halve_lr"))
        self.assertEqual(next_move(self.state(worse_streak=2, pivot_index=1), plan)["pivot"], "next_dataset")
        self.assertEqual(next_move(self.state(worse_streak=2, pivot_index=2), plan)["pivot"], "drop_lessons")
        self.assertEqual(next_move(self.state(worse_streak=2, pivot_index=3), plan), {"action": "stop", "reason": "pivots"})
        single = merge_plan({"data": {"sources": ["a"]}})  # next_dataset needs two or more sources
        self.assertEqual(next_move(self.state(worse_streak=2, pivot_index=1), single)["pivot"], "drop_lessons")


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.store = PassiveStore(self.dir.name)

    def test_create_get_update(self):
        self.assertIsNone(self.store.active())
        self.assertIsNone(self.store.latest())
        sid = self.store.create({"stop": {"duration": 3}})
        row = self.store.get(sid)
        self.assertEqual((row["status"], row["command"], row["plan"]["stop"]["duration"], row["state"], row["restarts"]), ("queued", "", 3, {}, []))
        self.store.update(sid, status="running", state={"round": 2}, restarts=[1.5])
        row = self.store.get(sid)
        self.assertEqual((row["status"], row["state"], row["restarts"]), ("running", {"round": 2}, [1.5]))
        self.assertEqual(self.store.active()["id"], sid)
        with self.assertRaises(ValueError):
            self.store.update(sid, nonsense=1)
        self.assertIsNone(self.store.get("missing"))

    def test_one_active_session(self):
        first = self.store.create({})
        with self.assertRaises(ValueError):
            self.store.create({})
        self.store.update(first, status="paused")
        with self.assertRaises(ValueError):
            self.store.create({})
        self.store.update(first, status="finished")
        second = self.store.create({})
        self.assertEqual(self.store.active()["id"], second)
        self.assertEqual(self.store.latest()["id"], second)
        self.assertEqual([r["id"] for r in self.store.list()], [second, first])

    def test_command_beat_journal_lessons(self):
        sid = self.store.create({})
        self.store.command(sid, "pause")
        self.assertEqual(self.store.get(sid)["command"], "pause")
        with self.assertRaises(ValueError):
            self.store.command(sid, "explode")
        self.store.beat(sid, 4242)
        row = self.store.get(sid)
        self.assertEqual(row["trainer_pid"], 4242)
        self.assertGreater(row["heartbeat"], 0)
        self.assertEqual(self.store.journal(sid), [])
        self.store.append(sid, {"type": "start"})
        self.store.append(sid, {"type": "stop", "reason": "user"})
        journal = self.store.journal(sid)
        self.assertEqual([e["type"] for e in journal], ["start", "stop"])
        self.assertIn("at", journal[0])
        self.assertEqual(self.store.lessons(sid), [])
        self.store.set_lessons(sid, [{"id": "a", "status": "pending"}])
        self.assertEqual(self.store.lessons(sid)[0]["status"], "pending")
        self.assertEqual(self.store.directory(sid), Path(self.dir.name) / "runs" / sid)

    def test_trainer_command(self):
        command = passive.trainer_command("/r", "passive_1")
        self.assertEqual(command[1:], ["-m", "studio_passive", "run", "passive_1", "--root", "/r"])


class LoopTests(unittest.TestCase):
    def test_step_back_and_recover(self):
        # 3 is the big step, 4 and 5 get worse, so it goes back to 3 with a smaller learning rate and 6 is better still.
        rig = Rig(self, [5.0, 4.0, 2.0, 3.0, 3.5, 1.5], stop={"target_error": 1.6}, plateau={"checks": 9, "max_steps": 50})
        state = rig.run()
        row = rig.row()
        self.assertEqual((row["status"], state["stop_reason"], state["best"]), ("finished", "target", 6))
        rollback = [e for e in rig.store.journal(rig.id) if e["type"] == "rollback"]
        self.assertEqual(len(rollback), 1)
        self.assertEqual((rollback[0]["from"], rollback[0]["to"], rollback[0]["pivot"], rollback[0]["discarded"]), (5, 3, "halve_lr", [4, 5]))
        discarded = {c["n"]: c["discarded"] for c in state["checkpoints"]}
        self.assertEqual(discarded, {1: False, 2: False, 3: False, 4: True, 5: True, 6: False})
        six = state["checkpoints"][-1]
        self.assertEqual((six["parent"], six["pivot"]), (3, "halve_lr"))
        self.assertEqual(rig.calls[5]["init_from"], state["checkpoints"][2]["path"])  # round 6 restarts from checkpoint 3
        self.assertAlmostEqual(rig.calls[5]["config"]["learning_rate"], rig.calls[0]["config"]["learning_rate"] / 2)
        self.assertEqual(rig.calls[3]["init_from"], state["checkpoints"][2]["path"])  # 4 followed on from 3 before the step back
        self.assertEqual(rig.calls[0]["config"]["steps"], 50)
        self.assertIn("start", rig.types())
        self.assertGreater(row["heartbeat"], 0)
        held = (rig.root / "runs" / rig.id / "eval.jsonl").read_text().count("\n")
        self.assertEqual(held + len((rig.root / "runs" / rig.id / "snapshot.jsonl").read_text().splitlines()), len(EXAMPLES))

    def test_pivots_exhausted(self):
        rig = Rig(self, [1.0, 2.0, 3.0, 4.0, 5.0], worse={"limit": 2, "pivots": ["halve_lr"]}, plateau={"checks": 9})
        state = rig.run()
        self.assertEqual(state["stop_reason"], "pivots")
        self.assertEqual(state["best"], 1)
        self.assertEqual(rig.types().count("rollback"), 1)

    def test_default_patience_lets_every_change_be_tried(self):
        rig = Rig(self, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], stop={"patience_rounds": 3}, worse={"limit": 2, "pivots": ["halve_lr", "drop_lessons"]}, plateau={"checks": 9})
        state = rig.run()
        self.assertEqual((state["stop_reason"], rig.types().count("rollback"), state["best"]), ("pivots", 2, 1))

    def test_stops_on_duration(self):
        rig = Rig(self, [3.0, 2.0, 1.0, 0.5], stop={"duration": 2, "unit": "hours"}, plateau={"checks": 9})
        rig.on_train = lambda r, n: r.time.__setitem__(0, r.time[0] + 3600)
        state = rig.run()
        self.assertEqual((state["stop_reason"], rig.row()["status"]), ("duration", "finished"))
        self.assertLessEqual(len(state["checkpoints"]), 2)
        self.assertGreaterEqual(state["active_seconds"], 7200)

    def test_duration_stops_inside_a_long_round(self):
        rig = Rig(self, [3.0], checks=[3.0] * 20, stop={"duration": 1, "unit": "hours"}, plateau={"checks": 99})
        rig.on_train = lambda r, n: r.time.__setitem__(0, r.time[0] + 4000)
        rig.run()
        self.assertEqual(rig.emitted, 1)  # the first check already showed the time was up

    def test_stops_on_patience(self):
        rig = Rig(self, [3.0, 4.0, 5.0, 6.0], stop={"patience_rounds": 2}, worse={"limit": 9}, plateau={"checks": 9})
        state = rig.run()
        self.assertEqual((state["stop_reason"], len(state["checkpoints"]), state["best"]), ("patience", 3, 1))

    def test_round_ends_when_error_plateaus(self):
        rig = Rig(self, [1.0], checks=[1.0] * 10, stop={"target_error": 5}, plateau={"checks": 3})
        state = rig.run()
        self.assertEqual(rig.emitted, 3)
        self.assertEqual(len(state["checkpoints"]), 1)
        self.assertEqual(len([e for e in rig.store.journal(rig.id) if e["type"] == "check"]), 3)

    def test_stop_command_keeps_best(self):
        rig = Rig(self, [3.0, 2.0, 1.0], plateau={"checks": 9})
        rig.on_train = lambda r, n: r.store.command(r.id, "stop") if n == 2 else None
        state = rig.run()
        row = rig.row()
        self.assertEqual((row["status"], state["stop_reason"], row["command"]), ("stopped", "user", ""))
        self.assertEqual(rig.types().count("round"), 2)
        self.assertTrue(Path(state["checkpoints"][state["best"] - 1]["path"]).exists())
        self.assertEqual(rig.types()[-1], "stop")

    def test_a_stop_survives_a_trainer_that_dies_before_acting_on_it(self):
        rig = Rig(self, [3.0, 2.0], plateau={"checks": 9})

        def dies(config, examples, output_dir, emit, stop, pause, init_from=None):
            rig.store.command(rig.id, "stop")
            emit({"type": "metric", "step": 1, "loss": 1.0, "val_loss": 3.0})  # the trainer takes the command here
            raise KeyboardInterrupt  # and is killed before it writes the stopped status
        with self.assertRaises(KeyboardInterrupt):
            run_session(rig.root, rig.id, train=dies, evaluate=rig.evaluate, build=lambda root, data: [dict(e) for e in EXAMPLES], clock=lambda: rig.time[0])
        self.assertEqual(rig.row()["command"], "stop")
        state = run_session(rig.root, rig.id, train=rig.train, evaluate=rig.evaluate, build=lambda root, data: [dict(e) for e in EXAMPLES], clock=lambda: rig.time[0])  # the watchdog restarts it
        self.assertEqual((rig.row()["status"], rig.row()["command"], state["stop_reason"], len(rig.calls)), ("stopped", "", "user", 0))

    def test_a_stop_while_the_data_is_still_being_prepared_ends_cleanly(self):
        rig = Rig(self, [3.0], plateau={"checks": 9})

        def cancelled(config, examples, output_dir, emit, stop, pause, init_from=None):
            rig.store.command(rig.id, "stop")
            self.assertTrue(stop.wait(5))  # the heartbeat thread takes the command
            raise InterruptedError("Training preparation cancelled.")
        with patch.object(passive, "POLL", 0.01):
            state = run_session(rig.root, rig.id, train=cancelled, evaluate=rig.evaluate, build=lambda root, data: [dict(e) for e in EXAMPLES], clock=lambda: rig.time[0])
        self.assertEqual((rig.row()["status"], state["stop_reason"], state["checkpoints"]), ("stopped", "user", []))

    def test_a_diverging_round_steps_back_and_halves_the_learning_rate(self):
        rig = Rig(self, [5.0, 4.0, 3.0], stop={"target_error": 3.5}, plateau={"checks": 9}, fail_train_on=3)
        rig.fail_with = ValueError("Non-finite loss. Reduce the learning rate.")
        state = rig.run()
        self.assertEqual((rig.row()["status"], state["stop_reason"], state["best"]), ("finished", "target", 3))
        self.assertEqual([c["n"] for c in state["checkpoints"]], [1, 2, 3])
        move = [e for e in rig.store.journal(rig.id) if e["type"] == "rollback"]
        self.assertEqual((move[0]["from"], move[0]["to"], move[0]["pivot"]), (2, 2, "halve_lr"))
        self.assertEqual(rig.calls[3]["init_from"], state["checkpoints"][1]["path"])
        self.assertAlmostEqual(rig.calls[3]["config"]["learning_rate"], rig.calls[0]["config"]["learning_rate"] / 2)
        self.assertIn("went wrong", [e for e in rig.store.journal(rig.id) if e["type"] == "error"][0]["message"])
        self.assertFalse(rig.calls[2]["dir"].exists())

    def test_diverging_with_nothing_left_to_try_ends_the_session_not_as_failed(self):
        rig = Rig(self, [5.0, 4.0], plateau={"checks": 9}, fail_train_on=3, worse={"limit": 2, "pivots": []})
        rig.fail_with = ValueError("Non-finite loss. Reduce the learning rate.")
        state = rig.run()
        self.assertEqual((rig.row()["status"], state["stop_reason"], state["best"]), ("finished", "pivots", 2))

    def test_diverging_before_any_checkpoint_halves_the_learning_rate_and_retries(self):
        rig = Rig(self, [5.0], stop={"target_error": 6}, plateau={"checks": 9}, fail_train_on=1)
        rig.fail_with = ValueError("Non-finite loss. Reduce the learning rate.")
        state = rig.run()
        self.assertEqual((rig.row()["status"], len(state["checkpoints"])), ("finished", 1))
        self.assertAlmostEqual(rig.calls[1]["config"]["learning_rate"], rig.calls[0]["config"]["learning_rate"] / 2)

    def test_other_training_errors_still_fail_the_session(self):
        rig = Rig(self, [5.0], plateau={"checks": 9}, fail_train_on=1)
        rig.fail_with = ValueError("The dataset is empty.")
        with self.assertRaises(ValueError):
            rig.run()
        self.assertIn("empty", rig.row()["error"])

    def test_heartbeat_runs_during_a_slow_setup_and_survives_any_error(self):
        rig = Rig(self, [3.0], stop={"target_error": 5}, plateau={"checks": 9})
        beats, real = [], PassiveStore.beat

        def build(root, data):
            time.sleep(0.15)
            beats.append(rig.row()["heartbeat"])
            return [dict(e) for e in EXAMPLES]

        calls = []

        def beat(self, session_id, pid):
            calls.append(1)
            if len(calls) == 2:  # the first beat is run()'s own, the second is the thread's
                raise OSError("not a SQLite error")
            real(self, session_id, pid)
        with patch.object(passive, "BEAT", 0.01), patch.object(passive, "POLL", 0.01), patch.object(PassiveStore, "beat", beat):
            run_session(rig.root, rig.id, train=rig.train, evaluate=rig.evaluate, build=build, clock=lambda: rig.time[0])
        self.assertIsNotNone(beats[0])
        self.assertGreater(len(calls), 2)  # still beating after the error

    def test_pruning_takes_discarded_branches_first_and_removes_their_run_folders(self):
        rig = Rig(self, [5.0, 4.0, 2.0, 3.0, 3.5, 1.5], limits={"keep_checkpoints": 3}, stop={"target_error": 1.6}, plateau={"checks": 9, "max_steps": 50})
        state = rig.run()
        kept = {c["n"]: c["kept"] for c in state["checkpoints"]}
        self.assertEqual(kept, {1: False, 2: False, 3: True, 4: False, 5: True, 6: True})
        for c, call in zip(state["checkpoints"], rig.calls):
            self.assertEqual(call["dir"].exists(), c["kept"])
        self.assertTrue(Path(state["checkpoints"][state["best"] - 1]["path"]).exists())

    def test_win_rate_without_a_teacher_is_journalled_and_the_session_still_ends(self):
        rig = Rig(self, [3.0, 2.0, 1.0], stop={"win_rate": 0.6, "target_error": 1.5}, plateau={"checks": 9})
        state = rig.run()
        self.assertEqual(state["stop_reason"], "target")
        notes = [e["message"] for e in rig.store.journal(rig.id) if e["type"] == "error"]
        self.assertEqual(len(notes), 1)
        self.assertIn("no teacher", notes[0])

    def test_win_rate_with_a_teacher_is_not_flagged(self):
        rig = Rig(self, [3.0, 2.0], stop={"win_rate": 0.6, "target_error": 2.5}, plateau={"checks": 9}, teacher=lambda n, c: {"checkpoint": c, "lessons": []})
        rig.set_teacher()
        rig.run()
        self.assertNotIn("error", rig.types())

    def test_pause_and_resume(self):
        rig = Rig(self, [3.0, 2.0], stop={"target_error": 2.5}, plateau={"checks": 9})
        rig.on_train = lambda r, n: r.store.command(r.id, "pause") if n == 1 else None
        seen = {}

        def resume_when_paused():
            while rig.row()["status"] != "paused":
                pass
            seen["status"] = "paused"
            rig.store.command(rig.id, "resume")
        thread = threading.Thread(target=resume_when_paused, daemon=True)
        thread.start()
        state = rig.run()
        thread.join(5)
        self.assertEqual(seen["status"], "paused")
        self.assertEqual((rig.row()["status"], state["stop_reason"]), ("finished", "target"))
        self.assertEqual([t for t in rig.types() if t in ("pause", "resume")], ["pause", "resume"])

    def test_no_teacher_skips_the_step(self):
        rig = Rig(self, [3.0, 2.0], stop={"target_error": 2.5}, plateau={"checks": 9}, teacher=lambda n, c: {"checkpoint": c, "lessons": []})
        state = rig.run()
        self.assertEqual(rig.teacher_calls, [])
        self.assertEqual((rig.row()["status"], state["stop_reason"]), ("finished", "target"))
        self.assertIsNone(state["checkpoints"][0]["teacher"])

    def test_teacher_failure_never_kills_the_session(self):
        def broken(n, c):
            raise RuntimeError("Cannot reach Ollama")
        rig = Rig(self, [3.0, 2.0], stop={"target_error": 2.5}, plateau={"checks": 9}, teacher=broken)
        rig.set_teacher()
        state = rig.run()
        self.assertEqual(rig.row()["status"], "finished")
        self.assertEqual(len(state["checkpoints"]), 2)
        self.assertFalse(state["teacher"]["reachable"])
        skipped = [e for e in rig.store.journal(rig.id) if e["type"] == "teacher"]
        self.assertEqual(len(skipped), 2)
        self.assertIn("Ollama", skipped[0]["skipped"])

    def test_lessons_are_stored_pending_and_fed_forward(self):
        lesson = {"messages": [{"role": "user", "content": "say yes"}], "chosen": "yes, indeed", "rejected": "y"}
        rig = Rig(self, [3.0, 2.0], stop={"target_error": 2.5}, plateau={"checks": 9}, teacher=lambda n, c: {"checkpoint": c, "win_rate": 0.7, "lessons": [lesson] if n == 1 else []})
        rig.set_teacher()
        state = rig.run()
        rows = rig.store.lessons(rig.id)
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]["status"], rows[0]["prompt"], rows[0]["answer"], rows[0]["reply"], rows[0]["round"], rows[0]["checkpoint"]),
                         ("pending", "say yes", "yes, indeed", "y", 1, 1))
        taught = {"messages": [{"role": "user", "content": "say yes"}, {"role": "assistant", "content": "yes, indeed"}]}
        self.assertNotIn(taught, rig.calls[0]["examples"])
        self.assertIn(taught, rig.calls[1]["examples"])
        self.assertEqual(state["checkpoints"][0]["teacher"], {"polished": False, "win_rate": 0.7, "lessons": 1})
        self.assertIn("lesson", rig.types())

    def test_polish_replaces_only_if_not_worse(self):
        for polished_error, expect in ((1.5, True), (2.5, False)):
            with self.subTest(polished_error=polished_error):
                rig = Rig(self, [2.0, polished_error], stop={"target_error": 5}, teacher=lambda n, c: {"checkpoint": "polished.npz", "win_rate": 0.8, "lessons": []})
                rig.set_teacher()
                state = rig.run()
                entry = state["checkpoints"][0]
                self.assertEqual(entry["teacher"]["polished"], expect)
                self.assertEqual(entry["path"] == "polished.npz", expect)
                self.assertEqual(entry["error"], polished_error if expect else 2.0)

    def test_drop_lessons_pivot_leaves_out_later_lessons(self):
        lesson = {"messages": [{"role": "user", "content": "q"}], "chosen": "a", "rejected": "b"}
        rig = Rig(self, [1.0, 2.0, 3.0, 0.5], stop={"target_error": 0.6}, worse={"limit": 2, "pivots": ["drop_lessons"]}, plateau={"checks": 9},
                  teacher=lambda n, c: {"checkpoint": c, "lessons": [lesson]})
        rig.set_teacher()
        rig.run()
        status = {r["checkpoint"]: r["status"] for r in rig.store.lessons(rig.id)}
        self.assertEqual(status[1], "pending")
        self.assertEqual((status[2], status[3]), ("left_out", "left_out"))
        self.assertEqual(sum(1 for e in rig.calls[3]["examples"] if e.get("messages")), 1)  # only checkpoint 1's lesson remains

    def test_restart_carries_on(self):
        rig = Rig(self, [3.0, 2.0, 1.0, 0.5], stop={"target_error": 0.6}, plateau={"checks": 9}, fail_train_on=2)
        with self.assertRaises(RuntimeError):
            rig.run()
        row = rig.row()
        self.assertEqual(row["status"], "running")  # the watchdog decides about failure
        self.assertIn("boom", row["error"])
        self.assertEqual([c["n"] for c in row["state"]["checkpoints"]], [1])
        self.assertIn("error", rig.types())
        first_path = row["state"]["checkpoints"][0]["path"]
        state = rig.run()
        self.assertEqual((rig.row()["status"], state["stop_reason"], state["best"]), ("finished", "target", 4))
        self.assertEqual(state["checkpoints"][0]["path"], first_path)
        self.assertTrue(Path(first_path).exists())
        self.assertIn("restart", rig.types())
        self.assertEqual(rig.types().count("start"), 1)
        self.assertEqual(rig.row()["error"], "")

    def test_finished_session_is_left_alone(self):
        rig = Rig(self, [1.0], stop={"target_error": 5})
        rig.run()
        calls = len(rig.calls)
        rig.run()
        self.assertEqual(len(rig.calls), calls)

    def test_signal_interrupt_leaves_session_running(self):
        rig = Rig(self, [3.0, 2.0], plateau={"checks": 9})
        interrupt = threading.Event()
        rig.on_train = lambda r, n: interrupt.set()
        run_session(rig.root, rig.id, train=rig.train, evaluate=rig.evaluate, build=lambda root, data: list(EXAMPLES), clock=lambda: rig.time[0], interrupt=interrupt)
        self.assertEqual(rig.row()["status"], "running")
        self.assertEqual(rig.emitted, 1)
        self.assertEqual(rig.row()["state"]["checkpoints"], [])

    def test_keep_checkpoints_never_deletes_best_or_current(self):
        rig = Rig(self, [1.0, 5.0, 6.0, 7.0, 8.0], limits={"keep_checkpoints": 2}, stop={"patience_rounds": 4}, worse={"limit": 9}, plateau={"checks": 9})
        state = rig.run()
        kept = {c["n"]: c["kept"] for c in state["checkpoints"]}
        self.assertEqual(kept, {1: True, 2: False, 3: False, 4: False, 5: True})
        for c in state["checkpoints"]:
            self.assertEqual(Path(c["path"]).exists(), c["kept"])

    def test_pruning_stays_inside_the_session_folder(self):
        rig = Rig(self, [], limits={"keep_checkpoints": 1})
        outside = Path(tempfile.mkdtemp()) / "keep.npz"
        outside.write_bytes(b"x")
        self.addCleanup(outside.unlink)
        trainer = passive._Trainer(rig.root, rig.id, None, None, None, None, None, lambda: 0.0, threading.Event())
        trainer.state = {"best": 2, "current": 2, "checkpoints": [{"n": 1, "path": str(outside), "kept": True}, {"n": 2, "path": "x/best.npz", "kept": True}]}
        trainer.prune()
        self.assertTrue(outside.exists())
        self.assertFalse(trainer.state["checkpoints"][0]["kept"])

    def test_a_paused_session_does_not_spin(self):
        rig = Rig(self, [])
        trainer = passive._Trainer(rig.root, rig.id, None, None, None, None, None, lambda: 0.0, threading.Event())
        trainer.pause.set()
        naps = []
        with patch.object(passive.time, "sleep", lambda seconds: (naps.append(seconds), trainer.pause.clear())):
            trainer.wait_if_paused()
        self.assertGreaterEqual(naps[0], 1.0)

    def test_memory_over_limit_reruns_with_smaller_batch(self):
        rig = Rig(self, [2.0], stop={"target_error": 5}, limits={"memory_fraction": 0.5}, plateau={"checks": 9})
        calls = []

        def memory():
            calls.append(1)
            return 900 if len(calls) == 1 else 100
        with patch.object(passive, "process_memory", memory), patch.object(passive, "total_memory", lambda: 1000):
            state = rig.run()
        self.assertEqual([c["config"]["batch_size"] for c in rig.calls], [4, 2])
        self.assertEqual(len(state["checkpoints"]), 1)
        self.assertEqual(len(rig.errors), 0)
        memory_events = [e for e in rig.store.journal(rig.id) if e["type"] == "memory"]
        self.assertEqual((len(memory_events), memory_events[0]["batch_size"]), (1, 2))
        self.assertEqual(state["memory"]["limit"], 500)

    def test_memory_probes_return_sane_numbers(self):
        self.assertGreater(passive.process_memory(), 1 << 20)
        self.assertGreater(passive.total_memory(), passive.process_memory())

    def test_a_failure_in_setup_is_recorded_with_its_reason(self):
        rig = Rig(self, [])
        with self.assertRaises(ValueError):
            run_session(rig.root, rig.id, train=rig.train, evaluate=rig.evaluate, build=lambda root, data: [dict(EXAMPLES[0])], clock=lambda: rig.time[0])
        row = rig.row()
        self.assertEqual(row["status"], "queued")  # still active: the watchdog restarts it and gives up after repeated failures
        self.assertIn("at least two examples", row["error"])
        self.assertEqual(rig.types(), ["error"])

    def test_the_per_round_copy_of_the_training_set_is_not_kept(self):
        rig = Rig(self, [3.0, 2.0], stop={"target_error": 2.5}, plateau={"checks": 9})
        rig.on_train = lambda r, n: (r.calls[-1]["dir"] / "dataset.jsonl").write_text("x")
        rig.run()
        self.assertEqual(len(rig.calls), 2)
        self.assertTrue(all((c["dir"] / "best.npz").exists() and not (c["dir"] / "dataset.jsonl").exists() for c in rig.calls))


class SmokeTests(unittest.TestCase):
    def test_real_training_for_a_few_steps(self):
        tiny = dict(device="cpu", width=32, layers=1, heads=2, context=32, expansion=2, dropout=0, batch_size=4, learning_rate=.004, warmup=0,
                    validation=0.2, eval_every=5, objective="assistant")
        rig = Rig(self, [], stop={"target_error": 100}, plateau={"max_steps": 10, "checks": 2}, config=tiny)
        state = run_session(rig.root, rig.id, build=lambda root, data: [dict(e) for e in EXAMPLES])
        self.assertEqual((rig.row()["status"], len(state["checkpoints"])), ("finished", 1))
        self.assertGreater(state["checkpoints"][0]["error"], 0)
        self.assertTrue(Path(state["checkpoints"][0]["path"]).exists())


class TrainerKeepAwakeTests(unittest.TestCase):
    def test_main_asks_to_stay_awake_only_when_the_plan_says_so(self):
        for awake in (True, False):
            with tempfile.TemporaryDirectory() as tmp:
                store = PassiveStore(tmp)
                sid = store.create(plan_with(limits={"keep_awake": awake}))
                with patch("studio_passive_watch.keep_awake") as keep, patch("signal.signal"), patch.object(passive, "run_session") as run:
                    passive.main(["run", sid, "--root", tmp])
                self.assertEqual(keep.called, awake)
                run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
