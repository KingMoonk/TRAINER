"""A passive session's best checkpoint becomes a trained version: on request and, once a session has ended, by itself."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import studio_eval
import studio_passive_watch as watch
from studio_ia import services as services_module
from studio_ia.services import WorkspaceServices
from studio_model import train
from studio_passive import PassiveStore, merge_plan
from studio_versions import VersionStore
from tests.test_publish import EXAMPLES, TINY


class Saved(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.passive = PassiveStore(self.root)
        self.id = self.passive.create(merge_plan({}))
        self.scored = []
        services = object.__new__(WorkspaceServices)
        services.root, services.store = self.root, VersionStore(self.root)
        services._held_scores = lambda checkpoint: self.scored.append(checkpoint) or np.array([[3., 10.]])
        services.emit = lambda *args: None
        self.services = services

    def session(self, status="finished", rounds=3, best=2):
        """A session with `rounds` checkpoints, the second of them the best, as a trainer leaves it; the checkpoints are real files."""
        entries = []
        for n in range(1, rounds + 1):
            out = self.root / "runs" / self.id / f"round{n:03d}_x"
            train(dict(TINY, steps=4, eval_every=4), EXAMPLES, out, telemetry_every=0)
            entries.append({"n": n, "round": n, "path": str(out / "best.npz"), "error": 2.0 + abs(n - best) / 10, "kept": True, "discarded": False})
        self.passive.update(self.id, status=status, state={"checkpoints": entries, "best": best if rounds else None})
        return entries

    def test_the_best_checkpoint_becomes_one_trained_version_with_its_provenance(self):
        entries = self.session()
        saved = self.services.save_passive(self.id)
        self.assertEqual(saved, {"version": 1, "created": True})
        row = self.services.store.get(1)
        self.assertEqual((row["status"], row["best_checkpoint"], row["runs"]), ("trained", entries[1]["path"], [str(Path(entries[1]["path"]).parent)]))
        self.assertEqual((row["passive"]["session"], row["passive"]["rounds"], row["passive"]["round"], row["passive"]["error"]), (self.id, 3, 2, 2.0))
        for part in (self.id, "best of 3 rounds", "round 2", "test error 2"):
            self.assertIn(part, row["change"])
        self.assertEqual((row["scorecard"]["bpc"], row["scorecard"]["bpc_method"]), (studio_eval.corpus_bpc(np.array([[3., 10.]])), studio_eval.SCORE_METHOD))
        self.assertEqual((row["train_device"], row["score_device"]), ("cpu", studio_eval.SCORE_DEVICE))
        self.assertEqual(self.scored, [entries[1]["path"]])
        self.assertEqual(row["spec"]["dims"]["d_model"], TINY["width"])

    def test_saving_the_same_best_again_changes_nothing(self):
        self.session()
        self.services.save_passive(self.id)
        again = self.services.save_passive(self.id)
        self.assertEqual(again, {"version": 1, "created": False})
        self.assertEqual(len(self.services.store.versions()), 1)
        self.assertEqual(len(self.scored), 1)

    def test_a_better_best_later_is_a_new_version(self):
        entries = self.session()
        self.services.save_passive(self.id)
        state = self.passive.get(self.id)["state"]
        state["best"] = 3
        self.passive.update(self.id, state=state)
        self.assertEqual(self.services.save_passive(self.id), {"version": 2, "created": True})
        self.assertEqual(self.services.store.get(2)["best_checkpoint"], entries[2]["path"])

    def test_nothing_to_save_before_a_checkpoint_or_for_an_unknown_session(self):
        self.passive.update(self.id, status="failed", state={"checkpoints": [], "best": None})
        with self.assertRaisesRegex(RuntimeError, "no checkpoint yet"):
            self.services.save_passive(self.id)
        self.assertIsNone(self.services.save_passive("nope"))
        self.assertEqual(self.services.store.versions(), [])

    def test_a_checkpoint_that_was_cleaned_away_is_said_plainly(self):
        entries = self.session(rounds=1, best=1)
        Path(entries[0]["path"]).unlink()
        with self.assertRaisesRegex(RuntimeError, "no longer on disk"):
            self.services.save_passive(self.id)

    def test_a_never_used_workspace_has_nothing(self):
        with tempfile.TemporaryDirectory() as other:
            services = object.__new__(WorkspaceServices)
            services.root = Path(other)
            self.assertIsNone(services.save_passive("p1"))
            self.assertFalse((Path(other) / "data" / "passive.sqlite").exists())

    def view(self):
        with patch.object(watch, "installed", return_value=False):
            return self.services.passive()["sessions"]

    def test_an_ended_session_is_saved_by_itself_once_and_the_list_says_so(self):
        self.session(status="stopped")
        ran = []
        with patch.object(services_module, "start_thread", lambda target, *args: ran.append(target) or target(*args)):
            first = self.view()
            second = self.view()
        self.assertEqual(len(ran), 1)  # tried once; the second look finds it saved
        self.assertEqual(len(self.services.store.versions()), 1)
        self.assertEqual((first[0]["checkpoints"], first[0]["version"]), (3, None))  # the first look starts the save
        self.assertEqual(second[0]["version"], 1)

    def test_an_active_session_is_left_for_the_button(self):
        self.session(status="running")
        with patch.object(services_module, "start_thread", lambda target, *args: self.fail("a running session is not saved by itself")):
            rows = self.view()
        self.assertEqual((rows[0]["checkpoints"], rows[0]["version"]), (3, None))

    def test_a_failed_automatic_save_is_logged_and_not_retried_every_look(self):
        self.session(status="finished")
        self.services._held_scores = lambda checkpoint: (_ for _ in ()).throw(OSError("disk"))
        calls = []
        with patch.object(services_module, "start_thread", lambda target, *args: calls.append(1) or target(*args)), self.assertLogs(services_module._log, "WARNING"):
            self.view()
        self.view()
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.services.store.versions(), [])  # no unscored version was left behind


if __name__ == "__main__":
    unittest.main()
