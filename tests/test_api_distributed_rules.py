"""A cluster run started with plateau, pivot and stages options: the real route over a real workspace stores them where the coordinator
reads them (the job's params and Settings), and refuses combinations that would do nothing with a sentence."""
import unittest

import studio_diloco
from studio_core import Workspace
from studio_ia.services import WorkspaceServices
from studio_spec import default_spec
from tests.test_api_distributed_teacher import TEXT, Base


class Reaches(Base):
    def setUp(self):
        super().setUp()
        self.workspace = Workspace(self.root / "ws")
        self.services = WorkspaceServices(self.workspace)
        self.addCleanup(self.services.close)
        self.services.save_version(default_spec(), "first")
        self.set = self.workspace.create_dataset("Pages")
        self.workspace.add(self.set, [{"text": t, "quality": "Good"} for t in TEXT])
        self.client = self.client_of(self.services)

    def start(self, **body):
        return self.client.post("/distributed/runs", json={"dataset_ids": [self.set], "rounds": 6, "inner_steps": 2, **body})

    def test_a_run_without_the_options_has_none(self):
        self.assertEqual(self.start().status_code, 201)
        params = self.queued[0].params
        self.assertNotIn("stages", params)
        settings = studio_diloco.Settings.from_params(params)
        self.assertEqual((settings.pivots, settings.plateau_checks), ((), 0))

    def test_the_options_reach_the_settings_and_the_job(self):
        response = self.start(pivots=["halve_lr", "next_dataset"], plateau_checks=3, plateau_min_improvement=.02, stages={"replay": .3})
        self.assertEqual(response.status_code, 201, response.text)
        params = self.queued[0].params
        self.assertEqual(params["stages"], {"replay": .3})
        settings = studio_diloco.Settings.from_params(params)
        self.assertEqual((settings.pivots, settings.plateau_checks, settings.plateau_min_improvement), (("halve_lr", "next_dataset"), 3, .02))

    def test_a_plateau_alone_is_a_stop_rule(self):
        self.assertEqual(self.start(plateau_checks=2).status_code, 201)
        self.assertEqual(studio_diloco.Settings.from_params(self.queued[0].params).plateau_checks, 2)

    def test_options_that_would_do_nothing_are_refused_in_a_sentence(self):
        for body, words in (({"pivots": ["next_dataset"], "plateau_checks": 2}, "needs stages"),
                            ({"stages": {"replay": .2}, "pivots": ["halve_lr"], "plateau_checks": 2}, "next_dataset"),
                            ({"pivots": ["halve_lr"]}, "plateau_checks"),
                            ({"plateau_checks": 1}, "at least 2"),
                            ({"pivots": ["drop_lessons"], "plateau_checks": 2}, "drop_lessons")):
            with self.subTest(body=body):
                response = self.start(**body)
                self.assertEqual(response.status_code, 400, response.text)
                self.assertIn(words, response.text)
        self.assertEqual(self.queued, [])


if __name__ == "__main__":
    unittest.main()
