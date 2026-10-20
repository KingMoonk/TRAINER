"""Issue #171: one memory budget shared by chat and the running job (studio_memory.Ledger, studio_queue.Runner, WorkspaceServices)."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from studio_core import Workspace
from studio_ia.services import CHAT_MODELS, WorkspaceServices
from studio_memory import Ledger, weight_bytes
from studio_queue import Job, Plan, Runner
from tests.test_queue import wait

GB = 2 ** 30
CONFIG = {"width": 32, "layers": 2, "heads": 2, "context": 64, "expansion": 2, "dropout": 0., "batch_size": 8,
          "steps": 5, "learning_rate": .003, "warmup": 1, "validation": 0, "eval_every": 5, "seed": 5, "device": "cpu"}


def note_share(job, context, emit, stop, pause):
    """Runs in the worker process: the share the job was handed, which is all the worker ever sees of the ledger."""
    with open(Path(context.root) / "share.txt", "w", encoding="utf-8") as f:
        f.write(str(job.params["memory_share"]))


class LedgerChecks(unittest.TestCase):
    def test_totals_follow_registrations_and_releases(self):
        ledger = Ledger()
        ledger.register("a", 3 * GB)
        ledger.register("b", 2 * GB)
        self.assertEqual((ledger.total("cpu"), ledger.largest("cpu")), (5 * GB, 3 * GB))
        ledger.release("a")
        ledger.release("never loaded")
        self.assertEqual((ledger.total("cpu"), ledger.largest("cpu")), (2 * GB, 2 * GB))

    def test_a_job_on_the_cpu_or_mps_counts_the_ram_a_job_on_cuda_counts_the_gpu(self):
        ledger = Ledger()
        ledger.register("chat", GB, "cpu")
        ledger.register("moved", 4 * GB, "cuda")
        self.assertEqual([ledger.total(d) for d in ("cpu", "mps", "cuda")], [GB, GB, 4 * GB])

    def test_weight_bytes_counts_every_parameter_in_its_dtype(self):
        model = SimpleNamespace(p={"w": torch.zeros(4, 8), "b": torch.zeros(8, dtype=torch.float16)})
        self.assertEqual(weight_bytes(model), 4 * 8 * 4 + 8 * 2)


class RunnerShare(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ledger, self.evicted = Ledger(), []
        self.params_when_asked = []
        ram = mock.patch("studio_memory.physical_memory", return_value=16 * GB)
        ram.start()
        self.addCleanup(ram.stop)

    def run_pretrain(self, memory):
        """Run one pretrain job through a real runner; returns the share its worker was handed."""
        self.job = Job("pretrain", "p", params={"config": dict(CONFIG)})
        runner = Runner(self.root / "data" / "queue.json", {"pretrain": note_share}, tick=.01, ledger=self.ledger, make_room=self.make_room)
        runner.set_plan(Plan(hours="any", idle_only=False, pause_on_battery=False, memory=memory))
        self.addCleanup(runner.shutdown)
        runner.enqueue(self.job)
        runner.start()
        self.assertTrue(wait(lambda: self.job.status == "done"), self.job.error)
        return float((self.root / "share.txt").read_text(encoding="utf-8"))

    def make_room(self):
        """Stands in for chat: evicts its oldest model, and notes whether the job's params had been set yet."""
        self.params_when_asked.append("memory_share" in self.job.params)
        oldest = next(iter(self.models))
        self.models.remove(oldest)
        self.ledger.release(oldest)
        self.evicted.append(oldest)
        return True

    models = ()

    def test_two_chat_models_make_the_jobs_share_smaller_by_their_bytes(self):
        self.assertEqual(self.run_pretrain("most"), .75)
        self.ledger.register("a", GB)
        self.ledger.register("b", GB)
        self.assertEqual(self.run_pretrain("most"), .75 - 2 * GB / (16 * GB))
        self.assertEqual(self.evicted, [])  # 12 GB - 2 GB is room for another model

    def test_a_job_with_no_room_makes_chat_evict_its_oldest_model_before_its_params_are_set(self):
        self.models = ["old", "newer"]
        self.ledger.register("old", 5 * GB)
        self.ledger.register("newer", 4 * GB)
        share = self.run_pretrain("half")  # 8 GB for the job and chat holds 9 GB
        self.assertEqual(self.evicted, ["old"])  # then 8 - 4 GB is room for one 4 GB model: it stops there
        self.assertEqual(self.params_when_asked, [False])
        self.assertEqual(share, .5 - 4 * GB / (16 * GB))

    def test_a_job_with_less_than_one_models_bytes_left_evicts_until_it_has_that_much(self):
        self.models = ["a", "b", "c"]
        for key in self.models:
            self.ledger.register(key, 3 * GB)
        share = self.run_pretrain("half")  # 8 GB, chat holds 9: evict a (6 left, 2 spare < 3), evict b (3 left, 5 spare)
        self.assertEqual(self.evicted, ["a", "b"])
        self.assertEqual(share, .5 - 3 * GB / (16 * GB))

    def test_a_job_that_is_not_a_pretrain_keeps_the_plans_share(self):
        self.ledger.register("a", 9 * GB)
        runner = Runner(self.root / "q.json", {}, ledger=self.ledger, make_room=self.make_room)
        self.assertEqual(runner._share(Job("quicktest")), .75)
        self.assertEqual(self.evicted, [])


class ChatLedger(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.svc = WorkspaceServices(Workspace(Path(self.tmp.name) / "ws"))
        self.addCleanup(self.svc.close)

    def chat(self, names):
        def local(path, device, data=None):
            return SimpleNamespace(model=SimpleNamespace(p={"w": torch.zeros(int(Path(path).stem[1:]) * 256)}),
                                   generate=lambda messages, on_text=None, stop=None, ablate=None: "ok")
        with mock.patch("studio_model.LocalModel", side_effect=local), mock.patch("studio_model.load_checkpoint", return_value={}), mock.patch.object(self.svc, "_checkpoint", side_effect=lambda v: v),                 mock.patch.object(self.svc, "_chat_device", return_value="cpu"):  # a machine with a CUDA GPU would register under "cuda"
            for name in names:
                self.svc.chat(name, [], None, None)

    def test_the_runner_is_wired_to_the_ledger_and_to_evicting(self):
        self.assertIs(self.svc.runner.ledger, self.svc.ledger)
        self.assertEqual(self.svc.runner.make_room, self.svc.evict_oldest_chat)

    def test_a_loaded_model_is_registered_and_the_lru_releases_what_it_evicts(self):
        names = [f"m{n}.npz" for n in range(1, CHAT_MODELS + 2)]
        self.chat(names)
        self.assertEqual(list(self.svc._models), names[1:])
        self.assertEqual(self.svc.ledger.total(), sum(n * 256 * 4 for n in range(2, CHAT_MODELS + 2)))

    def test_using_a_model_again_does_not_register_it_twice(self):
        self.chat(["m1.npz", "m1.npz"])
        self.assertEqual(self.svc.ledger.total(), 256 * 4)

    def test_evicting_for_the_queue_drops_the_oldest_and_says_whether_it_did(self):
        self.assertFalse(self.svc.evict_oldest_chat())
        self.chat(["m1.npz", "m2.npz"])
        self.assertTrue(self.svc.evict_oldest_chat())
        self.assertEqual(list(self.svc._models), ["m2.npz"])
        self.assertEqual(self.svc.ledger.total(), 2 * 256 * 4)


if __name__ == "__main__":
    unittest.main()
