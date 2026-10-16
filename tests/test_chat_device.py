"""Issue #172: chat runs on the CUDA GPU while the queue is idle and moves to the CPU when a job starts.

The CUDA path is only exercised with a faked gpu_status and a fake model here; moving a real model to a real CUDA GPU has not been run."""
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from studio_core import Workspace
from studio_ia.services import WorkspaceServices
from studio_model import LocalModel, ScratchTransformer, validate_config
from studio_modules.tokenizer.byte import Tokenizer
from studio_queue import Job, Plan, Runner
from tests.test_queue import wait

CUDA = {"available": True, "kind": "cuda", "name": "fake", "reason": "", "memory_bytes": 8 * 2 ** 30}
MPS = dict(CUDA, kind="mps")
NONE = {"available": False, "kind": "", "name": "", "reason": "no GPU", "memory_bytes": 0}


class FakeModel:
    """Stands in for LocalModel: knows its device, records every move, and notes the device each reply starts and ends on."""

    def __init__(self, path, device, log):
        self.device, self.log = device, log
        self.model = SimpleNamespace(p={"w": torch.zeros(256)})
        self.log.append(("load", device))
        self.hold = None  # a threading.Event a reply waits on, when a test wants it to stay in progress
        self.started = threading.Event()

    def to(self, device):
        if device == "cuda" and getattr(self, "full", False):
            raise torch.cuda.OutOfMemoryError("CUDA out of memory.")
        self.device = device
        self.log.append(("move", device))
        return self

    def generate(self, messages, on_text=None, stop=None):
        start = self.device
        self.started.set()
        if self.hold is not None:
            self.hold.wait(10)
        self.log.append(("reply", start, self.device))
        return "ok"


class ChatDevice(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.svc = WorkspaceServices(Workspace(Path(self.tmp.name) / "ws"))
        self.addCleanup(self.svc.close)
        self.log, self.made = [], {}

    def fake(self, path, device):
        self.made[Path(path).name] = model = FakeModel(path, device, self.log)
        return model

    def chat(self, name="m.npz"):
        with mock.patch("studio_model.LocalModel", side_effect=self.fake), mock.patch.object(self.svc, "_checkpoint", side_effect=lambda v: v):
            return self.svc.chat(name, [], None, None)

    def devices(self):
        """The device chat's model was on at each step, in order."""
        return [entry[-1] for entry in self.log if entry[0] in ("load", "move")]

    def test_cuda_while_idle_cpu_when_a_job_starts_cuda_again_after_it_ends(self):
        with mock.patch("studio_backend.gpu_status", return_value=CUDA):
            self.chat()
            self.assertEqual(self.devices(), ["cuda"])
            self.svc.chat_job_started(Job("pretrain"))
            self.assertEqual(self.devices(), ["cuda", "cpu"])
            self.chat()  # a reply during the job comes from the CPU
            self.svc.chat_job_ended(Job("pretrain"))
        self.assertEqual(self.devices(), ["cuda", "cpu", "cuda"])
        self.assertEqual([e[1:] for e in self.log if e[0] == "reply"], [("cuda", "cuda"), ("cpu", "cpu")])

    def test_a_model_loaded_during_a_job_is_on_the_cpu_and_goes_to_cuda_when_it_ends(self):
        with mock.patch("studio_backend.gpu_status", return_value=CUDA):
            self.svc.chat_job_started(Job("pretrain"))
            self.chat()
            self.assertEqual(self.devices(), ["cpu"])
            self.svc.chat_job_ended(Job("pretrain"))
        self.assertEqual(self.devices(), ["cpu", "cuda"])

    def test_on_mps_the_device_is_cpu_throughout(self):
        with mock.patch("studio_backend.gpu_status", return_value=MPS):
            self.chat()
            self.svc.chat_job_started(Job("pretrain"))
            self.chat()
            self.svc.chat_job_ended(Job("pretrain"))
            self.chat()
        self.assertEqual(set(self.devices()), {"cpu"})

    def test_on_a_machine_with_no_gpu_the_device_is_cpu_throughout(self):
        with mock.patch("studio_backend.gpu_status", return_value=NONE):
            self.chat()
            self.svc.chat_job_started(Job("pretrain"))
            self.svc.chat_job_ended(Job("pretrain"))
        self.assertEqual(set(self.devices()), {"cpu"})

    def test_the_gpu_cache_is_emptied_after_the_move_to_the_cpu(self):
        with mock.patch("studio_backend.gpu_status", return_value=CUDA), mock.patch("torch.cuda.is_available", return_value=True), \
                mock.patch("torch.cuda.empty_cache", side_effect=lambda: self.log.append(("empty_cache",))):
            self.chat()
            self.svc.chat_job_started(Job("pretrain"))
            self.svc.chat_job_ended(Job("pretrain"))
        self.assertEqual([e for e in self.log if e[0] in ("move", "empty_cache")], [("move", "cpu"), ("empty_cache",), ("move", "cuda")])

    def test_the_ledger_follows_the_model_between_the_gpu_and_the_ram(self):
        ledger, size = self.svc.ledger, 256 * 4
        with mock.patch("studio_backend.gpu_status", return_value=CUDA):
            self.chat()
            self.assertEqual((ledger.total("cuda"), ledger.total("cpu")), (size, 0))
            self.svc.chat_job_started(Job("pretrain"))
            self.assertEqual((ledger.total("cuda"), ledger.total("cpu")), (0, size))
            self.svc.chat_job_ended(Job("pretrain"))
            self.assertEqual((ledger.total("cuda"), ledger.total("cpu")), (size, 0))

    def test_a_reply_in_progress_finishes_on_the_device_it_started_on(self):
        with mock.patch("studio_backend.gpu_status", return_value=CUDA):
            self.chat()
            model = self.made["m.npz"]
            model.hold = threading.Event()
            reply = threading.Thread(target=self.chat)
            reply.start()
            self.assertTrue(model.started.wait(10))
            mover = threading.Thread(target=self.svc.chat_job_started, args=(Job("pretrain"),))
            mover.start()
            mover.join(.3)
            self.assertTrue(mover.is_alive())  # waiting for the reply, not moving the model under it
            self.assertEqual(model.device, "cuda")
            model.hold.set()
            reply.join(10)
            mover.join(10)
        self.assertEqual(model.device, "cpu")
        self.assertIn(("reply", "cuda", "cuda"), self.log)

    def test_a_model_the_gpu_has_no_room_for_stays_on_the_cpu_and_the_ledger_says_so(self):
        with mock.patch("studio_backend.gpu_status", return_value=CUDA):
            self.chat()
            self.svc.chat_job_started(Job("pretrain"))
            self.made["m.npz"].full = True
            with self.assertLogs("groundwork.chat", "WARNING"):
                self.svc.chat_job_ended(Job("pretrain"))
            self.assertEqual(self.made["m.npz"].device, "cpu")
            self.assertEqual((self.svc.ledger.total("cuda"), self.svc.ledger.total("cpu")), (0, 256 * 4))
            self.chat()  # still answers

    def test_the_runner_calls_the_hooks_around_a_job(self):
        self.assertEqual(self.svc.runner.on_start, self.svc.chat_job_started)
        self.assertEqual(self.svc.runner.on_end, self.svc.chat_job_ended)


def quick(job, context, emit, stop, pause):
    pass


class RunnerHooks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.calls = []

    def runner(self, **hooks):
        r = Runner(Path(self.tmp.name) / "data" / "queue.json", {"quicktest": quick}, tick=.01, **hooks)
        r.set_plan(Plan(hours="any", idle_only=False, pause_on_battery=False))
        self.addCleanup(r.shutdown)
        return r

    def test_a_job_is_announced_before_it_starts_and_after_it_ends(self):
        r = self.runner(on_start=lambda job: self.calls.append(("start", job.status)), on_end=lambda job: self.calls.append(("end", job.status)))
        job = r.enqueue(Job("quicktest"))
        r.start()
        self.assertTrue(wait(lambda: job.status == "done" and len(self.calls) == 2))
        self.assertEqual(self.calls, [("start", "waiting"), ("end", "done")])

    def test_a_hook_that_fails_does_not_stop_the_job(self):
        def boom(job):
            raise RuntimeError("no GPU")
        r = self.runner(on_start=boom, on_end=boom)
        job = r.enqueue(Job("quicktest"))
        with self.assertLogs("groundwork.queue", "ERROR"):
            r.start()
            self.assertTrue(wait(lambda: job.status == "done"))


def tiny(device="cpu"):
    config = validate_config({"architecture": {"schema": 1, "dims": {"d_model": 16, "n_layer": 2, "n_ctx": 32, "dropout": 0},
                                              "position": "sinusoidal", "block": {"mixer": {"id": "attention", "config": {"heads": 2}}}},
                              "device": device, "seed": 3})
    local = LocalModel.__new__(LocalModel)
    local.config, local.tokenizer = config, Tokenizer("byte")
    local.model = ScratchTransformer(config, local.tokenizer.vocab_size)
    local.rng, local.step = np.random.default_rng(1), 0
    return local


class ModelMove(unittest.TestCase):
    @unittest.skipUnless(torch.backends.mps.is_available(), "needs a second device; MPS stands in for CUDA")
    def test_a_model_moved_to_another_device_and_back_gives_the_same_logits_and_reply(self):
        local, ids = tiny(), np.array([[1, 5, 9, 12, 7]])
        indices = [layer.dims.layer for layer in local.model.layers]
        before = local.model.forward(ids).clone()
        reply = local.generate([{"role": "user", "content": "Hi"}], temperature=0, max_tokens=12)
        local.to("mps")
        model = local.model
        self.assertEqual((local.device, local.config["device"], model.device, model.dims.device), ("mps",) * 4)
        self.assertTrue(all(t.device.type == "mps" for t in [*model.p.values(), *model.g.values()]))
        self.assertTrue(all(layer.device == "mps" and layer.dims.device == "mps" for layer in model.layers))
        self.assertEqual(model.position.encoding.device.type, "mps")  # a tensor the layer keeps besides its weights
        self.assertTrue(torch.allclose(model.forward(ids).cpu(), before, atol=1e-4))
        self.assertEqual(local.generate([{"role": "user", "content": "Hi"}], temperature=0, max_tokens=12), reply)
        local.to("cpu")
        self.assertEqual(local.device, "cpu")
        self.assertTrue(torch.equal(model.forward(ids), before))
        self.assertEqual([layer.dims.layer for layer in model.layers], indices)  # each layer keeps the block index it was built with

    @unittest.skipIf(torch.cuda.is_available(), "this machine has CUDA")
    def test_a_move_to_a_device_that_is_not_there_changes_nothing(self):
        local = tiny()
        before = {key: value.clone() for key, value in local.model.p.items()}
        with mock.patch("studio_backend.gpu_status", return_value=NONE), self.assertRaises(RuntimeError):
            local.to("cuda")
        self.assertEqual((local.device, local.model.device, local.model.dims.device), ("cpu",) * 3)
        self.assertTrue(all(torch.equal(value, before[key]) for key, value in local.model.p.items()))

    def test_a_move_to_the_device_it_is_on_does_nothing(self):
        local = tiny()
        self.assertIs(local.model.to("cpu"), local.model)


if __name__ == "__main__":
    unittest.main()
