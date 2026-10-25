"""Issue #695: voice mode's engines (studio_voice), their services, and GET /voice, POST /voice/transcribe, POST /voice/speak.

The real models are never loaded: engines are fakes, or the heavy imports are absent and the status says so."""
import io
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

import studio_voice
from studio_api.app import create_app
from studio_api.hosts import ANY
from studio_core import Workspace
from studio_ia.services import FakeServices, WorkspaceServices
from studio_queue import Job


def wav(rate=16000, channels=1, width=2, frames=1600):
    out = io.BytesIO()
    with wave.open(out, "wb") as f:
        f.setnchannels(channels)
        f.setsampwidth(width)
        f.setframerate(rate)
        f.writeframes(b"\x01" * width * channels * frames)
    return out.getvalue()


class FakeEngine:
    def __init__(self, name, missing=""):
        self.name, self.label, self.model_name, self._missing = name, name, name, missing
        self.device, self.ran = None, []

    def missing(self):
        return self._missing

    def load(self, device):
        self.device = device

    def move(self, device):
        self.device = device if self.name == "stt" else None  # text to speech unloads instead of moving

    def unload(self):
        self.device = None

    def run(self, arg):
        self.ran.append(arg)
        return "hello" if self.name == "stt" else b"RIFF"


def voice(cuda=True, stt="", tts=""):
    return studio_voice.Voice(FakeEngine("stt", stt), FakeEngine("tts", tts), cuda=lambda: cuda)


class Wav(unittest.TestCase):
    def test_a_good_wav_gives_its_frames(self):
        self.assertEqual(len(studio_voice.wav_samples(wav(frames=800))), 1600)

    def test_bad_audio_is_a_sentence(self):
        for bad, words in ((b"", "No audio"), (b"not a wav", "not a WAV"), (wav(rate=44100), "16 kHz"), (wav(channels=2), "16 kHz"),
                           (wav(width=1), "16 kHz"), (wav(frames=0), "empty"), (wav(frames=16000 * 121), "longer")):
            with self.subTest(words), self.assertRaisesRegex(ValueError, words):
                studio_voice.wav_samples(bad)

    def test_speech_wav_round_trips(self):
        data = studio_voice.wav_bytes_from([0.0, 0.5, -0.5], 24000)
        with wave.open(io.BytesIO(data)) as f:
            self.assertEqual((f.getnchannels(), f.getsampwidth(), f.getframerate(), f.getnframes()), (1, 2, 24000, 3))


class Engines(unittest.TestCase):
    def test_status_before_and_after_use(self):
        v = voice()
        self.assertEqual([(r["engine"], r["installed"], r["loaded"], r["device"]) for r in v.status()],
                         [("stt", True, False, None), ("tts", True, False, None)])
        self.assertEqual(v.transcribe(wav()), "hello")
        self.assertEqual(v.speak(" Hi there "), b"RIFF")
        self.assertEqual(v.tts.ran, ["Hi there"])
        self.assertEqual([(r["loaded"], r["device"]) for r in v.status()], [(True, "cuda"), (True, "cuda")])

    def test_models_load_once_and_stay(self):
        v = voice()
        with mock.patch.object(v.stt, "load", wraps=v.stt.load) as load:
            v.transcribe(wav())
            v.transcribe(wav())
        load.assert_called_once_with("cuda")

    def test_empty_or_oversized_text_and_bad_wav_never_reach_the_engine(self):
        v = voice()
        for text in ("", "   ", None):
            with self.assertRaisesRegex(ValueError, "no text"):
                v.speak(text)
        with self.assertRaisesRegex(ValueError, "longer"):
            v.speak("x" * (studio_voice.MAX_TTS_CHARS + 1))
        with self.assertRaises(ValueError):
            v.transcribe(b"junk")
        self.assertEqual((v.stt.ran, v.tts.ran), ([], []))

    def test_an_engine_that_is_not_installed_says_what_to_run(self):
        v = voice(stt="Speech to text needs NeMo. Run `make voice-setup`.", tts="Text to speech needs PyTorch. Run `make voice-setup`.")
        self.assertEqual([r["installed"] for r in v.status()], [False, False])
        self.assertIn("make voice-setup", v.status()[0]["missing"])
        for call in (lambda: v.transcribe(wav()), lambda: v.speak("hi")):
            with self.assertRaisesRegex(RuntimeError, "make voice-setup"):
                call()

    def test_real_engines_without_their_packages_name_the_setup_command(self):
        with mock.patch("studio_voice._have", return_value=False):
            rows = studio_voice.Voice().status()
        self.assertTrue(all(not r["installed"] and "make voice-setup" in r["missing"] for r in rows))

    def test_a_job_moves_speech_to_text_to_the_cpu_and_unloads_the_voice(self):
        v = voice()
        v.transcribe(wav())
        v.speak("hi")
        v.job_started()
        self.assertEqual((v.stt.device, v.tts.device), ("cpu", None))
        self.assertEqual(v.transcribe(wav()), "hello")  # still works, on the CPU
        with self.assertRaisesRegex(RuntimeError, "training job"):
            v.speak("hi")
        v.job_ended()
        self.assertEqual(v.stt.device, "cuda")
        self.assertEqual(v.speak("hi"), b"RIFF")  # loads again after the job
        self.assertEqual(v.tts.device, "cuda")

    def test_a_model_first_used_during_a_job_loads_on_the_cpu(self):
        v = voice()
        v.job_started()
        v.transcribe(wav())
        self.assertEqual(v.stt.device, "cpu")

    def test_no_gpu_means_no_voice_but_transcription_runs_on_the_cpu(self):
        v = voice(cuda=False)
        self.assertEqual(v.transcribe(wav()), "hello")
        self.assertEqual(v.stt.device, "cpu")
        with self.assertRaisesRegex(RuntimeError, "NVIDIA GPU"):
            v.speak("hi")


class QueueJobStarting(unittest.TestCase):
    def test_the_queue_hooks_move_the_services_voice_off_the_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            svc = WorkspaceServices(Workspace(Path(tmp) / "ws"))
            self.addCleanup(svc.close)
            svc._voice = v = voice()
            svc.transcribe(wav())
            svc.speak("hi")
            self.assertEqual((v.stt.device, v.tts.device), ("cuda", "cuda"))
            no_gpu = {"available": False, "kind": "", "name": "", "reason": "", "memory_bytes": 0}
            with mock.patch("studio_backend.gpu_status", return_value=no_gpu):
                svc.chat_job_started(Job("pretrain"))
                self.assertEqual((v.stt.device, v.tts.device), ("cpu", None))
                svc.chat_job_ended(Job("pretrain"))
            self.assertFalse(svc.voice_status()[1]["loaded"])


class Api(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(create_app(FakeServices(), open=True, hosts=ANY))

    def test_status(self):
        body = self.client.get("/voice").json()
        self.assertEqual([e["engine"] for e in body["engines"]], ["stt", "tts"])
        self.assertEqual(set(body["engines"][0]), {"engine", "label", "model", "installed", "loaded", "device", "missing"})

    def test_transcribe(self):
        r = self.client.post("/voice/transcribe", files={"audio": ("a.wav", wav(), "audio/wav")})
        self.assertEqual((r.status_code, r.json()), (200, {"text": "This is a test transcript."}))

    def test_transcribe_refuses_empty_and_bad_audio(self):
        for data, words in ((b"", "No audio"), (b"junk", "not a WAV"), (wav(rate=8000), "16 kHz")):
            r = self.client.post("/voice/transcribe", files={"audio": ("a.wav", data, "audio/wav")})
            self.assertEqual(r.status_code, 400)
            self.assertRegex(r.json()["detail"], words)

    def test_speak_gives_wav(self):
        r = self.client.post("/voice/speak", json={"text": "Hello"})
        self.assertEqual((r.status_code, r.headers["content-type"]), (200, "audio/wav"))
        with wave.open(io.BytesIO(r.content)) as f:
            self.assertGreater(f.getnframes(), 0)

    def test_speak_refuses_empty_text(self):
        r = self.client.post("/voice/speak", json={"text": "  "})
        self.assertEqual((r.status_code, r.json()["detail"]), (400, "There is no text to speak."))

    def test_an_engine_that_cannot_run_is_a_conflict_with_a_sentence(self):
        services = FakeServices()
        services.speak = mock.Mock(side_effect=RuntimeError("Text to speech needs NeMo. Run `make voice-setup`."))
        r = TestClient(create_app(services, open=True, hosts=ANY)).post("/voice/speak", json={"text": "hi"})
        self.assertEqual(r.status_code, 409)
        self.assertIn("make voice-setup", r.json()["detail"])


if __name__ == "__main__":
    unittest.main()
