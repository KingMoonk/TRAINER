"""Voice mode's two local engines: speech to text (NVIDIA Parakeet) and text to speech (Breeze TTS 2).

Both load the first time they are used and stay loaded. Neither is a dependency of the app: `make voice-setup` installs them
(requirements-voice.txt), clones the Breeze inference code and fetches both models' weights; the app never installs anything.
Every heavy import is inside a method, so this module imports with nothing installed and `status()` says what is missing.

Sharing the GPU with the queue: a job about to start calls `job_started()`. Speech to text (0.6B parameters) moves to the CPU and
keeps working, slower; text to speech (about 7.7 GiB of VRAM, bfloat16) is unloaded, because it is far too slow on a CPU. While a job
runs, `speak` fails with a sentence saying so. After the job, `job_ended()` moves speech to text back to the GPU; text to speech
loads again on its next use. Errors: ValueError for what the caller sent (the API answers 400), RuntimeError for an engine that
cannot run now (409); both carry a sentence for the person.

Runtimes (see the PR for the research): Parakeet through NeMo (`nemo_toolkit[asr]`), because the transformers port needs a newer
transformers than the one Breeze TTS pins (4.57.3); Breeze through its repository's `breeze_infer` and `models` packages, which are
not pip-installable, so setup clones them to VOICE_DIR/breeze-tts and they are put on sys.path when the engine loads.
"""

import contextlib
import importlib.util
import io
import logging
import os
import sys
import tempfile
import threading
import wave
from pathlib import Path

_log = logging.getLogger("groundwork.voice")

STT_MODEL = "nvidia/parakeet-tdt-0.6b-v3"
TTS_MODEL = "BreezeBlue/Breeze-TTS-2"
TTS_REVISION = "3e28c5151381a722f1d8661b4118c298caa77aa4"
BREEZE_REPO = "https://github.com/breezeblue-ai/breeze-tts"
BREEZE_COMMIT = "6eba901d1845353802652c708a4740fdfc5baa75"  # the inference code setup checks out
VOICE_DIR = Path(os.environ.get("GROUNDWORK_VOICE_DIR") or Path(__file__).resolve().parent / ".voice")
SETUP = "Run `make voice-setup`, then restart the app."

STT_RATE = 16000
MAX_STT_SECONDS = 120  # a spoken turn, not a recording: longer audio is refused rather than held in memory
MAX_TTS_CHARS = 2000
# The one voice: a voice-design instruction (no reference recording), in the language of the text.
VOICE_INSTRUCTION = "A warm, calm adult voice with a clear, natural pace."
CFG_SCALE = 4.0  # the Breeze README's setting for stronger instruction following
MAX_NEW_TOKENS, MAX_SEQ_LEN, REPETITION_PENALTY, SEED = 1500, 2048, 1.1, 0


def _have(module):
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def breeze_dir():
    return VOICE_DIR / "breeze-tts"


def _breeze_missing():
    """What text to speech lacks, as one sentence; "" when it is all there."""
    lacks = []
    if not (breeze_dir() / "breeze_infer").is_dir():
        lacks.append("the Breeze TTS code")
    if not _have("qwen_tts"):
        lacks.append("the qwen-tts package")
    if not _have("torch"):
        lacks.append("PyTorch")
    if not (VOICE_DIR / "breeze-tts-2" / "audio_tokenizer").is_dir():
        lacks.append("the Breeze TTS 2 weights")
    return f"Text to speech needs {', '.join(lacks)}. {SETUP}" if lacks else ""


def _nemo_missing():
    lacks = []
    if not _have("nemo"):
        lacks.append("NeMo")
    if not _have("torch"):
        lacks.append("PyTorch")
    return f"Speech to text needs {' and '.join(lacks)}. {SETUP}" if lacks else ""


def wav_samples(wav_bytes):
    """The raw 16-bit PCM frames of a 16 kHz mono 16-bit WAV; a ValueError with a sentence for anything else."""
    if not wav_bytes:
        raise ValueError("No audio came with the request. Record something and try again.")
    try:
        with wave.open(io.BytesIO(wav_bytes)) as wav:
            channels, width, rate, frames = wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()
            data = wav.readframes(frames)
    except (wave.Error, EOFError):
        raise ValueError("That is not a WAV file. Send 16 kHz mono 16-bit WAV audio.") from None
    if (channels, width, rate) != (1, 2, STT_RATE):
        raise ValueError(f"The audio must be 16 kHz mono 16-bit WAV; this one is {rate} Hz, {channels} channel(s), {width * 8}-bit.")
    if not data:
        raise ValueError("The recording is empty. Record something and try again.")
    if frames > MAX_STT_SECONDS * STT_RATE:
        raise ValueError(f"The recording is longer than {MAX_STT_SECONDS} seconds. Say less at a time.")
    return data


def wav_bytes_from(samples, rate):
    """16-bit mono WAV bytes from float samples in [-1, 1] (a numpy array)."""
    import numpy as np
    pcm = (np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0) * 32767).astype("<i2").tobytes()
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm)
    return out.getvalue()


class SpeechToText:
    name, label = "stt", "Speech to text"
    model_name = STT_MODEL

    def __init__(self):
        self.model, self.device = None, None

    def missing(self):
        return _nemo_missing()

    def load(self, device):
        import nemo.collections.asr as nemo_asr
        _log.info("loading %s on %s", STT_MODEL, device)
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=STT_MODEL, map_location="cpu")
        self.model = model.to(device).eval()
        self.device = device

    def move(self, device):
        if self.model is not None and device != self.device:
            self.model.to(device)
            self.device = device
            _empty_cuda_cache()

    def unload(self):
        self.model, self.device = None, None
        _empty_cuda_cache()

    def run(self, pcm):
        """The text of 16 kHz mono 16-bit PCM frames."""
        handle, name = tempfile.mkstemp(suffix=".wav")  # NeMo transcribes files
        os.close(handle)
        try:
            with wave.open(name, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(STT_RATE)
                wav.writeframes(pcm)
            out = self.model.transcribe([name], batch_size=1, verbose=False)
        finally:
            with contextlib.suppress(OSError):
                os.unlink(name)
        if isinstance(out, tuple):  # older NeMo: (best, all)
            out = out[0]
        first = out[0] if out else ""
        return (getattr(first, "text", first) or "").strip()


class TextToSpeech:
    name, label = "tts", "Text to speech"
    model_name = TTS_MODEL

    def __init__(self):
        self.runtime, self.parts, self.device = None, None, None

    def missing(self):
        return _breeze_missing()

    def load(self, device):
        code = str(breeze_dir())
        if code not in sys.path:
            sys.path.insert(0, code)
        from breeze_infer.runtime import load_runtime, update_generation_config_for_breeze
        from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig
        _log.info("loading %s on %s", TTS_MODEL, device)
        tokenizer, model, audio_tokenizer = load_runtime(VOICE_DIR / "breeze-tts-2", device=device, attn_implementation="eager")  # eager: no flash-attn
        update_generation_config_for_breeze(model)
        config = FastStreamingConfig(max_new_tokens=MAX_NEW_TOKENS, max_seq_len=MAX_SEQ_LEN, fast_all=False, repetition_penalty=REPETITION_PENALTY)
        self.runtime = FastBreezeStreamingRuntime(model, audio_tokenizer, config, tokenizer=tokenizer)
        self.parts = (tokenizer, audio_tokenizer, model)
        self.device = device

    def move(self, device):
        """Text to speech does not run on a CPU: leaving the GPU unloads it."""
        if device != self.device:
            self.unload()

    def unload(self):
        self.runtime, self.parts, self.device = None, None, None
        _empty_cuda_cache()

    def run(self, text):
        import numpy as np
        from breeze_infer.runtime import set_all_seeds
        from breeze_infer.templates import get_template, prepare_inputs, select_template_name
        tokenizer, audio_tokenizer, model = self.parts
        request = {"id": "voice", "text": text, "speaker": "S0", "instruction": VOICE_INSTRUCTION}
        set_all_seeds(SEED)
        inputs = prepare_inputs(tokenizer, audio_tokenizer, model, [request], get_template(select_template_name(request)),
                                guidance_scale=CFG_SCALE, guidance_scale_ref=None, guidance_scale_ins=None)
        chunks = [np.asarray(chunk.audio, dtype=np.float32).reshape(-1)
                  for chunk in self.runtime.iter_audio_chunks(inputs, request_id="voice", seed=SEED)]
        if not chunks:
            raise RuntimeError("The voice made no sound for that text. Try different words.")
        return wav_bytes_from(np.concatenate(chunks), self.runtime.sample_rate)


def _empty_cuda_cache():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


class Voice:
    """Both engines and the one lock that serialises them. `cuda` and the engines are injectable for tests."""

    def __init__(self, stt=None, tts=None, cuda=_cuda, busy=None):
        self.stt, self.tts = stt or SpeechToText(), tts or TextToSpeech()
        self._cuda = cuda
        self._busy = busy or (lambda label: contextlib.nullcontext())  # the services pass the busy card
        self._lock = threading.RLock()
        self._job_running = False

    def _device(self):
        return "cuda" if self._cuda() else "cpu"

    def status(self):
        """[{engine, label, model, installed, loaded, device, missing}] for speech to text, then text to speech."""
        with self._lock:
            rows = []
            for engine in (self.stt, self.tts):
                missing = engine.missing()
                rows.append({"engine": engine.name, "label": engine.label, "model": engine.model_name, "installed": not missing,
                             "loaded": engine.device is not None, "device": engine.device, "missing": missing})
            return rows

    def transcribe(self, wav_bytes):
        pcm = wav_samples(wav_bytes)
        with self._lock:
            missing = self.stt.missing()
            if missing:
                raise RuntimeError(missing)
            if self.stt.device is None:
                with self._busy("Loading speech to text"):
                    self.stt.load("cpu" if self._job_running else self._device())
            return self.stt.run(pcm)

    def speak(self, text):
        text = (text or "").strip()
        if not text:
            raise ValueError("There is no text to speak.")
        if len(text) > MAX_TTS_CHARS:
            raise ValueError(f"That reply is longer than {MAX_TTS_CHARS} characters; the voice speaks shorter text.")
        with self._lock:
            missing = self.tts.missing()
            if missing:
                raise RuntimeError(missing)
            if self._job_running:
                raise RuntimeError("The voice is paused while a training job runs, because it needs the GPU. It comes back when the job ends.")
            if not self._cuda():
                raise RuntimeError("The voice needs an NVIDIA GPU with about 8 GB of free memory, and this computer has none available.")
            if self.tts.device is None:
                with self._busy("Loading the voice"):
                    self.tts.load(self._device())
            return self.tts.run(text)

    def job_started(self):
        """The queue's call as a job is about to start: speech to text goes to the CPU, text to speech is unloaded."""
        with self._lock:
            self._job_running = True
            if self.stt.device is not None:
                self._guard(lambda: self.stt.move("cpu"), "speech to text")
            if self.tts.device is not None:
                self._guard(self.tts.unload, "text to speech")

    def job_ended(self):
        """The queue's call after a job: speech to text may use the GPU again; text to speech loads on its next use."""
        with self._lock:
            self._job_running = False
            if self.stt.device is not None:
                self._guard(lambda: self.stt.move(self._device()), "speech to text")

    @staticmethod
    def _guard(action, what):
        try:
            action()
        except RuntimeError:  # e.g. no room on the GPU: the engine stays where it is
            _log.warning("could not move %s", what, exc_info=True)


def _install_setup(root=VOICE_DIR):
    """What `make voice-setup` does after pip: clone the Breeze code and fetch both weights. Not called by the app."""
    import subprocess
    root.mkdir(parents=True, exist_ok=True)
    code = root / "breeze-tts"
    if not code.is_dir():
        subprocess.run(["git", "clone", BREEZE_REPO, str(code)], check=True)
    subprocess.run(["git", "-C", str(code), "checkout", BREEZE_COMMIT], check=True)
    from huggingface_hub import snapshot_download
    snapshot_download(TTS_MODEL, revision=TTS_REVISION, local_dir=str(root / "breeze-tts-2"))
    import nemo.collections.asr as nemo_asr
    nemo_asr.models.ASRModel.from_pretrained(model_name=STT_MODEL)  # into the Hugging Face cache
    print(f"Voice engines ready. Weights: {STT_MODEL}, {TTS_MODEL}@{TTS_REVISION[:7]}.")


if __name__ == "__main__":
    _install_setup()
