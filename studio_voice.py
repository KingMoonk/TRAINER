"""Cross-platform local voice engines: sherpa-onnx speech recognition and Piper neural speech synthesis.

Both run on the CPU and load only on first use. `make voice-setup` installs their platform wheels and downloads the selected
models; the app never installs dependencies. Errors carry actionable sentences for the person.
"""

import contextlib
import importlib.util
import io
import logging
import os
import sys
import threading
import wave
from pathlib import Path

_log = logging.getLogger("groundwork.voice")

STT_MODEL = "sherpa-onnx-zipformer-small-en-2023-06-26"
STT_MODEL_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-small-en-2023-06-26.tar.bz2"
TTS_MODEL = "Piper TTS"
TTS_VOICES = {
    "lessac": {"label": "Lessac · American English · high quality", "model": "en_US-lessac-high"},
    "ryan": {"label": "Ryan · American English · high quality", "model": "en_US-ryan-high"},
    "amy": {"label": "Amy · American English · medium quality", "model": "en_US-amy-medium"},
}
DEFAULT_TTS_VOICE = "lessac"
VOICE_DIR = Path(os.environ.get("GROUNDWORK_VOICE_DIR") or Path(__file__).resolve().parent / ".voice")
SETUP = "Run `make voice-setup`, then restart the app."
STT_DIR = VOICE_DIR / STT_MODEL
TTS_DIR = VOICE_DIR / "piper"

STT_RATE = 16000
MAX_STT_SECONDS = 120  # a spoken turn, not a recording: longer audio is refused rather than held in memory
MAX_TTS_CHARS = 2000


def _have(module):
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _piper_missing():
    lacks = []
    if not _have("piper"):
        lacks.append("the Piper TTS package")
    absent = [spec["model"] for spec in TTS_VOICES.values() if not (TTS_DIR / f"{spec['model']}.onnx").is_file()]
    if absent:
        lacks.append("voice models " + ", ".join(absent))
    return f"Text to speech needs {', '.join(lacks)}. {SETUP}" if lacks else ""


def _sherpa_missing():
    lacks = []
    if not _have("sherpa_onnx"):
        lacks.append("sherpa-onnx")
    required = ("tokens.txt", "encoder-epoch-99-avg-1.int8.onnx", "decoder-epoch-99-avg-1.onnx", "joiner-epoch-99-avg-1.int8.onnx")
    if any(not (STT_DIR / filename).is_file() for filename in required):
        lacks.append("the offline speech-recognition model")
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
        return _sherpa_missing()

    def load(self, device):
        import sherpa_onnx
        _log.info("loading %s on CPU", STT_MODEL)
        self.model = sherpa_onnx.OfflineRecognizer.from_transducer(
            tokens=str(STT_DIR / "tokens.txt"),
            encoder=str(STT_DIR / "encoder-epoch-99-avg-1.int8.onnx"),
            decoder=str(STT_DIR / "decoder-epoch-99-avg-1.onnx"),
            joiner=str(STT_DIR / "joiner-epoch-99-avg-1.int8.onnx"),
            num_threads=max(1, (os.cpu_count() or 2) // 2),
            sample_rate=STT_RATE,
            feature_dim=80,
            decoding_method="greedy_search",
            provider="cpu",
        )
        self.device = "cpu"

    def move(self, device):
        self.device = "cpu" if self.model is not None else None

    def unload(self):
        self.model, self.device = None, None

    def run(self, pcm):
        """The text of 16 kHz mono 16-bit PCM frames."""
        import numpy as np
        stream = self.model.create_stream()
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
        stream.accept_waveform(STT_RATE, samples)
        self.model.decode_stream(stream)
        return (stream.result.text or "").strip()


class TextToSpeech:
    name, label = "tts", "Text to speech"
    model_name = TTS_MODEL

    def __init__(self):
        self.runtime, self.device, self.voice_id = None, None, None

    def missing(self):
        return _piper_missing()

    def load(self, device):
        self._load_voice(DEFAULT_TTS_VOICE)

    def move(self, device):
        self.device = "cpu" if self.runtime is not None else None

    def _load_voice(self, voice_id):
        import piper
        if voice_id not in TTS_VOICES:
            raise ValueError("Choose one of the available text-to-speech voices.")
        model = TTS_DIR / f"{TTS_VOICES[voice_id]['model']}.onnx"
        _log.info("loading %s voice %s on CPU", TTS_MODEL, voice_id)
        self.runtime = piper.PiperVoice.load(model, use_cuda=False)
        self.voice_id, self.device = voice_id, "cpu"

    def unload(self):
        self.runtime, self.device, self.voice_id = None, None, None

    def run(self, text, voice_id=DEFAULT_TTS_VOICE):
        if voice_id not in TTS_VOICES:
            raise ValueError("Choose one of the available text-to-speech voices.")
        if self.voice_id != voice_id:
            self._load_voice(voice_id)
        out = io.BytesIO()
        with wave.open(out, "wb") as wav:
            self.runtime.synthesize_wav(text, wav)
        return out.getvalue()


class Voice:
    """Both engines and the one lock that serialises them."""

    def __init__(self, stt=None, tts=None, busy=None):
        self.stt, self.tts = stt or SpeechToText(), tts or TextToSpeech()
        self._busy = busy or (lambda label: contextlib.nullcontext())  # the services pass the busy card
        self._lock = threading.RLock()
        self._job_running = False

    def status(self):
        """The install and load status of speech recognition and speech synthesis."""
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
                    self.stt.load("cpu")
            return self.stt.run(pcm)

    def speak(self, text, voice_id=DEFAULT_TTS_VOICE):
        text = (text or "").strip()
        if not text:
            raise ValueError("There is no text to speak.")
        if len(text) > MAX_TTS_CHARS:
            raise ValueError(f"That reply is longer than {MAX_TTS_CHARS} characters; the voice speaks shorter text.")
        if voice_id not in TTS_VOICES:
            raise ValueError("Choose one of the available text-to-speech voices.")
        with self._lock:
            missing = self.tts.missing()
            if missing:
                raise RuntimeError(missing)
            if self._job_running:
                raise RuntimeError("Speech is paused while a training job runs to leave CPU resources available. It comes back when the job ends.")
            if self.tts.device is None:
                with self._busy("Loading the voice"):
                    self.tts.load("cpu")
            return self.tts.run(text, voice_id)

    def job_started(self):
        """The queue's call as a job is about to start: speech to text goes to the CPU, text to speech is unloaded."""
        with self._lock:
            self._job_running = True
            if self.stt.device is not None:
                self._guard(lambda: self.stt.move("cpu"), "speech to text")
            if self.tts.device is not None:
                self._guard(self.tts.unload, "text to speech")

    def job_ended(self):
        """The queue's call after a job; text to speech loads on its next use."""
        with self._lock:
            self._job_running = False
            if self.stt.device is not None:
                self._guard(lambda: self.stt.move("cpu"), "speech to text")

    @staticmethod
    def _guard(action, what):
        try:
            action()
        except RuntimeError:  # e.g. no room on the GPU: the engine stays where it is
            _log.warning("could not move %s", what, exc_info=True)


def _install_setup(root=VOICE_DIR):
    """Download the cross-platform speech models after the optional packages are installed."""
    import subprocess
    import tarfile
    import urllib.request

    root.mkdir(parents=True, exist_ok=True)
    TTS_DIR.mkdir(parents=True, exist_ok=True)
    for voice in TTS_VOICES.values():
        subprocess.run([sys.executable, "-m", "piper.download_voices", "--download-dir", str(TTS_DIR), voice["model"]], check=True)
    if not (STT_DIR / "tokens.txt").is_file():
        archive_path = root / f"{STT_MODEL}.tar.bz2"
        partial_path = archive_path.with_suffix(archive_path.suffix + ".part")
        try:
            if not archive_path.is_file():
                urllib.request.urlretrieve(STT_MODEL_URL, partial_path)
                partial_path.replace(archive_path)
            with tarfile.open(archive_path, "r:bz2") as archive:
                archive.extractall(root, filter="data")
            archive_path.unlink(missing_ok=True)
        finally:
            partial_path.unlink(missing_ok=True)
    print(f"Voice engines ready: {STT_MODEL} and {len(TTS_VOICES)} Piper voices in {TTS_DIR}.")


if __name__ == "__main__":
    _install_setup()
