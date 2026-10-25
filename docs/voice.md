# Voice mode engines

Local speech for Eval voice mode (issue #694). `studio_voice.py` holds the engines, `studio_api/routers/voice.py` the routes.

- Speech to text: `nvidia/parakeet-tdt-0.6b-v3` through NeMo (`nemo_toolkit[asr]`). The transformers port is not in a stable release and needs a newer transformers than Breeze TTS pins (4.57.3), so NeMo is the one runtime that fits both in one `.venv`.
- Text to speech: `BreezeBlue/Breeze-TTS-2` at revision `3e28c51`. Its inference code (github.com/breezeblue-ai/breeze-tts, pinned commit in `studio_voice.BREEZE_COMMIT`) is not pip-installable, so setup clones it to `.voice/breeze-tts` and the weights go to `.voice/breeze-tts-2`. It runs in eager attention mode (flash-attn is not needed), bfloat16, about 7.7 GiB of VRAM, and needs CUDA. The weights are research and non-commercial only.
- `make voice-setup` installs `requirements-voice.txt`, clones the code and fetches both weights (several GB). The app never installs anything; `GET /voice` says `make voice-setup` for whatever is missing.
- Platforms: the Breeze README targets Linux. NeMo and qwen-tts list Python up to 3.13 and neither states Windows support, so on Windows with Python 3.14 the install may fail; `GET /voice` then keeps saying the engine is missing.
- GPU sharing: when a queue job starts, speech to text moves to the CPU (still works, slower) and text to speech is unloaded (a CPU is too slow for it); speaking fails with a sentence until the job ends. Text to speech loads again on its next use.
- API: `GET /voice` -> `{engines: [{engine, label, model, installed, loaded, device, missing}]}`; `POST /voice/transcribe` (multipart field `audio`, 16 kHz mono 16-bit WAV) -> `{text}`; `POST /voice/speak` (`{text}`) -> `audio/wav` (24 kHz mono 16-bit). A bad request is a 400 and an engine that cannot run now a 409, each with a sentence in `detail`.
