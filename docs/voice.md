# Voice mode engines

Voice mode uses local CPU inference, so speech works on Windows and macOS without an NVIDIA GPU. The app does not install packages or download models automatically.

- `make voice-setup` installs `piper-tts` and `sherpa-onnx` into `.venv`, then downloads the English Zipformer speech-recognition model and three selectable Piper voices into `.voice/`.
- Speech to text uses the small English Zipformer model through sherpa-onnx. Text to speech uses Piper's high-quality Lessac and Ryan voices plus Amy; change the spoken-reply voice in Settings. That preference is kept in the browser.
- The engine packages have binary wheels for current Windows and macOS Python versions, including Python 3.14. This removes the NeMo/Numba build failure and avoids the former Breeze model's CUDA-only runtime.
- Models load on first use and remain cached. TTS unloads while a training job runs to avoid competing for CPU resources; it reloads when next used.
- Piper is GPL-3.0-or-later; each downloaded voice model carries its own license in its model card. Check those terms before redistributing a bundled app or voice model.

The API remains `GET /voice`, `POST /voice/transcribe` (16 kHz mono 16-bit WAV), and `POST /voice/speak` (text and optional voice ID; response is WAV).
