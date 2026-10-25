"""Voice mode: speech to text and text to speech, both local (studio_voice).

GET /voice says which engines are installed and loaded. POST /voice/transcribe takes a 16 kHz mono 16-bit WAV (multipart, field
`audio`) and answers {text}. POST /voice/speak takes {text} and answers audio/wav. Audio that cannot be read or no text is a 400,
an engine that is missing or cannot run now (not installed, a training job has the GPU) a 409, both with a sentence in `detail`.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, File, Response, UploadFile
from pydantic import BaseModel, Field

from studio_api.app import get_services
from studio_voice import DEFAULT_TTS_VOICE

router = APIRouter(prefix="/voice", tags=["voice"])


class Engine(BaseModel):
    engine: str = Field(description="'stt' (speech to text) or 'tts' (text to speech)")
    label: str
    model: str = Field(description="the model's name on Hugging Face")
    installed: bool = Field(description="false until `make voice-setup` has installed it")
    loaded: bool = Field(description="true once a request has loaded it; it stays loaded until a training job needs the GPU")
    device: str | None = Field(description="where it is loaded ('cuda' or 'cpu'); null when not loaded")
    missing: str = Field(description="a sentence naming what is missing and the command that fixes it; empty when installed")


class VoiceStatus(BaseModel):
    engines: list[Engine]


class Transcript(BaseModel):
    text: str


class SpeakRequest(BaseModel):
    text: str
    voice: str = DEFAULT_TTS_VOICE


@router.get("", response_model=VoiceStatus)
def status(services=Depends(get_services)):
    """Both engines, speech to text first."""
    return VoiceStatus(engines=services.voice_status())


@router.post("/transcribe", response_model=Transcript)
def transcribe(audio: Annotated[UploadFile, File(description="16 kHz mono 16-bit PCM WAV")], services=Depends(get_services)):
    """The words in the recording. The first call loads the model, which takes a while."""
    return Transcript(text=services.transcribe(audio.file.read()))


@router.post("/speak", responses={200: {"content": {"audio/wav": {"schema": {"type": "string", "format": "binary"}}}}})
def speak(body: SpeakRequest, services=Depends(get_services)):
    """The text spoken in the selected voice, as WAV. The first call loads the model, which takes a while."""
    return Response(services.speak(body.text, body.voice), media_type="audio/wav")
