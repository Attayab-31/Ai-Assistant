"""
app/providers/stt/groq_stt.py — Groq Whisper STT (chunk-based fallback).

Uses Groq's Whisper implementation for high-speed batch transcription.
Used when Deepgram streaming is unavailable.
"""

import io
import logging

from groq import AsyncGroq

from app.providers.base import BaseSTTProvider, http_api_ping
from config import settings

logger = logging.getLogger(__name__)


class GroqSTTProvider(BaseSTTProvider):
    """
    Groq Whisper STT provider (chunk-based, non-streaming).
    Much faster than OpenAI Whisper. Used as Deepgram fallback.
    """

    provider_name = "groq_whisper"

    def __init__(self, model: str = "whisper-large-v3-turbo") -> None:
        self.model = model
        self._client: AsyncGroq | None = None
        logger.info(f"GroqSTTProvider initialized: model={model}")

    @property
    def client(self) -> AsyncGroq:
        if self._client is None:
            if not settings.groq_api_key:
                raise ValueError("GROQ_API_KEY not set")
            self._client = AsyncGroq(api_key=settings.groq_api_key)
        return self._client

    async def transcribe_chunk(self, audio_bytes: bytes) -> str:
        """Transcribe mulaw 8kHz audio using Groq Whisper API."""
        try:
            from app.utils.audio import mulaw_to_wav

            # Groq Whisper requires standard PCM16 WAV (not ULAW-compressed WAV).
            wav_bytes = mulaw_to_wav(audio_bytes)
            audio_file = io.BytesIO(wav_bytes)
            audio_file.name = "audio.wav"

            response = await self.client.audio.transcriptions.create(
                file=audio_file,
                model=self.model,
                response_format="text",
                language="en",
            )
            return str(response).strip()
        except Exception as e:
            logger.error(f"Groq STT error: {e}")
            return ""

    async def ping(self) -> tuple[bool, float]:
        """Verify the Groq API key and Whisper endpoint are reachable."""
        if not settings.groq_api_key:
            return False, 0.0
        return await http_api_ping(
            "https://api.groq.com/openai/v1/models",
            {"Authorization": f"Bearer {settings.groq_api_key}"},
        )
