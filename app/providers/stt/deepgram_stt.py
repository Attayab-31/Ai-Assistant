"""
app/providers/stt/deepgram_stt.py — Deepgram Nova-3 real-time streaming STT.

Primary STT provider. Deepgram Nova-3 offers industry-leading accuracy
for phone call audio (mulaw 8kHz). Provides real-time streaming transcription
via WebSocket connection.
"""

import io
import logging
import time
import wave

import httpx
from deepgram import (
    DeepgramClient,
    FileSource,
    PrerecordedOptions,
)

from app.providers.base import BaseSTTProvider
from config import settings

logger = logging.getLogger(__name__)


class DeepgramSTTProvider(BaseSTTProvider):
    """
    Deepgram Nova-3 STT provider.
    Supports both streaming (WebSocket) and batch transcription.
    """

    provider_name = "deepgram"

    def __init__(self, model: str = "nova-3", language: str = "en-US") -> None:
        self.model = model
        self.language = language
        self._client: DeepgramClient | None = None
        logger.info(f"DeepgramSTTProvider initialized: model={model}, lang={language}")

    @property
    def client(self) -> DeepgramClient:
        """Lazy-initialize Deepgram client."""
        if self._client is None:
            if not settings.deepgram_api_key:
                raise ValueError("DEEPGRAM_API_KEY not set in environment")
            self._client = DeepgramClient(settings.deepgram_api_key)
        return self._client

    async def transcribe_chunk(self, audio_bytes: bytes) -> str:
        """
        Transcribe mulaw 8kHz audio using Deepgram pre-recorded API.
        Sends mulaw directly (fast path); falls back to PCM WAV if needed.
        """
        if not audio_bytes:
            return ""

        try:
            source: FileSource = {"buffer": audio_bytes}
            options = PrerecordedOptions(
                model=self.model,
                language=self.language,
                punctuate=True,
                encoding="mulaw",
                sample_rate=8000,
            )
            response = await self.client.listen.asyncprerecorded.v("1").transcribe_file(
                source, options
            )
            transcript = response.results.channels[0].alternatives[0].transcript
            if transcript.strip():
                return transcript.strip()
        except Exception as e:
            logger.debug(f"Deepgram mulaw STT failed, trying WAV: {e}")

        from app.utils.audio import mulaw_to_wav

        return await self.transcribe_wav_chunk(mulaw_to_wav(audio_bytes))

    async def transcribe_wav_chunk(self, wav_bytes: bytes) -> str:
        """Transcribe PCM16 WAV audio (browser/test console path)."""
        if not wav_bytes:
            return ""
        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
                sample_rate = wav.getframerate()
            source: FileSource = {"buffer": wav_bytes}
            options = PrerecordedOptions(
                model=self.model,
                language=self.language,
                punctuate=True,
                encoding="linear16",
                sample_rate=sample_rate,
            )
            response = await self.client.listen.asyncprerecorded.v("1").transcribe_file(
                source, options
            )
            transcript = response.results.channels[0].alternatives[0].transcript
            return transcript.strip()
        except Exception as e:
            logger.error(f"Deepgram WAV transcription error: {e}")
            return ""

    async def ping(self) -> tuple[bool, float]:
        """Verify the Deepgram API key and STT service are reachable."""
        if not settings.deepgram_api_key:
            return False, 0.0
        start = time.time()
        try:
            async with httpx.AsyncClient(timeout=6.0) as client:
                resp = await client.get(
                    "https://api.deepgram.com/v1/projects",
                    headers={"Authorization": f"Token {settings.deepgram_api_key}"},
                )
            latency_ms = (time.time() - start) * 1000
            return resp.status_code == 200, round(latency_ms, 1)
        except Exception as e:
            logger.debug("Deepgram STT ping failed: %s", e)
            return False, 0.0
