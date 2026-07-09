"""
app/providers/tts/deepgram_tts.py — Deepgram Aura-2 TTS (fallback).
"""

import asyncio
import logging
import re

import httpx

from app.providers.base import BaseTTSProvider, resolve_frozen_credential

logger = logging.getLogger(__name__)

DEEPGRAM_TTS_URL = "https://api.deepgram.com/v1/speak"
DEFAULT_VOICE = "aura-2-thalia-en"
VOICE_NAME_PATTERN = re.compile(r"^aura(?:-2)?-[a-z0-9-]+-[a-z]{2}$")

_DEEPGRAM_TTS_MAX_CONCURRENCY = 8
_DEEPGRAM_TTS_SEMAPHORE = asyncio.Semaphore(_DEEPGRAM_TTS_MAX_CONCURRENCY)

# One shared async client per process — avoids a TLS handshake on every utterance.
_shared_client: httpx.AsyncClient | None = None
_client_lock = asyncio.Lock()

AVAILABLE_VOICES = [
    "aura-2-thalia-en",
    "aura-2-andromeda-en",
    "aura-2-helena-en",
    "aura-2-apollo-en",
    "aura-2-arcas-en",
    "aura-2-aries-en",
    "aura-2-asteria-en",
    "aura-asteria-en",
    "aura-luna-en",
    "aura-stella-en",
    "aura-zeus-en",
    "aura-orpheus-en",
    "aura-angus-en",
    "aura-helios-en",
    "aura-2-estrella-es",
    "aura-2-nestor-es",
    "aura-2-celeste-es",
    "aura-2-sirio-es",
    "aura-2-javier-es",
    "aura-2-alvaro-es",
]


async def _shared_http_client() -> httpx.AsyncClient:
    global _shared_client
    async with _client_lock:
        if _shared_client is None or _shared_client.is_closed:
            _shared_client = httpx.AsyncClient(
                timeout=30.0,
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            )
        return _shared_client


class DeepgramTTSProvider(BaseTTSProvider):
    """Deepgram Aura-2 TTS. Used as fallback when Google TTS is unavailable."""

    provider_name = "deepgram"

    def __init__(self, voice: str = DEFAULT_VOICE, *, api_key: str | None = None) -> None:
        self.voice = self._normalize_voice(voice)
        self._api_key = api_key
        logger.info("DeepgramTTSProvider initialized: voice=%s", self.voice)

    @staticmethod
    def _normalize_voice(voice: str) -> str:
        clean_voice = (voice or DEFAULT_VOICE).strip().lower()
        if clean_voice in AVAILABLE_VOICES or VOICE_NAME_PATTERN.match(clean_voice):
            return clean_voice
        logger.warning(
            "Unsupported Deepgram TTS voice '%s', using %s", voice, DEFAULT_VOICE
        )
        return DEFAULT_VOICE

    async def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
    ) -> bytes:
        key = resolve_frozen_credential(self._api_key, settings_attr="deepgram_api_key")
        if not key:
            raise ValueError("DEEPGRAM_API_KEY not set")

        active_voice = voice or self.voice
        params = {
            "model": active_voice,
            "encoding": "mulaw",
            "sample_rate": 8000,
            "container": "none",
        }

        headers = {
            "Authorization": f"Token {key}",
            "Content-Type": "application/json",
        }

        async with _DEEPGRAM_TTS_SEMAPHORE:
            client = await _shared_http_client()
            try:
                response = await client.post(
                    DEEPGRAM_TTS_URL,
                    json={"text": text},
                    params=params,
                    headers=headers,
                )
                response.raise_for_status()
                logger.debug(
                    "Deepgram TTS: %d chars → %d bytes",
                    len(text),
                    len(response.content),
                )
                return response.content
            except httpx.HTTPStatusError as e:
                logger.error(
                    "Deepgram TTS HTTP error %s: %s",
                    e.response.status_code,
                    e.response.text,
                )
                raise
            except Exception as e:
                logger.error("Deepgram TTS error: %s", e)
                raise
