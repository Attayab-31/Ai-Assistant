"""
app/providers/base.py — Abstract base classes for STT, LLM, and TTS providers.

All concrete providers must implement these interfaces. This ensures
the ProviderRegistry can hot-swap any provider without changing call code.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseSTTProvider(ABC):
    """Abstract Speech-to-Text provider interface."""

    provider_name: str = "base"

    @abstractmethod
    async def transcribe_chunk(self, audio_bytes: bytes) -> str:
        """
        Transcribe a single audio chunk (non-streaming fallback).
        Returns complete transcript string.
        """
        pass

    async def ping(self) -> tuple[bool, float]:
        """
        Health check. Returns (is_healthy, latency_ms).
        Override in subclass for real check.
        """
        return False, 0.0


class BaseLLMProvider(ABC):
    """Abstract Large Language Model provider interface."""

    provider_name: str = "base"
    model: str = ""

    @abstractmethod
    async def get_response(
        self,
        system_prompt: str,
        messages: list[dict],
        json_mode: bool = False,
        temperature: float = 0.3,
        max_tokens: int = 500,
    ) -> str:
        """
        Get a response from the LLM.

        Args:
            system_prompt: System instruction for the model
            messages: List of {"role": "user"/"assistant", "content": "..."}
            json_mode: If True, instruct model to return valid JSON
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum response tokens

        Returns:
            String response from the model
        """
        pass

    async def ping(self) -> tuple[bool, float]:
        """
        Health check by sending a minimal test prompt.
        Returns (is_healthy, latency_ms).
        """
        try:
            start = time.time()
            response = await self.get_response(
                system_prompt="You are a test assistant.",
                messages=[{"role": "user", "content": "Say 'ok' in one word only."}],
                max_tokens=5,
            )
            latency_ms = (time.time() - start) * 1000
            return bool(response), round(latency_ms, 1)
        except Exception as e:
            logger.debug("LLM health check failed: %s", e)
            return False, 0.0


class BaseTTSProvider(ABC):
    """Abstract Text-to-Speech provider interface."""

    provider_name: str = "base"

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
    ) -> bytes:
        """
        Convert text to speech audio bytes.

        Returns:
            Audio bytes in mulaw 8kHz format (Telnyx compatible).
            Falls back to WAV/PCM if mulaw conversion unavailable.
        """
        pass

    async def ping(self) -> tuple[bool, float]:
        """Health check. Returns (is_healthy, latency_ms)."""
        try:
            start = time.time()
            audio = await asyncio.wait_for(self.synthesize("Hello."), timeout=5.0)
            latency_ms = (time.time() - start) * 1000
            return len(audio) > 0, round(latency_ms, 1)
        except Exception as e:
            logger.debug("TTS health check failed: %s", e)
            return False, 0.0
