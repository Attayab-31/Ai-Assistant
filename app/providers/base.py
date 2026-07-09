"""
app/providers/base.py — Abstract base classes for STT, LLM, and TTS providers.

All concrete providers must implement these interfaces. This ensures
the ProviderRegistry can hot-swap any provider without changing call code.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

import httpx

logger = logging.getLogger(__name__)


def resolve_frozen_credential(
    frozen_value: str | None,
    *,
    settings_attr: str,
) -> str:
    """Use a per-call frozen credential when provided; else live settings."""
    if frozen_value is not None:
        return frozen_value.strip()
    from config import settings

    return (getattr(settings, settings_attr, None) or "").strip()


def build_llm_messages(system_prompt: str, messages: list[dict]) -> list[dict]:
    """Prepend the system prompt to a chat message list."""
    return [{"role": "system", "content": system_prompt}] + messages


def openai_chat_kwargs(
    *,
    model: str,
    system_prompt: str,
    messages: list[dict],
    json_mode: bool = False,
    temperature: float = 0.3,
    max_tokens: int = 500,
    stream: bool = False,
    extra: dict | None = None,
) -> dict:
    """Build kwargs for OpenAI-compatible ``chat.completions.create`` calls."""
    kwargs: dict = {
        "model": model,
        "messages": build_llm_messages(system_prompt, messages),
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if stream:
        kwargs["stream"] = True
    if extra:
        kwargs.update(extra)
    return kwargs


async def complete_openai_chat(
    provider: "BaseLLMProvider",
    client,
    kwargs: dict,
) -> str:
    """Run a non-streaming OpenAI-compatible chat completion."""
    provider.last_usage = None
    response = await client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content or ""
    provider.last_usage = usage_from_response(response)
    return content


async def stream_openai_chat(
    provider: "BaseLLMProvider",
    client,
    kwargs: dict,
) -> AsyncIterator[str]:
    """Yield text deltas from a streaming OpenAI-compatible chat completion."""
    provider.last_usage = None
    stream = await client.chat.completions.create(**kwargs)
    async for chunk in stream:
        if getattr(chunk, "usage", None):
            provider.last_usage = usage_from_response(chunk)
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def usage_from_response(response) -> dict | None:
    """Extract token usage from an OpenAI-style chat completion response.

    Groq, OpenAI, OpenRouter and Gemini (OpenAI-compatible endpoint) all return
    a ``usage`` object with prompt/completion/total token counts. Returns None
    when the provider omitted usage so callers can treat it as "unknown" rather
    than zero.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    try:
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion = int(getattr(usage, "completion_tokens", 0) or 0)
        total = int(getattr(usage, "total_tokens", 0) or 0) or (prompt + completion)
    except (TypeError, ValueError):
        return None
    if prompt == 0 and completion == 0 and total == 0:
        return None
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }


async def http_api_ping(
    url: str,
    headers: dict[str, str],
    *,
    timeout: float = 6.0,
) -> tuple[bool, float]:
    """GET *url* with *headers*; return (ok, latency_ms)."""
    start = time.time()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url, headers=headers)
        latency_ms = (time.time() - start) * 1000
        return resp.status_code == 200, round(latency_ms, 1)
    except Exception as e:
        logger.debug("HTTP API ping failed for %s: %s", url, e)
        return False, 0.0


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
    # Token usage from the most recent get_response() call (or None if the call
    # failed / the provider omitted usage). Read immediately after the call so a
    # later call on the same instance doesn't overwrite it before it's consumed.
    last_usage: dict | None = None

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
