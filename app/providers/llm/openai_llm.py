"""
app/providers/llm/openai_llm.py — OpenAI GPT provider (secondary fallback).
"""

import logging
from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from app.providers.base import (
    BaseLLMProvider,
    complete_openai_chat,
    openai_chat_kwargs,
    resolve_frozen_credential,
    stream_openai_chat,
)

logger = logging.getLogger(__name__)

AVAILABLE_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]


class OpenAILLMProvider(BaseLLMProvider):
    """OpenAI GPT provider. Used as secondary fallback when Groq is unavailable."""

    provider_name = "openai"

    def __init__(self, model: str = "gpt-4o-mini", *, api_key: str | None = None) -> None:
        self.model = model if model in AVAILABLE_MODELS else "gpt-4o-mini"
        self._api_key = api_key
        self._client: AsyncOpenAI | None = None
        logger.info(f"OpenAILLMProvider initialized with model: {self.model}")

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            key = resolve_frozen_credential(self._api_key, settings_attr="openai_api_key")
            if not key:
                raise ValueError("OPENAI_API_KEY not set in environment")
            self._client = AsyncOpenAI(api_key=key)
        return self._client

    async def get_response(
        self,
        system_prompt: str,
        messages: list[dict],
        json_mode: bool = False,
        temperature: float = 0.3,
        max_tokens: int = 500,
    ) -> str:
        kwargs = openai_chat_kwargs(
            model=self.model,
            system_prompt=system_prompt,
            messages=messages,
            json_mode=json_mode,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        try:
            content = await complete_openai_chat(self, self.client, kwargs)
            logger.debug(f"OpenAI response: {content[:100]}...")
            return content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    async def stream_response(
        self,
        system_prompt: str,
        messages: list[dict],
        json_mode: bool = False,
        temperature: float = 0.3,
        max_tokens: int = 500,
    ) -> AsyncIterator[str]:
        """Yield text deltas from a streaming OpenAI completion."""
        kwargs = openai_chat_kwargs(
            model=self.model,
            system_prompt=system_prompt,
            messages=messages,
            json_mode=json_mode,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        async for delta in stream_openai_chat(self, self.client, kwargs):
            yield delta
