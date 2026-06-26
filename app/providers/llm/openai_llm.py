"""
app/providers/llm/openai_llm.py — OpenAI GPT provider (secondary fallback).
"""

import logging

from openai import AsyncOpenAI

from app.providers.base import BaseLLMProvider
from config import settings

logger = logging.getLogger(__name__)

AVAILABLE_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]


class OpenAILLMProvider(BaseLLMProvider):
    """OpenAI GPT provider. Used as secondary fallback when Groq is unavailable."""

    provider_name = "openai"

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model if model in AVAILABLE_MODELS else "gpt-4o-mini"
        self._client: AsyncOpenAI | None = None
        logger.info(f"OpenAILLMProvider initialized with model: {self.model}")

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY not set in environment")
            self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        return self._client

    async def get_response(
        self,
        system_prompt: str,
        messages: list[dict],
        json_mode: bool = False,
        temperature: float = 0.3,
        max_tokens: int = 500,
    ) -> str:
        all_messages = [{"role": "system", "content": system_prompt}] + messages
        kwargs = {
            "model": self.model,
            "messages": all_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            response = await self.client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            logger.debug(f"OpenAI response: {content[:100]}...")
            return content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
