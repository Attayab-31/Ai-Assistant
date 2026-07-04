"""OpenRouter LLM provider."""

import logging
from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from app.providers.base import BaseLLMProvider, usage_from_response
from config import settings

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

AVAILABLE_MODELS = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "meta-llama/llama-3.1-8b-instruct:free",
    "openai/gpt-4o-mini",
    "anthropic/claude-3.5-haiku",
]


class OpenRouterLLMProvider(BaseLLMProvider):
    """OpenRouter provider using its OpenAI-compatible chat completions API."""

    provider_name = "openrouter"

    def __init__(self, model: str = "meta-llama/llama-3.3-70b-instruct:free") -> None:
        self.model = model if model in AVAILABLE_MODELS else AVAILABLE_MODELS[0]
        self._client: AsyncOpenAI | None = None
        logger.info(f"OpenRouterLLMProvider initialized with model: {self.model}")

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            if not settings.openrouter_api_key:
                raise ValueError("OPENROUTER_API_KEY not set in environment")
            self._client = AsyncOpenAI(
                api_key=settings.openrouter_api_key,
                base_url=OPENROUTER_BASE_URL,
                default_headers={
                    "HTTP-Referer": settings.app_url,
                    "X-Title": settings.app_name,
                },
            )
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

        self.last_usage = None
        try:
            response = await self.client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content or ""
            self.last_usage = usage_from_response(response)
            logger.debug(f"OpenRouter response: {content[:100]}...")
            return content
        except Exception as e:
            logger.error(f"OpenRouter API error: {e}")
            raise

    async def stream_response(
        self,
        system_prompt: str,
        messages: list[dict],
        json_mode: bool = False,
        temperature: float = 0.3,
        max_tokens: int = 500,
    ) -> AsyncIterator[str]:
        """Yield text deltas from a streaming OpenRouter completion."""
        all_messages = [{"role": "system", "content": system_prompt}] + messages
        kwargs = {
            "model": self.model,
            "messages": all_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        self.last_usage = None
        stream = await self.client.chat.completions.create(**kwargs)
        async for chunk in stream:
            if getattr(chunk, "usage", None):
                self.last_usage = usage_from_response(chunk)
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
