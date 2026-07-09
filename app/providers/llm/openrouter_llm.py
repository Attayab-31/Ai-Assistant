"""OpenRouter LLM provider."""

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

    def __init__(
        self, model: str = "meta-llama/llama-3.3-70b-instruct:free", *, api_key: str | None = None
    ) -> None:
        self.model = model if model in AVAILABLE_MODELS else AVAILABLE_MODELS[0]
        self._api_key = api_key
        self._client: AsyncOpenAI | None = None
        logger.info(f"OpenRouterLLMProvider initialized with model: {self.model}")

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            key = resolve_frozen_credential(
                self._api_key, settings_attr="openrouter_api_key"
            )
            if not key:
                raise ValueError("OPENROUTER_API_KEY not set in environment")
            self._client = AsyncOpenAI(
                api_key=key,
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
