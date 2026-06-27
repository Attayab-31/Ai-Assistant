"""Google Gemini LLM provider.

Uses Gemini's OpenAI-compatible endpoint so we can reuse the ``openai`` SDK
(no extra dependency). Gemini 2.5 Flash models apply *implicit* prompt caching
automatically, which discounts the large repeated system prompt, and the
Google AI Studio free tier has a more generous quota than Groq's daily wall.
"""

import logging

from openai import AsyncOpenAI

from app.providers.base import BaseLLMProvider
from config import settings

logger = logging.getLogger(__name__)

# Gemini's OpenAI-compatible base URL (chat.completions + response_format).
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

AVAILABLE_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
]


class GeminiLLMProvider(BaseLLMProvider):
    """Google Gemini provider via its OpenAI-compatible chat completions API."""

    provider_name = "gemini"

    def __init__(self, model: str = "gemini-2.5-flash") -> None:
        self.model = model if model in AVAILABLE_MODELS else AVAILABLE_MODELS[0]
        self._client: AsyncOpenAI | None = None
        logger.info(f"GeminiLLMProvider initialized with model: {self.model}")

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            if not settings.gemini_api_key:
                raise ValueError("GEMINI_API_KEY not set in environment")
            self._client = AsyncOpenAI(
                api_key=settings.gemini_api_key,
                base_url=GEMINI_BASE_URL,
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

        try:
            response = await self.client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content or ""
            logger.debug(f"Gemini response: {content[:100]}...")
            return content
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise
