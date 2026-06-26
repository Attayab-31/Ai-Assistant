"""
app/providers/llm/groq_llm.py — Groq LLM provider (Llama 3.3 70B).

Primary LLM provider. Groq offers free API access with GPT-4 level quality
at ~90ms latency. Uses the groq Python SDK.
"""

import logging

from groq import AsyncGroq

from app.providers.base import BaseLLMProvider
from config import settings

logger = logging.getLogger(__name__)

AVAILABLE_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama-3.1-70b-versatile",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]


class GroqLLMProvider(BaseLLMProvider):
    """
    Groq LLM provider using Llama 3.3 70B (default).
    Free API with very low latency — ideal for real-time voice calls.
    """

    provider_name = "groq"

    def __init__(self, model: str = "llama-3.3-70b-versatile") -> None:
        self.model = model if model in AVAILABLE_MODELS else "llama-3.3-70b-versatile"
        self._client: AsyncGroq | None = None
        logger.info(f"GroqLLMProvider initialized with model: {self.model}")

    @property
    def client(self) -> AsyncGroq:
        """Lazy-initialize Groq client."""
        if self._client is None:
            if not settings.groq_api_key:
                raise ValueError("GROQ_API_KEY not set in environment")
            self._client = AsyncGroq(api_key=settings.groq_api_key)
        return self._client

    async def get_response(
        self,
        system_prompt: str,
        messages: list[dict],
        json_mode: bool = False,
        temperature: float = 0.3,
        max_tokens: int = 500,
    ) -> str:
        """
        Get a response from Groq API.

        Args:
            system_prompt: Injected as the system message
            messages: Conversation history
            json_mode: If True, enforces JSON output
            temperature: Sampling temperature
            max_tokens: Maximum completion tokens

        Returns:
            Response text string
        """
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
            logger.debug(f"Groq response ({len(content)} chars): {content[:100]}...")
            return content
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            raise
