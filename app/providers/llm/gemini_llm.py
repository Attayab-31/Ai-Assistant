"""Google Gemini LLM provider.

Uses Gemini's OpenAI-compatible endpoint so we can reuse the ``openai`` SDK
(no extra dependency). Gemini 2.5 Flash models apply *implicit* prompt caching
automatically, which discounts the large repeated system prompt, and the
Google AI Studio free tier has a more generous quota than Groq's daily wall.
"""

import logging
import re

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

# Floor for JSON responses. Gemini tends to be verbose and the screening
# contract (response_text + extracted_data + flags) can run long; too small a
# budget truncates the JSON mid-string. Generous here is cheap because thinking
# is disabled (see reasoning_effort below).
_JSON_MIN_MAX_TOKENS = 768


def _extract_json(text: str) -> str:
    """Best-effort salvage of a single JSON object from a model reply.

    Strips ```json fences and any prose around the object so the caller's
    json.loads never trips on stray text. Returns the original string when no
    object is found (so the caller still sees a meaningful parse error).
    """
    if not text:
        return text
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s).strip()
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start : end + 1].strip()
    return s


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

        # Gemini 2.5 models "think" by default, and on the OpenAI-compatible
        # endpoint those reasoning tokens are billed against max_tokens. With a
        # small budget they consume it entirely and the body comes back empty or
        # truncated (the JSON parse failures seen in production). Disabling
        # thinking returns the budget to the actual answer and slashes latency.
        effective_max = max(max_tokens, _JSON_MIN_MAX_TOKENS) if json_mode else max_tokens
        kwargs = {
            "model": self.model,
            "messages": all_messages,
            "temperature": temperature,
            "max_tokens": effective_max,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        # "none" is only valid for 2.5 models; older flash models aren't thinking
        # models, so we skip it there to avoid a 400.
        if self.model.startswith("gemini-2.5"):
            kwargs["reasoning_effort"] = "none"

        try:
            response = await self.client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content or ""
            if json_mode:
                content = _extract_json(content)
            logger.debug(f"Gemini response: {content[:100]}...")
            return content
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise
