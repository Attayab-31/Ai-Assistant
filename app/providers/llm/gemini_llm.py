"""Google Gemini LLM provider.

Uses Gemini's OpenAI-compatible endpoint so we can reuse the ``openai`` SDK
(no extra dependency). Gemini 2.5 Flash models apply *implicit* prompt caching
automatically, which discounts the large repeated system prompt, and the
Google AI Studio free tier has a more generous quota than Groq's daily wall.
"""

import logging
import re
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
_VOICE_JSON_MIN_MAX_TOKENS = 256


def extract_json_from_llm_text(text: str) -> str:
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


# Registry / call_handler import the private name from earlier releases.
_extract_json = extract_json_from_llm_text


class GeminiLLMProvider(BaseLLMProvider):
    """Google Gemini provider via its OpenAI-compatible chat completions API."""

    provider_name = "gemini"

    def __init__(self, model: str = "gemini-2.5-flash", *, api_key: str | None = None) -> None:
        self.model = model if model in AVAILABLE_MODELS else AVAILABLE_MODELS[0]
        self._api_key = api_key
        self._client: AsyncOpenAI | None = None
        logger.info(f"GeminiLLMProvider initialized with model: {self.model}")

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            key = resolve_frozen_credential(self._api_key, settings_attr="gemini_api_key")
            if not key:
                raise ValueError("GEMINI_API_KEY not set in environment")
            self._client = AsyncOpenAI(
                api_key=key,
                base_url=GEMINI_BASE_URL,
            )
        return self._client

    def _gemini_request_extra(
        self, *, json_mode: bool, max_tokens: int
    ) -> tuple[int, dict]:
        if json_mode:
            floor = (
                _VOICE_JSON_MIN_MAX_TOKENS
                if max_tokens <= 200
                else _JSON_MIN_MAX_TOKENS
            )
            effective_max = max(max_tokens, floor)
        else:
            effective_max = max_tokens
        extra: dict = {}
        # Disable Gemini 2.5 "thinking" so reasoning tokens don't eat the budget.
        if self.model.startswith("gemini-2.5"):
            extra["extra_body"] = {"reasoning_effort": "none"}
        return effective_max, extra

    async def get_response(
        self,
        system_prompt: str,
        messages: list[dict],
        json_mode: bool = False,
        temperature: float = 0.3,
        max_tokens: int = 500,
    ) -> str:
        effective_max, extra = self._gemini_request_extra(
            json_mode=json_mode, max_tokens=max_tokens
        )
        kwargs = openai_chat_kwargs(
            model=self.model,
            system_prompt=system_prompt,
            messages=messages,
            json_mode=json_mode,
            temperature=temperature,
            max_tokens=effective_max,
            extra=extra,
        )
        try:
            content = await complete_openai_chat(self, self.client, kwargs)
            if json_mode:
                content = extract_json_from_llm_text(content)
            logger.debug(f"Gemini response: {content[:100]}...")
            return content
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise

    async def stream_response(
        self,
        system_prompt: str,
        messages: list[dict],
        json_mode: bool = False,
        temperature: float = 0.3,
        max_tokens: int = 500,
    ) -> AsyncIterator[str]:
        """Yield text deltas from a streaming Gemini completion."""
        effective_max, extra = self._gemini_request_extra(
            json_mode=json_mode, max_tokens=max_tokens
        )
        kwargs = openai_chat_kwargs(
            model=self.model,
            system_prompt=system_prompt,
            messages=messages,
            json_mode=json_mode,
            temperature=temperature,
            max_tokens=effective_max,
            stream=True,
            extra=extra,
        )
        async for delta in stream_openai_chat(self, self.client, kwargs):
            yield delta
