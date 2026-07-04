"""
Per-call settings snapshot and provider bundle.

Settings and provider instances are frozen when a call session starts so
admin changes apply to NEW calls only — never mid-call.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.question_flow import normalize_questions
from app.core.screening_flow import normalize_faqs
from app.core.voice_latency import resolve_voice_latency
from config import DEFAULT_FAQS, DEFAULT_QUESTIONS
from config import settings as env_settings

logger = logging.getLogger(__name__)

CALL_SETTINGS_KEYS = (
    "active_llm_provider",
    "active_stt_provider",
    "active_tts_provider",
    "active_groq_model",
    "active_openai_model",
    "active_openrouter_model",
    "active_gemini_model",
    "deepgram_model",
    "groq_stt_model",
    "tts_voice_google",
    "tts_voice_deepgram",
    "tts_speed",
    "auto_fallback_enabled",
    "llm_fallback_provider",
    "stt_fallback_provider",
    "tts_fallback_provider",
    "property_name",
    "greeting_message",
    "closing_message",
    "provider_failure_message",
    "screening_questions",
    "screening_faqs",
    "max_retries_per_question",
    "silence_timeout_seconds",
    "max_call_duration_seconds",
    "llm_temperature",
    "llm_max_tokens",
    "qualified_score_threshold",
    "review_score_threshold",
    "voice_latency_profile",
    "llm_streaming_enabled",
)

# Redis cache for the raw settings batch behind a call snapshot. A short TTL
# bounds memory and staleness; admin writes also invalidate it immediately so
# changes still apply to new calls right away. This spares the DB from one
# batch query per call when many calls start at once.
CALL_SETTINGS_SNAPSHOT_KEY = "call_settings:batch:v1"
CALL_SETTINGS_SNAPSHOT_TTL = 30


@dataclass(frozen=True)
class CallSettingsSnapshot:
    """Immutable settings captured at call start."""

    llm_provider: str
    stt_provider: str
    tts_provider: str
    llm_model: str
    stt_model: str
    groq_stt_model: str
    tts_voice: str
    tts_speed: float
    auto_fallback_enabled: bool
    llm_fallback_provider: str
    stt_fallback_provider: str
    tts_fallback_provider: str
    property_name: str
    greeting_message: str
    closing_message: str
    provider_failure_message: str
    questions: list
    faqs: list
    max_retries: int
    silence_timeout_seconds: int
    max_call_duration_seconds: int
    llm_temperature: float
    llm_max_tokens: int
    qualified_score_threshold: int = 75
    review_score_threshold: int = 40
    voice_latency_profile: str = "balanced"
    llm_streaming_enabled: bool = True
    turn_timeout_seconds: float = 15.0
    llm_timeout_voice_seconds: float = 5.5
    deepgram_endpointing_ms: int = 900
    deepgram_utterance_end_ms: int = 1000
    latency_alert_turn_p95_ms: int = 1200
    latency_alert_timeout_rate_pct: float = 2.0
    llm_models_by_provider: dict[str, str] = field(default_factory=dict)
    tts_voices_by_provider: dict[str, str] = field(default_factory=dict)
    captured_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@dataclass
class CallProviderBundle:
    """Provider instances bound to a single call (not shared with registry)."""

    llm: Any
    stt: Any
    tts: Any
    llm_name: str
    stt_name: str
    tts_name: str
    auto_fallback_enabled: bool
    tts_speed: float = 1.0
    llm_fallback_provider: str = "auto"
    stt_fallback_provider: str = "auto"
    tts_fallback_provider: str = "auto"
    llm_by_name: dict = field(default_factory=dict)
    tts_by_name: dict = field(default_factory=dict)


def _parse_setting(key: str, raw: Any, default: Any) -> Any:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    if key in ("screening_questions", "screening_faqs"):
        if isinstance(raw, list):
            return raw
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return default
    if key in (
        "max_retries_per_question",
        "silence_timeout_seconds",
        "max_call_duration_seconds",
        "llm_max_tokens",
        "qualified_score_threshold",
        "review_score_threshold",
    ):
        try:
            return int(raw)
        except (ValueError, TypeError):
            return default
    if key == "auto_fallback_enabled":
        return str(raw).lower() in ("true", "1", "yes")
    if key in ("tts_speed", "llm_temperature"):
        try:
            return float(raw)
        except (ValueError, TypeError):
            return default
    return raw


def snapshot_from_map(values: dict[str, Any]) -> CallSettingsSnapshot:
    """Build snapshot from a flat key→value map (DB batch or env defaults)."""
    llm = str(values.get("active_llm_provider") or env_settings.active_llm_provider)
    stt = str(values.get("active_stt_provider") or env_settings.active_stt_provider)
    tts = str(values.get("active_tts_provider") or env_settings.active_tts_provider)

    model_by_llm = {
        "groq": values.get("active_groq_model") or env_settings.active_groq_model,
        "openai": values.get("active_openai_model") or env_settings.active_openai_model,
        "openrouter": values.get("active_openrouter_model")
        or env_settings.active_openrouter_model,
        "gemini": values.get("active_gemini_model")
        or env_settings.active_gemini_model,
    }
    voice_by_tts = {
        "google": values.get("tts_voice_google") or env_settings.tts_voice_google,
        "deepgram": values.get("tts_voice_deepgram") or env_settings.tts_voice_deepgram,
    }

    return CallSettingsSnapshot(
        llm_provider=llm.lower(),
        stt_provider=stt.lower(),
        tts_provider=tts.lower(),
        llm_model=str(model_by_llm.get(llm.lower(), env_settings.active_groq_model)),
        stt_model=str(values.get("deepgram_model") or env_settings.deepgram_model),
        groq_stt_model=str(values.get("groq_stt_model") or "whisper-large-v3-turbo"),
        tts_voice=str(voice_by_tts.get(tts.lower(), env_settings.tts_voice_deepgram)),
        tts_speed=_parse_setting("tts_speed", values.get("tts_speed"), 1.0),
        auto_fallback_enabled=_parse_setting(
            "auto_fallback_enabled", values.get("auto_fallback_enabled"), True
        ),
        llm_fallback_provider=str(values.get("llm_fallback_provider") or "auto").lower(),
        stt_fallback_provider=str(values.get("stt_fallback_provider") or "auto").lower(),
        tts_fallback_provider=str(values.get("tts_fallback_provider") or "auto").lower(),
        property_name=str(
            values.get("property_name") or env_settings.default_property_name
        ),
        greeting_message=str(values.get("greeting_message") or "").strip(),
        closing_message=str(values.get("closing_message") or "").strip(),
        provider_failure_message=str(
            values.get("provider_failure_message") or ""
        ).strip(),
        questions=normalize_questions(
            _parse_setting(
                "screening_questions",
                values.get("screening_questions"),
                DEFAULT_QUESTIONS,
            )
        ),
        faqs=normalize_faqs(
            _parse_setting(
                "screening_faqs",
                values.get("screening_faqs"),
                DEFAULT_FAQS,
            )
        ),
        max_retries=_parse_setting(
            "max_retries_per_question", values.get("max_retries_per_question"), 2
        ),
        silence_timeout_seconds=_parse_setting(
            "silence_timeout_seconds", values.get("silence_timeout_seconds"), 12
        ),
        max_call_duration_seconds=_parse_setting(
            "max_call_duration_seconds", values.get("max_call_duration_seconds"), 600
        ),
        llm_temperature=_parse_setting(
            "llm_temperature", values.get("llm_temperature"), 0.3
        ),
        llm_max_tokens=_parse_setting(
            "llm_max_tokens", values.get("llm_max_tokens"), 0
        ),
        qualified_score_threshold=_parse_setting(
            "qualified_score_threshold", values.get("qualified_score_threshold"), 75
        ),
        review_score_threshold=_parse_setting(
            "review_score_threshold", values.get("review_score_threshold"), 40
        ),
        llm_models_by_provider={k: str(v) for k, v in model_by_llm.items()},
        tts_voices_by_provider={k: str(v) for k, v in voice_by_tts.items()},
        **{
            k: v
            for k, v in resolve_voice_latency(values).items()
            if k
            in (
                "voice_latency_profile",
                "llm_streaming_enabled",
                "turn_timeout_seconds",
                "llm_timeout_voice_seconds",
                "deepgram_endpointing_ms",
                "deepgram_utterance_end_ms",
                "latency_alert_turn_p95_ms",
                "latency_alert_timeout_rate_pct",
            )
        },
    )


async def load_call_settings_snapshot(db: AsyncSession) -> CallSettingsSnapshot:
    """Load all call-relevant settings, served from Redis when warm.

    Falls back to a single DB batch query on a cache miss (or if Redis is
    down), then repopulates the cache. Behavior is identical either way — the
    cache only removes redundant DB reads when calls start in bursts.
    """
    from app.core.redis_client import cache_get_json, cache_set_json
    from app.db.crud import fetch_settings_batch

    cached = await cache_get_json(CALL_SETTINGS_SNAPSHOT_KEY)
    if isinstance(cached, dict):
        return snapshot_from_map(cached)

    values = await fetch_settings_batch(db, CALL_SETTINGS_KEYS)
    await cache_set_json(CALL_SETTINGS_SNAPSHOT_KEY, values, CALL_SETTINGS_SNAPSHOT_TTL)
    return snapshot_from_map(values)


def build_call_provider_bundle(snapshot: CallSettingsSnapshot) -> CallProviderBundle:
    """Construct isolated provider instances for one call session."""
    from app.providers.llm.gemini_llm import GeminiLLMProvider
    from app.providers.llm.groq_llm import GroqLLMProvider
    from app.providers.llm.openai_llm import OpenAILLMProvider
    from app.providers.llm.openrouter_llm import OpenRouterLLMProvider
    from app.providers.stt.deepgram_stt import DeepgramSTTProvider
    from app.providers.stt.groq_stt import GroqSTTProvider
    from app.providers.tts.deepgram_tts import DeepgramTTSProvider
    from app.providers.tts.google_tts import GoogleTTSProvider

    llm_factories = {
        "groq": lambda m: GroqLLMProvider(model=m),
        "openai": lambda m: OpenAILLMProvider(model=m),
        "openrouter": lambda m: OpenRouterLLMProvider(model=m),
        "gemini": lambda m: GeminiLLMProvider(model=m),
    }
    stt_factories = {
        "deepgram": lambda: DeepgramSTTProvider(model=snapshot.stt_model),
        "groq": lambda: GroqSTTProvider(model=snapshot.groq_stt_model),
    }
    tts_factories = {
        "google": lambda v: GoogleTTSProvider(voice=v),
        "deepgram": lambda v: DeepgramTTSProvider(voice=v),
    }

    llm_name = snapshot.llm_provider
    stt_name = snapshot.stt_provider
    tts_name = snapshot.tts_provider

    if llm_name not in llm_factories:
        raise ValueError(f"Unknown LLM provider: {llm_name}")
    if stt_name not in stt_factories:
        raise ValueError(f"Unknown STT provider: {stt_name}")
    if tts_name not in tts_factories:
        raise ValueError(f"Unknown TTS provider: {tts_name}")

    llm_models = snapshot.llm_models_by_provider
    tts_voices = snapshot.tts_voices_by_provider
    llm_by_name = {
        name: factory(str(llm_models.get(name, snapshot.llm_model)))
        for name, factory in llm_factories.items()
    }
    tts_by_name = {
        name: factory(str(tts_voices.get(name, snapshot.tts_voice)))
        for name, factory in tts_factories.items()
    }

    return CallProviderBundle(
        llm=llm_by_name[llm_name],
        stt=stt_factories[stt_name](),
        tts=tts_by_name[tts_name],
        llm_name=llm_name,
        stt_name=stt_name,
        tts_name=tts_name,
        auto_fallback_enabled=snapshot.auto_fallback_enabled,
        tts_speed=snapshot.tts_speed,
        llm_fallback_provider=snapshot.llm_fallback_provider,
        stt_fallback_provider=snapshot.stt_fallback_provider,
        tts_fallback_provider=snapshot.tts_fallback_provider,
        llm_by_name=llm_by_name,
        tts_by_name=tts_by_name,
    )
