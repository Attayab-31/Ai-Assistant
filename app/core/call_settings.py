"""
Per-call settings snapshot and provider bundle.

Settings and provider instances are frozen when a call session starts so
admin changes apply to NEW calls only — never mid-call.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.question_flow import coerce_questions_for_runtime_with_reason
from app.core.screening_flow import normalize_faqs
from app.core.voice_latency import resolve_voice_latency
from config import DEFAULT_FAQS
from config import settings as env_settings

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
    "tts_voice_deepgram_es",
    "tts_voice_google_es",
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
    "latency_alert_turn_p95_ms",
    "latency_alert_turn_p95_crit_ms",
    "latency_alert_timeout_rate_pct",
    "latency_alert_timeout_rate_crit_pct",
    "llm_streaming_enabled",
    "email_notifications_enabled",
    "landlord_email",
    "email_from_name",
    "email_from_address",
    "email_subject_template",
    "email_body_template",
    "email_qualified_only",
    "email_include_transcript",
    "cc_emails",
    "bcc_emails",
    "timezone",
    "crm_webhook_url",
    "crm_webhook_secret",
    "crm_notifications_enabled",
)

# Redis cache for the raw settings batch behind a call snapshot. A short TTL
# bounds memory and staleness; admin writes also invalidate it immediately so
# changes still apply to new calls right away. This spares the DB from one
# batch query per call when many calls start at once.
CALL_SETTINGS_SNAPSHOT_KEY = "call_settings:batch:v2"
CALL_SETTINGS_SNAPSHOT_TTL = 30

NOTIFICATION_SETTINGS_KEYS = (
    "email_notifications_enabled",
    "landlord_email",
    "email_from_name",
    "email_from_address",
    "email_subject_template",
    "email_body_template",
    "email_qualified_only",
    "email_include_transcript",
    "cc_emails",
    "bcc_emails",
    "timezone",
    "crm_webhook_url",
    "crm_webhook_secret",
    "crm_notifications_enabled",
)

ENCRYPTED_PROVIDER_KEY_SETTINGS = (
    "groq_api_key_encrypted",
    "openai_api_key_encrypted",
    "openrouter_api_key_encrypted",
    "gemini_api_key_encrypted",
    "deepgram_api_key_encrypted",
)

_ENCRYPTED_TO_PROVIDER = {
    "groq_api_key_encrypted": "groq",
    "openai_api_key_encrypted": "openai",
    "openrouter_api_key_encrypted": "openrouter",
    "gemini_api_key_encrypted": "gemini",
    "deepgram_api_key_encrypted": "deepgram",
}


@dataclass(frozen=True)
class ProviderApiKeys:
    """API credentials frozen at call start (immune to mid-call admin rotation)."""

    groq: str = ""
    openai: str = ""
    openrouter: str = ""
    gemini: str = ""
    deepgram: str = ""
    google_application_credentials: str = ""

    def configured(self, provider: str) -> bool:
        """True when this provider had a usable credential at call start."""
        name = provider.lower().strip()
        if name == "google":
            return bool(self.google_application_credentials.strip())
        return bool(getattr(self, name, "") or "")


def capture_provider_api_keys_from_settings() -> ProviderApiKeys:
    """Snapshot provider credentials from in-memory settings."""
    return ProviderApiKeys(
        groq=(env_settings.groq_api_key or "").strip(),
        openai=(env_settings.openai_api_key or "").strip(),
        openrouter=(env_settings.openrouter_api_key or "").strip(),
        gemini=(env_settings.gemini_api_key or "").strip(),
        deepgram=(env_settings.deepgram_api_key or "").strip(),
        google_application_credentials=(
            env_settings.google_application_credentials or ""
        ).strip(),
    )


async def capture_provider_api_keys(db: AsyncSession) -> ProviderApiKeys:
    """Snapshot provider credentials for one call using DB keys first.

    Reads encrypted API keys from ``system_settings`` so NEW calls pick up a
    freshly-rotated key even if provider-registry reload failed. Falls back to
    env-backed in-memory settings when keys are missing or undecryptable.
    """
    from app.db.crud import fetch_settings_batch
    from app.utils.security import decrypt_value

    keys = capture_provider_api_keys_from_settings()
    values = await fetch_settings_batch(db, ENCRYPTED_PROVIDER_KEY_SETTINGS)
    resolved: dict[str, str] = {
        "groq": keys.groq,
        "openai": keys.openai,
        "openrouter": keys.openrouter,
        "gemini": keys.gemini,
        "deepgram": keys.deepgram,
    }
    for setting_key, provider in _ENCRYPTED_TO_PROVIDER.items():
        raw = values.get(setting_key)
        if not raw:
            continue
        try:
            decrypted = decrypt_value(str(raw)).strip()
        except Exception:
            decrypted = ""
        if decrypted:
            resolved[provider] = decrypted
    return ProviderApiKeys(
        groq=resolved["groq"],
        openai=resolved["openai"],
        openrouter=resolved["openrouter"],
        gemini=resolved["gemini"],
        deepgram=resolved["deepgram"],
        google_application_credentials=keys.google_application_credentials,
    )


async def provider_key_configured_map(db: AsyncSession) -> dict[str, bool]:
    """Which providers have usable credentials (env or DB), for admin UI/health."""
    keys = await capture_provider_api_keys(db)
    return {
        "groq": keys.configured("groq"),
        "openai": keys.configured("openai"),
        "openrouter": keys.configured("openrouter"),
        "gemini": keys.configured("gemini"),
        "deepgram": keys.configured("deepgram"),
    }


@dataclass(frozen=True)
class NotificationSettingsSnapshot:
    """Email + CRM notification settings frozen at call start."""

    email_notifications_enabled: bool = True
    landlord_email: str = ""
    email_from_name: str = ""
    email_from_address: str = ""
    cc_emails: str = ""
    bcc_emails: str = ""
    email_qualified_only: bool = False
    email_include_transcript: bool = False
    email_subject_template: str = ""
    email_body_template: str = ""
    timezone: str = ""
    crm_webhook_url: str = ""
    crm_webhook_secret: str = ""
    crm_notifications_enabled: bool = False


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
    latency_alert_turn_p95_crit_ms: int = 1800
    latency_alert_timeout_rate_pct: float = 2.0
    latency_alert_timeout_rate_crit_pct: float = 5.0
    llm_models_by_provider: dict[str, str] = field(default_factory=dict)
    tts_voices_by_provider: dict[str, str] = field(default_factory=dict)
    tts_voice_deepgram_es: str = "aura-2-estrella-es"
    tts_voice_google_es: str = "es-US-Neural2-A"
    notification_settings: NotificationSettingsSnapshot = field(
        default_factory=NotificationSettingsSnapshot
    )
    questions_runtime_fallback: str | None = None
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
    stt_models_by_provider: dict[str, str] = field(default_factory=dict)
    api_keys: ProviderApiKeys = field(default_factory=ProviderApiKeys)


def stt_model_for_provider(providers: CallProviderBundle, provider_name: str) -> str:
    """Return the STT model frozen for this call (or a safe registry default)."""
    name = provider_name.lower().strip()
    model = (providers.stt_models_by_provider or {}).get(name)
    if model:
        return str(model)
    if providers.stt_name == name:
        primary = getattr(providers.stt, "model", None)
        if primary:
            return str(primary)
    from config import settings

    if name == "deepgram":
        return settings.deepgram_model
    return "whisper-large-v3-turbo"


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
    if key in (
        "email_notifications_enabled",
        "email_qualified_only",
        "email_include_transcript",
        "crm_notifications_enabled",
    ):
        return str(raw).lower() in ("true", "1", "yes")
    if key in ("tts_speed", "llm_temperature"):
        try:
            return float(raw)
        except (ValueError, TypeError):
            return default
    return raw


def notification_settings_from_map(
    values: dict[str, Any],
) -> NotificationSettingsSnapshot:
    """Build frozen post-call notification settings from a settings map."""
    from app.utils.security import decrypt_value

    raw_crm_secret = str(values.get("crm_webhook_secret") or "")
    crm_secret = raw_crm_secret
    if raw_crm_secret:
        try:
            crm_secret = decrypt_value(raw_crm_secret).strip()
        except Exception:
            crm_secret = raw_crm_secret

    return NotificationSettingsSnapshot(
        email_notifications_enabled=bool(
            _parse_setting(
                "email_notifications_enabled",
                values.get("email_notifications_enabled"),
                True,
            )
        ),
        landlord_email=str(
            values.get("landlord_email") or env_settings.default_landlord_email or ""
        ),
        email_from_name=str(
            values.get("email_from_name") or env_settings.email_from_name or ""
        ),
        email_from_address=str(
            values.get("email_from_address") or env_settings.email_from or ""
        ),
        cc_emails=str(values.get("cc_emails") or ""),
        bcc_emails=str(values.get("bcc_emails") or ""),
        email_qualified_only=bool(
            _parse_setting(
                "email_qualified_only",
                values.get("email_qualified_only"),
                False,
            )
        ),
        email_include_transcript=bool(
            _parse_setting(
                "email_include_transcript",
                values.get("email_include_transcript"),
                False,
            )
        ),
        email_subject_template=str(values.get("email_subject_template") or ""),
        email_body_template=str(values.get("email_body_template") or ""),
        timezone=str(values.get("timezone") or ""),
        crm_webhook_url=(crm_url := str(values.get("crm_webhook_url") or "").strip()),
        crm_webhook_secret=crm_secret,
        crm_notifications_enabled=(
            bool(
                _parse_setting(
                    "crm_notifications_enabled",
                    values.get("crm_notifications_enabled"),
                    False,
                )
            )
            if "crm_notifications_enabled" in values
            else bool(crm_url)
        ),
    )


def crm_notifications_active(snapshot: NotificationSettingsSnapshot) -> bool:
    """True when post-call CRM delivery is enabled and a URL is configured."""
    return snapshot.crm_notifications_enabled and bool(
        snapshot.crm_webhook_url.strip()
    )


def notification_settings_email_dict(
    snapshot: NotificationSettingsSnapshot,
) -> dict[str, Any]:
    """Dict shape consumed by email_service Celery tasks."""
    return {
        "landlord_email": snapshot.landlord_email,
        "email_from_name": snapshot.email_from_name,
        "email_from_address": snapshot.email_from_address,
        "email_subject_template": snapshot.email_subject_template,
        "email_body_template": snapshot.email_body_template,
        "email_qualified_only": snapshot.email_qualified_only,
        "email_include_transcript": snapshot.email_include_transcript,
        "cc_emails": snapshot.cc_emails,
        "bcc_emails": snapshot.bcc_emails,
        "timezone": snapshot.timezone,
        "crm_webhook_secret": snapshot.crm_webhook_secret,
    }


async def load_notification_settings_from_db(
    db: AsyncSession,
) -> NotificationSettingsSnapshot:
    """Load runtime notification settings without masking sensitive keys."""
    from app.db.crud import fetch_settings_batch

    values = await fetch_settings_batch(db, NOTIFICATION_SETTINGS_KEYS)
    return notification_settings_from_map(values)


NOTIFICATION_SETTINGS_PERSIST_KEY = "notification_settings"

_EMAIL_SETTINGS_KEYS = (
    "landlord_email",
    "email_from_name",
    "email_from_address",
    "email_subject_template",
    "email_body_template",
    "email_qualified_only",
    "email_include_transcript",
    "cc_emails",
    "bcc_emails",
    "timezone",
    "crm_webhook_secret",
)


def notification_settings_persist_dict(
    snapshot: NotificationSettingsSnapshot,
) -> dict[str, Any]:
    """Persist email/CRM notification settings on the tenant for later resend."""
    out = notification_settings_email_dict(snapshot)
    out["email_notifications_enabled"] = snapshot.email_notifications_enabled
    out["crm_webhook_url"] = snapshot.crm_webhook_url
    out["crm_notifications_enabled"] = snapshot.crm_notifications_enabled
    return out


def notification_settings_email_dict_from_tenant(
    tenant: Any | None,
) -> dict[str, Any] | None:
    """Return frozen email settings saved at call finalize, if present."""
    if tenant is None or not isinstance(getattr(tenant, "normalized_data", None), dict):
        return None
    raw = tenant.normalized_data.get(NOTIFICATION_SETTINGS_PERSIST_KEY)
    if not isinstance(raw, dict) or not raw:
        return None
    if not any(k in raw for k in _EMAIL_SETTINGS_KEYS):
        return None
    return {k: raw.get(k, "") for k in _EMAIL_SETTINGS_KEYS}


def has_notification_settings_snapshot(tenant: Any | None) -> bool:
    """True when tenant.normalized_data carries frozen notification settings."""
    return notification_settings_email_dict_from_tenant(tenant) is not None


def _parse_screening_questions_for_runtime(raw_val: Any) -> tuple[list[dict[str, Any]] | None, str | None]:
    """Parse stored screening_questions without injecting install-time defaults."""
    if raw_val is None:
        return None, None
    if isinstance(raw_val, list):
        return raw_val, None
    if isinstance(raw_val, str):
        try:
            parsed = json.loads(raw_val)
        except (json.JSONDecodeError, TypeError):
            return None, "Screening questions setting is not valid JSON."
        if not isinstance(parsed, list):
            return None, "Screening questions setting must be a JSON array."
        return parsed, None
    return None, "Screening questions setting has an unsupported format."


def _load_runtime_questions(values: dict[str, Any]) -> tuple[list, str | None]:
    """Parse and validate screening questions for a new call (admin source of truth)."""
    parsed, parse_error = _parse_screening_questions_for_runtime(
        values.get("screening_questions")
    )
    if parse_error:
        import logging

        logging.getLogger(__name__).error(
            "Invalid screening_questions in settings (%s) — blocking live calls",
            parse_error,
        )
        return [], parse_error
    return coerce_questions_for_runtime_with_reason(parsed)


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
    tts_voice_deepgram_es = str(
        values.get("tts_voice_deepgram_es") or env_settings.tts_voice_deepgram_es
    )
    tts_voice_google_es = str(
        values.get("tts_voice_google_es") or env_settings.tts_voice_google_es
    )

    runtime_questions, questions_runtime_fallback = _load_runtime_questions(values)

    return CallSettingsSnapshot(
        llm_provider=llm.lower(),
        stt_provider=stt.lower(),
        tts_provider=tts.lower(),
        llm_model=str(model_by_llm.get(llm.lower(), env_settings.active_groq_model)),
        stt_model=str(values.get("deepgram_model") or env_settings.deepgram_model),
        groq_stt_model=str(values.get("groq_stt_model") or "whisper-large-v3-turbo"),
        tts_voice=str(voice_by_tts.get(tts.lower(), env_settings.tts_voice_deepgram)),
        tts_voice_deepgram_es=tts_voice_deepgram_es,
        tts_voice_google_es=tts_voice_google_es,
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
        questions=runtime_questions,
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
        notification_settings=notification_settings_from_map(values),
        questions_runtime_fallback=questions_runtime_fallback,
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
                "latency_alert_turn_p95_crit_ms",
                "latency_alert_timeout_rate_pct",
                "latency_alert_timeout_rate_crit_pct",
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


def build_call_provider_bundle(
    snapshot: CallSettingsSnapshot,
    *,
    api_keys: ProviderApiKeys | None = None,
) -> CallProviderBundle:
    """Construct isolated provider instances for one call session."""
    from app.providers.llm.gemini_llm import GeminiLLMProvider
    from app.providers.llm.groq_llm import GroqLLMProvider
    from app.providers.llm.openai_llm import OpenAILLMProvider
    from app.providers.llm.openrouter_llm import OpenRouterLLMProvider
    from app.providers.stt.deepgram_stt import DeepgramSTTProvider
    from app.providers.stt.groq_stt import GroqSTTProvider
    from app.providers.tts.deepgram_tts import DeepgramTTSProvider
    from app.providers.tts.google_tts import GoogleTTSProvider

    api_keys = api_keys or capture_provider_api_keys_from_settings()

    llm_factories = {
        "groq": lambda m: GroqLLMProvider(model=m, api_key=api_keys.groq),
        "openai": lambda m: OpenAILLMProvider(model=m, api_key=api_keys.openai),
        "openrouter": lambda m: OpenRouterLLMProvider(
            model=m, api_key=api_keys.openrouter
        ),
        "gemini": lambda m: GeminiLLMProvider(model=m, api_key=api_keys.gemini),
    }
    stt_factories = {
        "deepgram": lambda: DeepgramSTTProvider(
            model=snapshot.stt_model, api_key=api_keys.deepgram
        ),
        "groq": lambda: GroqSTTProvider(
            model=snapshot.groq_stt_model, api_key=api_keys.groq
        ),
    }
    tts_factories = {
        "google": lambda v: GoogleTTSProvider(
            voice=v,
            google_application_credentials=api_keys.google_application_credentials,
        ),
        "deepgram": lambda v: DeepgramTTSProvider(voice=v, api_key=api_keys.deepgram),
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
        stt_models_by_provider={
            "deepgram": snapshot.stt_model,
            "groq": snapshot.groq_stt_model,
        },
        api_keys=api_keys,
    )
