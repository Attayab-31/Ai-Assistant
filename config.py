"""
config.py — Application configuration, environment settings, and ProviderRegistry.

Loads all settings from environment variables (via .env), defines the
ProviderRegistry singleton for hot-swapping AI providers at runtime, and
exposes the DEFAULT_QUESTIONS for initial database seeding.
"""

import asyncio
import logging
import ssl
from typing import Optional

from dotenv import load_dotenv
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.seed_data import load_seed_faqs, load_seed_questions

load_dotenv(override=False)

DEFAULT_QUESTIONS = load_seed_questions()
DEFAULT_FAQS = load_seed_faqs()
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Application Settings
# ──────────────────────────────────────────────────────────────────────────────


class Settings(BaseSettings):
    """Typed settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # App
    app_name: str = "AI Tenant Screener"
    secret_key: str = "insecure-dev-key-change-in-production"
    admin_email: str = "admin@example.com"
    admin_password: str = "Admin123!"
    app_url: str = "http://localhost:8000"
    environment: str = "development"
    debug: bool = False
    log_dir: str = "./logs"  # Directory for rotating log files (production)
    # Dev: write logs/voice.trace.log and show [call_id|phase] on console.
    log_voice_trace: bool = True
    # Live call sessions are in-process; keep at 1 until Redis session store exists.
    web_workers: int = 1
    enable_test_console: bool = False
    # Security default: unsigned Telnyx webhooks are rejected unless explicitly
    # opted in for local development debugging.
    allow_unsigned_webhooks_in_dev: bool = False
    # Comma-separated peer IPs allowed to set X-Forwarded-For / X-Real-IP (e.g.
    # loopback when TLS terminates on the same host). Empty = never trust those
    # headers; rate limits and auth lockouts use the direct TCP peer only.
    trusted_proxy_ips: str = ""

    @field_validator("debug", mode="before")
    @classmethod
    def parse_debug(cls, value):
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"release", "prod", "production", "off"}:
                return False
            if normalized in {"debug", "dev", "development", "on"}:
                return True
        return value

    # Database
    database_url: str = (
        "postgresql+asyncpg://postgres:postgres@localhost:5432/tenant_screener"
    )

    @field_validator("database_url", mode="before")
    @classmethod
    def validate_database_url(cls, value):
        if isinstance(value, str) and value.strip().lower().startswith(
            ("http://", "https://")
        ):
            raise ValueError(
                "DATABASE_URL must be a Postgres connection string, not the "
                "Supabase project API URL. Use Supabase Dashboard > Connect "
                "to get a postgresql://... URL."
            )
        if isinstance(value, str) and value.strip():
            from sqlalchemy.engine import make_url

            try:
                url = make_url(value)
            except Exception as exc:
                raise ValueError(
                    "DATABASE_URL is not a valid SQLAlchemy database URL. "
                    "If the password contains special characters, URL-encode "
                    "them first."
                ) from exc

            if url.host and "@" in url.host:
                raise ValueError(
                    "DATABASE_URL appears to contain an unescaped '@' in the "
                    "username or password. URL-encode credentials before adding "
                    "them to the connection string, for example '@' becomes '%40'."
                )
        return value

    database_ssl: str = (
        "auto"  # auto, disable, allow, prefer, require, verify-ca, verify-full
    )
    database_pool_mode: str = "auto"  # auto, direct, session, transaction
    database_migration_mode: str = "auto"  # auto, upgrade, check, create_all, skip
    database_pool_size: int = 5
    database_max_overflow: int = 10
    database_pool_recycle_seconds: int = 1800
    database_connect_timeout_seconds: int = 10

    # Supabase
    supabase_url: str = ""
    supabase_secret_key: str = ""
    supabase_storage_bucket: str = "call-recordings"

    # Redis (separate databases for different purposes to avoid memory conflicts)
    # Cache & settings (analytics, configs)
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/1"  # Celery task queue
    celery_result_backend: str = "redis://localhost:6379/2"  # Celery result storage

    # Telephony
    telnyx_api_key: str = ""
    telnyx_public_key: str = ""

    # STT
    deepgram_api_key: str = ""

    # LLM
    groq_api_key: str = ""
    openai_api_key: str = ""
    openrouter_api_key: str = ""
    gemini_api_key: str = ""

    # TTS
    google_application_credentials: str = ""

    # Email
    resend_api_key: str = ""
    email_from: str = "screening@example.com"
    email_from_name: str = "AI Screening Platform"

    # Encryption
    encryption_key: str = ""

    # Defaults
    default_property_name: str = "Ready Rentals Online"
    default_landlord_email: str = ""

    # Active providers (used as env-backed defaults for DB runtime settings)
    active_llm_provider: str = "groq"
    active_stt_provider: str = "deepgram"
    active_tts_provider: str = "deepgram"
    active_groq_model: str = "llama-3.3-70b-versatile"
    active_openai_model: str = "gpt-4o-mini"
    active_openrouter_model: str = "meta-llama/llama-3.3-70b-instruct:free"
    active_gemini_model: str = "gemini-2.5-flash"
    deepgram_model: str = "nova-3"
    groq_stt_model: str = "whisper-large-v3-turbo"
    tts_voice_google: str = "en-US-Wavenet-D"
    tts_voice_deepgram: str = "aura-2-thalia-en"
    tts_voice_deepgram_es: str = "aura-2-estrella-es"
    tts_voice_google_es: str = "es-US-Neural2-A"
    tts_speed: float = 1.0

    @field_validator("environment", mode="before")
    @classmethod
    def normalize_environment(cls, value: object) -> str:
        if isinstance(value, str):
            return value.strip().lower()
        return str(value or "development")

    @field_validator("redis_url", "celery_broker_url", "celery_result_backend")
    @classmethod
    def warn_readonly_upstash_redis(cls, value: str) -> str:
        """Upstash read-only TCP user breaks cache invalidation and Celery."""
        if value and "default_ro" in value:
            logger.warning(
                "Redis URL uses Upstash read-only user 'default_ro' — "
                "switch to 'default' with the Standard TCP password for writes"
            )
        return value

    @property
    def is_production(self) -> bool:
        return self.environment.strip().lower() == "production"

    @property
    def allow_test_console(self) -> bool:
        return (not self.is_production) or self.enable_test_console

    def validate_runtime_secrets(self) -> list[str]:
        """Return production misconfiguration errors (empty list when safe).

        Used to fail fast at startup so we never serve real traffic with
        development defaults, weak secrets, or unverifiable webhooks.
        """
        if not self.is_production:
            return []

        errors: list[str] = []

        weak_secret_keys = {
            "",
            "change-me",
            "insecure-dev-key-change-in-production",
        }
        if self.secret_key in weak_secret_keys:
            errors.append(
                "SECRET_KEY must be set to a strong, unique value in production"
            )
        elif len(self.secret_key) < 32:
            errors.append("SECRET_KEY must be at least 32 characters in production")

        if self.admin_password in ("", "Admin123!"):
            errors.append(
                "ADMIN_PASSWORD must be changed from the default in production"
            )

        if not self.encryption_key:
            errors.append(
                "ENCRYPTION_KEY must be set in production "
                '(generate one with: python -c "from cryptography.fernet import '
                'Fernet; print(Fernet.generate_key().decode())")'
            )
        else:
            try:
                from cryptography.fernet import Fernet

                Fernet(self.encryption_key.encode())
            except Exception:
                errors.append(
                    "ENCRYPTION_KEY is not a valid Fernet key "
                    "(must be a 32-byte urlsafe base64 key)"
                )

        if not self.app_url.lower().startswith("https://"):
            errors.append("APP_URL must use https:// in production")

        if not self.telnyx_public_key:
            errors.append(
                "TELNYX_PUBLIC_KEY must be set in production to verify "
                "incoming Telnyx webhooks"
            )

        if self.web_workers != 1:
            errors.append(
                "WEB_WORKERS must be 1 in production until shared session "
                "store is implemented (live calls use in-process state; "
                "rate limits are also per-process)"
            )

        if self.debug:
            errors.append("DEBUG must be false in production")

        if not (self.telnyx_api_key or "").strip():
            errors.append(
                "TELNYX_API_KEY must be set in production to place and control calls"
            )
        if self.enable_test_console:
            errors.append(
                "ENABLE_TEST_CONSOLE must be false in production"
            )

        app_host = (self.app_url or "").lower()
        if (
            app_host.startswith("https://")
            and "localhost" not in app_host
            and "127.0.0.1" not in app_host
            and not (self.trusted_proxy_ips or "").strip()
        ):
            errors.append(
                "TRUSTED_PROXY_IPS must be set in production behind a reverse proxy "
                "(comma-separated load-balancer peer IPs) so per-client rate limits "
                "and login lockouts work correctly"
            )

        for name, url in (
            ("REDIS_URL", self.redis_url),
            ("CELERY_BROKER_URL", self.celery_broker_url),
            ("CELERY_RESULT_BACKEND", self.celery_result_backend),
        ):
            if url and "default_ro" in url:
                errors.append(
                    f"{name} uses Upstash read-only credentials (default_ro); "
                    "use the Standard TCP password with username default"
                )

        return errors

    def validate_celery_runtime_secrets(
        self, *, require_encryption: bool = True
    ) -> list[str]:
        """Production checks for Celery worker/beat processes.

        Unlike ``validate_runtime_secrets``, this omits web-only requirements
        (Telnyx, admin password, WEB_WORKERS, trusted proxy IPs). Workers that
        send email or fire CRM webhooks need ``require_encryption=True`` so
        encrypted settings can be decrypted; beat only schedules tasks.
        """
        if not self.is_production:
            return []

        errors: list[str] = []

        weak_secret_keys = {
            "",
            "change-me",
            "insecure-dev-key-change-in-production",
        }
        if self.secret_key in weak_secret_keys:
            errors.append(
                "SECRET_KEY must be set to a strong, unique value in production"
            )
        elif len(self.secret_key) < 32:
            errors.append("SECRET_KEY must be at least 32 characters in production")

        if require_encryption:
            if not self.encryption_key:
                errors.append(
                    "ENCRYPTION_KEY must be set in production "
                    '(generate one with: python -c "from cryptography.fernet import '
                    'Fernet; print(Fernet.generate_key().decode())")'
                )
            else:
                try:
                    from cryptography.fernet import Fernet

                    Fernet(self.encryption_key.encode())
                except Exception:
                    errors.append(
                        "ENCRYPTION_KEY is not a valid Fernet key "
                        "(must be a 32-byte urlsafe base64 key)"
                    )

        if not self.app_url.lower().startswith("https://"):
            errors.append("APP_URL must use https:// in production")

        if self.debug:
            errors.append("DEBUG must be false in production")

        for name, url in (
            ("REDIS_URL", self.redis_url),
            ("CELERY_BROKER_URL", self.celery_broker_url),
            ("CELERY_RESULT_BACKEND", self.celery_result_backend),
        ):
            if url and "default_ro" in url:
                errors.append(
                    f"{name} uses Upstash read-only credentials (default_ro); "
                    "use the Standard TCP password with username default"
                )

        return errors


settings = Settings()


def redis_url_connection_kwargs(url: str) -> dict:
    """TLS kwargs for redis-py / Celery when connecting to Upstash (rediss://)."""
    if (url or "").startswith("rediss://"):
        return {"ssl_cert_reqs": ssl.CERT_REQUIRED}
    return {}


def celery_redis_ssl_options() -> dict | None:
    """Celery broker/backend SSL options for Upstash TLS URLs."""
    url = settings.celery_broker_url or ""
    if url.startswith("rediss://"):
        return {"ssl_cert_reqs": ssl.CERT_REQUIRED}
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Default system settings to seed database with
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_SYSTEM_SETTINGS = [
    {
        "key": "active_llm_provider",
        "value": settings.active_llm_provider,
        "value_type": "string",
        "description": "Active LLM provider",
        "is_sensitive": False,
    },
    {
        "key": "active_stt_provider",
        "value": settings.active_stt_provider,
        "value_type": "string",
        "description": "Active STT provider",
        "is_sensitive": False,
    },
    {
        "key": "active_tts_provider",
        "value": settings.active_tts_provider,
        "value_type": "string",
        "description": "Active TTS provider",
        "is_sensitive": False,
    },
    {
        "key": "active_groq_model",
        "value": settings.active_groq_model,
        "value_type": "string",
        "description": "Active Groq model",
        "is_sensitive": False,
    },
    {
        "key": "active_openai_model",
        "value": settings.active_openai_model,
        "value_type": "string",
        "description": "Active OpenAI model",
        "is_sensitive": False,
    },
    {
        "key": "active_openrouter_model",
        "value": settings.active_openrouter_model,
        "value_type": "string",
        "description": "Active OpenRouter model",
        "is_sensitive": False,
    },
    {
        "key": "active_gemini_model",
        "value": settings.active_gemini_model,
        "value_type": "string",
        "description": "Active Gemini model",
        "is_sensitive": False,
    },
    {
        "key": "landlord_email",
        "value": settings.default_landlord_email,
        "value_type": "string",
        "description": "Landlord notification email",
        "is_sensitive": False,
    },
    {
        "key": "property_name",
        "value": settings.default_property_name,
        "value_type": "string",
        "description": "Property name for AI agent",
        "is_sensitive": False,
    },
    {
        "key": "greeting_message",
        "value": "",
        "value_type": "string",
        "description": (
            "Spoken opening before the first question. Blank uses the built-in "
            "script. Use {property_name} as a placeholder for the business name."
        ),
        "is_sensitive": False,
    },
    {
        "key": "closing_message",
        "value": "",
        "value_type": "string",
        "description": (
            "Spoken closing after screening completes. Blank uses the built-in "
            "wrap-up. Use {property_name} as a placeholder for the business name."
        ),
        "is_sensitive": False,
    },
    {
        "key": "provider_failure_message",
        "value": "",
        "value_type": "string",
        "description": (
            "Spoken when LLM, TTS, or STT is unavailable and the call must end. "
            "Blank uses the built-in script. Use {property_name} as a placeholder."
        ),
        "is_sensitive": False,
    },
    {
        "key": "call_recording_enabled",
        "value": "false",
        "value_type": "boolean",
        "description": "Enable call recording",
        "is_sensitive": False,
    },
    {
        "key": "max_call_duration_seconds",
        "value": "600",
        "value_type": "integer",
        "description": "Maximum call duration in seconds",
        "is_sensitive": False,
    },
    {
        "key": "silence_timeout_seconds",
        "value": "12",
        "value_type": "integer",
        "description": "Silence timeout before re-prompt",
        "is_sensitive": False,
    },
    {
        "key": "max_retries_per_question",
        "value": "2",
        "value_type": "integer",
        "description": "Max retries per screening question",
        "is_sensitive": False,
    },
    {
        "key": "email_notifications_enabled",
        "value": "true",
        "value_type": "boolean",
        "description": "Send email after each call",
        "is_sensitive": False,
    },
    {
        "key": "email_from_name",
        "value": settings.email_from_name,
        "value_type": "string",
        "description": "Sender display name for result emails",
        "is_sensitive": False,
    },
    {
        "key": "email_from_address",
        "value": settings.email_from,
        "value_type": "string",
        "description": "Sender address for result emails",
        "is_sensitive": False,
    },
    {
        "key": "email_subject_template",
        "value": "New Screening Result: {name}",
        "value_type": "string",
        "description": "Subject line template for result emails",
        "is_sensitive": False,
    },
    {
        "key": "email_body_template",
        "value": "",
        "value_type": "string",
        "description": "HTML body template for result emails (blank uses built-in layout)",
        "is_sensitive": False,
    },
    {
        "key": "email_qualified_only",
        "value": "false",
        "value_type": "boolean",
        "description": "Only email when applicant is qualified",
        "is_sensitive": False,
    },
    {
        "key": "email_include_transcript",
        "value": "false",
        "value_type": "boolean",
        "description": "Include call transcript in result emails",
        "is_sensitive": False,
    },
    {
        "key": "cc_emails",
        "value": "",
        "value_type": "string",
        "description": "CC addresses for result emails (comma-separated)",
        "is_sensitive": False,
    },
    {
        "key": "bcc_emails",
        "value": "",
        "value_type": "string",
        "description": "BCC addresses for result emails (comma-separated)",
        "is_sensitive": False,
    },
    {
        "key": "crm_webhook_url",
        "value": "",
        "value_type": "string",
        "description": "CRM webhook URL for post-call events",
        "is_sensitive": False,
    },
    {
        "key": "crm_notifications_enabled",
        "value": "false",
        "value_type": "boolean",
        "description": "Send post-call results to CRM webhook when URL is set",
        "is_sensitive": False,
    },
    {
        "key": "crm_webhook_secret",
        "value": "",
        "value_type": "string",
        "description": "HMAC secret for CRM webhook",
        "is_sensitive": True,
    },
    {
        "key": "blacklisted_numbers",
        "value": "[]",
        "value_type": "json",
        "description": "Blacklisted phone numbers",
        "is_sensitive": False,
    },
    {
        "key": "auto_fallback_enabled",
        "value": "true",
        "value_type": "boolean",
        "description": "Enable automatic provider fallback",
        "is_sensitive": False,
    },
    {
        "key": "llm_fallback_provider",
        "value": "auto",
        "value_type": "string",
        "description": "Backup AI brain when the main one fails "
        "(auto = try all others, or pick one, or none)",
        "is_sensitive": False,
    },
    {
        "key": "stt_fallback_provider",
        "value": "auto",
        "value_type": "string",
        "description": "Backup speech-to-text when the main one fails "
        "(auto = try all others, or pick one, or none)",
        "is_sensitive": False,
    },
    {
        "key": "tts_fallback_provider",
        "value": "auto",
        "value_type": "string",
        "description": "Backup voice when the main one fails "
        "(auto = try all others, or pick one, or none)",
        "is_sensitive": False,
    },
    {
        "key": "qualified_score_threshold",
        "value": "75",
        "value_type": "integer",
        "description": "Minimum score (0-100) for 'qualified' status",
        "is_sensitive": False,
    },
    {
        "key": "review_score_threshold",
        "value": "40",
        "value_type": "integer",
        "description": "Minimum score (0-100) for 'review' status; below this is 'unqualified'",
        "is_sensitive": False,
    },
    {
        "key": "timezone",
        "value": "America/New_York",
        "value_type": "string",
        "description": "Timezone for timestamps",
        "is_sensitive": False,
    },
    {
        "key": "tts_voice_google",
        "value": settings.tts_voice_google,
        "value_type": "string",
        "description": "Google TTS voice",
        "is_sensitive": False,
    },
    {
        "key": "tts_voice_deepgram",
        "value": settings.tts_voice_deepgram,
        "value_type": "string",
        "description": "Deepgram TTS voice",
        "is_sensitive": False,
    },
    {
        "key": "tts_voice_deepgram_es",
        "value": settings.tts_voice_deepgram_es,
        "value_type": "string",
        "description": "Deepgram TTS voice for Spanish calls",
        "is_sensitive": False,
    },
    {
        "key": "tts_voice_google_es",
        "value": settings.tts_voice_google_es,
        "value_type": "string",
        "description": "Google TTS voice for Spanish calls",
        "is_sensitive": False,
    },
    {
        "key": "tts_speed",
        "value": str(settings.tts_speed),
        "value_type": "string",
        "description": "TTS speech speed (0.75-1.25)",
        "is_sensitive": False,
    },
    {
        "key": "deepgram_model",
        "value": settings.deepgram_model,
        "value_type": "string",
        "description": "Deepgram STT model",
        "is_sensitive": False,
    },
    {
        "key": "groq_stt_model",
        "value": settings.groq_stt_model,
        "value_type": "string",
        "description": "Groq STT model",
        "is_sensitive": False,
    },
    {
        "key": "llm_temperature",
        "value": "0.3",
        "value_type": "string",
        "description": "AI reply creativity (0.0 = focused/consistent, 1.0 = varied)",
        "is_sensitive": False,
    },
    {
        "key": "llm_max_tokens",
        "value": "0",
        "value_type": "integer",
        "description": "Max length of an AI reply in tokens; 0 uses the tuned default",
        "is_sensitive": False,
    },
    {
        "key": "voice_latency_profile",
        "value": "balanced",
        "value_type": "string",
        "description": "Voice latency preset: fast, balanced, or quality",
        "is_sensitive": False,
    },
    {
        "key": "latency_alert_turn_p95_ms",
        "value": "1200",
        "value_type": "integer",
        "description": "Warning threshold for turn p95 latency in ms",
        "is_sensitive": False,
    },
    {
        "key": "latency_alert_turn_p95_crit_ms",
        "value": "1800",
        "value_type": "integer",
        "description": "Critical threshold for turn p95 latency in ms",
        "is_sensitive": False,
    },
    {
        "key": "latency_alert_timeout_rate_pct",
        "value": "2.0",
        "value_type": "string",
        "description": "Warning threshold for timed-out turns as percentage",
        "is_sensitive": False,
    },
    {
        "key": "latency_alert_timeout_rate_crit_pct",
        "value": "5.0",
        "value_type": "string",
        "description": "Critical threshold for timed-out turns as percentage",
        "is_sensitive": False,
    },
    {
        "key": "llm_streaming_enabled",
        "value": "true",
        "value_type": "boolean",
        "description": "Stream LLM tokens to TTS during live calls for faster replies",
        "is_sensitive": False,
    },
    # ── Data retention (automatic cleanup keeps the database from growing
    # forever). A value of 0 disables that particular cleanup. ──
    {
        "key": "retention_enabled",
        "value": "true",
        "value_type": "boolean",
        "description": "Master switch for the daily automatic data-cleanup job",
        "is_sensitive": False,
    },
    {
        "key": "retention_calls_days",
        "value": "365",
        "value_type": "integer",
        "description": "Permanently delete calls + applicants older than this many days (0 = keep forever)",
        "is_sensitive": False,
    },
    {
        "key": "retention_audit_days",
        "value": "365",
        "value_type": "integer",
        "description": "Delete activity-log entries older than this many days (0 = keep forever)",
        "is_sensitive": False,
    },
    {
        "key": "retention_recording_days",
        "value": "90",
        "value_type": "integer",
        "description": "Delete call recordings older than this many days, keeping the call record (0 = keep forever)",
        "is_sensitive": False,
    },
    {
        "key": "retention_soft_deleted_days",
        "value": "30",
        "value_type": "integer",
        "description": "Permanently remove already-deleted calls this many days after deletion (0 = keep forever)",
        "is_sensitive": False,
    },
    {
        "key": "retention_stale_call_hours",
        "value": "24",
        "value_type": "integer",
        "description": "Mark calls stuck in initiated/in_progress as failed after this many hours (0 = disable)",
        "is_sensitive": False,
    },
]

ENV_BACKED_SYSTEM_SETTING_KEYS = {
    "active_llm_provider",
    "active_stt_provider",
    "active_tts_provider",
    "active_groq_model",
    "active_openai_model",
    "active_openrouter_model",
    "active_gemini_model",
    "landlord_email",
    "property_name",
    "email_from_name",
    "email_from_address",
    "tts_voice_google",
    "tts_voice_deepgram",
    "tts_voice_deepgram_es",
    "tts_voice_google_es",
    "tts_speed",
    "deepgram_model",
    "groq_stt_model",
}


# ──────────────────────────────────────────────────────────────────────────────
# Provider Registry — Hot-swap AI providers at runtime
# ──────────────────────────────────────────────────────────────────────────────


_ENCRYPTED_KEY_MAP = {
    "groq_api_key_encrypted": "groq_api_key",
    "openai_api_key_encrypted": "openai_api_key",
    "openrouter_api_key_encrypted": "openrouter_api_key",
    "gemini_api_key_encrypted": "gemini_api_key",
    "deepgram_api_key_encrypted": "deepgram_api_key",
}
_ENV_API_KEY_DEFAULTS = {attr: getattr(settings, attr, "") for attr in _ENCRYPTED_KEY_MAP.values()}


def _apply_encrypted_api_keys(values: dict) -> None:
    """Decrypt admin-saved API keys and apply them onto the live settings.

    The admin Providers panel stores rotated keys as ``<provider>_api_key_encrypted``
    in ``system_settings``. Providers read ``settings.<provider>_api_key`` lazily,
    so updating the in-memory settings here makes rotation take effect on the next
    provider rebuild without a server restart. Decrypt failures are logged and the
    existing env-backed value is kept.
    """
    from app.utils.security import decrypt_value

    for enc_key, attr in _ENCRYPTED_KEY_MAP.items():
        raw = values.get(enc_key)
        if not raw:
            # Setting removed/rolled back in DB: restore env-backed baseline so
            # stale rotated keys do not linger in process memory.
            setattr(settings, attr, _ENV_API_KEY_DEFAULTS.get(attr, ""))
            continue
        try:
            decrypted = decrypt_value(str(raw)).strip()
        except Exception as e:  # noqa: BLE001 - never let a bad key break startup
            logger.warning("Could not decrypt %s: %s", enc_key, e)
            setattr(settings, attr, _ENV_API_KEY_DEFAULTS.get(attr, ""))
            continue
        if decrypted:
            setattr(settings, attr, decrypted)
        else:
            setattr(settings, attr, _ENV_API_KEY_DEFAULTS.get(attr, ""))


class ProviderRegistry:
    """
    Singleton that holds active provider instances.
    Admin panel calls switch_*() methods to hot-swap at runtime without
    restarting the server. All changes are persisted to system_settings.
    Thread-safe using asyncio.Lock.
    """

    _instance: Optional["ProviderRegistry"] = None
    _lock: asyncio.Lock = asyncio.Lock()

    def __new__(cls) -> "ProviderRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._stt = None
            cls._instance._llm = None
            cls._instance._tts = None
            cls._instance._stt_name = ""
            cls._instance._llm_name = ""
            cls._instance._tts_name = ""
            cls._instance._auto_fallback_enabled = True
            cls._instance._llm_fallback_provider = "auto"
            cls._instance._stt_fallback_provider = "auto"
            cls._instance._tts_fallback_provider = "auto"
        return cls._instance

    async def initialize(self) -> None:
        """Initialize providers from DB settings (fallback to env)."""
        from app.db.database import AsyncSessionLocal

        try:
            async with AsyncSessionLocal() as db:
                await self.reload_from_db(db)
        except Exception as e:
            logger.warning(f"Could not load provider settings from DB: {e}")
            if settings.is_production:
                raise
            logger.info("Initializing ProviderRegistry from env defaults...")
            await self.switch_llm(settings.active_llm_provider)
            await self.switch_stt(settings.active_stt_provider)
            await self.switch_tts(settings.active_tts_provider)

        logger.info(
            f"ProviderRegistry initialized — LLM: {self._llm_name}, "
            f"STT: {self._stt_name}, TTS: {self._tts_name}, "
            f"auto_fallback: {self._auto_fallback_enabled}"
        )

    async def reload_from_db(self, db) -> None:
        """Reload active providers from DB (affects new calls only)."""
        from app.db.crud import fetch_settings_batch

        keys = (
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
            "auto_fallback_enabled",
            "llm_fallback_provider",
            "stt_fallback_provider",
            "tts_fallback_provider",
            "groq_api_key_encrypted",
            "openai_api_key_encrypted",
            "openrouter_api_key_encrypted",
            "gemini_api_key_encrypted",
            "deepgram_api_key_encrypted",
        )
        values = await fetch_settings_batch(db, keys)

        # Apply admin-rotated API keys (saved encrypted in the DB) onto the
        # in-memory settings so freshly-built provider clients use them. Falls
        # back to env values when a key is absent or fails to decrypt.
        _apply_encrypted_api_keys(values)

        llm = str(values.get("active_llm_provider") or settings.active_llm_provider)
        stt = str(values.get("active_stt_provider") or settings.active_stt_provider)
        tts = str(values.get("active_tts_provider") or settings.active_tts_provider)
        self._auto_fallback_enabled = bool(values.get("auto_fallback_enabled", True))
        self._llm_fallback_provider = str(
            values.get("llm_fallback_provider") or "auto"
        ).lower()
        self._stt_fallback_provider = str(
            values.get("stt_fallback_provider") or "auto"
        ).lower()
        self._tts_fallback_provider = str(
            values.get("tts_fallback_provider") or "auto"
        ).lower()

        model_by_llm = {
            "groq": values.get("active_groq_model") or settings.active_groq_model,
            "openai": values.get("active_openai_model") or settings.active_openai_model,
            "openrouter": values.get("active_openrouter_model")
            or settings.active_openrouter_model,
            "gemini": values.get("active_gemini_model") or settings.active_gemini_model,
        }
        voice_by_tts = {
            "google": values.get("tts_voice_google") or settings.tts_voice_google,
            "deepgram": values.get("tts_voice_deepgram") or settings.tts_voice_deepgram,
        }

        stt_model_by_provider = {
            "deepgram": values.get("deepgram_model") or settings.deepgram_model,
            "groq": values.get("groq_stt_model") or settings.groq_stt_model,
        }
        await self.switch_llm(llm, model=model_by_llm.get(llm.lower()))
        await self.switch_stt(stt, model=stt_model_by_provider.get(stt.lower()))
        await self.switch_tts(tts, voice=voice_by_tts.get(tts.lower()))

    async def switch_llm(self, provider: str, model: str | None = None) -> None:
        """Switch LLM provider without server restart."""
        async with self._lock:
            try:
                from app.providers.llm.gemini_llm import GeminiLLMProvider
                from app.providers.llm.groq_llm import GroqLLMProvider
                from app.providers.llm.openai_llm import OpenAILLMProvider
                from app.providers.llm.openrouter_llm import OpenRouterLLMProvider

                provider = provider.lower().strip()
                if provider == "groq":
                    self._llm = GroqLLMProvider(
                        model=model or settings.active_groq_model
                    )
                elif provider == "openai":
                    self._llm = OpenAILLMProvider(
                        model=model or settings.active_openai_model
                    )
                elif provider == "openrouter":
                    self._llm = OpenRouterLLMProvider(
                        model=model or settings.active_openrouter_model
                    )
                elif provider == "gemini":
                    self._llm = GeminiLLMProvider(
                        model=model or settings.active_gemini_model
                    )
                else:
                    raise ValueError(f"Unknown LLM provider: {provider}")

                self._llm_name = provider
                logger.info(f"LLM switched to: {provider}")
            except Exception as e:
                logger.error(f"Failed to switch LLM to {provider}: {e}")
                raise

    async def switch_stt(self, provider: str, model: str | None = None) -> None:
        """Switch STT provider without server restart."""
        async with self._lock:
            try:
                from app.providers.stt.deepgram_stt import DeepgramSTTProvider
                from app.providers.stt.groq_stt import GroqSTTProvider

                provider = provider.lower().strip()
                if provider == "deepgram":
                    self._stt = DeepgramSTTProvider(
                        model=model or settings.deepgram_model
                    )
                elif provider == "groq":
                    self._stt = (
                        GroqSTTProvider(model=model)
                        if model
                        else GroqSTTProvider()
                    )
                else:
                    raise ValueError(f"Unknown STT provider: {provider}")

                self._stt_name = provider
                logger.info(f"STT switched to: {provider}")
            except Exception as e:
                logger.error(f"Failed to switch STT to {provider}: {e}")
                raise

    async def switch_tts(self, provider: str, voice: str | None = None) -> None:
        """Switch TTS provider without server restart."""
        async with self._lock:
            try:
                from app.providers.tts.deepgram_tts import DeepgramTTSProvider
                from app.providers.tts.google_tts import GoogleTTSProvider

                provider = provider.lower().strip()
                if provider == "google":
                    self._tts = GoogleTTSProvider(
                        voice=voice or settings.tts_voice_google
                    )
                elif provider == "deepgram":
                    self._tts = DeepgramTTSProvider(
                        voice=voice or settings.tts_voice_deepgram
                    )
                else:
                    raise ValueError(f"Unknown TTS provider: {provider}")

                self._tts_name = provider
                logger.info(f"TTS switched to: {provider}")
            except Exception as e:
                logger.error(f"Failed to switch TTS to {provider}: {e}")
                raise

    @property
    def llm(self):
        """Get active LLM provider."""
        if self._llm is None:
            raise RuntimeError("LLM provider not initialized. Call initialize() first.")
        return self._llm

    @property
    def stt(self):
        """Get active STT provider."""
        if self._stt is None:
            raise RuntimeError("STT provider not initialized. Call initialize() first.")
        return self._stt

    @property
    def tts(self):
        """Get active TTS provider."""
        if self._tts is None:
            raise RuntimeError("TTS provider not initialized. Call initialize() first.")
        return self._tts

    @property
    def llm_name(self) -> str:
        return self._llm_name

    @property
    def stt_name(self) -> str:
        return self._stt_name

    @property
    def tts_name(self) -> str:
        return self._tts_name

    @property
    def auto_fallback_enabled(self) -> bool:
        return self._auto_fallback_enabled

    @property
    def llm_fallback_provider(self) -> str:
        return self._llm_fallback_provider

    @property
    def stt_fallback_provider(self) -> str:
        return self._stt_fallback_provider

    @property
    def tts_fallback_provider(self) -> str:
        return self._tts_fallback_provider

    def get_status(self) -> dict:
        """Return current provider status summary."""
        return {
            "llm": self._llm_name,
            "stt": self._stt_name,
            "tts": self._tts_name,
            "auto_fallback_enabled": self._auto_fallback_enabled,
            "llm_fallback_provider": self._llm_fallback_provider,
            "stt_fallback_provider": self._stt_fallback_provider,
            "tts_fallback_provider": self._tts_fallback_provider,
            "initialized": all([self._llm, self._stt, self._tts]),
        }


# Global singleton instance
provider_registry = ProviderRegistry()
