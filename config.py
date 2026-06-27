"""
config.py — Application configuration, environment settings, and ProviderRegistry.

Loads all settings from environment variables (via .env), defines the
ProviderRegistry singleton for hot-swapping AI providers at runtime, and
exposes the DEFAULT_QUESTIONS for initial database seeding.
"""

import asyncio
import logging
from typing import Optional

from dotenv import load_dotenv
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.screening_flow import DEFAULT_FAQ_ENTRIES, DEFAULT_SCREENING_QUESTIONS

load_dotenv(override=False)
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
    supabase_publishable_key: str = ""
    supabase_secret_key: str = ""
    supabase_jwks_url: str = ""
    supabase_storage_bucket: str = "call-recordings"

    # Redis (separate databases for different purposes to avoid memory conflicts)
    # Cache & settings (analytics, configs)
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/1"  # Celery task queue
    celery_result_backend: str = "redis://localhost:6379/2"  # Celery result storage

    # Telephony
    telnyx_api_key: str = ""
    telnyx_public_key: str = ""
    telnyx_phone_number: str = ""

    # STT
    deepgram_api_key: str = ""

    # LLM
    groq_api_key: str = ""
    openai_api_key: str = ""
    openrouter_api_key: str = ""

    # TTS
    google_application_credentials: str = ""
    google_tts_project_id: str = ""

    # Email
    resend_api_key: str = ""
    email_from: str = "screening@example.com"
    email_from_name: str = "AI Screening Platform"

    # Encryption
    encryption_key: str = ""

    # Defaults
    default_property_name: str = "Ready Rentals Online"
    default_landlord_email: str = ""
    default_agent_name: str = "Ready Rentals assistant"

    # Active providers (used as env-backed defaults for DB runtime settings)
    active_llm_provider: str = "groq"
    active_stt_provider: str = "deepgram"
    active_tts_provider: str = "deepgram"
    active_groq_model: str = "llama-3.3-70b-versatile"
    active_openai_model: str = "gpt-4o-mini"
    active_openrouter_model: str = "meta-llama/llama-3.3-70b-instruct:free"
    deepgram_model: str = "nova-3"
    tts_voice_google: str = "en-US-Wavenet-D"
    tts_voice_deepgram: str = "aura-2-thalia-en"
    tts_speed: float = 1.0

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

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

        return errors


settings = Settings()


# ──────────────────────────────────────────────────────────────────────────────
# Default Screening Questions
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_QUESTIONS = list(DEFAULT_SCREENING_QUESTIONS)
DEFAULT_FAQS = list(DEFAULT_FAQ_ENTRIES)

# Default system settings to seed database with
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
        "key": "ai_agent_name",
        "value": settings.default_agent_name,
        "value_type": "string",
        "description": "AI agent name",
        "is_sensitive": False,
    },
    {
        "key": "min_income_threshold",
        "value": "0",
        "value_type": "integer",
        "description": "Optional absolute monthly income floor ($); 0 uses 3x rent policy",
        "is_sensitive": False,
    },
    {
        "key": "disqualify_on_eviction",
        "value": "false",
        "value_type": "boolean",
        "description": "Auto-disqualify if eviction disclosed (normally false; reviewed individually)",
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
        "key": "crm_webhook_url",
        "value": "",
        "value_type": "string",
        "description": "CRM webhook URL for post-call events",
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
        "key": "score_weight_income",
        "value": "35",
        "value_type": "integer",
        "description": "Score weight: income context (0-100)",
        "is_sensitive": False,
    },
    {
        "key": "score_weight_eviction",
        "value": "15",
        "value_type": "integer",
        "description": "Score weight: eviction context (0-100)",
        "is_sensitive": False,
    },
    {
        "key": "score_weight_completion",
        "value": "25",
        "value_type": "integer",
        "description": "Score weight: screening completion (0-100)",
        "is_sensitive": False,
    },
    {
        "key": "score_weight_move_date",
        "value": "10",
        "value_type": "integer",
        "description": "Score weight: move-in timing (0-100)",
        "is_sensitive": False,
    },
    {
        "key": "score_weight_rental_history",
        "value": "10",
        "value_type": "integer",
        "description": "Score weight: rental history context (0-100)",
        "is_sensitive": False,
    },
    {
        "key": "score_weight_household_fit",
        "value": "5",
        "value_type": "integer",
        "description": "Score weight: occupants and pet details (0-100)",
        "is_sensitive": False,
    },
    {
        "key": "monthly_rent_for_income_ratio",
        "value": "0",
        "value_type": "integer",
        "description": "Optional monthly rent used for income scoring; 0 means review income context",
        "is_sensitive": False,
    },
    {
        "key": "income_multiplier",
        "value": "3.0",
        "value_type": "string",
        "description": "Required income as a multiple of monthly rent (e.g. 3.0 = 3x rent)",
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
        "key": "hold_music_enabled",
        "value": "false",
        "value_type": "boolean",
        "description": "Play hold music on answer",
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
    "landlord_email",
    "property_name",
    "ai_agent_name",
    "tts_voice_google",
    "tts_voice_deepgram",
    "tts_speed",
    "deepgram_model",
}


# ──────────────────────────────────────────────────────────────────────────────
# Provider Registry — Hot-swap AI providers at runtime
# ──────────────────────────────────────────────────────────────────────────────


_ENCRYPTED_KEY_MAP = {
    "groq_api_key_encrypted": "groq_api_key",
    "openai_api_key_encrypted": "openai_api_key",
    "openrouter_api_key_encrypted": "openrouter_api_key",
    "deepgram_api_key_encrypted": "deepgram_api_key",
}


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
            continue
        try:
            decrypted = decrypt_value(str(raw)).strip()
        except Exception as e:  # noqa: BLE001 - never let a bad key break startup
            logger.warning("Could not decrypt %s: %s", enc_key, e)
            continue
        if decrypted:
            setattr(settings, attr, decrypted)


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
            "deepgram_model",
            "groq_stt_model",
            "tts_voice_google",
            "tts_voice_deepgram",
            "auto_fallback_enabled",
            "llm_fallback_provider",
            "stt_fallback_provider",
            "tts_fallback_provider",
            "groq_api_key_encrypted",
            "openai_api_key_encrypted",
            "openrouter_api_key_encrypted",
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
        }
        voice_by_tts = {
            "google": values.get("tts_voice_google") or settings.tts_voice_google,
            "deepgram": values.get("tts_voice_deepgram") or settings.tts_voice_deepgram,
        }

        stt_model_by_provider = {
            "deepgram": values.get("deepgram_model") or settings.deepgram_model,
            "groq": values.get("groq_stt_model") or "whisper-large-v3-turbo",
        }
        await self.switch_llm(llm, model=model_by_llm.get(llm.lower()))
        await self.switch_stt(stt, model=stt_model_by_provider.get(stt.lower()))
        await self.switch_tts(tts, voice=voice_by_tts.get(tts.lower()))

    async def switch_llm(self, provider: str, model: str | None = None) -> None:
        """Switch LLM provider without server restart."""
        async with self._lock:
            try:
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
