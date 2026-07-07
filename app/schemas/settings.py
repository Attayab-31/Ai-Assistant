"""Pydantic schemas for settings and provider management routes."""

from typing import Any, Literal

from pydantic import BaseModel, EmailStr, Field, TypeAdapter, field_validator

_email_adapter = TypeAdapter(EmailStr)


class LLMProviderSwitch(BaseModel):
    provider: Literal["groq", "openai", "openrouter", "gemini"]
    model: str | None = None
    api_key: str | None = None


class STTProviderSwitch(BaseModel):
    provider: Literal["deepgram", "groq"]
    model: str | None = None
    api_key: str | None = None


class TTSProviderSwitch(BaseModel):
    provider: Literal["google", "deepgram"]
    voice: str | None = None
    speed: float | None = Field(default=None, ge=0.75, le=1.25)


class ProviderApiKeyUpdate(BaseModel):
    """Set or rotate the API key for ANY provider without switching the active one."""

    provider: Literal["groq", "openai", "openrouter", "gemini", "deepgram"]
    api_key: str = Field(min_length=8, max_length=400)

    @field_validator("api_key")
    @classmethod
    def strip_key(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("API key cannot be empty")
        return value


class ConditionalRule(BaseModel):
    field: str
    operator: Literal["eq", "ne", "truthy", "falsy"] = "truthy"
    value: Any | None = None


class ScoringRule(BaseModel):
    enabled: bool = False
    max_points: int = Field(default=0, ge=0, le=100)
    rule_type: Literal[
        "any_answer", "yes_no", "numeric_range", "date_within", "required_field"
    ] = "any_answer"
    pass_config: dict[str, Any] = Field(default_factory=dict)


class LanguageOption(BaseModel):
    value: str
    label: str
    aliases: list[str] = Field(default_factory=list)


class ScreeningQuestion(BaseModel):
    id: str
    state: str
    question: str
    answer_type: Literal[
        "text",
        "long_text",
        "yes_no",
        "number",
        "currency",
        "date",
        "phone",
        "email",
        "language_choice",
    ] = "text"
    extract_fields: list[str] = Field(default_factory=list)
    field_labels: dict[str, str] = Field(default_factory=dict)
    validation: str | None = None
    retry_prompt: str | None = None
    retry_prompt_2: str | None = None
    retry_prompt_3: str | None = None
    active: bool = True
    order: int = 0
    required: bool = True
    requires_confirmation: bool = False
    conditional: ConditionalRule | None = None
    scoring: ScoringRule | None = None
    understanding_guide: str | None = None
    schema_version: int | None = None
    speech_mode: str | None = None
    require_all_extract_fields: bool = False
    language_options: list[LanguageOption] | None = None
    locales: dict[str, dict[str, str]] | None = None


class QuestionsUpdateRequest(BaseModel):
    questions: list[ScreeningQuestion]


class ScreeningFaq(BaseModel):
    id: str
    topic: str
    title: str
    pattern: str
    answer: str
    active: bool = True
    order: int = 0


class FaqsUpdateRequest(BaseModel):
    faqs: list[ScreeningFaq]


class EmailSettingsUpdate(BaseModel):
    landlord_email: EmailStr | None = None
    email_from_name: str | None = None
    email_from_address: str | None = None
    email_subject_template: str | None = None
    email_body_template: str | None = None
    cc_emails: str | None = None
    bcc_emails: str | None = None
    email_notifications_enabled: bool | None = None
    email_qualified_only: bool | None = None
    email_include_transcript: bool | None = None

    @field_validator("landlord_email", mode="before")
    @classmethod
    def strip_landlord_email(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
        return value

    @field_validator("email_from_address", mode="before")
    @classmethod
    def normalize_from_address(cls, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value.strip()  # "" clears the admin override (falls back to env)
        return value

    @field_validator("email_from_address")
    @classmethod
    def validate_from_address(cls, value: str | None) -> str | None:
        if not value:
            return ""
        _email_adapter.validate_python(value)
        return value

    @field_validator(
        "email_from_name",
        "email_subject_template",
        "email_body_template",
        "cc_emails",
        "bcc_emails",
        mode="before",
    )
    @classmethod
    def strip_optional_text(cls, value: Any) -> Any:
        if value is None:
            return None
        return value.strip() if isinstance(value, str) else value


class GeneralSettingsUpdate(BaseModel):
    property_name: str | None = None
    greeting_message: str | None = None
    closing_message: str | None = None
    provider_failure_message: str | None = None
    call_recording_enabled: bool | None = None
    max_call_duration_seconds: int | None = None
    silence_timeout_seconds: int | None = None
    max_retries_per_question: int | None = None
    auto_fallback_enabled: bool | None = None
    llm_fallback_provider: Literal[
        "auto", "groq", "openai", "openrouter", "gemini", "none"
    ] | None = None
    stt_fallback_provider: Literal["auto", "deepgram", "groq", "none"] | None = None
    tts_fallback_provider: Literal["auto", "google", "deepgram", "none"] | None = None
    qualified_score_threshold: int | None = None
    review_score_threshold: int | None = None
    llm_temperature: float | None = None
    llm_max_tokens: int | None = None
    timezone: str | None = None
    crm_webhook_url: str | None = None
    crm_webhook_secret: str | None = None
    retention_enabled: bool | None = None
    retention_calls_days: int | None = None
    retention_recording_days: int | None = None
    retention_audit_days: int | None = None
    retention_soft_deleted_days: int | None = None
    retention_stale_call_hours: int | None = None
    voice_latency_profile: Literal["fast", "balanced", "quality"] | None = None
    llm_streaming_enabled: bool | None = None

    @field_validator("max_retries_per_question")
    @classmethod
    def validate_max_retries(cls, value: int | None) -> int | None:
        if value is None:
            return value
        if not (1 <= int(value) <= 5):
            raise ValueError("max_retries_per_question must be between 1 and 5")
        return int(value)

    @field_validator("silence_timeout_seconds")
    @classmethod
    def validate_silence_timeout(cls, value: int | None) -> int | None:
        if value is None:
            return value
        if not (2 <= int(value) <= 30):
            raise ValueError("silence_timeout_seconds must be between 2 and 30")
        return int(value)

    @field_validator("max_call_duration_seconds")
    @classmethod
    def validate_max_duration(cls, value: int | None) -> int | None:
        if value is None:
            return value
        if not (60 <= int(value) <= 3600):
            raise ValueError("max_call_duration_seconds must be between 60 and 3600")
        return int(value)
