"""Pydantic schemas for settings and provider management routes."""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class LLMProviderSwitch(BaseModel):
    provider: Literal["groq", "openai", "openrouter"]
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


class ProviderTestRequest(BaseModel):
    provider_type: Literal["llm", "stt", "tts"]
    provider: str
    test_text: str = "Hello, this is a provider test."


class ProviderTestResponse(BaseModel):
    success: bool
    latency_ms: float
    response: str | None = None
    error: str | None = None


class ScreeningQuestion(BaseModel):
    id: str
    state: str
    question: str
    extract_fields: list[str] = Field(default_factory=list)
    validation: str | None = None
    retry_prompt: str | None = None
    retry_prompt_2: str | None = None
    retry_prompt_3: str | None = None
    active: bool = True
    order: int = 0


class QuestionsUpdateRequest(BaseModel):
    questions: list[ScreeningQuestion]

    @field_validator("questions")
    @classmethod
    def question_ids_must_be_unique(
        cls, value: list[ScreeningQuestion]
    ) -> list[ScreeningQuestion]:
        ids = [question.id for question in value]
        if len(ids) != len(set(ids)):
            raise ValueError("Question IDs must be unique")
        return value


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

    @field_validator("faqs")
    @classmethod
    def faq_topics_must_be_unique(cls, value: list[ScreeningFaq]) -> list[ScreeningFaq]:
        topics = [faq.topic for faq in value]
        if len(topics) != len(set(topics)):
            raise ValueError("FAQ topics must be unique")
        return value


class EmailSettingsUpdate(BaseModel):
    landlord_email: str | None = None
    email_from_name: str | None = None
    email_from_address: str | None = None
    email_subject_template: str | None = None
    email_body_template: str | None = None
    cc_emails: str | None = None
    bcc_emails: str | None = None
    email_notifications_enabled: bool | None = None
    email_qualified_only: bool | None = None
    email_include_transcript: bool | None = None


class GeneralSettingsUpdate(BaseModel):
    property_name: str | None = None
    ai_agent_name: str | None = None
    min_income_threshold: int | None = None
    disqualify_on_eviction: bool | None = None
    call_recording_enabled: bool | None = None
    max_call_duration_seconds: int | None = None
    silence_timeout_seconds: int | None = None
    max_retries_per_question: int | None = None
    auto_fallback_enabled: bool | None = None
    score_weight_income: int | None = None
    score_weight_eviction: int | None = None
    score_weight_completion: int | None = None
    score_weight_move_date: int | None = None
    score_weight_rental_history: int | None = None
    score_weight_household_fit: int | None = None
    monthly_rent_for_income_ratio: int | None = None
    income_multiplier: float | None = None
    qualified_score_threshold: int | None = None
    review_score_threshold: int | None = None
    timezone: str | None = None
    tts_voice_google: str | None = None
    tts_voice_deepgram: str | None = None
    tts_speed: float | None = None
    deepgram_model: str | None = None
    hold_music_enabled: bool | None = None
    crm_webhook_url: str | None = None
    crm_webhook_secret: str | None = None
    blacklisted_numbers: Any | None = None
