"""Tests for call-language → STT/TTS provider mapping."""

import pytest

from app.core.voice_language import (
    deepgram_stt_language,
    deepgram_tts_voice,
    google_tts_language_code,
    google_tts_voice,
    groq_stt_language,
    is_spanish_code,
)


def test_is_spanish_code():
    assert is_spanish_code("es")
    assert is_spanish_code("es-MX")
    assert not is_spanish_code("en")
    assert not is_spanish_code(None)


def test_deepgram_stt_language():
    assert deepgram_stt_language("en") == "en-US"
    assert deepgram_stt_language("es") == "es"


def test_groq_stt_language():
    assert groq_stt_language("en") == "en"
    assert groq_stt_language("es") == "es"


def test_deepgram_tts_voice_switches_on_spanish():
    assert (
        deepgram_tts_voice(
            language_code="es",
            english_voice="aura-2-thalia-en",
            spanish_voice="aura-2-estrella-es",
        )
        == "aura-2-estrella-es"
    )
    assert (
        deepgram_tts_voice(
            language_code="en",
            english_voice="aura-2-thalia-en",
            spanish_voice="aura-2-estrella-es",
        )
        == "aura-2-thalia-en"
    )


def test_google_tts_voice_and_language():
    assert google_tts_language_code("es") == "es-US"
    assert google_tts_language_code("en") == "en-US"
    assert (
        google_tts_voice(
            language_code="es",
            english_voice="en-US-Wavenet-D",
            spanish_voice="es-US-Neural2-A",
        )
        == "es-US-Neural2-A"
    )


def test_human_ack_spanish_avoids_english_transitions():
    from app.core.call_handler import human_ack
    from app.core.conversation import ConversationSession

    session = ConversationSession(call_id="t", phone_number="+1", call_language="es")
    english_markers = (
        "Got it.",
        "Thanks, that helps.",
        "Sounds good.",
        "Okay, noted.",
    )
    for _ in range(40):
        text = human_ack(session)
        assert not any(marker in text for marker in english_markers)


def test_human_ack_english_uses_transition_pool():
    from app.core.call_handler import human_ack
    from app.core.conversation import ConversationSession

    session = ConversationSession(call_id="t", phone_number="+1", call_language="en")
    outs = {human_ack(session) for _ in range(50)}
    assert any(
        "Got it" in o or "Thanks" in o or "Perfect" in o or "Great" in o for o in outs
    )


def test_stt_model_for_provider_uses_frozen_bundle_models():
    from app.core.call_settings import CallProviderBundle, stt_model_for_provider
    from app.providers.stt.deepgram_stt import DeepgramSTTProvider

    providers = CallProviderBundle(
        llm=object(),
        stt=DeepgramSTTProvider(model="nova-3-live"),
        tts=object(),
        llm_name="groq",
        stt_name="deepgram",
        tts_name="deepgram",
        auto_fallback_enabled=True,
        stt_models_by_provider={
            "deepgram": "nova-3-admin",
            "groq": "whisper-custom",
        },
    )
    assert stt_model_for_provider(providers, "groq") == "whisper-custom"
    assert stt_model_for_provider(providers, "deepgram") == "nova-3-admin"


@pytest.mark.asyncio
async def test_silence_liveness_reask_uses_localized_question_text(monkeypatch):
    from app.core.call_handler import process_tenant_speech
    from app.core.conversation import ConversationSession

    session = ConversationSession(
        call_id="silence-es",
        phone_number="+1",
        call_language="es",
        current_state="Q1_NAME",
        questions=[
            {
                "state": "Q1_NAME",
                "question": "What is your full legal name?",
                "locales": {
                    "es": {
                        "question": "Cual es su nombre legal completo?",
                    }
                },
                "extract_fields": ["full_name"],
                "active": True,
                "order": 1,
            }
        ],
    )
    session.silence_nudge_active = True

    from unittest.mock import AsyncMock

    fake_llm = AsyncMock(
        return_value={
            "response_text": "",
            "understood": True,
            "intent": "nothing",
            "relevance": "relevant",
            "question_complete": False,
            "extracted_data": {},
        }
    )

    async def _fake_synthesize(*args, **kwargs):
        return [b"ok"]

    monkeypatch.setattr("app.core.call_handler.get_llm_response_with_fallback", fake_llm)
    monkeypatch.setattr("app.core.call_handler.synthesize_speech_parts", _fake_synthesize)

    response_text, _audio_parts, complete = await process_tenant_speech(session, "si")

    assert complete is False
    assert "Cual es su nombre legal completo?" in response_text
    fake_llm.assert_not_called()


@pytest.mark.asyncio
async def test_partial_answer_empty_llm_response_uses_spanish_fallback(monkeypatch):
    from app.core.call_handler import process_tenant_speech
    from app.core.conversation import ConversationSession

    session = ConversationSession(
        call_id="partial-es",
        phone_number="+1",
        call_language="es",
        current_state="Q1_NAME",
        questions=[
            {
                "state": "Q1_NAME",
                "question": "Cual es su nombre completo?",
                "extract_fields": ["full_name"],
                "active": True,
                "order": 1,
            }
        ],
    )

    async def _fake_llm(*args, **kwargs):
        return {
            "response_text": "",
            "understood": True,
            "intent": "answer",
            "relevance": "relevant",
            "question_complete": False,
            "extracted_data": {},
        }

    async def _fake_synthesize(*args, **kwargs):
        return [b"ok"]

    monkeypatch.setattr("app.core.call_handler.get_llm_response_with_fallback", _fake_llm)
    monkeypatch.setattr("app.core.call_handler.synthesize_speech_parts", _fake_synthesize)

    response_text, _audio_parts, complete = await process_tenant_speech(
        session, "si, claro"
    )

    assert complete is False
    assert "Solo me falta un detalle mas" in response_text
