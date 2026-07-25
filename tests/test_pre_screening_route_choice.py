import pytest

from app.core.call_handler import (
    build_pre_screening_prompt,
    classify_pre_screening_route,
)
from app.core.conversation import CallState, ConversationSession


@pytest.mark.parametrize(
    ("transcript", "language_code", "expected"),
    [
        ("I want to answer a few questions", "en", "questions"),
        ("Quiero responder unas preguntas", "es", "questions"),
        ("I want to leave a message for a callback", "en", "callback"),
        ("Quiero dejar un mensaje para que me llamen", "es", "callback"),
        ("I want to talk to a person", "en", "human"),
        ("Quiero hablar con una persona", "es", "human"),
    ],
)
def test_classify_pre_screening_route_matches_expected(transcript, language_code, expected):
    assert classify_pre_screening_route(
        transcript, language_code=language_code) == expected


def test_build_pre_screening_prompt_uses_session_prompt_when_available():
    session = ConversationSession(call_id="test", phone_number="")
    session.pre_screening_enabled = True
    session.pre_screening_prompt = "Custom prompt for callers"

    assert build_pre_screening_prompt(session) == "Custom prompt for callers"


def test_build_pre_screening_prompt_falls_back_to_language_specific_default():
    session = ConversationSession(call_id="test", phone_number="")
    session.pre_screening_enabled = True
    session.call_language = "es"

    prompt = build_pre_screening_prompt(session)

    assert "callback" in prompt.lower() or "preguntas" in prompt.lower()


@pytest.mark.asyncio
async def test_pre_screening_route_choice_uses_short_ack_after_greeting_prompt(monkeypatch):
    from app.core.call_handler import process_tenant_speech

    session = ConversationSession(
        call_id="test",
        phone_number="",
        questions=[
            {
                "state": "Q1_NAME",
                "question": "What is your full legal name?",
                "retry_prompt": "Please tell me your first and last name.",
                "active": True,
                "order": 1,
            }
        ],
    )
    session.pre_screening_enabled = True
    session.pre_screening_prompt = "Would you like to answer a few questions?"
    session.pre_screening_prompt_spoken = True
    session.current_state = CallState.PRE_SCREENING.value
    session.route_choice_pending = True

    async def _fake_synthesize(*_args, **_kwargs):
        return [b"ok"]

    monkeypatch.setattr(
        "app.core.call_handler.synthesize_speech_parts", _fake_synthesize
    )

    response_text, _audio_parts, complete = await process_tenant_speech(
        session, "I want to answer a few questions"
    )

    assert complete is False
    assert session.route_choice_selected == "questions"
    assert response_text == "Great, let's get started. What is your full legal name?"
    assert "would you like to answer a few questions" not in response_text.lower()
