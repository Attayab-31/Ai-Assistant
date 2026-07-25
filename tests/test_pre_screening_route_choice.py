import pytest

from app.core.call_handler import (
    build_pre_screening_prompt,
    classify_pre_screening_route,
)
from app.core.conversation import ConversationSession


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
    assert classify_pre_screening_route(transcript, language_code=language_code) == expected


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
