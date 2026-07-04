"""Graceful shutdown when voice providers are unavailable."""

import pytest

from app.core.conversation import (
    CallState,
    ConversationSession,
    DEFAULT_PROVIDER_FAILURE_MESSAGE,
    provider_failure_message_for_session,
    STT_EMPTY_STRIKE_LIMIT,
)


def _session(**kwargs) -> ConversationSession:
    return ConversationSession(
        call_id="test-pf",
        phone_number="+15551234567",
        current_state="Q1_NAME",
        **kwargs,
    )


def test_default_provider_failure_message_uses_property_name():
    session = _session(property_name="Sunset Apartments")
    msg = provider_failure_message_for_session(session)
    assert "Sunset Apartments" in msg
    assert "{property_name}" not in msg


def test_custom_provider_failure_message():
    session = _session(
        property_name="Oak Grove",
        provider_failure_message="Sorry — {property_name} will call you back.",
    )
    assert provider_failure_message_for_session(session) == (
        "Sorry — Oak Grove will call you back."
    )


def test_builtin_default_when_blank():
    session = _session(property_name="Demo Property", provider_failure_message="")
    expected = DEFAULT_PROVIDER_FAILURE_MESSAGE.replace(
        "{property_name}", "Demo Property"
    )
    assert provider_failure_message_for_session(session) == expected


def test_session_tracks_stt_empty_strikes():
    session = _session()
    assert session.stt_empty_strikes == 0
    session.stt_empty_strikes = 2
    assert session.stt_empty_strikes == 2


def test_stt_strike_limit_is_three():
    assert STT_EMPTY_STRIKE_LIMIT == 3


@pytest.mark.asyncio
async def test_end_call_for_provider_failure_sets_review_flags(monkeypatch):
    from app.core.call_handler import end_call_for_provider_failure

    session = _session()

    async def fake_tts(text, sess, **kwargs):
        return b"\xff" * 80

    monkeypatch.setattr(
        "app.core.call_handler.synthesize_with_fallback",
        fake_tts,
    )

    text, parts, done = await end_call_for_provider_failure(
        session, "llm", "All LLM providers failed"
    )

    assert done is True
    assert session.current_state == CallState.ENDED.value
    assert session.control_flags["provider_failure"]["service"] == "llm"
    assert any(e.get("type") == "provider_failure" for e in session.errors)
    assert "technical issue" in text.lower() or "sorry" in text.lower()
    assert parts == [b"\xff" * 80]
    assert session.transcript[-1].speaker == "AI"


@pytest.mark.asyncio
async def test_end_call_for_provider_failure_idempotent():
    from app.core.call_handler import end_call_for_provider_failure

    session = _session()
    session.control_flags["provider_failure"] = {
        "service": "tts",
        "detail": "already",
    }

    text, parts, done = await end_call_for_provider_failure(session, "tts", "again")

    assert done is True
    assert parts == []
