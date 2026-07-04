"""Voice turn orchestration — admin recovery, meta nav, turn budgets."""

import time

from app.core.conversation import (
    ConversationSession,
    is_meta_navigation_request,
    mark_recovery_played,
    navigation_repeat_text,
    needs_extended_turn_budget,
    plan_turn_timeout_recovery,
    should_suppress_silence_nudge,
    turn_budget_seconds,
    turn_timeout_recovery_text,
    unsynthesized_speech_remainder,
)
from app.core.voice_latency import resolve_voice_latency


def _session(**kwargs) -> ConversationSession:
    questions = kwargs.pop(
        "questions",
        [
            {
                "state": "Q1_NAME",
                "question": "What is your full legal name?",
                "retry_prompt": "Please tell me your first and last name.",
                "retry_prompt_2": "I need your full legal name to continue.",
                "requires_confirmation": True,
                "extract_fields": ["full_name"],
                "active": True,
                "order": 1,
            },
            {
                "state": "Q2_PHONE",
                "question": "What is the best phone number for you?",
                "retry_prompt": "What number should we use to reach you?",
                "requires_confirmation": True,
                "extract_fields": ["contact_phone"],
                "active": True,
                "order": 2,
            },
        ],
    )
    session = ConversationSession(
        call_id="test-turn",
        phone_number="+15551234567",
        questions=questions,
        **kwargs,
    )
    return session


def test_meta_navigation_detects_next_question():
    assert is_meta_navigation_request("Yes. What is your next question?")
    assert is_meta_navigation_request("Can you repeat that again?")
    assert not is_meta_navigation_request("My name is Jane Doe.")


def test_navigation_repeat_uses_admin_question():
    session = _session(current_state="Q2_PHONE")
    assert navigation_repeat_text(session) == (
        "What is the best phone number for you?"
    )


def test_navigation_repeat_uses_admin_retry_prompt():
    session = _session(current_state="Q2_PHONE", retry_count=1)
    assert navigation_repeat_text(session) == "What number should we use to reach you?"


def test_turn_timeout_recovery_uses_admin_retry():
    session = _session(current_state="Q1_NAME", retry_count=0)
    text = turn_timeout_recovery_text(session, "John Smith.")
    assert "Sorry for the pause" in text
    assert "full legal name" in text.lower()


def test_extended_turn_budget_only_during_readback():
    session = _session(current_state="Q1_NAME")
    session.turn_timeout_seconds = 15
    assert not needs_extended_turn_budget(session)
    session.pending_confirmation = {
        "field": "full_name",
        "state": "Q1_NAME",
        "value": "John Smith",
        "attempts": 1,
    }
    assert needs_extended_turn_budget(session)
    assert turn_budget_seconds(session) == 20.0


def test_plan_timeout_recovery_readback_when_name_extracted():
    session = _session(current_state="Q1_NAME")
    session.extracted_data["full_name"] = "John Smith"
    text = plan_turn_timeout_recovery(session, "John Smith.")
    assert "John Smith" in text
    assert session.pending_confirmation is not None
    assert "didn't quite catch" not in text.lower()


def test_silence_suppressed_after_recovery():
    session = _session()
    now = time.monotonic()
    session.last_recovery_at_monotonic = now - 1.0
    assert should_suppress_silence_nudge(session, now=now)
    session.last_recovery_at_monotonic = now - 5.0
    assert not should_suppress_silence_nudge(session, now=now)


def test_unsynthesized_remainder_after_partial_stream():
    session = _session()
    session.turn_streaming_finalize = {
        "streamed_prefix": "I have your email as test@example.com.",
        "intended": "I have your email as test@example.com. Is that right?",
        "streamed_sent": True,
        "display": "I have your email as test@example.com. Is that right?",
    }
    remainder = unsynthesized_speech_remainder("", session)
    assert remainder == "Is that right?"


def test_mark_recovery_played_sets_timestamp():
    session = _session()
    mark_recovery_played(session)
    assert session.last_recovery_at_monotonic > 0


def test_fast_profile_turn_timeout_bumped():
    cfg = resolve_voice_latency({"voice_latency_profile": "fast"})
    assert cfg["turn_timeout_seconds"] == 14
