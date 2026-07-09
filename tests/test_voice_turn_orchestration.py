"""Voice turn orchestration — admin recovery, meta nav, turn budgets."""

import time

import pytest

from app.core.conversation import (
    ConversationSession,
    capture_turn_snapshot,
    handle_turn_timeout,
    is_meta_navigation_request,
    mark_recovery_played,
    navigation_repeat_text,
    needs_extended_turn_budget,
    plan_turn_timeout_recovery,
    should_suppress_silence_nudge,
    soft_callback_redirect_text,
    try_open_readback_confirmation,
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


def test_try_open_readback_requires_caller_spoke_for_timeout():
    session = _session(current_state="Q1_NAME")
    session.extracted_data["full_name"] = "John Smith"
    assert try_open_readback_confirmation(
        session, "Q1_NAME", require_caller_spoke=True, transcript=""
    ) is None
    assert session.pending_confirmation is None


def test_try_open_readback_normal_advance_does_not_require_transcript():
    session = _session(current_state="Q1_NAME")
    session.extracted_data["full_name"] = "Jane Doe"
    text = try_open_readback_confirmation(session, "Q1_NAME")
    assert text and "Jane Doe" in text
    assert session.pending_confirmation == {
        "field": "full_name",
        "state": "Q1_NAME",
        "value": "Jane Doe",
        "attempts": 1,
    }


def test_try_open_readback_skips_when_already_pending():
    session = _session(current_state="Q1_NAME")
    session.extracted_data["full_name"] = "Jane Doe"
    session.pending_confirmation = {
        "field": "full_name",
        "state": "Q1_NAME",
        "value": "Jane Doe",
        "attempts": 1,
    }
    assert try_open_readback_confirmation(session, "Q1_NAME") is None


def test_soft_callback_redirect_preserves_readback():
    session = _session(current_state="Q1_NAME")
    session.pending_confirmation = {
        "field": "full_name",
        "state": "Q1_NAME",
        "value": "John Smith",
        "attempts": 1,
    }
    text = soft_callback_redirect_text(session)
    assert "No problem" in text
    assert "John Smith" in text
    assert "Before you go" not in text
    assert session.pending_confirmation is not None
    assert session.control_flags.get("callback_soft_noted") is True


def test_soft_callback_redirect_without_readback_uses_question():
    session = _session(current_state="Q1_NAME")
    text = soft_callback_redirect_text(session)
    assert "No problem" in text
    assert "Before you go" in text
    assert "full legal name" in text.lower()
    assert session.pending_confirmation is None
    assert "callback_soft_noted" not in session.control_flags


@pytest.mark.asyncio
async def test_soft_callback_during_readback_keeps_pending(monkeypatch):
    from app.core.call_handler import process_tenant_speech

    session = _session(current_state="Q1_NAME")
    session.pending_confirmation = {
        "field": "full_name",
        "state": "Q1_NAME",
        "value": "John Smith",
        "attempts": 1,
    }

    async def _fake_llm(*_args, **_kwargs):
        return {
            "response_text": "",
            "understood": False,
            "intent": "callback",
            "relevance": "relevant",
            "question_complete": False,
            "extracted_data": {},
        }

    async def _fake_synthesize(*_args, **_kwargs):
        return [b"ok"]

    monkeypatch.setattr(
        "app.core.call_handler.get_llm_response_with_fallback", _fake_llm
    )
    monkeypatch.setattr(
        "app.core.call_handler.synthesize_speech_parts", _fake_synthesize
    )

    response_text, _audio_parts, complete = await process_tenant_speech(
        session, "can you call me back later?"
    )

    assert complete is False
    assert session.pending_confirmation is not None
    assert session.pending_confirmation["value"] == "John Smith"
    assert session.control_flags.get("callback_redirect_offered") is True
    assert "John Smith" in response_text
    assert "No problem" in response_text


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


def test_handle_turn_timeout_rolls_back_advanced_state_and_data():
    session = _session(current_state="Q1_NAME")
    snapshot = capture_turn_snapshot(session)
    session.extracted_data["full_name"] = "Jane Doe"
    session.raw_answers["Q1_NAME"] = "Jane Doe"
    session.current_state = "Q2_PHONE"
    session.answered_states.append("Q1_NAME")
    session.questions_answered = 1

    text = handle_turn_timeout(session, "Jane Doe", snapshot)

    assert session.current_state == "Q1_NAME"
    assert "full_name" not in session.extracted_data
    assert session.answered_states == []
    assert session.questions_answered == 0
    assert "full legal name" in text.lower()


def test_handle_turn_timeout_reconcile_drops_orphan_user_without_speech():
    session = _session(current_state="Q1_NAME")
    snapshot = capture_turn_snapshot(session)
    session.add_message("user", "Jane Doe")
    session.add_transcript("Tenant", "Jane Doe")

    handle_turn_timeout(session, "Jane Doe", snapshot)

    assert session.messages == []
    assert session.transcript == []


def test_handle_turn_timeout_keeps_partial_streaming_pair_in_history():
    session = _session(current_state="Q1_NAME")
    snapshot = capture_turn_snapshot(session)
    session.add_message("user", "Jane Doe")
    session.add_transcript("Tenant", "Jane Doe")
    session.streamed_speakable_prefix = "Thanks, Jane"
    session.streaming_ai_open = True
    session.add_transcript("AI", "Thanks, Jane")

    handle_turn_timeout(session, "Jane Doe", snapshot)

    assert session.messages == [
        {"role": "user", "content": "Jane Doe"},
        {"role": "assistant", "content": "Thanks, Jane"},
    ]
    assert session.transcript[-1].speaker == "AI"
    assert session.transcript[-1].text == "Thanks, Jane"


def test_plan_timeout_recovery_uses_answered_state_not_current():
    session = _session(current_state="Q2_PHONE")
    session.extracted_data["full_name"] = "Jane Doe"
    text = plan_turn_timeout_recovery(
        session,
        "Jane Doe",
        answered_state="Q1_NAME",
    )
    assert "Jane Doe" in text
    assert session.pending_confirmation is not None
    assert session.pending_confirmation["state"] == "Q1_NAME"
