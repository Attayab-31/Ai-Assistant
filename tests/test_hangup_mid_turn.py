"""Hangup cancels in-flight turns without mutating screening state."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.audio_stream import _await_turn_or_hangup
from app.core.conversation import (
    ConversationSession,
    capture_turn_snapshot,
    handle_hangup_cancelled_turn,
)


def _session(**kwargs) -> ConversationSession:
    questions = kwargs.pop(
        "questions",
        [
            {
                "state": "Q1_NAME",
                "question": "What is your full legal name?",
                "retry_prompt": "Please tell me your first and last name.",
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
    return ConversationSession(
        call_id="hangup-test",
        phone_number="+15551234567",
        questions=questions,
        **kwargs,
    )


def test_handle_hangup_cancelled_turn_restores_advanced_state():
    session = _session(current_state="Q1_NAME")
    snapshot = capture_turn_snapshot(session)
    session.extracted_data["full_name"] = "Jane Doe"
    session.current_state = "Q2_PHONE"
    session.answered_states.append("Q1_NAME")

    handle_hangup_cancelled_turn(session, snapshot)

    assert session.current_state == "Q1_NAME"
    assert "full_name" not in session.extracted_data
    assert session.answered_states == []


@pytest.mark.asyncio
async def test_await_turn_or_hangup_cancels_when_stop_event_set():
    session = _session()

    async def _slow_turn() -> tuple[str, list[bytes], bool]:
        await asyncio.sleep(30)
        return ("never", [], False)

    turn_task = asyncio.create_task(_slow_turn())
    stop_event = asyncio.Event()
    stop_event.set()

    result = await _await_turn_or_hangup(turn_task, stop_event, session=session)

    assert result is None
    assert turn_task.cancelled()


@pytest.mark.asyncio
async def test_await_turn_or_hangup_returns_completed_turn():
    session = _session()
    stop_event = asyncio.Event()

    async def _fast_turn() -> tuple[str, list[bytes], bool]:
        return ("Thanks", [b"audio"], False)

    turn_task = asyncio.create_task(_fast_turn())
    result = await _await_turn_or_hangup(turn_task, stop_event, session=session)

    assert result == ("Thanks", [b"audio"], False)


@pytest.mark.asyncio
async def test_await_turn_or_hangup_propagates_barge_in_cancel():
    session = _session()
    stop_event = asyncio.Event()

    async def _slow_turn() -> tuple[str, list[bytes], bool]:
        await asyncio.sleep(30)
        return ("never", [], False)

    turn_task = asyncio.create_task(_slow_turn())
    await asyncio.sleep(0)
    turn_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await _await_turn_or_hangup(turn_task, stop_event, session=session)


@pytest.mark.asyncio
async def test_finish_turn_noops_when_pending_hangup(monkeypatch):
    from app.core.call_handler import process_tenant_speech

    session = _session(current_state="Q1_NAME")
    session.pending_hangup = True

    async def _fake_llm(*_args, **_kwargs):
        return {
            "response_text": "Thanks!",
            "understood": True,
            "intent": "answer",
            "relevance": "relevant",
            "question_complete": True,
            "extracted_data": {"full_name": "Jane Doe"},
        }

    monkeypatch.setattr(
        "app.core.call_handler.get_llm_response_with_fallback", _fake_llm
    )
    monkeypatch.setattr(
        "app.core.call_handler.synthesize_speech_parts",
        AsyncMock(return_value=[b"audio"]),
    )

    text, audio, complete = await process_tenant_speech(session, "Jane Doe")

    assert text == ""
    assert audio == []
    assert complete is False
    assert "full_name" not in session.extracted_data


@pytest.mark.asyncio
async def test_merge_skipped_when_hangup_set_after_llm(monkeypatch):
    from app.core.call_handler import process_tenant_speech

    session = _session(current_state="Q1_NAME")

    async def _fake_llm(*_args, **_kwargs):
        session.pending_hangup = True
        return {
            "response_text": "Thanks!",
            "understood": True,
            "intent": "answer",
            "relevance": "relevant",
            "question_complete": True,
            "extracted_data": {"full_name": "Jane Doe"},
        }

    monkeypatch.setattr(
        "app.core.call_handler.get_llm_response_with_fallback", _fake_llm
    )

    text, audio, complete = await process_tenant_speech(session, "Jane Doe")

    assert text == ""
    assert "full_name" not in session.extracted_data
    assert complete is False
