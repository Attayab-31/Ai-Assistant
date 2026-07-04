"""In-memory session cleanup when a live audio stream ends."""

from datetime import UTC, datetime

from app.core.call_handler import (
    _active_sessions,
    finish_stream_session,
    get_active_sessions,
    remove_session,
)
from app.core.conversation import CallState, ConversationSession


def _register(session: ConversationSession) -> None:
    _active_sessions[session.call_id] = session


def _cleanup_call_id(call_id: str) -> None:
    _active_sessions.pop(call_id, None)


def test_get_active_sessions_hides_stream_ended():
    sid = "test-stream-ended"
    session = ConversationSession(call_id=sid, phone_number="+15550100")
    session.stream_ended_at = datetime.now(UTC)
    _register(session)
    try:
        assert get_active_sessions() == []
    finally:
        _cleanup_call_id(sid)


def test_finish_stream_session_removes_provider_failure():
    sid = "test-provider-fail"
    session = ConversationSession(call_id=sid, phone_number="+15550101")
    session.current_state = CallState.ENDED.value
    session.control_flags["provider_failure"] = {"service": "llm", "detail": "429"}
    _register(session)
    finish_stream_session(sid)
    assert get_session_missing(sid)
    assert get_active_sessions() == []


def test_finish_stream_session_keeps_completed_screening_for_end_and_score():
    sid = "test-complete-await-score"
    session = ConversationSession(
        call_id=sid,
        phone_number="+15550102",
        questions=[
            {
                "state": "Q1_NAME",
                "question": "Name?",
                "extract_fields": ["full_name"],
                "active": True,
                "order": 1,
            },
        ],
    )
    session.current_state = CallState.ENDED.value
    session.extracted_data["full_name"] = "Jane Doe"
    session.completed_states.add("Q1_NAME")
    _register(session)
    try:
        finish_stream_session(sid)
        assert sid in _active_sessions
        assert _active_sessions[sid].stream_ended_at is not None
        assert get_active_sessions() == []
    finally:
        remove_session(sid)


def get_session_missing(call_id: str) -> bool:
    return call_id not in _active_sessions
