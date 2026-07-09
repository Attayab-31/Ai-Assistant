"""Stream start vs call row readiness — phone backfill and DB polling."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.call_handler import (
    apply_call_phone_to_session,
    create_session,
    sync_session_phone_from_db,
)
from app.core.conversation import ConversationSession
from app.db.crud import wait_for_call_by_call_id


def test_apply_call_phone_to_session_skips_when_session_has_phone():
    session = ConversationSession(call_id="c1", phone_number="+15551234567")
    call = MagicMock(phone_number="+15559998888")
    assert apply_call_phone_to_session(session, call) is False
    assert session.phone_number == "+15551234567"


def test_apply_call_phone_to_session_backfills_from_call():
    session = ConversationSession(call_id="c2", phone_number="")
    call = MagicMock(phone_number="+15557654321")
    assert apply_call_phone_to_session(session, call) is True
    assert session.phone_number == "+15557654321"


def test_apply_call_phone_to_session_no_call():
    session = ConversationSession(call_id="c3", phone_number="")
    assert apply_call_phone_to_session(session, None) is False
    assert session.phone_number == ""


@pytest.mark.asyncio
async def test_sync_session_phone_from_db_uses_provided_call():
    session = ConversationSession(call_id="c4", phone_number="")
    call = MagicMock(phone_number="+15551112222")
    db = AsyncMock()
    assert await sync_session_phone_from_db(session, db, call=call) is True
    assert session.phone_number == "+15551112222"
    db.execute.assert_not_called()


@pytest.mark.asyncio
async def test_wait_for_call_by_call_id_returns_on_first_hit(monkeypatch):
    call = MagicMock(call_id="v3:abc")
    db = AsyncMock()
    monkeypatch.setattr(
        "app.db.crud.get_call_by_call_id",
        AsyncMock(side_effect=[call]),
    )
    result = await wait_for_call_by_call_id(db, "v3:abc", timeout=1.0)
    assert result is call


@pytest.mark.asyncio
async def test_wait_for_call_by_call_id_polls_until_found(monkeypatch):
    call = MagicMock(call_id="v3:late")
    get_call = AsyncMock(side_effect=[None, None, call])
    monkeypatch.setattr("app.db.crud.get_call_by_call_id", get_call)
    sleeps: list[float] = []

    async def _fast_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("app.db.crud.asyncio.sleep", _fast_sleep)
    db = AsyncMock()
    result = await wait_for_call_by_call_id(
        db, "v3:late", timeout=1.0, interval=0.05
    )
    assert result is call
    assert get_call.await_count == 3
    assert sleeps == [0.05, 0.05]


@pytest.mark.asyncio
async def test_create_session_backfills_phone_on_reuse(monkeypatch):
    from app.core import call_handler

    existing = ConversationSession(call_id="reuse-1", phone_number="")
    call_handler._active_sessions["reuse-1"] = existing

    snapshot = MagicMock()
    snapshot.property_name = "Test Property"
    snapshot.greeting_message = "Hi"
    snapshot.closing_message = "Bye"
    snapshot.provider_failure_message = "Sorry"
    snapshot.llm_temperature = 0.2
    snapshot.llm_max_tokens = 100
    snapshot.qualified_score_threshold = 70
    snapshot.review_score_threshold = 50
    snapshot.questions = []
    snapshot.faqs = []
    snapshot.max_retries = 2
    snapshot.silence_timeout_seconds = 12
    snapshot.max_call_duration_seconds = 600
    snapshot.auto_fallback_enabled = True
    snapshot.captured_at = "2026-01-01T00:00:00Z"
    snapshot.voice_latency_profile = "balanced"
    snapshot.llm_streaming_enabled = True
    snapshot.turn_timeout_seconds = 30
    snapshot.llm_timeout_voice_seconds = 20
    snapshot.deepgram_endpointing_ms = 300
    snapshot.deepgram_utterance_end_ms = 1000
    snapshot.latency_alert_turn_p95_ms = 5000
    snapshot.latency_alert_timeout_rate_pct = 10.0
    snapshot.tts_voice = "aura"
    snapshot.tts_provider = "deepgram"
    snapshot.tts_voice_deepgram_es = "aura-es"
    snapshot.tts_voice_google_es = "es-ES"
    snapshot.tts_voices_by_provider = {}
    snapshot.notification_settings = MagicMock()
    snapshot.questions_runtime_fallback = None

    bundle = MagicMock()
    bundle.stt_name = "deepgram"
    bundle.llm_name = "groq"
    bundle.tts_name = "deepgram"

    monkeypatch.setattr(
        call_handler, "load_call_settings_snapshot", AsyncMock(return_value=snapshot)
    )
    monkeypatch.setattr(
        call_handler, "build_call_provider_bundle", MagicMock(return_value=bundle)
    )
    monkeypatch.setattr(call_handler, "_prewarm_fallback_clients", lambda _b: None)

    db = AsyncMock()
    try:
        session = await create_session(
            call_id="reuse-1",
            phone_number="+15553334444",
            db=db,
        )
        assert session is existing
        assert session.phone_number == "+15553334444"
    finally:
        call_handler._active_sessions.pop("reuse-1", None)
