"""Empty STT strike handling — shared by live streaming and batch paths."""

import pytest

from app.core.audio_stream import (
    STT_EMPTY_RETRY_TEXT,
    STT_EMPTY_RETRY_TEXT_ES,
    handle_stt_empty_transcript,
    stt_empty_retry_text,
)
from app.core.conversation import STT_EMPTY_STRIKE_LIMIT, CallState, ConversationSession


def _session(**kwargs) -> ConversationSession:
    return ConversationSession(
        call_id="test-empty-stt",
        phone_number="+15551234567",
        current_state="Q1_NAME",
        **kwargs,
    )


@pytest.mark.asyncio
async def test_empty_stt_retries_before_limit(monkeypatch):
    session = _session()
    emitted: list[str] = []
    enqueued: list[list[bytes]] = []

    async def capture_emit(text: str) -> None:
        emitted.append(text)

    async def capture_enqueue(parts: list[bytes]) -> None:
        enqueued.append(parts)

    async def fake_tts(text, _session, **kwargs):
        return b"\xff" * 40

    monkeypatch.setattr(
        "app.core.audio_stream.call_handler.synthesize_with_fallback",
        fake_tts,
    )

    should_end, retry_queued = await handle_stt_empty_transcript(
        session,
        call_id=session.call_id,
        log_detail="(streaming)",
        emit_response=capture_emit,
        enqueue_retry=capture_enqueue,
    )

    assert should_end is False
    assert retry_queued is True
    assert session.stt_empty_strikes == 1
    assert session.transcript[-1].text == STT_EMPTY_RETRY_TEXT
    assert emitted == [STT_EMPTY_RETRY_TEXT]
    assert enqueued == [[b"\xff" * 40]]


@pytest.mark.asyncio
async def test_empty_stt_ends_call_at_strike_limit(monkeypatch):
    session = _session()
    session.stt_empty_strikes = STT_EMPTY_STRIKE_LIMIT - 1
    enqueued: list[list[bytes]] = []

    async def capture_enqueue(parts: list[bytes]) -> None:
        enqueued.append(parts)

    async def noop_playback() -> None:
        return None

    async def fake_tts(text, _session, **kwargs):
        return b"\xff" * 40

    monkeypatch.setattr(
        "app.core.audio_stream.call_handler.synthesize_with_fallback",
        fake_tts,
    )

    should_end, retry_queued = await handle_stt_empty_transcript(
        session,
        call_id=session.call_id,
        enqueue_retry=capture_enqueue,
        await_playback_done=noop_playback,
    )

    assert should_end is True
    assert retry_queued is False
    assert session.stt_empty_strikes == STT_EMPTY_STRIKE_LIMIT
    assert session.current_state == CallState.ENDED.value
    assert session.control_flags.get("provider_failure", {}).get("service") == "stt"
    assert len(enqueued) == 1


@pytest.mark.asyncio
async def test_empty_stt_without_tts_audio_opens_listen_gate(monkeypatch):
    session = _session()

    async def no_tts(text, _session, **kwargs):
        return b""

    monkeypatch.setattr(
        "app.core.audio_stream.call_handler.synthesize_with_fallback",
        no_tts,
    )

    should_end, retry_queued = await handle_stt_empty_transcript(
        session,
        call_id=session.call_id,
    )

    assert should_end is False
    assert retry_queued is False
    assert session.stt_empty_strikes == 1


@pytest.mark.asyncio
async def test_empty_stt_retry_uses_spanish_when_call_language_es(monkeypatch):
    session = _session(call_language="es")
    emitted: list[str] = []

    async def capture_emit(text: str) -> None:
        emitted.append(text)

    async def fake_tts(text, _session, **kwargs):
        assert text == STT_EMPTY_RETRY_TEXT_ES
        return b"\xff" * 40

    monkeypatch.setattr(
        "app.core.audio_stream.call_handler.synthesize_with_fallback",
        fake_tts,
    )

    should_end, retry_queued = await handle_stt_empty_transcript(
        session,
        call_id=session.call_id,
        emit_response=capture_emit,
    )

    assert should_end is False
    assert retry_queued is False
    assert stt_empty_retry_text(session) == STT_EMPTY_RETRY_TEXT_ES
    assert session.transcript[-1].text == STT_EMPTY_RETRY_TEXT_ES
    assert emitted == [STT_EMPTY_RETRY_TEXT_ES]
