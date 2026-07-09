"""Streaming STT reconnect / batch-degrade helpers."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.streaming_stt import (
    STREAMING_STT_RECONNECT_TIMEOUT_S,
    DeepgramStreamingSession,
    StreamingSttRelay,
)


def test_deepgram_session_marks_lost_on_error():
    session = DeepgramStreamingSession()
    assert session.lost is False
    session._lost = True
    assert session.lost is True


@pytest.mark.asyncio
async def test_streaming_relay_restart_replaces_inner_session():
    relay = StreamingSttRelay(model="nova-3", encoding="mulaw", language="en-US")
    first = MagicMock()
    first.close = AsyncMock()
    first.lost = False
    replacement = MagicMock()
    replacement.lost = False
    replacement.start = AsyncMock()
    replacement.close = AsyncMock()

    with patch.object(relay, "_build_session", return_value=replacement) as build:
        with patch.object(relay, "_connect_lock", asyncio.Lock()):
            relay._current = first
            await relay.restart()

    first.close.assert_awaited_once()
    build.assert_called_once()
    assert relay._current is replacement
    replacement.start.assert_awaited_once()
    await relay.close()


@pytest.mark.asyncio
async def test_streaming_relay_continues_forwarding_after_restart():
    relay = StreamingSttRelay(model="nova-3", encoding="mulaw", language="en-US")

    class FakeSession:
        def __init__(self) -> None:
            self.lost = False
            self.queue = asyncio.Queue()
            self.start = AsyncMock()
            self.feed = AsyncMock()

        async def close(self) -> None:
            self.lost = True
            await self.queue.put(None)

        async def transcripts(self):
            while True:
                item = await self.queue.get()
                if item is None:
                    break
                yield item

    first = FakeSession()
    replacement = FakeSession()
    relay._current = first
    relay._ensure_relay_task()

    first.lost = True
    await first.queue.put(None)

    transcript_stream = relay.transcripts()
    assert await asyncio.wait_for(anext(transcript_stream), timeout=1.0) is None

    with patch.object(relay, "_build_session", return_value=replacement):
        await relay.restart()

    await replacement.queue.put("after restart")
    assert await asyncio.wait_for(anext(transcript_stream), timeout=1.0) == "after restart"

    await relay.close()
    await transcript_stream.aclose()


@pytest.mark.asyncio
async def test_streaming_relay_buffers_audio_during_language_reconnect():
    relay = StreamingSttRelay(model="nova-3", encoding="mulaw", language="en-US")
    close_started = asyncio.Event()
    close_continue = asyncio.Event()

    first = MagicMock()
    first.lost = False

    async def close_first():
        close_started.set()
        await close_continue.wait()

    first.close = AsyncMock(side_effect=close_first)

    replacement = MagicMock()
    replacement.lost = False
    replacement.start = AsyncMock()
    replacement.feed = AsyncMock()
    replacement.close = AsyncMock()

    with patch.object(relay, "_build_session", return_value=replacement):
        relay._current = first
        reconnect_task = asyncio.create_task(relay.reconnect(language="es"))
        await asyncio.wait_for(close_started.wait(), timeout=1.0)

        await relay.feed(b"during-reconnect-1")
        await relay.feed(b"during-reconnect-2")
        close_continue.set()
        await reconnect_task

    first.close.assert_awaited_once()
    replacement.start.assert_awaited_once()
    assert relay._current is replacement
    assert [call.args[0] for call in replacement.feed.await_args_list] == [
        b"during-reconnect-1",
        b"during-reconnect-2",
    ]
    await relay.close()


@pytest.mark.asyncio
async def test_streaming_relay_buffers_audio_during_restart():
    relay = StreamingSttRelay(model="nova-3", encoding="mulaw", language="en-US")
    close_started = asyncio.Event()
    close_continue = asyncio.Event()

    first = MagicMock()
    first.lost = False

    async def close_first():
        close_started.set()
        await close_continue.wait()

    first.close = AsyncMock(side_effect=close_first)

    replacement = MagicMock()
    replacement.lost = False
    replacement.start = AsyncMock()
    replacement.feed = AsyncMock()
    replacement.close = AsyncMock()

    with patch.object(relay, "_build_session", return_value=replacement):
        relay._current = first
        restart_task = asyncio.create_task(relay.restart())
        await asyncio.wait_for(close_started.wait(), timeout=1.0)

        await relay.feed(b"restart-gap")
        close_continue.set()
        await restart_task

    assert [call.args[0] for call in replacement.feed.await_args_list] == [
        b"restart-gap"
    ]
    await relay.close()


@pytest.mark.asyncio
async def test_deepgram_feed_send_failure_marks_lost_and_signals_queue():
    session = DeepgramStreamingSession()
    session._connection = MagicMock()
    session._connection.send = AsyncMock(side_effect=RuntimeError("socket dead"))

    await session.feed(b"audio")

    assert session.lost is True
    assert session._connection is None
    assert session._transcript_queue.get_nowait() is None


@pytest.mark.asyncio
async def test_deepgram_keepalive_send_failure_marks_lost():
    session = DeepgramStreamingSession()
    session._connection = MagicMock()
    session._connection.send = AsyncMock(side_effect=OSError("broken pipe"))
    session._last_feed_at = 0.0

    with patch("app.core.streaming_stt.asyncio.sleep", new_callable=AsyncMock):
        await session._keepalive_loop()

    assert session.lost is True
    assert session._connection is None


@pytest.mark.asyncio
async def test_streaming_relay_feed_propagates_inner_lost():
    relay = StreamingSttRelay(model="nova-3", encoding="mulaw", language="en-US")
    inner = MagicMock()
    inner.lost = False
    inner.feed = AsyncMock(side_effect=lambda _chunk: setattr(inner, "lost", True))
    relay._current = inner

    await relay.feed(b"chunk")

    assert relay.lost is True
    assert relay._transcript_queue.get_nowait() is None


@pytest.mark.asyncio
async def test_streaming_relay_lost_when_inner_session_lost():
    relay = StreamingSttRelay(model="nova-3", encoding="mulaw", language="en-US")
    inner = DeepgramStreamingSession()
    inner._lost = True
    relay._current = inner
    assert relay.lost is True


@pytest.mark.asyncio
async def test_streaming_relay_reconnect_keeps_old_session_when_start_fails():
    relay = StreamingSttRelay(model="nova-3", encoding="mulaw", language="en-US")
    first = MagicMock()
    first.lost = False
    first.close = AsyncMock()
    relay._current = first

    replacement = MagicMock()
    replacement.start = AsyncMock(side_effect=RuntimeError("socket refused"))

    with patch.object(relay, "_build_session", return_value=replacement):
        with pytest.raises(RuntimeError, match="socket refused"):
            await relay.reconnect(language="es")

    assert relay._current is first
    assert relay._buffering_reconnect is False
    first.close.assert_not_awaited()


@pytest.mark.asyncio
async def test_apply_session_language_switches_tts_after_stt_reconnect():
    from app.core.call_handler import _apply_session_language
    from app.core.conversation import ConversationSession
    from app.providers.tts.deepgram_tts import DeepgramTTSProvider

    session = ConversationSession(
        call_id="lang-ok",
        phone_number="+1",
        call_language="en",
        tts_voice_en="aura-2-thalia-en",
        tts_voice_deepgram_es="aura-2-estrella-es",
    )
    relay = MagicMock()
    relay.lost = False
    relay.reconnect = AsyncMock()
    session.streaming_stt_relay = relay

    tts = DeepgramTTSProvider(voice="aura-2-thalia-en")
    providers = MagicMock()
    providers.stt = MagicMock(language="en-US")
    providers.tts_by_name = {"deepgram": tts}

    with patch("app.core.call_handler.get_call_providers", return_value=providers):
        await _apply_session_language(session, "es")

    relay.reconnect.assert_awaited_once_with(language="es")
    assert session.call_language == "es"
    assert tts.voice == "aura-2-estrella-es"


@pytest.mark.asyncio
async def test_apply_session_language_applies_language_when_stt_reconnect_fails():
    from app.core.call_handler import _apply_session_language
    from app.core.conversation import ConversationSession
    from app.providers.tts.deepgram_tts import DeepgramTTSProvider

    session = ConversationSession(
        call_id="lang-fail",
        phone_number="+1",
        call_language="en",
        tts_voice_en="aura-2-thalia-en",
        tts_voice_deepgram_es="aura-2-estrella-es",
    )
    relay = MagicMock()
    relay.lost = False
    relay.reconnect = AsyncMock(side_effect=TimeoutError())
    relay.close = AsyncMock()
    session.streaming_stt_relay = relay

    tts = DeepgramTTSProvider(voice="aura-2-thalia-en")
    providers = MagicMock()
    providers.stt = MagicMock(language="en-US")
    providers.tts_by_name = {"deepgram": tts}

    with patch("app.core.call_handler.get_call_providers", return_value=providers):
        await _apply_session_language(session, "es")

    assert session.call_language == "es"
    assert tts.voice == "aura-2-estrella-es"
    relay.close.assert_awaited_once()
    assert any(e.get("type") == "stt_language_sync_failed" for e in session.errors)


def test_audio_stream_reconnect_constants():
    from app.core.audio_stream import (
        STT_STREAM_RECONNECT_FLAG,
        STREAMING_STT_RECONNECT_TIMEOUT_S,
    )

    assert STREAMING_STT_RECONNECT_TIMEOUT_S > 0
    assert STT_STREAM_RECONNECT_FLAG
