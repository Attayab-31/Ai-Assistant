"""Cross-worker hangup stop when Redis is unavailable."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.db.crud import (
    STREAM_STOP_DB_KEY,
    clear_stream_stop_request,
    is_stream_stop_requested,
    persist_stream_stop_request,
)


@pytest.mark.asyncio
async def test_persist_stream_stop_request_writes_error_log():
    call_id = "v3:hangup-db"
    db = AsyncMock()
    result = MagicMock()
    result.rowcount = 1
    db.execute = AsyncMock(return_value=result)
    db.commit = AsyncMock()

    ok = await persist_stream_stop_request(db, call_id)

    assert ok is True
    db.execute.assert_awaited_once()
    db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_persist_stream_stop_request_ignores_terminal_calls():
    db = AsyncMock()
    result = MagicMock()
    result.rowcount = 0
    db.execute = AsyncMock(return_value=result)
    db.commit = AsyncMock()

    ok = await persist_stream_stop_request(db, "done-call")

    assert ok is False
    db.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_is_stream_stop_requested_reads_flag():
    call = MagicMock()
    call.error_log = {STREAM_STOP_DB_KEY: "2026-07-09T00:00:00+00:00"}
    db = AsyncMock()

    with patch("app.db.crud.get_call_by_call_id", AsyncMock(return_value=call)):
        assert await is_stream_stop_requested(db, "live-call") is True


@pytest.mark.asyncio
async def test_clear_stream_stop_request_removes_flag():
    call = MagicMock()
    call.error_log = {STREAM_STOP_DB_KEY: "2026-07-09T00:00:00+00:00"}
    db = AsyncMock()
    db.execute = AsyncMock()
    db.commit = AsyncMock()

    with patch("app.db.crud.get_call_by_call_id", AsyncMock(return_value=call)):
        await clear_stream_stop_request(db, "live-call")

    db.execute.assert_awaited_once()
    db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_request_stream_stop_persists_db_when_no_local_stream():
    from app.core import call_handler

    call_id = "v3:remote-stream"
    with patch("app.core.redis_client.ping", AsyncMock(return_value=False)):
        with patch(
            "app.core.call_handler._persist_stream_stop_request",
            AsyncMock(),
        ) as persist:
            local = await call_handler.request_stream_stop(call_id)

    assert local is False
    persist.assert_awaited_once_with(call_id)


@pytest.mark.asyncio
async def test_request_stream_stop_skips_db_when_local_stream_exists():
    from app.core import call_handler

    call_id = "v3:local-stream"
    stop = asyncio.Event()
    call_handler.register_stream_stop(call_id, stop)
    try:
        with patch("app.core.redis_client.ping", AsyncMock(return_value=False)):
            with patch(
                "app.core.call_handler._persist_stream_stop_request",
                AsyncMock(),
            ) as persist:
                local = await call_handler.request_stream_stop(call_id)

        assert local is True
        assert stop.is_set()
        persist.assert_not_awaited()
    finally:
        call_handler.unregister_stream_stop(call_id)


@pytest.mark.asyncio
async def test_check_stream_stop_signal_uses_db_when_redis_down():
    from app.core import call_handler
    from app.core.conversation import ConversationSession

    call_id = "v3:poll-db"
    stop = asyncio.Event()
    session = ConversationSession(call_id=call_id, phone_number="+1")
    call_handler._active_sessions[call_id] = session
    call_handler.register_stream_stop(call_id, stop)
    try:
        with patch(
            "app.core.redis_client.is_stream_stop_signaled",
            AsyncMock(return_value=False),
        ):
            with patch("app.core.redis_client.ping", AsyncMock(return_value=False)):
                with patch(
                    "app.core.call_handler._is_stream_stop_requested_db",
                    AsyncMock(return_value=True),
                ):
                    signaled = await call_handler.check_stream_stop_signal(call_id)

        assert signaled is True
        assert stop.is_set()
        assert session.pending_hangup is True
    finally:
        call_handler._active_sessions.pop(call_id, None)
        call_handler.unregister_stream_stop(call_id)


@pytest.mark.asyncio
async def test_check_stream_stop_signal_checks_db_even_when_redis_up():
    from app.core import call_handler

    call_id = "v3:redis-up-db-stop"
    with patch(
        "app.core.redis_client.is_stream_stop_signaled",
        AsyncMock(return_value=False),
    ):
        with patch(
            "app.core.call_handler._is_stream_stop_requested_db",
            AsyncMock(return_value=True),
        ) as db_check:
            signaled = await call_handler.check_stream_stop_signal(call_id)

    assert signaled is True
    db_check.assert_awaited_once_with(call_id)


@pytest.mark.asyncio
async def test_clear_stream_stop_signals_clears_redis_and_db():
    from app.core import call_handler

    call_id = "v3:clear"
    with patch(
        "app.core.redis_client.clear_stream_stop_signal", AsyncMock()
    ) as clear_redis:
        mock_db = AsyncMock()
        session_cm = MagicMock()
        session_cm.__aenter__ = AsyncMock(return_value=mock_db)
        session_cm.__aexit__ = AsyncMock(return_value=None)
        with patch(
            "app.db.database.AsyncSessionLocal",
            MagicMock(return_value=session_cm),
        ):
            with patch(
                "app.db.crud.clear_stream_stop_request", AsyncMock()
            ) as clear_db:
                await call_handler.clear_stream_stop_signals(call_id)

    clear_redis.assert_awaited_once_with(call_id)
    clear_db.assert_awaited_once_with(mock_db, call_id)
