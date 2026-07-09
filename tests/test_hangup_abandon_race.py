"""Multi-worker hangup / abandon race guards."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core import call_handler
from app.core.conversation import ConversationSession


@pytest.mark.asyncio
async def test_mark_call_abandoned_if_active_updates_in_progress(monkeypatch):
    from app.db.crud import mark_call_abandoned_if_active

    db = AsyncMock()
    result = MagicMock()
    result.rowcount = 1
    db.execute = AsyncMock(return_value=result)
    db.commit = AsyncMock()

    assert await mark_call_abandoned_if_active(db, "v3:call-1") is True
    db.execute.assert_awaited_once()
    db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_mark_call_abandoned_if_active_noops_on_terminal(monkeypatch):
    from app.db.crud import mark_call_abandoned_if_active

    db = AsyncMock()
    result = MagicMock()
    result.rowcount = 0
    db.execute = AsyncMock(return_value=result)
    db.commit = AsyncMock()

    assert await mark_call_abandoned_if_active(db, "v3:call-2") is False


@pytest.mark.asyncio
async def test_finalize_after_stream_timeout_skips_when_finalize_inflight(monkeypatch):
    monkeypatch.setattr(call_handler, "get_session", lambda _cid: None)
    monkeypatch.setattr(call_handler.asyncio, "sleep", AsyncMock())
    monkeypatch.setattr(
        "app.core.redis_client.is_finalize_inflight",
        AsyncMock(return_value=True),
    )
    mark = AsyncMock(return_value=True)
    monkeypatch.setattr("app.db.crud.mark_call_abandoned_if_active", mark)

    call = MagicMock(status="in_progress")
    monkeypatch.setattr(
        "app.db.crud.get_call_by_call_id",
        AsyncMock(return_value=call),
    )

    await call_handler.finalize_after_stream_timeout("v3:inflight", timeout=0.01)

    mark.assert_not_awaited()


@pytest.mark.asyncio
async def test_finalize_after_stream_timeout_preserves_in_progress_without_session(
    monkeypatch,
):
    monkeypatch.setattr(call_handler, "get_session", lambda _cid: None)
    monkeypatch.setattr(call_handler.asyncio, "sleep", AsyncMock())
    monkeypatch.setattr(
        "app.core.redis_client.is_finalize_inflight",
        AsyncMock(return_value=False),
    )
    monkeypatch.setattr(
        call_handler,
        "check_stream_stop_signal",
        AsyncMock(return_value=False),
    )
    mark = AsyncMock(return_value=True)
    monkeypatch.setattr("app.db.crud.mark_call_abandoned_if_active", mark)

    call = MagicMock(status="in_progress")
    monkeypatch.setattr(
        "app.db.crud.get_call_by_call_id",
        AsyncMock(return_value=call),
    )

    await call_handler.finalize_after_stream_timeout("v3:abandon", timeout=0.01)

    # Preserve in_progress to avoid racing a late successful finalize.
    mark.assert_not_awaited()


@pytest.mark.asyncio
async def test_finalize_call_sets_and_clears_inflight_marker(monkeypatch):
    session = ConversationSession(call_id="v3:finalize", phone_number="+15551234567")
    set_marker = AsyncMock()
    clear_marker = AsyncMock()
    monkeypatch.setattr("app.core.redis_client.set_finalize_inflight", set_marker)
    monkeypatch.setattr("app.core.redis_client.clear_finalize_inflight", clear_marker)
    monkeypatch.setattr(
        call_handler,
        "_finalize_call_impl",
        AsyncMock(return_value={"call_id": "v3:finalize", "status": "qualified"}),
    )

    db = AsyncMock()
    result = await call_handler.finalize_call(session, db)

    assert result["status"] == "qualified"
    set_marker.assert_awaited_once_with("v3:finalize")
    clear_marker.assert_awaited_once_with("v3:finalize")
