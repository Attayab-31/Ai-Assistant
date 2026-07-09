"""Call recording toggle is frozen at call.initiated."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_call_initiated_stores_recording_requested(monkeypatch):
    from app.api import webhook

    create_call = AsyncMock()
    monkeypatch.setattr(webhook, "create_call", create_call)
    monkeypatch.setattr(webhook, "_dedupe_webhook_event", AsyncMock(return_value=True))
    monkeypatch.setattr(webhook, "is_number_blacklisted", AsyncMock(return_value=False))

    async def _settings(db, key, default=None):
        if key == "blacklisted_numbers":
            return []
        if key == "call_recording_enabled":
            return True
        return default

    monkeypatch.setattr(webhook, "get_setting_value", _settings)
    monkeypatch.setattr(
        webhook.telnyx_service, "answer_call", AsyncMock(return_value={})
    )

    db = AsyncMock()
    await webhook.handle_call_initiated(
        db,
        {
            "call_control_id": "v3:rec-1",
            "from": "+15551234567",
            "direction": "incoming",
        },
    )

    create_call.assert_awaited_once()
    assert create_call.await_args.kwargs["recording_requested"] is True


@pytest.mark.asyncio
async def test_call_answered_uses_frozen_recording_flag(monkeypatch):
    from app.api import webhook

    call = MagicMock(recording_requested=True)
    monkeypatch.setattr(
        "app.db.crud.get_call_by_call_id", AsyncMock(return_value=call)
    )
    monkeypatch.setattr(webhook, "_dedupe_webhook_event", AsyncMock(return_value=True))
    start_recording = AsyncMock(return_value={})
    monkeypatch.setattr(webhook.telnyx_service, "start_recording", start_recording)
    monkeypatch.setattr(webhook.telnyx_service, "start_streaming", AsyncMock())
    monkeypatch.setattr("app.db.crud.update_call", AsyncMock())
    get_setting = AsyncMock(return_value=False)
    monkeypatch.setattr(webhook, "get_setting_value", get_setting)

    db = AsyncMock()
    with patch("app.utils.helpers.generate_stream_token", return_value="tok"):
        await webhook.handle_call_answered_event(
            db, {"call_control_id": "v3:rec-1"}
        )

    start_recording.assert_awaited_once_with("v3:rec-1")
    get_setting.assert_not_awaited()


@pytest.mark.asyncio
async def test_call_answered_skips_recording_when_frozen_off(monkeypatch):
    from app.api import webhook

    call = MagicMock(recording_requested=False)
    monkeypatch.setattr(
        "app.db.crud.get_call_by_call_id", AsyncMock(return_value=call)
    )
    monkeypatch.setattr(webhook, "_dedupe_webhook_event", AsyncMock(return_value=True))
    start_recording = AsyncMock()
    monkeypatch.setattr(webhook.telnyx_service, "start_recording", start_recording)
    monkeypatch.setattr(webhook.telnyx_service, "start_streaming", AsyncMock())
    monkeypatch.setattr("app.db.crud.update_call", AsyncMock())

    db = AsyncMock()
    with patch("app.utils.helpers.generate_stream_token", return_value="tok"):
        await webhook.handle_call_answered_event(
            db, {"call_control_id": "v3:rec-2"}
        )

    start_recording.assert_not_awaited()


@pytest.mark.asyncio
async def test_call_answered_duplicate_stream_start_is_idempotent(monkeypatch):
    from app.api import webhook

    call = MagicMock(recording_requested=False, status="in_progress")
    monkeypatch.setattr(
        "app.db.crud.get_call_by_call_id",
        AsyncMock(side_effect=[call, call]),
    )
    monkeypatch.setattr(webhook, "_dedupe_webhook_event", AsyncMock(return_value=True))
    monkeypatch.setattr(webhook.telnyx_service, "start_recording", AsyncMock())
    monkeypatch.setattr(
        webhook.telnyx_service,
        "start_streaming",
        AsyncMock(side_effect=RuntimeError("already streaming")),
    )
    update_call = AsyncMock()
    monkeypatch.setattr("app.db.crud.update_call", update_call)
    hangup = AsyncMock()
    monkeypatch.setattr(webhook.telnyx_service, "hangup_call", hangup)

    db = AsyncMock()
    with patch("app.utils.helpers.generate_stream_token", return_value="tok"):
        await webhook.handle_call_answered_event(
            db, {"call_control_id": "v3:dup-answered"}
        )

    # Duplicate answered must not fail/hangup a call that's already active.
    update_call.assert_not_awaited()
    hangup.assert_not_awaited()
