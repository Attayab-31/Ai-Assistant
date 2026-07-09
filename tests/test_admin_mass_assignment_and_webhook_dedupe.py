"""Security/reliability regressions for tenant update and webhook dedupe."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from starlette.requests import Request


@pytest.mark.asyncio
async def test_api_update_tenant_rejects_unknown_fields(monkeypatch):
    from app.api import admin as admin_api

    tenant = SimpleNamespace(id="tenant-1", normalized_data={})
    monkeypatch.setattr(admin_api.crud, "get_tenant_by_id", AsyncMock(return_value=tenant))
    update_tenant = AsyncMock()
    monkeypatch.setattr(admin_api.crud, "update_tenant", update_tenant)
    monkeypatch.setattr(admin_api, "_rescore_tenant", AsyncMock(return_value={}))
    monkeypatch.setattr(admin_api.crud, "create_audit_log", AsyncMock())

    payload = admin_api.TenantUpdateRequest.model_validate(
        {"full_name": "Jane Doe", "qualification_status": "qualified"}
    )
    request = SimpleNamespace(client=SimpleNamespace(host="127.0.0.1"))
    user = SimpleNamespace(id="admin-1")

    with pytest.raises(HTTPException, match="Unsupported tenant field: qualification_status"):
        await admin_api.api_update_tenant(
            tenant_id="tenant-1",
            payload=payload,
            request=request,
            db=object(),
            user=user,
        )

    update_tenant.assert_not_awaited()


@pytest.mark.asyncio
async def test_webhook_dedupe_fail_open_when_redis_unavailable(monkeypatch):
    from app.api import webhook

    monkeypatch.setattr("app.core.redis_client.get_redis", lambda: None)

    first = await webhook._dedupe_webhook_event(
        "call-123",
        "webhook:initiated:call-123",
        log_label="call.initiated",
    )
    assert first is True


@pytest.mark.asyncio
async def test_webhook_dedupe_still_blocks_real_duplicates(monkeypatch):
    from app.api import webhook

    mock_redis = AsyncMock()
    mock_redis.set.return_value = False  # NX key already exists
    monkeypatch.setattr("app.core.redis_client.get_redis", lambda: mock_redis)

    first = await webhook._dedupe_webhook_event(
        "call-dup",
        "webhook:hangup:call-dup",
        log_label="call.hangup",
    )
    assert first is False


@pytest.mark.asyncio
async def test_telnyx_webhook_retries_failed_known_handler(monkeypatch):
    from app.api import webhook

    payload = {
        "data": {
            "event_type": "call.answered",
            "payload": {"call_control_id": "call-retry"},
        }
    }
    monkeypatch.setattr(
        webhook,
        "verify_webhook",
        AsyncMock(return_value=json.dumps(payload).encode()),
    )
    monkeypatch.setattr(
        webhook,
        "handle_call_answered_event",
        AsyncMock(side_effect=RuntimeError("transient database error")),
    )
    cache_delete = AsyncMock()
    monkeypatch.setattr("app.core.redis_client.cache_delete", cache_delete)

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/telnyx/webhook",
            "headers": [],
            "client": ("127.0.0.1", 12345),
        }
    )
    with pytest.raises(HTTPException) as exc:
        await webhook.telnyx_webhook(request, db=object())

    assert exc.value.status_code == 500
    cache_delete.assert_awaited_once_with("webhook:answered:call-retry")


@pytest.mark.asyncio
async def test_call_initiated_duplicate_integrity_is_idempotent(monkeypatch):
    from app.api import webhook
    from sqlalchemy.exc import IntegrityError

    monkeypatch.setattr(webhook, "_dedupe_webhook_event", AsyncMock(return_value=True))
    monkeypatch.setattr(webhook, "is_number_blacklisted", AsyncMock(return_value=False))
    monkeypatch.setattr(webhook, "get_setting_value", AsyncMock(return_value=False))
    monkeypatch.setattr(
        webhook,
        "create_call",
        AsyncMock(side_effect=IntegrityError("dup", params=None, orig=None)),
    )
    monkeypatch.setattr(
        webhook,
        "get_call_by_call_id",
        AsyncMock(return_value=SimpleNamespace(call_id="call-dup")),
    )
    answer = AsyncMock()
    monkeypatch.setattr(webhook.telnyx_service, "answer_call", answer)

    db = AsyncMock()
    await webhook.handle_call_initiated(
        db,
        {
            "call_control_id": "call-dup",
            "from": "+15551234567",
            "direction": "incoming",
        },
    )

    db.rollback.assert_awaited_once()
    answer.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("event_type", ["call.initiated", "call.answered", "call.hangup"])
async def test_webhook_rejects_empty_call_control_id(monkeypatch, event_type):
    from app.api import webhook

    payload = {"data": {"event_type": event_type, "payload": {"call_control_id": ""}}}
    monkeypatch.setattr(
        webhook,
        "verify_webhook",
        AsyncMock(return_value=json.dumps(payload).encode()),
    )
    # Handlers must never be reached for an unidentifiable call.
    monkeypatch.setattr(webhook, "handle_call_initiated", AsyncMock())
    monkeypatch.setattr(webhook, "handle_call_answered_event", AsyncMock())
    monkeypatch.setattr(webhook, "handle_call_hangup", AsyncMock())

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/telnyx/webhook",
            "headers": [],
            "client": ("127.0.0.1", 12345),
        }
    )
    with pytest.raises(HTTPException) as exc:
        await webhook.telnyx_webhook(request, db=object())

    assert exc.value.status_code == 400
    webhook.handle_call_initiated.assert_not_awaited()
    webhook.handle_call_answered_event.assert_not_awaited()
    webhook.handle_call_hangup.assert_not_awaited()


@pytest.mark.asyncio
async def test_webhook_recording_saved_requires_identifier(monkeypatch):
    from app.api import webhook

    payload = {
        "data": {
            "event_type": "call.recording.saved",
            "payload": {"call_control_id": "", "recording_id": ""},
        }
    }
    monkeypatch.setattr(
        webhook,
        "verify_webhook",
        AsyncMock(return_value=json.dumps(payload).encode()),
    )
    monkeypatch.setattr(webhook, "handle_recording_saved", AsyncMock())

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/telnyx/webhook",
            "headers": [],
            "client": ("127.0.0.1", 12345),
        }
    )
    with pytest.raises(HTTPException) as exc:
        await webhook.telnyx_webhook(request, db=object())

    assert exc.value.status_code == 400
    webhook.handle_recording_saved.assert_not_awaited()


def _request_with_body(body: bytes, headers: list[tuple[bytes, bytes]] | None = None):
    headers = headers or []

    async def _receive():
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/telnyx/webhook",
            "headers": headers,
            "client": ("127.0.0.1", 12345),
        },
        _receive,
    )


@pytest.mark.asyncio
async def test_verify_webhook_rejects_unsigned_when_not_explicitly_allowed(monkeypatch):
    from app.api import webhook

    monkeypatch.setattr(webhook.settings, "environment", "development")
    monkeypatch.setattr(webhook.settings, "telnyx_public_key", "")
    monkeypatch.setattr(webhook.settings, "allow_unsigned_webhooks_in_dev", False)

    req = _request_with_body(b'{"data":{"event_type":"call.initiated"}}')
    with pytest.raises(HTTPException) as exc:
        await webhook.verify_webhook(req)

    assert exc.value.status_code == 503


@pytest.mark.asyncio
async def test_verify_webhook_allows_unsigned_only_when_opted_in(monkeypatch):
    from app.api import webhook

    monkeypatch.setattr(webhook.settings, "environment", "development")
    monkeypatch.setattr(webhook.settings, "telnyx_public_key", "")
    monkeypatch.setattr(webhook.settings, "allow_unsigned_webhooks_in_dev", True)
    body = b'{"data":{"event_type":"call.initiated"}}'

    req = _request_with_body(body)
    assert await webhook.verify_webhook(req) == body
