"""Reliability fixes for webhooks, blacklist, admin audit, and encryption."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from sqlalchemy.exc import IntegrityError


@pytest.mark.asyncio
async def test_recording_saved_skips_duplicate_delivery(monkeypatch):
    from app.api import webhook

    dedupe = AsyncMock(return_value=False)
    monkeypatch.setattr(webhook, "_dedupe_webhook_event", dedupe)
    update_call = AsyncMock()
    monkeypatch.setattr("app.db.crud.update_call", update_call)

    await webhook.handle_recording_saved(
        object(),
        {
            "call_control_id": "call-1",
            "recording_id": "rec-1",
            "recording_urls": {"mp3": "https://example.com/rec.mp3"},
        },
    )

    dedupe.assert_awaited_once_with(
        "rec-1",
        "webhook:recording:rec-1",
        log_label="call.recording.saved",
    )
    update_call.assert_not_awaited()


@pytest.mark.asyncio
async def test_recording_saved_dedupes_by_recording_id_when_call_id_missing(monkeypatch):
    from app.api import webhook

    dedupe = AsyncMock(return_value=False)
    monkeypatch.setattr(webhook, "_dedupe_webhook_event", dedupe)

    await webhook.handle_recording_saved(
        object(),
        {
            "call_control_id": "",
            "recording_id": "rec-only",
            "recording_urls": {"mp3": "https://example.com/rec.mp3"},
        },
    )

    dedupe.assert_awaited_once_with(
        "rec-only",
        "webhook:recording:rec-only",
        log_label="call.recording.saved",
    )


@pytest.mark.asyncio
async def test_remove_from_blacklist_sanitizes_path_phone(monkeypatch):
    from app.api import settings as settings_api

    removed: list[str] = []

    async def _fake_remove(_db, phone, **kwargs):
        removed.append(phone)
        return (["+15551234567"], True)

    monkeypatch.setattr(settings_api.crud, "remove_from_blacklist", _fake_remove)
    monkeypatch.setattr(settings_api, "_safe_create_audit_log", AsyncMock(return_value=True))

    request = SimpleNamespace(client=SimpleNamespace(host="127.0.0.1"))
    user = SimpleNamespace(id="admin-1")

    result = await settings_api.remove_from_blacklist(
        phone_number="(555) 123-4567",
        request=request,
        db=object(),
        user=user,
    )

    assert removed == ["+15551234567"]
    assert result["success"] is True
    assert "warnings" not in result


@pytest.mark.asyncio
async def test_remove_from_blacklist_rejects_invalid_phone():
    from app.api import settings as settings_api

    request = SimpleNamespace(client=SimpleNamespace(host="127.0.0.1"))
    user = SimpleNamespace(id="admin-1")

    with pytest.raises(HTTPException, match="Invalid phone number"):
        await settings_api.remove_from_blacklist(
            phone_number="not-a-phone",
            request=request,
            db=object(),
            user=user,
        )


@pytest.mark.asyncio
async def test_blacklist_add_returns_cache_warning(monkeypatch):
    from app.api import settings as settings_api

    monkeypatch.setattr(
        settings_api.crud,
        "add_to_blacklist",
        AsyncMock(return_value=(["+15550001"], False)),
    )
    monkeypatch.setattr(settings_api, "_safe_create_audit_log", AsyncMock(return_value=True))

    request = SimpleNamespace(client=SimpleNamespace(host="127.0.0.1"))
    user = SimpleNamespace(id="admin-1")

    result = await settings_api.add_to_blacklist(
        payload={"phone_number": "+15550001"},
        request=request,
        db=object(),
        user=user,
    )

    assert result["success"] is True
    assert any("cache" in w.lower() for w in result.get("warnings", []))


@pytest.mark.asyncio
async def test_admin_delete_call_survives_audit_failure(monkeypatch):
    from app.api import admin as admin_api

    call = SimpleNamespace(
        call_id="call-1",
        phone_number="+15550001",
        recording_url=None,
    )
    monkeypatch.setattr(admin_api.crud, "get_call_by_uuid", AsyncMock(return_value=call))
    monkeypatch.setattr(admin_api.crud, "hard_delete_call", AsyncMock())
    monkeypatch.setattr(
        admin_api,
        "_safe_create_audit_log",
        AsyncMock(return_value=False),
    )

    request = SimpleNamespace(client=SimpleNamespace(host="127.0.0.1"))
    user = SimpleNamespace(id="admin-1")

    result = await admin_api.api_delete_call(
        call_id="00000000-0000-0000-0000-000000000001",
        request=request,
        db=object(),
        user=user,
    )

    assert result["deleted"] is True
    assert any("audit" in w.lower() for w in result.get("warnings", []))


@pytest.mark.asyncio
async def test_api_mark_reviewed_writes_audit_log(monkeypatch):
    from app.api import admin as admin_api

    tenant = SimpleNamespace(id="tenant-1", reviewed_by_admin=False)
    monkeypatch.setattr(admin_api.crud, "get_tenant_by_call", AsyncMock(return_value=tenant))
    update_tenant = AsyncMock()
    monkeypatch.setattr(admin_api.crud, "update_tenant", update_tenant)
    audit = AsyncMock(return_value=True)
    monkeypatch.setattr(admin_api, "_safe_create_audit_log", audit)

    request = SimpleNamespace(client=SimpleNamespace(host="127.0.0.1"))
    user = SimpleNamespace(id="admin-1")

    result = await admin_api.api_mark_reviewed(
        call_id="00000000-0000-0000-0000-000000000002",
        request=request,
        db=object(),
        user=user,
    )

    assert result["reviewed"] is True
    audit.assert_awaited_once()
    assert audit.await_args.kwargs["action"] == "toggled_call_review"


def test_is_encrypted_value_detects_ciphertext():
    from app.utils.security import encrypt_value, is_encrypted_value

    secret = "super-secret-token"
    encrypted = encrypt_value(secret)
    assert is_encrypted_value(encrypted) is True
    assert is_encrypted_value(secret) is False


def test_crm_secret_not_double_encrypted():
    from app.utils.security import encrypt_value, is_encrypted_value

    plaintext = "crm-token-123"
    encrypted_once = encrypt_value(plaintext)
    assert is_encrypted_value(encrypted_once)
    assert encrypt_value(encrypted_once) != encrypted_once


def test_invalid_encryption_key_raises_in_production(monkeypatch):
    from app.utils import security

    monkeypatch.setattr(security.settings, "encryption_key", "not-a-valid-fernet-key")
    monkeypatch.setattr(security.settings, "environment", "production")

    with pytest.raises(RuntimeError, match="ENCRYPTION_KEY is invalid"):
        security._fernet()


@pytest.mark.asyncio
async def test_api_create_user_handles_integrity_race_as_conflict(monkeypatch):
    from app.api import admin as admin_api

    monkeypatch.setattr(admin_api.crud, "get_user_by_email", AsyncMock(return_value=None))
    monkeypatch.setattr(
        admin_api.crud,
        "create_user",
        AsyncMock(side_effect=IntegrityError("dup", params=None, orig=None)),
    )
    payload = admin_api.AdminUserCreateRequest(
        email="race@example.com",
        password="long-enough",
        full_name="Race",
        role="admin",
        scopes=[],
    )
    request = SimpleNamespace(client=SimpleNamespace(host="127.0.0.1"))
    user = SimpleNamespace(id="admin-1")
    db = AsyncMock()

    with pytest.raises(HTTPException) as exc:
        await admin_api.api_create_user(
            payload=payload,
            request=request,
            db=db,
            user=user,
        )

    assert exc.value.status_code == 409
    db.rollback.assert_awaited_once()
