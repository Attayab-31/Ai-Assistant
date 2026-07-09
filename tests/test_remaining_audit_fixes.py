"""Tests for remaining deep-audit fixes (#10–#18, admin ops)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from starlette.requests import Request


@pytest.mark.asyncio
async def test_purge_calls_before_skips_soft_deleted_rows():
    from app.db import crud

    db = AsyncMock()
    result = MagicMock()
    result.all.return_value = []
    db.execute = AsyncMock(return_value=result)

    await crud.purge_calls_before(db, datetime.now(UTC), batch_size=500)

    stmt = db.execute.await_args.args[0]
    compiled = str(stmt)
    assert "is_deleted" in compiled.lower()


@pytest.mark.asyncio
async def test_remove_from_blacklist_clears_tenant_flag():
    from app.db import crud

    row = MagicMock()
    row.value = '["+15551234567"]'
    db = AsyncMock()
    tenant_scan = MagicMock()
    tenant_scan.all.return_value = []
    db.execute = AsyncMock(
        side_effect=[
            MagicMock(scalar_one_or_none=lambda: row),
            MagicMock(),
            tenant_scan,
        ]
    )

    with patch(
        "app.services.settings_cache.invalidate_settings_cache",
        AsyncMock(),
    ):
        await crud.remove_from_blacklist(db, "+15551234567")

    assert db.execute.await_count >= 2


def test_numeric_range_max_only_awards_full_points():
    from app.core.question_scoring import evaluate_question_scoring

    question = {
        "state": "Q_INCOME",
        "extract_fields": ["monthly_income"],
        "scoring": {
            "enabled": True,
            "rule_type": "numeric_range",
            "max_points": 20,
            "pass_config": {"max": 5000},
        },
    }
    pts, reasons, _ = evaluate_question_scoring(
        question, {"monthly_income": 3000}
    )
    assert pts == 20
    assert reasons == []


def test_environment_normalization_treats_prod_as_production():
    from config import Settings

    settings = Settings(environment="Production")
    assert settings.is_production is True


def test_validate_runtime_secrets_requires_telnyx_and_no_debug(monkeypatch):
    from config import Settings

    settings = Settings(
        environment="production",
        secret_key="x" * 32,
        encryption_key="",
        app_url="https://example.com",
        telnyx_public_key="pub",
        telnyx_api_key="",
        debug=True,
        admin_password="strong-password-here",
        web_workers=1,
    )
    with patch.object(settings, "encryption_key", "dGVzdC1rZXktdGVzdC1rZXktdGVzdC1rZXk="):
        with patch("cryptography.fernet.Fernet"):
            errors = settings.validate_runtime_secrets()
    assert any("TELNYX_API_KEY" in e for e in errors)
    assert any("DEBUG" in e for e in errors)


def _auth_request(token: str = "") -> Request:
    headers = []
    if token:
        headers.append((b"cookie", f"access_token={token}".encode()))
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/admin",
            "headers": headers,
            "client": ("127.0.0.1", 12345),
            "scheme": "http",
        }
    )


@pytest.mark.asyncio
async def test_password_change_timestamp_invalidates_old_token():
    from app.models.user import AdminUser
    from app.utils.dependencies import get_current_user
    from app.utils.security import create_access_token

    token = create_access_token(
        {"sub": str(uuid.uuid4()), "pwd_at": "2026-01-01T00:00:00+00:00"},
        expires_delta=timedelta(hours=1),
    )
    user = AdminUser(
        id=uuid.uuid4(),
        email="admin@example.com",
        hashed_password="x",
        full_name="Admin",
        role="admin",
        is_active=True,
        password_changed_at=datetime(2026, 1, 2, tzinfo=UTC),
    )

    with patch("app.core.redis_client.is_token_revoked", AsyncMock(return_value=False)):
        with patch("app.utils.dependencies.get_user_by_id", AsyncMock(return_value=user)):
            with pytest.raises(HTTPException) as exc:
                await get_current_user(_auth_request(token), AsyncMock())

    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_logout_clears_cookie_when_revoke_fails():
    from unittest.mock import MagicMock

    from app.api.auth import logout
    from app.utils.security import create_access_token

    token = create_access_token({"sub": str(uuid.uuid4())}, expires_delta=timedelta(hours=1))
    request = _auth_request(token)
    response = MagicMock()
    with patch("app.core.redis_client.revoke_token", AsyncMock(return_value=False)):
        result = await logout(request, response)

    response.delete_cookie.assert_called_once()
    assert result.status_code == 200
    assert b"revoke_failed" in result.body


def test_encrypt_api_key_for_storage_skips_ciphertext():
    from app.api.settings import _encrypt_api_key_for_storage

    ciphertext = "gAAAAABmockciphertext"
    with patch(
        "app.utils.security.is_encrypted_value", return_value=True
    ):
        assert _encrypt_api_key_for_storage(ciphertext) == ciphertext


def test_general_settings_timezone_validator_rejects_invalid():
    from app.schemas.settings import GeneralSettingsUpdate

    with pytest.raises(ValueError, match="Unknown timezone"):
        GeneralSettingsUpdate(timezone="Not/A_Real_Zone")


@pytest.mark.asyncio
async def test_provider_overview_uses_db_configured_keys():
    from app.services.provider_usage import get_provider_overview

    db = object()
    keys = MagicMock()
    keys.configured.side_effect = lambda name: name == "groq"
    keys.openrouter = ""
    keys.deepgram = "dg-key"

    with patch(
        "app.core.call_settings.capture_provider_api_keys",
        AsyncMock(return_value=keys),
    ):
        with patch(
            "app.services.provider_usage._openrouter_balance",
            AsyncMock(return_value={"available": False}),
        ):
            with patch(
                "app.services.provider_usage._deepgram_balance",
                AsyncMock(return_value={"available": False}),
            ):
                with patch(
                    "app.services.provider_usage.get_internal_usage",
                    AsyncMock(return_value={}),
                ):
                    with patch(
                        "config.provider_registry.get_status",
                        return_value={},
                    ):
                        overview = await get_provider_overview(db)

    by_key = {p["key"]: p for p in overview["providers"]}
    assert by_key["groq"]["configured"] is True
    assert by_key["openai"]["configured"] is False


@pytest.mark.asyncio
async def test_mark_tenant_reviewed_writes_audit(monkeypatch):
    from types import SimpleNamespace

    from app.api import admin as admin_api

    tenant = SimpleNamespace(
        id=uuid.uuid4(),
        reviewed_by_admin=False,
    )
    monkeypatch.setattr(admin_api.crud, "get_tenant_by_id", AsyncMock(return_value=tenant))
    monkeypatch.setattr(admin_api.crud, "update_tenant", AsyncMock())
    audit = AsyncMock(return_value=True)
    monkeypatch.setattr(admin_api, "_safe_create_audit_log", audit)

    request = SimpleNamespace(client=SimpleNamespace(host="127.0.0.1"))
    result = await admin_api.api_mark_tenant_reviewed(
        tenant.id,
        request,
        {"reviewed": True},
        db=object(),
        user=SimpleNamespace(id=uuid.uuid4(), email="a@test.com"),
    )

    assert result["reviewed"] is True
    audit.assert_awaited_once()
    assert audit.await_args.kwargs["action"] == "tenant_review_updated"


def test_retention_calls_days_floor_enforced_in_api_validation():
    from app.api.settings import _validate_general_settings_updates

    with pytest.raises(HTTPException, match="at least 30"):
        _validate_general_settings_updates({"retention_calls_days": 7}, {})


def test_latency_critical_threshold_must_be_ge_warning():
    from app.api.settings import _validate_general_settings_updates

    with pytest.raises(HTTPException, match="critical p95 threshold must be >="):
        _validate_general_settings_updates(
            {
                "latency_alert_turn_p95_ms": 1500,
                "latency_alert_turn_p95_crit_ms": 1400,
            },
            {},
        )


@pytest.mark.asyncio
async def test_bulk_review_rejects_invalid_uuid():
    from types import SimpleNamespace

    from app.api import admin as admin_api

    with pytest.raises(HTTPException, match="invalid tenant id"):
        await admin_api.api_bulk_mark_reviewed(
            {"tenant_ids": ["not-a-uuid"], "reviewed": True},
            request=SimpleNamespace(client=SimpleNamespace(host="127.0.0.1")),
            db=object(),
            user=SimpleNamespace(id=uuid.uuid4()),
        )


@pytest.mark.asyncio
async def test_call_notes_reject_overlong():
    from types import SimpleNamespace

    from app.api import admin as admin_api

    tenant = SimpleNamespace(id=uuid.uuid4())
    with patch.object(admin_api.crud, "get_tenant_by_call", AsyncMock(return_value=tenant)):
        with pytest.raises(HTTPException, match="cannot exceed"):
            await admin_api.api_update_call_notes(
                uuid.uuid4(),
                {"notes": "x" * (admin_api.MAX_ADMIN_NOTES_LENGTH + 1)},
                request=SimpleNamespace(client=SimpleNamespace(host="127.0.0.1")),
                db=object(),
                user=SimpleNamespace(id=uuid.uuid4()),
            )


@pytest.mark.asyncio
async def test_call_notes_accepts_non_string_payload_without_crash():
    from types import SimpleNamespace

    from app.api import admin as admin_api

    tenant = SimpleNamespace(id=uuid.uuid4())
    update_tenant = AsyncMock()
    audit = AsyncMock(return_value=True)
    with patch.object(admin_api.crud, "get_tenant_by_call", AsyncMock(return_value=tenant)):
        with patch.object(admin_api.crud, "update_tenant", update_tenant):
            with patch.object(admin_api, "_safe_create_audit_log", audit):
                result = await admin_api.api_update_call_notes(
                    uuid.uuid4(),
                    {"notes": {"hello": "world"}},
                    request=SimpleNamespace(client=SimpleNamespace(host="127.0.0.1")),
                    db=object(),
                    user=SimpleNamespace(id=uuid.uuid4()),
                )

    assert result["saved"] is True
    assert update_tenant.await_args.kwargs["notes"] == "{'hello': 'world'}"
    assert audit.await_args.kwargs["new_value"]["notes_length"] == len("{'hello': 'world'}")


@pytest.mark.asyncio
async def test_tenant_notes_accepts_non_string_payload_without_crash():
    from types import SimpleNamespace

    from app.api import admin as admin_api

    tenant_id = uuid.uuid4()
    tenant = SimpleNamespace(id=tenant_id)
    update_tenant = AsyncMock()
    audit = AsyncMock(return_value=True)
    with patch.object(admin_api.crud, "get_tenant_by_id", AsyncMock(return_value=tenant)):
        with patch.object(admin_api.crud, "update_tenant", update_tenant):
            with patch.object(admin_api, "_safe_create_audit_log", audit):
                result = await admin_api.api_update_tenant_notes(
                    tenant_id,
                    {"notes": [1, 2, 3]},
                    request=SimpleNamespace(client=SimpleNamespace(host="127.0.0.1")),
                    db=object(),
                    user=SimpleNamespace(id=uuid.uuid4()),
                )

    assert result["saved"] is True
    assert update_tenant.await_args.kwargs["notes"] == "[1, 2, 3]"
    assert audit.await_args.kwargs["new_value"]["notes_length"] == len("[1, 2, 3]")


@pytest.mark.asyncio
async def test_send_test_email_error_is_sanitized():
    from app.api import settings as settings_api

    user = MagicMock()
    db = object()
    with patch.object(settings_api.crud, "get_setting_value", AsyncMock(return_value="admin@example.com")):
        with patch("app.services.email_service.build_test_email_preview", return_value=("Subject", "<p>x</p>")):
            with patch("app.services.email_service._split_emails", return_value=[]):
                with patch("resend.Emails.send", side_effect=RuntimeError("smtp secret details")):
                    with patch("config.settings.resend_api_key", "test-key"):
                        with pytest.raises(HTTPException) as exc:
                            await settings_api.send_test_email(
                                payload={"email": "landlord@example.com"},
                                request=MagicMock(headers={}, client=MagicMock(host="127.0.0.1")),
                                db=db,
                                user=user,
                            )

    assert exc.value.status_code == 500
    assert exc.value.detail == "Failed to send test email. Verify email settings and try again."
    assert "secret" not in exc.value.detail.lower()


@pytest.mark.asyncio
async def test_provider_health_errors_are_sanitized():
    from app.api import settings as settings_api

    failing = AsyncMock()
    failing.ping = AsyncMock(side_effect=RuntimeError("vendor secret details"))
    bundle = MagicMock()
    bundle.llm = failing
    bundle.llm_name = "groq"
    bundle.llm_by_name = {"groq": failing}
    bundle.stt = failing
    bundle.stt_name = "deepgram"
    bundle.tts = failing
    bundle.tts_name = "deepgram"
    bundle.tts_by_name = {"deepgram": failing}

    snapshot = MagicMock(stt_model="nova-3", groq_stt_model="whisper-large-v3-turbo")
    keys = MagicMock()
    keys.configured.return_value = False

    with patch("app.core.call_settings.load_call_settings_snapshot", AsyncMock(return_value=snapshot)):
        with patch("app.core.call_settings.build_call_provider_bundle", return_value=bundle):
            with patch("app.core.call_settings.capture_provider_api_keys", AsyncMock(return_value=keys)):
                result = await settings_api.check_provider_health(db=object(), user=MagicMock())

    assert result["llm"]["error"] == "Provider health check failed"


@pytest.mark.asyncio
async def test_is_token_revoked_fails_closed_in_production():
    from app.core import redis_client

    with patch("app.core.redis_client.get_redis", return_value=None):
        with patch.object(redis_client.settings, "environment", "production"):
            assert await redis_client.is_token_revoked("token") is True


@pytest.mark.asyncio
async def test_retention_runs_when_lock_unavailable():
    from app.services import retention_service

    async def _capture(*args, **kwargs):
        assert kwargs.get("fail_closed") is False
        return False

    with patch("app.core.redis_client.acquire_once", AsyncMock(side_effect=_capture)):
        result = await retention_service._run_retention()
    assert result.get("skipped") is True


@pytest.mark.asyncio
async def test_safe_create_audit_log_accepts_positional_db_admin():
    from app.api import admin as admin_api

    with patch.object(admin_api.crud, "create_audit_log", AsyncMock()) as create:
        ok = await admin_api._safe_create_audit_log(
            object(),
            action="x",
            admin_user_id=uuid.uuid4(),
            entity_type="setting",
        )
    assert ok is True
    assert create.await_args.kwargs.get("db") is not None


@pytest.mark.asyncio
async def test_safe_create_audit_log_accepts_positional_db_settings():
    from app.api import settings as settings_api

    with patch.object(settings_api.crud, "create_audit_log", AsyncMock()) as create:
        ok = await settings_api._safe_create_audit_log(
            object(),
            action="x",
            admin_user_id=uuid.uuid4(),
            entity_type="setting",
        )
    assert ok is True
    assert create.await_args.kwargs.get("db") is not None


@pytest.mark.asyncio
async def test_provider_switch_lock_uses_fail_closed_acquire():
    from app.api import settings as settings_api

    async def _capture(*args, **kwargs):
        assert kwargs.get("fail_closed") is True
        assert kwargs.get("token")
        return True

    fake_redis = AsyncMock()
    fake_redis.set = AsyncMock()
    fake_redis.eval = AsyncMock(return_value=1)
    with patch("app.core.redis_client.acquire_once", AsyncMock(side_effect=_capture)):
        with patch("app.core.redis_client.get_redis", return_value=fake_redis):
            async with settings_api._provider_switch_lock():
                pass


@pytest.mark.asyncio
async def test_provider_switch_lock_release_does_not_delete_other_owner():
    from app.api import settings as settings_api

    fake_redis = AsyncMock()
    fake_redis.eval = AsyncMock(return_value=0)
    fake_redis.get = AsyncMock(return_value="another-owner-token")
    with patch("app.core.redis_client.get_redis", return_value=fake_redis):
        with patch("app.core.redis_client.cache_delete", AsyncMock()) as cache_delete:
            await settings_api._release_provider_switch_lock("my-token")
    cache_delete.assert_not_awaited()
