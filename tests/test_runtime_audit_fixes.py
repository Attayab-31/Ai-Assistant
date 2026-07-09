"""Tests for the 24-issue runtime audit fix batch."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def test_numeric_range_above_max_scores_partial_points():
    from app.core.question_scoring import evaluate_question_scoring

    question = {
        "state": "Q_INCOME",
        "extract_fields": ["monthly_income"],
        "scoring": {
            "enabled": True,
            "rule_type": "numeric_range",
            "max_points": 20,
            "pass_config": {"min": 1000, "max": 5000},
        },
    }
    pts, reasons, _ = evaluate_question_scoring(
        question, {"monthly_income": 9000}
    )
    assert pts == 5
    assert any("above maximum" in r for r in reasons)


def test_is_number_blacklisted_normalizes_stored_entries(monkeypatch):
    from app.db import crud

    async def fake_get_setting_value(db, key, default=None):
        return ["(555) 987-6543"]

    monkeypatch.setattr(crud, "get_setting_value", fake_get_setting_value)

    async def _run():
        return await crud.is_number_blacklisted(object(), "+15559876543")

    import asyncio

    assert asyncio.run(_run()) is True


def test_verify_stream_token_allows_longer_ttl():
    from app.utils.helpers import generate_stream_token, verify_stream_token

    call_id = "v3:test-call"
    secret = "x" * 32
    token = generate_stream_token(call_id, secret)
    assert verify_stream_token(call_id, token, secret, max_age=900) is True


@pytest.mark.asyncio
async def test_finalize_active_session_keeps_session_on_concurrent_skip(monkeypatch):
    from app.core import call_handler
    from app.core.conversation import ConversationSession

    session = ConversationSession(call_id="v3:skip", phone_number="+15551234567")
    call_handler._active_sessions["v3:skip"] = session
    monkeypatch.setattr(
        call_handler,
        "finalize_call",
        AsyncMock(return_value={"call_id": "v3:skip", "status": "skipped"}),
    )
    remove = MagicMock()
    monkeypatch.setattr(call_handler, "remove_session", remove)
    monkeypatch.setattr(call_handler, "unregister_stream_stop", MagicMock())

    await call_handler.finalize_active_session_background("v3:skip")

    remove.assert_not_called()
    assert call_handler.get_session("v3:skip") is session
    call_handler._active_sessions.pop("v3:skip", None)


@pytest.mark.asyncio
async def test_finalize_after_stream_timeout_skips_force_without_hangup_signal(
    monkeypatch,
):
    from app.core import call_handler
    from app.core.conversation import ConversationSession

    session = ConversationSession(call_id="v3:active", phone_number="+15551234567")
    monkeypatch.setattr(call_handler, "get_session", lambda _cid: session)
    monkeypatch.setattr(call_handler.asyncio, "sleep", AsyncMock())
    monkeypatch.setattr(
        call_handler,
        "check_stream_stop_signal",
        AsyncMock(return_value=False),
    )
    force = AsyncMock()
    monkeypatch.setattr(call_handler, "finalize_active_session_background", force)

    await call_handler.finalize_after_stream_timeout("v3:active", timeout=0.01)

    force.assert_not_awaited()


@pytest.mark.asyncio
async def test_finalize_after_stream_timeout_abandons_when_stop_signaled(monkeypatch):
    from app.core import call_handler

    monkeypatch.setattr(call_handler, "get_session", lambda _cid: None)
    monkeypatch.setattr(call_handler.asyncio, "sleep", AsyncMock())
    monkeypatch.setattr(
        "app.core.redis_client.is_finalize_inflight",
        AsyncMock(return_value=False),
    )
    monkeypatch.setattr(
        call_handler,
        "check_stream_stop_signal",
        AsyncMock(return_value=True),
    )
    mark = AsyncMock(return_value=True)
    monkeypatch.setattr("app.db.crud.mark_call_abandoned_if_active", mark)

    call = MagicMock(status="in_progress")
    monkeypatch.setattr(
        "app.db.crud.get_call_by_call_id",
        AsyncMock(return_value=call),
    )

    await call_handler.finalize_after_stream_timeout("v3:stop", timeout=0.01)

    mark.assert_awaited_once()


def test_sync_post_safe_external_pins_dns(monkeypatch):
    import httpx

    from app.utils import security

    calls: list[tuple] = []
    original = security.socket.getaddrinfo

    def fake_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        calls.append((host, port))
        if host == "crm.example.com":
            return [(2, 1, 6, "", ("93.184.216.34", port))]
        return original(host, port, family, type, proto, flags)

    monkeypatch.setattr(security.socket, "getaddrinfo", fake_getaddrinfo)

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, content, headers):
            return FakeResponse()

    monkeypatch.setattr(httpx, "Client", FakeClient)

    security.sync_post_safe_external(
        "https://crm.example.com/hook",
        content=b"{}",
        headers={"Content-Type": "application/json"},
        require_https=True,
    )
    assert calls.count(("crm.example.com", 443)) >= 1


def test_send_screening_email_skips_when_already_sent(monkeypatch):
    from app.services import email_service

    monkeypatch.setattr(email_service.settings, "resend_api_key", "re_test")
    monkeypatch.setattr(email_service, "_is_email_already_sent_sync", lambda _cid: True)
    send = MagicMock()
    monkeypatch.setattr(email_service.resend.Emails, "send", send)

    result = email_service.send_screening_email_task.run(
        call_id="11111111-1111-1111-1111-111111111111",
        phone_number="+15551234567",
        tenant_data={},
        score=80,
        status="qualified",
        reasons=[],
        transcript="",
        duration=60,
        providers={},
    )
    assert result["reason"] == "already_sent"
    send.assert_not_called()


def test_fire_crm_webhook_skips_when_already_delivered(monkeypatch):
    from app.services import email_service

    monkeypatch.setattr(email_service, "_is_crm_already_delivered_sync", lambda _cid: True)
    post = MagicMock()
    monkeypatch.setattr("app.utils.security.sync_post_safe_external", post)

    result = email_service.fire_crm_webhook_task.run(
        webhook_url="https://crm.example.com/hook",
        call_id="11111111-1111-1111-1111-111111111111",
        phone_number="+15551234567",
        status="qualified",
        score=80,
        tenant_data={},
        app_url="https://example.com",
    )
    assert result["reason"] == "already_delivered"
    post.assert_not_called()


def test_digest_respects_notifications_disabled(monkeypatch):
    from app.services import email_service

    monkeypatch.setattr(email_service.settings, "resend_api_key", "re_test")
    monkeypatch.setattr(
        email_service,
        "resolve_email_settings",
        lambda: {
            "email_notifications_enabled": False,
            "timezone": "UTC",
            "landlord_email": "a@example.com",
        },
    )

    result = email_service.send_daily_digest_task.run()
    assert result["reason"] == "notifications_disabled"


def test_digest_skips_outside_local_nine_am(monkeypatch):
    from app.services import email_service

    monkeypatch.setattr(email_service.settings, "resend_api_key", "re_test")
    monkeypatch.setattr(
        email_service,
        "resolve_email_settings",
        lambda: {
            "email_notifications_enabled": True,
            "timezone": "UTC",
            "landlord_email": "a@example.com",
        },
    )
    monkeypatch.setattr(email_service, "_is_digest_send_hour", lambda _tz: False)

    result = email_service.send_daily_digest_task.run()
    assert result["reason"] == "outside_digest_window"


@pytest.mark.asyncio
async def test_post_login_tasks_uses_crud_helpers(monkeypatch):
    from app.api import auth

    update_login = AsyncMock()
    create_audit = AsyncMock()
    monkeypatch.setattr(auth.crud, "update_last_login", update_login)
    monkeypatch.setattr(auth.crud, "create_audit_log", create_audit)
    monkeypatch.setattr(auth, "password_needs_rehash", lambda _h: False)

    user_id = auth.uuid.uuid4()
    await auth._post_login_tasks(
        user_id,
        ip_address="127.0.0.1",
        user_agent="test",
        plain_password="secret",
        stored_hash="hash",
    )

    update_login.assert_awaited_once()
    create_audit.assert_awaited_once()
