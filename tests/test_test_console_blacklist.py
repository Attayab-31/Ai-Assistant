"""Test console must enforce the same DNC blacklist as live Telnyx calls."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException


@pytest.mark.asyncio
async def test_start_test_call_rejects_blacklisted_number(monkeypatch):
    from app.api import test_console

    monkeypatch.setattr(
        test_console,
        "is_number_blacklisted",
        AsyncMock(return_value=True),
    )
    create_session = AsyncMock()
    monkeypatch.setattr(test_console.call_handler, "create_session", create_session)

    payload = test_console.StartCallRequest(phone_number="+15551234567")
    request = type("Req", (), {"url": type("U", (), {"scheme": "http", "netloc": "localhost"})()})()

    with pytest.raises(HTTPException) as exc:
        await test_console.start_test_call(request, payload, db=object())

    assert exc.value.status_code == 403
    assert "do-not-call" in str(exc.value.detail).lower()
    create_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_start_test_call_allows_non_blacklisted_number(monkeypatch):
    from app.api import test_console

    monkeypatch.setattr(
        test_console,
        "is_number_blacklisted",
        AsyncMock(return_value=False),
    )

    class FakeSession:
        transcript = []

    session = FakeSession()
    session.call_id = "test-abc"
    session.phone_number = "+15551234567"
    create_session = AsyncMock(return_value=session)
    monkeypatch.setattr(test_console.call_handler, "create_session", create_session)
    monkeypatch.setattr(
        test_console.call_handler,
        "handle_call_answered",
        AsyncMock(return_value=b""),
    )

    monkeypatch.setattr(
        test_console,
        "_session_snapshot",
        lambda s: {"call_id": s.call_id, "phone_number": s.phone_number},
    )

    payload = test_console.StartCallRequest(phone_number="555-123-4567")
    request = type("Req", (), {"url": type("U", (), {"scheme": "http", "netloc": "localhost"})()})()

    result = await test_console.start_test_call(request, payload, db=object())

    assert result["call_id"].startswith("test-")
    create_session.assert_awaited_once()
    assert create_session.await_args.kwargs["phone_number"] == "+15551234567"


@pytest.mark.asyncio
async def test_is_number_blacklisted_matches_sanitized_entries(monkeypatch):
    from app.db import crud

    async def fake_get_setting_value(db, key, default=None):
        assert key == "blacklisted_numbers"
        return ["+15559876543"]

    monkeypatch.setattr(crud, "get_setting_value", fake_get_setting_value)

    assert await crud.is_number_blacklisted(object(), "(555) 987-6543") is True
    assert await crud.is_number_blacklisted(object(), "+15551234567") is False


@pytest.mark.asyncio
async def test_verify_ws_auth_requires_token_for_non_loopback(monkeypatch):
    from app.api import test_console

    ws = type(
        "WS",
        (),
        {
            "client": type("C", (), {"host": "203.0.113.10"})(),
            "cookies": {},
        },
    )()
    assert await test_console._verify_ws_auth(ws) is False


@pytest.mark.asyncio
async def test_verify_ws_auth_accepts_loopback_in_non_production():
    from app.api import test_console

    test_console._trusted_proxy_peers.cache_clear()
    ws = type(
        "WS",
        (),
        {
            "client": type("C", (), {"host": "127.0.0.1"})(),
            "cookies": {},
            "headers": {},
        },
    )()
    assert await test_console._verify_ws_auth(ws) is True


@pytest.mark.asyncio
async def test_verify_ws_auth_rejects_revoked_token(monkeypatch):
    from app.api import test_console

    ws = type(
        "WS",
        (),
        {
            "client": type("C", (), {"host": "203.0.113.10"})(),
            "cookies": {"access_token": "revoked-token"},
            "headers": {},
        },
    )()
    monkeypatch.setattr(
        "app.utils.security.decode_access_token",
        lambda token: {"sub": "11111111-1111-1111-1111-111111111111"},
    )
    monkeypatch.setattr(
        "app.core.redis_client.is_token_revoked",
        AsyncMock(return_value=True),
    )

    assert await test_console._verify_ws_auth(ws) is False


@pytest.mark.asyncio
async def test_verify_ws_auth_does_not_exempt_loopback_proxy_remote_client(monkeypatch):
    from app.api import test_console

    monkeypatch.setattr(test_console.settings, "trusted_proxy_ips", "127.0.0.1")
    test_console._trusted_proxy_peers.cache_clear()
    ws = type(
        "WS",
        (),
        {
            "client": type("C", (), {"host": "127.0.0.1"})(),
            "cookies": {},
            "headers": {"x-forwarded-for": "203.0.113.25"},
        },
    )()
    # No token: should be rejected (not loopback-exempt) for proxied remote source.
    assert await test_console._verify_ws_auth(ws) is False


@pytest.mark.asyncio
async def test_require_test_console_access_checks_forwarded_remote_client(monkeypatch):
    from app.api import test_console

    monkeypatch.setattr(test_console.settings, "trusted_proxy_ips", "127.0.0.1")
    test_console._trusted_proxy_peers.cache_clear()
    request = SimpleNamespace(
        client=SimpleNamespace(host="127.0.0.1"),
        headers={"x-forwarded-for": "203.0.113.31"},
    )
    user = SimpleNamespace(can=lambda scope: False, can_edit=False)
    monkeypatch.setattr(test_console, "get_current_user", AsyncMock(return_value=user))

    with pytest.raises(HTTPException, match="Settings access"):
        await test_console.require_test_console_access(request=request, db=object())


@pytest.mark.asyncio
async def test_say_rejects_non_test_call_id(monkeypatch):
    from app.api import test_console

    monkeypatch.setattr(test_console.call_handler, "get_session", AsyncMock())
    payload = test_console.SayRequest(call_id="prod-call-1", text="hello")

    with pytest.raises(HTTPException, match="Invalid test call_id"):
        await test_console.say(payload, _auth=None)
