"""Regressions for ratelimit, stream teardown, retention purge, CRM, health checks."""

import asyncio
import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.requests import Request

from app.utils.security import UnsafeURLError


def _request(
    *,
    client_host: str = "203.0.113.50",
    headers: list[tuple[bytes, bytes]] | None = None,
) -> Request:
    hdrs = headers or []
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/auth/login",
            "headers": hdrs,
            "client": (client_host, 12345),
        }
    )


def test_client_ip_ignores_spoofed_forwarded_headers_without_trusted_proxy(monkeypatch):
    from app.core import ratelimit
    from config import settings

    monkeypatch.setattr(settings, "trusted_proxy_ips", "")
    ratelimit._trusted_proxy_peers.cache_clear()

    req = _request(
        client_host="203.0.113.50",
        headers=[(b"x-forwarded-for", b"1.2.3.4")],
    )
    assert ratelimit.client_ip(req) == "203.0.113.50"


def test_client_ip_honors_forwarded_headers_from_trusted_proxy(monkeypatch):
    from app.core import ratelimit
    from config import settings

    monkeypatch.setattr(settings, "trusted_proxy_ips", "127.0.0.1")
    ratelimit._trusted_proxy_peers.cache_clear()

    req = _request(
        client_host="127.0.0.1",
        headers=[(b"x-forwarded-for", b"198.51.100.10, 10.0.0.1")],
    )
    assert ratelimit.client_ip(req) == "198.51.100.10"


@pytest.mark.asyncio
async def test_sender_exits_after_worker_sets_stop_on_empty_sentinel():
    """Documents worker/sender contract: empty queue item must pair with stop_event."""
    stop_event = asyncio.Event()
    outbound_queue: asyncio.Queue[bytes] = asyncio.Queue()

    async def sender() -> None:
        while True:
            item = await outbound_queue.get()
            if not item and stop_event.is_set():
                return

    task = asyncio.create_task(sender())
    await outbound_queue.put(b"")
    await asyncio.sleep(0.02)
    assert not task.done()

    stop_event.set()
    await outbound_queue.put(b"")
    await asyncio.wait_for(task, timeout=1.0)


@pytest.mark.asyncio
async def test_recording_pointer_safe_to_drop_false_when_delete_and_queue_fail(
    monkeypatch,
):
    from app.services.recording_cleanup import (
        RecordingRemovalResult,
        recording_pointer_safe_to_drop,
    )

    monkeypatch.setattr(
        "app.services.recording_cleanup.remove_recording",
        AsyncMock(return_value=RecordingRemovalResult.FAILED),
    )
    monkeypatch.setattr(
        "app.services.recording_cleanup.enqueue_orphaned_recording",
        AsyncMock(return_value=False),
    )

    assert await recording_pointer_safe_to_drop("recordings/stuck.mp3") is False


@pytest.mark.asyncio
async def test_purge_calls_before_skips_row_when_orphan_queue_fails(monkeypatch):
    from app.db import crud

    cid = uuid.uuid4()
    rows = [(cid, "recordings/stuck.mp3")]

    db = AsyncMock()
    db.execute = AsyncMock(
        return_value=MagicMock(all=MagicMock(return_value=rows))
    )

    monkeypatch.setattr(
        "app.services.recording_cleanup.recording_pointer_safe_to_drop",
        AsyncMock(return_value=False),
    )

    total = await crud.purge_calls_before(db, datetime.now(UTC), batch_size=500)

    assert total == 0
    db.commit.assert_not_awaited()


def test_fire_crm_webhook_records_unsafe_url_failure(monkeypatch):
    from app.services import email_service

    record = MagicMock()
    monkeypatch.setattr(email_service, "_record_side_effect_failure_sync", record)
    monkeypatch.setattr(
        "app.utils.security.assert_safe_external_url",
        MagicMock(side_effect=UnsafeURLError("blocked")),
    )

    result = email_service.fire_crm_webhook_task.run(
        "http://127.0.0.1/hook",
        "call-1",
        "+15551234567",
        "qualified",
        80,
        {},
        "http://localhost:8000",
    )

    assert result["sent"] is False
    assert result["error"] == "unsafe_webhook_url"
    record.assert_called_once_with(
        "call-1",
        key="crm_webhook",
        detail="unsafe_webhook_url",
    )


def test_provider_health_check_pings_all_roles(monkeypatch):
    from app.services import email_service

    class _Provider:
        def __init__(self, ok: bool):
            self._ok = ok

        async def ping(self):
            return self._ok, 12.5

    async def _init():
        return None

    registry = SimpleNamespace(
        initialize=_init,
        llm=_Provider(True),
        stt=_Provider(False),
        tts=_Provider(True),
    )
    monkeypatch.setattr("config.provider_registry", registry)

    with patch.object(email_service.logger, "info") as log_info:
        email_service.provider_health_check_task()

    logged = " ".join(str(call.args[0]) for call in log_info.call_args_list)
    assert "llm" in logged
    assert "stt" in logged
    assert "tts" in logged


def test_apply_encrypted_api_keys_resets_missing_key_to_env_default(monkeypatch):
    import config

    original = config._ENV_API_KEY_DEFAULTS.get("openai_api_key", "")
    monkeypatch.setattr(config.settings, "openai_api_key", "stale-rotated-key")

    config._apply_encrypted_api_keys({})

    assert config.settings.openai_api_key == original
