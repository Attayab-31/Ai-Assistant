"""Production-readiness defaults and infrastructure helpers."""

from unittest.mock import AsyncMock, patch

import pytest

from config import DEFAULT_SYSTEM_SETTINGS


EMAIL_TEMPLATE_KEYS = {
    "email_from_name",
    "email_from_address",
    "email_subject_template",
    "email_body_template",
    "email_qualified_only",
    "email_include_transcript",
    "cc_emails",
    "bcc_emails",
}


def test_default_system_settings_include_email_templates():
    keys = {item["key"] for item in DEFAULT_SYSTEM_SETTINGS}
    missing = EMAIL_TEMPLATE_KEYS - keys
    assert not missing, f"Missing seeded email keys: {sorted(missing)}"


def test_default_system_settings_include_groq_stt_model():
    keys = {item["key"] for item in DEFAULT_SYSTEM_SETTINGS}
    assert "groq_stt_model" in keys


def test_validate_runtime_secrets_allows_test_console_in_production_when_enabled():
    from config import Settings

    settings = Settings(
        environment="production",
        secret_key="x" * 32,
        encryption_key="",
        app_url="https://example.com",
        telnyx_public_key="pub",
        telnyx_api_key="telnyx",
        debug=False,
        admin_password="strong-password-here",
        web_workers=1,
        enable_test_console=True,
        trusted_proxy_ips="127.0.0.1",
    )
    with patch.object(settings, "encryption_key", "dGVzdC1rZXktdGVzdC1rZXktdGVzdC1rZXk="):
        with patch("cryptography.fernet.Fernet"):
            errors = settings.validate_runtime_secrets()
    assert errors == []


def test_side_effect_alerts_include_permanent_failures():
    from app.utils.helpers import side_effect_alerts_from_error_log

    alerts = side_effect_alerts_from_error_log(
        {
            "side_effect_failures": {
                "email_delivery": "smtp timeout",
                "crm_webhook": "500",
            }
        }
    )
    by_kind = {a["kind"]: a for a in alerts}
    assert "permanently failed" in by_kind["email"]["title"].lower()
    assert "permanently failed" in by_kind["crm"]["title"].lower()


@pytest.mark.asyncio
async def test_invalidate_settings_cache_retries_then_succeeds():
    from app.services import settings_cache

    with patch.object(
        settings_cache, "_delete_snapshot_key", new_callable=AsyncMock
    ) as mock_delete:
        mock_delete.side_effect = [False, True]
        await settings_cache.invalidate_settings_cache()
        assert mock_delete.call_count == 2


@pytest.mark.asyncio
async def test_invalidate_settings_cache_raises_after_exhausted_retries():
    from app.services import settings_cache

    with patch.object(
        settings_cache, "_delete_snapshot_key", new_callable=AsyncMock
    ) as mock_delete:
        mock_delete.return_value = False
        with pytest.raises(RuntimeError, match="invalidation failed"):
            await settings_cache.invalidate_settings_cache()
        assert mock_delete.call_count == settings_cache._MAX_ATTEMPTS


@pytest.mark.asyncio
async def test_acquire_once_fail_open_when_redis_down():
    from app.core.redis_client import acquire_once

    with patch("app.core.redis_client.get_redis", return_value=None):
        assert await acquire_once("test:key", 60) is True
        assert await acquire_once("test:key", 60, fail_closed=True) is False


@pytest.mark.asyncio
async def test_acquire_once_fail_closed_on_redis_error():
    from app.core.redis_client import acquire_once

    mock_redis = AsyncMock()
    mock_redis.set.side_effect = ConnectionError("redis unavailable")
    with patch("app.core.redis_client.get_redis", return_value=mock_redis):
        assert await acquire_once("test:key", 60, fail_closed=False) is True
        assert await acquire_once("test:key", 60, fail_closed=True) is False
