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
