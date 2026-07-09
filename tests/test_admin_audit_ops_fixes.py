"""Admin ops fixes: audit summaries, general reset, provider keys, blacklist cache, custom fields."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException


def test_summarize_questions_audit_change_detects_edits():
    from app.services.admin_audit_helpers import summarize_questions_audit_change

    old = [{"state": "income", "text": "Income?", "order": 1}]
    new = [
        {"state": "pets", "text": "Pets?", "order": 1},
        {"state": "income", "text": "Monthly income?", "order": 2},
    ]
    summary = summarize_questions_audit_change(old, new)
    assert summary["count_before"] == 1
    assert summary["count_after"] == 2
    assert summary["added_states"] == ["pets"]
    assert summary["changed_states"] == ["income"]
    assert summary["reordered"] is True


def test_format_audit_change_summary_questions():
    from app.services.admin_audit_helpers import format_audit_change_summary

    text = format_audit_change_summary(
        {"count": 3},
        {
            "count_before": 3,
            "count_after": 4,
            "added_states": ["pets"],
            "changed_states": ["income"],
            "reordered": False,
        },
    )
    assert "3 → 4 questions" in text
    assert "+1 added" in text
    assert "1 edited" in text


def test_validate_custom_tenant_updates_rejects_long_text():
    from app.services.admin_audit_helpers import validate_custom_tenant_updates

    with pytest.raises(ValueError, match="exceeds"):
        validate_custom_tenant_updates({"custom_notes": "x" * 2001})


def test_validate_custom_tenant_updates_rejects_bad_type():
    from app.services.admin_audit_helpers import validate_custom_tenant_updates

    with pytest.raises(ValueError, match="must be text"):
        validate_custom_tenant_updates({"custom_flag": {"nested": True}})


def test_general_reset_keys_include_voice_and_fallback():
    from app.api.settings import GENERAL_RESET_KEYS

    for key in (
        "voice_latency_profile",
        "llm_streaming_enabled",
        "auto_fallback_enabled",
        "llm_fallback_provider",
        "stt_fallback_provider",
        "tts_fallback_provider",
    ):
        assert key in GENERAL_RESET_KEYS


@pytest.mark.asyncio
async def test_provider_key_configured_map_reflects_db_keys(monkeypatch):
    from app.core.call_settings import ProviderApiKeys, provider_key_configured_map

    keys = ProviderApiKeys(
        groq="",
        openai="",
        openrouter="",
        gemini="",
        deepgram="db-deepgram-key",
        google_application_credentials="creds.json",
    )
    monkeypatch.setattr(
        "app.core.call_settings.capture_provider_api_keys",
        AsyncMock(return_value=keys),
    )

    result = await provider_key_configured_map(object())
    assert result["deepgram"] is True
    assert result["groq"] is False


@pytest.mark.asyncio
async def test_check_provider_health_includes_deepgram_backup_when_key_set(monkeypatch):
    from app.api import settings as settings_api
    from app.core.call_settings import ProviderApiKeys

    snapshot = SimpleNamespace(
        stt_model="nova-2",
        groq_stt_model="whisper-large-v3",
    )
    bundle = SimpleNamespace(
        llm=SimpleNamespace(),
        llm_name="openai",
        llm_by_name={},
        stt=SimpleNamespace(),
        stt_name="groq",
        tts=SimpleNamespace(),
        tts_name="google",
        tts_by_name={},
    )

    class _StubProvider:
        async def ping(self):
            return True, 5

    bundle.llm = _StubProvider()
    bundle.stt = _StubProvider()
    bundle.tts = _StubProvider()

    keys = ProviderApiKeys(
        groq="groq-key",
        openai="",
        openrouter="",
        gemini="",
        deepgram="deepgram-key",
        google_application_credentials="",
    )

    monkeypatch.setattr(
        "app.core.call_settings.load_call_settings_snapshot",
        AsyncMock(return_value=snapshot),
    )
    monkeypatch.setattr(
        "app.core.call_settings.build_call_provider_bundle",
        lambda _s, **kwargs: bundle,
    )
    monkeypatch.setattr(
        "app.core.call_settings.capture_provider_api_keys",
        AsyncMock(return_value=keys),
    )

    class _StubSTT:
        def __init__(self, *args, **kwargs):
            pass

        async def ping(self):
            return True, 3

    monkeypatch.setattr(
        "app.providers.stt.deepgram_stt.DeepgramSTTProvider",
        _StubSTT,
    )
    monkeypatch.setattr(
        "app.providers.stt.groq_stt.GroqSTTProvider",
        _StubSTT,
    )

    result = await settings_api.check_provider_health(
        db=object(),
        user=SimpleNamespace(id="admin-1"),
    )

    backup_names = [row["provider"] for row in result["stt_backups"]]
    assert "deepgram" in backup_names


@pytest.mark.asyncio
async def test_api_blacklist_tenant_warns_when_cache_stale(monkeypatch):
    from app.api import admin as admin_api

    tenant = SimpleNamespace(
        id="tenant-1",
        phone_number="+15551234567",
    )
    monkeypatch.setattr(admin_api.crud, "get_tenant_by_id", AsyncMock(return_value=tenant))
    monkeypatch.setattr(
        admin_api.crud,
        "add_to_blacklist",
        AsyncMock(return_value=(["+15551234567"], False)),
    )
    monkeypatch.setattr(admin_api.crud, "update_tenant", AsyncMock())
    monkeypatch.setattr(admin_api, "_safe_create_audit_log", AsyncMock(return_value=True))

    request = SimpleNamespace(client=SimpleNamespace(host="127.0.0.1"))
    user = SimpleNamespace(id="admin-1")

    result = await admin_api.api_blacklist_tenant(
        tenant_id="tenant-1",
        request=request,
        db=object(),
        user=user,
    )

    assert result["blacklisted"] is True
    assert "warnings" in result
    assert any("cache" in w.lower() for w in result["warnings"])


@pytest.mark.asyncio
async def test_api_update_tenant_records_old_values(monkeypatch):
    from app.api import admin as admin_api

    tenant = SimpleNamespace(
        id="tenant-1",
        full_name="Jane",
        monthly_income=4000,
        normalized_data={"custom_fields": {"custom_pet": "dog"}},
    )
    monkeypatch.setattr(admin_api.crud, "get_tenant_by_id", AsyncMock(return_value=tenant))
    monkeypatch.setattr(admin_api.crud, "update_tenant", AsyncMock())
    monkeypatch.setattr(admin_api, "_rescore_tenant", AsyncMock(return_value={"score": 80}))
    audit = AsyncMock(return_value=True)
    monkeypatch.setattr(admin_api, "_safe_create_audit_log", audit)

    payload = admin_api.TenantUpdateRequest.model_validate(
        {"full_name": "Janet", "custom_pet": "cat"}
    )
    request = SimpleNamespace(client=SimpleNamespace(host="127.0.0.1"))
    user = SimpleNamespace(id="admin-1")

    result = await admin_api.api_update_tenant(
        tenant_id="tenant-1",
        payload=payload,
        request=request,
        db=object(),
        user=user,
    )

    assert result["success"] is True
    audit.assert_awaited_once()
    kwargs = audit.await_args.kwargs
    assert kwargs["old_value"]["full_name"] == "Jane"
    assert kwargs["old_value"]["custom_pet"] == "dog"
    assert kwargs["new_value"]["full_name"] == "Janet"
    assert kwargs["new_value"]["custom_pet"] == "cat"
    assert kwargs["new_value"]["score"] == 80


@pytest.mark.asyncio
async def test_api_update_tenant_rejects_invalid_custom_field(monkeypatch):
    from app.api import admin as admin_api

    tenant = SimpleNamespace(id="tenant-1", normalized_data={})
    monkeypatch.setattr(admin_api.crud, "get_tenant_by_id", AsyncMock(return_value=tenant))

    payload = admin_api.TenantUpdateRequest.model_validate({"custom_x": ["list"]})
    request = SimpleNamespace(client=SimpleNamespace(host="127.0.0.1"))
    user = SimpleNamespace(id="admin-1")

    with pytest.raises(HTTPException, match="must be text"):
        await admin_api.api_update_tenant(
            tenant_id="tenant-1",
            payload=payload,
            request=request,
            db=object(),
            user=user,
        )
