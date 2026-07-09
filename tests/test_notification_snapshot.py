"""Tests for frozen email/CRM notification settings at call start."""

from unittest.mock import AsyncMock

import pytest

from app.core.call_settings import (
    NotificationSettingsSnapshot,
    load_notification_settings_from_db,
    notification_settings_email_dict,
    notification_settings_from_map,
    snapshot_from_map,
)


def test_notification_settings_from_map_defaults():
    ns = notification_settings_from_map({})
    assert ns.email_notifications_enabled is True
    assert ns.email_qualified_only is False
    assert ns.crm_webhook_url == ""
    assert ns.crm_notifications_enabled is False


def test_crm_notifications_enabled_legacy_url_without_flag():
    """Existing installs with a webhook URL keep CRM delivery until explicitly disabled."""
    ns = notification_settings_from_map(
        {"crm_webhook_url": "https://crm.example.com/hook"}
    )
    assert ns.crm_notifications_enabled is True


def test_crm_notifications_enabled_explicit_false_with_url():
    ns = notification_settings_from_map(
        {
            "crm_webhook_url": "https://crm.example.com/hook",
            "crm_notifications_enabled": "false",
        }
    )
    assert ns.crm_notifications_enabled is False


def test_crm_notifications_active():
    from app.core.call_settings import crm_notifications_active

    off = NotificationSettingsSnapshot(
        crm_notifications_enabled=False,
        crm_webhook_url="https://crm.example/hook",
    )
    assert crm_notifications_active(off) is False

    on = NotificationSettingsSnapshot(
        crm_notifications_enabled=True,
        crm_webhook_url="https://crm.example/hook",
    )
    assert crm_notifications_active(on) is True

    no_url = NotificationSettingsSnapshot(
        crm_notifications_enabled=True,
        crm_webhook_url="",
    )
    assert crm_notifications_active(no_url) is False


def test_notification_settings_from_map_parses_admin_values():
    ns = notification_settings_from_map(
        {
            "email_notifications_enabled": "false",
            "landlord_email": "owner@example.com",
            "email_from_name": "Screening Bot",
            "email_from_address": "bot@example.com",
            "email_qualified_only": "true",
            "email_include_transcript": "true",
            "email_subject_template": "Result: {name}",
            "email_body_template": "Score {score}",
            "cc_emails": "cc@example.com",
            "bcc_emails": "bcc@example.com",
            "timezone": "America/Chicago",
            "crm_webhook_url": "https://crm.example.com/hook",
            "crm_webhook_secret": "sekret",
        }
    )
    assert ns.email_notifications_enabled is False
    assert ns.landlord_email == "owner@example.com"
    assert ns.email_qualified_only is True
    assert ns.email_include_transcript is True
    assert ns.crm_webhook_url == "https://crm.example.com/hook"
    assert ns.crm_webhook_secret == "sekret"
    assert ns.crm_notifications_enabled is True


def test_notification_settings_from_map_decrypts_crm_webhook_secret():
    from app.utils.security import encrypt_value

    secret = "super-secret-signing-key"
    ns = notification_settings_from_map(
        {
            "crm_webhook_secret": encrypt_value(secret),
        }
    )
    assert ns.crm_webhook_secret == secret


def test_notification_settings_email_dict_shape():
    ns = NotificationSettingsSnapshot(
        landlord_email="a@b.com",
        crm_webhook_secret="xyz",
    )
    payload = notification_settings_email_dict(ns)
    assert payload["landlord_email"] == "a@b.com"
    assert payload["crm_webhook_secret"] == "xyz"
    assert "email_notifications_enabled" not in payload


def test_call_settings_snapshot_includes_notification_settings():
    snap = snapshot_from_map(
        {
            "active_llm_provider": "groq",
            "active_stt_provider": "deepgram",
            "active_tts_provider": "deepgram",
            "landlord_email": "snap@example.com",
            "email_notifications_enabled": "true",
            "crm_webhook_url": "https://hooks.example.com/screen",
        }
    )
    assert snap.notification_settings.landlord_email == "snap@example.com"
    assert snap.notification_settings.email_notifications_enabled is True
    assert snap.notification_settings.crm_webhook_url == "https://hooks.example.com/screen"
    assert snap.notification_settings.crm_notifications_enabled is True


def test_resolve_email_settings_uses_frozen_dict():
    from app.services.email_service import resolve_email_settings

    frozen = notification_settings_email_dict(
        NotificationSettingsSnapshot(
            landlord_email="frozen@example.com",
            email_qualified_only=True,
        )
    )
    resolved = resolve_email_settings(frozen)
    assert resolved["landlord_email"] == "frozen@example.com"
    assert resolved["email_qualified_only"] is True


def test_notification_settings_persist_and_read_from_tenant():
    from types import SimpleNamespace

    from app.core.call_settings import (
        NOTIFICATION_SETTINGS_PERSIST_KEY,
        has_notification_settings_snapshot,
        notification_settings_email_dict_from_tenant,
        notification_settings_persist_dict,
    )

    snap = NotificationSettingsSnapshot(
        landlord_email="frozen@example.com",
        email_subject_template="Hi {name}",
        email_include_transcript=True,
        crm_webhook_url="https://crm.example/hook",
        crm_notifications_enabled=True,
    )
    persisted = notification_settings_persist_dict(snap)
    assert persisted["landlord_email"] == "frozen@example.com"
    assert persisted["crm_webhook_url"] == "https://crm.example/hook"
    assert persisted["crm_notifications_enabled"] is True
    assert persisted["email_notifications_enabled"] is True

    tenant = SimpleNamespace(
        normalized_data={NOTIFICATION_SETTINGS_PERSIST_KEY: persisted}
    )
    assert has_notification_settings_snapshot(tenant) is True
    email = notification_settings_email_dict_from_tenant(tenant)
    assert email is not None
    assert email["landlord_email"] == "frozen@example.com"
    assert email["email_include_transcript"] is True
    assert "crm_webhook_url" not in email


def test_notification_settings_from_tenant_missing_returns_none():
    from types import SimpleNamespace

    from app.core.call_settings import notification_settings_email_dict_from_tenant

    assert notification_settings_email_dict_from_tenant(None) is None
    assert notification_settings_email_dict_from_tenant(
        SimpleNamespace(normalized_data={})
    ) is None


@pytest.mark.asyncio
async def test_load_notification_settings_from_db_uses_unmasked_batch(monkeypatch):
    fetch = AsyncMock(
        return_value={
            "landlord_email": "owner@example.com",
            "crm_webhook_url": "https://crm.example/hook",
            "crm_webhook_secret": "real-secret",
            "crm_notifications_enabled": True,
        }
    )
    monkeypatch.setattr("app.db.crud.fetch_settings_batch", fetch)

    snap = await load_notification_settings_from_db(db=object())
    assert snap.landlord_email == "owner@example.com"
    assert snap.crm_webhook_url == "https://crm.example/hook"
    assert snap.crm_webhook_secret == "real-secret"
    assert snap.crm_notifications_enabled is True


def test_normalize_email_settings_decrypts_crm_secret():
    from app.services.email_service import normalize_email_settings
    from app.utils.security import encrypt_value

    secret = "crm-signature-secret"
    out = normalize_email_settings(
        {
            "crm_webhook_secret": encrypt_value(secret),
        }
    )
    assert out["crm_webhook_secret"] == secret
