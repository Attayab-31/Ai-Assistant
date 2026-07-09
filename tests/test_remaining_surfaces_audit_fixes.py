"""Tests for remaining-surfaces audit fixes (Celery validation, test console, TTS, STT)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException


def test_validate_celery_runtime_secrets_omits_telnyx():
    from config import Settings

    settings = Settings(
        environment="production",
        secret_key="x" * 32,
        encryption_key="",
        app_url="https://example.com",
        telnyx_public_key="",
        telnyx_api_key="",
        debug=False,
        admin_password="Admin123!",
        web_workers=4,
    )
    with patch.object(settings, "encryption_key", "dGVzdC1rZXktdGVzdC1rZXktdGVzdC1rZXk="):
        with patch("cryptography.fernet.Fernet"):
            errors = settings.validate_celery_runtime_secrets(require_encryption=True)
    assert not any("TELNYX" in e for e in errors)
    assert not any("ADMIN_PASSWORD" in e for e in errors)
    assert not any("WEB_WORKERS" in e for e in errors)


def test_validate_celery_beat_skips_encryption_requirement():
    from config import Settings

    settings = Settings(
        environment="production",
        secret_key="x" * 32,
        encryption_key="",
        app_url="https://example.com",
        debug=False,
    )
    errors = settings.validate_celery_runtime_secrets(require_encryption=False)
    assert not any("ENCRYPTION_KEY" in e for e in errors)


def test_validate_tts_voice_rejects_unknown_google_voice():
    from app.api.settings import _validate_tts_voice

    with pytest.raises(HTTPException) as exc:
        _validate_tts_voice("google", "not-a-real-voice")
    assert exc.value.status_code == 400
    assert "Google TTS" in str(exc.value.detail)


def test_validate_tts_voice_accepts_deepgram_pattern_voice():
    from app.api.settings import _validate_tts_voice

    _validate_tts_voice("deepgram", "aura-2-thalia-en")


@pytest.mark.asyncio
async def test_list_test_sessions_hides_production_calls(monkeypatch):
    from app.api import test_console

    monkeypatch.setattr(
        test_console.call_handler,
        "get_active_sessions",
        lambda: [
            {"call_id": "test-abc123", "phone_number": "+15551234567"},
            {"call_id": "v3:telnyx-call-id", "phone_number": "+15559876543"},
        ],
    )
    result = await test_console.list_test_sessions(_auth=None)
    assert len(result["sessions"]) == 1
    assert result["sessions"][0]["call_id"] == "test-abc123"


def _args_namespace(*, force: bool):
    ns = MagicMock()
    ns.force = force
    return ns


def test_reset_database_refuses_production_without_force(monkeypatch):
    import scripts.reset_database as reset_mod

    monkeypatch.setattr(
        "config.settings",
        MagicMock(is_production=True),
    )
    monkeypatch.setattr(
        reset_mod.argparse.ArgumentParser,
        "parse_args",
        lambda self: _args_namespace(force=False),
    )
    with pytest.raises(SystemExit) as exc:
        reset_mod.main()
    assert exc.value.code == 1


def test_reset_database_allows_production_with_force(monkeypatch):
    import scripts.reset_database as reset_mod

    monkeypatch.setattr(
        "config.settings",
        MagicMock(is_production=True),
    )
    monkeypatch.setattr(
        reset_mod.argparse.ArgumentParser,
        "parse_args",
        lambda self: _args_namespace(force=True),
    )
    monkeypatch.setattr(reset_mod, "reset_database", AsyncMock())
    assert reset_mod.main() == 0


@pytest.mark.asyncio
async def test_deepgram_prerecorded_empty_response_returns_blank():
    from app.providers.stt.deepgram_stt import DeepgramSTTProvider

    provider = DeepgramSTTProvider(api_key="test-key")
    empty_response = MagicMock()
    empty_response.results.channels = []
    mock_client = MagicMock()
    mock_client.listen.asyncprerecorded.v.return_value.transcribe_file = AsyncMock(
        return_value=empty_response
    )
    provider._client = mock_client
    text = await provider._transcribe_prerecorded(
        b"\xff", encoding="mulaw", sample_rate=8000
    )
    assert text == ""
