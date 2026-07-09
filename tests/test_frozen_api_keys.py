"""Provider API keys are frozen per call at session start."""

import pytest
from unittest.mock import AsyncMock

from app.core.call_settings import (
    ProviderApiKeys,
    build_call_provider_bundle,
    capture_provider_api_keys,
    snapshot_from_map,
)
from app.providers.base import resolve_frozen_credential
from app.providers.llm.groq_llm import GroqLLMProvider
from app.providers.tts.deepgram_tts import DeepgramTTSProvider


def test_provider_api_keys_configured():
    keys = ProviderApiKeys(
        groq="g1",
        openai="o1",
        deepgram="d1",
        google_application_credentials="/path/to/creds.json",
    )
    assert keys.configured("groq")
    assert keys.configured("deepgram")
    assert keys.configured("google")
    assert not keys.configured("gemini")


def test_resolve_frozen_credential_prefers_frozen_value(monkeypatch):
    monkeypatch.setattr("config.settings.groq_api_key", "live-key")
    assert (
        resolve_frozen_credential("frozen-key", settings_attr="groq_api_key")
        == "frozen-key"
    )


def test_resolve_frozen_credential_uses_live_when_not_frozen(monkeypatch):
    monkeypatch.setattr("config.settings.groq_api_key", "live-key")
    assert resolve_frozen_credential(None, settings_attr="groq_api_key") == "live-key"


def test_groq_llm_uses_frozen_api_key():
    provider = GroqLLMProvider(api_key="call-start-groq")
    assert provider._api_key == "call-start-groq"
    assert (
        resolve_frozen_credential(provider._api_key, settings_attr="groq_api_key")
        == "call-start-groq"
    )


@pytest.mark.asyncio
async def test_build_call_provider_bundle_freezes_keys(monkeypatch):
    monkeypatch.setattr("config.settings.groq_api_key", "snap-groq")
    monkeypatch.setattr("config.settings.openai_api_key", "snap-openai")
    monkeypatch.setattr("config.settings.openrouter_api_key", "")
    monkeypatch.setattr("config.settings.gemini_api_key", "")
    monkeypatch.setattr("config.settings.deepgram_api_key", "snap-deepgram")
    monkeypatch.setattr("config.settings.google_application_credentials", "")

    snapshot = snapshot_from_map(
        {
            "active_llm_provider": "groq",
            "active_stt_provider": "deepgram",
            "active_tts_provider": "deepgram",
            "auto_fallback_enabled": True,
        }
    )
    bundle = build_call_provider_bundle(snapshot)
    assert bundle.api_keys.groq == "snap-groq"
    assert bundle.api_keys.deepgram == "snap-deepgram"
    assert isinstance(bundle.llm_by_name["groq"], GroqLLMProvider)
    assert bundle.llm_by_name["groq"]._api_key == "snap-groq"
    assert isinstance(bundle.tts_by_name["deepgram"], DeepgramTTSProvider)
    assert bundle.tts_by_name["deepgram"]._api_key == "snap-deepgram"

    monkeypatch.setattr("config.settings.groq_api_key", "rotated-live")
    assert bundle.llm_by_name["groq"]._api_key == "snap-groq"
    assert (
        resolve_frozen_credential(
            bundle.llm_by_name["groq"]._api_key, settings_attr="groq_api_key"
        )
        == "snap-groq"
    )


@pytest.mark.asyncio
async def test_capture_provider_api_keys_prefers_db_encrypted(monkeypatch):
    monkeypatch.setattr("config.settings.groq_api_key", "live-groq")
    monkeypatch.setattr("config.settings.deepgram_api_key", "live-deepgram")
    monkeypatch.setattr(
        "app.db.crud.fetch_settings_batch",
        AsyncMock(
            return_value={
                "groq_api_key_encrypted": "enc-groq",
                "deepgram_api_key_encrypted": "enc-deepgram",
            }
        ),
    )

    def _decrypt(value: str) -> str:
        return {
            "enc-groq": "db-groq",
            "enc-deepgram": "db-deepgram",
        }[value]

    monkeypatch.setattr("app.utils.security.decrypt_value", _decrypt)

    keys = await capture_provider_api_keys(db=object())
    assert keys.groq == "db-groq"
    assert keys.deepgram == "db-deepgram"
