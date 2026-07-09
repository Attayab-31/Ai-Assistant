"""Tests for admin source-of-truth fixes (latency, health, preview, webhooks)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from app.api import settings as settings_api


def test_merge_voice_latency_profile_alert_keys_overwrites_stale_db_values():
    updates = {"voice_latency_profile": "fast"}
    settings_api._merge_voice_latency_profile_alert_keys(updates)
    assert updates["latency_alert_turn_p95_ms"] == 1000
    assert updates["latency_alert_turn_p95_crit_ms"] == 1800
    assert updates["latency_alert_timeout_rate_pct"] == 3.0


def test_validate_llm_model_rejects_unknown_model():
    with pytest.raises(HTTPException) as exc:
        settings_api._validate_llm_model("openai", "not-a-real-model")
    assert exc.value.status_code == 400
    assert "Unknown model" in str(exc.value.detail)


@pytest.mark.asyncio
async def test_check_provider_health_passes_db_api_keys(monkeypatch):
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
    captured: dict = {}

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

    def _capture_bundle(_snapshot, *, api_keys=None):
        captured["api_keys"] = api_keys
        return bundle

    monkeypatch.setattr(
        "app.core.call_settings.load_call_settings_snapshot",
        AsyncMock(return_value=snapshot),
    )
    monkeypatch.setattr(
        "app.core.call_settings.build_call_provider_bundle",
        _capture_bundle,
    )
    monkeypatch.setattr(
        "app.core.call_settings.capture_provider_api_keys",
        AsyncMock(return_value=keys),
    )

    class _StubSTT:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

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

    await settings_api.check_provider_health(
        db=object(),
        user=SimpleNamespace(id="admin-1"),
    )

    assert captured["api_keys"] is keys


@pytest.mark.asyncio
async def test_recording_saved_rejects_missing_call_control_id(monkeypatch):
    from app.api import webhook as webhook_api

    body = (
        b'{"data":{"event_type":"call.recording.saved",'
        b'"payload":{"recording_id":"rec-1"}}}'
    )
    monkeypatch.setattr(webhook_api, "verify_webhook", AsyncMock(return_value=body))

    from starlette.requests import Request

    request = Request({"type": "http", "method": "POST", "headers": [], "path": "/"})
    with pytest.raises(HTTPException) as exc:
        await webhook_api.telnyx_webhook(request, db=AsyncMock())
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_publish_display_timezone_writes_redis(monkeypatch):
    from app.core import redis_client

    redis = AsyncMock()
    redis.set = AsyncMock()
    monkeypatch.setattr(redis_client, "get_redis", lambda: redis)

    await redis_client.publish_display_timezone("Europe/London")

    redis.set.assert_awaited_once_with(
        redis_client.DISPLAY_TIMEZONE_KEY, "Europe/London"
    )


@pytest.mark.asyncio
async def test_preview_conversation_flow_draft_uses_payload(monkeypatch):
    draft = [
        {
            "id": "q1",
            "state": "CUSTOM_Q1",
            "question": "Draft only?",
            "answer_type": "yes_no",
            "extract_fields": ["draft_only"],
            "order": 1,
            "active": True,
        }
    ]
    monkeypatch.setattr(
        settings_api.crud,
        "get_setting_value",
        AsyncMock(side_effect=lambda _db, key, default=None: default if key != "screening_questions" else [{"id": "saved"}]),
    )
    captured: dict = {}

    async def _fake_preview(db, questions, *, path=None, language="en"):
        captured["questions"] = questions
        return {"flow": [], "paths": [], "selected_path": "", "selected_language": "en"}

    monkeypatch.setattr(settings_api, "_build_questions_preview_response", _fake_preview)

    from app.schemas.settings import QuestionsUpdateRequest, ScreeningQuestion

    payload = QuestionsUpdateRequest(
        questions=[ScreeningQuestion.model_validate(draft[0])]
    )
    await settings_api.preview_conversation_flow_draft(
        payload,
        db=object(),
        user=SimpleNamespace(id="admin-1"),
    )
    assert captured["questions"][0]["state"] == "CUSTOM_Q1"
