"""Structured provider failure classification and session events."""

import pytest

from app.core.conversation import ConversationSession
from app.core.provider_errors import (
    build_provider_message,
    classify_provider_failure,
    summarize_provider_attempts,
)


class FakeAPIError(Exception):
    status_code = 401
    body = {"error": {"message": "Invalid API Key", "code": "invalid_api_key"}}


def test_classify_invalid_api_key_from_status_error():
    info = classify_provider_failure(
        FakeAPIError("auth failed"),
        service="llm",
        provider="groq",
    )
    assert info["reason"] == "invalid_api_key"
    assert info["http_status"] == 401
    assert info["provider_message"] == "Invalid API Key"
    assert info["provider"] == "groq"


def test_classify_rate_limit_from_string():
    exc = Exception(
        "Error code: 429 - {'error': {'message': 'Rate limit reached', 'code': 'rate_limit_exceeded'}}"
    )
    info = classify_provider_failure(exc, service="llm", provider="openai")
    assert info["reason"] == "rate_limit"
    assert info["http_status"] == 429


def test_classify_quota_exceeded():
    exc = Exception("RESOURCE_EXHAUSTED: quota exceeded for project")
    info = classify_provider_failure(exc, service="llm", provider="gemini")
    assert info["reason"] == "quota_exceeded"


def test_build_provider_message_includes_provider_role_and_reason():
    msg = build_provider_message(
        service="llm",
        provider="groq",
        role="primary",
        outcome="failed",
        reason="invalid_api_key",
        http_status=401,
        provider_message="Invalid API Key",
    )
    assert "Groq" in msg
    assert "primary AI assistant" in msg
    assert "Invalid or missing API key" in msg
    assert "HTTP 401" in msg
    assert "Invalid API Key" in msg


def test_summarize_provider_attempts_lists_each_provider():
    summary = summarize_provider_attempts(
        [
            {"provider": "groq", "role": "primary", "reason": "invalid_api_key"},
            {"provider": "gemini", "role": "fallback", "reason": "rate_limit"},
        ]
    )
    assert "Groq (primary)" in summary
    assert "Google Gemini (fallback)" in summary
    assert "Invalid or missing API key" in summary
    assert "Rate limit exceeded" in summary


def test_session_add_provider_event_structured_fields():
    session = ConversationSession(call_id="t", phone_number="+1")
    info = session.add_provider_event(
        service="llm",
        provider="groq",
        role="primary",
        outcome="failed",
        exc=FakeAPIError("bad key"),
    )
    assert info["reason"] == "invalid_api_key"
    assert len(session.errors) == 1
    err = session.errors[0]
    assert err["type"] == "llm_error"
    assert err["provider"] == "groq"
    assert err["role"] == "primary"
    assert err["outcome"] == "failed"
    assert "Groq" in err["message"]
    assert "Invalid or missing API key" in err["message"]


def test_session_add_provider_event_fallback_success():
    session = ConversationSession(call_id="t", phone_number="+1")
    session.add_provider_event(
        service="llm",
        provider="gemini",
        role="fallback",
        outcome="succeeded",
    )
    err = session.errors[0]
    assert err["type"] == "provider_ok"
    assert "Google Gemini" in err["message"]
    assert "succeeded" in err["message"]
