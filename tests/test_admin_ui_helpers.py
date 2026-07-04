"""Tests for admin UI helper polish (Phase 9)."""

from app.utils.helpers import (
    friendly_audit_action,
    friendly_audit_entity,
    friendly_provider_name,
    pagination_url,
)


def test_friendly_provider_name():
    assert friendly_provider_name("groq", "llm") == "Groq"
    assert friendly_provider_name("deepgram", "stt") == "Deepgram"
    assert friendly_provider_name("google", "tts") == "Google"


def test_friendly_audit_action():
    assert friendly_audit_action("updated_general_settings") == "Saved general settings"
    assert friendly_audit_action("admin_login") == "Signed in"


def test_friendly_audit_entity():
    assert friendly_audit_entity("tenant") == "Applicant"
    assert friendly_audit_entity("call") == "Call"


def test_pagination_url_preserves_filters():
    url = pagination_url("/admin/calls", 2, {"status": "completed", "phone": "+1555"})
    assert url.startswith("/admin/calls?")
    assert "page=2" in url
    assert "status=completed" in url
