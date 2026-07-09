"""Tests for admin UI helper polish (Phase 9)."""

from app.utils.helpers import (
    date_range_from_days,
    friendly_audit_action,
    friendly_audit_entity,
    friendly_provider_name,
    list_filter_url,
    pagination_url,
    side_effect_alerts_from_error_log,
    tenant_display_name,
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


def test_list_filter_url_clears_days():
    url = list_filter_url("/admin/tenants", {"review": "unreviewed", "days": 7}, days=None)
    assert "review=unreviewed" in url
    assert "days" not in url
    assert "page=1" in url


def test_date_range_from_days():
    start, end = date_range_from_days(7)
    assert start is not None
    assert end is None
    assert date_range_from_days(None) == (None, None)


def test_tenant_display_name():
    class T:
        def __init__(self, name, phone):
            self.full_name = name
            self.phone_number = phone

    assert tenant_display_name(T("Jordan Lee", "+15551234567")) == "Jordan Lee"
    assert "(" in tenant_display_name(T("", "+15551234567"))


def test_side_effect_alerts_from_error_log():
    alerts = side_effect_alerts_from_error_log(
        {"email_queue": "Connection refused", "errors": []}
    )
    assert len(alerts) == 1
    assert alerts[0]["kind"] == "email"
    assert "Connection refused" in alerts[0]["detail"]

    both = side_effect_alerts_from_error_log(
        {"email_queue": "x", "crm_queue": "y"}
    )
    assert {a["kind"] for a in both} == {"email", "crm"}

    questions = side_effect_alerts_from_error_log(
        {"questions_config_fallback": "Duplicate question states are not allowed"}
    )
    assert len(questions) == 1
    assert questions[0]["kind"] == "questions"
    assert "invalid question configuration" in questions[0]["title"].lower()
    assert "Duplicate" in questions[0]["detail"]
