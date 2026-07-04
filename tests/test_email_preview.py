"""Tests for email template preview and test-send rendering."""

from app.services.email_service import build_test_email_preview


def test_build_test_email_preview_uses_saved_templates():
    settings = {
        "email_subject_template": "Applicant {name} — {score}",
        "email_body_template": "<p>Phone: {phone}, status {status}</p>",
        "email_include_transcript": False,
    }
    subject, html = build_test_email_preview(settings)
    assert "Jane Doe" in subject
    assert "82" in subject
    assert "+1 (555) 123-4567" in html
    assert "PRE-QUALIFIED" in html or "qualified" in html.lower()
