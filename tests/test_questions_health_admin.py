"""Tests for admin questions health banner context."""

import pytest

from app.core.question_flow import default_questions_v2, runtime_question_errors


@pytest.mark.asyncio
async def test_render_admin_page_includes_runtime_errors_on_dashboard(monkeypatch):
    from app.api import admin as admin_module

    class FakeUser:
        email = "admin@test.com"
        full_name = "Admin"
        role = "admin"
        can_edit = True

        def can(self, scope: str) -> bool:
            return True

    async def fake_get_setting_value(db, key, default=None):
        if key == "screening_questions":
            return []
        return default

    async def fake_count_review(db):
        return 0

    monkeypatch.setattr(admin_module.crud, "get_setting_value", fake_get_setting_value)
    monkeypatch.setattr(
        admin_module.crud, "count_tenants_needing_review", fake_count_review
    )

    captured: dict = {}

    def fake_template_response(name, context, **kwargs):
        captured["template"] = name
        captured["context"] = context
        return "ok"

    monkeypatch.setattr(admin_module.templates, "TemplateResponse", fake_template_response)

    await admin_module._render_admin_page(
        None,
        None,
        "dashboard.html",
        FakeUser(),
        active_page="dashboard",
    )

    assert captured["template"] == "dashboard.html"
    errors = captured["context"]["questions_runtime_errors"]
    assert errors
    assert "system defaults" not in errors[0].lower()
    assert "blocked" in errors[0].lower()


@pytest.mark.asyncio
async def test_render_admin_page_skips_question_health_without_settings_scope(
    monkeypatch,
):
    from app.api import admin as admin_module

    class Viewer:
        email = "view@test.com"
        full_name = "Viewer"
        role = "viewer"
        can_edit = False

        def can(self, scope: str) -> bool:
            return scope == "calls"

    async def fake_count_review(db):
        return 0

    monkeypatch.setattr(
        admin_module.crud, "count_tenants_needing_review", fake_count_review
    )

    captured: dict = {}

    def fake_template_response(name, context, **kwargs):
        captured["context"] = context
        return "ok"

    monkeypatch.setattr(admin_module.templates, "TemplateResponse", fake_template_response)

    await admin_module._render_admin_page(
        None,
        None,
        "dashboard.html",
        Viewer(),
        active_page="dashboard",
    )

    assert "questions_runtime_errors" not in captured["context"]


def test_runtime_errors_match_between_admin_and_call_snapshot():
    """Admin banner and call snapshot should agree on invalid saved config."""
    from app.core.call_settings import snapshot_from_map

    invalid = default_questions_v2()
    for i, q in enumerate(invalid):
        if q["state"] == "Q1_FULL_NAME":
            bad = dict(q)
            bad["extract_fields"] = ["renamed_name"]
            invalid[i] = bad
            break

    admin_errors = runtime_question_errors(invalid)
    snap = snapshot_from_map({"screening_questions": invalid})
    assert admin_errors
    assert snap.questions_runtime_fallback == admin_errors[0]
