"""Tests for home onboarding checklist helpers."""

from app.utils.helpers import build_onboarding_checklist, is_property_configured


def test_is_property_configured_detects_custom_name():
    assert is_property_configured("Sunset Apartments", default_property_name="Ready Rentals Online")


def test_is_property_configured_detects_custom_greeting():
    assert is_property_configured(
        "Ready Rentals Online",
        greeting_message="Hello from our team.",
        default_property_name="Ready Rentals Online",
    )


def test_is_property_configured_default_seed_is_incomplete():
    assert not is_property_configured(
        "Ready Rentals Online",
        default_property_name="Ready Rentals Online",
    )


def test_build_onboarding_checklist_hides_when_complete():
    result = build_onboarding_checklist(
        property_name="Oak Grove",
        total_calls=3,
        reviewed_applicants=1,
        can_settings=True,
        can_edit=True,
        can_tenants=True,
    )
    assert result["complete"] is True
    assert result["show"] is False
    assert result["done_count"] == 3


def test_build_onboarding_checklist_shows_pending_steps():
    result = build_onboarding_checklist(
        total_calls=0,
        reviewed_applicants=0,
        needs_review_count=2,
        can_settings=True,
        can_edit=True,
        can_tenants=True,
    )
    assert result["show"] is True
    assert result["done_count"] == 0
    assert len(result["steps"]) == 3
    assert result["steps"][0]["label"] == "Review your property details"
    assert result["steps"][2]["href"] == "/admin/tenants?review=unreviewed"


def test_build_onboarding_checklist_property_saved():
    result = build_onboarding_checklist(
        property_settings_saved=True,
        can_settings=True,
        can_edit=False,
        can_tenants=False,
    )
    assert result["steps"][0]["done"] is True


def test_build_onboarding_checklist_respects_permissions():
    result = build_onboarding_checklist(
        can_settings=True,
        can_edit=False,
        can_tenants=False,
    )
    assert len(result["steps"]) == 1
    assert result["steps"][0]["id"] == "property"
