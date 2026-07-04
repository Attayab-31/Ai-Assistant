"""Tests for admin team accounts — roles, scopes, and validation."""

import pytest

from app.models.user import (
    ALL_SCOPES,
    PERMISSION_SCOPES,
    AdminUser,
    validate_assignable_scopes,
)


def _user(
    *,
    role: str = "staff",
    permissions: str | None = "calls,tenants",
    is_active: bool = True,
    email: str = "staff@example.com",
) -> AdminUser:
    return AdminUser(
        email=email,
        hashed_password="x",
        full_name="Test User",
        role=role,
        permissions=permissions,
        is_active=is_active,
    )


def test_super_admin_has_all_scopes_and_accounts():
    user = _user(role="super_admin", permissions=None)
    assert user.can("accounts") is True
    assert user.can("settings") is True
    assert user.effective_scopes == set(ALL_SCOPES)
    assert user.can_edit is True


def test_admin_has_all_scopes_but_not_accounts():
    user = _user(role="admin", permissions=None)
    assert user.can("settings") is True
    assert user.can("accounts") is False
    assert user.can_edit is True


def test_staff_limited_to_granted_scopes():
    user = _user(role="staff", permissions="calls,monitor")
    assert user.can("calls") is True
    assert user.can("monitor") is True
    assert user.can("settings") is False
    assert user.can("accounts") is False
    assert user.can_edit is True


def test_viewer_is_read_only():
    user = _user(role="viewer", permissions="calls")
    assert user.can("calls") is True
    assert user.can_edit is False


def test_legacy_null_permissions_default_excludes_settings():
    user = _user(role="staff", permissions=None)
    assert "settings" not in user.effective_scopes
    assert "calls" in user.effective_scopes


def test_dashboard_always_accessible():
    user = _user(role="viewer", permissions="audit")
    assert user.can("dashboard") is True


def test_validate_scopes_requires_at_least_one_for_staff():
    with pytest.raises(ValueError, match="at least one"):
        validate_assignable_scopes("staff", [])


def test_validate_scopes_rejects_unknown_scope():
    with pytest.raises(ValueError, match="Unknown access areas"):
        validate_assignable_scopes("staff", ["calls", "not_real"])


def test_validate_scopes_accepts_known_scopes():
    cleaned = validate_assignable_scopes("viewer", ["audit", "calls"])
    assert cleaned == ["audit", "calls"]


def test_permission_scope_labels_complete():
    assert set(PERMISSION_SCOPES) == set(ALL_SCOPES)
