"""app/models/user.py — Admin user ORM model + role/permission helpers."""

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, String, Text, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base

# ──────────────────────────────────────────────────────────────────────────────
# Access-control model
# ──────────────────────────────────────────────────────────────────────────────
# Roles describe *who* an account is; scopes describe *which areas* they may use.
#
#   super_admin  full access incl. user management + settings (env-seeded only)
#   admin        full operational access + settings, but NOT user management
#   staff        custom scoped areas, may edit within them
#   viewer       custom scoped areas, read-only
#
# Scopes are the feature areas of the admin panel that can be granted to staff
# and viewer accounts. "dashboard" (Home) is always available; "accounts" (user
# management) is implicitly super_admin only.
PERMISSION_SCOPES: dict[str, str] = {
    "monitor": "Live Monitor",
    "calls": "Calls",
    "tenants": "Applicants",
    "analytics": "Analytics",
    "settings": "Settings",
    "audit": "Activity Log",
}

ALL_SCOPES: frozenset[str] = frozenset(PERMISSION_SCOPES)

# Roles that may modify data (the rest are read-only).
EDIT_ROLES: frozenset[str] = frozenset({"super_admin", "admin", "staff"})

# Roles that can be assigned through the admin UI (super_admin is env-only).
ASSIGNABLE_ROLES: tuple[str, ...] = ("admin", "staff", "viewer")

# Fallback scopes for a legacy staff/viewer row whose permissions were never set
# (NULL). New accounts always get an explicit list, so this only protects old
# data: everything except settings.
_LEGACY_DEFAULT_SCOPES: frozenset[str] = ALL_SCOPES - {"settings"}


def validate_assignable_scopes(role: str, scopes: list[str]) -> list[str]:
    """Validate and normalize scope list for staff/viewer roles."""
    if role not in ASSIGNABLE_ROLES:
        raise ValueError(f"Invalid role: {role}")
    cleaned = sorted({s.strip() for s in scopes if s.strip()})
    invalid = set(cleaned) - ALL_SCOPES
    if invalid:
        raise ValueError(
            f"Unknown access areas: {', '.join(sorted(invalid))}"
        )
    if role in ("staff", "viewer") and not cleaned:
        raise ValueError(
            "Pick at least one access area for staff and viewer accounts."
        )
    return cleaned


class AdminUser(Base):
    """Admin user with role + per-area permission scopes."""

    __tablename__ = "admin_users"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    email: Mapped[str] = mapped_column(
        String(255), unique=True, nullable=False, index=True
    )
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str | None] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(
        String(20), default="admin"
    )  # super_admin, admin, staff, viewer
    # Comma-separated list of granted scope keys (only meaningful for staff /
    # viewer; super_admin and admin always get everything). NULL = legacy row.
    permissions: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    last_login: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # ── Access helpers ────────────────────────────────────────────────────────
    @property
    def is_env_account(self) -> bool:
        """True only for the permanent super admin defined by ADMIN_EMAIL.

        This is the single protected account (it is re-seeded from the
        environment). Any *other* account — including a leftover super admin
        from a previous ADMIN_EMAIL value — can be edited or deleted by a
        super admin.
        """
        from config import settings

        admin_email = (settings.admin_email or "").strip().lower()
        return bool(admin_email) and (self.email or "").strip().lower() == admin_email

    @property
    def can_edit(self) -> bool:
        """Whether this account may modify data (vs. read-only viewer)."""
        return self.role in EDIT_ROLES

    @property
    def effective_scopes(self) -> set[str]:
        """The set of feature-area scopes this account can access."""
        if self.role in ("super_admin", "admin"):
            return set(ALL_SCOPES)
        if self.permissions is None:
            return set(_LEGACY_DEFAULT_SCOPES)
        return {p.strip() for p in self.permissions.split(",") if p.strip()}

    def can(self, scope: str) -> bool:
        """True if this account may access the given area/scope."""
        if scope == "accounts":
            return self.role == "super_admin"
        if scope == "dashboard":
            return True
        return scope in self.effective_scopes

    def __repr__(self) -> str:
        return f"<AdminUser {self.email} | {self.role}>"
