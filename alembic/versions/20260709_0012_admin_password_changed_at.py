"""Track admin password changes for session invalidation.

Revision ID: 20260709_0012_password_changed
Revises: 20260708_0011_recording
Create Date: 2026-07-09
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260709_0012_password_changed"
down_revision: str | None = "20260708_0011_recording"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "admin_users",
        sa.Column("password_changed_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("admin_users", "password_changed_at")
