"""Add per-area permission scopes to admin users.

Revision ID: 20260627_0004_user_perms
Revises: 20260623_0003_ready_rentals
Create Date: 2026-06-27
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "20260627_0004_user_perms"
down_revision: Union[str, None] = "20260623_0003_ready_rentals"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "admin_users",
        sa.Column("permissions", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("admin_users", "permissions")
