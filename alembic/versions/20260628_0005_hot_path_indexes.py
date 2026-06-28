"""Add hot-path indexes for admin performance.

Revision ID: 20260628_0005_hot_indexes
Revises: 20260627_0004_user_perms
Create Date: 2026-06-28

Adds the indexes the admin panel's hottest queries rely on:
- tenants.call_id      → every call-detail view, resend-email, notes, review,
                         qualification override, and the Call↔Tenant joins used
                         by stats/analytics/qualification filters did a seq scan.
- tenants.created_at   → the Applicants list orders by created_at DESC.
- audit_logs.admin_user_id → Activity Log filter by user.
"""

from typing import Sequence, Union

from alembic import op

revision: str = "20260628_0005_hot_indexes"
down_revision: Union[str, None] = "20260627_0004_user_perms"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_tenants_call_id ON tenants (call_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_tenants_created_at ON tenants (created_at DESC)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_audit_logs_admin_user_id "
        "ON audit_logs (admin_user_id)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_audit_logs_admin_user_id")
    op.execute("DROP INDEX IF EXISTS idx_tenants_created_at")
    op.execute("DROP INDEX IF EXISTS idx_tenants_call_id")
