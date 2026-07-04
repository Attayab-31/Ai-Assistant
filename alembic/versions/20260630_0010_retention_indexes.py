"""Indexes for retention sweeps and stale-call cleanup.

Revision ID: 20260630_0010_retention_indexes
Revises: 20260630_0009_prune_scoring
Create Date: 2026-06-30

The daily retention job deletes by ``created_at`` / ``updated_at`` and scans
for recordings — partial indexes on ``calls`` keep those sweeps fast as the
table grows.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260630_0010_retention_indexes"
down_revision: str | None = "20260630_0009_prune_scoring"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_calls_created_at_all "
        "ON calls (created_at)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_calls_soft_deleted_updated "
        "ON calls (updated_at) WHERE is_deleted = true"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_calls_recording_created "
        "ON calls (created_at) WHERE recording_url IS NOT NULL"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_calls_status_created "
        "ON calls (status, created_at) "
        "WHERE status IN ('initiated', 'in_progress')"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_calls_status_created")
    op.execute("DROP INDEX IF EXISTS idx_calls_recording_created")
    op.execute("DROP INDEX IF EXISTS idx_calls_soft_deleted_updated")
    op.execute("DROP INDEX IF EXISTS idx_calls_created_at_all")
