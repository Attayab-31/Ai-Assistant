"""Add performance indexes

Revision ID: 20260622_0002_perf_indexes
Revises: 20260619_0001
Create Date: 2026-06-22 12:20:00.000000

"""
from alembic import op


# revision identifiers, used by Alembic.
revision = '20260622_0002_perf_indexes'
down_revision = '20260619_0001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create performance indexes."""
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_calls_created_at
        ON calls (created_at)
        WHERE is_deleted = false
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_calls_created_status
        ON calls (created_at, status)
        WHERE is_deleted = false
        """
    )


def downgrade() -> None:
    """Drop performance indexes."""
    op.execute('DROP INDEX IF EXISTS idx_calls_created_status')
    op.execute('DROP INDEX IF EXISTS idx_calls_created_at')
