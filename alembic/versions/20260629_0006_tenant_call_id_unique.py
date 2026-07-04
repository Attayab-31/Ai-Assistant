"""Enforce one tenant per call (unique tenants.call_id).

Revision ID: 20260629_0006_tenant_unique
Revises: 20260628_0005_hot_indexes
Create Date: 2026-06-29

Finalize can run on more than one worker (hangup webhook + media-stream end).
The in-process lock only dedups within a single worker, so a cross-worker race
could create two tenant rows (and two result emails) for one call. A unique
index on call_id makes the second insert fail fast, which the finalize path
now catches and treats as "already finalized".

Pre-existing duplicates (from before this guard) are collapsed to the earliest
row per call so the unique index can be created.
"""

from typing import Sequence, Union

from alembic import op

revision: str = "20260629_0006_tenant_unique"
down_revision: Union[str, None] = "20260628_0005_hot_indexes"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Collapse any pre-existing duplicates, keeping the earliest tenant per call.
    op.execute(
        """
        DELETE FROM tenants t
        USING tenants dup
        WHERE t.call_id IS NOT NULL
          AND t.call_id = dup.call_id
          AND t.created_at > dup.created_at
        """
    )
    # The plain lookup index is superseded by the unique index below.
    op.execute("DROP INDEX IF EXISTS idx_tenants_call_id")
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_tenants_call_id "
        "ON tenants (call_id) WHERE call_id IS NOT NULL"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_tenants_call_id")
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_tenants_call_id ON tenants (call_id)"
    )
