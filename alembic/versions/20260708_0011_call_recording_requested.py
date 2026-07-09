"""Freeze call recording decision at call.initiated.

Revision ID: 20260708_0011_recording
Revises: 20260630_0010_retention_indexes
Create Date: 2026-07-08

Stores whether Telnyx recording was requested when the call row was created so
a mid-call admin toggle cannot change recording for a call already ringing.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260708_0011_recording"
down_revision: str | None = "20260630_0010_retention_indexes"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "calls",
        sa.Column(
            "recording_requested",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )


def downgrade() -> None:
    op.drop_column("calls", "recording_requested")
