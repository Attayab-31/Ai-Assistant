"""Add per-call LLM token usage columns.

Revision ID: 20260629_0007_call_tokens
Revises: 20260629_0006_tenant_unique
Create Date: 2026-06-29

Records real token usage per call (summed across the per-turn conversational
LLM calls and any end-of-call extraction) so the admin can see actual spend
instead of estimates. All columns default to 0 for existing rows.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260629_0007_call_tokens"
down_revision: Union[str, None] = "20260629_0006_tenant_unique"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "calls",
        sa.Column("prompt_tokens", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "calls",
        sa.Column(
            "completion_tokens", sa.Integer(), nullable=False, server_default="0"
        ),
    )
    op.add_column(
        "calls",
        sa.Column("total_tokens", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "calls",
        sa.Column("llm_calls", sa.Integer(), nullable=False, server_default="0"),
    )


def downgrade() -> None:
    op.drop_column("calls", "llm_calls")
    op.drop_column("calls", "total_tokens")
    op.drop_column("calls", "completion_tokens")
    op.drop_column("calls", "prompt_tokens")
