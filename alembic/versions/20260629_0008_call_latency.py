"""Add per-call response-latency columns.

Revision ID: 20260629_0008_call_latency
Revises: 20260629_0007_call_tokens
Create Date: 2026-06-29

Records real per-call response latency (milliseconds) so the admin can see
WHERE time is spent on each turn: the LLM brain, TTS voice synthesis, and the
full turn (transcript-in to audio-ready). Averages are computed per call at
finalize; ``turn_count`` is the number of timed turns (used for weighting in
dashboard rollups). All columns default to 0 for existing rows.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260629_0008_call_latency"
down_revision: Union[str, None] = "20260629_0007_call_tokens"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "calls",
        sa.Column("avg_llm_ms", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "calls",
        sa.Column("avg_tts_ms", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "calls",
        sa.Column("avg_turn_ms", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "calls",
        sa.Column("max_turn_ms", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "calls",
        sa.Column("turn_count", sa.Integer(), nullable=False, server_default="0"),
    )


def downgrade() -> None:
    op.drop_column("calls", "turn_count")
    op.drop_column("calls", "max_turn_ms")
    op.drop_column("calls", "avg_turn_ms")
    op.drop_column("calls", "avg_tts_ms")
    op.drop_column("calls", "avg_llm_ms")
