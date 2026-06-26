"""Add Ready Rentals screening fields.

Revision ID: 20260623_0003_ready_rentals
Revises: 20260622_0002_perf_indexes
Create Date: 2026-06-23
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "20260623_0003_ready_rentals"
down_revision: Union[str, None] = "20260622_0002_perf_indexes"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("tenants", sa.Column("full_name", sa.String(length=255), nullable=True))
    op.add_column("tenants", sa.Column("contact_phone", sa.String(length=20), nullable=True))
    op.add_column("tenants", sa.Column("email", sa.String(length=255), nullable=True))
    op.add_column("tenants", sa.Column("occupants_count", sa.Integer(), nullable=True))
    op.add_column("tenants", sa.Column("has_pets", sa.Boolean(), nullable=True))
    op.add_column("tenants", sa.Column("pets_raw", sa.Text(), nullable=True))
    op.add_column("tenants", sa.Column("pet_type", sa.String(length=100), nullable=True))
    op.add_column("tenants", sa.Column("pet_breed", sa.String(length=100), nullable=True))
    op.add_column("tenants", sa.Column("pet_weight", sa.Integer(), nullable=True))
    op.alter_column("tenants", "eviction_raw", type_=sa.Text(), existing_type=sa.String(length=500))
    op.add_column("tenants", sa.Column("eviction_circumstances", sa.Text(), nullable=True))
    op.add_column("tenants", sa.Column("current_residence", sa.String(length=500), nullable=True))
    op.add_column("tenants", sa.Column("residence_duration", sa.String(length=255), nullable=True))
    op.add_column("tenants", sa.Column("move_timing", sa.String(length=255), nullable=True))
    op.add_column("tenants", sa.Column("employer", sa.String(length=255), nullable=True))
    op.add_column("tenants", sa.Column("employment_duration", sa.String(length=255), nullable=True))
    op.add_column("tenants", sa.Column("general_notes", sa.Text(), nullable=True))
    op.add_column("tenants", sa.Column("special_notes", sa.Text(), nullable=True))
    op.add_column("tenants", sa.Column("human_requested", sa.Boolean(), nullable=True))
    op.add_column("tenants", sa.Column("callback_requested", sa.Boolean(), nullable=True))
    op.add_column("tenants", sa.Column("stop_requested", sa.Boolean(), nullable=True))
    op.add_column("tenants", sa.Column("raw_answers", postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column("tenants", sa.Column("normalized_data", postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column("tenants", sa.Column("answered_states", postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column("tenants", sa.Column("refused_states", postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column("tenants", sa.Column("faq_topics", postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column("tenants", sa.Column("control_flags", postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column("tenants", sa.Column("qualification_details", postgresql.JSONB(astext_type=sa.Text()), nullable=True))


def downgrade() -> None:
    op.drop_column("tenants", "qualification_details")
    op.drop_column("tenants", "control_flags")
    op.drop_column("tenants", "faq_topics")
    op.drop_column("tenants", "refused_states")
    op.drop_column("tenants", "answered_states")
    op.drop_column("tenants", "normalized_data")
    op.drop_column("tenants", "raw_answers")
    op.drop_column("tenants", "stop_requested")
    op.drop_column("tenants", "callback_requested")
    op.drop_column("tenants", "human_requested")
    op.drop_column("tenants", "special_notes")
    op.drop_column("tenants", "general_notes")
    op.drop_column("tenants", "employment_duration")
    op.drop_column("tenants", "employer")
    op.drop_column("tenants", "move_timing")
    op.drop_column("tenants", "residence_duration")
    op.drop_column("tenants", "current_residence")
    op.drop_column("tenants", "eviction_circumstances")
    op.alter_column("tenants", "eviction_raw", type_=sa.String(length=500), existing_type=sa.Text())
    op.drop_column("tenants", "pet_weight")
    op.drop_column("tenants", "pet_breed")
    op.drop_column("tenants", "pet_type")
    op.drop_column("tenants", "pets_raw")
    op.drop_column("tenants", "has_pets")
    op.drop_column("tenants", "occupants_count")
    op.drop_column("tenants", "email")
    op.drop_column("tenants", "contact_phone")
    op.drop_column("tenants", "full_name")

