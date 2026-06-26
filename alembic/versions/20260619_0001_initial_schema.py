"""Initial schema.

Revision ID: 20260619_0001
Revises:
Create Date: 2026-06-19
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "20260619_0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "admin_users",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("hashed_password", sa.String(length=255), nullable=False),
        sa.Column("full_name", sa.String(length=255), nullable=True),
        sa.Column("role", sa.String(length=20), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("last_login", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_admin_users_email", "admin_users", ["email"], unique=True)

    op.create_table(
        "calls",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("call_id", sa.String(length=255), nullable=False),
        sa.Column("phone_number", sa.String(length=20), nullable=False),
        sa.Column("direction", sa.String(length=10), nullable=False),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_seconds", sa.Integer(), nullable=True),
        sa.Column("full_transcript", sa.Text(), nullable=True),
        sa.Column("recording_url", sa.Text(), nullable=True),
        sa.Column("questions_answered", sa.Integer(), nullable=False),
        sa.Column("call_completed", sa.Boolean(), nullable=False),
        sa.Column("stt_provider", sa.String(length=50), nullable=True),
        sa.Column("llm_provider", sa.String(length=50), nullable=True),
        sa.Column("tts_provider", sa.String(length=50), nullable=True),
        sa.Column("error_log", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("is_deleted", sa.Boolean(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_calls_call_id", "calls", ["call_id"], unique=True)
    op.create_index("ix_calls_phone_number", "calls", ["phone_number"], unique=False)
    op.create_index("ix_calls_status", "calls", ["status"], unique=False)

    op.create_table(
        "system_settings",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("key", sa.String(length=255), nullable=False),
        sa.Column("value", sa.Text(), nullable=False),
        sa.Column("value_type", sa.String(length=20), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("is_sensitive", sa.Boolean(), nullable=False),
        sa.Column("updated_by", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["updated_by"], ["admin_users.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_system_settings_key", "system_settings", ["key"], unique=True)

    op.create_table(
        "tenants",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("call_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("phone_number", sa.String(length=20), nullable=False),
        sa.Column("adults_count", sa.Integer(), nullable=True),
        sa.Column("children_count", sa.Integer(), nullable=True),
        sa.Column("monthly_income", sa.Numeric(10, 2), nullable=True),
        sa.Column("income_raw", sa.String(length=255), nullable=True),
        sa.Column("has_eviction", sa.Boolean(), nullable=True),
        sa.Column("eviction_raw", sa.String(length=500), nullable=True),
        sa.Column("move_in_date", sa.Date(), nullable=True),
        sa.Column("move_in_raw", sa.String(length=255), nullable=True),
        sa.Column("move_reason", sa.Text(), nullable=True),
        sa.Column("qualification_score", sa.Integer(), nullable=True),
        sa.Column("qualification_status", sa.String(length=20), nullable=True),
        sa.Column(
            "disqualify_reasons",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("email_sent", sa.Boolean(), nullable=False),
        sa.Column("email_sent_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("reviewed_by_admin", sa.Boolean(), nullable=False),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_blacklisted", sa.Boolean(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["call_id"], ["calls.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_tenants_phone_number", "tenants", ["phone_number"], unique=False)
    op.create_index(
        "ix_tenants_qualification_status",
        "tenants",
        ["qualification_status"],
        unique=False,
    )

    op.create_table(
        "audit_logs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("admin_user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("action", sa.String(length=100), nullable=False),
        sa.Column("entity_type", sa.String(length=50), nullable=True),
        sa.Column("entity_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("old_value", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("new_value", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("ip_address", sa.String(length=45), nullable=True),
        sa.Column("user_agent", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["admin_user_id"], ["admin_users.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_audit_logs_action", "audit_logs", ["action"], unique=False)
    op.create_index(
        "ix_audit_logs_created_at", "audit_logs", ["created_at"], unique=False
    )


def downgrade() -> None:
    op.drop_index("ix_audit_logs_created_at", table_name="audit_logs")
    op.drop_index("ix_audit_logs_action", table_name="audit_logs")
    op.drop_table("audit_logs")

    op.drop_index("ix_tenants_qualification_status", table_name="tenants")
    op.drop_index("ix_tenants_phone_number", table_name="tenants")
    op.drop_table("tenants")

    op.drop_index("ix_system_settings_key", table_name="system_settings")
    op.drop_table("system_settings")

    op.drop_index("ix_calls_status", table_name="calls")
    op.drop_index("ix_calls_phone_number", table_name="calls")
    op.drop_index("ix_calls_call_id", table_name="calls")
    op.drop_table("calls")

    op.drop_index("ix_admin_users_email", table_name="admin_users")
    op.drop_table("admin_users")
