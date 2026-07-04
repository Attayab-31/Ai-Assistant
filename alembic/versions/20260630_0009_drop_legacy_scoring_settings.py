"""Prune obsolete legacy-bucket scoring settings.

Revision ID: 20260630_0009_prune_scoring
Revises: 20260629_0008_call_latency
Create Date: 2026-06-30

Scoring is now driven entirely by per-question definitions (each question owns
its own ``max_points`` and rules). The old fixed "scoring weight" buckets and
the income-policy knobs they fed are gone, so the matching rows in
``system_settings`` are dead data. This deletes them so the settings store has
no stale keys. The two status thresholds (qualified/review cutoffs) are kept —
the per-question score still needs cutoffs to decide the final status.

Idempotent: deleting absent keys is a no-op, and downgrade re-seeds the legacy
defaults so the migration is reversible.
"""

import uuid
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision: str = "20260630_0009_prune_scoring"
down_revision: Union[str, None] = "20260629_0008_call_latency"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_OBSOLETE_KEYS = [
    "score_weight_income",
    "score_weight_eviction",
    "score_weight_completion",
    "score_weight_move_date",
    "score_weight_rental_history",
    "score_weight_household_fit",
    "monthly_rent_for_income_ratio",
    "income_multiplier",
    "min_income_threshold",
    "disqualify_on_eviction",
]

# (key, value, value_type, description) used only to restore on downgrade.
_LEGACY_DEFAULTS = [
    ("score_weight_income", "35", "integer", "Score weight: income context (0-100)"),
    ("score_weight_eviction", "15", "integer", "Score weight: eviction context (0-100)"),
    ("score_weight_completion", "25", "integer", "Score weight: screening completion (0-100)"),
    ("score_weight_move_date", "10", "integer", "Score weight: move-in timing (0-100)"),
    ("score_weight_rental_history", "10", "integer", "Score weight: rental history context (0-100)"),
    ("score_weight_household_fit", "5", "integer", "Score weight: occupants and pet details (0-100)"),
    ("monthly_rent_for_income_ratio", "0", "integer", "Optional monthly rent used for income scoring; 0 means review income context"),
    ("income_multiplier", "3.0", "string", "Required income as a multiple of monthly rent (e.g. 3.0 = 3x rent)"),
    ("min_income_threshold", "0", "integer", "Optional absolute monthly income floor ($); 0 uses 3x rent policy"),
    ("disqualify_on_eviction", "false", "boolean", "Auto-disqualify if eviction disclosed (normally false; reviewed individually)"),
]


def upgrade() -> None:
    settings = sa.table(
        "system_settings",
        sa.column("key", sa.String),
    )
    op.execute(
        settings.delete().where(settings.c.key.in_(_OBSOLETE_KEYS))
    )


def downgrade() -> None:
    settings = sa.table(
        "system_settings",
        sa.column("id", UUID(as_uuid=True)),
        sa.column("key", sa.String),
        sa.column("value", sa.Text),
        sa.column("value_type", sa.String),
        sa.column("description", sa.Text),
        sa.column("is_sensitive", sa.Boolean),
    )
    bind = op.get_bind()
    existing = {
        row[0]
        for row in bind.execute(
            sa.select(settings.c.key).where(settings.c.key.in_(_OBSOLETE_KEYS))
        )
    }
    rows = [
        {
            "id": uuid.uuid4(),
            "key": key,
            "value": value,
            "value_type": value_type,
            "description": description,
            "is_sensitive": False,
        }
        for key, value, value_type, description in _LEGACY_DEFAULTS
        if key not in existing
    ]
    if rows:
        op.bulk_insert(settings, rows)
