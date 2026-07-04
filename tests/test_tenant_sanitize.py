"""Tests for last-mile tenant payload sanitization before DB insert."""

from datetime import date
from decimal import Decimal

import pytest

from app.core.tenant_sanitize import sanitize_tenant_payload


def test_bool_in_text_column_moves_to_overflow():
    overflow: dict = {}
    out = sanitize_tenant_payload(
        {"eviction_raw": False, "has_eviction": False},
        overflow=overflow,
    )
    assert out["eviction_raw"] is None
    assert out["has_eviction"] is False
    assert overflow["eviction_raw"] is False


def test_string_text_fields_coerce_numbers():
    out = sanitize_tenant_payload(
        {
            "employer": "DevWorks",
            "income_raw": "50,000 USD per year",
            "occupants_count": "4",
        }
    )
    assert out["employer"] == "DevWorks"
    assert out["income_raw"] == "50,000 USD per year"
    assert out["occupants_count"] == 4


def test_date_and_numeric_coercion():
    out = sanitize_tenant_payload(
        {
            "move_in_date": "2026-07-24",
            "monthly_income": "4000.00",
        }
    )
    assert out["move_in_date"] == date(2026, 7, 24)
    assert out["monthly_income"] == Decimal("4000.00")


def test_jsonb_columns_preserved():
    payload = {
        "normalized_data": {"custom_fields": {"custom_x": "yes"}},
        "answered_states": ["Q1_FULL_NAME"],
        "refused_states": [],
    }
    out = sanitize_tenant_payload(payload)
    assert out["normalized_data"]["custom_fields"]["custom_x"] == "yes"
    assert out["answered_states"] == ["Q1_FULL_NAME"]


def test_invalid_jsonb_value_overflows():
    overflow: dict = {}
    out = sanitize_tenant_payload(
        {"disqualify_reasons": "not a list"},
        overflow=overflow,
    )
    assert out["disqualify_reasons"] is None
    assert overflow["disqualify_reasons"] == "not a list"


def test_long_string_truncated_to_column_limit():
    out = sanitize_tenant_payload({"full_name": "x" * 300})
    assert out["full_name"] is not None
    assert len(out["full_name"]) == 255


def test_dawn_smith_style_payload_survives_sanitize():
    """Regression: bool eviction_raw must not block tenant insert."""
    overflow: dict = {}
    out = sanitize_tenant_payload(
        {
            "full_name": "Dawn Smith",
            "contact_phone": "+13174026038",
            "email": "attayabpc@gmail.com",
            "occupants_count": 4,
            "monthly_income": Decimal("4000.00"),
            "has_pets": True,
            "pets_raw": "dog",
            "has_eviction": False,
            "eviction_raw": False,
            "move_in_date": date(2026, 7, 24),
            "move_in_raw": "2026-07-24",
        },
        overflow=overflow,
    )
    assert out["full_name"] == "Dawn Smith"
    assert out["has_eviction"] is False
    assert out["eviction_raw"] is None
    assert overflow["eviction_raw"] is False
    assert out["monthly_income"] == Decimal("4000.00")
