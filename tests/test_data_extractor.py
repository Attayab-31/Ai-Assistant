"""Tests for extraction safety nets: move-in year and pet weight units."""

from datetime import date

from app.core.data_extractor import (
    _normalize_pet_weight_lbs,
    _roll_future_date,
    coerce_extracted_data,
)


def test_roll_future_date_fixes_past_year():
    today = date(2026, 6, 30)
    # LLM emitted a past year for an upcoming move-in ("July 26").
    assert _roll_future_date(date(2024, 7, 26), today) == date(2026, 7, 26)


def test_roll_future_date_advances_to_next_year_when_month_passed():
    today = date(2026, 6, 30)
    # A month/day already passed this year rolls to next year.
    assert _roll_future_date(date(2024, 3, 5), today) == date(2027, 3, 5)


def test_roll_future_date_leaves_future_untouched():
    today = date(2026, 6, 30)
    future = date(2026, 8, 1)
    assert _roll_future_date(future, today) == future
    assert _roll_future_date(None, today) is None


def test_pet_weight_converts_kg_adjacent_to_number():
    # "2 kg" -> ~4 lbs (2 * 2.20462 rounded).
    assert _normalize_pet_weight_lbs(2, "2 kg", "small dog") == 4


def test_pet_weight_no_double_conversion_when_already_pounds():
    # Model already converted to 4 lbs; raw mentions kg but not "4 kg" — leave it.
    assert _normalize_pet_weight_lbs(4, "weighs about two kg", "dog") == 4


def test_pet_weight_plain_pounds_unchanged():
    assert _normalize_pet_weight_lbs(30, "30 pound lab", "dog") == 30
    assert _normalize_pet_weight_lbs(None) is None


def test_coerce_applies_extraction_safety_nets():
    out = coerce_extracted_data(
        {
            "move_in_date": "2024-07-26",
            "pet_weight": "2 kg",
            "pets_raw": "German Shepherd, two kg",
        }
    )
    # Past year rolled forward (assuming this test runs in 2026+).
    assert out["move_in_date"].year >= date.today().year
    assert out["move_in_date"].month == 7
    assert out["move_in_date"].day == 26
    # "2 kg" recognised next to the digit and converted to pounds.
    assert out["pet_weight"] == 4
