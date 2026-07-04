"""Extract and normalize Ready Rentals screening data from transcripts."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from app.core.question_flow import (
    active_extract_fields,
    extract_fields_from_speech,
    field_labels_from_questions,
    normalize_questions,
)
from app.core.screening_flow import (
    normalize_email,
    normalize_money,
    normalize_phone,
    parse_relative_date,
)

logger = logging.getLogger(__name__)

_CONTROL_FIELDS = (
    "human_requested",
    "callback_requested",
    "stop_requested",
    "special_notes",
)

_EXTRACTION_RULES = """Rules:
- Preserve raw caller wording in *_raw fields.
- For income, the question asks for MONTHLY income. Respect the period the caller states:
  - If they say "monthly", "a month", "per month", "/mo", or give no period at all, use the number AS-IS as monthly_income. Do NOT divide it, even if it seems large.
  - Only divide by 12 when the caller clearly says it is yearly ("a year", "yearly", "annually", "per year", "annual salary").
  - For hourly pay, convert only if the hours are stated; otherwise leave monthly_income null and preserve income_raw.
  - Always preserve the caller's exact wording in income_raw.
- Eviction means an eviction or landlord-tenant court filing. If unclear, leave has_eviction null.
- If the caller says bad credit or eviction should be reviewed, put that context in general_notes or special_notes.
- Return JSON only, no markdown.

Today's date: {today}"""


def build_extraction_prompt(
    transcript: str,
    questions: list[dict[str, Any]] | None = None,
    *,
    today: str | None = None,
) -> str:
    """Build an end-of-call extraction prompt from the admin question list."""
    today = today or date.today().isoformat()
    normalized = normalize_questions(questions)
    labels = field_labels_from_questions(normalized)
    fields = sorted(active_extract_fields(normalized))
    for control in _CONTROL_FIELDS:
        if control not in fields:
            fields.append(control)

    field_lines = []
    for field in fields:
        label = labels.get(field, field.replace("_", " "))
        field_lines.append(f"- {field}: {label}")

    return (
        "You extract structured data from a tenant screening phone call.\n"
        "Return ONLY a valid JSON object. Use null when a field is not mentioned.\n\n"
        f"TRANSCRIPT:\n{transcript}\n\n"
        "Extract these fields:\n"
        + "\n".join(field_lines)
        + "\n\n"
        + _EXTRACTION_RULES.format(today=today)
    )


def _clean_text(value: Any, max_len: int | None = None) -> str | None:
    if value in (None, ""):
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    if not text:
        return None
    return text[:max_len] if max_len else text


def _parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    norm = str(value).strip().lower()
    if norm in {"true", "yes", "y", "1", "yeah", "yep"}:
        return True
    if norm in {"false", "no", "n", "0", "nope", "none", "never"}:
        return False
    return None


def _parse_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        match = re.search(r"\d+", str(value))
        return int(match.group(0)) if match else None


def _parse_decimal(value: Any) -> Decimal | None:
    if value in (None, ""):
        return None
    try:
        return Decimal(str(value)).quantize(Decimal("0.01"))
    except (InvalidOperation, ValueError):
        return normalize_money(value)


def _parse_date(value: Any) -> date | None:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if not value:
        return None
    text = str(value)
    try:
        return datetime.strptime(text[:10], "%Y-%m-%d").date()
    except ValueError:
        parsed, _raw = parse_relative_date(text)
        return parsed


def _roll_future_date(d: date | None, today: date | None = None) -> date | None:
    """Roll a past move-in date forward to its next sensible future occurrence.

    Screening move-in dates are always upcoming, but small LLMs frequently emit
    a plausible-looking but PAST year (e.g. the caller says "July 26" and the
    model returns 2024-07-26 when today is 2026). A past date here is almost
    always a wrong year, and it actively corrupts scoring (the qualifier treats
    a past move-in date as a negative signal). When the date lands in the past,
    keep the same month/day and advance to this year, then next year. Dates that
    are already today-or-future are returned untouched.

    This is keyed off the *value*, not any specific question — if an admin
    removes the move-in question there's simply no date to adjust.
    """
    if d is None:
        return None
    today = today or date.today()
    if d >= today:
        return d
    for year in (today.year, today.year + 1):
        try:
            candidate = d.replace(year=year)
        except ValueError:
            # Feb 29 on a non-leap year — fall back to Feb 28.
            candidate = d.replace(year=year, month=2, day=28)
        if candidate >= today:
            return candidate
    return d


def _normalize_pet_weight_lbs(weight: int | None, *context: Any) -> int | None:
    """Pet weight is stored in POUNDS. Small models often skip the kg/oz->lbs
    conversion the prompt asks for, so convert deterministically as a safety net.

    To avoid double-converting a value the model *did* convert, only act when the
    metric unit is written directly next to THIS number in the caller's raw
    wording (e.g. "2 kg"). A spelled-out unit not adjacent to the same digit is
    left to the prompt, so an already-correct pound value is never inflated.
    """
    if weight is None or weight <= 0:
        return weight
    blob = " ".join(str(c) for c in context if c).lower()
    if re.search(rf"\b{weight}\s*(kg|kgs|kilo|kilos|kilogram|kilograms)\b", blob):
        return max(1, round(weight * 2.20462))
    if re.search(rf"\b{weight}\s*(gram|grams)\b", blob):
        return max(1, round(weight / 453.592))
    if re.search(rf"\b{weight}\s*(oz|ounce|ounces)\b", blob):
        return max(1, round(weight / 16))
    return weight


def coerce_extracted_data(raw: dict[str, Any]) -> dict[str, Any]:
    """Type-coerce and validate extracted fields."""
    raw = raw or {}
    result: dict[str, Any] = {}

    result["full_name"] = _clean_text(raw.get("full_name"), 255)

    phone = raw.get("contact_phone") or raw.get("phone_number")
    result["contact_phone"] = normalize_phone(str(phone)) if phone else None

    email_raw = _clean_text(raw.get("email"), 255)
    result["email"] = normalize_email(str(email_raw)) if email_raw else None

    result["move_in_date"] = _roll_future_date(_parse_date(raw.get("move_in_date")))
    result["move_in_raw"] = _clean_text(raw.get("move_in_raw"), 255)

    result["occupants_count"] = _parse_int(raw.get("occupants_count"))
    result["adults_count"] = _parse_int(raw.get("adults_count"))
    result["children_count"] = _parse_int(raw.get("children_count"))
    if result["children_count"] is None:
        result["children_count"] = 0
    if result["occupants_count"] is None and result["adults_count"] is not None:
        result["occupants_count"] = result["adults_count"] + (
            result["children_count"] or 0
        )

    result["has_pets"] = _parse_bool(raw.get("has_pets"))
    result["pets_raw"] = _clean_text(raw.get("pets_raw"), 1000)
    result["pet_type"] = _clean_text(raw.get("pet_type"), 100)
    result["pet_breed"] = _clean_text(raw.get("pet_breed"), 100)
    result["pet_weight"] = _normalize_pet_weight_lbs(
        _parse_int(raw.get("pet_weight")),
        raw.get("pet_weight"),
        raw.get("pets_raw"),
    )

    result["current_residence"] = _clean_text(raw.get("current_residence"), 500)
    result["residence_duration"] = _clean_text(raw.get("residence_duration"), 255)
    result["move_reason"] = _clean_text(raw.get("move_reason"), 1000)
    result["move_timing"] = _clean_text(raw.get("move_timing"), 255)

    result["has_eviction"] = _parse_bool(raw.get("has_eviction"))
    result["eviction_raw"] = _clean_text(raw.get("eviction_raw"), 1000)
    result["eviction_circumstances"] = _clean_text(
        raw.get("eviction_circumstances"), 2000
    )

    result["monthly_income"] = _parse_decimal(raw.get("monthly_income"))
    result["income_raw"] = _clean_text(raw.get("income_raw"), 255)

    result["employer"] = _clean_text(raw.get("employer"), 255)
    result["employment_duration"] = _clean_text(raw.get("employment_duration"), 255)
    result["general_notes"] = _clean_text(raw.get("general_notes"), 2000)
    result["special_notes"] = _clean_text(raw.get("special_notes"), 2000)

    for flag in ("human_requested", "callback_requested", "stop_requested"):
        result[flag] = _parse_bool(raw.get(flag))

    for key, value in raw.items():
        if str(key).startswith("custom_") and key not in result and value not in (None, ""):
            result[key] = value

    for json_field in (
        "raw_answers",
        "normalized_data",
        "answered_states",
        "refused_states",
        "faq_topics",
        "control_flags",
    ):
        value = raw.get(json_field)
        if value not in (None, ""):
            result[json_field] = value

    return result


async def extract_tenant_data(
    transcript: str,
    llm_provider,
    questions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Extract structured tenant data using the active LLM, with safe fallback."""
    today = date.today().isoformat()
    prompt = build_extraction_prompt(transcript, questions, today=today)

    try:
        raw_response = await asyncio.wait_for(
            llm_provider.get_response(
                system_prompt="You extract structured data from transcripts. Return only valid JSON.",
                messages=[{"role": "user", "content": prompt}],
                json_mode=True,
                temperature=0.1,
                max_tokens=1200,
            ),
            timeout=30.0,
        )
        clean = re.sub(r"```json\s*|\s*```", "", raw_response).strip()
        extracted = json.loads(clean)
        result = coerce_extracted_data(extracted)
        logger.info("Data extraction successful: %s", sorted(result.keys()))
        return result
    except TimeoutError:
        logger.error("Data extraction timed out")
    except json.JSONDecodeError as e:
        logger.error("Failed to parse extraction JSON: %s", e)
    except Exception as e:
        logger.error("Data extraction failed: %s", e)

    return extract_from_transcript_heuristic(transcript, questions)


def extract_from_transcript_heuristic(
    transcript: str,
    questions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Fallback extraction when the LLM is unavailable or returns bad JSON."""
    extracted: dict[str, Any] = {}
    normalized = normalize_questions(questions)
    for line in (transcript or "").splitlines():
        if "] AI:" in line:
            continue
        if "] Tenant:" not in line:
            continue
        utterance = line.split("] Tenant:", 1)[-1].strip()
        for question in normalized:
            if not question.get("active", True):
                continue
            fields = extract_fields_from_speech(utterance, question, extracted)
            if fields:
                extracted.update(fields)
                break
    return coerce_extracted_data(extracted)
