"""Extract and normalize Ready Rentals screening data from transcripts."""

from __future__ import annotations

import json
import logging
import re
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from app.core.screening_flow import (
    extract_fields_from_text,
    normalize_email,
    normalize_money,
    normalize_phone,
    parse_relative_date,
)

logger = logging.getLogger(__name__)


EXTRACTION_PROMPT = """You extract structured data from a tenant screening phone call for Ready Rentals Online.
Return ONLY a valid JSON object. Use null when a field is not mentioned.

TRANSCRIPT:
{transcript}

Extract these fields:
- full_name
- contact_phone
- email
- move_in_date: ISO date YYYY-MM-DD when clear
- move_in_raw: caller's exact move-in wording
- occupants_count: total occupants
- adults_count
- children_count
- has_pets
- pets_raw
- pet_type
- pet_breed
- pet_weight: number in POUNDS; convert if the caller used kg/grams/ounces (1 kg = 2.2 lbs)
- current_residence
- residence_duration
- move_reason
- move_timing
- has_eviction
- eviction_raw
- eviction_circumstances
- monthly_income: monthly household income before taxes in USD
- income_raw
- employer
- employment_duration
- general_notes
- special_notes
- human_requested
- callback_requested
- stop_requested

Rules:
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


def coerce_extracted_data(raw: dict[str, Any]) -> dict[str, Any]:
    """Type-coerce and validate extracted fields."""
    raw = raw or {}
    result: dict[str, Any] = {}

    result["full_name"] = _clean_text(raw.get("full_name"), 255)

    phone = raw.get("contact_phone") or raw.get("phone_number")
    result["contact_phone"] = normalize_phone(str(phone)) if phone else None

    email_raw = _clean_text(raw.get("email"), 255)
    result["email"] = normalize_email(str(email_raw)) if email_raw else None

    result["move_in_date"] = _parse_date(raw.get("move_in_date"))
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
    result["pet_weight"] = _parse_int(raw.get("pet_weight"))

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


async def extract_tenant_data(transcript: str, llm_provider) -> dict[str, Any]:
    """Extract structured tenant data using the active LLM, with safe fallback."""
    today = date.today().isoformat()
    prompt = EXTRACTION_PROMPT.format(transcript=transcript, today=today)

    try:
        raw_response = await llm_provider.get_response(
            system_prompt="You extract structured data from transcripts. Return only valid JSON.",
            messages=[{"role": "user", "content": prompt}],
            json_mode=True,
            temperature=0.1,
            max_tokens=1200,
        )
        clean = re.sub(r"```json\s*|\s*```", "", raw_response).strip()
        extracted = json.loads(clean)
        result = coerce_extracted_data(extracted)
        logger.info("Data extraction successful: %s", sorted(result.keys()))
        return result
    except json.JSONDecodeError as e:
        logger.error("Failed to parse extraction JSON: %s", e)
    except Exception as e:
        logger.error("Data extraction failed: %s", e)

    return extract_from_transcript_heuristic(transcript)


def extract_from_transcript_heuristic(transcript: str) -> dict[str, Any]:
    """Fallback extraction when the LLM is unavailable or returns bad JSON."""
    extracted: dict[str, Any] = {}
    state = ""
    for line in (transcript or "").splitlines():
        if "] AI:" in line:
            continue
        if "] Tenant:" not in line:
            continue
        utterance = line.split("] Tenant:", 1)[-1].strip()
        # Try each state until the utterance yields useful data.
        for state in (
            "Q1_FULL_NAME",
            "Q2_PHONE",
            "Q3_EMAIL",
            "Q4_MOVE_IN_DATE",
            "Q5_OCCUPANTS",
            "Q6_PETS",
            "Q6A_PET_DETAILS",
            "Q7_CURRENT_RESIDENCE",
            "Q8_RESIDENCE_DURATION",
            "Q9_MOVE_REASON",
            "Q10_MOVE_TIMING",
            "Q11_EVICTION",
            "Q11A_EVICTION_DETAILS",
            "Q12_INCOME",
            "Q13_EMPLOYER",
            "Q14_EMPLOYMENT_DURATION",
            "Q15_GENERAL_NOTES",
        ):
            fields = extract_fields_from_text(utterance, state, extracted)
            if fields:
                extracted.update(fields)
                break
    return coerce_extracted_data(extracted)
