"""Ready Rentals Online qualification scoring engine."""

from __future__ import annotations

import logging
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from app.core.screening_flow import count_active_questions, count_answered_questions

logger = logging.getLogger(__name__)


def _decimal(value: Any) -> Decimal | None:
    if value in (None, ""):
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _int_setting(settings: dict, key: str, default: int) -> int:
    try:
        return int(settings.get(key, default))
    except (TypeError, ValueError):
        return default


def _bool_setting(settings: dict, key: str, default: bool = False) -> bool:
    value = settings.get(key, default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _decimal_setting(settings: dict, key: str, default: Decimal) -> Decimal:
    try:
        value = Decimal(str(settings.get(key, default)))
        return value if value > 0 else default
    except (InvalidOperation, ValueError, TypeError):
        return default


def _fmt_multiplier(multiplier: Decimal) -> str:
    """Render a multiplier for written reasons, e.g. 3.0 -> '3x', 2.5 -> '2.5x'."""
    s = format(multiplier, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return f"{s}x"


def _parse_date(value: Any) -> date | None:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if not value:
        return None
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _add_reason(reasons: list[str], reason: str) -> None:
    if reason and reason not in reasons:
        reasons.append(reason)


def calculate_qualification_score(
    tenant_data: dict,
    settings: dict,
) -> tuple[int, str, list[str]]:
    """
    Calculate qualified/review/unqualified with explainable policy reasons.

    Ready Rentals policy:
    - Household income standard is 3x monthly rent when rent is configured.
    - Eviction history is reviewed individually, not automatically denied unless
      explicit config enables auto-disqualification.
    - Credit/background concerns belong in review; credit alone is not an
      automatic disqualifier.
    """
    reasons: list[str] = []
    score = 0

    weights = {
        "income": _int_setting(settings, "score_weight_income", 35),
        "completion": _int_setting(settings, "score_weight_completion", 25),
        "eviction": _int_setting(settings, "score_weight_eviction", 15),
        "rental_history": _int_setting(settings, "score_weight_rental_history", 10),
        "move_date": _int_setting(settings, "score_weight_move_date", 10),
        "household_fit": _int_setting(settings, "score_weight_household_fit", 5),
    }

    disqualify_on_eviction = _bool_setting(
        settings,
        "disqualify_on_eviction",
        False,
    )
    monthly_rent = _decimal(settings.get("monthly_rent_for_income_ratio"))
    min_income = _decimal(settings.get("min_income_threshold"))
    if min_income == Decimal("0"):
        min_income = None

    # Admin-configurable scoring policy (income multiple + status cutoffs).
    income_multiplier = _decimal_setting(settings, "income_multiplier", Decimal("3"))
    multiplier_label = _fmt_multiplier(income_multiplier)
    qualified_threshold = _int_setting(settings, "qualified_score_threshold", 75)
    review_threshold = _int_setting(settings, "review_score_threshold", 40)

    # Completion
    refused_states = tenant_data.get("refused_states") or []
    answered = int(
        tenant_data.get("questions_answered")
        or count_answered_questions(tenant_data, refused_states)
    )
    active_total = max(count_active_questions(tenant_data), 1)
    completion_ratio = min(1.0, answered / active_total)
    score += int(weights["completion"] * completion_ratio)
    if completion_ratio < 1:
        _add_reason(
            reasons,
            f"Screening incomplete - {answered}/{active_total} required items captured",
        )

    # Income
    monthly_income = _decimal(tenant_data.get("monthly_income"))
    if monthly_income is None:
        if tenant_data.get("income_raw"):
            score += int(weights["income"] * 0.35)
            _add_reason(
                reasons,
                "Income needs review - caller gave income in a format that needs verification",
            )
        else:
            _add_reason(reasons, "Income not disclosed")
    else:
        required_income = None
        if monthly_rent and monthly_rent > 0:
            required_income = monthly_rent * income_multiplier
        elif min_income and min_income > 0:
            required_income = min_income

        if required_income:
            ratio = monthly_income / required_income
            if ratio >= 1:
                score += weights["income"]
            elif ratio >= Decimal("0.8"):
                score += int(weights["income"] * 0.65)
                _add_reason(
                    reasons,
                    f"Income is close to the standard {multiplier_label} monthly rent "
                    "requirement and should be reviewed",
                )
            else:
                score += int(weights["income"] * 0.25)
                _add_reason(
                    reasons,
                    f"Income appears below the standard {multiplier_label} monthly "
                    "rent requirement",
                )
        else:
            score += int(weights["income"] * 0.75)
            _add_reason(
                reasons,
                f"Income captured; compare against the property's {multiplier_label} "
                "rent standard during review",
            )

    # Eviction / court filing
    has_eviction = tenant_data.get("has_eviction")
    if has_eviction is True:
        if disqualify_on_eviction:
            _add_reason(
                reasons,
                "Eviction or landlord-tenant court filing disclosed and auto-disqualification is enabled",
            )
            logger.info("Auto-disqualified by eviction config")
            return 0, "unqualified", reasons
        score += int(weights["eviction"] * 0.35)
        _add_reason(
            reasons,
            "Eviction or landlord-tenant court filing disclosed - reviewed individually",
        )
        if not tenant_data.get("eviction_circumstances") and not tenant_data.get(
            "eviction_raw"
        ):
            _add_reason(reasons, "Eviction circumstances need follow-up")
    elif has_eviction is False:
        score += weights["eviction"]
    else:
        score += int(weights["eviction"] * 0.4)
        _add_reason(reasons, "Eviction/court filing answer not confirmed")

    # Rental history
    rental_fields = [
        tenant_data.get("current_residence"),
        tenant_data.get("residence_duration"),
        tenant_data.get("move_reason"),
    ]
    rental_count = sum(1 for value in rental_fields if value)
    score += int(weights["rental_history"] * (rental_count / len(rental_fields)))
    if rental_count < len(rental_fields):
        _add_reason(reasons, "Rental history needs follow-up")

    # Move timing
    move_date = _parse_date(tenant_data.get("move_in_date"))
    if move_date:
        days_until = (move_date - date.today()).days
        if days_until < 0:
            score += int(weights["move_date"] * 0.25)
            _add_reason(reasons, "Move-in date appears to be in the past")
        elif days_until <= 120:
            score += weights["move_date"]
        else:
            score += int(weights["move_date"] * 0.5)
            _add_reason(
                reasons, "Move-in timing is farther out and may depend on availability"
            )
    elif tenant_data.get("move_in_raw") or tenant_data.get("move_timing"):
        score += int(weights["move_date"] * 0.6)
        _add_reason(reasons, "Move-in timing captured but needs date confirmation")
    else:
        _add_reason(reasons, "Move-in timing not disclosed")

    # Household / pet fit
    occupants = tenant_data.get("occupants_count") or tenant_data.get("adults_count")
    has_pets = tenant_data.get("has_pets")
    pet_detail_ok = has_pets is False or (
        has_pets is True
        and (tenant_data.get("pet_type") or tenant_data.get("pets_raw"))
        and (tenant_data.get("pet_weight") or tenant_data.get("pets_raw"))
    )
    if occupants and pet_detail_ok:
        score += weights["household_fit"]
    elif occupants:
        score += int(weights["household_fit"] * 0.6)
        _add_reason(reasons, "Pet details need follow-up for property matching")
    else:
        _add_reason(reasons, "Occupant count not disclosed")

    if tenant_data.get("general_notes") and re_search_credit_issue(
        tenant_data.get("general_notes")
    ):
        _add_reason(
            reasons,
            "Credit/background note disclosed - full picture should be reviewed",
        )

    if tenant_data.get("human_requested"):
        _add_reason(reasons, "Caller requested a leasing specialist follow-up")
    if tenant_data.get("callback_requested"):
        _add_reason(reasons, "Caller requested a call back later")
    if tenant_data.get("stop_requested"):
        _add_reason(reasons, "Caller asked to stop before completing screening")

    score = max(0, min(100, score))

    review_only_reasons = [
        reason
        for reason in reasons
        if any(
            phrase in reason.lower()
            for phrase in (
                "review",
                "follow-up",
                "not confirmed",
                "not disclosed",
                "incomplete",
                "needs",
            )
        )
    ]

    if score >= qualified_threshold and not review_only_reasons:
        status = "qualified"
    elif score >= review_threshold or review_only_reasons:
        status = "review"
    else:
        status = "unqualified"
        _add_reason(reasons, f"Insufficient qualification score ({score}/100)")

    logger.info(
        "Qualification: score=%s, status=%s, reasons=%s", score, status, reasons
    )
    return score, status, reasons


def re_search_credit_issue(text: Any) -> bool:
    import re

    return bool(
        re.search(
            r"\b(credit|background|criminal|bankrupt|collection|judgment|judgement)\b",
            str(text or ""),
            re.I,
        )
    )


def get_score_breakdown(tenant_data: dict, settings: dict) -> dict:
    """Return admin-friendly scoring context."""
    score, status, reasons = calculate_qualification_score(tenant_data, settings)
    monthly_rent = _decimal(settings.get("monthly_rent_for_income_ratio"))
    income_multiplier = _decimal_setting(settings, "income_multiplier", Decimal("3"))
    multiplier_label = _fmt_multiplier(income_multiplier)
    required_income = (
        monthly_rent * income_multiplier
        if monthly_rent and monthly_rent > 0
        else None
    )
    return {
        "score": score,
        "status": status,
        "reasons": reasons,
        "income": {
            "monthly_income": str(tenant_data.get("monthly_income") or ""),
            "standard": f"{multiplier_label} monthly rent",
            "required_income": str(required_income) if required_income else None,
        },
        "eviction": {
            "has_eviction": tenant_data.get("has_eviction"),
            "policy": "Reviewed individually unless auto-disqualification is enabled.",
        },
        "completion": {
            "questions_answered": tenant_data.get("questions_answered"),
            "refused_states": tenant_data.get("refused_states") or [],
        },
        "household": {
            "occupants_count": tenant_data.get("occupants_count"),
            "has_pets": tenant_data.get("has_pets"),
            "pets_raw": tenant_data.get("pets_raw"),
        },
    }
