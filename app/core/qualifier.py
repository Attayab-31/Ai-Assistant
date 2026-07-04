"""Ready Rentals Online qualification scoring engine.

Scoring is driven entirely by admin-defined per-question rules. Each question
with scoring enabled contributes its ``max_points`` to the achievable total and
its earned points to the score; the result is normalized to 0–100. There is no
separate built-in weighting — the admin's question definitions are the single
source of truth, and deleting a question removes its score automatically.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _int_setting(settings: dict, key: str, default: int) -> int:
    try:
        return int(settings.get(key, default))
    except (TypeError, ValueError):
        return default


def _add_reason(reasons: list[str], reason: str) -> None:
    if reason and reason not in reasons:
        reasons.append(reason)


def calculate_qualification_score(
    tenant_data: dict,
    settings: dict,
    *,
    questions: list[dict] | None = None,
) -> tuple[int, str, list[str]]:
    """Score an applicant from the admin-defined per-question scoring rules.

    The score is the earned points as a percentage of the points achievable for
    THIS question set, so it always reflects the admin's chosen criteria and is
    never capped by questions the admin chose not to score or chose to delete.

    - A question's ``scoring`` config is the single source of truth.
    - A question flagged to disqualify (e.g. "disqualify on yes") returns 0 /
      unqualified immediately.
    - When no question has scoring enabled, applicants can't be scored, so the
      call routes to manual review rather than auto-qualifying or zeroing out.
    """
    from app.core.question_scoring import score_custom_questions

    reasons: list[str] = []
    qualified_threshold = _int_setting(settings, "qualified_score_threshold", 75)
    review_threshold = _int_setting(settings, "review_score_threshold", 40)

    earned, q_reasons, breakdown, disqualified = score_custom_questions(
        questions, tenant_data
    )
    reasons.extend(q_reasons)
    if breakdown:
        tenant_data["custom_question_scoring"] = breakdown
    if disqualified:
        logger.info("Auto-disqualified by a per-question rule")
        return 0, "unqualified", reasons or ["Disqualifying answer provided"]

    possible = sum(int(b.get("max_points") or 0) for b in breakdown)

    # Caller-control signals always warrant a human look, regardless of score.
    force_review = False
    if tenant_data.get("general_notes") and re_search_credit_issue(
        tenant_data.get("general_notes")
    ):
        _add_reason(
            reasons,
            "Credit/background note disclosed - full picture should be reviewed",
        )
        force_review = True
    if tenant_data.get("human_requested"):
        _add_reason(reasons, "Caller requested a leasing specialist follow-up")
        force_review = True
    if tenant_data.get("callback_requested"):
        _add_reason(reasons, "Caller requested a call back later")
        force_review = True
    if tenant_data.get("stop_requested"):
        _add_reason(reasons, "Caller asked to stop before completing screening")
        force_review = True

    if possible <= 0:
        # No per-question scoring configured — we can't qualify automatically.
        _add_reason(
            reasons,
            "No qualification scoring is configured - manual review recommended",
        )
        return 0, "review", reasons

    score = max(0, min(100, int(round(earned / possible * 100))))

    if score >= qualified_threshold and not force_review:
        status = "qualified"
    elif score >= review_threshold or force_review:
        status = "review"
    else:
        status = "unqualified"
        _add_reason(reasons, f"Insufficient qualification score ({score}/100)")

    logger.info(
        "Qualification: %s/%s normalized=%s, status=%s",
        round(earned, 1),
        round(possible, 1),
        score,
        status,
    )
    return score, status, reasons


def build_tenant_scoring_data(tenant: Any, questions_answered: int = 0) -> dict:
    """Reconstruct the full scoring input for a stored tenant.

    This MUST mirror the data the live finalize scored, so any recomputed score
    matches the one persisted at the end of the call. It includes the standard
    column fields, the per-call state lists (answered/refused), and any dynamic
    custom-question fields stored in ``normalized_data['custom_fields']``.

    Using anything less (e.g. a partial stub) makes the admin breakdown disagree
    with the headline score — the exact bug this replaces.
    """
    data: dict[str, Any] = {
        "monthly_income": float(tenant.monthly_income)
        if tenant.monthly_income is not None
        else None,
        "income_raw": tenant.income_raw,
        "has_eviction": tenant.has_eviction,
        "eviction_circumstances": tenant.eviction_circumstances,
        "eviction_raw": tenant.eviction_raw,
        "current_residence": tenant.current_residence,
        "residence_duration": tenant.residence_duration,
        "move_reason": tenant.move_reason,
        "move_in_date": tenant.move_in_date,
        "move_in_raw": tenant.move_in_raw,
        "move_timing": tenant.move_timing,
        "occupants_count": tenant.occupants_count,
        "adults_count": tenant.adults_count,
        "children_count": tenant.children_count,
        "has_pets": tenant.has_pets,
        "pet_type": tenant.pet_type,
        "pets_raw": tenant.pets_raw,
        "pet_weight": tenant.pet_weight,
        "general_notes": tenant.general_notes,
        "questions_answered": questions_answered,
        "answered_states": list(tenant.answered_states or []),
        "refused_states": list(tenant.refused_states or []),
    }
    # Dynamic (non-column) custom question fields live in normalized_data. Don't
    # clobber a real column if a custom field happens to share its key.
    normalized = getattr(tenant, "normalized_data", None)
    if isinstance(normalized, dict):
        custom = normalized.get("custom_fields")
        if isinstance(custom, dict):
            for key, value in custom.items():
                data.setdefault(key, value)
    return data


def re_search_credit_issue(text: Any) -> bool:
    import re

    return bool(
        re.search(
            r"\b(credit|background|criminal|bankrupt|collection|judgment|judgement)\b",
            str(text or ""),
            re.I,
        )
    )


def get_score_breakdown(
    tenant_data: dict,
    settings: dict,
    *,
    questions: list[dict] | None = None,
) -> dict:
    """Return admin-friendly scoring context built from per-question scoring."""
    score, status, reasons = calculate_qualification_score(
        tenant_data, settings, questions=questions
    )
    breakdown = tenant_data.get("custom_question_scoring") or []
    points_earned = sum(int(b.get("points") or 0) for b in breakdown)
    points_possible = sum(int(b.get("max_points") or 0) for b in breakdown)
    return {
        "score": score,
        "status": status,
        "reasons": reasons,
        "points_earned": points_earned,
        "points_possible": points_possible,
        "questions": breakdown,
        "completion": {
            "questions_answered": tenant_data.get("questions_answered"),
            "refused_states": tenant_data.get("refused_states") or [],
        },
    }
