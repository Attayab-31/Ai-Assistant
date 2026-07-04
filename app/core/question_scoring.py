"""Per-question scoring rules for dynamic screening questions."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any


def _decimal(value: Any) -> Decimal | None:
    if value in (None, ""):
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _safe_int(value: Any, default: int = 0) -> int:
    """Coerce an admin-supplied config value to int, never raising.

    pass_config values are typed ``Any`` in the schema, so a malformed entry
    (e.g. a non-numeric string) must not crash scoring — which runs inside
    finalize and would otherwise lose the whole screening result.
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        try:
            return int(float(str(value).strip()))
        except (TypeError, ValueError):
            return default


def _parse_date(value: Any) -> date | None:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if not value:
        return None
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def evaluate_question_scoring(
    question: dict[str, Any],
    tenant_data: dict[str, Any],
) -> tuple[int, list[str], bool]:
    """Return (points_earned, reasons, auto_disqualify)."""
    scoring = question.get("scoring") or {}
    if not scoring.get("enabled"):
        return 0, [], False

    # NOTE: do NOT early-return when max_points == 0. A question can be enabled
    # purely as a disqualifying gate (0 points, "disqualify on yes"), and that
    # gate must still be evaluated.
    max_points = _safe_int(scoring.get("max_points"), 0)

    rule_type = str(scoring.get("rule_type") or "any_answer")
    config = dict(scoring.get("pass_config") or {})
    fields = list(question.get("extract_fields") or [])
    primary = str(fields[0]) if fields else ""
    value = tenant_data.get(primary)
    label = (question.get("field_labels") or {}).get(primary, primary)
    q_label = question.get("question") or question.get("id") or "Question"
    reasons: list[str] = []

    if rule_type == "any_answer":
        if value not in (None, ""):
            return max_points, [], False
        reasons.append(f"{q_label}: no answer captured")
        return 0, reasons, False

    if rule_type == "required_field":
        if value not in (None, ""):
            return max_points, [], False
        reasons.append(f"{q_label}: required field missing")
        return 0, reasons, False

    if rule_type == "yes_no":
        yes_pts = _safe_int(config.get("yes", max_points), max_points)
        no_pts = _safe_int(config.get("no", 0), 0)
        # Disqualify flags come straight from the admin's pass_config — they are
        # the only disqualification mechanism, so honor them directly.
        if value is True:
            if config.get("disqualify_on_yes"):
                return 0, [f"{q_label}: disqualifying answer (yes)"], True
            return yes_pts, [], False
        if value is False:
            if config.get("disqualify_on_no"):
                return 0, [f"{q_label}: disqualifying answer (no)"], True
            return no_pts, [], False
        reasons.append(f"{q_label}: yes/no not captured")
        return 0, reasons, False

    if rule_type == "numeric_range":
        num = _decimal(value)
        minimum = _decimal(config.get("min"))
        maximum = _decimal(config.get("max"))
        if num is None:
            reasons.append(f"{q_label}: numeric value missing")
            return 0, reasons, False
        if minimum is not None and num < minimum:
            reasons.append(f"{q_label}: below minimum ({label})")
            return int(max_points * 0.25), reasons, False
        if maximum is not None and num > maximum:
            return max_points, [], False
        if minimum is not None and num >= minimum:
            return max_points, [], False
        return int(max_points * 0.5), [], False

    if rule_type == "date_within":
        parsed = _parse_date(value)
        if not parsed:
            raw = tenant_data.get(f"{primary}_raw") or tenant_data.get("move_in_raw")
            if raw:
                return int(max_points * 0.6), [], False
            reasons.append(f"{q_label}: date not captured")
            return 0, reasons, False
        days = _safe_int(config.get("max_days_ahead"), 90) or 90
        delta = (parsed - date.today()).days
        if delta < 0:
            reasons.append(f"{q_label}: move date is in the past")
            return int(max_points * 0.4), reasons, False
        if delta <= days:
            return max_points, [], False
        reasons.append(f"{q_label}: move date farther than {days} days out")
        return int(max_points * 0.5), reasons, False

    if value not in (None, ""):
        return max_points, [], False
    return 0, [f"{q_label}: not answered"], False


def score_custom_questions(
    questions: list[dict[str, Any]] | None,
    tenant_data: dict[str, Any],
) -> tuple[int, list[str], list[dict[str, Any]], bool]:
    """Sum enabled per-question scores. Returns total, reasons, breakdown, disqualified."""
    if not questions:
        return 0, [], [], False

    total = 0
    reasons: list[str] = []
    breakdown: list[dict[str, Any]] = []
    disqualified = False

    for q in questions:
        if not q.get("active", True):
            continue
        scoring = q.get("scoring") or {}
        if not scoring.get("enabled"):
            continue
        pts, q_reasons, auto_dq = evaluate_question_scoring(q, tenant_data)
        total += pts
        reasons.extend(q_reasons)
        breakdown.append(
            {
                "question_id": q.get("id"),
                "state": q.get("state"),
                "points": pts,
                "max_points": _safe_int(scoring.get("max_points"), 0),
            }
        )
        if auto_dq:
            disqualified = True

    return total, reasons, breakdown, disqualified
