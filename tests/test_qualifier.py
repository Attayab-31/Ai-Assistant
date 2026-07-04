"""Tests for per-question qualification scoring.

Scoring is driven entirely by the admin's per-question definitions. The score
is a percentage of the points ACHIEVABLE for the configured flow, so adding or
removing questions never silently caps the score below the qualified threshold
("0%/nobody qualifies" trap). Deleting a question removes its points too.
"""

from datetime import date, timedelta

from app.core.qualifier import calculate_qualification_score
from app.core.question_flow import default_questions_v2, new_custom_question


def _future(days: int = 30) -> date:
    return date.today() + timedelta(days=days)


def _strong_applicant() -> dict:
    """Answers every scored default question well → full marks."""
    return {
        "questions_answered": 100,
        "has_eviction": False,
        "current_residence": "New York",
        "residence_duration": "3 years",
        "move_reason": "closer to work",
        "move_in_date": _future(30),
        "occupants_count": 2,
        "has_pets": False,
        "monthly_income": 9000,
    }


def test_full_default_flow_strong_applicant_qualifies():
    qs = default_questions_v2()
    score, status, _ = calculate_qualification_score(
        _strong_applicant(), {}, questions=qs
    )
    assert score == 100
    assert status == "qualified"


def test_removing_income_question_still_allows_full_score():
    # Admin removes the income question. A strong applicant on the REMAINING
    # scored questions must still be able to reach 100 / qualified.
    qs = [q for q in default_questions_v2() if q.get("state") != "Q12_INCOME"]
    data = _strong_applicant()
    data.pop("monthly_income", None)
    score, status, _ = calculate_qualification_score(data, {}, questions=qs)
    assert score == 100
    assert status == "qualified"


def test_removing_all_but_one_scored_question_does_not_cap_score():
    # Keep only the eviction question. A clean applicant should still hit 100.
    qs = [q for q in default_questions_v2() if q.get("state") == "Q11_EVICTION"]
    data = {"has_eviction": False}
    score, status, _ = calculate_qualification_score(data, {}, questions=qs)
    assert score == 100
    assert status == "qualified"


def test_no_scoring_criteria_routes_to_review_not_auto_qualify():
    # A custom question with scoring disabled: nothing is scorable, so the call
    # routes to human review instead of auto-qualifying or zeroing out.
    q = new_custom_question(question="Tell us about yourself", order=1)
    field = q["extract_fields"][0]
    data = {"questions_answered": 1, field: "some answer"}
    score, status, reasons = calculate_qualification_score(data, {}, questions=[q])
    assert score == 0
    assert status == "review"
    assert any("manual review" in r.lower() for r in reasons)


def test_custom_only_scoring_can_qualify():
    q = new_custom_question(question="Do you smoke?", answer_type="yes_no", order=1)
    field = q["extract_fields"][0]
    q["scoring"] = {
        "enabled": True,
        "max_points": 50,
        "rule_type": "yes_no",
        "pass_config": {"yes": 0, "no": 50},
    }
    data = {field: False}
    score, status, _ = calculate_qualification_score(data, {}, questions=[q])
    assert score == 100
    assert status == "qualified"


def test_partial_answers_score_is_proportional():
    # Only the eviction question answered well; everything else missing.
    qs = default_questions_v2()
    data = {"has_eviction": False}
    score, status, _ = calculate_qualification_score(data, {}, questions=qs)
    assert 0 < score < 100


def test_disqualify_on_yes_marks_unqualified():
    # A per-question "disqualify on yes" gate is the ONLY disqualification
    # mechanism and must fire from pass_config alone.
    q = new_custom_question(question="Any prior eviction?", answer_type="yes_no", order=1)
    field = q["extract_fields"][0]
    q["scoring"] = {
        "enabled": True,
        "max_points": 20,
        "rule_type": "yes_no",
        "pass_config": {"yes": 0, "no": 20, "disqualify_on_yes": True},
    }
    data = {field: True}
    score, status, _ = calculate_qualification_score(data, {}, questions=[q])
    assert score == 0
    assert status == "unqualified"


def test_zero_point_question_can_still_disqualify():
    # A pure gate (0 points) must still be evaluated for disqualification.
    q = new_custom_question(question="Are you a smoker?", answer_type="yes_no", order=1)
    field = q["extract_fields"][0]
    q["scoring"] = {
        "enabled": True,
        "max_points": 0,
        "rule_type": "yes_no",
        "pass_config": {"yes": 0, "no": 0, "disqualify_on_yes": True},
    }
    # Pair it with a real scored question so there are achievable points.
    scored = new_custom_question(question="Income?", answer_type="currency", order=2)
    sfield = scored["extract_fields"][0]
    scored["scoring"] = {
        "enabled": True,
        "max_points": 40,
        "rule_type": "any_answer",
        "pass_config": {},
    }
    data = {field: True, sfield: 8000}
    score, status, _ = calculate_qualification_score(data, {}, questions=[q, scored])
    assert score == 0
    assert status == "unqualified"
