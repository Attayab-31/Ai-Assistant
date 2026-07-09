"""Tests for dynamic screening question flow."""

import pytest

from app.core.question_flow import (
    default_questions_v2,
    migrate_questions_to_v2,
    new_custom_question,
    next_unanswered_state,
    normalize_questions,
    screening_complete,
    validate_questions_for_save,
)


def test_default_questions_are_v2():
    questions = default_questions_v2()
    assert len(questions) >= 1
    assert all(q.get("schema_version") == 2 for q in questions)
    assert all(q.get("answer_type") for q in questions)


def test_migrate_v1_to_v2():
    v1 = [{"id": "Q1", "state": "Q1_FULL_NAME", "question": "Name?", "order": 1}]
    v2 = migrate_questions_to_v2(v1)
    assert v2[0]["answer_type"] == "text"
    assert v2[0]["schema_version"] == 2


def test_add_delete_questions_validation():
    base = default_questions_v2()[:3]
    custom = new_custom_question(question="Do you smoke?", answer_type="yes_no", order=4)
    saved = validate_questions_for_save(base + [custom])
    assert len(saved) == 4
    assert saved[-1]["question"] == "Do you smoke?"


def test_next_unanswered_respects_order():
    questions = [
        {
            "schema_version": 2,
            "id": "A",
            "state": "A",
            "question": "First?",
            "answer_type": "text",
            "extract_fields": ["a_field"],
            "field_labels": {"a_field": "a"},
            "order": 1,
            "active": True,
        },
        {
            "schema_version": 2,
            "id": "B",
            "state": "B",
            "question": "Second?",
            "answer_type": "text",
            "extract_fields": ["b_field"],
            "field_labels": {"b_field": "b"},
            "order": 2,
            "active": True,
        },
    ]
    assert next_unanswered_state({}, questions=questions) == "A"
    assert (
        next_unanswered_state({"a_field": "yes"}, questions=questions) == "B"
    )
    assert next_unanswered_state(
        {"a_field": "yes", "b_field": "ok"}, questions=questions
    ) is None


def test_normalize_accepts_partial_custom_list():
    custom_only = [new_custom_question()]
    normalized = normalize_questions(custom_only)
    assert len(normalized) == 1
    assert normalized[0]["schema_version"] == 2


def test_screening_question_schema_accepts_admin_conditional_operators():
    from app.schemas.settings import ScreeningQuestion

    q = ScreeningQuestion(
        id="q1",
        state="Q_INCOME_HIGH",
        question="High income follow-up?",
        conditional={"field": "monthly_income", "operator": "gte", "value": 5000},
    )
    assert q.conditional is not None
    assert q.conditional.operator == "gte"

    asked = ScreeningQuestion(
        id="q2",
        state="Q_PETS_DETAIL",
        question="Pet details?",
        conditional={"field": "has_pets", "operator": "asked"},
    )
    assert asked.conditional.operator == "asked"


    from app.core.question_flow import ordered_active_questions, should_skip_question

    questions = default_questions_v2()
    no_pets = {"has_pets": False}
    active = ordered_active_questions(questions, no_pets)
    states = [q["state"] for q in active]
    assert "Q6_PETS" in states
    assert "Q6A_PET_DETAILS" not in states
    pet_q = next(q for q in questions if q["state"] == "Q6A_PET_DETAILS")
    assert should_skip_question(pet_q, no_pets)


def test_custom_question_scoring_yes_no():
    from app.core.question_scoring import evaluate_question_scoring

    q = {
        "question": "Do you smoke?",
        "extract_fields": ["smokes"],
        "field_labels": {"smokes": "smoking"},
        "scoring": {
            "enabled": True,
            "max_points": 10,
            "rule_type": "yes_no",
            "pass_config": {"yes": 0, "no": 10},
        },
    }
    pts, reasons, dq = evaluate_question_scoring(q, {"smokes": False})
    assert pts == 10
    assert not reasons
    assert not dq
    pts2, _, _ = evaluate_question_scoring(q, {"smokes": True})
    assert pts2 == 0


def test_question_save_warnings():
    from app.core.question_flow import (
        missing_contact_fields,
        question_save_warnings,
        total_enabled_scoring_points,
    )

    questions = default_questions_v2()
    assert not missing_contact_fields(questions)
    assert total_enabled_scoring_points(questions) >= 0

    custom = new_custom_question(question="Extra?", answer_type="text", order=99)
    custom["scoring"] = {
        "enabled": True,
        "max_points": 60,
        "rule_type": "any_answer",
        "pass_config": {},
    }
    warnings = question_save_warnings([custom])
    assert any("contact" in w.lower() for w in warnings)


def test_coerce_bool_yes_no_strings():
    from app.core.screening_flow import _coerce_bool

    assert _coerce_bool("yes") is True
    assert _coerce_bool("no") is False


def test_scoring_thresholds_from_tenant_snapshot():
    from types import SimpleNamespace

    from app.core.question_flow import scoring_thresholds_from_tenant

    tenant = SimpleNamespace(
        normalized_data={
            "qualified_score_threshold": 80,
            "review_score_threshold": 50,
        }
    )
    assert scoring_thresholds_from_tenant(tenant) == {
        "qualified_score_threshold": 80,
        "review_score_threshold": 50,
    }
    assert scoring_thresholds_from_tenant(
        None, fallback_settings={"qualified_score_threshold": 70, "review_score_threshold": 35}
    ) == {"qualified_score_threshold": 70, "review_score_threshold": 35}


def test_coerce_bool_extended():
    from app.core.screening_flow import _coerce_bool

    assert _coerce_bool("true") is True
    assert _coerce_bool("false") is False
    assert _coerce_bool(True) is True
    assert _coerce_bool("") is None


def test_normalize_coerces_has_flags_to_bool():
    from app.core.screening_flow import normalize_extracted_fields

    out = normalize_extracted_fields({"has_pets": "no", "has_eviction": "yes"})
    # Critically these become real booleans, not truthy strings.
    assert out["has_pets"] is False
    assert out["has_eviction"] is True


def test_normalize_date_object_does_not_raise_name_error():
    """Regression: date values used isinstance(..., datetime) without importing datetime."""
    from datetime import date

    from app.core.screening_flow import normalize_extracted_fields

    out = normalize_extracted_fields(
        {"move_in_date": date(2026, 7, 24)},
        questions=[{"extract_fields": ["move_in_date"], "field_types": {"move_in_date": "date"}}],
    )
    assert out["move_in_date"] == "2026-07-24"


def test_normalize_keeps_eviction_raw_as_caller_text():
    from app.core.screening_flow import normalize_extracted_fields
    from app.core.seed_data import load_seed_questions

    utterance = "No. I have not experienced any such kind of"
    out = normalize_extracted_fields(
        {
            "has_eviction": False,
            "eviction_raw": utterance,
        },
        questions=load_seed_questions(),
    )
    assert out["has_eviction"] is False
    assert out["eviction_raw"] == utterance
    assert isinstance(out["eviction_raw"], str)


def test_coerce_preserves_admin_extract_fields_without_custom_prefix():
    from app.core.data_extractor import coerce_extracted_data
    from app.core.question_flow import new_custom_question

    q = new_custom_question(question="Do you smoke?", answer_type="yes_no", order=1)
    q["extract_fields"] = ["smokes"]
    out = coerce_extracted_data(
        {"smokes": True, "full_name": "Jane Doe"},
        questions=[q],
    )
    assert out["smokes"] is True
    assert out["full_name"] == "Jane Doe"


def test_coerce_extracted_data_drops_bool_raw_fields():
    from app.core.data_extractor import coerce_extracted_data

    out = coerce_extracted_data({"eviction_raw": False, "has_eviction": False})
    assert out["has_eviction"] is False
    assert out["eviction_raw"] is None


def test_infer_monthly_income_from_stt_annual_garble():
    from app.core.screening_flow import (
        infer_monthly_income_from_raw,
        normalize_extracted_fields,
    )

    monthly = infer_monthly_income_from_raw("100,000 k means per year")
    assert monthly is not None
    assert float(monthly) == pytest.approx(8333.33, rel=0.01)

    out = normalize_extracted_fields(
        {"income_raw": "100,000 k means per year", "monthly_income": None}
    )
    assert out["monthly_income"] is not None
    assert float(out["monthly_income"]) == pytest.approx(8333.33, rel=0.01)


def test_infer_monthly_income_defaults_unqualified_period_to_monthly():
    from app.core.screening_flow import infer_monthly_income_from_raw

    monthly = infer_monthly_income_from_raw("4800 per month")
    assert monthly is not None
    assert float(monthly) == pytest.approx(4800.0, rel=0.01)

def test_conditional_truthy_treats_no_string_as_false():
    from app.core.question_flow import evaluate_conditional

    cond = {"field": "has_pets", "operator": "truthy"}
    # "no" as a raw string is truthy in Python but must NOT trigger follow-up.
    assert evaluate_conditional(cond, {"has_pets": "no"}) is False
    assert evaluate_conditional(cond, {"has_pets": "yes"}) is True
    assert evaluate_conditional(cond, {"has_pets": False}) is False
    assert evaluate_conditional(cond, {"has_pets": True}) is True


def test_conditional_eq_understands_bool_strings():
    from app.core.question_flow import evaluate_conditional

    cond = {"field": "flag", "operator": "eq", "value": "yes"}
    assert evaluate_conditional(cond, {"flag": True}) is True
    assert evaluate_conditional(cond, {"flag": "yes"}) is True
    assert evaluate_conditional(cond, {"flag": False}) is False


def test_conditional_numeric_operators():
    from app.core.question_flow import evaluate_conditional

    data = {"monthly_income": "3500"}
    assert evaluate_conditional(
        {"field": "monthly_income", "operator": "gte", "value": 3000}, data
    )
    assert not evaluate_conditional(
        {"field": "monthly_income", "operator": "gt", "value": 3500}, data
    )
    assert evaluate_conditional(
        {"field": "monthly_income", "operator": "lt", "value": 4000}, data
    )


def test_conditional_in_operator():
    from app.core.question_flow import evaluate_conditional

    cond = {"field": "pet_type", "operator": "in", "value": "dog, cat"}
    assert evaluate_conditional(cond, {"pet_type": "dog"})
    assert not evaluate_conditional(cond, {"pet_type": "bird"})


def test_conditional_asked_uses_flow_context():
    from app.core.question_flow import ConditionalFlowContext, evaluate_conditional

    questions = [
        {
            "state": "Q1",
            "extract_fields": ["has_pets"],
            "active": True,
            "order": 1,
        },
        {
            "state": "Q2",
            "extract_fields": ["pet_type"],
            "active": True,
            "order": 2,
        },
    ]
    cond = {"field": "has_pets", "operator": "asked"}
    ctx = ConditionalFlowContext(
        answered_states=frozenset({"Q1"}),
        questions=tuple(questions),
    )
    assert evaluate_conditional(cond, {}, flow_context=ctx)
    assert not evaluate_conditional(
        cond, {}, flow_context=ConditionalFlowContext(questions=tuple(questions))
    )


def test_text_answer_uses_session_raw_answers():
    from app.core.question_flow import is_question_answered_for_def

    q = {
        "state": "Q_NOTES",
        "answer_type": "text",
        "extract_fields": ["notes"],
        "active": True,
        "order": 1,
    }
    assert not is_question_answered_for_def(q, {})
    assert is_question_answered_for_def(q, {}, raw_answers={"Q_NOTES": "hello"})
    assert is_question_answered_for_def(q, {"notes": "typed"})


def test_validate_questions_rejects_numeric_conditional_without_value():
    from app.core.question_flow import new_custom_question, validate_questions_for_save

    base = new_custom_question(question="Income?", answer_type="currency", order=1)
    base["extract_fields"] = ["custom_income"]
    q = new_custom_question(question="High income follow-up?", answer_type="text", order=2)
    q["conditional"] = {
        "field": "custom_income",
        "operator": "gte",
        "value": "",
    }
    with pytest.raises(ValueError, match="numeric value"):
        validate_questions_for_save([base, q])


def test_admin_can_detach_builtin_conditional_on_v2_save():
    """Admin sets a built-in follow-up to 'always ask' -> null must survive.

    Once a question is v2 (admin-managed), an explicit ``conditional: null``
    means the admin deliberately detached the built-in follow-up from its
    parent yes/no. The migration must not silently re-inject the legacy rule.
    """
    from app.core.question_flow import migrate_question_to_v2

    pet_details = {
        "schema_version": 2,
        "id": "Q6A",
        "state": "Q6A_PET_DETAILS",
        "question": "Tell me about your pets.",
        "answer_type": "text",
        "extract_fields": ["pet_type", "pet_breed"],
        "conditional": None,
        "active": True,
        "order": 7,
    }
    migrated = migrate_question_to_v2(dict(pet_details))
    assert migrated["conditional"] is None


def test_legacy_v1_migration_still_injects_builtin_conditional():
    """First-time v1 -> v2 upgrade must still pull in the built-in rule."""
    from app.core.question_flow import migrate_question_to_v2

    legacy_pet_details = {
        "id": "Q6A",
        "state": "Q6A_PET_DETAILS",
        "question": "Tell me about your pets.",
        "extract_fields": ["pet_type", "pet_breed"],
        "active": True,
        "order": 7,
    }
    migrated = migrate_question_to_v2(dict(legacy_pet_details))
    assert migrated["conditional"] == {
        "field": "has_pets",
        "operator": "eq",
        "value": True,
    }


def test_advance_stays_when_llm_incomplete_even_without_question_mark():
    """Follow-up without '?' must not advance while question_complete=false."""
    from app.core.call_handler import question_advance_ready

    assert not question_advance_ready(
        question_complete=False,
        deterministic_done=True,
        understood=True,
    )


def test_advance_proceeds_when_llm_marks_complete_despite_rhetorical_question():
    """Rhetorical '?' in speech must not block when question_complete=true."""
    from app.core.call_handler import question_advance_ready

    assert question_advance_ready(
        question_complete=True,
        deterministic_done=True,
        understood=True,
    )


def test_advance_when_deterministic_done_and_caller_not_understood():
    from app.core.call_handler import question_advance_ready

    assert question_advance_ready(
        question_complete=False,
        deterministic_done=True,
        understood=False,
    )


def test_advance_when_llm_marks_complete_without_deterministic_slots():
    from app.core.call_handler import question_advance_ready

    assert question_advance_ready(
        question_complete=True,
        deterministic_done=False,
        understood=True,
    )


def test_scoring_does_not_crash_on_malformed_config():
    from app.core.question_scoring import evaluate_question_scoring

    # Non-numeric pass_config values must not raise (this runs inside finalize).
    q = {
        "question": "Move in soon?",
        "extract_fields": ["smokes"],
        "scoring": {
            "enabled": True,
            "max_points": 10,
            "rule_type": "yes_no",
            "pass_config": {"yes": "abc", "no": "xyz"},
        },
    }
    pts, _, _ = evaluate_question_scoring(q, {"smokes": True})
    assert isinstance(pts, int)
    pts2, _, _ = evaluate_question_scoring(q, {"smokes": False})
    assert isinstance(pts2, int)

    date_q = {
        "question": "When?",
        "extract_fields": ["move_in_date"],
        "scoring": {
            "enabled": True,
            "max_points": 10,
            "rule_type": "date_within",
            "pass_config": {"max_days_ahead": "not-a-number"},
        },
    }
    pts3, _, _ = evaluate_question_scoring(date_q, {"move_in_date": "2099-01-01"})
    assert isinstance(pts3, int)


def test_save_warns_on_unreachable_and_unscored_flow():
    from app.core.question_flow import new_custom_question, question_save_warnings

    # Only an all-conditional active flow → unreachable at start.
    base = new_custom_question(question="Got X?", answer_type="yes_no", order=1)
    follow = new_custom_question(question="Details?", answer_type="text", order=2)
    follow["conditional"] = {
        "field": base["extract_fields"][0],
        "operator": "truthy",
    }
    base["active"] = False
    warnings = question_save_warnings([base, follow])
    assert any("start of the call" in w for w in warnings)


def test_analyze_questions_draft_no_scoring_warning():
    from app.core.question_flow import analyze_questions_draft, new_custom_question

    q = new_custom_question(question="Notes?", answer_type="text", order=1)
    q["scoring"] = {"enabled": False, "max_points": 0, "rule_type": "any_answer", "pass_config": {}}
    analysis = analyze_questions_draft([q])
    assert analysis["validation_error"] is None
    assert analysis["runtime_valid"] is True
    assert any("scoring enabled" in w.lower() for w in analysis["warnings"])


def test_analyze_questions_draft_validation_error():
    from app.core.question_flow import analyze_questions_draft, new_custom_question

    base = new_custom_question(question="Gate?", answer_type="yes_no", order=1)
    base["extract_fields"] = ["custom_gate_flag"]
    base["active"] = False
    follow = new_custom_question(question="Follow?", answer_type="text", order=2)
    follow["conditional"] = {"field": "custom_gate_flag", "operator": "eq", "value": True}
    analysis = analyze_questions_draft([base, follow])
    assert analysis["validation_error"] is not None
    assert analysis["runtime_valid"] is False
    assert analysis["runtime_errors"]


def test_analyze_questions_draft_uses_validated_override():
    from app.core.question_flow import analyze_questions_draft, default_questions_v2

    validated = default_questions_v2()
    analysis = analyze_questions_draft([], validated=validated)
    assert analysis["validation_error"] is None
    assert analysis["runtime_valid"] is True


def test_custom_question_cannot_use_reserved_column():
    from app.core.question_flow import new_custom_question

    bad = new_custom_question(question="Your email?", answer_type="email", order=1)
    bad["extract_fields"] = ["email"]
    bad["state"] = "CUSTOM_EMAIL"
    try:
        validate_questions_for_save([bad])
    except ValueError as exc:
        assert "reserved" in str(exc).lower()
    else:
        raise AssertionError("Expected ValueError for reserved field name")


class _FakeTenant:
    """Minimal stand-in for the Tenant ORM row (scoring reads attributes)."""

    def __init__(self, **kw):
        defaults = {
            "monthly_income": None,
            "income_raw": None,
            "has_eviction": None,
            "eviction_circumstances": None,
            "eviction_raw": None,
            "current_residence": None,
            "residence_duration": None,
            "move_reason": None,
            "move_in_date": None,
            "move_in_raw": None,
            "move_timing": None,
            "occupants_count": None,
            "adults_count": None,
            "children_count": None,
            "has_pets": None,
            "pet_type": None,
            "pet_breed": None,
            "pets_raw": None,
            "pet_weight": None,
            "employer": None,
            "employment_duration": None,
            "general_notes": None,
            "special_notes": None,
            "human_requested": None,
            "callback_requested": None,
            "stop_requested": None,
            "answered_states": None,
            "refused_states": None,
            "normalized_data": None,
        }
        defaults.update(kw)
        for k, v in defaults.items():
            setattr(self, k, v)


def test_scoring_data_reconstructs_full_record_for_consistency():
    """The admin breakdown must score the SAME data finalize did.

    Regression: the call-detail page used a 4-field stub, so it recomputed a
    different (lower) score than the one stored at finalize. The shared
    reconstruction must surface every scored field, incl. custom fields.
    """
    from decimal import Decimal

    from app.core.qualifier import (
        build_tenant_scoring_data,
        calculate_qualification_score,
    )

    tenant = _FakeTenant(
        monthly_income=Decimal("70000"),
        has_eviction=False,
        current_residence="New York",
        residence_duration="3 years",
        move_reason="commute",
        occupants_count=3,
        has_pets=True,
        pet_type="dog",
        pets_raw="German Shepherd",
        pet_weight="medium",
        answered_states=["Q1_FULL_NAME", "Q5_OCCUPANTS"],
        normalized_data={"custom_fields": {"smokes": False}},
    )

    data = build_tenant_scoring_data(tenant, questions_answered=16)
    # Full record, not a stub: household + rental fields are present.
    assert data["occupants_count"] == 3
    assert data["has_pets"] is True
    assert data["current_residence"] == "New York"
    # Custom (dynamic, non-column) fields are surfaced for per-question scoring.
    assert data["smokes"] is False
    assert data["answered_states"] == ["Q1_FULL_NAME", "Q5_OCCUPANTS"]

    # Scoring the reconstructed data is stable across calls (deterministic).
    score_a, status_a, _ = calculate_qualification_score(data, {})
    score_b, status_b, _ = calculate_qualification_score(
        build_tenant_scoring_data(tenant, questions_answered=16), {}
    )
    assert (score_a, status_a) == (score_b, status_b)


def test_scoring_skips_inactive_questions():
    from app.core.question_scoring import score_custom_questions

    q = new_custom_question(question="Inactive scored?", answer_type="yes_no", order=1)
    q["active"] = False
    q["scoring"] = {
        "enabled": True,
        "max_points": 50,
        "rule_type": "any_answer",
        "pass_config": {},
    }
    field = q["extract_fields"][0]
    total, _, breakdown, _ = score_custom_questions([q], {field: "yes"})
    assert total == 0
    assert breakdown == []


def test_scoring_excludes_conditionally_skipped_questions():
    from app.core.question_scoring import score_custom_questions

    parent = new_custom_question(question="Pets?", answer_type="yes_no", order=1)
    parent["extract_fields"] = ["has_pets"]
    parent["state"] = "Q_PETS"
    follow = new_custom_question(question="Pet breed?", answer_type="text", order=2)
    follow["state"] = "Q_PET_BREED"
    follow["conditional"] = {"field": "has_pets", "operator": "truthy"}
    follow["scoring"] = {
        "enabled": True,
        "max_points": 40,
        "rule_type": "any_answer",
        "pass_config": {},
    }
    tenant_data = {
        "has_pets": False,
        "answered_states": ["Q_PETS"],
        "refused_states": [],
    }
    total, _, breakdown, _ = score_custom_questions(
        [parent, follow], tenant_data
    )
    assert total == 0
    assert breakdown == []


    from app.core.question_flow import (
        build_question_slot_config,
        default_questions_v2,
        is_question_answered_for_def,
    )

    pet_followup = next(
        q for q in default_questions_v2() if q["state"] == "Q6A_PET_DETAILS"
    )
    assert pet_followup.get("require_all_extract_fields") is True

    partial = {"has_pets": True, "pet_type": "dog"}
    assert not is_question_answered_for_def(pet_followup, partial)

    complete = {
        **partial,
        "pet_breed": "labrador",
        "pet_weight": "60 lbs",
    }
    assert is_question_answered_for_def(pet_followup, complete)

    slots = build_question_slot_config(pet_followup)
    assert set(slots["required"]) == {"pet_type", "pet_breed", "pet_weight"}


def test_optional_question_not_required_by_default():
    from app.core.question_flow import default_questions_v2, is_question_required

    notes = next(
        q for q in default_questions_v2() if q["state"] == "Q15_GENERAL_NOTES"
    )
    assert is_question_required(notes) is False


def test_skipped_conditional_is_not_counted_answered():
    from app.core.question_flow import (
        build_flow_rows,
        is_question_answered_for_def,
    )

    questions = default_questions_v2()
    pet_followup = next(q for q in questions if q["state"] == "Q6A_PET_DETAILS")
    assert not is_question_answered_for_def(pet_followup, {"has_pets": False})
    assert not is_question_answered_for_def(
        pet_followup, {"has_pets": False, "pet_type": "dog"}
    )

    rows = build_flow_rows(
        questions,
        answered_states=["Q6A_PET_DETAILS"],
        refused_states=[],
        scoring_data={"has_pets": False},
    )
    pet_row = next(r for r in rows if r["state"] == "Q6A_PET_DETAILS")
    assert pet_row["status"] == "Skipped"


def test_build_flow_rows_asked_conditional_uses_visit_history():
    from app.core.question_flow import build_flow_rows

    questions = [
        {
            "id": "q1",
            "state": "Q1",
            "question": "Pets?",
            "answer_type": "yes_no",
            "extract_fields": ["has_pets"],
            "active": True,
            "order": 1,
        },
        {
            "id": "q2",
            "state": "Q2",
            "question": "Pet type?",
            "answer_type": "text",
            "extract_fields": ["pet_type"],
            "active": True,
            "order": 2,
            "conditional": {"field": "has_pets", "operator": "asked"},
        },
    ]
    rows = build_flow_rows(
        questions,
        answered_states=["Q1"],
        refused_states=[],
        scoring_data={},
    )
    followup = next(r for r in rows if r["state"] == "Q2")
    assert followup["status"] == "—"
    rows_skipped = build_flow_rows(
        questions,
        answered_states=[],
        refused_states=[],
        scoring_data={},
    )
    skipped = next(r for r in rows_skipped if r["state"] == "Q2")
    assert skipped["status"] == "Skipped"


def test_language_choice_must_be_first_question():
    from app.core.question_flow import validate_questions_for_save

    with pytest.raises(ValueError, match="order 1"):
        validate_questions_for_save(
            [
                {
                    "id": "q1",
                    "state": "Q1",
                    "question": "Name?",
                    "answer_type": "text",
                    "extract_fields": ["full_name"],
                    "order": 1,
                    "active": True,
                },
                {
                    "id": "lang",
                    "state": "LANG",
                    "question": "Language?",
                    "answer_type": "language_choice",
                    "extract_fields": ["preferred_language"],
                    "language_options": [
                        {"value": "en", "label": "English"},
                        {"value": "es", "label": "Spanish"},
                    ],
                    "order": 2,
                    "active": True,
                },
            ]
        )


def test_confirmation_required_before_question_counts_answered():
    from app.core.question_flow import (
        count_answered_questions,
        is_question_answered_for_def,
        next_unanswered_state,
    )

    questions = default_questions_v2()
    phone_q = next(q for q in questions if q["state"] == "Q2_PHONE")
    data = {"full_name": "Stone Smith", "contact_phone": "+13174026780"}
    confirmed = {"full_name", "contact_phone"}
    assert not is_question_answered_for_def(phone_q, data, confirmed_fields=set())
    assert next_unanswered_state(
        data, questions=questions, confirmed_fields={"full_name"}
    ) == "Q2_PHONE"
    assert is_question_answered_for_def(phone_q, data, confirmed_fields=confirmed)
    assert count_answered_questions(
        data, questions=questions, confirmed_fields=confirmed
    ) == 2


def test_readback_prompt_uses_admin_question_metadata():
    from app.core.question_flow import readback_prompt_for_question

    q = {
        "state": "CUSTOM_PHONE",
        "answer_type": "phone",
        "extract_fields": ["callback_number"],
        "field_labels": {"callback_number": "callback number"},
    }
    spoken = readback_prompt_for_question(q, "+15551234567")
    assert "callback number" not in spoken.lower()  # phone type uses digit read-back
    assert "Is that correct?" in spoken


def test_needs_readback_confirmation_respects_confirmed_fields():
    from app.core.question_flow import needs_readback_confirmation

    questions = default_questions_v2()
    data = {"full_name": "John Smith"}
    assert needs_readback_confirmation("Q1_FULL_NAME", data, questions, set())
    assert not needs_readback_confirmation(
        "Q1_FULL_NAME", data, questions, {"full_name"}
    )


def test_screening_complete_requires_confirmation_context():
    questions = [
        {
            "schema_version": 2,
            "id": "Q1",
            "state": "Q1",
            "question": "Name?",
            "answer_type": "text",
            "extract_fields": ["full_name"],
            "field_labels": {"full_name": "full name"},
            "requires_confirmation": True,
            "order": 1,
            "active": True,
        },
        {
            "schema_version": 2,
            "id": "Q2",
            "state": "Q2",
            "question": "Phone?",
            "answer_type": "phone",
            "extract_fields": ["contact_phone"],
            "field_labels": {"contact_phone": "contact phone"},
            "requires_confirmation": True,
            "order": 2,
            "active": True,
        },
    ]
    data = {
        "full_name": "John Smith",
        "contact_phone": "+13174026780",
    }
    # Name confirmed, phone captured but not yet confirmed -> still incomplete.
    assert not screening_complete(
        data,
        questions=questions,
        confirmed_fields={"full_name"},
    )
    # Once both are confirmed, flow can continue past both confirmation-gated steps.
    assert screening_complete(
        data,
        questions=questions,
        confirmed_fields={"full_name", "contact_phone"},
    )


def test_session_is_screening_complete_uses_confirmed_fields():
    from app.core.conversation import ConversationSession

    questions = [
        {
            "schema_version": 2,
            "id": "Q1",
            "state": "Q1",
            "question": "Name?",
            "answer_type": "text",
            "extract_fields": ["full_name"],
            "field_labels": {"full_name": "full name"},
            "requires_confirmation": True,
            "order": 1,
            "active": True,
        },
        {
            "schema_version": 2,
            "id": "Q2",
            "state": "Q2",
            "question": "Phone?",
            "answer_type": "phone",
            "extract_fields": ["contact_phone"],
            "field_labels": {"contact_phone": "contact phone"},
            "requires_confirmation": True,
            "order": 2,
            "active": True,
        },
    ]
    session = ConversationSession(
        call_id="complete-check",
        phone_number="+15550000000",
        questions=questions,
    )
    session.extracted_data.update(
        {
            "full_name": "John Smith",
            "contact_phone": "+13174026780",
        }
    )

    assert session.is_screening_complete() is False
    session.mark_field_confirmed("full_name")
    session.mark_field_confirmed("contact_phone")
    assert session.is_screening_complete() is True


def test_screening_complete_honors_raw_answers_context():
    questions = [
        {
            "schema_version": 2,
            "id": "Q_NOTES",
            "state": "Q_NOTES",
            "question": "Any notes?",
            "answer_type": "text",
            "extract_fields": ["notes"],
            "field_labels": {"notes": "notes"},
            "requires_confirmation": False,
            "order": 1,
            "active": True,
        }
    ]
    # Text questions can be considered answered from raw_answers even when no
    # normalized extract slot was written.
    assert screening_complete({}, questions=questions, raw_answers={"Q_NOTES": "hello"})


def test_compose_agent_response_dedupes_follow_up():
    from app.core.conversation import (
        ConversationSession,
        compose_agent_response,
        dedupe_repeated_speech,
        strip_upcoming_question_from_ack,
    )

    session = ConversationSession(
        call_id="t",
        phone_number="+15550000000",
        questions=default_questions_v2(),
        current_state="Q2_PHONE",
    )
    question = "What is the best phone number for you?"
    ack, follow_up = compose_agent_response(
        session,
        f"Great, thanks. {question}",
        "Q1_FULL_NAME",
    )
    assert question in ack
    assert follow_up == ""
    trimmed = strip_upcoming_question_from_ack(
        session, f"Great, thanks. {question} {question}"
    )
    assert trimmed == "Great, thanks"
    assert (
        dedupe_repeated_speech(f"{question} {question}")
        == question
    )


def test_prompt_fields_catalog_from_admin_questions():
    from app.core.question_flow import new_custom_question, prompt_fields_catalog

    custom = new_custom_question(
        question="Do you smoke?",
        answer_type="yes_no",
        order=99,
    )
    catalog = prompt_fields_catalog([custom])
    field = custom["extract_fields"][0]
    assert f"- {field}:" in catalog
    assert "(yes_no)" in catalog


def test_two_date_questions_do_not_share_move_in_raw():
    from app.core.question_flow import is_question_answered_for_def

    move_in = {
        "schema_version": 2,
        "id": "MOVE",
        "state": "MOVE",
        "question": "Move in?",
        "answer_type": "date",
        "extract_fields": ["move_in_date", "move_in_raw"],
        "order": 1,
        "active": True,
    }
    lease_end = {
        "schema_version": 2,
        "id": "LEASE",
        "state": "LEASE",
        "question": "Lease ends?",
        "answer_type": "date",
        "extract_fields": ["lease_end_date", "lease_end_raw"],
        "order": 2,
        "active": True,
    }
    data = {"move_in_date": "2026-07-24", "move_in_raw": "July 24"}
    assert is_question_answered_for_def(move_in, data)
    assert not is_question_answered_for_def(lease_end, data)


def test_normalize_extracted_fields_applies_custom_phone_type():
    from app.core.question_flow import new_custom_question
    from app.core.screening_flow import normalize_extracted_fields

    custom = new_custom_question(question="Callback number?", answer_type="phone", order=1)
    field = custom["extract_fields"][0]
    normalized = normalize_extracted_fields(
        {field: "5551234567"},
        questions=[custom],
    )
    assert normalized[field] == "+15551234567"


def test_build_system_prompt_lists_admin_fields_not_hardcoded_only():
    from app.core.conversation import ConversationSession, build_system_prompt
    from app.core.question_flow import new_custom_question

    custom = new_custom_question(
        question="Do you have a vehicle?",
        answer_type="yes_no",
        order=1,
    )
    field = custom["extract_fields"][0]
    session = ConversationSession(
        call_id="t",
        phone_number="+15550000000",
        questions=[custom],
        current_state=custom["state"],
    )
    prompt = build_system_prompt(session, transcript="Yes I do")
    assert field in prompt
    assert "Fields:" in prompt
    assert "has_pets" not in prompt or field in prompt


def test_builtin_primary_field_cannot_change_on_save():
    from app.core.question_flow import default_questions_v2, validate_questions_for_save

    questions = default_questions_v2()
    for i, q in enumerate(questions):
        if q["state"] == "Q1_FULL_NAME":
            bad = dict(q)
            bad["extract_fields"] = ["renamed_name"]
            questions[i] = bad
            break
    with pytest.raises(ValueError, match="primary field"):
        validate_questions_for_save(questions)


def test_custom_question_primary_field_can_change():
    from app.core.question_flow import new_custom_question, validate_questions_for_save

    custom = new_custom_question(question="Vehicle?", answer_type="yes_no", order=1)
    custom["extract_fields"] = ["custom_vehicle"]
    custom["field_labels"] = {"custom_vehicle": "has a vehicle"}
    saved = validate_questions_for_save([custom])
    assert saved[0]["extract_fields"][0] == "custom_vehicle"


def test_understanding_guide_reaches_system_prompt():
    from app.core.conversation import ConversationSession, build_system_prompt
    from app.core.question_flow import new_custom_question

    custom = new_custom_question(question="Any vehicles?", answer_type="text", order=1)
    custom["understanding_guide"] = "Count cars and motorcycles separately."
    session = ConversationSession(
        call_id="t",
        phone_number="+15550000000",
        questions=[custom],
        current_state=custom["state"],
    )
    prompt = build_system_prompt(session, transcript="Two cars")
    assert "Count cars and motorcycles separately." in prompt


def test_build_system_prompt_includes_admin_flow_outline():
    from app.core.conversation import ConversationSession, build_system_prompt
    from app.core.question_flow import new_custom_question

    custom = new_custom_question(question="Do you smoke?", answer_type="yes_no", order=1)
    custom["requires_confirmation"] = True
    session = ConversationSession(
        call_id="t",
        phone_number="+15550000000",
        questions=[custom],
        current_state=custom["state"],
    )
    prompt = build_system_prompt(session, transcript="No")
    assert "Flow order" in prompt
    assert custom["state"] in prompt
    assert "Do you smoke?" in prompt
    assert "[confirm]" in prompt
    field = custom["extract_fields"][0]
    assert field in prompt
    assert "[confirm]" in prompt


def test_prompt_flow_outline_windows_long_admin_flow():
    from app.core.question_flow import default_questions_v2, prompt_screening_flow_outline

    questions = default_questions_v2()
    mid_state = questions[8]["state"]
    outline = prompt_screening_flow_outline(questions, current_state=mid_state)
    assert "← CURRENT" in outline
    assert mid_state in outline
    assert "earlier step(s)" in outline or "more step(s)" in outline
    assert outline.count("\n") < len(questions)


def test_merge_extracted_data_rejects_unknown_fields():
    from app.core.conversation import ConversationSession
    from app.core.question_flow import default_questions_v2

    session = ConversationSession(
        call_id="t",
        phone_number="+15550000000",
        questions=default_questions_v2(),
    )
    session.merge_extracted_data(
        {
            "full_name": "Dawn Smith",
            "pet_age": "three years",
            "made_up_field": "nope",
        }
    )
    assert session.extracted_data.get("full_name") == "Dawn Smith"
    assert "pet_age" not in session.extracted_data
    assert "made_up_field" not in session.extracted_data


def test_merge_extracted_data_allows_admin_custom_field():
    from app.core.conversation import ConversationSession
    from app.core.question_flow import new_custom_question, validate_questions_for_save

    custom = new_custom_question(question="Pet age?", answer_type="text", order=99)
    custom["extract_fields"] = ["pet_age"]
    custom["field_labels"] = {"pet_age": "pet age"}
    questions = validate_questions_for_save([custom])
    session = ConversationSession(
        call_id="t",
        phone_number="+15550000000",
        questions=questions,
        current_state=custom["state"],
    )
    session.merge_extracted_data({"pet_age": "three years"})
    assert session.extracted_data.get("pet_age") == "three years"


def test_merge_extracted_data_skips_confirmed_admin_field():
    from app.core.conversation import ConversationSession
    from app.core.question_flow import new_custom_question, validate_questions_for_save

    custom = new_custom_question(
        question="What is your ID number?",
        answer_type="text",
        order=1,
    )
    custom["extract_fields"] = ["custom_id_number"]
    custom["field_labels"] = {"custom_id_number": "ID number"}
    custom["requires_confirmation"] = True
    questions = validate_questions_for_save([custom])
    session = ConversationSession(
        call_id="lock",
        phone_number="+1",
        questions=questions,
        current_state=custom["state"],
    )
    session.extracted_data["custom_id_number"] = "ID-111"
    session.mark_field_confirmed("custom_id_number")

    session.merge_extracted_data({"custom_id_number": "ID-999", "other_note": "x"})

    assert session.extracted_data["custom_id_number"] == "ID-111"
    assert "other_note" not in session.extracted_data


def test_merge_extracted_data_allows_explicit_correction_of_confirmed_field():
    from app.core.conversation import ConversationSession
    from app.core.question_flow import default_questions_v2

    session = ConversationSession(
        call_id="corr",
        phone_number="+1",
        questions=default_questions_v2(),
    )
    session.extracted_data["full_name"] = "Jane Doe"
    session.mark_field_confirmed("full_name")

    session.merge_extracted_data(
        {"full_name": "Janet Doe"},
        allow_overwrite=frozenset({"full_name"}),
    )

    assert session.extracted_data["full_name"] == "Janet Doe"


def test_filter_extracted_to_allowed_fields_at_finalize():
    from app.core.conversation import filter_extracted_to_allowed_fields
    from app.core.question_flow import default_questions_v2

    filtered = filter_extracted_to_allowed_fields(
        {
            "full_name": "Dawn Smith",
            "pet_age": "three years",
            "invented": "x",
        },
        default_questions_v2(),
    )
    assert filtered == {"full_name": "Dawn Smith"}


def test_retry_prompt_for_count_uses_admin_escalation():
    from app.core.question_flow import retry_prompt_for_count

    q = {
        "question": "Base?",
        "retry_prompt": "Retry 1",
        "retry_prompt_2": "Retry 2",
        "retry_prompt_3": "Retry 3",
    }
    assert retry_prompt_for_count(q, 0) == "Base?"
    assert retry_prompt_for_count(q, 1) == "Retry 2"
    assert retry_prompt_for_count(q, 2) == "Retry 3"


def test_retry_prompt_for_count_uses_localized_overrides():
    from app.core.question_flow import retry_prompt_for_count

    q = {
        "question": "What email address should we use?",
        "retry_prompt": "Could you spell it?",
        "retry_prompt_2": "Please say it slowly.",
        "retry_prompt_3": "One more time please.",
        "locales": {
            "es": {
                "question": "Que correo electronico debemos usar?",
                "retry_prompt": "Podria deletrearlo?",
                "retry_prompt_2": "Digalo despacio, por favor.",
                "retry_prompt_3": "Una vez mas, por favor.",
            }
        },
    }
    assert retry_prompt_for_count(q, 0, language_code="es") == (
        "Que correo electronico debemos usar?"
    )
    assert retry_prompt_for_count(q, 1, language_code="es") == (
        "Digalo despacio, por favor."
    )
    assert retry_prompt_for_count(q, 2, language_code="es") == (
        "Una vez mas, por favor."
    )


def test_polite_redirect_uses_call_language_for_retry_prompt():
    from app.core.conversation import ConversationSession, polite_redirect

    questions = [
        {
            "schema_version": 2,
            "id": "Q1",
            "state": "Q1_FULL_NAME",
            "question": "Can I have your full name?",
            "answer_type": "text",
            "extract_fields": ["full_name"],
            "field_labels": {"full_name": "full name"},
            "retry_prompt": "What is your full name?",
            "retry_prompt_2": "Could you tell me your first and last name?",
            "retry_prompt_3": "Just your first and last name is perfect.",
            "order": 1,
            "active": True,
            "locales": {
                "es": {
                    "question": "Cual es su nombre completo?",
                    "retry_prompt": "Cual es su nombre completo?",
                    "retry_prompt_2": "Podria decirme su nombre y apellido?",
                    "retry_prompt_3": "Solo su nombre y apellido esta perfecto.",
                }
            },
        }
    ]
    session = ConversationSession(
        call_id="t",
        phone_number="+15550000000",
        questions=questions,
        current_state="Q1_FULL_NAME",
        call_language="es",
        retry_count=1,
    )

    text = polite_redirect(session, "unclear")
    assert "Podria decirme su nombre y apellido?" in text
    assert "Could you tell me your first and last name?" not in text


def test_localized_question_text_falls_back_to_base():
    from app.core.question_flow import localized_question_text

    q = {
        "question": "Where do you work?",
        "retry_prompt": "Could you share your employer?",
        "locales": {"es": {"question": "Donde trabaja?"}},
    }
    assert localized_question_text(q, language_code="es", key="question") == "Donde trabaja?"
    assert localized_question_text(q, language_code="es", key="retry_prompt") == (
        "Could you share your employer?"
    )


def test_count_missing_spanish_question_overrides():
    from app.core.question_flow import (
        count_missing_spanish_question_overrides,
        new_custom_question,
    )

    custom = new_custom_question(question="Smoke?", answer_type="yes_no", order=1)
    assert count_missing_spanish_question_overrides([custom]) == 1
    custom["locales"] = {"es": {"question": "Fuma?"}}
    assert count_missing_spanish_question_overrides([custom]) == 0


def test_merge_seed_spanish_locales_preserves_admin_spanish():
    from app.core.question_flow import merge_seed_spanish_locales

    stored = [
        {
            "state": "Q1_FULL_NAME",
            "question": "Name?",
            "locales": {"es": {"question": "Nombre admin?"}},
        }
    ]
    seed = {
        "Q1_FULL_NAME": {
            "locales": {
                "es": {
                    "question": "Seed Spanish?",
                    "retry_prompt": "Seed retry?",
                }
            }
        }
    }
    merged, updated = merge_seed_spanish_locales(stored, seed)
    assert updated == 1
    es = merged[0]["locales"]["es"]
    assert es["question"] == "Nombre admin?"
    assert es["retry_prompt"] == "Seed retry?"


def test_default_seed_questions_include_spanish_locales():
    from app.core.question_flow import default_questions_v2, localized_question_text

    from app.core.seed_data import load_seed_questions

    load_seed_questions.cache_clear()
    q1 = next(q for q in default_questions_v2() if q["state"] == "Q1_FULL_NAME")
    assert localized_question_text(q1, language_code="es", key="question").startswith(
        "¿"
    )


def test_question_save_warnings_missing_spanish():
    from app.core.question_flow import new_custom_question, question_save_warnings

    custom = new_custom_question(question="Pets?", answer_type="yes_no", order=1)
    warnings = question_save_warnings([custom])
    assert any("Spanish wording" in w for w in warnings)


def test_raw_field_types_always_text():
    from app.core.question_flow import default_questions_v2, field_answer_types_from_questions

    types = field_answer_types_from_questions(default_questions_v2())
    assert types["income_raw"] == "text"
    assert types["eviction_raw"] == "text"
    assert types["monthly_income"] == "currency"


def test_normalize_extracted_fields_sanitizes_raw_fields():
    from decimal import Decimal

    from app.core.screening_flow import normalize_extracted_fields

    out = normalize_extracted_fields(
        {
            "has_eviction": True,
            "eviction_raw": True,
            "monthly_income": 50000,
            "income_raw": Decimal("50000.00"),
        }
    )
    assert out["has_eviction"] is True
    assert "eviction_raw" not in out
    assert out["income_raw"] == "50000.00"


def test_build_system_prompt_token_budget():
    from app.core.conversation import ConversationSession, build_system_prompt
    from app.core.question_flow import default_questions_v2

    session = ConversationSession(
        call_id="t",
        phone_number="+15550000000",
        questions=default_questions_v2(),
        property_name="Ready Rentals",
    )
    session.extracted_data = {
        "full_name": "Dawn Smith",
        "contact_phone": "+13174026038",
        "email": "test@gmail.com",
    }
    prompt = build_system_prompt(session, transcript="Three people.")
    assert len(prompt) < 8500
    assert "Relevant FAQ data this turn" not in prompt
    assert "Local fallback hints" not in prompt
    assert "income_raw" in prompt
    assert "(text)" in prompt


def test_prompt_extraction_rules_follow_active_questions_only():
    from app.core.question_flow import new_custom_question, prompt_extraction_rules

    custom = new_custom_question(question="Favorite color?", answer_type="text", order=1)
    rules = prompt_extraction_rules([custom])
    assert "Eviction means" not in rules
    assert "monthly_income" not in rules
    assert "Today's date:" in rules

    income_q = new_custom_question(question="Income?", answer_type="currency", order=2)
    rules_income = prompt_extraction_rules([income_q])
    assert "money fields" in rules_income


def test_normalize_rolls_future_for_any_date_field():
    from datetime import date

    from app.core.question_flow import new_custom_question
    from app.core.screening_flow import normalize_extracted_fields

    custom = new_custom_question(
        question="When do you need parking?",
        answer_type="date",
        order=99,
    )
    custom["extract_fields"] = ["custom_parking_date"]
    out = normalize_extracted_fields(
        {"custom_parking_date": "2020-09-01"},
        questions=[custom],
    )
    parsed = date.fromisoformat(out["custom_parking_date"])
    assert parsed >= date.today()


def test_build_preview_sample_paths_includes_followups():
    from app.core.question_flow import build_preview_sample_paths, default_questions_v2

    paths = build_preview_sample_paths(default_questions_v2())
    ids = {p["id"] for p in paths}
    assert "default" in ids
    assert "has_pets__true" in ids
    assert "has_eviction__true" in ids
    assert "all_followups" in ids


def test_build_conversation_preview_flow_uses_localized_question_text():
    from app.core.question_flow import build_conversation_preview_flow

    questions = [
        {
            "schema_version": 2,
            "id": "Q1",
            "state": "Q1_FULL_NAME",
            "question": "Can I start with your full name?",
            "answer_type": "text",
            "extract_fields": ["full_name"],
            "field_labels": {"full_name": "full name"},
            "retry_prompt": "",
            "retry_prompt_2": "",
            "retry_prompt_3": "",
            "active": True,
            "order": 1,
            "required": True,
            "requires_confirmation": False,
            "conditional": None,
            "scoring": {
                "enabled": False,
                "max_points": 0,
                "rule_type": "any_answer",
                "pass_config": {},
            },
            "locales": {"es": {"question": "Puedo comenzar con su nombre completo?"}},
        }
    ]
    flow = build_conversation_preview_flow(
        questions,
        {},
        business="Ready Rentals",
        language_code="es",
    )
    assert "Puedo comenzar con su nombre completo?" in flow[0]["text"]


def test_speech_mode_pet_bundle_extraction():
    from app.core.question_flow import default_questions_v2, extract_fields_from_speech

    pet_q = next(q for q in default_questions_v2() if q["state"] == "Q6A_PET_DETAILS")
    out = extract_fields_from_speech("German shepherd about 70 pounds", pet_q)
    assert out.get("pet_weight") == 70
    assert out.get("pets_raw")


def test_build_system_prompt_includes_today():
    from datetime import date

    from app.core.conversation import ConversationSession, build_system_prompt
    from app.core.question_flow import default_questions_v2

    session = ConversationSession(
        call_id="t",
        phone_number="+15550000000",
        questions=default_questions_v2(),
    )
    prompt = build_system_prompt(session, transcript="July 24")
    assert date.today().isoformat() in prompt
    assert "past" in prompt.lower()


def test_validate_blocks_all_conditional_flow():
    from app.core.question_flow import new_custom_question, validate_questions_for_save

    gate = new_custom_question(question="Gate?", answer_type="yes_no", order=1)
    gate["extract_fields"] = ["custom_gate_flag"]
    gate["active"] = False
    follow = new_custom_question(question="Follow?", answer_type="text", order=2)
    follow["extract_fields"] = ["custom_follow_note"]
    follow["conditional"] = {"field": "custom_gate_flag", "operator": "eq", "value": True}
    with pytest.raises(ValueError, match="start of a call"):
        validate_questions_for_save([gate, follow])


def test_coerce_questions_for_runtime_fail_closed_on_empty():
    from app.core.question_flow import coerce_questions_for_runtime

    assert coerce_questions_for_runtime([]) == []


def test_coerce_questions_for_runtime_with_reason():
    from app.core.question_flow import (
        coerce_questions_for_runtime_with_reason,
        default_questions_v2,
    )

    qs, reason = coerce_questions_for_runtime_with_reason([])
    assert qs == []
    assert reason is not None
    assert "blocked" in reason.lower()

    valid, no_reason = coerce_questions_for_runtime_with_reason(default_questions_v2())
    assert no_reason is None
    assert len(valid) == len(default_questions_v2())


@pytest.mark.asyncio
async def test_handle_call_answered_blocked_on_invalid_questions_config(monkeypatch):
    from app.core.call_handler import handle_call_answered
    from app.core.conversation import CallState, ConversationSession

    session = ConversationSession(
        call_id="blocked-q",
        phone_number="+1",
        property_name="Demo Property",
        questions=[],
    )
    session.control_flags["questions_config_blocked"] = (
        "Duplicate question states are not allowed"
    )

    async def fake_synth(*args, **kwargs):
        return [b"audio"]

    monkeypatch.setattr(
        "app.core.call_handler.synthesize_speech_parts",
        fake_synth,
    )

    audio = await handle_call_answered(session)
    assert audio == [b"audio"]
    assert session.current_state == CallState.ENDED.value
    assert session.control_flags.get("provider_failure", {}).get("service") == "questions"
    assert session.control_flags.get("questions_config_fallback") == (
        "Duplicate question states are not allowed"
    )


def test_snapshot_blocks_calls_when_questions_invalid():
    from app.core.call_settings import snapshot_from_map

    snap = snapshot_from_map({"screening_questions": []})
    assert snap.questions_runtime_fallback is not None
    assert snap.questions == []

    snap_bad_json = snapshot_from_map({"screening_questions": "not-json"})
    assert snap_bad_json.questions == []
    assert snap_bad_json.questions_runtime_fallback is not None
    assert "json" in snap_bad_json.questions_runtime_fallback.lower()

    from app.core.question_flow import default_questions_v2

    ok = snapshot_from_map(
        {"screening_questions": default_questions_v2()}
    )
    assert ok.questions_runtime_fallback is None
    assert len(ok.questions) > 0


def test_runtime_question_errors_empty():
    from app.core.question_flow import runtime_question_errors

    errors = runtime_question_errors([])
    assert len(errors) == 1
    assert "blocked" in errors[0].lower()


def test_runtime_question_errors_valid_defaults():
    from app.core.question_flow import default_questions_v2, runtime_question_errors

    assert runtime_question_errors(default_questions_v2()) == []


def test_runtime_question_errors_invalid():
    from app.core.question_flow import default_questions_v2, runtime_question_errors

    questions = default_questions_v2()
    for i, q in enumerate(questions):
        if q["state"] == "Q1_FULL_NAME":
            bad = dict(q)
            bad["extract_fields"] = ["renamed_name"]
            questions[i] = bad
            break
    errors = runtime_question_errors(questions)
    assert len(errors) == 1
    assert "primary field" in errors[0].lower()


def test_has_sufficient_extraction_requires_required_and_confirmed():
    from app.core.call_handler import _has_sufficient_extraction
    from app.core.question_flow import default_questions_v2

    questions = default_questions_v2()
    data = {"full_name": "Jane Doe", "has_pets": False, "has_eviction": False}
    assert not _has_sufficient_extraction(data, questions)

    data["contact_phone"] = "+15551234567"
    data["email"] = "j@test.com"
    still_sparse = dict(data)
    still_sparse.update(
        {
            "move_in_date": "2026-08-01",
            "occupants_count": 2,
            "monthly_income": 5000,
            "employer": "Acme",
        }
    )
    assert not _has_sufficient_extraction(
        still_sparse,
        questions,
        confirmed_fields={"full_name", "contact_phone", "email"},
    )


def test_new_language_question_has_options_and_field():
    from app.core.question_flow import new_language_question

    q = new_language_question(order=1)
    assert q["answer_type"] == "language_choice"
    assert q["extract_fields"] == ["preferred_language"]
    assert len(q["language_options"]) >= 2
    assert q["requires_confirmation"] is False
    assert q["scoring"]["enabled"] is False


def test_resolve_language_choice_uses_admin_aliases():
    from app.core.question_flow import new_language_question, resolve_language_choice

    q = new_language_question()
    assert resolve_language_choice("I'd like español please", q) == "es"
    assert resolve_language_choice("English please", q) == "en"
    assert resolve_language_choice("French", q) is None


def test_extract_fields_from_speech_language_choice():
    from app.core.question_flow import extract_fields_from_speech, new_language_question

    q = new_language_question()
    out = extract_fields_from_speech("Spanish", q)
    assert out["preferred_language"] == "es"


def test_validate_language_choice_rejects_scoring():
    from app.core.question_flow import new_language_question, validate_questions_for_save

    q = new_language_question()
    q["scoring"] = {
        "enabled": True,
        "max_points": 5,
        "rule_type": "any_answer",
        "pass_config": {},
    }
    with pytest.raises(ValueError, match="cannot affect score"):
        validate_questions_for_save([q])


def test_validate_only_one_active_language_choice():
    from app.core.question_flow import new_language_question, validate_questions_for_save

    q1 = new_language_question(order=1)
    q2 = dict(q1)
    q2["id"] = "Q0_LANGUAGE_OTHER"
    q2["state"] = "Q0_LANGUAGE_OTHER"
    with pytest.raises(ValueError, match="Only one language choice"):
        validate_questions_for_save([q1, q2])

