"""Tests for dynamic screening question flow."""

import pytest

from app.core.question_flow import (
    default_questions_v2,
    migrate_questions_to_v2,
    new_custom_question,
    next_unanswered_state,
    normalize_questions,
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


def test_conditional_skip_pets_followup():
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
    assert "Extract ONLY these configured fields" in prompt
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
    assert "Active screening flow" in prompt
    assert "Do you smoke?" in prompt
    assert "Read-back confirmation fields" in prompt
    field = custom["extract_fields"][0]
    assert field in prompt
    assert "read-back confirm" in prompt


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

