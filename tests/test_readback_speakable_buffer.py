"""Read-back turns must not play a buffered LLM ack before the admin confirm prompt."""

from app.core.conversation import ConversationSession
from app.core.question_flow import needs_readback_confirmation


def _questions():
    return [
        {
            "state": "Q1_FULL_NAME",
            "question": "Can I start with your full name?",
            "requires_confirmation": True,
            "extract_fields": ["full_name"],
            "active": True,
            "order": 1,
        },
        {
            "state": "Q2_PHONE",
            "question": "What is the best phone number for you?",
            "requires_confirmation": True,
            "extract_fields": ["contact_phone"],
            "active": True,
            "order": 2,
        },
    ]


def test_needs_readback_after_name_extracted():
    session = ConversationSession(
        call_id="t",
        phone_number="+1",
        questions=_questions(),
        current_state="Q1_FULL_NAME",
    )
    session.extracted_data["full_name"] = "John Smith"
    assert needs_readback_confirmation(
        "Q1_FULL_NAME",
        session.extracted_data,
        session.questions,
        session.confirmed_fields,
    )


def test_no_readback_when_field_already_confirmed():
    session = ConversationSession(
        call_id="t",
        phone_number="+1",
        questions=_questions(),
        current_state="Q1_FULL_NAME",
    )
    session.extracted_data["full_name"] = "John Smith"
    session.confirmed_fields.add("full_name")
    assert not needs_readback_confirmation(
        "Q1_FULL_NAME",
        session.extracted_data,
        session.questions,
        session.confirmed_fields,
    )
