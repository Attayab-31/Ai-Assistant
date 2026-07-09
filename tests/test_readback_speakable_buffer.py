"""Read-back turns must not play a buffered LLM ack before the admin confirm prompt."""

from app.core.conversation import ConversationSession, build_correction_readback
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


def test_correction_readback_english_without_session():
    text = build_correction_readback(
        [{"field": "contact_phone", "value": "555-1234"}],
    )
    assert "Did I get that right" in text
    assert "Es correcto" not in text


def test_correction_readback_spanish_with_session():
    session = ConversationSession(
        call_id="t",
        phone_number="+1",
        call_language="es",
        questions=_questions(),
    )
    text = build_correction_readback(
        [{"field": "contact_phone", "value": "555-1234"}],
        session=session,
    )
    assert "Es correcto" in text
    assert "Did I get that right" not in text


def test_reopen_question_after_failed_confirmation_unskips_admin_step():
    """Repair must honor admin read-back config by re-opening the question."""
    from app.core.question_flow import next_unanswered_state

    questions = [
        {
            "state": "CUSTOM_PHONE",
            "question": "What is your phone?",
            "required": False,
            "requires_confirmation": True,
            "extract_fields": ["contact_phone"],
            "active": True,
            "order": 1,
        },
        {
            "state": "Q_NEXT",
            "question": "Next question?",
            "required": True,
            "extract_fields": ["move_in_raw"],
            "active": True,
            "order": 2,
        },
    ]
    session = ConversationSession(
        call_id="repair",
        phone_number="+1",
        questions=questions,
        current_state="CUSTOM_PHONE",
    )
    session.extracted_data["contact_phone"] = "+15551234567"
    session.mark_completed("CUSTOM_PHONE")

    assert next_unanswered_state(
        session.extracted_data,
        session.skip_states,
        questions=session.questions,
        confirmed_fields=session.confirmed_fields,
    ) != "CUSTOM_PHONE"

    session.reopen_question_after_failed_confirmation(
        "CUSTOM_PHONE", "contact_phone"
    )

    assert "contact_phone" not in session.extracted_data
    assert "CUSTOM_PHONE" not in session.completed_states
    assert session.current_state == "CUSTOM_PHONE"
    assert next_unanswered_state(
        session.extracted_data,
        session.skip_states,
        questions=session.questions,
        confirmed_fields=session.confirmed_fields,
    ) == "CUSTOM_PHONE"


def _interrupt_session() -> ConversationSession:
    session = ConversationSession(
        call_id="barge",
        phone_number="+1",
        questions=_questions(),
        current_state="Q1_FULL_NAME",
    )
    # Simulate the start of a caller turn: user line committed up front.
    session.add_transcript("Tenant", "My name is John")
    session.add_message("user", "My name is John")
    return session


def test_reconcile_pairs_user_turn_when_agent_streamed_partial_speech():
    """Barge-in mid-stream: keep the utterance, pair it with what was spoken."""
    session = _interrupt_session()
    # Agent streamed a partial sentence before the caller barged in.
    session.append_streaming_ai_transcript("Great, and what's your")

    session.reconcile_interrupted_turn()

    # User turn is now paired with an assistant turn (no orphan user message).
    assert session.messages[-2] == {"role": "user", "content": "My name is John"}
    assert session.messages[-1]["role"] == "assistant"
    assert session.messages[-1]["content"] == "Great, and what's your"
    # Streaming state is cleared for the next turn.
    assert session.streaming_ai_open is False
    assert session.streamed_speakable_prefix == ""
    # Transcript keeps the tenant line and the finalized partial AI line.
    assert session.transcript[-2].speaker == "Tenant"
    assert session.transcript[-1].speaker == "AI"
    assert session.transcript[-1].text == "Great, and what's your"


def test_reconcile_drops_orphan_user_turn_when_agent_said_nothing():
    """Barge-in before any agent speech: drop the unanswered user turn."""
    session = _interrupt_session()

    session.reconcile_interrupted_turn()

    # No dangling user message remains in the LLM history.
    assert all(
        not (m["role"] == "user" and m["content"] == "My name is John")
        for m in session.messages
    )
    # Tenant transcript line is rolled back too (nothing was answered).
    assert all(entry.speaker != "Tenant" for entry in session.transcript)
    assert session.streaming_ai_open is False


def test_reconcile_never_fabricates_assistant_speech():
    """When nothing was streamed, no assistant message is invented."""
    session = _interrupt_session()

    session.reconcile_interrupted_turn()

    assert all(m["role"] != "assistant" for m in session.messages)
