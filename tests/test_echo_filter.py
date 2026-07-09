"""Agent echo filter must not drop valid short screening answers."""

from types import SimpleNamespace

import pytest

from app.core.conversation import ConversationSession, is_echo_of_agent


def _session_with_last_ai(text: str) -> ConversationSession:
    session = ConversationSession(
        call_id="test-echo",
        phone_number="+15550001",
        questions=[],
    )
    session.add_transcript("AI", text)
    return session


@pytest.mark.parametrize("answer", ["yes", "no", "ok", "sure", "correct"])
def test_short_screening_answers_not_echo_when_in_agent_question(answer: str):
    session = _session_with_last_ai(
        f"Do you have pets? Please say {answer} or no."
    )
    assert not is_echo_of_agent(answer, session)


def test_gratitude_still_filtered():
    session = _session_with_last_ai("What is your monthly income?")
    assert is_echo_of_agent("thank you", session)


def test_long_agent_phrase_repetition_still_echo():
    session = _session_with_last_ai(
        "Your move in date is March fifteenth twenty twenty six"
    )
    repeated = "your move in date is march fifteenth twenty twenty six"
    assert is_echo_of_agent(repeated, session)
