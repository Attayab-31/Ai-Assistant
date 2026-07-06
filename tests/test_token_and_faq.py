"""Tests for LLM token accounting and token-aware FAQ injection."""

from app.core.conversation import _faq_topic_index, _select_faq_block
from app.providers.base import usage_from_response


class _FakeUsage:
    def __init__(self, prompt, completion, total=None):
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = total if total is not None else prompt + completion


class _FakeResponse:
    def __init__(self, usage):
        self.usage = usage


def test_usage_from_response_parses_counts():
    out = usage_from_response(_FakeResponse(_FakeUsage(100, 20)))
    assert out == {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
    }


def test_usage_from_response_handles_missing():
    assert usage_from_response(_FakeResponse(None)) is None
    assert usage_from_response(object()) is None
    # All-zero usage is treated as "unknown".
    assert usage_from_response(_FakeResponse(_FakeUsage(0, 0, 0))) is None


_FAQS = [
    {
        "topic": "pet_policy",
        "title": "Pet policy",
        "pattern": r"\b(pet|pets|dog|cat)\b",
        "answer": "Some properties are pet friendly.",
        "active": True,
    },
    {
        "topic": "application_fee",
        "title": "Application fee",
        "pattern": r"\b(fee|cost to apply)\b",
        "answer": "Our application fee is around fifty dollars.",
        "active": True,
    },
]


def test_faq_plain_answer_turn_uses_compact_index():
    # A normal screening answer (no question) → cheap topic index, not answers.
    block, is_full = _select_faq_block(_FAQS, "My name is John Smith")
    assert is_full is False
    assert "fifty dollars" not in block
    assert "pet_policy" in block


def test_faq_pattern_match_includes_only_matched_answer():
    block, is_full = _select_faq_block(_FAQS, "Do you allow pets?")
    assert is_full is True
    assert "pet friendly" in block
    # The unrelated FAQ answer is not embedded.
    assert "fifty dollars" not in block


def test_faq_unmatched_question_includes_relevant_subset():
    # Looks like a question but matches no pattern → top relevant FAQs, not all.
    block, is_full = _select_faq_block(_FAQS, "What is your policy on something?")
    assert is_full is True
    assert "pet friendly" in block or "fifty dollars" in block
    assert block.count('topic "') <= 3


def test_faq_unmatched_question_many_entries_caps_at_top_k():
    many = _FAQS + [
        {
            "topic": f"topic_{i}",
            "title": f"Title {i}",
            "pattern": rf"\\bunique{i}\\b",
            "answer": f"Answer number {i} about housing.",
            "active": True,
        }
        for i in range(8)
    ]
    block, is_full = _select_faq_block(many, "What about parking and fees?")
    assert is_full is True
    assert block.count('topic "') <= 3


def test_faq_topic_index_compact():
    idx = _faq_topic_index(_FAQS)
    assert "pet_policy" in idx
    assert "application_fee" in idx
    # No answer text in the index.
    assert "fifty dollars" not in idx


def test_session_records_token_usage():
    from app.core.conversation import ConversationSession

    s = ConversationSession(call_id="test-1", phone_number="+15555550100")
    s.record_llm_usage({"prompt_tokens": 100, "completion_tokens": 20})
    s.record_llm_usage({"prompt_tokens": 50, "completion_tokens": 10})
    s.record_llm_usage(None)  # a call with no reported usage still counts
    assert s.prompt_tokens == 150
    assert s.completion_tokens == 30
    assert s.total_tokens == 180
    assert s.llm_calls == 3


def test_session_records_latency_breakdown():
    from app.core.conversation import ConversationSession

    s = ConversationSession(call_id="test-2", phone_number="+15555550101")
    # Two turns; each turn made 1 LLM call and 2 TTS calls (ack + question).
    s.record_turn_latency(2000)
    s.record_llm_latency(800)
    s.record_tts_latency(500)
    s.record_tts_latency(400)
    s.record_turn_latency(3000)
    s.record_llm_latency(1000)
    s.record_tts_latency(600)
    s.record_tts_latency(300)

    # Per-stage averages use the TURN denominator so they stay additive:
    # llm 1800/2 = 900, tts 1800/2 = 900, turn 5000/2 = 2500.
    assert s.avg_llm_latency_ms == 900
    assert s.avg_tts_latency_ms == 900
    assert s.avg_turn_latency_ms == 2500
    # other = turn - llm - tts = 2500 - 900 - 900 = 700 (>= 0, additive).
    assert s.avg_turn_latency_ms - s.avg_llm_latency_ms - s.avg_tts_latency_ms == 700
    assert s.max_turn_latency_ms == 3000
    assert s.last_turn_latency_ms == 3000
    assert s.turn_latency_samples == 2


def test_session_latency_ignores_bad_values():
    from app.core.conversation import ConversationSession

    s = ConversationSession(call_id="test-3", phone_number="+15555550102")
    # None, zero, negative, and non-numeric must never raise or skew the math.
    for bad in (None, 0, -5, "oops"):
        s.record_turn_latency(bad)
        s.record_llm_latency(bad)
        s.record_tts_latency(bad)
    assert s.avg_turn_latency_ms == 0
    assert s.avg_llm_latency_ms == 0
    assert s.avg_tts_latency_ms == 0
    assert s.turn_latency_samples == 0


def test_session_to_dict_exposes_live_latency():
    from app.core.conversation import ConversationSession

    s = ConversationSession(call_id="test-4", phone_number="+15555550103")
    s.record_turn_latency(2400)
    d = s.to_dict()
    assert d["avg_turn_latency_ms"] == 2400
    assert d["last_turn_latency_ms"] == 2400
    assert "avg_llm_latency_ms" in d
    assert "avg_tts_latency_ms" in d
