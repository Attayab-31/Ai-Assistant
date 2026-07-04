"""Tests for streaming JSON speakable extraction."""

from app.core.llm_streaming import drain_speakable_sentences, partial_json_string_value


def test_partial_json_string_value_incomplete():
    buf = '{"response_text": "Hello there.'
    assert partial_json_string_value(buf, "response_text") == "Hello there."


def test_drain_sentences_from_partial_json():
    buf = '{"response_text": "Thanks for calling. What is your name?'
    sentences, offset = drain_speakable_sentences(buf, 0)
    assert sentences == ["Thanks for calling."]
    assert offset > 0

    more, offset2 = drain_speakable_sentences(buf, offset)
    assert more == []


def test_drain_phrase_chunk_before_sentence_end():
    buf = '{"response_text": "Thanks for calling today we apprec'
    sentences, offset = drain_speakable_sentences(buf, 0)
    assert sentences == ["Thanks for calling today we"]
    assert offset > 0


def test_drain_skips_already_spoken():
    buf = '{"response_text": "Thanks for calling. Next question?"}'
    _, offset = drain_speakable_sentences(buf, 0)
    more, offset2 = drain_speakable_sentences(buf, offset)
    assert more == ["Next question?"]
    assert offset2 > offset


def test_drain_waits_on_abbreviation_mid_stream():
    buf = '{"response_text": "Please see Dr.'
    sentences, _ = drain_speakable_sentences(buf, 0)
    assert sentences == []

    closed = '{"response_text": "Please see Dr. Smith tomorrow."}'
    sentences2, _ = drain_speakable_sentences(closed, 0)
    assert sentences2 == ["Please see Dr. Smith tomorrow."]
