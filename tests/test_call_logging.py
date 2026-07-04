"""Tests for structured voice-call logging helpers."""

import logging

from app.core.call_logging import (
    Phase,
    VoiceTraceFilter,
    format_call_prefix,
    vinfo,
    voice_context_from_record,
)


def test_format_call_prefix_full():
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="hello",
        args=(),
        exc_info=None,
    )
    record.call_id = "test-abc123"
    record.phase = Phase.LLM_TRY
    record.service = "llm"
    record.provider = "gemini"
    assert format_call_prefix(record) == "[test-abc123 | llm:try | llm/gemini] "


def test_format_call_prefix_empty():
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="hello",
        args=(),
        exc_info=None,
    )
    assert format_call_prefix(record) == ""


def test_voice_trace_filter():
    filt = VoiceTraceFilter()
    plain = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="plain",
        args=(),
        exc_info=None,
    )
    voice = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="voice",
        args=(),
        exc_info=None,
    )
    voice.call_id = "test-x"
    assert filt.filter(plain) is False
    assert filt.filter(voice) is True


def test_vinfo_attaches_extra(caplog):
    logger = logging.getLogger("test.call_logging")
    session = type("S", (), {"call_id": "test-99", "current_state": "full_name"})()
    with caplog.at_level(logging.INFO):
        vinfo(
            logger,
            "Primary LLM attempt",
            session=session,
            phase=Phase.LLM_TRY,
            service="llm",
            provider="groq",
            timeout_s=5.5,
        )
    assert len(caplog.records) == 1
    rec = caplog.records[0]
    assert rec.call_id == "test-99"
    assert rec.phase == Phase.LLM_TRY
    assert rec.provider == "groq"
    ctx = voice_context_from_record(rec)
    assert ctx["state"] == "full_name"
    assert ctx["timeout_s"] == 5.5


def test_voice_context_from_record_for_json():
    record = logging.LogRecord(
        name="app.core.call_handler",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="LLM OK",
        args=(),
        exc_info=None,
    )
    record.call_id = "test-json"
    record.phase = Phase.LLM_OK
    record.service = "llm"
    record.provider = "groq"
    record.latency_ms = 420
    ctx = voice_context_from_record(record)
    payload = {"message": record.getMessage(), "voice": ctx}
    assert payload["voice"]["call_id"] == "test-json"
    assert payload["voice"]["phase"] == "llm:ok"
    assert payload["voice"]["latency_ms"] == 420
