"""LLM fallback health hints — time-windowed skip and probe retry."""

import time

import pytest

from app.core import call_handler


@pytest.fixture(autouse=True)
def _clear_hints():
    call_handler._llm_health_hints.clear()
    yield
    call_handler._llm_health_hints.clear()


def test_success_clears_recent_failures():
    call_handler._llm_health_hints["groq"] = {
        "fail": 3.0,
        "last_fail_at": time.time(),
    }
    call_handler._record_llm_health_success("groq", 120.0)
    hint = call_handler._llm_health_hints["groq"]
    assert hint["fail"] == 0.0
    assert hint["ok"] == 1.0
    assert hint["latency_ms"] == 120.0


def test_unhealthy_skipped_during_cooldown():
    now = time.time()
    call_handler._llm_health_hints["openai"] = {
        "fail": 3.0,
        "ok": 0.0,
        "last_fail_at": now,
    }
    assert call_handler._llm_provider_healthy_for_fallback("openai") is False


def test_probe_allowed_after_cooldown(monkeypatch):
    now = time.time()
    call_handler._llm_health_hints["openai"] = {
        "fail": 3.0,
        "ok": 0.0,
        "last_fail_at": now - call_handler.LLM_HEALTH_PROBE_COOLDOWN_S - 1,
    }
    assert call_handler._llm_provider_healthy_for_fallback("openai") is True


def test_stale_failures_decay_outside_window():
    now = time.time()
    call_handler._llm_health_hints["gemini"] = {
        "fail": 5.0,
        "ok": 0.0,
        "last_fail_at": now - call_handler.LLM_HEALTH_WINDOW_S - 1,
    }
    assert call_handler._llm_provider_healthy_for_fallback("gemini") is True
    assert call_handler._llm_health_hints["gemini"]["fail"] == 0.0


def test_rank_prefers_higher_success_ratio():
    call_handler._llm_health_hints["groq"] = {
        "ok": 5.0,
        "fail": 0.0,
        "last_ok_at": time.time(),
        "latency_ms": 200.0,
    }
    call_handler._llm_health_hints["openai"] = {
        "ok": 1.0,
        "fail": 4.0,
        "last_ok_at": time.time(),
        "latency_ms": 100.0,
    }
    assert call_handler._llm_health_rank("groq")[0] < call_handler._llm_health_rank("openai")[0]


def test_health_clock_uses_wall_clock_not_monotonic():
    # Wall-clock so timestamps written to Redis are comparable across workers.
    before = time.time()
    value = call_handler._llm_health_now()
    after = time.time()
    assert before <= value <= after


def test_recorded_timestamps_are_wall_clock_comparable():
    # A timestamp recorded here must be usable as a cross-worker "now" reference:
    # a hint written "now" is not treated as stale by a fresh wall-clock read.
    call_handler._record_llm_health_failure("groq")
    last_fail = call_handler._llm_health_hints["groq"]["last_fail_at"]
    assert abs(time.time() - last_fail) < call_handler.LLM_HEALTH_WINDOW_S
