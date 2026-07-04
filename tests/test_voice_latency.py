"""Voice latency profile resolution."""

from app.core.voice_latency import LATENCY_PROFILES, resolve_voice_latency


def test_balanced_profile_defaults():
    cfg = resolve_voice_latency({})
    assert cfg["voice_latency_profile"] == "balanced"
    assert cfg["turn_timeout_seconds"] == 15
    assert cfg["llm_streaming_enabled"] is True


def test_fast_profile():
    cfg = resolve_voice_latency({"voice_latency_profile": "fast"})
    assert cfg["turn_timeout_seconds"] == LATENCY_PROFILES["fast"]["turn_timeout_seconds"]
    assert cfg["deepgram_endpointing_ms"] == 700
    assert cfg["deepgram_utterance_end_ms"] >= 1000


def test_utterance_end_clamped_to_deepgram_minimum():
    from app.core.voice_latency import DEEPGRAM_UTTERANCE_END_MIN_MS

    LATENCY_PROFILES["_test_low"] = {
        "turn_timeout_seconds": 12,
        "llm_timeout_voice_seconds": 4.5,
        "deepgram_endpointing_ms": 700,
        "deepgram_utterance_end_ms": 800,
        "latency_alert_turn_p95_ms": 1000,
        "latency_alert_timeout_rate_pct": 3.0,
    }
    try:
        cfg = resolve_voice_latency({"voice_latency_profile": "_test_low"})
        assert cfg["deepgram_utterance_end_ms"] == DEEPGRAM_UTTERANCE_END_MIN_MS
    finally:
        LATENCY_PROFILES.pop("_test_low", None)


def test_unknown_profile_falls_back():
    cfg = resolve_voice_latency({"voice_latency_profile": "turbo"})
    assert cfg["voice_latency_profile"] == "balanced"
