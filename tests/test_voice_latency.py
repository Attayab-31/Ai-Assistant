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


def test_explicit_critical_threshold_overrides_profile_defaults():
    cfg = resolve_voice_latency(
        {
            "voice_latency_profile": "balanced",
            "latency_alert_turn_p95_crit_ms": 2200,
            "latency_alert_timeout_rate_crit_pct": 7.5,
        }
    )
    assert cfg["latency_alert_turn_p95_crit_ms"] == 2200
    assert cfg["latency_alert_timeout_rate_crit_pct"] == 7.5


def test_critical_thresholds_clamped_to_warning_floor():
    cfg = resolve_voice_latency(
        {
            "voice_latency_profile": "balanced",
            "latency_alert_turn_p95_ms": 1500,
            "latency_alert_turn_p95_crit_ms": 1200,
            "latency_alert_timeout_rate_pct": 3.0,
            "latency_alert_timeout_rate_crit_pct": 2.0,
        }
    )
    assert cfg["latency_alert_turn_p95_crit_ms"] >= cfg["latency_alert_turn_p95_ms"]
    assert cfg["latency_alert_timeout_rate_crit_pct"] >= cfg["latency_alert_timeout_rate_pct"]


def test_conversation_session_accepts_critical_latency_alert_fields():
    from app.core.conversation import ConversationSession

    session = ConversationSession(
        call_id="test-latency",
        phone_number="+15555550123",
        latency_alert_turn_p95_ms=1200,
        latency_alert_turn_p95_crit_ms=1800,
        latency_alert_timeout_rate_pct=2.0,
        latency_alert_timeout_rate_crit_pct=5.0,
    )
    assert session.latency_alert_turn_p95_crit_ms == 1800
    assert session.latency_alert_timeout_rate_crit_pct == 5.0


def test_profile_latency_alert_defaults_match_profile():
    from app.core.voice_latency import profile_latency_alert_defaults

    fast = profile_latency_alert_defaults("fast")
    assert fast["latency_alert_turn_p95_ms"] == 1000
    assert fast["latency_alert_timeout_rate_pct"] == 3.0
