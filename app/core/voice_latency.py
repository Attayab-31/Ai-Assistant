"""Admin voice-latency presets — frozen per call at session start."""

from __future__ import annotations

from typing import Any

LATENCY_PROFILES: dict[str, dict[str, float | int]] = {
    "fast": {
        "turn_timeout_seconds": 14,
        "llm_timeout_voice_seconds": 5.0,
        "deepgram_endpointing_ms": 700,
        # Utterance-end floor is 1000ms (Deepgram API); fast tuning uses shorter
        # endpointing + turn timeouts instead.
        "deepgram_utterance_end_ms": 1000,
        "latency_alert_turn_p95_ms": 1000,
        "latency_alert_timeout_rate_pct": 3.0,
    },
    "balanced": {
        "turn_timeout_seconds": 15,
        "llm_timeout_voice_seconds": 5.5,
        "deepgram_endpointing_ms": 900,
        "deepgram_utterance_end_ms": 1000,
        "latency_alert_turn_p95_ms": 1200,
        "latency_alert_timeout_rate_pct": 2.0,
    },
    "quality": {
        "turn_timeout_seconds": 20,
        "llm_timeout_voice_seconds": 7.0,
        "deepgram_endpointing_ms": 1200,
        "deepgram_utterance_end_ms": 1400,
        "latency_alert_turn_p95_ms": 1800,
        "latency_alert_timeout_rate_pct": 2.0,
    },
}

DEFAULT_VOICE_LATENCY_PROFILE = "balanced"

# Deepgram live API rejects utterance_end_ms < 1000 with HTTP 400.
DEEPGRAM_UTTERANCE_END_MIN_MS = 1000


def _clamp_utterance_end_ms(ms: int) -> int:
    try:
        value = int(ms)
    except (TypeError, ValueError):
        value = DEEPGRAM_UTTERANCE_END_MIN_MS
    return max(DEEPGRAM_UTTERANCE_END_MIN_MS, value)


def resolve_voice_latency(values: dict[str, Any] | None) -> dict[str, Any]:
    """Merge admin profile choice into concrete per-call latency numbers."""
    values = values or {}
    profile = str(values.get("voice_latency_profile") or DEFAULT_VOICE_LATENCY_PROFILE).lower()
    if profile not in LATENCY_PROFILES:
        profile = DEFAULT_VOICE_LATENCY_PROFILE
    cfg = dict(LATENCY_PROFILES[profile])
    cfg["voice_latency_profile"] = profile
    cfg["deepgram_utterance_end_ms"] = _clamp_utterance_end_ms(
        cfg.get("deepgram_utterance_end_ms", DEEPGRAM_UTTERANCE_END_MIN_MS)
    )
    raw_stream = values.get("llm_streaming_enabled", "true")
    cfg["llm_streaming_enabled"] = str(raw_stream).lower() in ("true", "1", "yes")
    return cfg
