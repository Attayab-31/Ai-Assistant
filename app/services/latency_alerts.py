"""Lightweight latency SLO checks and admin email alerts."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Mirrors docs/latency_slo_runbook.md — defaults; per-call admin profile overrides.
TURN_P95_WARN_MS = 1200
TURN_P95_CRIT_MS = 1800
TIMEOUT_RATE_WARN_PCT = 2.0
TIMEOUT_RATE_CRIT_PCT = 5.0


def _alert_thresholds(session) -> tuple[float, float, float, float]:
    turn_warn = float(getattr(session, "latency_alert_turn_p95_ms", TURN_P95_WARN_MS) or TURN_P95_WARN_MS)
    timeout_warn = float(
        getattr(session, "latency_alert_timeout_rate_pct", TIMEOUT_RATE_WARN_PCT)
        or TIMEOUT_RATE_WARN_PCT
    )
    turn_crit_cfg = getattr(session, "latency_alert_turn_p95_crit_ms", None)
    timeout_crit_cfg = getattr(session, "latency_alert_timeout_rate_crit_pct", None)
    turn_crit = (
        float(turn_crit_cfg)
        if turn_crit_cfg not in (None, "")
        else max(turn_warn * 1.5, TURN_P95_CRIT_MS)
    )
    timeout_crit = (
        float(timeout_crit_cfg)
        if timeout_crit_cfg not in (None, "")
        else max(timeout_warn * 2.5, TIMEOUT_RATE_CRIT_PCT)
    )
    turn_crit = max(turn_crit, turn_warn)
    timeout_crit = max(timeout_crit, timeout_warn)
    return turn_warn, turn_crit, timeout_warn, timeout_crit


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def evaluate_call_latency(session) -> list[str]:
    """Return human-readable warnings when a call breached latency SLOs."""
    traces = getattr(session, "turn_traces", None) or []
    if not traces:
        return []

    turn_ms = [
        float(t["turn_ms"]) for t in traces if t.get("turn_ms") is not None
    ]
    if not turn_ms:
        return []

    warnings: list[str] = []
    timeouts = sum(1 for t in traces if t.get("timed_out"))
    timeout_rate = 100.0 * timeouts / len(traces)
    turn_p95 = _percentile(turn_ms, 95) or 0.0
    max_turn = max(turn_ms)
    turn_warn, turn_crit, timeout_warn, timeout_crit = _alert_thresholds(session)

    if timeout_rate >= timeout_crit:
        warnings.append(
            f"Turn timeout rate {timeout_rate:.1f}% (critical ≥ {timeout_crit:.0f}%)"
        )
    elif timeout_rate >= timeout_warn:
        warnings.append(
            f"Turn timeout rate {timeout_rate:.1f}% (warning ≥ {timeout_warn:.0f}%)"
        )

    if turn_p95 >= turn_crit:
        warnings.append(
            f"Turn p95 {turn_p95:.0f}ms (critical ≥ {turn_crit:.0f}ms)"
        )
    elif turn_p95 >= turn_warn:
        warnings.append(
            f"Turn p95 {turn_p95:.0f}ms (warning ≥ {turn_warn:.0f}ms)"
        )

    if max_turn >= turn_crit:
        warnings.append(f"Slowest turn {max_turn:.0f}ms")

    return warnings


def queue_latency_alert_if_needed(
    session, *, call_id: str, email_settings: dict | None = None
) -> bool:
    """Email the landlord when a call breached latency thresholds.

    Returns True when a Celery task was queued, False when skipped or enqueue failed.
    """
    warnings = evaluate_call_latency(session)
    if not warnings:
        return False
    try:
        from app.services.email_service import send_latency_alert_task

        send_latency_alert_task.delay(
            call_id=call_id,
            warnings=warnings,
            turn_count=len(session.turn_traces or []),
            avg_turn_ms=getattr(session, "avg_turn_latency_ms", 0),
            max_turn_ms=int(round(getattr(session, "max_turn_latency_ms", 0))),
            llm_provider=getattr(session, "llm_provider", ""),
            tts_provider=getattr(session, "tts_provider", ""),
            email_settings=email_settings,
        )
        logger.warning(
            "[%s] Latency SLO breach — queued alert: %s",
            call_id,
            "; ".join(warnings),
        )
        return True
    except Exception as exc:
        logger.error("Failed to queue latency alert for %s: %s", call_id, exc)
        return False
