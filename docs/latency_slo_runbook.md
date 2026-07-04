# Voice Latency SLO Runbook

This runbook defines practical production targets for PSTN/phone calls and the
alerts to watch when latency degrades.

## Quick benchmark (before/after)

Run a few test-console or PSTN calls, then:

```bash
python scripts/benchmark_latency.py --last 10 --label before
# deploy / tune
python scripts/benchmark_latency.py --last 10 --label after
```

Per-turn detail is stored in `calls.error_log.turn_traces` (turn_ms, llm_ms, tts_ms, ttfa_ms).

## Automatic alerts

When a call finishes with high latency or turn timeouts, an email is sent to the
configured landlord address (same as screening emails, requires
`email_notifications_enabled=true` and Resend configured).

## SLO targets (phone calls)

- Turn response gap (tenant stop speaking -> first AI audio): p95 < 1200 ms
- LLM latency per turn: p95 < 700 ms
- TTS latency per turn: p95 < 500 ms
- Turn timeout rate: < 2% of turns
- Fallback rate (any stage): < 5% of turns
- Barge-in suppression time: < 200 ms

## Operational thresholds

- Warning:
  - p95 turn gap > 1200 ms for 10 minutes
  - fallback rate > 5% for 10 minutes
  - turn timeout rate > 2% for 10 minutes
- Critical:
  - p95 turn gap > 1800 ms for 5 minutes
  - fallback rate > 12% for 5 minutes
  - turn timeout rate > 5% for 5 minutes

## What to check first

1. Provider health from Admin -> AI & Voice -> Check now.
2. Logs for:
   - `Turn timed out after 15s`
   - `Falling back to ...`
   - provider quota/429 errors
   - Redis permission/write errors
3. Confirm current active LLM/STT/TTS in admin matches expected.
4. Confirm Redis is read-write (not read-only credentials).

## Immediate mitigations

- Keep `auto_fallback_enabled=true`.
- Use a low-latency primary LLM (admin-controlled).
- Ensure backup providers have valid keys.
- Reduce LLM token caps if responses are too long.
- Keep endpointing tuned for your traffic profile.

