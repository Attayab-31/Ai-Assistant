"""
Summarize voice-call latency from persisted call rows and per-turn traces.

Usage:
  python scripts/benchmark_latency.py
  python scripts/benchmark_latency.py --call-id test-abc123
  python scripts/benchmark_latency.py --last 20 --label after-overlap
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select

from app.db.database import AsyncSessionLocal, engine
from app.models.call import Call


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


def _fmt_ms(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.0f}ms"


def _summarize_turn_traces(traces: list[dict]) -> dict:
    turn_ms = [float(t["turn_ms"]) for t in traces if t.get("turn_ms") is not None]
    llm_ms = [float(t["llm_ms"]) for t in traces if t.get("llm_ms") is not None]
    tts_ms = [float(t["tts_ms"]) for t in traces if t.get("tts_ms") is not None]
    ttfa_ms = [float(t["ttfa_ms"]) for t in traces if t.get("ttfa_ms") is not None]
    timeouts = sum(1 for t in traces if t.get("timed_out"))
    streamed = sum(1 for t in traces if t.get("streamed"))

    return {
        "turns": len(traces),
        "timeouts": timeouts,
        "timeout_rate_pct": round(100 * timeouts / len(traces), 1) if traces else 0.0,
        "streamed_turns": streamed,
        "turn_p50": _percentile(turn_ms, 50),
        "turn_p95": _percentile(turn_ms, 95),
        "turn_p99": _percentile(turn_ms, 99),
        "llm_p50": _percentile(llm_ms, 50),
        "llm_p95": _percentile(llm_ms, 95),
        "tts_p50": _percentile(tts_ms, 50),
        "tts_p95": _percentile(tts_ms, 95),
        "ttfa_p50": _percentile(ttfa_ms, 50),
        "ttfa_p95": _percentile(ttfa_ms, 95),
    }


async def run(args: argparse.Namespace) -> int:
    async with AsyncSessionLocal() as db:
        stmt = select(Call).order_by(Call.started_at.desc())
        if args.call_id:
            stmt = stmt.where(Call.call_id == args.call_id)
        else:
            stmt = stmt.limit(args.last)

        result = await db.execute(stmt)
        calls = list(result.scalars().all())

    if not calls:
        print("No calls found.")
        return 1

    all_traces: list[dict] = []
    call_rows: list[dict] = []

    for call in calls:
        error_log = call.error_log or {}
        traces = error_log.get("turn_traces") or []
        all_traces.extend(traces)
        call_rows.append(
            {
                "call_id": call.call_id,
                "status": call.status,
                "turn_count": call.turn_count or 0,
                "avg_turn_ms": call.avg_turn_ms,
                "max_turn_ms": call.max_turn_ms,
                "avg_llm_ms": call.avg_llm_ms,
                "avg_tts_ms": call.avg_tts_ms,
                "trace_turns": len(traces),
                "llm": call.llm_provider,
                "tts": call.tts_provider,
            }
        )

    summary = _summarize_turn_traces(all_traces)
    label = f" ({args.label})" if args.label else ""

    print(f"Latency benchmark{label}")
    print(f"Calls analyzed: {len(calls)}")
    print()
    print("Per-turn traces (from error_log.turn_traces):")
    print(f"  turns:          {summary['turns']}")
    print(f"  timeouts:       {summary['timeouts']} ({summary['timeout_rate_pct']}%)")
    print(f"  streamed turns: {summary['streamed_turns']}")
    print(f"  turn p50/p95/p99: {_fmt_ms(summary['turn_p50'])} / {_fmt_ms(summary['turn_p95'])} / {_fmt_ms(summary['turn_p99'])}")
    print(f"  llm  p50/p95:     {_fmt_ms(summary['llm_p50'])} / {_fmt_ms(summary['llm_p95'])}")
    print(f"  tts  p50/p95:     {_fmt_ms(summary['tts_p50'])} / {_fmt_ms(summary['tts_p95'])}")
    print(f"  ttfa p50/p95:     {_fmt_ms(summary['ttfa_p50'])} / {_fmt_ms(summary['ttfa_p95'])}")
    print()
    print("Calls:")
    for row in call_rows:
        print(
            f"  {row['call_id']}  status={row['status']}  "
            f"turns={row['turn_count']}  avg_turn={row['avg_turn_ms']}ms  "
            f"max_turn={row['max_turn_ms']}ms  traces={row['trace_turns']}  "
            f"llm={row['llm']}  tts={row['tts']}"
        )

    if args.json:
        print()
        print(json.dumps({"summary": summary, "calls": call_rows}, indent=2))

    await engine.dispose()
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark voice call latency")
    parser.add_argument("--call-id", help="Single call_id to analyze")
    parser.add_argument("--last", type=int, default=10, help="Recent calls (default 10)")
    parser.add_argument("--label", help="Optional label for before/after comparisons")
    parser.add_argument("--json", action="store_true", help="Also print JSON summary")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
