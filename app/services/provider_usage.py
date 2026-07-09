"""Provider usage & balance monitoring for the admin Live Monitor.

Two complementary sources, because vendors expose very different things:

1. LIVE vendor balance/credits — only where the provider has a usable API:
   - OpenRouter: remaining credits via ``/api/v1/credits`` (or ``/api/v1/key``).
   - Deepgram: remaining project balance via ``/v1/projects/{id}/balances``.
   Groq, OpenAI, Gemini and Google do not expose a real-time "remaining quota" number,
   so those are reported honestly as "no usage API" rather than faked.

2. INTERNAL usage rollup — computed from our own ``calls`` table (always
   available, vendor-neutral): how many calls ran and how many minutes of audio
   we handled per provider over a window. This is the reliable signal for "how
   much are we using each API".

Everything here is best-effort and never raises into the request path: a vendor
outage or a missing key degrades to an "unavailable" card, never a 500. Vendor
lookups are cached in-process (the project's Redis user is read-only) so polling
the monitor never hammers billing APIs.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any

import httpx
from sqlalchemy import func, select

from app.models.call import Call
from config import settings

logger = logging.getLogger(__name__)

_VENDOR_TIMEOUT_S = 6.0
_CACHE_TTL_S = 120

# Process-local cache: {key: (expires_at, value)}. Used instead of Redis because
# the deployed Redis user is read-only and cannot persist writes.
_cache: dict[str, tuple[float, Any]] = {}


def _cache_get(key: str) -> Any | None:
    hit = _cache.get(key)
    if hit and hit[0] > time.monotonic():
        return hit[1]
    return None


def _cache_set(key: str, value: Any, ttl: int = _CACHE_TTL_S) -> None:
    _cache[key] = (time.monotonic() + ttl, value)


def _key_fingerprint(key: str) -> str:
    """Short, non-reversible fingerprint of an API key for cache scoping.

    Balance responses are per-credential, so the cache must be keyed by which
    key produced them. Otherwise rotating a key (or supplying a per-tenant key)
    would surface another key's cached balance until the TTL expired.
    """
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _money(amount: Any, currency: str = "USD") -> str | None:
    try:
        return f"${float(amount):,.2f}" if currency == "USD" else f"{float(amount):,.2f} {currency}"
    except (TypeError, ValueError):
        return None


async def _openrouter_balance(*, api_key: str | None = None) -> dict[str, Any]:
    """Remaining OpenRouter credits (best-effort)."""
    key = (api_key or settings.openrouter_api_key or "").strip()
    if not key:
        return {"available": False, "reason": "No API key configured"}

    cache_key = f"openrouter_balance:{_key_fingerprint(key)}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    headers = {"Authorization": f"Bearer {key}"}
    result: dict[str, Any] = {"available": False, "reason": "Could not reach OpenRouter"}
    try:
        async with httpx.AsyncClient(timeout=_VENDOR_TIMEOUT_S) as client:
            # Newer credits endpoint: { data: { total_credits, total_usage } }
            resp = await client.get(
                "https://openrouter.ai/api/v1/credits", headers=headers
            )
            if resp.status_code == 200:
                data = resp.json().get("data", {})
                total = data.get("total_credits")
                used = data.get("total_usage")
                if total is not None and used is not None:
                    remaining = float(total) - float(used)
                    result = {
                        "available": True,
                        "remaining_label": _money(remaining),
                        "used_label": _money(used),
                        "raw": {"total_credits": total, "total_usage": used},
                    }
            else:
                # Fallback: /key returns { data: { limit, usage, limit_remaining } }
                resp = await client.get(
                    "https://openrouter.ai/api/v1/key", headers=headers
                )
                if resp.status_code == 200:
                    data = resp.json().get("data", {})
                    remaining = data.get("limit_remaining")
                    used = data.get("usage")
                    result = {
                        "available": True,
                        "remaining_label": _money(remaining)
                        if remaining is not None
                        else "Unlimited",
                        "used_label": _money(used),
                        "raw": data,
                    }
    except Exception as e:
        logger.debug("OpenRouter balance lookup failed: %s", e)

    _cache_set(cache_key, result)
    return result


async def _deepgram_balance(*, api_key: str | None = None) -> dict[str, Any]:
    """Remaining Deepgram project balance (best-effort)."""
    key = (api_key or settings.deepgram_api_key or "").strip()
    if not key:
        return {"available": False, "reason": "No API key configured"}

    cache_key = f"deepgram_balance:{_key_fingerprint(key)}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    headers = {"Authorization": f"Token {key}"}
    result: dict[str, Any] = {
        "available": False,
        "reason": "Couldn't read balance right now.",
    }
    try:
        async with httpx.AsyncClient(timeout=_VENDOR_TIMEOUT_S) as client:
            proj_resp = await client.get(
                "https://api.deepgram.com/v1/projects", headers=headers
            )
            if proj_resp.status_code != 200:
                result = {
                    "available": False,
                    "reachable": False,
                    "reason": "Deepgram rejected this API key.",
                }
            else:
                projects = proj_resp.json().get("projects", [])
                if not projects:
                    result = {
                        "available": False,
                        "reachable": True,
                        "reason": "No Deepgram project found for this key.",
                    }
                else:
                    project_id = projects[0].get("project_id")
                    bal_resp = await client.get(
                        f"https://api.deepgram.com/v1/projects/{project_id}/balances",
                        headers=headers,
                    )
                    if bal_resp.status_code == 200:
                        balances = bal_resp.json().get("balances", [])
                        if balances:
                            amount = balances[0].get("amount")
                            units = balances[0].get("units", "usd")
                            result = {
                                "available": True,
                                "reachable": True,
                                "remaining_label": _money(amount)
                                if str(units).lower() == "usd"
                                else f"{amount} {units}",
                                "raw": balances[0],
                            }
                        else:
                            result = {
                                "available": False,
                                "reachable": True,
                                "reason": "Connected, but no prepaid balance on this account (pay-as-you-go).",
                            }
                    elif bal_resp.status_code in (401, 403):
                        # Deepgram works fine for calls; this key just can't read billing.
                        result = {
                            "available": False,
                            "reachable": True,
                            "reason": "Connected and working — but this API key can't read billing. To see your balance here, create a Deepgram key with the 'billing:read' scope.",
                        }
                    else:
                        result = {
                            "available": False,
                            "reachable": True,
                            "reason": "Connected, but balance is unavailable for this account.",
                        }
    except Exception as e:
        logger.debug("Deepgram balance lookup failed: %s", e)
        result = {
            "available": False,
            "reachable": False,
            "reason": "Couldn't reach Deepgram just now.",
        }

    _cache_set(cache_key, result)
    return result


async def get_internal_usage(db, days: int = 30) -> dict[str, Any]:
    """Vendor-neutral usage rollup computed from our own call records."""
    from datetime import UTC, datetime, timedelta

    since = datetime.now(UTC) - timedelta(days=days)

    totals = await db.execute(
        select(
            func.count(Call.id),
            func.coalesce(func.sum(Call.duration_seconds), 0),
            func.coalesce(func.sum(Call.prompt_tokens), 0),
            func.coalesce(func.sum(Call.completion_tokens), 0),
            func.coalesce(func.sum(Call.total_tokens), 0),
            func.coalesce(func.sum(Call.llm_calls), 0),
            # Latency: turn-weighted sums so we can derive true weighted averages.
            func.coalesce(func.sum(Call.avg_llm_ms * Call.turn_count), 0),
            func.coalesce(func.sum(Call.avg_tts_ms * Call.turn_count), 0),
            func.coalesce(func.sum(Call.avg_turn_ms * Call.turn_count), 0),
            func.coalesce(func.sum(Call.turn_count), 0),
            func.coalesce(func.max(Call.max_turn_ms), 0),
        ).where(Call.is_deleted == False, Call.created_at >= since)  # noqa: E712
    )
    (
        total_calls,
        total_seconds,
        prompt_tokens,
        completion_tokens,
        total_tokens,
        llm_calls,
        llm_ms_weighted,
        tts_ms_weighted,
        turn_ms_weighted,
        turn_samples,
        max_turn_ms,
    ) = totals.one()

    async def _by(column):
        rows = await db.execute(
            select(column, func.count(Call.id))
            .where(
                Call.is_deleted == False,  # noqa: E712
                Call.created_at >= since,
                column.isnot(None),
            )
            .group_by(column)
        )
        return {str(name): int(count) for name, count in rows.all()}

    calls = int(total_calls or 0)
    total_tok = int(total_tokens or 0)
    samples = int(turn_samples or 0)

    def _wavg(weighted) -> int:
        return round(int(weighted or 0) / samples) if samples else 0

    avg_llm = _wavg(llm_ms_weighted)
    avg_tts = _wavg(tts_ms_weighted)
    avg_turn = _wavg(turn_ms_weighted)
    # "Other" = full turn minus the two measured stages (STT-final assembly,
    # normalization, queueing). Clamped so rounding noise never goes negative.
    avg_other = max(0, avg_turn - avg_llm - avg_tts)
    return {
        "days": days,
        "total_calls": calls,
        "total_minutes": round(int(total_seconds or 0) / 60, 1),
        "by_llm": await _by(Call.llm_provider),
        "by_stt": await _by(Call.stt_provider),
        "by_tts": await _by(Call.tts_provider),
        "tokens": {
            "prompt": int(prompt_tokens or 0),
            "completion": int(completion_tokens or 0),
            "total": total_tok,
            "llm_calls": int(llm_calls or 0),
            "avg_per_call": round(total_tok / calls) if calls else 0,
        },
        "latency": {
            "turn_samples": samples,
            "avg_turn_ms": avg_turn,
            "avg_llm_ms": avg_llm,
            "avg_tts_ms": avg_tts,
            "avg_other_ms": avg_other,
            "max_turn_ms": int(max_turn_ms or 0),
            # Convenience percentages for the breakdown bar.
            "llm_pct": round(avg_llm / avg_turn * 100) if avg_turn else 0,
            "tts_pct": round(avg_tts / avg_turn * 100) if avg_turn else 0,
            "other_pct": round(avg_other / avg_turn * 100) if avg_turn else 0,
        },
    }


async def get_provider_overview(
    db, days: int = 30, *, status: dict | None = None
) -> dict[str, Any]:
    """Full provider monitoring payload: live balances + internal usage + config."""
    from app.core.call_settings import capture_provider_api_keys
    from config import provider_registry

    if status is None:
        status = provider_registry.get_status()
    keys = await capture_provider_api_keys(db)
    configured = {
        "groq": keys.configured("groq"),
        "openai": keys.configured("openai"),
        "openrouter": keys.configured("openrouter"),
        "gemini": keys.configured("gemini"),
        "deepgram": keys.configured("deepgram"),
    }
    openrouter = await _openrouter_balance(api_key=keys.openrouter)
    deepgram = await _deepgram_balance(api_key=keys.deepgram)
    usage = await get_internal_usage(db, days=days)

    # Per-vendor "remaining" cards. Only OpenRouter & Deepgram expose a number.
    no_api = {
        "available": False,
        "reason": "This provider has no real-time balance API. Track spend in the provider's own dashboard.",
    }
    providers = [
        {
            "key": "deepgram",
            "name": "Deepgram",
            "role": "Speech-to-text / Voice",
            "configured": configured["deepgram"],
            "balance": deepgram,
        },
        {
            "key": "openrouter",
            "name": "OpenRouter",
            "role": "AI brain (LLM)",
            "configured": configured["openrouter"],
            "balance": openrouter,
        },
        {
            "key": "groq",
            "name": "Groq",
            "role": "AI brain (LLM)",
            "configured": configured["groq"],
            "balance": no_api,
        },
        {
            "key": "openai",
            "name": "OpenAI",
            "role": "AI brain (LLM)",
            "configured": configured["openai"],
            "balance": no_api,
        },
        {
            "key": "gemini",
            "name": "Google Gemini",
            "role": "AI brain (LLM)",
            "configured": configured["gemini"],
            "balance": no_api,
        },
    ]

    return {
        "active": {
            "llm": status.get("llm"),
            "stt": status.get("stt"),
            "tts": status.get("tts"),
            "auto_fallback_enabled": status.get("auto_fallback_enabled"),
        },
        "providers": providers,
        "usage": usage,
    }
