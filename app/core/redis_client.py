"""Shared, connection-pooled async Redis client.

One pooled client is reused process-wide so we never leak connections (every
call site previously opened its own client). All helpers degrade gracefully:
if Redis is unreachable they fail open (return None / allow), so a Redis hiccup
never takes down calls, logins, or webhooks.

Memory safety: every value written through here MUST carry a TTL. With short
TTLs the platform's Redis footprint stays in single-digit MB regardless of call
volume, which keeps us comfortably under small (e.g. 256 MB) Redis plans.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

import redis.asyncio as aioredis

from config import redis_url_connection_kwargs, settings

logger = logging.getLogger(__name__)

# Bound the pool so a burst of concurrent calls can never exhaust a small
# managed Redis plan's connection limit.
_MAX_CONNECTIONS = 20

_client: aioredis.Redis | None = None


def get_redis() -> aioredis.Redis | None:
    """Return the shared pooled client (lazy, never raises)."""
    global _client
    if _client is None:
        try:
            _client = aioredis.from_url(
                settings.redis_url,
                decode_responses=True,
                max_connections=_MAX_CONNECTIONS,
                socket_connect_timeout=2,
                socket_timeout=2,
                health_check_interval=30,
                retry_on_timeout=True,
                **redis_url_connection_kwargs(settings.redis_url),
            )
        except Exception as e:  # pragma: no cover - construction is lazy
            logger.warning("Redis client init failed: %s", e)
            return None
    return _client


async def close_redis() -> None:
    """Close the pool on shutdown."""
    global _client
    if _client is not None:
        try:
            await _client.aclose()
        except Exception as e:  # pragma: no cover
            logger.debug("Redis close error: %s", e)
        finally:
            _client = None


async def ping() -> bool:
    """True when Redis is reachable."""
    r = get_redis()
    if r is None:
        return False
    try:
        return bool(await r.ping())
    except Exception:
        return False


async def cache_get_json(key: str) -> Any | None:
    """Return a decoded JSON value, or None on miss/error."""
    r = get_redis()
    if r is None:
        return None
    try:
        raw = await r.get(key)
        return json.loads(raw) if raw else None
    except Exception as e:
        logger.debug("cache_get_json(%s) failed: %s", key, e)
        return None


async def cache_set_json(key: str, value: Any, ttl_seconds: int) -> None:
    """Store a JSON value with a mandatory TTL (keeps memory bounded)."""
    r = get_redis()
    if r is None:
        return
    try:
        await r.setex(key, ttl_seconds, json.dumps(value, default=str))
    except Exception as e:
        logger.debug("cache_set_json(%s) failed: %s", key, e)


async def cache_delete(*keys: str) -> None:
    """Delete one or more keys (no-op if Redis is down)."""
    r = get_redis()
    if r is None or not keys:
        return
    try:
        await r.delete(*keys)
    except Exception as e:
        logger.debug("cache_delete failed: %s", e)


async def acquire_once(
    key: str,
    ttl_seconds: int,
    *,
    fail_closed: bool = False,
    token: str = "1",
) -> bool:
    """Idempotency guard: True the first time, False if already seen.

    By default fails OPEN (returns True when Redis is down). Set
    ``fail_closed=True`` for production webhooks so duplicate deliveries are
    rejected rather than double-processed.

    Pass a unique ``token`` when the holder must release the key later via
    compare-and-delete (e.g. provider-switch serialization).
    """
    r = get_redis()
    if r is None:
        return not fail_closed
    try:
        return bool(await r.set(key, token, nx=True, ex=ttl_seconds))
    except Exception as e:
        logger.debug("acquire_once(%s) failed: %s", key, e)
        return not fail_closed


_ADMIN_DELETED_CALL_TTL_SECONDS = 7 * 24 * 3600


async def mark_call_admin_deleted(call_id: str) -> None:
    """Tombstone so a live stream cannot recreate an admin-deleted call row."""
    r = get_redis()
    if r is None:
        return
    try:
        await r.setex(
            f"call:admin_deleted:{call_id}",
            _ADMIN_DELETED_CALL_TTL_SECONDS,
            "1",
        )
    except Exception as e:
        logger.debug("mark_call_admin_deleted(%s) failed: %s", call_id, e)


async def is_call_admin_deleted(call_id: str) -> bool:
    r = get_redis()
    if r is None:
        return False
    try:
        return bool(await r.get(f"call:admin_deleted:{call_id}"))
    except Exception as e:
        logger.debug("is_call_admin_deleted(%s) failed: %s", call_id, e)
        return False


async def set_stream_stop_signal(call_id: str, ttl_seconds: int = 3600) -> None:
    """Signal a live audio stream to stop (cross-process hangup support)."""
    r = get_redis()
    if r is None:
        return
    try:
        await r.setex(f"stream:stop:{call_id}", ttl_seconds, "1")
    except Exception as e:
        logger.debug("set_stream_stop_signal(%s) failed: %s", call_id, e)


async def is_stream_stop_signaled(call_id: str) -> bool:
    """True when another process (or hangup webhook) requested stream stop."""
    r = get_redis()
    if r is None:
        return False
    try:
        return bool(await r.get(f"stream:stop:{call_id}"))
    except Exception as e:
        logger.debug("is_stream_stop_signaled(%s) failed: %s", call_id, e)
        return False


async def clear_stream_stop_signal(call_id: str) -> None:
    """Remove a stream-stop flag after the stream has shut down."""
    await cache_delete(f"stream:stop:{call_id}")


async def set_finalize_inflight(call_id: str, ttl_seconds: int = 60) -> None:
    """Mark that a worker is finalizing this call (cross-process hangup guard)."""
    r = get_redis()
    if r is None:
        return
    try:
        await r.setex(f"finalize:inflight:{call_id}", ttl_seconds, "1")
    except Exception as e:
        logger.debug("set_finalize_inflight(%s) failed: %s", call_id, e)


async def is_finalize_inflight(call_id: str) -> bool:
    """True when another worker is actively finalizing this call."""
    r = get_redis()
    if r is None:
        return False
    try:
        return bool(await r.get(f"finalize:inflight:{call_id}"))
    except Exception as e:
        logger.debug("is_finalize_inflight(%s) failed: %s", call_id, e)
        return False


async def clear_finalize_inflight(call_id: str) -> None:
    """Remove finalize-inflight marker after finalize completes."""
    await cache_delete(f"finalize:inflight:{call_id}")


async def revoke_token(token: str, ttl_seconds: int) -> bool:
    """Denylist a JWT until it expires (logout / forced sign-out)."""
    r = get_redis()
    if r is None or not token:
        return not settings.is_production
    digest = hashlib.sha256(token.encode()).hexdigest()
    try:
        await r.setex(f"auth:revoked:{digest}", max(1, ttl_seconds), "1")
        return True
    except Exception as e:
        logger.debug("revoke_token failed: %s", e)
        return not settings.is_production


async def is_token_revoked(token: str) -> bool:
    """True when the token was explicitly revoked before expiry."""
    r = get_redis()
    if r is None or not token:
        return bool(token) and settings.is_production
    digest = hashlib.sha256(token.encode()).hexdigest()
    try:
        return bool(await r.get(f"auth:revoked:{digest}"))
    except Exception as e:
        logger.debug("is_token_revoked failed: %s", e)
        return settings.is_production


DISPLAY_TIMEZONE_KEY = "settings:display_timezone"


async def publish_display_timezone(name: str | None) -> None:
    """Share display timezone across workers (Jinja filters are sync/per-process)."""
    from app.utils.helpers import set_display_timezone

    set_display_timezone(name)
    candidate = (name or "").strip()
    if not candidate:
        return
    r = get_redis()
    if r is None:
        return
    try:
        await r.set(DISPLAY_TIMEZONE_KEY, candidate)
    except Exception as e:
        logger.debug("publish_display_timezone failed: %s", e)


async def sync_display_timezone_from_redis() -> None:
    """Refresh in-process display timezone from Redis before rendering admin pages."""
    from app.utils.helpers import set_display_timezone

    r = get_redis()
    if r is None:
        return
    try:
        raw = await r.get(DISPLAY_TIMEZONE_KEY)
    except Exception as e:
        logger.debug("sync_display_timezone_from_redis failed: %s", e)
        return
    if not raw:
        return
    value = raw.decode() if isinstance(raw, bytes) else str(raw)
    set_display_timezone(value.strip())


MONITOR_SESSION_KEY_PREFIX = "monitor:session:"
MONITOR_SESSION_SET = "monitor:active_sessions"
MONITOR_SESSION_TTL_SECONDS = 300


async def upsert_monitor_session(call_id: str, payload: dict) -> None:
    """Publish a live-call summary for cross-worker Live Monitor aggregation."""
    cid = str(call_id or "").strip()
    if not cid:
        return
    r = get_redis()
    if r is None:
        return
    try:
        import json

        await r.setex(
            f"{MONITOR_SESSION_KEY_PREFIX}{cid}",
            MONITOR_SESSION_TTL_SECONDS,
            json.dumps(payload),
        )
        await r.sadd(MONITOR_SESSION_SET, cid)
        await r.expire(MONITOR_SESSION_SET, MONITOR_SESSION_TTL_SECONDS)
    except Exception as e:
        logger.debug("upsert_monitor_session(%s) failed: %s", cid, e)


async def remove_monitor_session(call_id: str) -> None:
    cid = str(call_id or "").strip()
    if not cid:
        return
    r = get_redis()
    if r is None:
        return
    try:
        await r.delete(f"{MONITOR_SESSION_KEY_PREFIX}{cid}")
        await r.srem(MONITOR_SESSION_SET, cid)
    except Exception as e:
        logger.debug("remove_monitor_session(%s) failed: %s", cid, e)


async def list_remote_monitor_sessions() -> list[dict]:
    """Read live-call summaries published by any worker."""
    r = get_redis()
    if r is None:
        return []
    try:
        import json

        call_ids = list(await r.smembers(MONITOR_SESSION_SET))
    except Exception as e:
        logger.debug("list_remote_monitor_sessions smembers failed: %s", e)
        return []

    sessions: list[dict] = []
    for cid in call_ids:
        key = f"{MONITOR_SESSION_KEY_PREFIX}{cid}"
        try:
            raw = await r.get(key)
        except Exception:
            raw = None
        if not raw:
            try:
                await r.srem(MONITOR_SESSION_SET, cid)
            except Exception:
                pass
            continue
        try:
            payload = json.loads(raw)
        except Exception:
            continue
        if isinstance(payload, dict) and payload.get("call_id"):
            sessions.append(payload)
    return sessions
