"""Settings cache invalidation hook.

New calls load their settings snapshot via a short-TTL Redis cache (see
``app.core.call_settings.load_call_settings_snapshot``). This module is the
single place admin writes call to drop that cached snapshot so configuration
changes apply to new calls immediately instead of after the TTL.
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)

_MAX_ATTEMPTS = 3
_RETRY_DELAY_SECONDS = 0.15


async def _delete_snapshot_key(key: str) -> bool:
    """Delete the settings snapshot key; return True when Redis confirms removal."""
    from app.core.redis_client import get_redis

    r = get_redis()
    if r is None:
        return False
    try:
        deleted = await r.delete(key)
        if deleted > 0:
            return True
        # Key absent — nothing stale to clear.
        return not await r.exists(key)
    except Exception as exc:
        logger.debug("Settings cache delete failed: %s", exc)
        return False


async def invalidate_settings_cache() -> None:
    """Call after any admin settings write.

    Retries a few times so a transient Redis blip does not leave new calls on
    a stale snapshot for the full TTL (~30s).
    """
    from app.core.call_settings import CALL_SETTINGS_SNAPSHOT_KEY

    key = CALL_SETTINGS_SNAPSHOT_KEY
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        if await _delete_snapshot_key(key):
            logger.debug("Settings cache invalidated")
            return
        if attempt < _MAX_ATTEMPTS:
            await asyncio.sleep(_RETRY_DELAY_SECONDS * attempt)

    raise RuntimeError(
        f"Settings cache invalidation failed after {_MAX_ATTEMPTS} attempts"
    )
