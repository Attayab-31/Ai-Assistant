"""Settings cache invalidation hook.

New calls load their settings snapshot via a short-TTL Redis cache (see
``app.core.call_settings.load_call_settings_snapshot``). This module is the
single place admin writes call to drop that cached snapshot so configuration
changes apply to new calls immediately instead of after the TTL.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


async def invalidate_settings_cache() -> None:
    """Call after any admin settings write."""
    from app.core.call_settings import CALL_SETTINGS_SNAPSHOT_KEY
    from app.core.redis_client import cache_delete

    await cache_delete(CALL_SETTINGS_SNAPSHOT_KEY)
    logger.debug("Settings cache invalidated")
