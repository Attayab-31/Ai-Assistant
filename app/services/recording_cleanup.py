"""
Recording removal orchestration — retries, result checking, orphan queue.

Call sites must not clear DB pointers or delete call rows until managed storage
objects are removed, except when the pointer is an external URL we do not own.
Failed deletes are queued in Redis for the daily retention sweep to retry.
"""

from __future__ import annotations

import asyncio
import logging
from enum import Enum

logger = logging.getLogger(__name__)

PENDING_DELETES_KEY = "storage:pending_recording_deletes"
PENDING_DELETES_TTL_SECONDS = 90 * 86400
DEFAULT_DELETE_RETRIES = 3


class RecordingRemovalResult(str, Enum):
    """Outcome of attempting to remove a stored recording."""

    REMOVED = "removed"
    EXTERNAL = "external"
    NOTHING = "nothing"
    FAILED = "failed"


async def recording_pointer_safe_to_drop(recording_url: str | None) -> bool:
    """True when a call row can be deleted without losing a managed recording."""
    if not recording_url or not str(recording_url).strip():
        return True
    result = await remove_recording(recording_url)
    if removal_ok_for_db_clear(result):
        return True
    if result == RecordingRemovalResult.FAILED:
        return await enqueue_orphaned_recording(recording_url)
    return False


def is_managed_recording_path(object_path: str | None) -> bool:
    """True for Supabase object paths we should delete (not third-party URLs)."""
    if not object_path or not str(object_path).strip():
        return False
    return not str(object_path).strip().startswith(("http://", "https://"))


def removal_ok_for_db_clear(result: RecordingRemovalResult) -> bool:
    """True when it is safe to drop the DB pointer to this recording."""
    return result in (
        RecordingRemovalResult.REMOVED,
        RecordingRemovalResult.EXTERNAL,
        RecordingRemovalResult.NOTHING,
    )


async def remove_recording(
    object_path: str | None,
    *,
    retries: int = DEFAULT_DELETE_RETRIES,
) -> RecordingRemovalResult:
    """Delete a recording from managed storage, with bounded retries."""
    if not object_path or not str(object_path).strip():
        return RecordingRemovalResult.NOTHING
    path = str(object_path).strip()
    if not is_managed_recording_path(path):
        return RecordingRemovalResult.EXTERNAL

    from app.services.storage_service import storage_service

    delay = 0.5
    for attempt in range(max(1, retries)):
        if attempt > 0:
            await asyncio.sleep(delay)
            delay = min(delay * 2, 4.0)
        if await storage_service.delete_recording(path):
            return RecordingRemovalResult.REMOVED

    logger.warning("Recording delete failed after %s attempts: %s", retries, path)
    return RecordingRemovalResult.FAILED


async def enqueue_orphaned_recording(object_path: str) -> bool:
    """Queue a storage path for retry when the DB pointer is already gone.

    Returns True when the path was persisted to the orphan queue.
    """
    path = str(object_path or "").strip()
    if not is_managed_recording_path(path):
        return True
    from app.core.redis_client import get_redis

    r = get_redis()
    if r is None:
        logger.error(
            "Cannot queue orphaned recording delete (Redis unavailable): %s", path
        )
        return False
    try:
        await r.sadd(PENDING_DELETES_KEY, path)
        await r.expire(PENDING_DELETES_KEY, PENDING_DELETES_TTL_SECONDS)
        logger.info("Queued orphaned recording for retry: %s", path)
        return True
    except Exception as e:
        logger.error("Failed to queue orphaned recording %s: %s", path, e)
        return False


async def retry_pending_recording_deletes(*, limit: int = 200) -> dict[str, int]:
    """Retry storage deletes queued after DB rows were removed."""
    from app.core.redis_client import get_redis

    summary = {"retried": 0, "removed": 0, "remaining": 0}
    r = get_redis()
    if r is None:
        return summary
    try:
        paths = list(await r.smembers(PENDING_DELETES_KEY))
    except Exception as e:
        logger.error("Failed to read pending recording delete queue: %s", e)
        return summary

    for path in paths[:limit]:
        summary["retried"] += 1
        result = await remove_recording(path, retries=2)
        if result == RecordingRemovalResult.REMOVED:
            try:
                await r.srem(PENDING_DELETES_KEY, path)
            except Exception as e:
                logger.debug("srem pending delete %s failed: %s", path, e)
            summary["removed"] += 1

    try:
        summary["remaining"] = int(await r.scard(PENDING_DELETES_KEY))
    except Exception:
        pass
    if summary["retried"]:
        try:
            await r.expire(PENDING_DELETES_KEY, PENDING_DELETES_TTL_SECONDS)
        except Exception as e:
            logger.debug("Could not refresh orphan delete queue TTL: %s", e)
    if summary["removed"]:
        logger.info("Pending recording delete retry removed %s object(s)", summary["removed"])
    return summary
