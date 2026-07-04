"""
app/services/retention_service.py — automatic data-retention cleanup.

A single daily Celery beat task prunes data so the database (and recording
storage) can't grow without bound. Every window is admin-configurable via
Settings → General (keys ``retention_*``); a window of 0 disables that
particular sweep, and ``retention_enabled=false`` disables the whole job.
"""

import asyncio
import logging
from datetime import UTC, datetime, timedelta

from app.core.celery_app import celery_app

logger = logging.getLogger(__name__)

RECORDING_BATCH_SIZE = 200


def _to_int(value, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


async def _run_retention() -> dict:
    """Apply all configured retention windows. Returns a summary of counts."""
    from app.core.redis_client import acquire_once
    from app.db import crud
    from app.db.database import AsyncSessionLocal
    from app.services.storage_service import storage_service

    if not await acquire_once("retention:purge:lock", 7200):
        logger.info("Retention sweep already running — skipping")
        return {"enabled": False, "skipped": True}

    summary = {
        "soft_deleted_calls": 0,
        "recordings": 0,
        "calls": 0,
        "audit_logs": 0,
        "stale_calls_closed": 0,
    }

    async with AsyncSessionLocal() as db:
        settings_map = await crud.get_all_settings(db)

        enabled = str(settings_map.get("retention_enabled", "true")).lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if not enabled:
            logger.info("Retention disabled by admin setting — skipping")
            return {"enabled": False, **summary}

        now = datetime.now(UTC)
        soft_days = _to_int(settings_map.get("retention_soft_deleted_days"), 30)
        recording_days = _to_int(settings_map.get("retention_recording_days"), 90)
        calls_days = _to_int(settings_map.get("retention_calls_days"), 365)
        audit_days = _to_int(settings_map.get("retention_audit_days"), 365)
        stale_hours = _to_int(settings_map.get("retention_stale_call_hours"), 24)

        if stale_hours > 0:
            summary["stale_calls_closed"] = await crud.close_stale_calls(
                db, now - timedelta(hours=stale_hours)
            )

        if soft_days > 0:
            summary["soft_deleted_calls"] = await crud.purge_soft_deleted_calls_before(
                db, now - timedelta(days=soft_days)
            )

        if recording_days > 0:
            cutoff = now - timedelta(days=recording_days)
            while True:
                rows = await crud.get_recordings_before(
                    db, cutoff, limit=RECORDING_BATCH_SIZE
                )
                if not rows:
                    break
                for call_id, url in rows:
                    await storage_service.delete_recording(url)
                    await crud.clear_recording_url(db, call_id)
                summary["recordings"] += len(rows)
                if len(rows) < RECORDING_BATCH_SIZE:
                    break

        if calls_days > 0:
            summary["calls"] = await crud.purge_calls_before(
                db, now - timedelta(days=calls_days)
            )

        if audit_days > 0:
            summary["audit_logs"] = await crud.purge_audit_logs_before(
                db, now - timedelta(days=audit_days)
            )

    logger.info("Retention sweep complete: %s", summary)
    return {"enabled": True, **summary}


@celery_app.task(
    soft_time_limit=600,
    time_limit=900,
    name="app.services.retention_service.purge_expired_data_task",
)
def purge_expired_data_task():
    """Celery beat task: run the retention sweep once per day."""
    try:
        return asyncio.run(_run_retention())
    except Exception as e:  # pragma: no cover - defensive
        logger.error("Retention task failed: %s", e)
        return {"error": str(e)}
