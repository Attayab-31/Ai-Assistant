"""Lightweight Celery broker + worker health checks for admin monitoring."""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)


def _check_celery_health_sync() -> dict:
    """Ping the broker and active workers (sync — run via ``asyncio.to_thread``)."""
    from app.core.celery_app import celery_app

    broker_ok = False
    broker_detail = "Not connected"
    try:
        conn = celery_app.connection()
        conn.ensure_connection(max_retries=1, timeout=2.0)
        broker_ok = True
        broker_detail = "Connected"
        conn.release()
    except Exception as exc:
        logger.debug("Celery broker check failed: %s", exc)
        broker_detail = "Unreachable"

    workers = 0
    worker_detail = "No workers responded"
    try:
        insp = celery_app.control.inspect(timeout=2.0)
        ping = insp.ping() or {}
        workers = len(ping)
        if workers:
            worker_detail = f"{workers} worker{'s' if workers != 1 else ''} online"
    except Exception as exc:
        logger.debug("Celery worker inspect failed: %s", exc)

    ok = broker_ok and workers > 0
    return {
        "ok": ok,
        "broker": broker_ok,
        "broker_detail": broker_detail,
        "workers": workers,
        "worker_detail": worker_detail,
        "detail": worker_detail if ok else (
            "Background tasks (emails, retention purge) need a Celery worker "
            "and beat scheduler — see README / render.yaml."
        ),
    }


async def check_celery_health() -> dict:
    """Non-blocking Celery health snapshot for admin APIs."""
    return await asyncio.to_thread(_check_celery_health_sync)
