"""Persist post-call side-effect failures for admin visibility."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import uuid as uuid_module

logger = logging.getLogger(__name__)

_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=2,
    thread_name_prefix="side_effect_db",
)


async def _resolve_call(db, call_id: str):
    from app.db.crud import get_call_by_call_id, get_call_by_uuid

    call = None
    try:
        call = await get_call_by_uuid(db, uuid_module.UUID(call_id))
    except (TypeError, ValueError):
        pass
    if call is None:
        call = await get_call_by_call_id(db, call_id)
    return call


async def record_side_effect_queue_failure(
    db,
    call_id: str,
    kind: str,
    detail: str,
) -> None:
    """Record Celery enqueue failures (email_queue / crm_queue) on the call row."""
    from app.db.crud import merge_call_error_log

    call = await _resolve_call(db, call_id)
    if call is None:
        return
    await merge_call_error_log(db, call.call_id, {kind: detail}, commit=True)


async def record_side_effect_delivery_failure(
    call_id: str,
    *,
    key: str,
    detail: str,
) -> None:
    """Record permanent delivery failures under error_log.side_effect_failures."""
    from app.db.crud import merge_call_error_log
    from app.db.database import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        call = await _resolve_call(db, call_id)
        if call is None:
            return
        failures = dict(getattr(call, "error_log", {}) or {}).get("side_effect_failures") or {}
        failures = dict(failures)
        failures[key] = detail
        await merge_call_error_log(
            db,
            call.call_id,
            {"side_effect_failures": failures},
            commit=True,
        )


def run_side_effect_db_write(coro) -> None:
    """Run async DB persistence from sync Celery tasks or async request handlers."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        try:
            asyncio.run(coro)
        except Exception as exc:
            logger.warning("Side-effect DB write failed: %s", exc)
        return

    future = _executor.submit(asyncio.run, coro)
    try:
        future.result(timeout=10)
    except Exception as exc:
        logger.warning("Side-effect DB write failed: %s", exc)


def record_side_effect_delivery_failure_sync(
    call_id: str,
    *,
    key: str,
    detail: str,
) -> None:
    """Sync entry point for Celery workers."""
    run_side_effect_db_write(
        record_side_effect_delivery_failure(call_id, key=key, detail=detail)
    )
