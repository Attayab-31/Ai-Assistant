"""
app/db/crud.py — All CRUD operations for the application.

Provides async functions for creating, reading, updating, and deleting
records across all models. Used by API routes and background tasks.
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import (
    and_,
    cast,
    delete,
    desc,
    extract,
    func,
    literal,
    literal_column,
    or_,
    select,
    update,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.redis_client import cache_get_json, cache_set_json
from app.models.audit_log import AuditLog
from app.models.call import Call
from app.models.settings import SystemSetting
from app.models.tenant import Tenant
from app.models.user import AdminUser
from app.utils.security import is_sensitive_setting_key, mask_email, mask_phone
from config import (
    DEFAULT_FAQS,
    DEFAULT_QUESTIONS,
    DEFAULT_SYSTEM_SETTINGS,
    ENV_BACKED_SYSTEM_SETTING_KEYS,
)

logger = logging.getLogger(__name__)

_SETTING_VALUE_TYPES = {
    item["key"]: item.get("value_type") or "string"
    for item in DEFAULT_SYSTEM_SETTINGS
}
_JSON_SETTING_KEYS = frozenset(
    {"screening_questions", "screening_faqs", "blacklisted_numbers"}
)


def _default_value_type_for_key(key: str) -> str:
    if key in _JSON_SETTING_KEYS:
        return "json"
    return _SETTING_VALUE_TYPES.get(key, "string")


# ──────────────────────────────────────────────────────────────────────────────
# Seeding
# ──────────────────────────────────────────────────────────────────────────────


async def seed_defaults(db: AsyncSession) -> None:
    """Seed default system settings and admin user if not present."""
    from app.utils.security import hash_password
    from config import settings

    default_settings = [
        *DEFAULT_SYSTEM_SETTINGS,
        {
            "key": "screening_questions",
            "value": json.dumps(DEFAULT_QUESTIONS),
            "value_type": "json",
            "description": "Tenant screening questions (ordered)",
            "is_sensitive": False,
        },
        {
            "key": "screening_faqs",
            "value": json.dumps(DEFAULT_FAQS),
            "value_type": "json",
            "description": "Approved FAQ answers for live calls (ordered)",
            "is_sensitive": False,
        },
    ]

    setting_keys = [item["key"] for item in default_settings]
    existing_keys = set(
        (
            await db.execute(
                select(SystemSetting.key).where(SystemSetting.key.in_(setting_keys))
            )
        ).scalars()
    )

    missing_settings = [
        SystemSetting(**setting_data)
        for setting_data in default_settings
        if setting_data["key"] not in existing_keys
    ]
    if missing_settings:
        db.add_all(missing_settings)

    env_backed_defaults = {
        setting_data["key"]: setting_data
        for setting_data in default_settings
        if setting_data["key"] in ENV_BACKED_SYSTEM_SETTING_KEYS
    }
    synced_settings = []
    if env_backed_defaults:
        existing_env_backed = (
            await db.execute(
                select(SystemSetting).where(
                    SystemSetting.key.in_(list(env_backed_defaults))
                )
            )
        ).scalars()
        for existing in existing_env_backed:
            default_setting = env_backed_defaults[existing.key]
            default_value = str(default_setting["value"])
            if existing.updated_by is None and existing.value != default_value:
                existing.value = default_value
                existing.value_type = default_setting.get(
                    "value_type", existing.value_type
                )
                existing.description = default_setting.get(
                    "description", existing.description
                )
                existing.is_sensitive = default_setting.get(
                    "is_sensitive", existing.is_sensitive
                )
                synced_settings.append(existing.key)

    # Seed default admin user
    existing_admin = await get_user_by_email(db, settings.admin_email)
    admin_created = False
    if not existing_admin:
        admin = AdminUser(
            email=settings.admin_email,
            hashed_password=hash_password(settings.admin_password),
            full_name="Super Admin",
            role="super_admin",
        )
        db.add(admin)
        admin_created = True
        logger.info("Created default admin: %s", mask_email(settings.admin_email))

    if not missing_settings and not admin_created and not synced_settings:
        logger.info("Default seed data already present")
    else:
        await db.commit()
        logger.info(
            "Seeded defaults: %s settings inserted, %s env-backed settings synced, "
            "admin_created=%s",
            len(missing_settings),
            len(synced_settings),
            admin_created,
        )

    await ensure_screening_questions_v2(db)
    await ensure_screening_questions_spanish_locales(db)


async def ensure_screening_questions_spanish_locales(db: AsyncSession) -> bool:
    """Backfill missing locales.es on built-in questions from seed defaults."""
    from app.core.question_flow import merge_seed_spanish_locales, normalize_questions
    from app.core.seed_data import load_seed_questions

    raw = await get_setting_value(db, "screening_questions", [])
    if not raw or not isinstance(raw, list):
        return False
    seed_by_state = {str(q["state"]): q for q in load_seed_questions()}
    merged, updated = merge_seed_spanish_locales(raw, seed_by_state)
    if not updated:
        return False
    normalized = normalize_questions(merged)
    await set_setting(db, "screening_questions", json.dumps(normalized))
    logger.info(
        "Backfilled Spanish locales on %d screening question(s)", updated
    )
    return True


async def ensure_screening_questions_v2(db: AsyncSession) -> bool:
    """Persist v1-shaped screening_questions JSON as schema v2 on startup."""
    from app.core.question_flow import is_v2_question, normalize_questions

    raw = await get_setting_value(db, "screening_questions", [])
    if not raw or not isinstance(raw, list):
        return False
    if all(is_v2_question(q) for q in raw):
        return False

    normalized = normalize_questions(raw)
    await set_setting(db, "screening_questions", json.dumps(normalized))
    logger.info(
        "Upgraded screening_questions to schema v2 (%d questions)", len(normalized)
    )
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Call CRUD
# ──────────────────────────────────────────────────────────────────────────────


async def create_call(
    db: AsyncSession, call_id: str, phone_number: str, *, commit: bool = True, **kwargs
) -> Call:
    """Create a new call record."""
    status = kwargs.pop("status", "initiated")
    call = Call(
        call_id=call_id,
        phone_number=phone_number,
        status=status,
        started_at=datetime.now(UTC),
        **kwargs,
    )
    db.add(call)
    if commit:
        await db.commit()
        await db.refresh(call)
    else:
        await db.flush()
    logger.info("Call created: %s from %s", call_id, mask_phone(phone_number))
    return call


async def get_call_by_call_id(db: AsyncSession, call_id: str) -> Call | None:
    """Get call by Telnyx call_control_id."""
    result = await db.execute(
        select(Call)
        .where(Call.call_id == call_id, Call.is_deleted == False)
        .options(selectinload(Call.tenant))
    )
    return result.scalar_one_or_none()


async def get_call_by_call_id_for_update(
    db: AsyncSession, call_id: str
) -> Call | None:
    """Get call by call_id with FOR UPDATE lock (RMW safety)."""
    result = await db.execute(
        select(Call)
        .where(Call.call_id == call_id, Call.is_deleted == False)
        .with_for_update()
    )
    return result.scalar_one_or_none()


async def wait_for_call_by_call_id(
    db: AsyncSession,
    call_id: str,
    *,
    timeout: float = 3.0,
    interval: float = 0.1,
) -> Call | None:
    """Poll until a call row exists or *timeout* elapses (webhook ordering races)."""
    deadline = time.monotonic() + timeout
    while True:
        call = await get_call_by_call_id(db, call_id)
        if call is not None:
            return call
        if time.monotonic() >= deadline:
            return None
        await asyncio.sleep(interval)


async def get_call_by_uuid(db: AsyncSession, call_uuid: uuid.UUID) -> Call | None:
    """Get call by internal UUID."""
    result = await db.execute(
        select(Call)
        .where(Call.id == call_uuid, Call.is_deleted == False)
        .options(selectinload(Call.tenant))
    )
    return result.scalar_one_or_none()


async def update_call(
    db: AsyncSession, call_id: str, *, commit: bool = True, **kwargs
) -> Call | None:
    """Update call record fields."""
    kwargs["updated_at"] = datetime.now(UTC)
    await db.execute(update(Call).where(Call.call_id == call_id).values(**kwargs))
    if commit:
        await db.commit()
        return await get_call_by_call_id(db, call_id)
    await db.flush()
    return await get_call_by_call_id(db, call_id)


async def update_call_if_active(
    db: AsyncSession, call_id: str, *, commit: bool = True, **kwargs
) -> bool:
    """Update a call only while it is still ``initiated`` or ``in_progress``.

    Returns True when a row was updated. Never overwrites ``completed``,
    ``failed``, or ``abandoned`` (race-safe vs late finalize / abandon).
    """
    kwargs["updated_at"] = datetime.now(UTC)
    result = await db.execute(
        update(Call)
        .where(
            Call.call_id == call_id,
            Call.is_deleted == False,
            Call.status.in_(("initiated", "in_progress")),
        )
        .values(**kwargs)
    )
    if commit:
        await db.commit()
    else:
        await db.flush()
    return (result.rowcount or 0) > 0


STREAM_STOP_DB_KEY = "stream_stop_requested_at"


def _error_log_merge_expr(patch: dict) -> Any:
    return func.coalesce(Call.error_log, cast("{}", JSONB)).op("||")(cast(patch, JSONB))


async def merge_call_error_log(
    db: AsyncSession,
    call_id: str,
    patch: dict,
    *,
    commit: bool = True,
) -> bool:
    """Atomically merge keys into a call's error_log JSON."""
    if not patch:
        return False
    result = await db.execute(
        update(Call)
        .where(Call.call_id == call_id, Call.is_deleted == False)
        .values(
            error_log=_error_log_merge_expr(patch),
            updated_at=datetime.now(UTC),
        )
    )
    if commit:
        await db.commit()
    else:
        await db.flush()
    return (result.rowcount or 0) > 0


async def persist_stream_stop_request(db: AsyncSession, call_id: str) -> bool:
    """Persist a cross-worker hangup stop on the call row (Redis fallback)."""
    call = await get_call_by_call_id(db, call_id)
    if call is None or call.status not in ("initiated", "in_progress"):
        return False
    log = call.error_log if isinstance(call.error_log, dict) else {}
    if log.get(STREAM_STOP_DB_KEY):
        return True
    return await merge_call_error_log(
        db,
        call_id,
        {STREAM_STOP_DB_KEY: datetime.now(UTC).isoformat()},
        commit=True,
    )


async def is_stream_stop_requested(db: AsyncSession, call_id: str) -> bool:
    """True when a hangup stop was persisted on the call row."""
    call = await get_call_by_call_id(db, call_id)
    if call is None:
        return False
    log = call.error_log if isinstance(call.error_log, dict) else {}
    return bool(log.get(STREAM_STOP_DB_KEY))


async def clear_stream_stop_request(db: AsyncSession, call_id: str) -> None:
    """Remove a persisted hangup stop after the media stream has shut down."""
    call = await get_call_by_call_id(db, call_id)
    if call is None:
        return
    log = call.error_log if isinstance(call.error_log, dict) else {}
    if STREAM_STOP_DB_KEY not in log:
        return
    from sqlalchemy import literal

    await db.execute(
        update(Call)
        .where(Call.call_id == call_id, Call.is_deleted == False)
        .values(
            error_log=func.coalesce(Call.error_log, cast("{}", JSONB)).op("-")(
                literal(STREAM_STOP_DB_KEY)
            ),
            updated_at=datetime.now(UTC),
        )
    )
    await db.commit()


async def merge_call_side_effect_failure(
    db: AsyncSession,
    call_id: str,
    key: str,
    detail: str,
    *,
    commit: bool = True,
) -> bool:
    """Atomically record one nested delivery failure on the call row."""
    if not key:
        return False
    path = literal_column(
        "ARRAY['side_effect_failures', :failure_key]::text[]"
    ).bindparams(failure_key=key)
    expr = func.jsonb_set(
        func.coalesce(Call.error_log, cast("{}", JSONB)),
        path,
        cast(json.dumps(detail), JSONB),
        True,
    )
    result = await db.execute(
        update(Call)
        .where(Call.call_id == call_id, Call.is_deleted == False)
        .values(
            error_log=expr,
            updated_at=datetime.now(UTC),
        )
    )
    if commit:
        await db.commit()
    else:
        await db.flush()
    return (result.rowcount or 0) > 0


async def mark_call_abandoned_if_active(
    db: AsyncSession,
    call_id: str,
    *,
    commit: bool = True,
    error_log: dict | None = None,
) -> bool:
    """Mark a call abandoned only when still ``initiated`` or ``in_progress``.

    Atomic compare-and-set: never overwrites ``completed``, ``failed``, or
    ``abandoned``. Returns True when a row was updated.
    """
    now = datetime.now(UTC)
    values: dict[str, Any] = {
        "status": "abandoned",
        "ended_at": now,
        "updated_at": now,
    }
    if error_log is not None:
        values["error_log"] = _error_log_merge_expr(error_log)
    result = await db.execute(
        update(Call)
        .where(
            Call.call_id == call_id,
            Call.is_deleted == False,
            Call.status.in_(("initiated", "in_progress")),
        )
        .values(**values)
    )
    if commit:
        await db.commit()
    else:
        await db.flush()
    return (result.rowcount or 0) > 0


async def list_calls(
    db: AsyncSession,
    page: int = 1,
    per_page: int = 20,
    status: str | None = None,
    qualification_status: str | None = None,
    phone_search: str | None = None,
    name_search: str | None = None,
    text_search: str | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    llm_provider: str | None = None,
) -> tuple[list[Call], int]:
    """List calls with filters and pagination. Returns (calls, total_count)."""
    query = (
        select(Call).where(Call.is_deleted == False).options(selectinload(Call.tenant))
    )

    if status:
        query = query.where(Call.status == status)
    if phone_search:
        query = query.where(Call.phone_number.ilike(f"%{phone_search}%"))
    if name_search:
        query = query.outerjoin(Tenant, Call.id == Tenant.call_id).where(
            or_(
                Tenant.full_name.ilike(f"%{name_search}%"),
                Tenant.email.ilike(f"%{name_search}%"),
            )
        )
    if text_search:
        query = query.outerjoin(Tenant, Call.id == Tenant.call_id).where(
            or_(
                Call.phone_number.ilike(f"%{text_search}%"),
                Tenant.full_name.ilike(f"%{text_search}%"),
                Tenant.email.ilike(f"%{text_search}%"),
            )
        )
    if date_from:
        query = query.where(Call.created_at >= date_from)
    if date_to:
        query = query.where(Call.created_at <= date_to)
    if llm_provider:
        query = query.where(Call.llm_provider == llm_provider)
    if qualification_status:
        query = query.join(Tenant).where(
            Tenant.qualification_status == qualification_status
        )

    count_query = select(func.count(Call.id)).where(Call.is_deleted == False)
    if status:
        count_query = count_query.where(Call.status == status)
    if phone_search:
        count_query = count_query.where(Call.phone_number.ilike(f"%{phone_search}%"))
    if name_search:
        count_query = count_query.outerjoin(Tenant, Call.id == Tenant.call_id).where(
            or_(
                Tenant.full_name.ilike(f"%{name_search}%"),
                Tenant.email.ilike(f"%{name_search}%"),
            )
        )
    if text_search:
        count_query = count_query.outerjoin(Tenant, Call.id == Tenant.call_id).where(
            or_(
                Call.phone_number.ilike(f"%{text_search}%"),
                Tenant.full_name.ilike(f"%{text_search}%"),
                Tenant.email.ilike(f"%{text_search}%"),
            )
        )
    if date_from:
        count_query = count_query.where(Call.created_at >= date_from)
    if date_to:
        count_query = count_query.where(Call.created_at <= date_to)
    if llm_provider:
        count_query = count_query.where(Call.llm_provider == llm_provider)
    if qualification_status:
        count_query = count_query.join(Tenant).where(
            Tenant.qualification_status == qualification_status
        )

    count_result = await db.execute(count_query)
    total = count_result.scalar() or 0

    query = query.order_by(desc(Call.created_at))
    query = query.offset((page - 1) * per_page).limit(per_page)

    result = await db.execute(query)
    return result.scalars().all(), total


async def hard_delete_call(db: AsyncSession, call_uuid: uuid.UUID) -> bool:
    """Permanently delete a call. The linked tenant row is removed by the
    ``ON DELETE CASCADE`` on ``tenants.call_id`` (see the initial migration)."""
    await db.execute(delete(Call).where(Call.id == call_uuid))
    await db.commit()
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Retention / cleanup (used by the daily Celery purge task)
# ──────────────────────────────────────────────────────────────────────────────


async def purge_calls_before(
    db: AsyncSession, cutoff: datetime, *, batch_size: int = 500
) -> int:
    """Hard-delete calls created before ``cutoff`` (CASCADE removes tenants).

    Any stored recording is removed from storage first so objects don't orphan
    when the DB row (the only pointer to them) is deleted — this holds even if
    ``retention_calls_days`` is shorter than ``retention_recording_days``."""
    from app.services.recording_cleanup import recording_pointer_safe_to_drop

    total = 0
    while True:
        rows = (
            await db.execute(
                select(Call.id, Call.recording_url)
                .where(
                    Call.created_at < cutoff,
                    Call.is_deleted == False,  # noqa: E712
                )
                .order_by(Call.created_at, Call.id)
                .limit(batch_size)
            )
        ).all()
        if not rows:
            break
        safe_ids: list[uuid.UUID] = []
        for call_id, recording_url in rows:
            if await recording_pointer_safe_to_drop(recording_url):
                safe_ids.append(call_id)
            else:
                logger.warning(
                    "Skipping call purge for %s — recording delete and orphan "
                    "queue both failed (%s)",
                    call_id,
                    recording_url,
                )
        if not safe_ids:
            break
        result = await db.execute(delete(Call).where(Call.id.in_(safe_ids)))
        await db.commit()
        total += result.rowcount or 0
        if len(rows) < batch_size:
            break
    return total


async def purge_soft_deleted_calls_before(
    db: AsyncSession, cutoff: datetime, *, batch_size: int = 500
) -> int:
    """Hard-delete legacy soft-deleted calls last touched before ``cutoff``.

    Removes each call's stored recording first so storage objects don't orphan
    when the row is deleted."""
    from app.services.recording_cleanup import recording_pointer_safe_to_drop

    total = 0
    while True:
        rows = (
            await db.execute(
                select(Call.id, Call.recording_url)
                .where(Call.is_deleted == True, Call.updated_at < cutoff)  # noqa: E712
                .order_by(Call.updated_at, Call.id)
                .limit(batch_size)
            )
        ).all()
        if not rows:
            break
        safe_ids: list[uuid.UUID] = []
        for call_id, recording_url in rows:
            if await recording_pointer_safe_to_drop(recording_url):
                safe_ids.append(call_id)
            else:
                logger.warning(
                    "Skipping soft-deleted call purge for %s — recording delete "
                    "and orphan queue both failed (%s)",
                    call_id,
                    recording_url,
                )
        if not safe_ids:
            break
        result = await db.execute(delete(Call).where(Call.id.in_(safe_ids)))
        await db.commit()
        total += result.rowcount or 0
        if len(rows) < batch_size:
            break
    return total


async def purge_audit_logs_before(
    db: AsyncSession, cutoff: datetime, *, batch_size: int = 500
) -> int:
    """Delete audit-log rows created before ``cutoff``."""
    total = 0
    while True:
        ids_result = await db.execute(
            select(AuditLog.id).where(AuditLog.created_at < cutoff).limit(batch_size)
        )
        ids = [row[0] for row in ids_result.all()]
        if not ids:
            break
        result = await db.execute(delete(AuditLog).where(AuditLog.id.in_(ids)))
        await db.commit()
        total += result.rowcount or 0
        if len(ids) < batch_size:
            break
    return total


async def close_stale_calls(
    db: AsyncSession, older_than: datetime, *, batch_size: int = 500
) -> int:
    """Mark calls stuck in initiated/in_progress as failed (zombie cleanup).

    Prevents rows from lingering forever when a webhook or stream dies without
    a hangup event. Does not delete — retention handles hard-delete by age.
    """
    total = 0
    stale_statuses = ("initiated", "in_progress")
    while True:
        ids_result = await db.execute(
            select(Call.id)
            .where(
                Call.status.in_(stale_statuses),
                Call.is_deleted == False,
                Call.created_at < older_than,
            )
            .limit(batch_size)
        )
        ids = [row[0] for row in ids_result.all()]
        if not ids:
            break
        now = datetime.now(UTC)
        result = await db.execute(
            update(Call)
            .where(Call.id.in_(ids))
            .values(
                status="failed",
                ended_at=now,
                updated_at=now,
                error_log=_error_log_merge_expr(
                    {"stale_cleanup": "Auto-closed after exceeding stale window"}
                ),
            )
        )
        await db.commit()
        total += result.rowcount or 0
        if len(ids) < batch_size:
            break
    return total


async def get_recordings_before(
    db: AsyncSession,
    cutoff: datetime,
    *,
    limit: int = 200,
    after_created_at: datetime | None = None,
    after_id: uuid.UUID | None = None,
) -> list[tuple[uuid.UUID, str, datetime]]:
    """Return (call_id, recording_url) for calls older than ``cutoff`` that
    still have a stored recording, so the storage object can be removed."""
    conditions = [Call.recording_url.isnot(None), Call.created_at < cutoff]
    if after_created_at is not None and after_id is not None:
        conditions.append(
            or_(
                Call.created_at > after_created_at,
                and_(Call.created_at == after_created_at, Call.id > after_id),
            )
        )

    result = await db.execute(
        select(Call.id, Call.recording_url, Call.created_at)
        .where(*conditions)
        .order_by(Call.created_at, Call.id)
        .limit(limit)
    )
    return [(row[0], row[1], row[2]) for row in result.all()]


async def clear_recording_url(db: AsyncSession, call_id: uuid.UUID) -> None:
    """Null out a call's recording_url after its storage object is deleted."""
    await db.execute(
        update(Call).where(Call.id == call_id).values(recording_url=None)
    )
    await db.commit()


async def get_call_stats(db: AsyncSession) -> dict:
    """Get aggregate call statistics for dashboard - OPTIMIZED: 2 queries instead of 7."""
    now = datetime.now(UTC)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = today_start - timedelta(days=today_start.weekday())
    month_start = today_start.replace(day=1)

    # Query 1: All call statistics in one aggregation using conditional filters
    call_result = await db.execute(
        select(
            func.count(Call.id).filter(Call.is_deleted == False).label("total"),
            func.count(Call.id)
            .filter(and_(Call.is_deleted == False, Call.created_at >= today_start))
            .label("today"),
            func.count(Call.id)
            .filter(and_(Call.is_deleted == False, Call.created_at >= week_start))
            .label("week"),
            func.count(Call.id)
            .filter(and_(Call.is_deleted == False, Call.created_at >= month_start))
            .label("month"),
            func.avg(Call.duration_seconds)
            .filter(and_(Call.is_deleted == False, Call.duration_seconds.isnot(None)))
            .label("avg_dur"),
        )
    )
    call_stats = call_result.one()

    # Query 2: Tenant qualification stats (requires JOIN, so separate from above)
    tenant_result = await db.execute(
        select(
            func.count(Tenant.id)
            .filter(Tenant.qualification_status == "qualified")
            .label("qualified"),
            func.count(Tenant.id)
            .filter(Tenant.qualification_status == "unqualified")
            .label("unqualified"),
        )
        .join(Call, Tenant.call_id == Call.id)
        .where(Call.is_deleted == False, Call.created_at >= month_start)
    )
    tenant_stats = tenant_result.one()

    return {
        "total_calls": call_stats.total or 0,
        "calls_today": call_stats.today or 0,
        "calls_this_week": call_stats.week or 0,
        "calls_this_month": call_stats.month or 0,
        "avg_duration_seconds": int(call_stats.avg_dur) if call_stats.avg_dur else 0,
        "qualified_this_month": tenant_stats.qualified or 0,
        "disqualified_this_month": tenant_stats.unqualified or 0,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Tenant CRUD
# ──────────────────────────────────────────────────────────────────────────────


async def create_tenant(
    db: AsyncSession,
    call_id: uuid.UUID,
    phone_number: str,
    *,
    commit: bool = True,
    **kwargs,
) -> Tenant:
    """Create or update a tenant record after a call."""
    tenant = Tenant(call_id=call_id, phone_number=phone_number, **kwargs)
    db.add(tenant)
    if commit:
        await db.commit()
        await db.refresh(tenant)
    else:
        await db.flush()
    return tenant


async def get_tenant_by_call(db: AsyncSession, call_uuid: uuid.UUID) -> Tenant | None:
    """Get tenant record for a specific call (excluding soft-deleted calls)."""
    result = await db.execute(
        select(Tenant)
        .join(Call, Tenant.call_id == Call.id)
        .where(Tenant.call_id == call_uuid, Call.is_deleted == False)
    )
    return result.scalar_one_or_none()


async def get_tenant_by_id(db: AsyncSession, tenant_id: uuid.UUID) -> Tenant | None:
    """Get a tenant by its internal UUID."""
    result = await db.execute(select(Tenant).where(Tenant.id == tenant_id))
    return result.scalar_one_or_none()


SIDE_EFFECTS_CLAIM_KEY = "side_effects_claimed"
SIDE_EFFECTS_CHANNELS_KEY = "side_effects_channels"
VALID_SIDE_EFFECT_CHANNELS = frozenset({"email", "crm", "latency"})


def _side_effect_channels_claimed(nd: dict) -> dict[str, str]:
    """Return channel -> claimed_at ISO timestamps from tenant normalized_data."""
    if nd.get(SIDE_EFFECTS_CLAIM_KEY) is True:
        ts = str(nd.get("side_effects_claimed_at") or "")
        return {channel: ts for channel in VALID_SIDE_EFFECT_CHANNELS}
    raw = nd.get(SIDE_EFFECTS_CHANNELS_KEY)
    if not isinstance(raw, dict):
        return {}
    return {
        channel: str(ts)
        for channel, ts in raw.items()
        if channel in VALID_SIDE_EFFECT_CHANNELS and ts not in (None, "")
    }


async def is_finalize_side_effects_claimed(
    db: AsyncSession, tenant_id: uuid.UUID
) -> bool:
    """True when legacy all-channels claim is set for this tenant."""
    tenant = await get_tenant_by_id(db, tenant_id)
    if tenant is None:
        return False
    nd = tenant.normalized_data if isinstance(tenant.normalized_data, dict) else {}
    return bool(nd.get(SIDE_EFFECTS_CLAIM_KEY))


async def is_finalize_side_effect_channel_claimed(
    db: AsyncSession, tenant_id: uuid.UUID, channel: str
) -> bool:
    """True when a specific post-call side effect was already queued."""
    if channel not in VALID_SIDE_EFFECT_CHANNELS:
        return False
    tenant = await get_tenant_by_id(db, tenant_id)
    if tenant is None:
        return False
    nd = tenant.normalized_data if isinstance(tenant.normalized_data, dict) else {}
    return channel in _side_effect_channels_claimed(nd)


async def claim_finalize_side_effects(
    db: AsyncSession, tenant_id: uuid.UUID
) -> bool:
    """Atomically claim all post-call side-effect channels (legacy helper)."""
    result = await db.execute(
        select(Tenant).where(Tenant.id == tenant_id).with_for_update()
    )
    tenant = result.scalar_one_or_none()
    if tenant is None:
        return False
    nd = dict(tenant.normalized_data or {})
    if nd.get(SIDE_EFFECTS_CLAIM_KEY) or _side_effect_channels_claimed(nd):
        return False
    claimed_at = datetime.now(UTC).isoformat()
    nd[SIDE_EFFECTS_CLAIM_KEY] = True
    nd["side_effects_claimed_at"] = claimed_at
    nd[SIDE_EFFECTS_CHANNELS_KEY] = {
        channel: claimed_at for channel in VALID_SIDE_EFFECT_CHANNELS
    }
    tenant.normalized_data = nd
    await db.commit()
    return True


async def claim_finalize_side_effect_channel(
    db: AsyncSession, tenant_id: uuid.UUID, channel: str
) -> bool:
    """Atomically claim one side-effect channel; False if already claimed."""
    if channel not in VALID_SIDE_EFFECT_CHANNELS:
        return False
    result = await db.execute(
        select(Tenant).where(Tenant.id == tenant_id).with_for_update()
    )
    tenant = result.scalar_one_or_none()
    if tenant is None:
        return False
    nd = dict(tenant.normalized_data or {})
    channels = _side_effect_channels_claimed(nd)
    if channel in channels:
        return False
    channels = dict(channels)
    channels[channel] = datetime.now(UTC).isoformat()
    nd[SIDE_EFFECTS_CHANNELS_KEY] = channels
    nd.pop(SIDE_EFFECTS_CLAIM_KEY, None)
    nd.pop("side_effects_claimed_at", None)
    tenant.normalized_data = nd
    await db.commit()
    return True


async def release_finalize_side_effect_channel(
    db: AsyncSession, tenant_id: uuid.UUID, channel: str
) -> None:
    """Undo a channel claim when Celery enqueue fails (Redis-down path)."""
    if channel not in VALID_SIDE_EFFECT_CHANNELS:
        return
    result = await db.execute(
        select(Tenant).where(Tenant.id == tenant_id).with_for_update()
    )
    tenant = result.scalar_one_or_none()
    if tenant is None:
        return
    nd = dict(tenant.normalized_data or {})
    if nd.get(SIDE_EFFECTS_CLAIM_KEY) is True:
        return
    channels = dict(_side_effect_channels_claimed(nd))
    if channel not in channels:
        return
    channels.pop(channel, None)
    if channels:
        nd[SIDE_EFFECTS_CHANNELS_KEY] = channels
    else:
        nd.pop(SIDE_EFFECTS_CHANNELS_KEY, None)
    tenant.normalized_data = nd
    await db.commit()


async def update_tenant(
    db: AsyncSession, tenant_id: uuid.UUID, **kwargs
) -> Tenant | None:
    """Update tenant record."""
    kwargs["updated_at"] = datetime.now(UTC)
    await db.execute(update(Tenant).where(Tenant.id == tenant_id).values(**kwargs))
    await db.commit()
    result = await db.execute(select(Tenant).where(Tenant.id == tenant_id))
    return result.scalar_one_or_none()


def _tenant_visible_clause():
    """Applicants visible when unlinked or their call is not soft-deleted."""
    return or_(Tenant.call_id.is_(None), Call.is_deleted == False)


def _apply_tenant_list_filters(
    stmt,
    *,
    qualification_status: str | None = None,
    phone_search: str | None = None,
    name_search: str | None = None,
    text_search: str | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    review_filter: str | None = None,
):
    if qualification_status:
        stmt = stmt.where(Tenant.qualification_status == qualification_status)
    if phone_search:
        stmt = stmt.where(Tenant.phone_number.ilike(f"%{phone_search}%"))
    if name_search:
        stmt = stmt.where(
            or_(
                Tenant.full_name.ilike(f"%{name_search}%"),
                Tenant.email.ilike(f"%{name_search}%"),
            )
        )
    if text_search:
        stmt = stmt.where(
            or_(
                Tenant.full_name.ilike(f"%{text_search}%"),
                Tenant.email.ilike(f"%{text_search}%"),
                Tenant.phone_number.ilike(f"%{text_search}%"),
            )
        )
    if date_from:
        stmt = stmt.where(Tenant.created_at >= date_from)
    if date_to:
        stmt = stmt.where(Tenant.created_at <= date_to)
    if review_filter == "unreviewed":
        stmt = stmt.where(Tenant.reviewed_by_admin == False)
    elif review_filter == "reviewed":
        stmt = stmt.where(Tenant.reviewed_by_admin == True)
    return stmt


async def count_tenants_needing_review(db: AsyncSession) -> int:
    """Count visible applicants not yet marked reviewed by an admin."""
    result = await db.execute(
        select(func.count(Tenant.id))
        .outerjoin(Call, Tenant.call_id == Call.id)
        .where(_tenant_visible_clause(), Tenant.reviewed_by_admin == False)
    )
    return result.scalar() or 0


async def count_reviewed_tenants(db: AsyncSession) -> int:
    """Count visible applicants marked reviewed by an admin."""
    result = await db.execute(
        select(func.count(Tenant.id))
        .outerjoin(Call, Tenant.call_id == Call.id)
        .where(_tenant_visible_clause(), Tenant.reviewed_by_admin == True)
    )
    return result.scalar() or 0


async def settings_touched_by_admin(
    db: AsyncSession, keys: tuple[str, ...] = ("property_name", "greeting_message", "closing_message", "landlord_email")
) -> bool:
    """True when an admin has saved any of the given settings keys."""
    if not keys:
        return False
    result = await db.execute(
        select(func.count(SystemSetting.id)).where(
            SystemSetting.key.in_(list(keys)),
            SystemSetting.updated_by.isnot(None),
        )
    )
    return (result.scalar() or 0) > 0


async def list_tenants(
    db: AsyncSession,
    page: int = 1,
    per_page: int = 20,
    qualification_status: str | None = None,
    phone_search: str | None = None,
    name_search: str | None = None,
    text_search: str | None = None,
    review_filter: str | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
) -> tuple[list[Tenant], int]:
    """List tenants with pagination and filters.

    Excludes applicants whose call was deleted (legacy soft-deleted rows), so a
    deleted call's applicant never lingers in the Applicants list. Going forward
    deletes are hard deletes (CASCADE removes the tenant outright).
    """
    visible = _tenant_visible_clause()
    filter_kwargs = {
        "qualification_status": qualification_status,
        "phone_search": phone_search,
        "name_search": name_search,
        "text_search": text_search,
        "date_from": date_from,
        "date_to": date_to,
        "review_filter": review_filter,
    }

    query = (
        select(Tenant)
        .outerjoin(Call, Tenant.call_id == Call.id)
        .where(visible)
    )
    query = _apply_tenant_list_filters(query, **filter_kwargs)

    count_query = (
        select(func.count(Tenant.id))
        .outerjoin(Call, Tenant.call_id == Call.id)
        .where(visible)
    )
    count_query = _apply_tenant_list_filters(count_query, **filter_kwargs)

    count_result = await db.execute(count_query)
    total = count_result.scalar() or 0

    query = (
        query.order_by(desc(Tenant.created_at))
        .offset((page - 1) * per_page)
        .limit(per_page)
    )
    result = await db.execute(query)
    return result.scalars().all(), total


async def list_tenant_ids_for_navigation(
    db: AsyncSession,
    *,
    review_filter: str | None = "unreviewed",
    qualification_status: str | None = None,
    limit: int = 500,
) -> list[uuid.UUID]:
    """Ordered tenant IDs for prev/next navigation in a review queue."""
    query = (
        select(Tenant.id)
        .outerjoin(Call, Tenant.call_id == Call.id)
        .where(_tenant_visible_clause())
    )
    query = _apply_tenant_list_filters(
        query,
        qualification_status=qualification_status,
        review_filter=review_filter,
    )
    query = query.order_by(desc(Tenant.created_at)).limit(limit)
    result = await db.execute(query)
    return list(result.scalars().all())


async def bulk_set_tenants_reviewed(
    db: AsyncSession,
    tenant_ids: list[uuid.UUID],
    *,
    reviewed: bool = True,
) -> int:
    """Mark multiple applicants reviewed/unreviewed. Returns rows updated."""
    if not tenant_ids:
        return 0
    now = datetime.now(UTC)
    result = await db.execute(
        update(Tenant)
        .where(Tenant.id.in_(tenant_ids))
        .values(
            reviewed_by_admin=reviewed,
            reviewed_at=now if reviewed else None,
            updated_at=now,
        )
    )
    await db.commit()
    return result.rowcount or 0


# ──────────────────────────────────────────────────────────────────────────────
# User CRUD
# ──────────────────────────────────────────────────────────────────────────────


async def get_user_by_email(db: AsyncSession, email: str) -> AdminUser | None:
    """Get admin user by email."""
    result = await db.execute(select(AdminUser).where(AdminUser.email == email))
    return result.scalar_one_or_none()


async def get_user_by_id(db: AsyncSession, user_id: uuid.UUID) -> AdminUser | None:
    """Get admin user by UUID."""
    result = await db.execute(select(AdminUser).where(AdminUser.id == user_id))
    return result.scalar_one_or_none()


async def update_last_login(db: AsyncSession, user_id: uuid.UUID) -> None:
    """Update user's last login timestamp."""
    await db.execute(
        update(AdminUser)
        .where(AdminUser.id == user_id)
        .values(last_login=datetime.now(UTC))
    )
    await db.commit()


async def list_users(db: AsyncSession) -> list[AdminUser]:
    """List all admin users."""
    result = await db.execute(select(AdminUser).order_by(AdminUser.created_at))
    return result.scalars().all()


def _encode_permissions(scopes: list[str] | None) -> str | None:
    """Persist a scope list as a comma-separated string (or None)."""
    if not scopes:
        return None
    return ",".join(sorted(set(scopes)))


async def create_user(
    db: AsyncSession,
    email: str,
    hashed_password: str,
    full_name: str,
    role: str = "admin",
    permissions: list[str] | None = None,
) -> AdminUser:
    """Create a new admin user."""
    user = AdminUser(
        email=email,
        hashed_password=hashed_password,
        full_name=full_name,
        role=role,
        permissions=_encode_permissions(permissions),
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


async def update_user_password(
    db: AsyncSession, user_id: uuid.UUID, hashed_password: str
) -> None:
    """Replace a user's stored password hash."""
    await db.execute(
        update(AdminUser)
        .where(AdminUser.id == user_id)
        .values(hashed_password=hashed_password)
    )
    await db.commit()


async def delete_user(db: AsyncSession, user_id: uuid.UUID) -> bool:
    """Delete an admin user.

    The audit_log.admin_user_id and system_settings.updated_by foreign keys use
    ON DELETE SET NULL, so the user's connected history is preserved (those rows
    remain, just no longer attributed to a named user) rather than cascade-deleted.
    """
    user = await get_user_by_id(db, user_id)
    if not user:
        return False
    await db.delete(user)
    await db.commit()
    return True


async def update_user(
    db: AsyncSession,
    user_id: uuid.UUID,
    *,
    full_name: str | None = None,
    role: str | None = None,
    permissions: list[str] | None = None,
    is_active: bool | None = None,
) -> AdminUser | None:
    """Update an admin user's profile, role, scopes, or active status."""
    user = await get_user_by_id(db, user_id)
    if not user:
        return None

    if full_name is not None:
        user.full_name = full_name.strip() or user.email
    if role is not None:
        user.role = role
    if permissions is not None:
        user.permissions = _encode_permissions(permissions)
    if is_active is not None:
        user.is_active = is_active

    await db.commit()
    await db.refresh(user)
    return user


# ──────────────────────────────────────────────────────────────────────────────
# System Settings CRUD
# ──────────────────────────────────────────────────────────────────────────────


async def fetch_settings_batch(
    db: AsyncSession,
    keys: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    """Fetch multiple settings in one query with typed parsing."""
    if not keys:
        return {}
    result = await db.execute(
        select(SystemSetting).where(SystemSetting.key.in_(list(keys)))
    )
    rows = {s.key: s for s in result.scalars()}
    out: dict[str, Any] = {}
    for key in keys:
        setting = rows.get(key)
        if not setting:
            continue
        if setting.value_type == "integer":
            try:
                out[key] = int(setting.value)
            except (ValueError, TypeError):
                out[key] = setting.value
        elif setting.value_type == "boolean":
            out[key] = setting.value.lower() in ("true", "1", "yes")
        elif setting.value_type == "json":
            try:
                out[key] = json.loads(setting.value)
            except (json.JSONDecodeError, TypeError):
                out[key] = setting.value
        else:
            out[key] = setting.value
    return out


async def get_setting(db: AsyncSession, key: str) -> SystemSetting | None:
    """Get a system setting by key."""
    result = await db.execute(select(SystemSetting).where(SystemSetting.key == key))
    return result.scalar_one_or_none()


async def get_setting_value(db: AsyncSession, key: str, default: Any = None) -> Any:
    """Get a setting value, parsed by its type."""
    setting = await get_setting(db, key)
    if not setting:
        return default

    if setting.value_type == "integer":
        try:
            return int(setting.value)
        except (ValueError, TypeError):
            return default
    elif setting.value_type == "boolean":
        return setting.value.lower() in ("true", "1", "yes")
    elif setting.value_type == "json":
        try:
            return json.loads(setting.value)
        except (json.JSONDecodeError, TypeError):
            return default
    return setting.value


async def set_setting(
    db: AsyncSession,
    key: str,
    value: str,
    updated_by: uuid.UUID | None = None,
    *,
    is_sensitive: bool | None = None,
) -> tuple[SystemSetting, bool]:
    """Create or update a system setting.

    Returns ``(setting, cache_invalidated)``. Cache invalidation failures are
    logged but do not roll back the DB write.
    """
    if is_sensitive is None:
        is_sensitive = is_sensitive_setting_key(key)
    setting = await get_setting(db, key)
    if setting:
        setting.value = value
        setting.is_sensitive = is_sensitive
        if updated_by:
            setting.updated_by = updated_by
        setting.updated_at = datetime.now(UTC)
    else:
        setting = SystemSetting(
            key=key,
            value=value,
            value_type=_default_value_type_for_key(key),
            updated_by=updated_by,
            is_sensitive=is_sensitive,
        )
        db.add(setting)
    await db.commit()
    await db.refresh(setting)
    from app.services.settings_cache import invalidate_settings_cache

    try:
        await invalidate_settings_cache()
    except Exception as exc:
        # The DB write is already committed; cache invalidation failure should
        # not turn a successful settings save into a user-visible API failure.
        logger.warning("Settings cache invalidation failed for %s: %s", key, exc)
        return setting, False
    return setting, True


async def set_settings_bulk(
    db: AsyncSession,
    updates: dict[str, str],
    updated_by: uuid.UUID | None = None,
) -> bool:
    """Create/update multiple settings atomically, then invalidate cache once.

    Returns True when cache invalidation succeeded, False otherwise. DB writes
    are committed in a single transaction so partial admin saves are avoided.
    """
    if not updates:
        return True

    keys = list(updates.keys())
    result = await db.execute(select(SystemSetting).where(SystemSetting.key.in_(keys)))
    existing = {s.key: s for s in result.scalars()}
    now = datetime.now(UTC)

    for key, value in updates.items():
        is_sensitive = is_sensitive_setting_key(key)
        setting = existing.get(key)
        if setting is not None:
            setting.value = value
            setting.is_sensitive = is_sensitive
            if updated_by:
                setting.updated_by = updated_by
            setting.updated_at = now
            continue
        db.add(
            SystemSetting(
                key=key,
                value=value,
                value_type=_default_value_type_for_key(key),
                updated_by=updated_by,
                is_sensitive=is_sensitive,
            )
        )

    await db.commit()
    from app.services.settings_cache import invalidate_settings_cache

    try:
        await invalidate_settings_cache()
    except Exception as exc:
        logger.warning("Settings cache invalidation failed for bulk save: %s", exc)
        return False
    return True


async def get_all_settings(db: AsyncSession) -> dict[str, str]:
    """Get all settings as a flat dict (sensitive values masked)."""
    result = await db.execute(select(SystemSetting))
    settings_list = result.scalars().all()
    return {
        s.key: (
            "****"
            if (s.is_sensitive or is_sensitive_setting_key(s.key))
            else s.value
        )
        for s in settings_list
    }


async def is_number_blacklisted(db: AsyncSession, phone_number: str) -> bool:
    """True when *phone_number* (after sanitization) is on the DNC list."""
    from app.utils.helpers import sanitize_phone_number

    normalized = sanitize_phone_number(phone_number)
    if not normalized:
        return False
    blacklist = await get_setting_value(db, "blacklisted_numbers", [])
    return normalized in (blacklist or [])


async def add_to_blacklist(
    db: AsyncSession, phone_number: str, updated_by: uuid.UUID | None = None
) -> tuple[list[str], bool]:
    """Add a phone number to the Do-Not-Call blacklist (idempotent)."""
    from app.utils.helpers import sanitize_phone_number

    phone = sanitize_phone_number(phone_number)
    if not phone:
        current = await get_setting_value(db, "blacklisted_numbers", [])
        return current, True
    return await _mutate_blacklist_numbers(
        db,
        updated_by=updated_by,
        add_phone=phone,
    )


async def remove_from_blacklist(
    db: AsyncSession, phone_number: str, updated_by: uuid.UUID | None = None
) -> tuple[list[str], bool]:
    """Remove a phone number from the Do-Not-Call blacklist (idempotent)."""
    from app.utils.helpers import sanitize_phone_number

    phone = sanitize_phone_number(phone_number)
    if not phone:
        current = await get_setting_value(db, "blacklisted_numbers", [])
        return current, True
    return await _mutate_blacklist_numbers(
        db,
        updated_by=updated_by,
        remove_phone=phone,
    )


def _normalize_blacklist_numbers(numbers: list[str]) -> list[str]:
    """Normalize stored blacklist entries to canonical E.164-like values."""
    from app.utils.helpers import sanitize_phone_number

    normalized: list[str] = []
    seen: set[str] = set()
    for item in numbers:
        phone = sanitize_phone_number(str(item))
        if phone and phone not in seen:
            seen.add(phone)
            normalized.append(phone)
    return normalized


async def _mutate_blacklist_numbers(
    db: AsyncSession,
    *,
    updated_by: uuid.UUID | None = None,
    add_phone: str | None = None,
    remove_phone: str | None = None,
) -> tuple[list[str], bool]:
    """Atomically mutate blacklisted_numbers with a row lock.

    Prevents lost updates when two admin requests edit the JSON list at once.
    """
    row = (
        await db.execute(
            select(SystemSetting)
            .where(SystemSetting.key == "blacklisted_numbers")
            .with_for_update()
        )
    ).scalar_one_or_none()

    current: list[str] = []
    if row and row.value:
        try:
            parsed = json.loads(row.value)
            if isinstance(parsed, list):
                current = [str(item) for item in parsed if item]
        except (TypeError, json.JSONDecodeError):
            current = []

    current = _normalize_blacklist_numbers(current)

    if add_phone and add_phone not in current:
        current.append(add_phone)
    if remove_phone and remove_phone in current:
        current.remove(remove_phone)
        await db.execute(
            update(Tenant)
            .where(Tenant.phone_number == remove_phone)
            .values(is_blacklisted=False, updated_at=datetime.now(UTC))
        )

    payload = json.dumps(current)
    if row is None:
        row = SystemSetting(
            key="blacklisted_numbers",
            value=payload,
            value_type="json",
            updated_by=updated_by,
            is_sensitive=False,
        )
        db.add(row)
    else:
        row.value = payload
        row.value_type = "json"
        row.is_sensitive = False
        row.updated_at = datetime.now(UTC)
        if updated_by:
            row.updated_by = updated_by

    await db.commit()
    from app.services.settings_cache import invalidate_settings_cache

    try:
        await invalidate_settings_cache()
    except Exception as exc:
        logger.warning("Settings cache invalidation failed for blacklist update: %s", exc)
        return current, False
    return current, True


# ──────────────────────────────────────────────────────────────────────────────
# Audit Log CRUD
# ──────────────────────────────────────────────────────────────────────────────


async def create_audit_log(
    db: AsyncSession,
    action: str,
    admin_user_id: uuid.UUID | None = None,
    entity_type: str | None = None,
    entity_id: uuid.UUID | None = None,
    old_value: dict | None = None,
    new_value: dict | None = None,
    ip_address: str | None = None,
    user_agent: str | None = None,
) -> AuditLog:
    """Create an audit log entry."""
    log = AuditLog(
        admin_user_id=admin_user_id,
        action=action,
        entity_type=entity_type,
        entity_id=entity_id,
        old_value=old_value,
        new_value=new_value,
        ip_address=ip_address,
        user_agent=user_agent,
    )
    db.add(log)
    await db.commit()
    await db.refresh(log)
    return log


async def list_audit_logs(
    db: AsyncSession,
    page: int = 1,
    per_page: int = 50,
    admin_user_id: uuid.UUID | None = None,
    action: str | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
) -> tuple[list[AuditLog], int]:
    """List audit logs with filters and pagination."""
    filters = []
    if admin_user_id:
        filters.append(AuditLog.admin_user_id == admin_user_id)
    if action:
        filters.append(AuditLog.action.ilike(f"%{action}%"))
    if date_from:
        filters.append(AuditLog.created_at >= date_from)
    if date_to:
        filters.append(AuditLog.created_at <= date_to)

    query = select(AuditLog)
    count_query = select(func.count(AuditLog.id))
    if filters:
        query = query.where(*filters)
        count_query = count_query.where(*filters)

    count_result = await db.execute(count_query)
    total = count_result.scalar() or 0

    query = (
        query.order_by(desc(AuditLog.created_at))
        .offset((page - 1) * per_page)
        .limit(per_page)
    )
    result = await db.execute(query)
    return result.scalars().all(), total


async def get_analytics_data(db: AsyncSession, days: int = 30) -> dict:
    """Get comprehensive analytics data - OPTIMIZED: 2 queries + Redis caching instead of 7."""
    cache_key = f"analytics:{days}:{datetime.now(UTC).date()}"
    cached = await cache_get_json(cache_key)
    if cached is not None:
        logger.debug("Analytics cache hit for %s", cache_key)
        return cached

    now = datetime.now(UTC)
    start_date = now - timedelta(days=days)

    # Query 1: All call data aggregations in one query.
    # NOTE: the date_trunc unit is inlined as a literal (not a bound param) so
    # the SELECT and GROUP BY expressions render to identical SQL. Passing
    # "day" as a string makes SQLAlchemy emit separate bind params ($1 vs $3),
    # which Postgres treats as different expressions and rejects with
    # "column calls.created_at must appear in the GROUP BY clause".
    day_expr = func.date_trunc(literal_column("'day'"), Call.created_at)
    hour_expr = extract("hour", Call.created_at)
    call_result = await db.execute(
        select(
            day_expr.label("day"),
            Call.status.label("status"),
            Call.llm_provider.label("llm_provider"),
            Call.questions_answered.label("questions_answered"),
            hour_expr.label("hour"),
            func.count(Call.id).label("count"),
        )
        .where(Call.is_deleted == False, Call.created_at >= start_date)
        .group_by(
            day_expr,
            Call.status,
            Call.llm_provider,
            Call.questions_answered,
            hour_expr,
        )
    )

    # Query 2: Tenant qualification data with average scores
    tenant_result = await db.execute(
        select(
            Tenant.qualification_status,
            func.count(Tenant.id).label("count"),
            func.avg(Tenant.qualification_score).label("avg_score"),
        )
        .join(Call, Call.id == Tenant.call_id)
        .where(Call.is_deleted == False, Call.created_at >= start_date)
        .group_by(Tenant.qualification_status)
    )

    # Process aggregated results
    calls_by_day_dict = {}
    status_breakdown = {}
    llm_usage = {}
    questions_dist = {}
    calls_by_hour = {}

    for row in call_result:
        # Calls by day
        day_str = str(row.day.date()) if row.day else "unknown"
        if day_str not in calls_by_day_dict:
            calls_by_day_dict[day_str] = 0
        calls_by_day_dict[day_str] += row.count

        # Status breakdown
        if row.status:
            status_breakdown[row.status] = (
                status_breakdown.get(row.status, 0) + row.count
            )

        # LLM usage
        if row.llm_provider:
            llm_usage[row.llm_provider] = llm_usage.get(row.llm_provider, 0) + row.count

        # Questions distribution
        q_key = (
            str(row.questions_answered) if row.questions_answered is not None else "0"
        )
        questions_dist[q_key] = questions_dist.get(q_key, 0) + row.count

        # Calls by hour
        if row.hour is not None:
            hour_key = str(int(row.hour))
            calls_by_hour[hour_key] = calls_by_hour.get(hour_key, 0) + row.count

    # Process tenant results
    qual_breakdown = {}
    avg_qualified_score = 0

    for row in tenant_result:
        status_key = row.qualification_status or "unknown"
        qual_breakdown[status_key] = row.count
        if row.qualification_status == "qualified" and row.avg_score:
            avg_qualified_score = round(float(row.avg_score), 1)

    # Build response
    result = {
        "calls_by_day": [
            {"date": d, "count": c} for d, c in sorted(calls_by_day_dict.items())
        ],
        "status_breakdown": status_breakdown,
        "qualification_breakdown": qual_breakdown,
        "llm_usage": llm_usage,
        "questions_distribution": questions_dist,
        "calls_by_hour": calls_by_hour,
        "avg_qualified_score": avg_qualified_score,
    }

    # Cache result for 5 minutes (TTL keeps memory bounded).
    await cache_set_json(cache_key, result, 300)
    logger.debug("Cached analytics for %s", cache_key)

    return result
