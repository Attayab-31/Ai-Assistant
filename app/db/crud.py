"""
app/db/crud.py — All CRUD operations for the application.

Provides async functions for creating, reading, updating, and deleting
records across all models. Used by API routes and background tasks.
"""

import json
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import and_, desc, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.audit_log import AuditLog
from app.models.call import Call
from app.models.settings import SystemSetting
from app.models.tenant import Tenant
from app.models.user import AdminUser
from app.utils.security import mask_email, mask_phone
from config import (
    DEFAULT_FAQS,
    DEFAULT_QUESTIONS,
    DEFAULT_SYSTEM_SETTINGS,
    ENV_BACKED_SYSTEM_SETTING_KEYS,
)

logger = logging.getLogger(__name__)


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
        return

    await db.commit()
    logger.info(
        "Seeded defaults: %s settings inserted, %s env-backed settings synced, "
        "admin_created=%s",
        len(missing_settings),
        len(synced_settings),
        admin_created,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Call CRUD
# ──────────────────────────────────────────────────────────────────────────────


async def create_call(
    db: AsyncSession, call_id: str, phone_number: str, **kwargs
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
    await db.commit()
    await db.refresh(call)
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


async def get_call_by_uuid(db: AsyncSession, call_uuid: uuid.UUID) -> Call | None:
    """Get call by internal UUID."""
    result = await db.execute(
        select(Call)
        .where(Call.id == call_uuid, Call.is_deleted == False)
        .options(selectinload(Call.tenant))
    )
    return result.scalar_one_or_none()


async def update_call(db: AsyncSession, call_id: str, **kwargs) -> Call | None:
    """Update call record fields."""
    kwargs["updated_at"] = datetime.now(UTC)
    await db.execute(update(Call).where(Call.call_id == call_id).values(**kwargs))
    await db.commit()
    return await get_call_by_call_id(db, call_id)


async def list_calls(
    db: AsyncSession,
    page: int = 1,
    per_page: int = 20,
    status: str | None = None,
    qualification_status: str | None = None,
    phone_search: str | None = None,
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


async def soft_delete_call(db: AsyncSession, call_uuid: uuid.UUID) -> bool:
    """Soft delete a call record."""
    await db.execute(
        update(Call)
        .where(Call.id == call_uuid)
        .values(is_deleted=True, updated_at=datetime.now(UTC))
    )
    await db.commit()
    return True


async def get_call_stats(db: AsyncSession) -> dict:
    """Get aggregate call statistics for dashboard - OPTIMIZED: 2 queries instead of 7."""
    from datetime import timedelta

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
    db: AsyncSession, call_id: uuid.UUID, phone_number: str, **kwargs
) -> Tenant:
    """Create or update a tenant record after a call."""
    tenant = Tenant(call_id=call_id, phone_number=phone_number, **kwargs)
    db.add(tenant)
    await db.commit()
    await db.refresh(tenant)
    return tenant


async def get_tenant_by_call(db: AsyncSession, call_uuid: uuid.UUID) -> Tenant | None:
    """Get tenant record for a specific call."""
    result = await db.execute(select(Tenant).where(Tenant.call_id == call_uuid))
    return result.scalar_one_or_none()


async def update_tenant(
    db: AsyncSession, tenant_id: uuid.UUID, **kwargs
) -> Tenant | None:
    """Update tenant record."""
    kwargs["updated_at"] = datetime.now(UTC)
    await db.execute(update(Tenant).where(Tenant.id == tenant_id).values(**kwargs))
    await db.commit()
    result = await db.execute(select(Tenant).where(Tenant.id == tenant_id))
    return result.scalar_one_or_none()


async def list_tenants(
    db: AsyncSession,
    page: int = 1,
    per_page: int = 20,
    qualification_status: str | None = None,
    phone_search: str | None = None,
) -> tuple[list[Tenant], int]:
    """List tenants with pagination and filters."""
    query = select(Tenant)
    if qualification_status:
        query = query.where(Tenant.qualification_status == qualification_status)
    if phone_search:
        query = query.where(Tenant.phone_number.ilike(f"%{phone_search}%"))

    count_query = select(func.count(Tenant.id))
    if qualification_status:
        count_query = count_query.where(
            Tenant.qualification_status == qualification_status
        )
    if phone_search:
        count_query = count_query.where(Tenant.phone_number.ilike(f"%{phone_search}%"))

    count_result = await db.execute(count_query)
    total = count_result.scalar() or 0

    query = (
        query.order_by(desc(Tenant.created_at))
        .offset((page - 1) * per_page)
        .limit(per_page)
    )
    result = await db.execute(query)
    return result.scalars().all(), total


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


async def create_user(
    db: AsyncSession,
    email: str,
    hashed_password: str,
    full_name: str,
    role: str = "admin",
) -> AdminUser:
    """Create a new admin user."""
    user = AdminUser(
        email=email, hashed_password=hashed_password, full_name=full_name, role=role
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


async def count_users(db: AsyncSession) -> int:
    """Return the number of admin users (used to gate first-user signup)."""
    result = await db.execute(select(func.count()).select_from(AdminUser))
    return int(result.scalar() or 0)


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


async def count_active_super_admins(db: AsyncSession) -> int:
    """Number of active super_admins — used to prevent deleting the last one."""
    result = await db.execute(
        select(func.count())
        .select_from(AdminUser)
        .where(AdminUser.role == "super_admin", AdminUser.is_active == True)  # noqa: E712
    )
    return int(result.scalar() or 0)


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
    db: AsyncSession, key: str, value: str, updated_by: uuid.UUID | None = None
) -> SystemSetting:
    """Create or update a system setting."""
    setting = await get_setting(db, key)
    if setting:
        setting.value = value
        if updated_by:
            setting.updated_by = updated_by
        setting.updated_at = datetime.now(UTC)
    else:
        setting = SystemSetting(key=key, value=value, updated_by=updated_by)
        db.add(setting)
    await db.commit()
    await db.refresh(setting)
    from app.services.settings_cache import invalidate_settings_cache

    await invalidate_settings_cache()
    return setting


async def get_all_settings(db: AsyncSession) -> dict[str, str]:
    """Get all settings as a flat dict (sensitive values masked)."""
    result = await db.execute(select(SystemSetting))
    settings_list = result.scalars().all()
    return {s.key: ("****" if s.is_sensitive else s.value) for s in settings_list}


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
    query = select(AuditLog)
    if admin_user_id:
        query = query.where(AuditLog.admin_user_id == admin_user_id)
    if action:
        query = query.where(AuditLog.action.ilike(f"%{action}%"))
    if date_from:
        query = query.where(AuditLog.created_at >= date_from)
    if date_to:
        query = query.where(AuditLog.created_at <= date_to)

    # Get total count - OPTIMIZED: direct count without subquery
    count_result = await db.execute(select(func.count(AuditLog.id)))
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
    from datetime import timedelta

    from sqlalchemy import extract, literal_column

    # Try Redis cache first (shared pooled client — no per-call connections).
    from app.core.redis_client import cache_get_json, cache_set_json

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
