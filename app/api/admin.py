"""
app/api/admin.py — Admin dashboard routes: pages (Jinja2) + JSON API endpoints.

Covers:
- Dashboard overview
- Calls list/detail/delete/resend-email
- Tenants list/detail/blacklist
- Analytics
- Audit log
- Account/users
"""

import csv
import io
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import crud
from app.db.database import get_db
from app.models.user import AdminUser
from app.utils.dependencies import (
    get_current_user,
    get_current_user_optional,
    invalidate_user_cache,
    require_role,
    require_scope,
)
from app.utils.helpers import (
    format_currency,
    format_duration,
    format_phone_display,
    friendly_state,
    score_color,
    status_badge_color,
    time_ago,
)

logger = logging.getLogger(__name__)
router = APIRouter()

TEMPLATES_DIR = Path(__file__).parent.parent / "admin" / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Register custom Jinja2 filters
templates.env.filters["duration"] = format_duration
templates.env.filters["currency"] = format_currency
templates.env.filters["phone_display"] = format_phone_display
templates.env.filters["time_ago"] = time_ago
templates.env.filters["status_color"] = status_badge_color
templates.env.filters["score_color"] = score_color
templates.env.filters["friendly_state"] = friendly_state


# ──────────────────────────────────────────────────────────────────────────────
# Page Routes (HTML)
# ──────────────────────────────────────────────────────────────────────────────


def _guard_page(user: AdminUser | None, scope: str) -> RedirectResponse | None:
    """Page-access guard for HTML routes.

    Returns a RedirectResponse to send back (login if not authenticated, or the
    dashboard if authenticated but lacking the area scope), or None when the
    user is allowed to view the page.
    """
    if not user:
        return RedirectResponse(url="/admin/login")
    if not user.can(scope):
        return RedirectResponse(url="/admin/dashboard")
    return None


@router.get("/login", response_class=HTMLResponse, include_in_schema=False)
async def login_page(request: Request):
    """Render the admin login page."""
    return templates.TemplateResponse("login.html", {"request": request})


@router.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
async def dashboard_page(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Render the main dashboard page."""
    user = await get_current_user_optional(request, db)
    if not user:
        return RedirectResponse(url="/admin/login")

    stats = await crud.get_call_stats(db)
    recent_calls, _ = await crud.list_calls(db, page=1, per_page=10)
    active_sessions = []
    try:
        from app.core.call_handler import get_active_sessions

        active_sessions = get_active_sessions()
    except (ImportError, AttributeError) as e:
        logger.debug("Could not fetch active sessions: %s", e)

    from config import provider_registry

    provider_status = provider_registry.get_status()

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "user": user,
            "stats": stats,
            "recent_calls": recent_calls,
            "active_sessions": active_sessions,
            "provider_status": provider_status,
            "active_page": "dashboard",
        },
    )


@router.get("/calls", response_class=HTMLResponse, include_in_schema=False)
async def calls_list_page(
    request: Request,
    db: AsyncSession = Depends(get_db),
    page: int = 1,
    status_filter: str | None = Query(None, alias="status"),
    qualification: str | None = None,
    phone: str | None = None,
):
    """Render the all calls list page."""
    user = await get_current_user_optional(request, db)
    if guard := _guard_page(user, "calls"):
        return guard

    calls, total = await crud.list_calls(
        db,
        page=page,
        per_page=20,
        status=status_filter,
        qualification_status=qualification,
        phone_search=phone,
    )

    return templates.TemplateResponse(
        "calls/list.html",
        {
            "request": request,
            "user": user,
            "calls": calls,
            "total": total,
            "page": page,
            "per_page": 20,
            "total_pages": max(1, (total + 19) // 20),
            "active_page": "calls",
            "filters": {
                "status": status_filter,
                "qualification": qualification,
                "phone": phone,
            },
        },
    )


@router.get("/calls/{call_id}", response_class=HTMLResponse, include_in_schema=False)
async def call_detail_page(
    request: Request,
    call_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Render the single call detail page."""
    user = await get_current_user_optional(request, db)
    if guard := _guard_page(user, "calls"):
        return guard

    call = await crud.get_call_by_uuid(db, call_id)
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")

    tenant = await crud.get_tenant_by_call(db, call_id)

    active_question_count = None
    if tenant:
        from app.core.screening_flow import count_active_questions

        tenant_data = {
            "has_pets": tenant.has_pets,
            "has_eviction": tenant.has_eviction,
        }
        active_question_count = count_active_questions(tenant_data)

    score_breakdown = None
    if tenant:
        from app.core.qualifier import get_score_breakdown

        all_settings = await crud.get_all_settings(db)
        tenant_dict = {
            "monthly_income": tenant.monthly_income,
            "has_eviction": tenant.has_eviction,
            "move_in_date": tenant.move_in_date,
            "questions_answered": call.questions_answered,
        }
        score_breakdown = get_score_breakdown(tenant_dict, all_settings)

    return templates.TemplateResponse(
        "calls/detail.html",
        {
            "request": request,
            "user": user,
            "call": call,
            "tenant": tenant,
            "score_breakdown": score_breakdown,
            "active_question_count": active_question_count,
            "active_page": "calls",
        },
    )


@router.get("/tenants", response_class=HTMLResponse, include_in_schema=False)
async def tenants_list_page(
    request: Request,
    db: AsyncSession = Depends(get_db),
    page: int = 1,
    qualification: str | None = None,
    phone: str | None = None,
):
    """Render the all tenants list page."""
    user = await get_current_user_optional(request, db)
    if guard := _guard_page(user, "tenants"):
        return guard

    tenants, total = await crud.list_tenants(
        db,
        page=page,
        per_page=20,
        qualification_status=qualification,
        phone_search=phone,
    )

    return templates.TemplateResponse(
        "tenants/list.html",
        {
            "request": request,
            "user": user,
            "tenants": tenants,
            "total": total,
            "page": page,
            "total_pages": max(1, (total + 19) // 20),
            "active_page": "tenants",
            "filters": {"qualification": qualification, "phone": phone},
        },
    )


@router.get(
    "/tenants/{tenant_id}", response_class=HTMLResponse, include_in_schema=False
)
async def tenant_detail_page(
    request: Request,
    tenant_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Render single tenant profile page."""
    user = await get_current_user_optional(request, db)
    if guard := _guard_page(user, "tenants"):
        return guard

    from sqlalchemy import select

    from app.models.tenant import Tenant

    result = await db.execute(select(Tenant).where(Tenant.id == tenant_id))
    tenant = result.scalar_one_or_none()
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")

    # Get call history for this phone number
    history_calls, _ = await crud.list_calls(
        db, page=1, per_page=50, phone_search=tenant.phone_number
    )

    return templates.TemplateResponse(
        "tenants/detail.html",
        {
            "request": request,
            "user": user,
            "tenant": tenant,
            "history_calls": history_calls,
            "active_page": "tenants",
        },
    )


@router.get("/analytics", response_class=HTMLResponse, include_in_schema=False)
async def analytics_page(
    request: Request,
    db: AsyncSession = Depends(get_db),
    days: int = 30,
):
    """Render analytics page with charts."""
    user = await get_current_user_optional(request, db)
    if guard := _guard_page(user, "analytics"):
        return guard

    analytics = await crud.get_analytics_data(db, days=days)
    stats = await crud.get_call_stats(db)

    return templates.TemplateResponse(
        "analytics.html",
        {
            "request": request,
            "user": user,
            "analytics": analytics,
            "stats": stats,
            "days": days,
            "active_page": "analytics",
        },
    )


@router.get("/monitor", response_class=HTMLResponse, include_in_schema=False)
async def monitor_page(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Render the Live Monitor page (live calls, system health, API usage)."""
    user = await get_current_user_optional(request, db)
    if guard := _guard_page(user, "monitor"):
        return guard

    return templates.TemplateResponse(
        "monitor.html",
        {
            "request": request,
            "user": user,
            "active_page": "monitor",
        },
    )


@router.get("/audit-log", response_class=HTMLResponse, include_in_schema=False)
async def audit_log_page(
    request: Request,
    db: AsyncSession = Depends(get_db),
    page: int = 1,
    action: str | None = None,
):
    """Render audit log page."""
    user = await get_current_user_optional(request, db)
    if guard := _guard_page(user, "audit"):
        return guard

    logs, total = await crud.list_audit_logs(db, page=page, per_page=50, action=action)
    users = await crud.list_users(db)
    user_map = {str(u.id): u.email for u in users}

    return templates.TemplateResponse(
        "audit_log.html",
        {
            "request": request,
            "user": user,
            "logs": logs,
            "user_map": user_map,
            "total": total,
            "page": page,
            "total_pages": max(1, (total + 49) // 50),
            "active_page": "audit_log",
        },
    )


@router.get("/settings/providers", response_class=HTMLResponse, include_in_schema=False)
async def settings_providers_page(request: Request, db: AsyncSession = Depends(get_db)):
    """Render AI providers settings page."""
    user = await get_current_user_optional(request, db)
    if guard := _guard_page(user, "settings"):
        return guard

    all_settings = await crud.get_all_settings(db)
    from config import provider_registry
    from config import settings as env_settings

    provider_status = provider_registry.get_status()

    # Which providers already have a usable API key — from an env var OR a
    # key rotated through the admin panel (stored as <provider>_api_key_encrypted).
    # We expose only booleans, never the key material.
    def _key_set(provider: str, env_attr: str) -> bool:
        if (getattr(env_settings, env_attr, "") or "").strip():
            return True
        return bool(all_settings.get(f"{provider}_api_key_encrypted"))

    provider_keys = {
        "groq": _key_set("groq", "groq_api_key"),
        "openai": _key_set("openai", "openai_api_key"),
        "openrouter": _key_set("openrouter", "openrouter_api_key"),
        "gemini": _key_set("gemini", "gemini_api_key"),
        "deepgram": _key_set("deepgram", "deepgram_api_key"),
    }

    return templates.TemplateResponse(
        "settings/providers.html",
        {
            "request": request,
            "user": user,
            "settings": all_settings,
            "provider_status": provider_status,
            "provider_keys": provider_keys,
            "active_page": "settings",
            "section": "providers",
        },
    )


@router.get("/settings/questions", response_class=HTMLResponse, include_in_schema=False)
async def settings_questions_page(request: Request, db: AsyncSession = Depends(get_db)):
    """Render screening questions editor page."""
    user = await get_current_user_optional(request, db)
    if guard := _guard_page(user, "settings"):
        return guard

    questions = await crud.get_setting_value(db, "screening_questions", [])
    from app.core.screening_flow import FLOW_STATE_VALUES, normalize_questions

    normalized = normalize_questions(questions)
    return templates.TemplateResponse(
        "settings/questions.html",
        {
            "request": request,
            "user": user,
            "questions": normalized,
            "flow_state_count": len(FLOW_STATE_VALUES),
            "active_page": "settings",
            "section": "questions",
        },
    )


@router.get("/settings/faqs", response_class=HTMLResponse, include_in_schema=False)
async def settings_faqs_page(request: Request, db: AsyncSession = Depends(get_db)):
    """Render screening FAQ editor page."""
    user = await get_current_user_optional(request, db)
    if guard := _guard_page(user, "settings"):
        return guard

    faqs = await crud.get_setting_value(db, "screening_faqs", [])
    from app.core.screening_flow import FAQ_TOPIC_VALUES, normalize_faqs

    normalized = normalize_faqs(faqs)
    return templates.TemplateResponse(
        "settings/faqs.html",
        {
            "request": request,
            "user": user,
            "faqs": normalized,
            "faq_topic_count": len(FAQ_TOPIC_VALUES),
            "active_page": "settings",
            "section": "faqs",
        },
    )


@router.get("/settings/email", response_class=HTMLResponse, include_in_schema=False)
async def settings_email_page(request: Request, db: AsyncSession = Depends(get_db)):
    """Render email settings page."""
    user = await get_current_user_optional(request, db)
    if guard := _guard_page(user, "settings"):
        return guard

    all_settings = await crud.get_all_settings(db)
    return templates.TemplateResponse(
        "settings/email.html",
        {
            "request": request,
            "user": user,
            "settings": all_settings,
            "active_page": "settings",
            "section": "email",
        },
    )


@router.get("/settings/general", response_class=HTMLResponse, include_in_schema=False)
async def settings_general_page(request: Request, db: AsyncSession = Depends(get_db)):
    """Render general settings page."""
    user = await get_current_user_optional(request, db)
    if guard := _guard_page(user, "settings"):
        return guard

    all_settings = await crud.get_all_settings(db)
    return templates.TemplateResponse(
        "settings/general.html",
        {
            "request": request,
            "user": user,
            "settings": all_settings,
            "active_page": "settings",
            "section": "general",
        },
    )


@router.get("/account", response_class=HTMLResponse, include_in_schema=False)
async def account_page(request: Request, db: AsyncSession = Depends(get_db)):
    """Render account management page (super admin only)."""
    user = await get_current_user_optional(request, db)
    if guard := _guard_page(user, "accounts"):
        return guard

    users = await crud.list_users(db)
    super_admin_count = await crud.count_active_super_admins(db)
    from app.models.user import PERMISSION_SCOPES

    return templates.TemplateResponse(
        "account.html",
        {
            "request": request,
            "user": user,
            "users": users,
            "super_admin_count": super_admin_count,
            "permission_scopes": PERMISSION_SCOPES,
            "active_page": "account",
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# JSON API Routes
# ──────────────────────────────────────────────────────────────────────────────


@router.get("/api/stats")
async def api_get_stats(
    db: AsyncSession = Depends(get_db),
    user: AdminUser = Depends(get_current_user),
):
    """Get dashboard stats as JSON (for AJAX refresh)."""
    return await crud.get_call_stats(db)


# Approx number of core questions, used only for the live progress bar.
_PROGRESS_DENOMINATOR = 15


@router.get("/api/monitor")
async def api_monitor(
    db: AsyncSession = Depends(get_db),
    user: AdminUser = Depends(require_scope("monitor")),
):
    """Everything the Live Monitor needs in one poll: live calls, system health,
    today's activity, and provider usage/balances.

    Designed to be polled every few seconds, so each piece is cheap or cached and
    nothing here is allowed to fail the whole response.
    """
    # 1) Live calls in progress (in-memory, this worker).
    active_calls = []
    try:
        from app.core.call_handler import get_active_sessions

        for s in get_active_sessions():
            answered = s.get("questions_answered") or 0
            active_calls.append(
                {
                    "call_id": s.get("call_id"),
                    "phone_display": format_phone_display(s.get("phone_number")),
                    "state": s.get("state"),
                    "state_label": friendly_state(s.get("state")),
                    "duration_seconds": s.get("duration"),
                    "duration_label": format_duration(s.get("duration")),
                    "questions_answered": answered,
                    "progress_pct": min(
                        100, round(answered / _PROGRESS_DENOMINATOR * 100)
                    ),
                    "started_at": s.get("started_at"),
                }
            )
    except Exception as e:
        logger.debug("Could not collect active sessions: %s", e)

    # 2) System health (DB is implicitly OK — this query ran).
    redis_ok = False
    try:
        from app.core.redis_client import ping as redis_ping

        redis_ok = await redis_ping()
    except Exception as e:
        logger.debug("Redis ping failed: %s", e)

    uptime_seconds = None
    try:
        import main

        uptime_seconds = round(__import__("time").time() - main.APP_START_TIME)
    except Exception:
        uptime_seconds = None

    from config import provider_registry

    provider_status = provider_registry.get_status()

    # 3) Today's activity + provider usage/balances.
    stats = await crud.get_call_stats(db)

    usage = {}
    try:
        from app.services.provider_usage import get_provider_overview

        usage = await get_provider_overview(db, days=30)
    except Exception as e:
        logger.debug("Provider usage lookup failed: %s", e)

    return {
        "active_calls": active_calls,
        "active_count": len(active_calls),
        "health": {
            "database": True,
            "redis": redis_ok,
            "uptime_seconds": uptime_seconds,
            "providers": provider_status,
        },
        "stats": stats,
        "usage": usage,
    }


@router.get("/api/calls")
async def api_list_calls(
    db: AsyncSession = Depends(get_db),
    user: AdminUser = Depends(require_scope("calls")),
    page: int = 1,
    per_page: int = 20,
    status: str | None = None,
    qualification: str | None = None,
    phone: str | None = None,
):
    """List calls as JSON."""
    calls, total = await crud.list_calls(
        db,
        page=page,
        per_page=per_page,
        status=status,
        qualification_status=qualification,
        phone_search=phone,
    )
    return {
        "calls": [
            {
                "id": str(c.id),
                "call_id": c.call_id,
                "phone_number": c.phone_number,
                "status": c.status,
                "duration_seconds": c.duration_seconds,
                "questions_answered": c.questions_answered,
                "qualification_status": c.tenant.qualification_status
                if c.tenant
                else None,
                "qualification_score": c.tenant.qualification_score
                if c.tenant
                else None,
                "llm_provider": c.llm_provider,
                "created_at": c.created_at.isoformat(),
            }
            for c in calls
        ],
        "total": total,
        "page": page,
        "per_page": per_page,
    }


@router.get("/api/calls/{call_id}/recording")
async def api_get_call_recording(
    call_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user: AdminUser = Depends(require_scope("calls")),
):
    """Redirect to the call recording behind a short-lived, authenticated link.

    Recordings are stored privately; we mint a signed URL on demand so the audio
    is never exposed through a permanent public link. Older rows that stored a
    full URL (Telnyx fallback) are redirected to as-is.
    """
    call = await crud.get_call_by_uuid(db, call_id)
    if not call or not call.recording_url:
        raise HTTPException(status_code=404, detail="Recording not found")

    value = call.recording_url
    if value.startswith(("http://", "https://")):
        return RedirectResponse(url=value)

    from app.services.storage_service import storage_service

    signed = await storage_service.create_signed_url(value)
    if not signed:
        raise HTTPException(status_code=404, detail="Recording unavailable")
    return RedirectResponse(url=signed)


@router.delete("/api/calls/{call_id}")
async def api_delete_call(
    call_id: uuid.UUID,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: AdminUser = Depends(require_scope("calls", edit=True)),
):
    """Soft delete a call record."""
    call = await crud.get_call_by_uuid(db, call_id)
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")

    await crud.soft_delete_call(db, call_id)
    await crud.create_audit_log(
        db,
        action="deleted_call",
        admin_user_id=user.id,
        entity_type="call",
        entity_id=call_id,
        old_value={"call_id": call.call_id, "phone": call.phone_number},
        ip_address=request.client.host if request.client else None,
    )
    return {"deleted": True}


@router.post("/api/calls/{call_id}/resend-email")
async def api_resend_email(
    call_id: uuid.UUID,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: AdminUser = Depends(require_scope("calls", edit=True)),
):
    """Re-send the screening summary email for a call."""
    call = await crud.get_call_by_uuid(db, call_id)
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")

    tenant = await crud.get_tenant_by_call(db, call_id)
    if not tenant:
        raise HTTPException(
            status_code=400, detail="No tenant data found for this call"
        )

    from app.services.email_service import send_screening_email_task

    send_screening_email_task.delay(
        call_id=str(call.id),
        phone_number=call.phone_number,
        tenant_data={
            "full_name": tenant.full_name,
            "contact_phone": tenant.contact_phone,
            "email": tenant.email,
            "adults_count": tenant.adults_count,
            "children_count": tenant.children_count,
            "occupants_count": tenant.occupants_count,
            "monthly_income": float(tenant.monthly_income)
            if tenant.monthly_income
            else None,
            "income_raw": tenant.income_raw,
            "has_pets": tenant.has_pets,
            "pets_raw": tenant.pets_raw,
            "pet_type": tenant.pet_type,
            "pet_breed": tenant.pet_breed,
            "pet_weight": tenant.pet_weight,
            "has_eviction": tenant.has_eviction,
            "eviction_circumstances": tenant.eviction_circumstances,
            "eviction_raw": tenant.eviction_raw,
            "move_in_date": str(tenant.move_in_date) if tenant.move_in_date else None,
            "move_in_raw": tenant.move_in_raw,
            "move_timing": tenant.move_timing,
            "current_residence": tenant.current_residence,
            "residence_duration": tenant.residence_duration,
            "move_reason": tenant.move_reason,
            "employer": tenant.employer,
            "employment_duration": tenant.employment_duration,
            "general_notes": tenant.general_notes,
        },
        score=tenant.qualification_score or 0,
        status=tenant.qualification_status or "review",
        reasons=tenant.disqualify_reasons or [],
        transcript=call.full_transcript or "",
        duration=call.duration_seconds or 0,
        providers={
            "stt": call.stt_provider,
            "llm": call.llm_provider,
            "tts": call.tts_provider,
        },
        bypass_filters=True,
    )

    await crud.create_audit_log(
        db,
        action="resent_email",
        admin_user_id=user.id,
        entity_type="call",
        entity_id=call_id,
        ip_address=request.client.host if request.client else None,
    )
    return {"queued": True}


@router.patch("/api/calls/{call_id}/notes")
async def api_update_call_notes(
    call_id: uuid.UUID,
    payload: dict,
    db: AsyncSession = Depends(get_db),
    user: AdminUser = Depends(require_scope("calls", edit=True)),
):
    """Update admin notes on a call's tenant record."""
    tenant = await crud.get_tenant_by_call(db, call_id)
    if not tenant:
        raise HTTPException(status_code=404, detail="No tenant record for this call")

    await crud.update_tenant(db, tenant.id, notes=payload.get("notes", ""))
    return {"saved": True}


@router.patch("/api/calls/{call_id}/review")
async def api_mark_reviewed(
    call_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user: AdminUser = Depends(require_scope("calls", edit=True)),
):
    """Toggle reviewed status on a call's tenant record."""
    tenant = await crud.get_tenant_by_call(db, call_id)
    if not tenant:
        raise HTTPException(status_code=404, detail="No tenant record for this call")

    new_status = not tenant.reviewed_by_admin
    await crud.update_tenant(
        db,
        tenant.id,
        reviewed_by_admin=new_status,
        reviewed_at=datetime.now(UTC) if new_status else None,
    )
    return {"reviewed": new_status}


@router.patch("/api/calls/{call_id}/qualification")
async def api_override_qualification(
    call_id: uuid.UUID,
    payload: dict,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: AdminUser = Depends(require_scope("calls", edit=True)),
):
    """Manually override a tenant's qualification status."""
    new_status = payload.get("status")
    if new_status not in ("qualified", "review", "unqualified"):
        raise HTTPException(status_code=400, detail="Invalid status")

    tenant = await crud.get_tenant_by_call(db, call_id)
    if not tenant:
        raise HTTPException(status_code=404, detail="No tenant record for this call")

    old_status = tenant.qualification_status
    await crud.update_tenant(db, tenant.id, qualification_status=new_status)

    await crud.create_audit_log(
        db,
        action="overrode_qualification",
        admin_user_id=user.id,
        entity_type="tenant",
        entity_id=tenant.id,
        old_value={"status": old_status},
        new_value={"status": new_status},
        ip_address=request.client.host if request.client else None,
    )
    return {"updated": True, "status": new_status}


@router.get("/api/calls/export")
async def api_export_calls_csv(
    db: AsyncSession = Depends(get_db),
    user: AdminUser = Depends(require_scope("calls")),
    status: str | None = None,
    qualification: str | None = None,
):
    """Export filtered calls as CSV."""
    calls, _ = await crud.list_calls(
        db, page=1, per_page=10000, status=status, qualification_status=qualification
    )

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "Phone Number",
            "Date",
            "Duration (s)",
            "Status",
            "Questions Answered",
            "Qualification Status",
            "Score",
            "LLM Provider",
            "Email Sent",
        ]
    )
    for c in calls:
        writer.writerow(
            [
                c.phone_number,
                c.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                c.duration_seconds or "",
                c.status,
                c.questions_answered,
                c.tenant.qualification_status if c.tenant else "",
                c.tenant.qualification_score if c.tenant else "",
                c.llm_provider or "",
                "Yes" if (c.tenant and c.tenant.email_sent) else "No",
            ]
        )

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=calls_export_{datetime.now().strftime('%Y%m%d')}.csv"
        },
    )


@router.post("/api/tenants/{tenant_id}/blacklist")
async def api_blacklist_tenant(
    tenant_id: uuid.UUID,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: AdminUser = Depends(require_scope("tenants", edit=True)),
):
    """Blacklist a tenant's phone number."""
    from sqlalchemy import select

    from app.models.tenant import Tenant

    result = await db.execute(select(Tenant).where(Tenant.id == tenant_id))
    tenant = result.scalar_one_or_none()
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")

    blacklist = await crud.get_setting_value(db, "blacklisted_numbers", [])
    if tenant.phone_number not in blacklist:
        blacklist.append(tenant.phone_number)
        import json

        await crud.set_setting(
            db, "blacklisted_numbers", json.dumps(blacklist), updated_by=user.id
        )

    await crud.update_tenant(db, tenant_id, is_blacklisted=True)
    await crud.create_audit_log(
        db,
        action="blacklisted_number",
        admin_user_id=user.id,
        entity_type="tenant",
        entity_id=tenant_id,
        new_value={"phone_number": tenant.phone_number},
        ip_address=request.client.host if request.client else None,
    )
    return {"blacklisted": True}


@router.patch("/api/tenants/{tenant_id}/notes")
async def api_update_tenant_notes(
    tenant_id: uuid.UUID,
    payload: dict,
    db: AsyncSession = Depends(get_db),
    user: AdminUser = Depends(require_scope("tenants", edit=True)),
):
    """Update notes on a tenant profile."""
    await crud.update_tenant(db, tenant_id, notes=payload.get("notes", ""))
    return {"saved": True}


@router.get("/api/analytics/export")
async def api_export_analytics_csv(
    db: AsyncSession = Depends(get_db),
    user: AdminUser = Depends(require_scope("analytics")),
    days: int = 30,
):
    """Export analytics raw data as CSV."""
    analytics = await crud.get_analytics_data(db, days=days)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Date", "Call Count"])
    for row in analytics["calls_by_day"]:
        writer.writerow([row["date"], row["count"]])

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=analytics_{datetime.now().strftime('%Y%m%d')}.csv"
        },
    )


@router.get("/api/analytics/data")
async def api_get_analytics(
    db: AsyncSession = Depends(get_db),
    user: AdminUser = Depends(require_scope("analytics")),
    days: int = 30,
):
    """Get analytics data as JSON for Chart.js rendering."""
    return await crud.get_analytics_data(db, days=days)


# ──────────────────────────────────────────────────────────────────────────────
# Update Endpoints (Admin Editing)
# ──────────────────────────────────────────────────────────────────────────────


class TenantUpdateRequest(BaseModel):
    full_name: str | None = None
    contact_phone: str | None = None
    email: str | None = None
    monthly_income: int | None = None
    adults_count: int | None = None
    children_count: int | None = None
    occupants_count: int | None = None
    move_in_raw: str | None = None
    move_reason: str | None = None
    move_timing: str | None = None
    current_residence: str | None = None
    residence_duration: str | None = None
    employer: str | None = None
    employment_duration: str | None = None
    general_notes: str | None = None
    notes: str | None = None
    has_eviction: bool | None = None
    has_pets: bool | None = None


@router.put("/api/tenants/{tenant_id}")
async def api_update_tenant(
    tenant_id: uuid.UUID,
    payload: TenantUpdateRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: AdminUser = Depends(require_scope("tenants", edit=True)),
):
    """Update tenant information."""
    from sqlalchemy import select

    from app.models.tenant import Tenant

    result = await db.execute(select(Tenant).where(Tenant.id == tenant_id))
    tenant = result.scalar_one_or_none()
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")

    update_data = {}
    for field, value in payload.model_dump(exclude_unset=True).items():
        update_data[field] = value

    rescore = {}
    if update_data:
        await crud.update_tenant(db, tenant_id, **update_data)

        # Re-run qualification scoring so manual edits (income, eviction, etc.)
        # are reflected in the stored score/status instead of going stale.
        rescore = await _rescore_tenant(db, tenant_id)

        await crud.create_audit_log(
            db,
            action="updated_tenant",
            admin_user_id=user.id,
            entity_type="tenant",
            entity_id=tenant_id,
            old_value={},
            new_value={**update_data, **rescore},
            ip_address=request.client.host if request.client else None,
        )

    return {
        "success": True,
        "updated_fields": list(update_data.keys()),
        **rescore,
    }


async def _rescore_tenant(db: AsyncSession, tenant_id: uuid.UUID) -> dict:
    """Recalculate and persist a tenant's qualification score after an edit."""
    from sqlalchemy import select

    from app.core.qualifier import calculate_qualification_score
    from app.models.tenant import Tenant

    result = await db.execute(select(Tenant).where(Tenant.id == tenant_id))
    tenant = result.scalar_one_or_none()
    if not tenant:
        return {}

    questions_answered = 0
    if tenant.call_id:
        call = await crud.get_call_by_uuid(db, tenant.call_id)
        if call:
            questions_answered = call.questions_answered or 0

    tenant_data = {
        "monthly_income": float(tenant.monthly_income)
        if tenant.monthly_income is not None
        else None,
        "income_raw": tenant.income_raw,
        "has_eviction": tenant.has_eviction,
        "eviction_circumstances": tenant.eviction_circumstances,
        "eviction_raw": tenant.eviction_raw,
        "current_residence": tenant.current_residence,
        "residence_duration": tenant.residence_duration,
        "move_reason": tenant.move_reason,
        "move_in_date": tenant.move_in_date,
        "move_in_raw": tenant.move_in_raw,
        "move_timing": tenant.move_timing,
        "occupants_count": tenant.occupants_count,
        "adults_count": tenant.adults_count,
        "children_count": tenant.children_count,
        "has_pets": tenant.has_pets,
        "pet_type": tenant.pet_type,
        "pets_raw": tenant.pets_raw,
        "pet_weight": tenant.pet_weight,
        "general_notes": tenant.general_notes,
        "questions_answered": questions_answered,
    }

    settings_map = await crud.get_all_settings(db)
    score, status, reasons = calculate_qualification_score(tenant_data, settings_map)
    await crud.update_tenant(
        db,
        tenant_id,
        qualification_score=score,
        qualification_status=status,
        disqualify_reasons=reasons if reasons else None,
    )
    return {"qualification_score": score, "qualification_status": status}


# ──────────────────────────────────────────────────────────────────────────────
# User management (super admin only)
# ──────────────────────────────────────────────────────────────────────────────


def _validate_scopes(role: str, scopes: list[str]) -> list[str]:
    """Validate and normalize scope list for staff/viewer roles."""
    from app.models.user import ALL_SCOPES, ASSIGNABLE_ROLES

    if role not in ASSIGNABLE_ROLES:
        raise HTTPException(status_code=400, detail=f"Invalid role: {role}")
    cleaned = sorted({s.strip() for s in scopes if s.strip()})
    invalid = set(cleaned) - ALL_SCOPES
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown access areas: {', '.join(sorted(invalid))}",
        )
    if role in ("staff", "viewer") and not cleaned:
        raise HTTPException(
            status_code=400,
            detail="Pick at least one access area for staff and viewer accounts.",
        )
    return cleaned


class AdminUserCreateRequest(BaseModel):
    email: str
    password: str
    full_name: str
    role: str = "staff"
    scopes: list[str] = []


class AdminUserUpdateRequest(BaseModel):
    full_name: str | None = None
    role: str | None = None
    scopes: list[str] | None = None
    is_active: bool | None = None


@router.post("/api/users")
async def api_create_user(
    payload: AdminUserCreateRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: AdminUser = Depends(require_role("super_admin")),
):
    """Create a new admin account (super admin only).

    super_admin accounts cannot be created here — the env-seeded account is the
    only super admin. Pick admin / staff / viewer and, for staff & viewer, choose
    which areas of the panel they may access.
    """
    from app.models.user import ASSIGNABLE_ROLES
    from app.utils.security import hash_password

    role = payload.role.strip().lower()
    if role not in ASSIGNABLE_ROLES:
        raise HTTPException(status_code=400, detail=f"Invalid role: {role}")

    if len(payload.password) < 8:
        raise HTTPException(
            status_code=400, detail="Password must be at least 8 characters."
        )

    email = payload.email.lower().strip()
    if await crud.get_user_by_email(db, email):
        raise HTTPException(status_code=409, detail="Email already in use.")

    scopes = _validate_scopes(role, payload.scopes)
    # admin role gets all scopes implicitly; store None in DB.
    stored_scopes = scopes if role in ("staff", "viewer") else None

    new_user = await crud.create_user(
        db,
        email=email,
        hashed_password=hash_password(payload.password),
        full_name=payload.full_name.strip() or email,
        role=role,
        permissions=stored_scopes,
    )

    await crud.create_audit_log(
        db,
        action="created_admin_user",
        admin_user_id=user.id,
        entity_type="admin_user",
        entity_id=new_user.id,
        new_value={"email": email, "role": role, "scopes": stored_scopes or "all"},
        ip_address=request.client.host if request.client else None,
    )

    logger.info("Admin user %s created by %s (role=%s)", email, user.email, role)
    invalidate_user_cache(new_user.id)
    return {
        "success": True,
        "user": {
            "id": str(new_user.id),
            "email": new_user.email,
            "full_name": new_user.full_name,
            "role": new_user.role,
            "scopes": list(new_user.effective_scopes),
        },
    }


@router.patch("/api/users/{user_id}")
async def api_update_user(
    user_id: uuid.UUID,
    payload: AdminUserUpdateRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: AdminUser = Depends(require_role("super_admin")),
):
    """Update an admin account's name, role, access areas, or active status."""
    from app.models.user import ASSIGNABLE_ROLES

    target = await crud.get_user_by_id(db, user_id)
    if not target:
        raise HTTPException(status_code=404, detail="User not found")

    if target.role == "super_admin":
        raise HTTPException(
            status_code=400,
            detail="The super admin account is managed via environment variables "
            "and can't be edited here.",
        )

    old_value = {
        "email": target.email,
        "role": target.role,
        "scopes": list(target.effective_scopes),
        "is_active": target.is_active,
    }

    new_role = payload.role.strip().lower() if payload.role else target.role
    if new_role not in ASSIGNABLE_ROLES:
        raise HTTPException(status_code=400, detail=f"Invalid role: {new_role}")

    perm_update: list[str] | None = None  # None = leave permissions unchanged
    if payload.role is not None:
        if new_role in ("staff", "viewer"):
            if payload.scopes is not None:
                perm_update = _validate_scopes(new_role, payload.scopes)
            else:
                perm_update = list(target.effective_scopes)
                if not perm_update:
                    raise HTTPException(
                        status_code=400,
                        detail="Pick at least one access area for staff and viewer accounts.",
                    )
        else:
            perm_update = []  # admin — implicit full access, clear stored scopes
    elif payload.scopes is not None:
        perm_update = _validate_scopes(new_role, payload.scopes)

    updated = await crud.update_user(
        db,
        user_id,
        full_name=payload.full_name,
        role=new_role if payload.role else None,
        permissions=perm_update,
        is_active=payload.is_active,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="User not found")

    await crud.create_audit_log(
        db,
        action="updated_admin_user",
        admin_user_id=user.id,
        entity_type="admin_user",
        entity_id=user_id,
        old_value=old_value,
        new_value={
            "role": updated.role,
            "scopes": list(updated.effective_scopes),
            "is_active": updated.is_active,
        },
        ip_address=request.client.host if request.client else None,
    )

    invalidate_user_cache(user_id)
    return {
        "success": True,
        "user": {
            "id": str(updated.id),
            "email": updated.email,
            "full_name": updated.full_name,
            "role": updated.role,
            "scopes": list(updated.effective_scopes),
            "is_active": updated.is_active,
        },
    }


@router.delete("/api/users/{user_id}")
async def api_delete_user(
    user_id: uuid.UUID,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: AdminUser = Depends(require_role("super_admin")),
):
    """Delete an admin account (super admin only).

    Guard rails:
    - You cannot delete your own account (avoid locking yourself out mid-session).
    - You cannot delete the last active super admin (avoid orphaning the system).
    The deleted user's audit-log entries and setting changes are kept — those
    foreign keys are ON DELETE SET NULL, so the history survives, just shown as
    an unnamed user.
    """
    target = await crud.get_user_by_id(db, user_id)
    if not target:
        raise HTTPException(status_code=404, detail="User not found")

    if target.id == user.id:
        raise HTTPException(
            status_code=400,
            detail="You can't delete your own account while signed in.",
        )

    if target.role == "super_admin":
        remaining = await crud.count_active_super_admins(db)
        # If this target is the only active super admin, block the delete.
        if remaining <= 1:
            raise HTTPException(
                status_code=400,
                detail="Can't delete the last super admin. Promote another "
                "account to super admin first.",
            )

    deleted_email = target.email
    await crud.delete_user(db, user_id)
    invalidate_user_cache(user_id)

    await crud.create_audit_log(
        db,
        action="deleted_admin_user",
        admin_user_id=user.id,
        entity_type="admin_user",
        entity_id=user_id,
        old_value={"email": deleted_email, "role": target.role},
        new_value={},
        ip_address=request.client.host if request.client else None,
    )

    logger.info("Admin user %s deleted by %s", deleted_email, user.email)
    return {"success": True, "deleted": deleted_email}
