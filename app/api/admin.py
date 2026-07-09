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
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import crud
from app.db.database import get_db
from app.models.user import AdminUser
from app.utils.dependencies import (
    get_current_user_optional,
    invalidate_user_cache,
    require_role,
    require_scope,
)
from app.utils.helpers import (
    audit_action_choices,
    date_range_from_days,
    format_currency,
    format_duration,
    format_phone_display,
    friendly_audit_action,
    friendly_audit_entity,
    friendly_call_status,
    friendly_provider_name,
    friendly_qualification,
    friendly_state,
    glossary_label,
    glossary_tip,
    list_filter_url,
    localtime,
    pagination_url,
    score_color,
    status_badge_color,
    tenant_display_name,
    time_ago,
)
from app.services.admin_audit_helpers import (
    audit_client_ip,
    format_audit_change_summary as _format_audit_change_summary,
)


def _audit_change_summary_jinja(values: object) -> str:
    if isinstance(values, (list, tuple)) and len(values) == 2:
        old_value, new_value = values
    else:
        old_value, new_value = None, None
    return _format_audit_change_summary(old_value, new_value)

logger = logging.getLogger(__name__)
router = APIRouter()
MAX_ADMIN_NOTES_LENGTH = 4000
MAX_BULK_REVIEW_IDS = 500

_AUDIT_STALE_WARNING = (
    "Action completed, but audit logging failed. Check server logs."
)


def _add_audit_warning(response: dict, audit_ok: bool) -> dict:
    """Attach a standard warning when audit-log persistence failed."""
    if not audit_ok:
        existing = response.get("warnings")
        if isinstance(existing, list):
            if _AUDIT_STALE_WARNING not in existing:
                existing.append(_AUDIT_STALE_WARNING)
        else:
            response["warnings"] = [_AUDIT_STALE_WARNING]
    return response


async def _safe_create_audit_log(*args, **kwargs) -> bool:
    """Try to persist audit metadata without failing a successful admin mutation."""
    if args:
        if len(args) == 1 and "db" not in kwargs:
            kwargs["db"] = args[0]
        else:
            logger.error("Invalid _safe_create_audit_log call signature")
            return False
    try:
        await crud.create_audit_log(**kwargs)
        return True
    except Exception as exc:
        logger.error("Audit log write failed after admin action: %s", exc)
        return False


TEMPLATES_DIR = Path(__file__).parent.parent / "admin" / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Register custom Jinja2 filters (list_filter_url is a global — templates call it directly)
_ADMIN_JINJA_FILTERS = {
    "duration": format_duration,
    "currency": format_currency,
    "phone_display": format_phone_display,
    "time_ago": time_ago,
    "localtime": localtime,
    "status_color": status_badge_color,
    "score_color": score_color,
    "friendly_state": friendly_state,
    "friendly_call_status": friendly_call_status,
    "friendly_qualification": friendly_qualification,
    "glossary_label": glossary_label,
    "glossary_tip": glossary_tip,
    "friendly_provider": friendly_provider_name,
    "friendly_audit_action": friendly_audit_action,
    "friendly_audit_entity": friendly_audit_entity,
    "audit_change_summary": _audit_change_summary_jinja,
    "pagination_url": pagination_url,
    "tenant_display": tenant_display_name,
}
for _filter_name, _filter_fn in _ADMIN_JINJA_FILTERS.items():
    templates.env.filters[_filter_name] = _filter_fn
templates.env.globals["list_filter_url"] = list_filter_url

# Contextual in-app help links (href, label)
PAGE_HELP: dict[str, tuple[str, str]] = {
    "dashboard": ("/admin/settings/questions/guide", "How screening works"),
    "calls": ("/admin/settings/questions/guide", "How screening works"),
    "tenants": ("/admin/settings/questions/guide", "How screening works"),
    "settings": ("/admin/settings/questions/guide", "Questions guide"),
    "analytics": ("/admin/analytics", "Analytics"),
    "monitor": ("/admin/monitor", "Live monitor"),
    "audit_log": ("/admin/audit-log", "Activity log"),
    "account": ("/admin/account", "Team accounts"),
}


async def _render_admin_page(
    db: AsyncSession,
    request: Request,
    template_name: str,
    user: AdminUser | None,
    **context,
):
    """Render an admin template with shared nav context."""
    needs_review_count = 0
    if user and user.can("tenants"):
        needs_review_count = await crud.count_tenants_needing_review(db)
    active = context.get("active_page")
    help_href, help_label = PAGE_HELP.get(active or "", ("", ""))
    if user and user.can("settings"):
        from app.core.question_flow import (
            count_missing_spanish_question_overrides,
            runtime_question_errors,
        )

        questions = await crud.get_setting_value(db, "screening_questions", [])
        raw = questions if isinstance(questions, list) else None
        extra: dict = {}
        if "questions_runtime_errors" not in context:
            extra["questions_runtime_errors"] = runtime_question_errors(raw)
        if "questions_spanish_warnings" not in context:
            missing_es = count_missing_spanish_question_overrides(raw)
            if missing_es:
                extra["questions_spanish_warnings"] = [
                    f"{missing_es} active question{'s' if missing_es != 1 else ''} "
                    "have no Spanish wording. Spanish calls will read those "
                    "questions in English until you add Spanish text."
                ]
            else:
                extra["questions_spanish_warnings"] = []
        context = {**context, **extra}
    from app.core.redis_client import sync_display_timezone_from_redis

    await sync_display_timezone_from_redis()
    return templates.TemplateResponse(
        template_name,
        {
            "request": request,
            "user": user,
            "needs_review_count": needs_review_count,
            "page_help_href": help_href,
            "page_help_label": help_label,
            **context,
        },
    )


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

    can_calls = user.can("calls")
    can_monitor = user.can("monitor")
    can_tenants = user.can("tenants")
    can_analytics = user.can("analytics")
    can_settings = user.can("settings")

    stats = await crud.get_call_stats(db) if can_calls else {}
    recent_calls: list = []
    if can_calls:
        recent_calls, _ = await crud.list_calls(db, page=1, per_page=10)

    active_count = 0
    if can_monitor:
        try:
            from app.core.call_handler import list_monitor_sessions

            active_count = len(await list_monitor_sessions())
        except (ImportError, AttributeError) as e:
            logger.debug("Could not fetch active sessions: %s", e)

    analytics_preview = None
    if can_analytics:
        analytics_preview = await crud.get_analytics_data(db, days=7)

    needs_review_count = 0
    reviewed_applicants = 0
    if can_tenants:
        needs_review_count = await crud.count_tenants_needing_review(db)
        reviewed_applicants = await crud.count_reviewed_tenants(db)

    onboarding = {"show": False, "steps": [], "complete": True, "done_count": 0, "total_count": 0}
    if can_settings or can_tenants:
        from app.utils.helpers import build_onboarding_checklist
        from config import settings as env_settings

        property_name = await crud.get_setting_value(
            db, "property_name", env_settings.default_property_name
        )
        greeting_message = await crud.get_setting_value(db, "greeting_message", "")
        closing_message = await crud.get_setting_value(db, "closing_message", "")
        landlord_email = await crud.get_setting_value(db, "landlord_email", "")
        property_settings_saved = await crud.settings_touched_by_admin(db)
        onboarding = build_onboarding_checklist(
            property_name=str(property_name or ""),
            greeting_message=str(greeting_message or ""),
            closing_message=str(closing_message or ""),
            landlord_email=str(landlord_email or ""),
            default_property_name=env_settings.default_property_name,
            property_settings_saved=property_settings_saved,
            total_calls=int(stats.get("total_calls") or 0) if can_calls else 0,
            reviewed_applicants=reviewed_applicants,
            needs_review_count=needs_review_count,
            can_settings=can_settings,
            can_edit=bool(user.can_edit),
            can_tenants=can_tenants,
        )

    celery_health = None
    if can_monitor or can_settings:
        try:
            from app.services.celery_health import check_celery_health

            celery_health = await check_celery_health()
        except Exception as e:
            logger.debug("Dashboard Celery health check failed: %s", e)
            celery_health = {"ok": False, "workers": 0, "detail": "Unavailable"}

    from config import settings as env_settings

    return await _render_admin_page(
        db,
        request,
        "dashboard.html",
        user,
        stats=stats,
        recent_calls=recent_calls,
        active_count=active_count,
        analytics_preview=analytics_preview,
        onboarding=onboarding,
        can_calls=can_calls,
        can_monitor=can_monitor,
        can_tenants=can_tenants,
        can_analytics=can_analytics,
        can_settings=can_settings,
        celery_health=celery_health,
        is_production=env_settings.is_production,
        active_page="dashboard",
    )


@router.get("/calls", response_class=HTMLResponse, include_in_schema=False)
async def calls_list_page(
    request: Request,
    db: AsyncSession = Depends(get_db),
    page: int = 1,
    status_filter: str | None = Query(None, alias="status"),
    qualification: str | None = None,
    phone: str | None = None,
    q: str | None = None,
    days: int | None = None,
):
    """Render the all calls list page."""
    user = await get_current_user_optional(request, db)
    if guard := _guard_page(user, "calls"):
        return guard

    date_from, date_to = date_range_from_days(days)
    search = (q or phone or "").strip() or None
    calls, total = await crud.list_calls(
        db,
        page=page,
        per_page=20,
        status=status_filter,
        qualification_status=qualification,
        text_search=search,
        date_from=date_from,
        date_to=date_to,
    )

    return await _render_admin_page(
        db,
        request,
        "calls/list.html",
        user,
        calls=calls,
        total=total,
        page=page,
        total_pages=max(1, (total + 19) // 20),
        active_page="calls",
        filters={
            "status": status_filter,
            "qualification": qualification,
            "phone": phone,
            "q": q,
            "days": days,
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

    from app.core.call_settings import has_notification_settings_snapshot
    from app.core.question_flow import (
        build_applicant_summary_rows,
        field_labels_from_questions,
        normalize_questions,
        questions_snapshot_from_tenant,
    )

    questions_config = await crud.get_setting_value(db, "screening_questions", [])
    normalized = normalize_questions(questions_config)
    snapshot = questions_snapshot_from_tenant(tenant) or normalized
    field_labels = field_labels_from_questions(snapshot)

    active_question_count = sum(1 for q in snapshot if q.get("active", True))
    flow_rows: list[dict] = []
    score_breakdown = None
    custom_fields: dict = {}
    if tenant:
        from app.core.qualifier import build_tenant_scoring_data, get_score_breakdown
        from app.core.question_flow import build_flow_rows, count_active_questions

        # One authoritative reconstruction of the data this tenant was scored on.
        scoring_data = build_tenant_scoring_data(
            tenant, questions_answered=call.questions_answered or 0
        )
        skip = set(tenant.refused_states or [])
        active_question_count = count_active_questions(
            scoring_data, skip, questions=snapshot
        )

        flow_rows = build_flow_rows(
            snapshot,
            tenant.answered_states,
            tenant.refused_states,
            scoring_data=scoring_data,
        )

        all_settings = await crud.get_all_settings(db)
        from app.core.question_flow import scoring_thresholds_from_tenant

        score_breakdown = get_score_breakdown(
            scoring_data,
            scoring_thresholds_from_tenant(tenant, fallback_settings=all_settings),
            questions=snapshot,
        )
        # The page headline shows the score STORED at finalize. Keep the
        # breakdown panel in agreement with it (settings may have changed since
        # the call), so the two can never contradict each other again.
        if tenant.qualification_score is not None:
            score_breakdown["score"] = tenant.qualification_score
        if tenant.qualification_status:
            score_breakdown["status"] = tenant.qualification_status
        # Attach the human-readable question text to each scored row so the
        # template can render a readable table instead of a raw dict dump.
        if score_breakdown.get("questions"):
            label_by_state = {
                str(q.get("state")): q.get("question") for q in snapshot
            }
            for row in score_breakdown["questions"]:
                row["question"] = label_by_state.get(
                    str(row.get("state")), row.get("state")
                )

        if isinstance(tenant.normalized_data, dict):
            cf = tenant.normalized_data.get("custom_fields")
            if isinstance(cf, dict):
                custom_fields = cf

    from app.utils.helpers import parse_transcript_lines, side_effect_alerts_from_error_log

    error_log = call.error_log if isinstance(call.error_log, dict) else {}
    turn_traces = error_log.get("turn_traces") or []
    trace_errors = error_log.get("errors") or []
    side_effect_alerts = side_effect_alerts_from_error_log(error_log)

    nav_prev_id = nav_next_id = None
    if tenant:
        queue_ids = await crud.list_tenant_ids_for_navigation(db, review_filter="unreviewed")
        if tenant.id in queue_ids:
            idx = queue_ids.index(tenant.id)
            nav_prev_id = queue_ids[idx - 1] if idx > 0 else None
            nav_next_id = queue_ids[idx + 1] if idx < len(queue_ids) - 1 else None

    return await _render_admin_page(
        db,
        request,
        "calls/detail.html",
        user,
        call=call,
        tenant=tenant,
        score_breakdown=score_breakdown,
        active_question_count=active_question_count,
        custom_fields=custom_fields,
        field_labels=field_labels,
        flow_rows=flow_rows,
        flow_from_snapshot=questions_snapshot_from_tenant(tenant) is not None,
        using_legacy_question_config=tenant is not None
        and questions_snapshot_from_tenant(tenant) is None,
        using_legacy_notification_settings=tenant is not None
        and not has_notification_settings_snapshot(tenant),
        transcript_lines=parse_transcript_lines(call.full_transcript),
        turn_traces=turn_traces,
        trace_errors=trace_errors,
        side_effect_alerts=side_effect_alerts,
        summary_rows=build_applicant_summary_rows(tenant, snapshot)
        if tenant
        else [],
        active_page="calls",
        nav_prev_id=nav_prev_id,
        nav_next_id=nav_next_id,
    )


@router.get("/tenants", response_class=HTMLResponse, include_in_schema=False)
async def tenants_list_page(
    request: Request,
    db: AsyncSession = Depends(get_db),
    page: int = 1,
    qualification: str | None = None,
    phone: str | None = None,
    q: str | None = None,
    review: str | None = None,
    days: int | None = None,
):
    """Render the all tenants list page."""
    user = await get_current_user_optional(request, db)
    if guard := _guard_page(user, "tenants"):
        return guard

    review_filter = review if review in ("unreviewed", "reviewed") else None
    date_from, date_to = date_range_from_days(days)
    search = (q or phone or "").strip() or None

    tenants, total = await crud.list_tenants(
        db,
        page=page,
        per_page=20,
        qualification_status=qualification,
        text_search=search,
        review_filter=review_filter,
        date_from=date_from,
        date_to=date_to,
    )

    return await _render_admin_page(
        db,
        request,
        "tenants/list.html",
        user,
        tenants=tenants,
        total=total,
        page=page,
        total_pages=max(1, (total + 19) // 20),
        active_page="tenants",
        filters={
            "qualification": qualification,
            "phone": phone,
            "q": q,
            "review": review_filter,
            "days": days,
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

    tenant = await crud.get_tenant_by_id(db, tenant_id)
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")

    linked_call = None
    if tenant.call_id:
        linked_call = await crud.get_call_by_uuid(db, tenant.call_id)

    # Get call history for this phone number
    history_calls, _ = await crud.list_calls(
        db, page=1, per_page=50, phone_search=tenant.phone_number
    )

    custom_fields: dict = {}
    field_labels: dict = {}
    flow_rows: list[dict] = []
    snapshot = None
    if isinstance(tenant.normalized_data, dict):
        cf = tenant.normalized_data.get("custom_fields")
        if isinstance(cf, dict):
            custom_fields = cf
        from app.core.qualifier import build_tenant_scoring_data
        from app.core.question_flow import (
            build_applicant_summary_rows,
            build_flow_rows,
            build_tenant_edit_fields,
            field_labels_from_questions,
            questions_snapshot_from_tenant,
        )

        snapshot = questions_snapshot_from_tenant(tenant)
        if snapshot:
            field_labels = field_labels_from_questions(snapshot)
            scoring_data = build_tenant_scoring_data(
                tenant,
                questions_answered=linked_call.questions_answered
                if linked_call
                else 0,
            )
            flow_rows = build_flow_rows(
                snapshot,
                tenant.answered_states,
                tenant.refused_states,
                scoring_data=scoring_data,
            )

    has_snapshot = snapshot is not None
    active_question_count = sum(1 for q in snapshot if q.get("active", True)) if snapshot else 0

    nav_prev_id = nav_next_id = None
    queue_ids = await crud.list_tenant_ids_for_navigation(db, review_filter="unreviewed")
    if tenant.id in queue_ids:
        idx = queue_ids.index(tenant.id)
        nav_prev_id = queue_ids[idx - 1] if idx > 0 else None
        nav_next_id = queue_ids[idx + 1] if idx < len(queue_ids) - 1 else None

    return await _render_admin_page(
        db,
        request,
        "tenants/detail.html",
        user,
        tenant=tenant,
        linked_call=linked_call,
        active_question_count=active_question_count,
        history_calls=history_calls,
        custom_fields=custom_fields,
        field_labels=field_labels,
        flow_rows=flow_rows,
        flow_from_snapshot=has_snapshot,
        using_legacy_question_config=not has_snapshot,
        summary_rows=build_applicant_summary_rows(tenant, snapshot)
        if snapshot
        else [],
        edit_fields=build_tenant_edit_fields(tenant, snapshot) if snapshot else [],
        active_page="tenants",
        nav_prev_id=nav_prev_id,
        nav_next_id=nav_next_id,
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
    qual = analytics.get("qualification_breakdown") or {}
    range_stats = {
        "total_calls": sum(
            row.get("count", 0) for row in analytics.get("calls_by_day", [])
        ),
        "qualified": qual.get("qualified", 0),
        "unqualified": qual.get("unqualified", 0),
        "avg_qualified_score": analytics.get("avg_qualified_score", 0),
    }

    return await _render_admin_page(
        db,
        request,
        "analytics.html",
        user,
        analytics=analytics,
        range_stats=range_stats,
        days=days,
        active_page="analytics",
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

    return await _render_admin_page(
        db, request, "monitor.html", user, active_page="monitor",
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

    return await _render_admin_page(
        db,
        request,
        "audit_log.html",
        user,
        logs=logs,
        user_map=user_map,
        total=total,
        page=page,
        total_pages=max(1, (total + 49) // 50),
        action_filter=action,
        action_choices=audit_action_choices(),
        active_page="audit_log",
    )


@router.get("/settings/providers", response_class=HTMLResponse, include_in_schema=False)
async def settings_providers_page(request: Request, db: AsyncSession = Depends(get_db)):
    """Render AI providers settings page."""
    user = await get_current_user_optional(request, db)
    if guard := _guard_page(user, "settings"):
        return guard

    all_settings = await crud.get_all_settings(db)
    from app.core.call_settings import provider_key_configured_map
    from config import provider_registry

    try:
        await provider_registry.reload_from_db(db)
    except Exception as e:
        logger.warning("Provider registry reload on settings page failed: %s", e)

    provider_status = provider_registry.get_status()
    provider_keys = await provider_key_configured_map(db)

    return await _render_admin_page(
        db,
        request,
        "settings/providers.html",
        user,
        settings=all_settings,
        provider_status=provider_status,
        provider_keys=provider_keys,
        active_page="settings",
        section="providers",
    )


@router.get("/settings/questions", response_class=HTMLResponse, include_in_schema=False)
async def settings_questions_page(request: Request, db: AsyncSession = Depends(get_db)):
    """Render screening questions editor page."""
    user = await get_current_user_optional(request, db)
    if guard := _guard_page(user, "settings"):
        return guard

    questions = await crud.get_setting_value(db, "screening_questions", [])
    from app.core.question_flow import (
        ANSWER_TYPES,
        CONDITIONAL_OPERATORS,
        SCORING_RULE_TYPES,
        normalize_questions,
        question_save_warnings,
        total_enabled_scoring_points,
    )

    normalized = normalize_questions(questions)
    return await _render_admin_page(
        db,
        request,
        "settings/questions.html",
        user,
        questions=normalized,
        flow_state_count=len(normalized),
        active_flow_count=sum(1 for q in normalized if q.get("active", True)),
        scoring_points_total=total_enabled_scoring_points(normalized),
        save_warnings=question_save_warnings(normalized),
        answer_types=sorted(ANSWER_TYPES),
        conditional_operators=sorted(CONDITIONAL_OPERATORS),
        scoring_rule_types=sorted(SCORING_RULE_TYPES),
        active_page="settings",
        section="questions",
    )


def _markdown_to_html(text: str) -> str:
    """Minimal markdown → HTML for the questions admin guide."""
    import html as html_mod
    import re

    lines = text.splitlines()
    parts: list[str] = []
    in_code = False
    in_list = False

    def close_list():
        nonlocal in_list
        if in_list:
            parts.append("</ul>")
            in_list = False

    for line in lines:
        if line.startswith("```"):
            close_list()
            if not in_code:
                parts.append("<pre><code>")
                in_code = True
            else:
                parts.append("</code></pre>")
                in_code = False
            continue
        if in_code:
            parts.append(html_mod.escape(line))
            continue
        if line.startswith("### "):
            close_list()
            parts.append(f"<h3>{html_mod.escape(line[4:])}</h3>")
        elif line.startswith("## "):
            close_list()
            parts.append(f"<h2>{html_mod.escape(line[3:])}</h2>")
        elif line.startswith("# "):
            close_list()
            parts.append(f"<h1>{html_mod.escape(line[2:])}</h1>")
        elif line.strip() == "":
            close_list()
        elif line.startswith("- "):
            if not in_list:
                parts.append("<ul>")
                in_list = True
            body = html_mod.escape(line[2:])
            body = re.sub(r"`([^`]+)`", r"<code>\1</code>", body)
            parts.append(f"<li>{body}</li>")
        else:
            close_list()
            body = html_mod.escape(line)
            body = re.sub(r"`([^`]+)`", r"<code>\1</code>", body)
            parts.append(f"<p>{body}</p>")
    close_list()
    if in_code:
        parts.append("</code></pre>")
    return "\n".join(parts)


@router.get("/settings/questions/guide", response_class=HTMLResponse, include_in_schema=False)
async def settings_questions_guide_page(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Render the screening questions admin guide."""
    user = await get_current_user_optional(request, db)
    if guard := _guard_page(user, "settings"):
        return guard

    guide_path = Path(__file__).resolve().parents[2] / "docs" / "admin_questions_guide.md"
    raw = guide_path.read_text(encoding="utf-8") if guide_path.exists() else ""
    content_html = _markdown_to_html(raw) if raw else "<p class=\"muted\">Guide file not found.</p>"

    return await _render_admin_page(
        db,
        request,
        "settings/questions_guide.html",
        user,
        content_html=content_html,
        active_page="settings",
        section="questions",
    )


@router.get("/settings/faqs", response_class=HTMLResponse, include_in_schema=False)
async def settings_faqs_page(request: Request, db: AsyncSession = Depends(get_db)):
    """Render screening FAQ editor page."""
    user = await get_current_user_optional(request, db)
    if guard := _guard_page(user, "settings"):
        return guard

    faqs = await crud.get_setting_value(db, "screening_faqs", [])
    from app.core.screening_flow import normalize_faqs

    normalized = normalize_faqs(faqs)
    return await _render_admin_page(
        db,
        request,
        "settings/faqs.html",
        user,
        faqs=normalized,
        faq_topic_count=len(normalized),
        active_page="settings",
        section="faqs",
    )


@router.get("/settings/email", response_class=HTMLResponse, include_in_schema=False)
async def settings_email_page(request: Request, db: AsyncSession = Depends(get_db)):
    """Render email settings page."""
    user = await get_current_user_optional(request, db)
    if guard := _guard_page(user, "settings"):
        return guard

    all_settings = await crud.get_all_settings(db)
    return await _render_admin_page(
        db,
        request,
        "settings/email.html",
        user,
        settings=all_settings,
        active_page="settings",
        section="email",
    )


@router.get("/settings/general", response_class=HTMLResponse, include_in_schema=False)
async def settings_general_page(request: Request, db: AsyncSession = Depends(get_db)):
    """Render general settings page."""
    user = await get_current_user_optional(request, db)
    if guard := _guard_page(user, "settings"):
        return guard

    all_settings = await crud.get_all_settings(db)
    return await _render_admin_page(
        db,
        request,
        "settings/general.html",
        user,
        settings=all_settings,
        active_page="settings",
        section="general",
    )


@router.get("/account", response_class=HTMLResponse, include_in_schema=False)
async def account_page(request: Request, db: AsyncSession = Depends(get_db)):
    """Render account management page (super admin only)."""
    user = await get_current_user_optional(request, db)
    if guard := _guard_page(user, "accounts"):
        return guard

    users = await crud.list_users(db)
    from app.models.user import PERMISSION_SCOPES

    return await _render_admin_page(
        db,
        request,
        "account.html",
        user,
        users=users,
        permission_scopes=PERMISSION_SCOPES,
        active_page="account",
    )


# ──────────────────────────────────────────────────────────────────────────────
# JSON API Routes
# ──────────────────────────────────────────────────────────────────────────────


# Approx fallback for the live progress bar when session question count is unknown.
_PROGRESS_DENOMINATOR = 15


def _monitor_progress_pct(answered: int, total: int | None) -> int:
    denom = total if total and total > 0 else _PROGRESS_DENOMINATOR
    return min(100, round((answered or 0) / denom * 100))


def _csv_attachment_response(
    *,
    filename_stem: str,
    header: list[str],
    rows: list[list],
    extra_headers: dict[str, str] | None = None,
) -> StreamingResponse:
    """Build a one-shot CSV download response."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(header)
    writer.writerows(rows)
    output.seek(0)
    headers = {
        "Content-Disposition": (
            f"attachment; filename={filename_stem}_"
            f"{datetime.now().strftime('%Y%m%d')}.csv"
        )
    }
    if extra_headers:
        headers.update(extra_headers)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers=headers,
    )


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
    # 1) Live calls in progress (local + other workers via Redis when available).
    active_calls = []
    latency_slo: dict[str, float | int] = {}
    try:
        from app.core.call_handler import list_monitor_sessions

        for s in await list_monitor_sessions():
            answered = s.get("questions_answered") or 0
            total_q = s.get("active_question_count") or 0
            active_calls.append(
                {
                    "call_id": s.get("call_id"),
                    "phone_display": format_phone_display(s.get("phone_number")),
                    "state": s.get("state"),
                    "state_label": friendly_state(s.get("state")),
                    "duration_seconds": s.get("duration"),
                    "duration_label": format_duration(s.get("duration")),
                    "questions_answered": answered,
                    "progress_pct": _monitor_progress_pct(answered, total_q),
                    "started_at": s.get("started_at"),
                    "avg_turn_latency_ms": s.get("avg_turn_latency_ms") or 0,
                    "last_turn_latency_ms": s.get("last_turn_latency_ms") or 0,
                    "avg_llm_latency_ms": s.get("avg_llm_latency_ms") or 0,
                    "avg_tts_latency_ms": s.get("avg_tts_latency_ms") or 0,
                }
            )
    except Exception as e:
        logger.debug("Could not collect active sessions: %s", e)

    try:
        from app.core.call_settings import load_call_settings_snapshot

        snapshot = await load_call_settings_snapshot(db)
        latency_slo = {
            "turn_warn_ms": int(snapshot.latency_alert_turn_p95_ms),
            "turn_crit_ms": int(snapshot.latency_alert_turn_p95_crit_ms),
        }
    except Exception as e:
        logger.debug("Could not load latency SLO settings for monitor: %s", e)

    # 2) System health (DB is implicitly OK — this query ran).
    redis_ok = False
    try:
        from app.core.redis_client import ping as redis_ping

        redis_ok = await redis_ping()
    except Exception as e:
        logger.debug("Redis ping failed: %s", e)

    celery_health = {"ok": True, "broker": False, "workers": 0, "detail": ""}
    try:
        from app.services.celery_health import check_celery_health

        celery_health = await check_celery_health()
    except Exception as e:
        logger.debug("Celery health check failed: %s", e)
        celery_health = {
            "ok": False,
            "broker": False,
            "workers": 0,
            "detail": "Health check failed",
        }

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

        usage = await get_provider_overview(db, days=30, status=provider_status)
    except Exception as e:
        logger.debug("Provider usage lookup failed: %s", e)

    return {
        "active_calls": active_calls,
        "active_count": len(active_calls),
        "health": {
            "database": True,
            "redis": redis_ok,
            "celery": celery_health,
            "uptime_seconds": uptime_seconds,
            "providers": provider_status,
        },
        "stats": stats,
        "usage": usage,
        "latency_slo": latency_slo,
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
        from app.services.recording_cleanup import is_managed_recording_path
        from app.utils.security import UnsafeURLError, assert_safe_external_url

        # Legacy absolute URLs must still pass outbound URL safety checks.
        if is_managed_recording_path(value):
            raise HTTPException(status_code=404, detail="Recording unavailable")
        try:
            assert_safe_external_url(value)
        except UnsafeURLError as exc:
            raise HTTPException(status_code=404, detail="Recording unavailable") from exc
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
    """Permanently delete a call, its applicant (CASCADE), and its recording."""
    call = await crud.get_call_by_uuid(db, call_id)
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")

    audit_value = {"call_id": call.call_id, "phone": call.phone_number}
    recording_url = call.recording_url
    recording_delete_failed = False

    if recording_url:
        from app.services.recording_cleanup import (
            RecordingRemovalResult,
            enqueue_orphaned_recording,
            remove_recording,
        )

        removal = await remove_recording(recording_url)
        if removal == RecordingRemovalResult.FAILED:
            recording_delete_failed = True
            await enqueue_orphaned_recording(recording_url)

    deleted = await crud.hard_delete_call(db, call_id)
    if not deleted:
        raise HTTPException(
            status_code=503,
            detail=(
                "Call was not deleted because its recording could not be "
                "removed or queued for retry."
            ),
        )
    from app.core.redis_client import mark_call_admin_deleted

    await mark_call_admin_deleted(call.call_id)
    audit_ok = await _safe_create_audit_log(
        db,
        action="deleted_call",
        admin_user_id=user.id,
        entity_type="call",
        entity_id=call_id,
        old_value=audit_value,
        ip_address=audit_client_ip(request),
    )
    response: dict = {"deleted": True}
    if recording_delete_failed:
        response["recording_delete_failed"] = True
        response["warnings"] = [
            "Call deleted, but the stored recording could not be removed from "
            "storage. It will be retried automatically."
        ]
    return _add_audit_warning(response, audit_ok)


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

    from app.core.call_settings import (
        load_notification_settings_from_db,
        notification_settings_email_dict,
        notification_settings_email_dict_from_tenant,
    )
    from app.services.email_service import send_screening_email_task

    tenant_payload = {
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
    }
    if isinstance(tenant.normalized_data, dict):
        custom = tenant.normalized_data.get("custom_fields")
        if isinstance(custom, dict):
            tenant_payload["custom_fields"] = custom

    email_settings = notification_settings_email_dict_from_tenant(tenant)
    used_live_settings = email_settings is None
    if email_settings is None:
        email_settings = notification_settings_email_dict(
            await load_notification_settings_from_db(db)
        )

    send_screening_email_task.delay(
        call_id=str(call.id),
        phone_number=call.phone_number,
        tenant_data=tenant_payload,
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
        email_settings=email_settings,
    )

    audit_ok = await _safe_create_audit_log(
        db,
        action="resent_email",
        admin_user_id=user.id,
        entity_type="call",
        entity_id=call_id,
        ip_address=audit_client_ip(request),
    )
    return _add_audit_warning(
        {"queued": True, "used_live_settings": used_live_settings},
        audit_ok,
    )


@router.patch("/api/calls/{call_id}/notes")
async def api_update_call_notes(
    call_id: uuid.UUID,
    payload: dict,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: AdminUser = Depends(require_scope("calls", edit=True)),
):
    """Update admin notes on a call's tenant record."""
    tenant = await crud.get_tenant_by_call(db, call_id)
    if not tenant:
        raise HTTPException(status_code=404, detail="No tenant record for this call")

    notes_text = str(payload.get("notes", "") or "")
    if len(notes_text) > MAX_ADMIN_NOTES_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Notes cannot exceed {MAX_ADMIN_NOTES_LENGTH} characters.",
        )
    await crud.update_tenant(db, tenant.id, notes=notes_text)
    audit_ok = await _safe_create_audit_log(
        db,
        action="updated_call_notes",
        admin_user_id=user.id,
        entity_type="tenant",
        entity_id=tenant.id,
        new_value={"call_id": str(call_id), "notes_length": len(notes_text)},
        ip_address=audit_client_ip(request),
    )
    return _add_audit_warning({"saved": True}, audit_ok)


@router.patch("/api/calls/{call_id}/review")
async def api_mark_reviewed(
    call_id: uuid.UUID,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: AdminUser = Depends(require_scope("calls", edit=True)),
):
    """Toggle reviewed status on a call's tenant record."""
    tenant = await crud.get_tenant_by_call(db, call_id)
    if not tenant:
        raise HTTPException(status_code=404, detail="No tenant record for this call")

    old_status = bool(tenant.reviewed_by_admin)
    new_status = not old_status
    await crud.update_tenant(
        db,
        tenant.id,
        reviewed_by_admin=new_status,
        reviewed_at=datetime.now(UTC) if new_status else None,
    )
    audit_ok = await _safe_create_audit_log(
        db,
        action="toggled_call_review",
        admin_user_id=user.id,
        entity_type="tenant",
        entity_id=tenant.id,
        old_value={"reviewed": old_status},
        new_value={"reviewed": new_status, "call_id": str(call_id)},
        ip_address=audit_client_ip(request),
    )
    return _add_audit_warning({"reviewed": new_status}, audit_ok)


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

    reason = (payload.get("reason") or "").strip()

    tenant = await crud.get_tenant_by_call(db, call_id)
    if not tenant:
        raise HTTPException(status_code=404, detail="No tenant record for this call")

    old_status = tenant.qualification_status
    flags = dict(tenant.control_flags or {})
    flags["qualification_status_overridden"] = True

    qualified_threshold = await crud.get_setting_value(
        db, "qualified_score_threshold", 75
    )
    review_threshold = await crud.get_setting_value(db, "review_score_threshold", 50)
    try:
        qualified_threshold = int(qualified_threshold)
    except (TypeError, ValueError):
        qualified_threshold = 75
    try:
        review_threshold = int(review_threshold)
    except (TypeError, ValueError):
        review_threshold = 50
    score_for_status = {
        "qualified": qualified_threshold,
        "review": review_threshold,
        "unqualified": max(0, review_threshold - 1),
    }

    update_kwargs: dict = {
        "qualification_status": new_status,
        "qualification_score": score_for_status[new_status],
        "control_flags": flags,
    }
    if reason:
        existing = (tenant.notes or "").strip()
        line = f"[Manual result override: {old_status} → {new_status}] {reason}"
        update_kwargs["notes"] = f"{existing}\n\n{line}".strip() if existing else line

    await crud.update_tenant(db, tenant.id, **update_kwargs)

    audit_ok = await _safe_create_audit_log(
        db,
        action="overrode_qualification",
        admin_user_id=user.id,
        entity_type="tenant",
        entity_id=tenant.id,
        old_value={"status": old_status},
        new_value={"status": new_status, "reason": reason or None},
        ip_address=audit_client_ip(request),
    )
    return _add_audit_warning({"updated": True, "status": new_status}, audit_ok)


@router.patch("/api/tenants/{tenant_id}/review")
async def api_mark_tenant_reviewed(
    tenant_id: uuid.UUID,
    request: Request,
    payload: dict | None = None,
    db: AsyncSession = Depends(get_db),
    user: AdminUser = Depends(require_scope("tenants", edit=True)),
):
    """Set reviewed status on an applicant (list inline actions)."""
    tenant = await crud.get_tenant_by_id(db, tenant_id)
    if not tenant:
        raise HTTPException(status_code=404, detail="Applicant not found")
    payload = payload or {}
    old_reviewed = bool(tenant.reviewed_by_admin)
    if "reviewed" in payload:
        new_status = bool(payload["reviewed"])
    else:
        new_status = not tenant.reviewed_by_admin
    await crud.update_tenant(
        db,
        tenant.id,
        reviewed_by_admin=new_status,
        reviewed_at=datetime.now(UTC) if new_status else None,
    )
    audit_ok = await _safe_create_audit_log(
        db,
        action="tenant_review_updated",
        admin_user_id=user.id,
        entity_type="tenant",
        entity_id=tenant.id,
        old_value={"reviewed_by_admin": old_reviewed},
        new_value={"reviewed_by_admin": new_status},
        ip_address=audit_client_ip(request),
    )
    return _add_audit_warning(
        {"reviewed": new_status, "tenant_id": str(tenant.id)},
        audit_ok,
    )


@router.post("/api/tenants/bulk-review")
async def api_bulk_mark_reviewed(
    payload: dict,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: AdminUser = Depends(require_scope("tenants", edit=True)),
):
    """Mark multiple applicants reviewed in one action."""
    raw_ids = payload.get("tenant_ids") or []
    if len(raw_ids) > MAX_BULK_REVIEW_IDS:
        raise HTTPException(
            status_code=400,
            detail=f"Bulk review supports at most {MAX_BULK_REVIEW_IDS} applicants per request.",
        )
    reviewed = bool(payload.get("reviewed", True))
    tenant_ids: list[uuid.UUID] = []
    invalid_ids = 0
    for raw in raw_ids:
        try:
            tenant_ids.append(uuid.UUID(str(raw)))
        except ValueError:
            invalid_ids += 1
    if invalid_ids:
        raise HTTPException(
            status_code=400,
            detail=f"{invalid_ids} invalid tenant id value(s) provided.",
        )
    updated = await crud.bulk_set_tenants_reviewed(db, tenant_ids, reviewed=reviewed)
    audit_ok = await _safe_create_audit_log(
        db,
        action="tenant_bulk_review_updated",
        admin_user_id=user.id,
        entity_type="tenant",
        new_value={"reviewed": reviewed, "updated": updated, "requested": len(raw_ids)},
        ip_address=audit_client_ip(request),
    )
    return _add_audit_warning(
        {"updated": updated, "reviewed": reviewed},
        audit_ok,
    )


@router.get("/api/calls/export")
async def api_export_calls_csv(
    db: AsyncSession = Depends(get_db),
    user: AdminUser = Depends(require_scope("calls")),
    status: str | None = None,
    qualification: str | None = None,
    phone: str | None = None,
    q: str | None = None,
    days: int | None = None,
):
    """Export filtered calls as CSV."""
    date_from, date_to = date_range_from_days(days)
    search = (q or phone or "").strip() or None
    calls, total = await crud.list_calls(
        db,
        page=1,
        per_page=10000,
        status=status,
        qualification_status=qualification,
        text_search=search,
        date_from=date_from,
        date_to=date_to,
    )

    extra_headers: dict[str, str] = {"X-Export-Total-Count": str(total)}
    if total > len(calls):
        extra_headers["X-Export-Truncated"] = "true"

    rows = [
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
        for c in calls
    ]
    return _csv_attachment_response(
        filename_stem="calls_export",
        header=[
            "Phone Number",
            "Date",
            "Duration (s)",
            "Status",
            "Questions Answered",
            "Qualification Status",
            "Score",
            "LLM Provider",
            "Email Sent",
        ],
        rows=rows,
        extra_headers=extra_headers,
    )


@router.post("/api/tenants/{tenant_id}/blacklist")
async def api_blacklist_tenant(
    tenant_id: uuid.UUID,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: AdminUser = Depends(require_scope("tenants", edit=True)),
):
    """Blacklist a tenant's phone number."""
    from app.utils.helpers import sanitize_phone_number

    tenant = await crud.get_tenant_by_id(db, tenant_id)
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")

    phone = sanitize_phone_number(tenant.phone_number or "")
    if not phone:
        raise HTTPException(status_code=400, detail="Tenant has no valid phone number")

    _, cache_ok = await crud.add_to_blacklist(
        db, phone, updated_by=user.id, tenant_id=tenant_id
    )
    audit_ok = await _safe_create_audit_log(
        db,
        action="blacklisted_number",
        admin_user_id=user.id,
        entity_type="tenant",
        entity_id=tenant_id,
        new_value={"phone_number": phone},
        ip_address=audit_client_ip(request),
    )
    from app.api.settings import _add_cache_warning

    return _add_cache_warning(_add_audit_warning({"blacklisted": True}, audit_ok), cache_ok)


@router.patch("/api/tenants/{tenant_id}/notes")
async def api_update_tenant_notes(
    tenant_id: uuid.UUID,
    payload: dict,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: AdminUser = Depends(require_scope("tenants", edit=True)),
):
    """Update notes on a tenant profile."""
    tenant = await crud.get_tenant_by_id(db, tenant_id)
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")

    notes_text = str(payload.get("notes", "") or "")
    if len(notes_text) > MAX_ADMIN_NOTES_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Notes cannot exceed {MAX_ADMIN_NOTES_LENGTH} characters.",
        )
    await crud.update_tenant(db, tenant_id, notes=notes_text)
    audit_ok = await _safe_create_audit_log(
        db,
        action="updated_tenant_notes",
        admin_user_id=user.id,
        entity_type="tenant",
        entity_id=tenant_id,
        new_value={"notes_length": len(notes_text)},
        ip_address=audit_client_ip(request),
    )
    return _add_audit_warning({"saved": True}, audit_ok)


@router.get("/api/analytics/export")
async def api_export_analytics_csv(
    db: AsyncSession = Depends(get_db),
    user: AdminUser = Depends(require_scope("analytics")),
    days: int = 30,
):
    """Export analytics raw data as CSV."""
    analytics = await crud.get_analytics_data(db, days=days)

    rows = [
        [row["date"], row["count"]] for row in analytics["calls_by_day"]
    ]
    return _csv_attachment_response(
        filename_stem="analytics",
        header=["Date", "Call Count"],
        rows=rows,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Update Endpoints (Admin Editing)
# ──────────────────────────────────────────────────────────────────────────────


class TenantUpdateRequest(BaseModel):
    model_config = {"extra": "allow"}

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
    tenant = await crud.get_tenant_by_id(db, tenant_id)
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")

    update_data = {}
    custom_updates: dict = {}
    declared = set(TenantUpdateRequest.model_fields.keys())
    payload_fields = payload.model_dump(exclude_unset=True)
    for field, value in payload_fields.items():
        if str(field).startswith("custom_"):
            custom_updates[field] = value
        elif field in declared:
            update_data[field] = value
        else:
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported tenant field: {field}",
            )

    if custom_updates:
        from app.services.admin_audit_helpers import validate_custom_tenant_updates

        try:
            validate_custom_tenant_updates(custom_updates)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    from app.services.admin_audit_helpers import build_tenant_audit_old_value

    old_audit = build_tenant_audit_old_value(tenant, payload_fields)

    if custom_updates:
        nd = dict(tenant.normalized_data or {})
        cf = dict(nd.get("custom_fields") or {})
        cf.update(custom_updates)
        nd["custom_fields"] = cf
        update_data["normalized_data"] = nd

    rescore = {}
    if update_data:
        await crud.update_tenant(db, tenant_id, **update_data)

        # Re-run qualification scoring so manual edits (income, eviction, etc.)
        # are reflected in the stored score/status instead of going stale.
        rescore = await _rescore_tenant(db, tenant_id)

        new_audit = {
            k: v for k, v in payload_fields.items() if k != "normalized_data"
        }
        if rescore:
            new_audit.update(rescore)

        audit_ok = await _safe_create_audit_log(
            db,
            action="updated_tenant",
            admin_user_id=user.id,
            entity_type="tenant",
            entity_id=tenant_id,
            old_value=old_audit,
            new_value=new_audit,
            ip_address=audit_client_ip(request),
        )
    else:
        audit_ok = True

    response = {
        "success": True,
        "updated_fields": list(update_data.keys()),
        **rescore,
    }
    return _add_audit_warning(response, audit_ok)


async def _rescore_tenant(db: AsyncSession, tenant_id: uuid.UUID) -> dict:
    """Recalculate and persist a tenant's qualification score after an edit."""
    from app.core.qualifier import (
        build_tenant_scoring_data,
        calculate_qualification_score,
    )
    from app.core.question_flow import (
        normalize_questions,
        questions_snapshot_from_tenant,
        scoring_thresholds_from_tenant,
    )

    tenant = await crud.get_tenant_by_id(db, tenant_id)
    if not tenant:
        return {}

    questions_answered = 0
    if tenant.call_id:
        call = await crud.get_call_by_uuid(db, tenant.call_id)
        if call:
            questions_answered = call.questions_answered or 0

    tenant_data = build_tenant_scoring_data(
        tenant, questions_answered=questions_answered
    )

    settings_map = await crud.get_all_settings(db)
    # Re-score against the questions ACTUALLY ASKED on this call (the per-call
    # snapshot), not the current config — otherwise editing a tenant after an
    # admin changed the question set would silently rescore against a different
    # flow. Fall back to current config only for legacy rows without a snapshot.
    questions_cfg = questions_snapshot_from_tenant(tenant) or normalize_questions(
        await crud.get_setting_value(db, "screening_questions", [])
    )
    scoring_settings = scoring_thresholds_from_tenant(tenant, fallback_settings=settings_map)
    score, status, reasons = calculate_qualification_score(
        tenant_data, scoring_settings, questions=questions_cfg
    )
    flags = tenant.control_flags or {}
    status_overridden = bool(flags.get("qualification_status_overridden"))
    update_kwargs: dict = {
        "qualification_score": score,
        "disqualify_reasons": reasons if reasons else None,
    }
    if not status_overridden:
        update_kwargs["qualification_status"] = status
    else:
        status = tenant.qualification_status or status
    await crud.update_tenant(db, tenant_id, **update_kwargs)
    return {"qualification_score": score, "qualification_status": status}


# ──────────────────────────────────────────────────────────────────────────────
# User management (super admin only)
# ──────────────────────────────────────────────────────────────────────────────


def _validate_scopes(role: str, scopes: list[str]) -> list[str]:
    """Validate and normalize scope list for staff/viewer roles."""
    from app.models.user import validate_assignable_scopes

    try:
        return validate_assignable_scopes(role, scopes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


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
    password: str | None = None


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

    try:
        new_user = await crud.create_user(
            db,
            email=email,
            hashed_password=hash_password(payload.password),
            full_name=payload.full_name.strip() or email,
            role=role,
            permissions=stored_scopes,
        )
    except IntegrityError as e:
        await db.rollback()
        raise HTTPException(status_code=409, detail="Email already in use.") from e

    audit_ok = await _safe_create_audit_log(
        db,
        action="created_admin_user",
        admin_user_id=user.id,
        entity_type="admin_user",
        entity_id=new_user.id,
        new_value={"email": email, "role": role, "scopes": stored_scopes or "all"},
        ip_address=audit_client_ip(request),
    )

    logger.info("Admin user %s created by %s (role=%s)", email, user.email, role)
    invalidate_user_cache(new_user.id)
    return _add_audit_warning(
        {
            "success": True,
            "user": {
                "id": str(new_user.id),
                "email": new_user.email,
                "full_name": new_user.full_name,
                "role": new_user.role,
                "scopes": list(new_user.effective_scopes),
            },
        },
        audit_ok,
    )


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

    if target.is_env_account:
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

    if payload.password is not None:
        from app.utils.security import hash_password

        new_password = payload.password.strip()
        if len(new_password) < 8:
            raise HTTPException(
                status_code=400,
                detail="Password must be at least 8 characters.",
            )
        await crud.update_user_password(
            db, user_id, hash_password(new_password)
        )

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

    audit_ok = await _safe_create_audit_log(
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
        ip_address=audit_client_ip(request),
    )

    invalidate_user_cache(user_id)
    return _add_audit_warning(
        {
            "success": True,
            "user": {
                "id": str(updated.id),
                "email": updated.email,
                "full_name": updated.full_name,
                "role": updated.role,
                "scopes": list(updated.effective_scopes),
                "is_active": updated.is_active,
            },
        },
        audit_ok,
    )


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
    - You cannot delete the env-managed super admin (set by ADMIN_EMAIL); it is
      re-seeded from the environment anyway. Every other account, including a
      leftover super admin from a previous ADMIN_EMAIL, can be removed.
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

    # Only the env-managed super admin is protected. Any other account —
    # including a leftover super admin from a previous ADMIN_EMAIL — can be
    # removed by a super admin.
    if target.is_env_account:
        raise HTTPException(
            status_code=400,
            detail="The super admin account is set by ADMIN_EMAIL in your "
            "environment and can't be deleted here. Change ADMIN_EMAIL to move "
            "it to a different account.",
        )

    deleted_email = target.email
    await crud.delete_user(db, user_id)
    invalidate_user_cache(user_id)

    audit_ok = await _safe_create_audit_log(
        db,
        action="deleted_admin_user",
        admin_user_id=user.id,
        entity_type="admin_user",
        entity_id=user_id,
        old_value={"email": deleted_email, "role": target.role},
        new_value={},
        ip_address=audit_client_ip(request),
    )

    logger.info("Admin user %s deleted by %s", deleted_email, user.email)
    return _add_audit_warning({"success": True, "deleted": deleted_email}, audit_ok)
