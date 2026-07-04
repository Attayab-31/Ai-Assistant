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
    get_current_user_optional,
    invalidate_user_cache,
    require_role,
    require_scope,
)
from app.utils.helpers import (
    audit_action_choices,
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
    localtime,
    pagination_url,
    list_filter_url,
    score_color,
    status_badge_color,
    tenant_display_name,
    time_ago,
    date_range_from_days,
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
templates.env.filters["localtime"] = localtime
templates.env.filters["status_color"] = status_badge_color
templates.env.filters["score_color"] = score_color
templates.env.filters["friendly_state"] = friendly_state
templates.env.filters["friendly_call_status"] = friendly_call_status
templates.env.filters["friendly_qualification"] = friendly_qualification
templates.env.filters["glossary_label"] = glossary_label
templates.env.filters["glossary_tip"] = glossary_tip
templates.env.filters["friendly_provider"] = friendly_provider_name
templates.env.filters["friendly_audit_action"] = friendly_audit_action
templates.env.filters["friendly_audit_entity"] = friendly_audit_entity
templates.env.filters["pagination_url"] = pagination_url
templates.env.filters["list_filter_url"] = list_filter_url
templates.env.globals["list_filter_url"] = list_filter_url
templates.env.filters["tenant_display"] = tenant_display_name

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
            from app.core.call_handler import get_active_sessions

            active_count = len(get_active_sessions())
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
        from config import settings as env_settings

        from app.utils.helpers import build_onboarding_checklist

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

    from app.core.question_flow import (
        field_labels_from_questions,
        normalize_questions,
        questions_snapshot_from_tenant,
        build_applicant_summary_rows,
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
        score_breakdown = get_score_breakdown(
            scoring_data, all_settings, questions=snapshot
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

    from app.utils.helpers import parse_transcript_lines

    error_log = call.error_log if isinstance(call.error_log, dict) else {}
    turn_traces = error_log.get("turn_traces") or []
    trace_errors = error_log.get("errors") or []

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
        transcript_lines=parse_transcript_lines(call.full_transcript),
        turn_traces=turn_traces,
        trace_errors=trace_errors,
        summary_rows=build_applicant_summary_rows(tenant, snapshot)
        if tenant
        else [],
        active_page="calls",
        nav_prev_id=nav_prev_id,
        nav_next_id=nav_next_id,
        nav_queue="unreviewed",
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
        from app.core.question_flow import (
            build_applicant_summary_rows,
            build_flow_rows,
            field_labels_from_questions,
            questions_snapshot_from_tenant,
        )

        snapshot = questions_snapshot_from_tenant(tenant)
        if snapshot:
            field_labels = field_labels_from_questions(snapshot)
            flow_rows = build_flow_rows(
                snapshot, tenant.answered_states, tenant.refused_states
            )

    has_snapshot = snapshot is not None
    active_question_count = sum(1 for q in snapshot if q.get("active", True)) if snapshot else 0

    linked_call = None
    if tenant.call_id:
        linked_call = await crud.get_call_by_uuid(db, tenant.call_id)

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
        active_page="tenants",
        nav_prev_id=nav_prev_id,
        nav_next_id=nav_next_id,
        nav_queue="unreviewed",
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
        "review": qual.get("review", 0),
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
    super_admin_count = await crud.count_active_super_admins(db)
    from app.models.user import PERMISSION_SCOPES

    return await _render_admin_page(
        db,
        request,
        "account.html",
        user,
        users=users,
        super_admin_count=super_admin_count,
        permission_scopes=PERMISSION_SCOPES,
        active_page="account",
    )


# ──────────────────────────────────────────────────────────────────────────────
# JSON API Routes
# ──────────────────────────────────────────────────────────────────────────────


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
                    "avg_turn_latency_ms": s.get("avg_turn_latency_ms") or 0,
                    "last_turn_latency_ms": s.get("last_turn_latency_ms") or 0,
                    "avg_llm_latency_ms": s.get("avg_llm_latency_ms") or 0,
                    "avg_tts_latency_ms": s.get("avg_tts_latency_ms") or 0,
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

        usage = await get_provider_overview(db, days=30, status=provider_status)
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
    """Permanently delete a call, its applicant (CASCADE), and its recording."""
    call = await crud.get_call_by_uuid(db, call_id)
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")

    audit_value = {"call_id": call.call_id, "phone": call.phone_number}
    recording_url = call.recording_url

    # Remove the stored recording first (best-effort), then the DB rows.
    if recording_url:
        from app.services.storage_service import storage_service

        await storage_service.delete_recording(recording_url)

    await crud.hard_delete_call(db, call_id)
    await crud.create_audit_log(
        db,
        action="deleted_call",
        admin_user_id=user.id,
        entity_type="call",
        entity_id=call_id,
        old_value=audit_value,
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

    reason = (payload.get("reason") or "").strip()

    tenant = await crud.get_tenant_by_call(db, call_id)
    if not tenant:
        raise HTTPException(status_code=404, detail="No tenant record for this call")

    old_status = tenant.qualification_status
    flags = dict(tenant.control_flags or {})
    flags["qualification_status_overridden"] = True
    update_kwargs: dict = {
        "qualification_status": new_status,
        "control_flags": flags,
    }
    if reason:
        existing = (tenant.notes or "").strip()
        line = f"[Manual result override: {old_status} → {new_status}] {reason}"
        update_kwargs["notes"] = f"{existing}\n\n{line}".strip() if existing else line

    await crud.update_tenant(db, tenant.id, **update_kwargs)

    await crud.create_audit_log(
        db,
        action="overrode_qualification",
        admin_user_id=user.id,
        entity_type="tenant",
        entity_id=tenant.id,
        old_value={"status": old_status},
        new_value={"status": new_status, "reason": reason or None},
        ip_address=request.client.host if request.client else None,
    )
    return {"updated": True, "status": new_status}


@router.patch("/api/tenants/{tenant_id}/review")
async def api_mark_tenant_reviewed(
    tenant_id: uuid.UUID,
    payload: dict | None = None,
    db: AsyncSession = Depends(get_db),
    user: AdminUser = Depends(require_scope("tenants", edit=True)),
):
    """Set reviewed status on an applicant (list inline actions)."""
    tenant = await crud.get_tenant_by_id(db, tenant_id)
    if not tenant:
        raise HTTPException(status_code=404, detail="Applicant not found")
    payload = payload or {}
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
    return {"reviewed": new_status, "tenant_id": str(tenant.id)}


@router.post("/api/tenants/bulk-review")
async def api_bulk_mark_reviewed(
    payload: dict,
    db: AsyncSession = Depends(get_db),
    user: AdminUser = Depends(require_scope("tenants", edit=True)),
):
    """Mark multiple applicants reviewed in one action."""
    raw_ids = payload.get("tenant_ids") or []
    reviewed = bool(payload.get("reviewed", True))
    tenant_ids: list[uuid.UUID] = []
    for raw in raw_ids:
        try:
            tenant_ids.append(uuid.UUID(str(raw)))
        except ValueError:
            continue
    updated = await crud.bulk_set_tenants_reviewed(db, tenant_ids, reviewed=reviewed)
    return {"updated": updated, "reviewed": reviewed}


@router.get("/api/calls/export")
async def api_export_calls_csv(
    db: AsyncSession = Depends(get_db),
    user: AdminUser = Depends(require_scope("calls")),
    status: str | None = None,
    qualification: str | None = None,
    phone: str | None = None,
):
    """Export filtered calls as CSV."""
    calls, _ = await crud.list_calls(
        db,
        page=1,
        per_page=10000,
        status=status,
        qualification_status=qualification,
        phone_search=phone,
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
    tenant = await crud.get_tenant_by_id(db, tenant_id)
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")

    await crud.add_to_blacklist(db, tenant.phone_number, updated_by=user.id)
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
    tenant = await crud.get_tenant_by_id(db, tenant_id)
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
