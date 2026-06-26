"""
app/services/email_service.py — Resend email integration and Celery background tasks.

Handles:
- Sending screening summary emails after each call
- Daily digest emails
- CRM webhook delivery with HMAC signing
- Retry logic with exponential backoff
"""

import logging
from datetime import UTC, datetime

import resend

from app.core.celery_app import celery_app
from config import settings

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Email Template Rendering
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_EMAIL_TEMPLATE = """
<div style="font-family: -apple-system, sans-serif; max-width: 600px; margin: 0 auto;">
  <div style="background: #1e293b; padding: 24px; border-radius: 8px 8px 0 0;">
    <h1 style="color: white; margin: 0; font-size: 20px;">🏠 New Tenant Screening</h1>
  </div>
  <div style="background: white; padding: 24px; border: 1px solid #e2e8f0; border-top: none;">
    <table style="width: 100%; margin-bottom: 16px;">
      <tr><td style="color: #64748b; padding: 4px 0;">Date</td><td style="text-align:right; font-weight:600;">{date}</td></tr>
      <tr><td style="color: #64748b; padding: 4px 0;">Phone</td><td style="text-align:right; font-weight:600;">{phone_number}</td></tr>
      <tr><td style="color: #64748b; padding: 4px 0;">Duration</td><td style="text-align:right; font-weight:600;">{duration}</td></tr>
      <tr><td style="color: #64748b; padding: 4px 0;">Status</td><td style="text-align:right;"><span style="background:{status_color}; color:white; padding:4px 10px; border-radius:12px; font-size:13px; font-weight:600;">{status_label}</span></td></tr>
    </table>
    <hr style="border:none; border-top:1px solid #e2e8f0; margin: 16px 0;">
    <h3 style="color:#1e293b; font-size:15px;">Tenant Answers</h3>
    <table style="width:100%; font-size:14px;">
      <tr><td style="padding:6px 0;">👥 Household</td><td style="text-align:right;">{adults} adult(s), {children} child(ren)</td></tr>
      <tr><td style="padding:6px 0;">💰 Monthly Income</td><td style="text-align:right;">{income}</td></tr>
      <tr><td style="padding:6px 0;">⚖️ Prior Eviction</td><td style="text-align:right;">{eviction}</td></tr>
      <tr><td style="padding:6px 0;">📅 Move-In Date</td><td style="text-align:right;">{move_date}</td></tr>
      <tr><td style="padding:6px 0;">📝 Move Reason</td><td style="text-align:right;">{move_reason}</td></tr>
    </table>
    <hr style="border:none; border-top:1px solid #e2e8f0; margin: 16px 0;">
    <div style="text-align:center; padding: 12px; background:#f8fafc; border-radius:8px;">
      <span style="font-size:13px; color:#64748b;">Qualification Score</span><br>
      <span style="font-size:28px; font-weight:700; color:{status_color};">{score}/100</span>
    </div>
    {disqualify_section}
    <p style="font-size:12px; color:#94a3b8; margin-top:20px;">🤖 STT: {stt} | LLM: {llm} | TTS: {tts}</p>
    <div style="text-align:center; margin-top:20px;">
      <a href="{dashboard_url}" style="background:#3b82f6; color:white; padding:10px 20px; border-radius:6px; text-decoration:none; font-size:14px; font-weight:600;">View in Dashboard</a>
    </div>
  </div>
</div>
"""

STATUS_LABELS = {
    "qualified": ("✅ PRE-QUALIFIED", "#22c55e"),
    "review": ("⚠️ NEEDS REVIEW", "#f59e0b"),
    "unqualified": ("❌ DISQUALIFIED", "#ef4444"),
}


def _apply_tokens(text: str, tokens: dict) -> str:
    """Safely substitute {token} placeholders without breaking on stray braces."""
    out = text or ""
    for key, value in tokens.items():
        out = out.replace("{" + key + "}", str(value))
    return out


def render_email_template(
    phone_number: str,
    tenant_data: dict,
    score: int,
    status: str,
    reasons: list[str],
    duration: int,
    providers: dict,
    call_id: str,
    subject_template: str | None = None,
    body_template: str | None = None,
    include_transcript: bool = False,
    transcript: str = "",
) -> tuple[str, str]:
    """
    Render the email subject and HTML body from tenant data.

    Honors admin-configured subject/body templates when provided. Custom
    templates use simple ``{token}`` placeholders (name, phone, status, score,
    income, adults, children, eviction, move_date, move_reason, date,
    duration, dashboard_url) so they can't raise on unknown fields.

    Returns:
        Tuple of (subject, html_body)
    """
    import html as _html

    from app.utils.helpers import format_currency, format_duration

    status_label, status_color = STATUS_LABELS.get(status, ("UNKNOWN", "#64748b"))

    def _esc(value) -> str:
        """HTML-escape a value before it lands in the email body.

        Tenant fields (name, move reason, etc.) are caller-controlled via STT, so
        they must be escaped to avoid HTML/script injection into the landlord's
        inbox — both in the default template and in admin custom templates.
        """
        return _html.escape(str(value), quote=True)

    dashboard_url = f"{settings.app_url}/admin/calls/{call_id}"
    eviction_text = (
        "Yes"
        if tenant_data.get("has_eviction")
        else ("No" if tenant_data.get("has_eviction") is False else "—")
    )
    move_date_text = (
        tenant_data.get("move_in_raw") or tenant_data.get("move_in_date") or "—"
    )
    date_text = datetime.now(UTC).strftime("%B %d, %Y at %I:%M %p UTC")

    tokens = {
        "name": _esc(tenant_data.get("full_name") or "—"),
        "phone": _esc(phone_number),
        "status": status_label,
        "score": score,
        "income": _esc(format_currency(tenant_data.get("monthly_income"))),
        "adults": _esc(tenant_data.get("adults_count") or "—"),
        "children": _esc(tenant_data.get("children_count") or "0"),
        "eviction": _esc(eviction_text),
        "move_date": _esc(move_date_text),
        "move_reason": _esc(tenant_data.get("move_reason") or "—"),
        "date": date_text,
        "duration": format_duration(duration),
        "dashboard_url": dashboard_url,
    }

    disqualify_section = ""
    if reasons:
        items = "".join(f"<li style='margin:4px 0;'>{_esc(r)}</li>" for r in reasons)
        disqualify_section = f"""
        <div style="margin-top:12px; padding:12px; background:#fef2f2; border-radius:6px; border-left:3px solid #ef4444;">
          <strong style="font-size:13px; color:#991b1b;">Flags:</strong>
          <ul style="margin:6px 0 0 16px; padding:0; font-size:13px; color:#7f1d1d;">{items}</ul>
        </div>"""

    if body_template and body_template.strip():
        html = _apply_tokens(body_template, tokens)
    else:
        html = DEFAULT_EMAIL_TEMPLATE.format(
            date=date_text,
            phone_number=tokens["phone"],
            duration=format_duration(duration),
            status_label=status_label,
            status_color=status_color,
            adults=tokens["adults"],
            children=tokens["children"],
            income=tokens["income"],
            eviction=tokens["eviction"],
            move_date=tokens["move_date"],
            move_reason=tokens["move_reason"],
            score=score,
            disqualify_section=disqualify_section,
            stt=providers.get("stt", "—"),
            llm=providers.get("llm", "—"),
            tts=providers.get("tts", "—"),
            dashboard_url=dashboard_url,
        )

    if include_transcript and transcript:
        import html as _html_escape

        safe_transcript = _html_escape.escape(transcript)
        html += f"""
        <div style="max-width:600px; margin:16px auto 0;">
          <h3 style="color:#1e293b; font-size:15px;">Full Transcript</h3>
          <pre style="white-space:pre-wrap; word-break:break-word; background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:12px; font-size:12px; color:#334155;">{safe_transcript}</pre>
        </div>"""

    if subject_template and subject_template.strip():
        subject = _apply_tokens(subject_template, tokens)
    else:
        subject = f"🏠 New Tenant Screening — {phone_number} — {status_label}"
    return subject, html


# ──────────────────────────────────────────────────────────────────────────────
# Celery Tasks
# ──────────────────────────────────────────────────────────────────────────────


@celery_app.task(
    bind=True,
    max_retries=3,
    default_retry_delay=10,
    # The Resend SDK has no request timeout; cap the task so a hung HTTP call
    # can't pin a worker slot indefinitely.
    soft_time_limit=30,
    time_limit=45,
    name="app.services.email_service.send_screening_email_task",
)
def send_screening_email_task(
    self,
    call_id: str,
    phone_number: str,
    tenant_data: dict,
    score: int,
    status: str,
    reasons: list,
    transcript: str,
    duration: int,
    providers: dict,
    bypass_filters: bool = False,
):
    """
    Celery task: send screening summary email to landlord.

    Honors the admin email settings: qualified-only filter, custom
    subject/body templates, transcript inclusion, and sender identity.
    ``bypass_filters`` skips the qualified-only check (used by manual resend).
    Retries up to 3 times with exponential backoff on failure.
    """
    try:
        if not settings.resend_api_key:
            logger.warning("RESEND_API_KEY not set — skipping email")
            return {"sent": False, "reason": "no_api_key"}

        resend.api_key = settings.resend_api_key

        email_settings = _get_email_settings_sync()

        if (
            not bypass_filters
            and email_settings.get("email_qualified_only")
            and status != "qualified"
        ):
            logger.info(
                "Skipping email for call %s — qualified-only is on and status is %s",
                call_id,
                status,
            )
            return {"sent": False, "reason": "qualified_only_filter"}

        landlord_email = (
            email_settings.get("landlord_email") or settings.default_landlord_email
        )
        if not landlord_email:
            logger.warning("No landlord email configured — skipping email")
            return {"sent": False, "reason": "no_recipient"}

        subject, html_body = render_email_template(
            phone_number=phone_number,
            tenant_data=tenant_data,
            score=score,
            status=status,
            reasons=reasons or [],
            duration=duration,
            providers=providers,
            call_id=call_id,
            subject_template=email_settings.get("email_subject_template") or None,
            body_template=email_settings.get("email_body_template") or None,
            include_transcript=email_settings.get("email_include_transcript", False),
            transcript=transcript or "",
        )

        from_name = email_settings.get("email_from_name") or settings.email_from_name
        from_address = email_settings.get("email_from_address") or settings.email_from

        params = {
            "from": f"{from_name} <{from_address}>",
            "to": [landlord_email],
            "subject": subject,
            "html": html_body,
        }

        cc = _split_emails(email_settings.get("cc_emails"))
        bcc = _split_emails(email_settings.get("bcc_emails"))
        if cc:
            params["cc"] = cc
        if bcc:
            params["bcc"] = bcc

        result = resend.Emails.send(params)
        logger.info(f"Email sent for call {call_id}: {result}")

        _mark_email_sent_sync(call_id)
        return {"sent": True, "email_id": result.get("id")}

    except Exception as e:
        logger.error(f"Email send failed for call {call_id}: {e}")
        try:
            raise self.retry(exc=e, countdown=10 * (2**self.request.retries))
        except self.MaxRetriesExceededError:
            logger.error(f"Email permanently failed for call {call_id} after retries")
            return {"sent": False, "reason": "max_retries_exceeded", "error": str(e)}


@celery_app.task(
    bind=True,
    max_retries=3,
    default_retry_delay=15,
    name="app.services.email_service.fire_crm_webhook_task",
)
def fire_crm_webhook_task(
    self,
    webhook_url: str,
    call_id: str,
    phone_number: str,
    status: str,
    score: int,
    tenant_data: dict,
    app_url: str,
):
    """
    Celery task: fire CRM webhook with HMAC-SHA256 signature.
    """
    import json

    import httpx

    from app.utils.helpers import generate_hmac_signature
    from app.utils.security import UnsafeURLError, assert_safe_external_url

    # The CRM URL is admin-configured but still externally influenced; validate it
    # targets a public host so it can't be pointed at internal services (SSRF).
    try:
        assert_safe_external_url(webhook_url)
    except UnsafeURLError as e:
        logger.error("Refusing unsafe CRM webhook URL for call %s: %s", call_id, e)
        return {"sent": False, "error": "unsafe_webhook_url"}

    try:
        payload = {
            "event": "tenant_screened",
            "timestamp": datetime.now(UTC).isoformat(),
            "call_id": call_id,
            "phone_number": phone_number,
            "qualification_status": status,
            "score": score,
            "tenant_data": {
                "adults": tenant_data.get("adults_count"),
                "children": tenant_data.get("children_count"),
                "monthly_income": float(tenant_data["monthly_income"])
                if tenant_data.get("monthly_income")
                else None,
                "has_eviction": tenant_data.get("has_eviction"),
                "move_in_date": str(tenant_data.get("move_in_date"))
                if tenant_data.get("move_in_date")
                else None,
                "move_reason": tenant_data.get("move_reason"),
            },
            "transcript_url": f"{app_url}/admin/calls/{call_id}",
        }
        body = json.dumps(payload).encode()

        secret = _get_webhook_secret_sync()
        headers = {"Content-Type": "application/json"}
        if secret:
            headers["X-Signature-SHA256"] = generate_hmac_signature(body, secret)

        with httpx.Client(timeout=10.0) as client:
            response = client.post(webhook_url, content=body, headers=headers)
            response.raise_for_status()

        logger.info(f"CRM webhook fired for call {call_id}: {response.status_code}")
        return {"sent": True, "status_code": response.status_code}

    except Exception as e:
        logger.error(f"CRM webhook failed for call {call_id}: {e}")
        try:
            raise self.retry(exc=e, countdown=15 * (2**self.request.retries))
        except self.MaxRetriesExceededError:
            logger.error(f"CRM webhook permanently failed for call {call_id}")
            return {"sent": False, "error": str(e)}


@celery_app.task(
    soft_time_limit=30,
    time_limit=45,
    name="app.services.email_service.send_daily_digest_task",
)
def send_daily_digest_task():
    """
    Celery beat task: send daily digest email summarizing previous day's calls.
    Runs every 24 hours via Celery beat schedule.
    """
    try:
        if not settings.resend_api_key:
            return {"sent": False, "reason": "no_api_key"}

        stats = _get_yesterday_stats_sync()
        if stats["total_calls"] == 0:
            logger.info("No calls yesterday — skipping digest")
            return {"sent": False, "reason": "no_calls"}

        # Use the admin-configured email settings (recipient, sender identity,
        # CC/BCC) just like the per-call result email, falling back to env.
        email_settings = _get_email_settings_sync()
        landlord_email = email_settings.get("landlord_email")
        if not landlord_email:
            return {"sent": False, "reason": "no_recipient"}

        from_name = email_settings.get("email_from_name") or settings.email_from_name
        from_address = email_settings.get("email_from_address") or settings.email_from

        resend.api_key = settings.resend_api_key
        html = f"""
        <div style="font-family: sans-serif; max-width: 500px; margin: 0 auto;">
          <h2>📊 Daily Screening Digest</h2>
          <p>Yesterday's summary:</p>
          <ul>
            <li>Total calls: {stats['total_calls']}</li>
            <li>Qualified: {stats['qualified']}</li>
            <li>Needs review: {stats['review']}</li>
            <li>Disqualified: {stats['unqualified']}</li>
          </ul>
          <a href="{settings.app_url}/admin/dashboard">View Dashboard</a>
        </div>
        """
        params = {
            "from": f"{from_name} <{from_address}>",
            "to": [landlord_email],
            "subject": f"📊 Daily Screening Digest — {stats['total_calls']} calls",
            "html": html,
        }
        cc = _split_emails(email_settings.get("cc_emails"))
        bcc = _split_emails(email_settings.get("bcc_emails"))
        if cc:
            params["cc"] = cc
        if bcc:
            params["bcc"] = bcc

        resend.Emails.send(params)
        return {"sent": True}
    except Exception as e:
        logger.error(f"Daily digest failed: {e}")
        return {"sent": False, "error": str(e)}


@celery_app.task(name="app.services.email_service.provider_health_check_task")
def provider_health_check_task():
    """
    Celery beat task: ping all configured providers every 5 minutes.
    Logs results for the API Health Monitor panel.
    """
    import asyncio

    logger.info("Running provider health check...")
    try:
        asyncio.run(_run_health_checks())
    except Exception as e:
        logger.error(f"Health check task failed: {e}")
    return {"checked": True}


async def _run_health_checks():
    """Async helper to ping all providers."""
    from config import provider_registry

    results = {}
    try:
        ok, latency = await provider_registry.llm.ping()
        results["llm"] = {"healthy": ok, "latency_ms": latency}
    except Exception as e:
        results["llm"] = {"healthy": False, "error": str(e)}

    logger.info(f"Health check results: {results}")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Sync DB helpers (Celery tasks run in sync context)
# ──────────────────────────────────────────────────────────────────────────────


def _split_emails(value) -> list[str]:
    """Parse a comma/semicolon-separated email string into a clean list."""
    if not value:
        return []
    import re

    parts = re.split(r"[,;\s]+", str(value).strip())
    return [p for p in parts if p]


def _coerce_bool(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return default
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _get_email_settings_sync() -> dict:
    """Load all email-related settings synchronously (Celery worker context)."""
    import asyncio

    from app.db.crud import get_setting_value
    from app.db.database import AsyncSessionLocal

    keys = (
        "landlord_email",
        "email_from_name",
        "email_from_address",
        "email_subject_template",
        "email_body_template",
        "email_qualified_only",
        "email_include_transcript",
        "cc_emails",
        "bcc_emails",
    )

    async def _fetch():
        async with AsyncSessionLocal() as db:
            data = {}
            for key in keys:
                data[key] = await get_setting_value(db, key, "")
            return data

    try:
        raw = asyncio.run(_fetch())
    except RuntimeError:
        raw = {}

    return {
        "landlord_email": raw.get("landlord_email") or settings.default_landlord_email,
        "email_from_name": raw.get("email_from_name") or "",
        "email_from_address": raw.get("email_from_address") or "",
        "email_subject_template": raw.get("email_subject_template") or "",
        "email_body_template": raw.get("email_body_template") or "",
        "email_qualified_only": _coerce_bool(raw.get("email_qualified_only")),
        "email_include_transcript": _coerce_bool(raw.get("email_include_transcript")),
        "cc_emails": raw.get("cc_emails") or "",
        "bcc_emails": raw.get("bcc_emails") or "",
    }


def _get_landlord_email_sync() -> str:
    """Get landlord email using a sync DB connection (Celery worker context)."""
    import asyncio

    from app.db.crud import get_setting_value
    from app.db.database import AsyncSessionLocal

    async def _fetch():
        async with AsyncSessionLocal() as db:
            email = await get_setting_value(
                db, "landlord_email", settings.default_landlord_email
            )
            return email or settings.default_landlord_email

    try:
        return asyncio.run(_fetch())
    except RuntimeError:
        # Event loop already running — fallback to default
        return settings.default_landlord_email


def _get_webhook_secret_sync() -> str:
    """Get CRM webhook secret synchronously."""
    import asyncio

    from app.db.crud import get_setting_value
    from app.db.database import AsyncSessionLocal

    async def _fetch():
        async with AsyncSessionLocal() as db:
            return await get_setting_value(db, "crm_webhook_secret", "")

    try:
        return asyncio.run(_fetch())
    except RuntimeError:
        return ""


def _mark_email_sent_sync(call_id: str) -> None:
    """Mark email as sent on the tenant record."""
    import asyncio
    import uuid as uuid_module

    from app.db.crud import get_tenant_by_call, update_tenant
    from app.db.database import AsyncSessionLocal

    async def _update():
        async with AsyncSessionLocal() as db:
            try:
                call_uuid = uuid_module.UUID(call_id)
                tenant = await get_tenant_by_call(db, call_uuid)
                if tenant:
                    await update_tenant(
                        db,
                        tenant.id,
                        email_sent=True,
                        email_sent_at=datetime.now(UTC),
                    )
            except Exception as e:
                logger.error(f"Failed to mark email sent: {e}")

    try:
        asyncio.run(_update())
    except RuntimeError:
        pass


def _get_yesterday_stats_sync() -> dict:
    """Get yesterday's call stats synchronously for digest email."""
    import asyncio
    from datetime import timedelta

    from sqlalchemy import and_, func, select

    from app.db.database import AsyncSessionLocal
    from app.models.call import Call
    from app.models.tenant import Tenant

    async def _fetch():
        async with AsyncSessionLocal() as db:
            now = datetime.now(UTC)
            yesterday_start = (now - timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            yesterday_end = yesterday_start + timedelta(days=1)

            total = (
                await db.execute(
                    select(func.count(Call.id)).where(
                        and_(
                            Call.created_at >= yesterday_start,
                            Call.created_at < yesterday_end,
                            Call.is_deleted == False,
                        )
                    )
                )
            ).scalar() or 0

            quals = {}
            for status in ["qualified", "review", "unqualified"]:
                count = (
                    await db.execute(
                        select(func.count(Tenant.id))
                        .join(Call, Call.id == Tenant.call_id)
                        .where(
                            and_(
                                Call.created_at >= yesterday_start,
                                Call.created_at < yesterday_end,
                                Tenant.qualification_status == status,
                            )
                        )
                    )
                ).scalar() or 0
                quals[status] = count

            return {"total_calls": total, **quals}

    try:
        return asyncio.run(_fetch())
    except RuntimeError:
        return {"total_calls": 0, "qualified": 0, "review": 0, "unqualified": 0}
