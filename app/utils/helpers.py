"""Formatting, phone normalization, and webhook-signing helpers."""

import hashlib
import hmac
import logging
import re
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from urllib.parse import urlencode

logger = logging.getLogger(__name__)

# Admin-configured display timezone, cached in-process so the synchronous Jinja
# filters (which can't await the DB) can localize timestamps. Seeded at startup
# and refreshed whenever the admin saves the "timezone" setting.
_DEFAULT_DISPLAY_TZ = "America/New_York"
_display_tz_name = _DEFAULT_DISPLAY_TZ


def set_display_timezone(name: str | None) -> None:
    """Update the cached display timezone (called on startup and on save)."""
    global _display_tz_name
    candidate = (name or "").strip()
    if candidate:
        _display_tz_name = candidate


def _zone(name: str | None) -> ZoneInfo:
    """Resolve a timezone name, falling back to the default then UTC."""
    for candidate in (name, _display_tz_name, _DEFAULT_DISPLAY_TZ):
        if not candidate:
            continue
        try:
            return ZoneInfo(candidate)
        except (ZoneInfoNotFoundError, ValueError, OSError):
            logger.debug("Unknown timezone %r, trying next fallback", candidate)
    return ZoneInfo("UTC")


def format_in_timezone(
    value: datetime | None,
    tz_name: str | None = None,
    fmt: str = "%B %d, %Y at %I:%M %p %Z",
) -> str:
    """Format a (UTC-assumed) datetime in the given/admin display timezone."""
    if not value:
        return "—"
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(_zone(tz_name)).strftime(fmt)


def localtime(value: datetime | None, fmt: str = "%b %d, %Y %I:%M %p %Z") -> str:
    """Jinja filter: render a timestamp in the admin's display timezone."""
    return format_in_timezone(value, None, fmt)


def sanitize_phone_number(phone_number: str) -> str:
    """Normalize a phone number to a simple E.164-like value."""
    raw = (phone_number or "").strip()
    if raw.startswith("tel:"):
        raw = raw[4:]

    digits = re.sub(r"\D", "", raw)
    if not digits:
        return ""

    if raw.startswith("+"):
        return f"+{digits}"
    if len(digits) == 10:
        return f"+1{digits}"
    if len(digits) == 11 and digits.startswith("1"):
        return f"+{digits}"
    return f"+{digits}"


def format_phone_display(phone_number: str | None) -> str:
    """Format a stored phone number for admin display."""
    if not phone_number:
        return "-"

    digits = re.sub(r"\D", "", phone_number)
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    if len(digits) == 10:
        return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
    return phone_number


_TRANSCRIPT_LINE_RE = re.compile(
    r"^\[([^\]]+)\]\s+(AI|Tenant):\s*(.*)$",
    re.IGNORECASE,
)


def parse_transcript_lines(text: str | None) -> list[dict[str, str]]:
    """Split a stored call transcript into speaker-labelled rows."""
    if not text:
        return []
    rows: list[dict[str, str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        match = _TRANSCRIPT_LINE_RE.match(line)
        if match:
            speaker = match.group(2).upper()
            if speaker == "AI":
                speaker_label = "AI"
            else:
                speaker_label = match.group(2).title()
            rows.append(
                {
                    "time": match.group(1),
                    "speaker": speaker_label,
                    "text": match.group(3),
                }
            )
        else:
            rows.append({"time": "", "speaker": "", "text": line})
    return rows


def format_duration(seconds: int | None) -> str:
    """Format a duration in seconds as a compact label."""
    if seconds is None:
        return "-"
    try:
        total = max(0, int(seconds))
    except (TypeError, ValueError):
        return "-"

    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def format_currency(value: Any) -> str:
    """Format a numeric value as US dollars."""
    if value in (None, ""):
        return "-"
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return "-"
    return f"${amount:,.0f}" if amount == amount.to_integral() else f"${amount:,.2f}"


def time_ago(value: datetime | None) -> str:
    """Return a human-friendly relative timestamp."""
    if not value:
        return "-"
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)

    seconds = int((datetime.now(UTC) - value).total_seconds())
    if seconds < 60:
        return "just now"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h ago"
    days = hours // 24
    if days < 30:
        return f"{days}d ago"
    return value.astimezone(_zone(None)).strftime("%Y-%m-%d")


_FRIENDLY_STATE_LABELS = {
    "IDLE": "Starting up",
    "GREETING": "Greeting caller",
    "Q1_FULL_NAME": "Asking for name",
    "Q2_PHONE": "Asking for phone",
    "Q3_EMAIL": "Asking for email",
    "Q4_MOVE_IN_DATE": "Move-in date",
    "Q5_OCCUPANTS": "Who's moving in",
    "Q6_PETS": "Pets",
    "Q6A_PET_DETAILS": "Pet details",
    "Q7_CURRENT_RESIDENCE": "Current home",
    "Q8_RESIDENCE_DURATION": "Time at address",
    "Q9_MOVE_REASON": "Reason for moving",
    "Q10_MOVE_TIMING": "Move timing",
    "Q11_EVICTION": "Eviction history",
    "Q11A_EVICTION_DETAILS": "Eviction details",
    "Q12_INCOME": "Income",
    "Q13_EMPLOYER": "Employer",
    "Q14_EMPLOYMENT_DURATION": "Time employed",
    "Q15_GENERAL_NOTES": "Final notes",
    "WRAP_UP": "Wrapping up",
    "ENDED": "Finishing call",
}


def friendly_state(state: str | None) -> str:
    """Turn an internal call-flow state into a plain-language label for admins."""
    if not state:
        return "In progress"
    label = _FRIENDLY_STATE_LABELS.get(state)
    if label:
        return label
    return state.replace("_", " ").title()


_CALL_STATUS_LABELS = {
    "initiated": "Starting",
    "in_progress": "On call",
    "completed": "Completed",
    "failed": "Failed",
    "abandoned": "Abandoned",
}

_QUALIFICATION_LABELS = {
    "qualified": "Qualified",
    "review": "Needs review",
    "unqualified": "Not qualified",
}


def friendly_call_status(status: str | None) -> str:
    """Plain-language label for a call row status."""
    if not status:
        return "Unknown"
    return _CALL_STATUS_LABELS.get(status.lower(), status.replace("_", " ").title())


def friendly_qualification(status: str | None) -> str:
    """Plain-language label for an applicant qualification status."""
    if not status:
        return "—"
    return _QUALIFICATION_LABELS.get(status.lower(), status.replace("_", " ").title())


_GLOSSARY: dict[str, tuple[str, str]] = {
    "llm": ("AI assistant", "Understands callers and writes replies (also called LLM)"),
    "stt": ("Speech recognition", "Turns caller audio into text (also called STT)"),
    "tts": ("Spoken replies", "Turns written replies into voice (also called TTS)"),
    "tokens": ("AI usage", "How much text the AI processed — billed as tokens by providers"),
    "ttfa": ("First audio", "Time from reply start to first spoken audio (TTFA)"),
    "webhook": ("Send to another system", "Posts screening results to a URL when a call ends"),
    "redis": ("Speed cache", "Optional Redis layer for faster settings — not required"),
    "api_key": ("Provider key", "Secret key from your AI or voice provider account"),
    "prompt": ("AI input", "Text sent to the AI — not spoken to callers"),
    "latency": ("Reply delay", "Wait time between caller speech and assistant response"),
    "uptime": ("Server uptime", "How long this app has been running without restart"),
    "pipeline": ("Voice pipeline", "The speech → AI → voice chain for each call turn"),
}


def glossary_label(key: str | None) -> str:
    """Plain-language label for a technical admin term."""
    if not key:
        return ""
    entry = _GLOSSARY.get(str(key).lower())
    return entry[0] if entry else str(key)


def glossary_tip(key: str | None) -> str:
    """Tooltip explaining a technical admin term."""
    if not key:
        return ""
    entry = _GLOSSARY.get(str(key).lower())
    return entry[1] if entry else ""


def is_property_configured(
    property_name: str = "",
    *,
    greeting_message: str = "",
    closing_message: str = "",
    landlord_email: str = "",
    default_property_name: str = "Ready Rentals Online",
) -> bool:
    """True when the tenant has customized property details beyond seeded defaults."""
    name = (property_name or "").strip()
    default = (default_property_name or "").strip()
    if name and default and name.lower() != default.lower():
        return True
    if (greeting_message or "").strip():
        return True
    if (closing_message or "").strip():
        return True
    if (landlord_email or "").strip():
        return True
    return False


def build_onboarding_checklist(
    *,
    property_name: str = "",
    greeting_message: str = "",
    closing_message: str = "",
    landlord_email: str = "",
    default_property_name: str = "Ready Rentals Online",
    property_settings_saved: bool = False,
    total_calls: int = 0,
    reviewed_applicants: int = 0,
    needs_review_count: int = 0,
    can_settings: bool = False,
    can_edit: bool = False,
    can_tenants: bool = False,
) -> dict[str, Any]:
    """Build home-page onboarding steps for new accounts."""
    steps: list[dict[str, Any]] = []

    if can_settings:
        property_done = is_property_configured(
            property_name,
            greeting_message=greeting_message,
            closing_message=closing_message,
            landlord_email=landlord_email,
            default_property_name=default_property_name,
        ) or property_settings_saved
        steps.append(
            {
                "id": "property",
                "label": "Review your property details",
                "detail": "Confirm your business name, greeting, and notification email in General settings.",
                "done": property_done,
                "href": "/admin/settings/general",
                "cta": "Open settings",
            }
        )

    if can_settings and can_edit:
        steps.append(
            {
                "id": "test_call",
                "label": "Run a test call",
                "detail": "Try the test console to hear your screening flow end-to-end.",
                "done": total_calls > 0,
                "href": "/test",
                "cta": "Start test call",
                "new_tab": True,
            }
        )

    if can_tenants:
        review_href = (
            "/admin/tenants?review=unreviewed"
            if needs_review_count > 0
            else "/admin/tenants"
        )
        steps.append(
            {
                "id": "review",
                "label": "Review your first applicant",
                "detail": (
                    "Open an applicant profile and mark them reviewed when you are done."
                    if reviewed_applicants == 0
                    else "You have reviewed at least one applicant."
                ),
                "done": reviewed_applicants > 0,
                "href": review_href,
                "cta": "Review applicants" if needs_review_count else "Open applicants",
            }
        )

    done_count = sum(1 for step in steps if step["done"])
    complete = bool(steps) and done_count == len(steps)
    return {
        "steps": steps,
        "complete": complete,
        "show": bool(steps) and not complete,
        "done_count": done_count,
        "total_count": len(steps),
    }


_LLM_PROVIDER_LABELS = {
    "groq": "Groq",
    "openai": "OpenAI",
    "openrouter": "OpenRouter",
    "gemini": "Google Gemini",
}
_STT_PROVIDER_LABELS = {
    "deepgram": "Deepgram",
    "groq": "Groq Whisper",
}
_TTS_PROVIDER_LABELS = {
    "google": "Google",
    "deepgram": "Deepgram",
}

_PROVIDER_LABELS_BY_ROLE = {
    "llm": _LLM_PROVIDER_LABELS,
    "stt": _STT_PROVIDER_LABELS,
    "tts": _TTS_PROVIDER_LABELS,
}


def friendly_provider_name(name: str | None, role: str = "llm") -> str:
    """Plain display name for an AI/voice engine slug."""
    if not name:
        return "—"
    slug = str(name).strip().lower()
    labels = _PROVIDER_LABELS_BY_ROLE.get(str(role).lower(), {})
    return labels.get(slug, str(name).replace("_", " ").title())


_AUDIT_ACTION_LABELS: dict[str, str] = {
    "admin_login": "Signed in",
    "switched_llm_provider": "Changed AI assistant engine",
    "switched_stt_provider": "Changed speech recognition engine",
    "switched_tts_provider": "Changed spoken-reply engine",
    "rotated_provider_api_key": "Updated provider API key",
    "updated_screening_questions": "Saved screening questions",
    "reset_screening_questions": "Reset screening questions to defaults",
    "updated_screening_faqs": "Saved caller FAQs",
    "reset_screening_faqs": "Reset caller FAQs to defaults",
    "updated_email_settings": "Saved email settings",
    "reset_email_settings": "Reset email settings to defaults",
    "updated_general_settings": "Saved general settings",
    "reset_general_settings": "Reset general settings to defaults",
    "added_to_blacklist": "Added number to do-not-call list",
    "removed_from_blacklist": "Removed number from do-not-call list",
    "deleted_call": "Deleted a call",
    "resent_email": "Resent screening email",
    "overrode_qualification": "Changed applicant result manually",
    "blacklisted_number": "Blocked a caller number",
    "updated_tenant": "Updated applicant profile",
    "created_admin_user": "Created team account",
    "updated_admin_user": "Updated team account",
    "deleted_admin_user": "Deleted team account",
}

_AUDIT_ENTITY_LABELS: dict[str, str] = {
    "call": "Call",
    "tenant": "Applicant",
    "user": "Team account",
    "settings": "Settings",
    "auth": "Sign-in",
}


def friendly_audit_action(action: str | None) -> str:
    """Plain-language label for an audit-log action code."""
    if not action:
        return "—"
    key = str(action).strip().lower()
    return _AUDIT_ACTION_LABELS.get(key, key.replace("_", " ").capitalize())


def friendly_audit_entity(entity_type: str | None) -> str:
    """Plain-language label for an audit-log entity type."""
    if not entity_type:
        return "—"
    key = str(entity_type).strip().lower()
    return _AUDIT_ENTITY_LABELS.get(key, key.replace("_", " ").title())


def audit_action_choices() -> list[tuple[str, str]]:
    """Sorted (code, label) pairs for activity-log filters."""
    return sorted(_AUDIT_ACTION_LABELS.items(), key=lambda item: item[1].lower())


def pagination_url(path: str, page: int, params: dict[str, Any] | None = None) -> str:
    """Build a list-page URL preserving active filters."""
    q: dict[str, str] = {}
    for key, value in (params or {}).items():
        if value is None:
            continue
        text = str(value).strip()
        if text:
            q[str(key)] = text
    q["page"] = str(max(1, page))
    return f"{path}?{urlencode(q)}"


def list_filter_url(
    path: str,
    filters: dict[str, Any] | None = None,
    **overrides: Any,
) -> str:
    """Build a list-page filter URL (resets to page 1). Pass overrides as kwargs."""
    q: dict[str, str] = {}
    for key, value in (filters or {}).items():
        if key == "page" or value is None:
            continue
        text = str(value).strip()
        if text:
            q[str(key)] = text
    for key, value in overrides.items():
        if value is None or (isinstance(value, str) and not str(value).strip()):
            q.pop(str(key), None)
            continue
        q[str(key)] = str(value)
    q["page"] = "1"
    return f"{path}?{urlencode(q)}"


def date_range_from_days(days: int | None) -> tuple[datetime | None, datetime | None]:
    """Return UTC (date_from, date_to) for a rolling window; all time if days unset."""
    if not days or days <= 0:
        return None, None
    now = datetime.now(UTC)
    return now - timedelta(days=days), None


def tenant_display_name(tenant: Any) -> str:
    """Primary label for an applicant in lists and breadcrumbs."""
    if not tenant:
        return "—"
    name = (getattr(tenant, "full_name", None) or "").strip()
    if name:
        return name
    return format_phone_display(getattr(tenant, "phone_number", None))


def status_badge_color(status: str | None) -> str:
    """Return a stable color for a qualification or call status."""
    colors = {
        "qualified": "#16a34a",
        "review": "#d97706",
        "unqualified": "#dc2626",
        "completed": "#2563eb",
        "in_progress": "#7c3aed",
        "initiated": "#64748b",
        "failed": "#dc2626",
        "abandoned": "#6b7280",
    }
    return colors.get((status or "").lower(), "#64748b")


def score_color(score: int | None) -> str:
    """Return a color based on a 0-100 qualification score."""
    try:
        value = int(score or 0)
    except (TypeError, ValueError):
        value = 0
    if value >= 75:
        return "#16a34a"
    if value >= 50:
        return "#d97706"
    return "#dc2626"


def generate_hmac_signature(body: bytes, secret: str) -> str:
    """Generate an HMAC-SHA256 signature for outbound webhooks."""
    digest = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def generate_stream_token(call_id: str, secret: str) -> str:
    """Create a short-lived signed token authorizing a media-stream WebSocket.

    The token is bound to the call id and a timestamp and signed with the app
    secret. It is appended to the stream URL we hand to Telnyx so that only a
    connection initiated by us (with this exact URL) can drive the call's audio.
    """
    import time

    ts = str(int(time.time()))
    msg = f"{call_id}:{ts}".encode()
    sig = hmac.new(secret.encode(), msg, hashlib.sha256).hexdigest()
    return f"{ts}.{sig}"


def verify_stream_token(
    call_id: str, token: str, secret: str, max_age: int = 300
) -> bool:
    """Validate a stream token created by ``generate_stream_token``.

    Returns False if malformed, expired (older than ``max_age`` seconds), or the
    signature does not match. Uses a constant-time comparison.
    """
    import time

    if not token or "." not in token:
        return False
    ts_str, _, sig = token.partition(".")
    try:
        ts = int(ts_str)
    except ValueError:
        return False
    if abs(int(time.time()) - ts) > max_age:
        return False
    expected = hmac.new(
        secret.encode(), f"{call_id}:{ts_str}".encode(), hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, sig)
