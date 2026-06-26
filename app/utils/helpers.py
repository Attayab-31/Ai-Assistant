"""Formatting, phone normalization, and webhook-signing helpers."""

import hashlib
import hmac
import re
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any


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
    return value.strftime("%Y-%m-%d")


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
