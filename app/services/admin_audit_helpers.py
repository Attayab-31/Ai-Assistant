"""Admin audit summaries and tenant custom-field validation."""

from __future__ import annotations

import json
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from fastapi import Request

from app.core.question_flow import normalize_questions

_CUSTOM_FIELD_MAX_LEN = 2000
_CUSTOM_FIELD_MAX_KEYS = 50
_ALLOWED_CUSTOM_TYPES = (str, int, float, bool, type(None))


def audit_client_ip(request: Request | None) -> str | None:
    """Resolve the client IP for audit logs behind trusted proxies."""
    if request is None:
        return None
    if not hasattr(request, "headers"):
        client = getattr(request, "client", None)
        return getattr(client, "host", None) if client else None
    from app.core.ratelimit import client_ip

    return client_ip(request)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def summarize_questions_audit_change(
    old_questions: list[dict[str, Any]] | None,
    new_questions: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Compact audit payload for screening-question saves."""
    old_norm = normalize_questions(old_questions or [])
    new_norm = normalize_questions(new_questions or [])
    old_by_state = {str(q.get("state") or ""): q for q in old_norm}
    new_by_state = {str(q.get("state") or ""): q for q in new_norm}
    old_states = set(old_by_state)
    new_states = set(new_by_state)

    added = sorted(new_states - old_states)
    removed = sorted(old_states - new_states)
    changed: list[str] = []
    for state in sorted(old_states & new_states):
        if old_by_state[state] != new_by_state[state]:
            changed.append(state)

    old_order = [str(q.get("state") or "") for q in old_norm]
    new_order = [str(q.get("state") or "") for q in new_norm]

    return {
        "count_before": len(old_norm),
        "count_after": len(new_norm),
        "added_states": added[:30],
        "removed_states": removed[:30],
        "changed_states": changed[:30],
        "reordered": old_order != new_order,
    }


def build_tenant_audit_old_value(tenant: Any, payload_fields: dict[str, Any]) -> dict[str, Any]:
    """Snapshot tenant fields before an admin edit (no secrets)."""
    old: dict[str, Any] = {}
    custom_keys = [
        k for k in payload_fields if str(k).startswith("custom_")
    ]
    for field, _new in payload_fields.items():
        if field == "normalized_data":
            continue
        if str(field).startswith("custom_"):
            continue
        if hasattr(tenant, field):
            old[field] = _json_safe(getattr(tenant, field))

    if custom_keys:
        cf = dict((tenant.normalized_data or {}).get("custom_fields") or {})
        for key in custom_keys:
            old[key] = _json_safe(cf.get(key))
    return old


def validate_custom_tenant_updates(custom_updates: dict[str, Any]) -> None:
    """Reject oversized or unsupported custom_* applicant field edits."""
    if len(custom_updates) > _CUSTOM_FIELD_MAX_KEYS:
        raise ValueError(
            f"Too many custom fields in one request (max {_CUSTOM_FIELD_MAX_KEYS})"
        )
    for key, value in custom_updates.items():
        if not str(key).startswith("custom_") or len(str(key)) > 80:
            raise ValueError(f"Invalid custom field name: {key!r}")
        if type(value) not in _ALLOWED_CUSTOM_TYPES:
            raise ValueError(
                f"Custom field {key!r} must be text, number, true/false, or empty"
            )
        if isinstance(value, str) and len(value) > _CUSTOM_FIELD_MAX_LEN:
            raise ValueError(
                f"Custom field {key!r} exceeds {_CUSTOM_FIELD_MAX_LEN} characters"
            )


def format_audit_change_summary(
    old_value: dict[str, Any] | None,
    new_value: dict[str, Any] | None,
    *,
    max_len: int = 220,
) -> str:
    """One-line human summary for the activity log table."""
    old_value = old_value or {}
    new_value = new_value or {}

    if "count_before" in new_value or "count_after" in new_value:
        parts = [
            f"{new_value.get('count_before', '?')} → {new_value.get('count_after', '?')} questions"
        ]
        if new_value.get("added_states"):
            parts.append(f"+{len(new_value['added_states'])} added")
        if new_value.get("removed_states"):
            parts.append(f"-{len(new_value['removed_states'])} removed")
        if new_value.get("changed_states"):
            parts.append(f"{len(new_value['changed_states'])} edited")
        if new_value.get("reordered"):
            parts.append("reordered")
        text = "; ".join(parts)
        return text if len(text) <= max_len else text[: max_len - 1] + "…"

    if not old_value and not new_value:
        return "—"

    keys = sorted(set(old_value) | set(new_value))
    if not keys:
        return "—"

    parts: list[str] = []
    for key in keys[:8]:
        if old_value.get(key) == new_value.get(key):
            continue
        ov = old_value.get(key, "—")
        nv = new_value.get(key, "—")
        parts.append(f"{key}: {_short(ov)} → {_short(nv)}")
    if not parts:
        return "Updated"
    extra = len(keys) - 8
    text = "; ".join(parts)
    if extra > 0:
        text += f"; +{extra} more"
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def _short(value: Any, limit: int = 40) -> str:
    if value is None:
        return "empty"
    if isinstance(value, (dict, list)):
        raw = json.dumps(_json_safe(value), default=str)
    else:
        raw = str(value)
    raw = raw.replace("\n", " ").strip()
    if len(raw) > limit:
        return raw[: limit - 1] + "…"
    return raw or "empty"
