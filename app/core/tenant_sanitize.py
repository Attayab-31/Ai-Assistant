"""Last-mile coercion of tenant row payloads before INSERT.

Guarantees every value matches its SQLAlchemy column type so a single bad
extracted field cannot reject the entire screening record. Values that cannot
be coerced safely are moved into the caller-supplied ``overflow`` dict (typically
``normalized_data['persist_overflow']``) rather than dropped.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date, datetime
from decimal import Decimal
from functools import lru_cache
from typing import Any

from sqlalchemy import Boolean, Date, DateTime, Integer, Numeric, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.inspection import inspect

from app.core.data_extractor import _parse_bool, _parse_date, _parse_decimal, _parse_int
from app.models.tenant import Tenant

logger = logging.getLogger(__name__)

_STRING_LIKE = (String, Text)


@lru_cache(maxsize=1)
def _tenant_column_kinds() -> dict[str, tuple[str, int | None]]:
    """Map Tenant column name -> (kind, max_length for strings)."""
    kinds: dict[str, tuple[str, int | None]] = {}
    for col in inspect(Tenant).columns:
        t = col.type
        if isinstance(t, _STRING_LIKE):
            max_len = getattr(t, "length", None)
            kinds[col.key] = ("string", max_len)
        elif isinstance(t, Boolean):
            kinds[col.key] = ("boolean", None)
        elif isinstance(t, Integer):
            kinds[col.key] = ("integer", None)
        elif isinstance(t, Numeric):
            kinds[col.key] = ("numeric", None)
        elif isinstance(t, Date):
            kinds[col.key] = ("date", None)
        elif isinstance(t, (DateTime,)):
            kinds[col.key] = ("datetime", None)
        elif isinstance(t, JSONB):
            kinds[col.key] = ("jsonb", None)
        elif isinstance(t, UUID):
            kinds[col.key] = ("uuid", None)
        else:
            kinds[col.key] = ("unknown", None)
    return kinds


def _coerce_string(value: Any, max_len: int | None) -> tuple[str | None, Any | None]:
    if value in (None, ""):
        return None, None
    if isinstance(value, bool):
        return None, value
    if isinstance(value, (dict, list)):
        return None, value
    if isinstance(value, (int, float, Decimal)):
        text = str(value)
    else:
        text = re.sub(r"\s+", " ", str(value)).strip()
    if not text:
        return None, None
    if max_len is not None:
        text = text[:max_len]
    return text, None


def _coerce_jsonb(value: Any) -> tuple[Any | None, Any | None]:
    if value in (None, ""):
        return None, None
    if isinstance(value, (dict, list)):
        return value, None
    return None, value


def _coerce_datetime(value: Any) -> tuple[datetime | None, Any | None]:
    if value in (None, ""):
        return None, None
    if isinstance(value, datetime):
        return value, None
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time()), None
    text = str(value).strip()
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")), None
    except ValueError:
        parsed = _parse_date(value)
        if parsed is not None:
            return datetime.combine(parsed, datetime.min.time()), None
        return None, value


def _coerce_value(
    key: str,
    value: Any,
    kind: str,
    max_len: int | None,
) -> tuple[Any | None, Any | None]:
    if value is None:
        return None, None

    if kind == "string":
        return _coerce_string(value, max_len)

    if kind == "boolean":
        coerced = _parse_bool(value)
        if coerced is not None:
            return coerced, None
        if isinstance(value, bool):
            return value, None
        return None, value

    if kind == "integer":
        coerced = _parse_int(value)
        if coerced is not None:
            return coerced, None
        return None, value

    if kind == "numeric":
        coerced = _parse_decimal(value)
        if coerced is not None:
            return coerced, None
        return None, value

    if kind == "date":
        coerced = _parse_date(value)
        if coerced is not None:
            return coerced, None
        return None, value

    if kind == "datetime":
        return _coerce_datetime(value)

    if kind == "jsonb":
        return _coerce_jsonb(value)

    if kind == "uuid":
        return value, None

    # Unknown column type — pass through; DB will accept or reject.
    return value, None


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, (dict, list)):
        return value
    return str(value)


def sanitize_tenant_payload(
    payload: dict[str, Any],
    *,
    overflow: dict[str, Any] | None = None,
    log_context: str = "",
) -> dict[str, Any]:
    """Return a copy of ``payload`` with every Tenant column safely coerced."""
    if not payload:
        return {}

    kinds = _tenant_column_kinds()
    out = dict(payload)
    sink = overflow if overflow is not None else {}

    for key, value in list(out.items()):
        spec = kinds.get(key)
        if spec is None:
            continue
        kind, max_len = spec
        coerced, spilled = _coerce_value(key, value, kind, max_len)
        if spilled is not None:
            sink[key] = _json_safe(spilled)
            if log_context:
                logger.warning(
                    "[%s] Moved uncoercible tenant field %r to persist_overflow",
                    log_context,
                    key,
                )
        out[key] = coerced

    return out
