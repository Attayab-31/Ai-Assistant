"""
Structured voice-call logging helpers.

Attach ``call_id``, ``phase``, ``service``, and ``provider`` via ``extra`` so
formatters can render a consistent prefix in development and JSON fields in
production.  Use ``vinfo`` / ``vwarn`` / ``verror`` / ``vdebug`` instead of
plain ``logger.info`` on the hot voice path.
"""

from __future__ import annotations

import logging
import re
from typing import Any

# Grep-friendly phase tags (e.g. ``grep 'llm:fail' logs/voice.trace.log``).
class Phase:
    CALL_START = "call:start"
    CALL_END = "call:end"
    STT = "stt"
    STT_STREAM = "stt:stream"
    STT_FALLBACK = "stt:fallback"
    TURN_START = "turn:start"
    TURN_END = "turn:end"
    TURN_TIMEOUT = "turn:timeout"
    TURN_RECOVERY = "turn:recovery"
    LLM_TRY = "llm:try"
    LLM_OK = "llm:ok"
    LLM_FAIL = "llm:fail"
    LLM_FALLBACK = "llm:fallback"
    LLM_SKIP = "llm:skip"
    LLM_HARDCODED = "llm:hardcoded"
    TTS_TRY = "tts:try"
    TTS_OK = "tts:ok"
    TTS_FAIL = "tts:fail"
    TTS_FALLBACK = "tts:fallback"
    TTS_SKIP = "tts:skip"
    STREAM_TTS = "tts:stream"
    TTS_FINISH = "tts:finish"
    TTS_REMAINDER = "tts:remainder"
    TTS_DEDUP_SKIP = "tts:skip_dup"
    UI_STREAM = "ui:stream"
    UI_FINAL = "ui:final"
    AUDIO_ENQUEUE = "audio:enqueue"
    BARGE_IN = "audio:barge-in"
    ECHO = "audio:echo"
    TENANT = "tenant"


# Keys copied into JSON logs and shown in dev console prefix (after message).
VOICE_CTX_KEYS: tuple[str, ...] = (
    "call_id",
    "phase",
    "service",
    "provider",
    "state",
    "latency_ms",
    "reason",
    "timeout_s",
    "attempt",
    "detail",
    "bytes",
    "duration_s",
    "budget_s",
    "primary",
    "fallback_chain",
)


def _extra(
    session: Any | None = None,
    *,
    call_id: str = "",
    phase: str = "",
    service: str = "",
    provider: str = "",
    state: str = "",
    **fields: Any,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if session is not None:
        cid = getattr(session, "call_id", "") or ""
        if cid:
            out["call_id"] = cid
        st = getattr(session, "current_state", "") or ""
        if st:
            out["state"] = st
    if call_id:
        out["call_id"] = call_id
    if state:
        out["state"] = state
    if phase:
        out["phase"] = phase
    if service:
        out["service"] = service
    if provider:
        out["provider"] = provider
    for key in VOICE_CTX_KEYS:
        if key in fields and fields[key] is not None and fields[key] != "":
            out[key] = fields[key]
    for key, value in fields.items():
        if key in VOICE_CTX_KEYS:
            continue
        if value is not None and value != "":
            out[key] = value
    return out


def voice_log(
    logger: logging.Logger,
    level: int,
    msg: str,
    *,
    session: Any | None = None,
    call_id: str = "",
    **ctx: Any,
) -> None:
    extra = _extra(session, call_id=call_id, **ctx)
    if extra:
        logger.log(level, msg, extra=extra)
    else:
        logger.log(level, msg)


def vdebug(
    logger: logging.Logger,
    msg: str,
    *,
    session: Any | None = None,
    call_id: str = "",
    **ctx: Any,
) -> None:
    voice_log(logger, logging.DEBUG, msg, session=session, call_id=call_id, **ctx)


def vinfo(
    logger: logging.Logger,
    msg: str,
    *,
    session: Any | None = None,
    call_id: str = "",
    **ctx: Any,
) -> None:
    voice_log(logger, logging.INFO, msg, session=session, call_id=call_id, **ctx)


def vwarn(
    logger: logging.Logger,
    msg: str,
    *,
    session: Any | None = None,
    call_id: str = "",
    **ctx: Any,
) -> None:
    voice_log(logger, logging.WARNING, msg, session=session, call_id=call_id, **ctx)


def verror(
    logger: logging.Logger,
    msg: str,
    *,
    session: Any | None = None,
    call_id: str = "",
    **ctx: Any,
) -> None:
    voice_log(logger, logging.ERROR, msg, session=session, call_id=call_id, **ctx)


def format_call_prefix(record: logging.LogRecord) -> str:
    """Human-readable prefix for development console output."""
    parts: list[str] = []
    cid = getattr(record, "call_id", None)
    if cid:
        parts.append(str(cid))
    phase = getattr(record, "phase", None)
    if phase:
        parts.append(str(phase))
    svc = getattr(record, "service", None)
    prov = getattr(record, "provider", None)
    if svc and prov:
        parts.append(f"{svc}/{prov}")
    elif prov:
        parts.append(str(prov))
    elif svc:
        parts.append(str(svc))
    st = getattr(record, "state", None)
    if st and not phase:
        parts.append(str(st))
    if not parts:
        return ""
    return "[" + " | ".join(parts) + "] "


def voice_context_from_record(record: logging.LogRecord) -> dict[str, Any]:
    """Extract voice context fields from a log record for JSON output."""
    ctx: dict[str, Any] = {}
    for key in VOICE_CTX_KEYS:
        if hasattr(record, key):
            value = getattr(record, key)
            if value is not None and value != "":
                ctx[key] = value
    return ctx


class VoiceTraceFilter(logging.Filter):
    """Keep voice-pipeline lines (structured extra or known voice loggers/messages)."""

    _CALL_ID_RE = re.compile(r"\b(test-[a-f0-9]+)\b", re.I)
    _VOICE_LOGGER_PREFIXES = (
        "app.core.call_handler",
        "app.core.audio_stream",
        "app.core.conversation",
        "app.core.streaming_stt",
        "app.core.llm_streaming",
        "app.providers.stt",
        "app.providers.tts",
        "app.providers.llm",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        if getattr(record, "call_id", None) or getattr(record, "phase", None):
            return True
        name = record.name
        if any(name.startswith(prefix) for prefix in self._VOICE_LOGGER_PREFIXES):
            return True
        return bool(self._CALL_ID_RE.search(record.getMessage()))
