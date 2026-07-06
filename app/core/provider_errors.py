"""Classify voice-provider failures into structured, operator-friendly errors."""

from __future__ import annotations

import ast
import re
from typing import Any

PROVIDER_LABELS: dict[str, str] = {
    "groq": "Groq",
    "gemini": "Google Gemini",
    "openai": "OpenAI",
    "openrouter": "OpenRouter",
    "deepgram": "Deepgram",
    "google": "Google TTS",
}

SERVICE_LABELS: dict[str, str] = {
    "llm": "AI assistant",
    "stt": "speech recognition",
    "tts": "spoken voice",
}

REASON_LABELS: dict[str, str] = {
    "invalid_api_key": "Invalid or missing API key",
    "auth_failed": "Authentication failed",
    "rate_limit": "Rate limit exceeded",
    "quota_exceeded": "Quota or usage limit exceeded",
    "billing": "Billing or payment issue",
    "forbidden": "Access forbidden",
    "not_found": "Model or endpoint not found",
    "payload_too_large": "Request too large",
    "timeout": "Request timed out",
    "network_error": "Network or connection error",
    "provider_unavailable": "Provider temporarily unavailable",
    "ssl_error": "Secure connection failed",
    "invalid_response": "Invalid or unparseable response",
    "budget_exhausted": "Turn time budget exhausted",
    "unhealthy": "Skipped after repeated recent failures",
    "no_key": "No API key configured",
    "unknown": "Unknown error",
}


def provider_label(provider: str) -> str:
    key = (provider or "").strip().lower()
    return PROVIDER_LABELS.get(key, key.replace("_", " ").title() or "Unknown")


def service_label(service: str) -> str:
    return SERVICE_LABELS.get((service or "").strip().lower(), service or "provider")


def reason_label(reason: str) -> str:
    return REASON_LABELS.get((reason or "").strip().lower(), reason or "Unknown error")


def _extract_http_status(exc: BaseException) -> int | None:
    for attr in ("status_code", "http_status", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    text = str(exc)
    match = re.search(r"Error code:\s*(\d{3})", text)
    if match:
        return int(match.group(1))
    match = re.search(r"\b(4\d{2}|5\d{2})\b", text)
    if match and "invalid_api_key" not in text.lower():
        return int(match.group(1))
    return None


def _extract_error_body(exc: BaseException) -> dict[str, Any] | None:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        return body
    text = str(exc)
    match = re.search(r"Error code:\s*\d+\s*-\s*(\{.*\})\s*$", text, re.DOTALL)
    if not match:
        return None
    try:
        parsed = ast.literal_eval(match.group(1))
    except (SyntaxError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _nested_error_fields(body: dict[str, Any] | None) -> tuple[str | None, str | None]:
    if not body:
        return None, None
    err = body.get("error")
    if not isinstance(err, dict):
        return None, None
    message = str(err.get("message") or "").strip() or None
    code = str(err.get("code") or err.get("type") or "").strip() or None
    return message, code


def classify_provider_failure(
    exc: BaseException,
    *,
    service: str,
    provider: str,
) -> dict[str, Any]:
    """Return structured fields for a provider failure."""
    http_status = _extract_http_status(exc)
    body = _extract_error_body(exc)
    provider_message, provider_code = _nested_error_fields(body)
    text = " ".join(
        part
        for part in (
            str(provider_code or ""),
            str(provider_message or ""),
            str(exc),
        )
        if part
    ).lower()

    reason = "unknown"
    if isinstance(exc, TimeoutError) or "timed out" in text or "timeout" in text:
        reason = "timeout"
    elif http_status == 401 or "invalid api key" in text or "invalid_api_key" in text:
        reason = "invalid_api_key"
    elif http_status == 403 or "forbidden" in text:
        reason = "forbidden"
    elif http_status == 402 or "billing" in text or "payment" in text:
        reason = "billing"
    elif (
        http_status == 429
        or "rate limit" in text
        or "rate_limit" in text
        or "too many requests" in text
    ):
        reason = "rate_limit"
    elif (
        "quota" in text
        or "usage limit" in text
        or "insufficient_quota" in text
        or "resource_exhausted" in text
    ):
        reason = "quota_exceeded"
    elif http_status == 404 or "not found" in text or "model_not_found" in text:
        reason = "not_found"
    elif http_status == 413 or "too large" in text:
        reason = "payload_too_large"
    elif "ssl" in text or "decryption_failed" in text or "certificate" in text:
        reason = "ssl_error"
    elif http_status in (500, 502, 503, 504) or "unavailable" in text:
        reason = "provider_unavailable"
    elif any(token in text for token in ("connection", "network", "dns", "refused")):
        reason = "network_error"
    elif http_status == 400 and ("auth" in text or "credential" in text):
        reason = "auth_failed"

    return {
        "service": service,
        "provider": (provider or "").lower(),
        "reason": reason,
        "http_status": http_status,
        "provider_code": provider_code,
        "provider_message": provider_message,
    }


def build_provider_message(
    *,
    service: str,
    provider: str,
    role: str,
    outcome: str,
    reason: str | None = None,
    http_status: int | None = None,
    provider_message: str | None = None,
    detail: str | None = None,
) -> str:
    """Human-readable one-line summary for operators."""
    prov = provider_label(provider)
    svc = service_label(service)
    role_key = (role or "primary").lower()
    role_text = f"{role_key} {svc}"

    if outcome == "succeeded":
        return f"{prov} ({role_text}): succeeded"

    if outcome == "skipped":
        why = reason_label(reason or "unknown")
        if detail:
            return f"{prov} ({role_text}): skipped — {why} ({detail})"
        return f"{prov} ({role_text}): skipped — {why}"

    why = reason_label(reason or "unknown")
    parts = [f"{prov} ({role_text}): {why}"]
    if http_status:
        parts.append(f"HTTP {http_status}")
    if provider_message and provider_message.lower() not in why.lower():
        parts.append(f"— {provider_message}")
    elif detail:
        parts.append(f"— {detail}")
    return " ".join(parts)


def summarize_provider_attempts(attempts: list[dict[str, Any]]) -> str:
    """Build a compact summary when every provider in a chain failed."""
    if not attempts:
        return "All providers failed"
    parts: list[str] = []
    for item in attempts:
        prov = provider_label(str(item.get("provider") or ""))
        role = str(item.get("role") or "primary")
        reason = reason_label(str(item.get("reason") or "unknown"))
        parts.append(f"{prov} ({role}) — {reason}")
    return "All providers failed. Tried: " + "; ".join(parts)


def legacy_error_type(service: str, reason: str | None) -> str:
    svc = (service or "").lower()
    if reason == "timeout":
        return f"{svc}_timeout"
    if svc == "llm":
        return "llm_error"
    if svc == "stt":
        return "stt_error"
    if svc == "tts":
        return "tts_error"
    return "provider_error"
