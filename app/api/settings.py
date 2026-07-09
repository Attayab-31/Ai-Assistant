"""
app/api/settings.py — Provider switching API and settings management routes.

The most critical API for the platform: allows hot-swapping LLM/STT/TTS
providers from the admin panel without restarting the server.
"""

import json
import logging
import secrets
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import crud
from app.db.database import get_db
from app.models.settings import SystemSetting
from app.models.user import AdminUser
from app.schemas.settings import (
    EmailSettingsUpdate,
    FaqsUpdateRequest,
    GeneralSettingsUpdate,
    LLMProviderSwitch,
    ProviderApiKeyUpdate,
    QuestionsUpdateRequest,
    STTProviderSwitch,
    TTSProviderSwitch,
)
from app.services.admin_audit_helpers import audit_client_ip
from app.utils.dependencies import require_scope
from app.utils.security import encrypt_value
from config import provider_registry

logger = logging.getLogger(__name__)


def _encrypt_api_key_for_storage(api_key: str) -> str:
    """Encrypt a plaintext key; accept already-encrypted ciphertext as-is."""
    from app.utils.security import is_encrypted_value

    cleaned = (api_key or "").strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="API key is required")
    if is_encrypted_value(cleaned):
        return cleaned
    return encrypt_value(cleaned)

router = APIRouter()

_PROVIDER_SWITCH_LOCK_KEY = "settings:provider_switch:lock"
_PROVIDER_SWITCH_LOCK_TTL_SECONDS = 60


@dataclass(frozen=True)
class _SettingSnapshotEntry:
    """Pre-switch DB state for one settings key."""

    existed: bool
    value: str
    is_sensitive: bool
    updated_at: datetime | None = None


async def _release_provider_switch_lock(lock_token: str) -> None:
    """Release switch lock only when owned by this caller."""
    from app.core.redis_client import cache_delete, get_redis

    r = get_redis()
    if r is None:
        return
    try:
        released = await r.eval(
            "if redis.call('get', KEYS[1]) == ARGV[1] "
            "then return redis.call('del', KEYS[1]) else return 0 end",
            1,
            _PROVIDER_SWITCH_LOCK_KEY,
            lock_token,
        )
        if not released:
            logger.debug("Provider switch lock already expired/replaced")
    except Exception as exc:
        logger.debug("Provider switch lock release fallback: %s", exc)
        try:
            value = await r.get(_PROVIDER_SWITCH_LOCK_KEY)
        except Exception:
            value = None
        if value == lock_token:
            await cache_delete(_PROVIDER_SWITCH_LOCK_KEY)


@asynccontextmanager
async def _provider_switch_lock():
    """Serialize provider switches so stale rollbacks cannot clobber newer writes."""
    from app.core.redis_client import acquire_once

    lock_token = secrets.token_urlsafe(16)
    acquired = await acquire_once(
        _PROVIDER_SWITCH_LOCK_KEY,
        _PROVIDER_SWITCH_LOCK_TTL_SECONDS,
        fail_closed=True,
    )
    if not acquired:
        raise HTTPException(
            status_code=409,
            detail="Another provider switch is in progress. Try again in a moment.",
        )
    from app.core.redis_client import get_redis

    r = get_redis()
    if r is not None:
        try:
            await r.set(
                _PROVIDER_SWITCH_LOCK_KEY,
                lock_token,
                xx=True,
                ex=_PROVIDER_SWITCH_LOCK_TTL_SECONDS,
            )
        except Exception:
            # Best effort: if this fails we still rely on fail-closed acquisition.
            pass
    try:
        yield
    finally:
        await _release_provider_switch_lock(lock_token)

EMAIL_SETTING_KEYS = (
    "landlord_email",
    "email_from_name",
    "email_from_address",
    "email_subject_template",
    "email_body_template",
    "email_include_transcript",
    "cc_emails",
    "bcc_emails",
    "timezone",
)

_CACHE_STALE_WARNING = (
    "Settings saved, but the live settings cache could not be refreshed. "
    "New calls may use stale values for up to 30 seconds. Check that Redis "
    "is reachable with read-write credentials."
)
_AUDIT_STALE_WARNING = (
    "Settings saved, but audit logging failed. Check server logs."
)


def _add_cache_warning(response: dict, cache_ok: bool) -> dict:
    """Attach a standard stale-cache warning when invalidation failed."""
    if not cache_ok:
        existing = response.get("warnings")
        if isinstance(existing, list):
            if _CACHE_STALE_WARNING not in existing:
                existing.append(_CACHE_STALE_WARNING)
        else:
            response["warnings"] = [_CACHE_STALE_WARNING]
    return response


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
    """Try to persist audit metadata without failing a successful settings write."""
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
        logger.error("Audit log write failed after settings save: %s", exc)
        return False


async def _snapshot_settings(
    db: AsyncSession, keys: list[str]
) -> dict[str, _SettingSnapshotEntry]:
    """Capture exact raw values (and versions) for rollback."""
    result = await db.execute(select(SystemSetting).where(SystemSetting.key.in_(keys)))
    rows = {row.key: row for row in result.scalars()}
    out: dict[str, _SettingSnapshotEntry] = {}
    for key in keys:
        row = rows.get(key)
        if row is None:
            out[key] = _SettingSnapshotEntry(False, "", False)
        else:
            updated_at = row.updated_at
            if updated_at is not None and updated_at.tzinfo is None:
                updated_at = updated_at.replace(tzinfo=UTC)
            out[key] = _SettingSnapshotEntry(
                True,
                str(row.value),
                bool(row.is_sensitive),
                updated_at,
            )
    return out


async def _restore_settings(
    db: AsyncSession,
    snapshot: dict[str, _SettingSnapshotEntry],
    keys: list[str],
    *,
    written_values: dict[str, str] | None = None,
) -> None:
    """Best-effort rollback to pre-change values in one transaction.

    When ``written_values`` is supplied, each key is only reverted if the row
    still holds the value this switch wrote — a newer successful switch is left
    intact (stale-rollback clobber guard).
    """
    from app.services.settings_cache import invalidate_settings_cache

    if not keys:
        return

    result = await db.execute(select(SystemSetting).where(SystemSetting.key.in_(keys)))
    existing = {row.key: row for row in result.scalars()}
    changed = False

    for key in keys:
        entry = snapshot.get(key, _SettingSnapshotEntry(False, "", False))
        row = existing.get(key)
        expected_written = (written_values or {}).get(key)

        if expected_written is not None:
            if row is None:
                continue
            if str(row.value) != expected_written:
                logger.warning(
                    "Skipping rollback for %s: newer value present (ours=%r, current=%r)",
                    key,
                    expected_written,
                    row.value,
                )
                continue

        if entry.existed:
            if row is None:
                db.add(
                    SystemSetting(
                        key=key,
                        value=entry.value,
                        is_sensitive=entry.is_sensitive,
                    )
                )
                changed = True
            else:
                row.value = entry.value
                row.is_sensitive = entry.is_sensitive
                changed = True
            continue
        if row is not None:
            await db.delete(row)
            changed = True

    if not changed:
        return

    await db.commit()
    try:
        await invalidate_settings_cache()
    except Exception:
        logger.warning("Settings cache invalidation failed during provider rollback")


async def _apply_provider_switch_settings(
    db: AsyncSession,
    *,
    updates: dict[str, str],
    label: str,
    updated_by,
) -> bool:
    """Persist provider switch settings and rollback on reload failure."""
    async with _provider_switch_lock():
        keys = list(updates.keys())
        snapshot = await _snapshot_settings(db, keys)
        try:
            cache_ok = await crud.set_settings_bulk(db, updates, updated_by=updated_by)
            await provider_registry.reload_from_db(db)
            return cache_ok
        except Exception as e:
            logger.error(
                "Failed to switch %s provider; restoring previous settings: %s",
                label,
                e,
            )
            await _restore_settings(
                db, snapshot, keys, written_values=updates
            )
            try:
                await provider_registry.reload_from_db(db)
            except Exception as restore_err:
                logger.error(
                    "Registry reload after rollback failed for %s switch: %s",
                    label,
                    restore_err,
                )
            raise HTTPException(
                status_code=400,
                detail="Failed to switch provider. Verify credentials and try again.",
            ) from e


def _validate_general_settings_updates(updates: dict, current: dict) -> None:
    """Cross-field and numeric guards for general settings saves."""

    def _as_int(key: str, fallback: int) -> int:
        try:
            return int(updates.get(key, current.get(key, fallback) or fallback))
        except (TypeError, ValueError):
            return fallback

    if (
        "qualified_score_threshold" in updates
        or "review_score_threshold" in updates
    ):
        qualified = _as_int("qualified_score_threshold", 75)
        review = _as_int("review_score_threshold", 40)
        if not (0 <= review <= qualified <= 100):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Score cutoffs must satisfy 0 ≤ review ≤ qualified ≤ 100 "
                    f"(got review={review}, qualified={qualified})"
                ),
            )

    if updates.get("llm_temperature") is not None:
        try:
            temp = float(updates["llm_temperature"])
        except (TypeError, ValueError):
            temp = -1.0
        if not (0.0 <= temp <= 1.0):
            raise HTTPException(
                status_code=400,
                detail="AI reply creativity must be between 0.0 and 1.0.",
            )

    if updates.get("llm_max_tokens") is not None:
        try:
            max_tok = int(updates["llm_max_tokens"])
        except (TypeError, ValueError):
            max_tok = -1
        if not (0 <= max_tok <= 2000):
            raise HTTPException(
                status_code=400,
                detail="Max reply length must be between 0 and 2000 tokens (0 = default).",
            )

    for ret_key in (
        "retention_calls_days",
        "retention_recording_days",
        "retention_audit_days",
        "retention_soft_deleted_days",
    ):
        if updates.get(ret_key) is not None:
            try:
                days = int(updates[ret_key])
            except (TypeError, ValueError):
                days = -1
            if days < 0:
                raise HTTPException(
                    status_code=400,
                    detail="Retention windows must be 0 or more days (0 = keep forever).",
                )
            from app.services.retention_service import (
                MIN_RETENTION_AUDIT_DAYS,
                MIN_RETENTION_CALLS_DAYS,
            )

            if (
                ret_key == "retention_calls_days"
                and 0 < days < MIN_RETENTION_CALLS_DAYS
            ):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Call retention must be at least {MIN_RETENTION_CALLS_DAYS} "
                        "days when enabled."
                    ),
                )
            if (
                ret_key == "retention_audit_days"
                and 0 < days < MIN_RETENTION_AUDIT_DAYS
            ):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Audit retention must be at least {MIN_RETENTION_AUDIT_DAYS} "
                        "days when enabled."
                    ),
                )

    if updates.get("timezone"):
        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

        tz = str(updates["timezone"]).strip()
        try:
            ZoneInfo(tz)
        except ZoneInfoNotFoundError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown timezone: {tz}",
            ) from exc
        updates["timezone"] = tz

    if updates.get("retention_stale_call_hours") is not None:
        try:
            hours = int(updates["retention_stale_call_hours"])
        except (TypeError, ValueError):
            hours = -1
        if hours < 0:
            raise HTTPException(
                status_code=400,
                detail="Stale-call window must be 0 or more hours (0 = disable).",
            )

    if updates.get("latency_alert_turn_p95_ms") is not None:
        try:
            turn_warn = int(updates["latency_alert_turn_p95_ms"])
        except (TypeError, ValueError):
            turn_warn = -1
        if turn_warn <= 0:
            raise HTTPException(
                status_code=400,
                detail="Latency warning p95 threshold must be greater than 0 ms.",
            )
    else:
        turn_warn = int(current.get("latency_alert_turn_p95_ms") or 1200)

    if updates.get("latency_alert_turn_p95_crit_ms") is not None:
        try:
            turn_crit = int(updates["latency_alert_turn_p95_crit_ms"])
        except (TypeError, ValueError):
            turn_crit = -1
        if turn_crit <= 0:
            raise HTTPException(
                status_code=400,
                detail="Latency critical p95 threshold must be greater than 0 ms.",
            )
        if turn_crit < turn_warn:
            raise HTTPException(
                status_code=400,
                detail="Latency critical p95 threshold must be >= warning threshold.",
            )

    if updates.get("latency_alert_timeout_rate_pct") is not None:
        try:
            timeout_warn = float(updates["latency_alert_timeout_rate_pct"])
        except (TypeError, ValueError):
            timeout_warn = -1.0
        if timeout_warn <= 0:
            raise HTTPException(
                status_code=400,
                detail="Latency warning timeout-rate threshold must be > 0%.",
            )
    else:
        timeout_warn = float(current.get("latency_alert_timeout_rate_pct") or 2.0)

    if updates.get("latency_alert_timeout_rate_crit_pct") is not None:
        try:
            timeout_crit = float(updates["latency_alert_timeout_rate_crit_pct"])
        except (TypeError, ValueError):
            timeout_crit = -1.0
        if timeout_crit <= 0:
            raise HTTPException(
                status_code=400,
                detail="Latency critical timeout-rate threshold must be > 0%.",
            )
        if timeout_crit < timeout_warn:
            raise HTTPException(
                status_code=400,
                detail="Latency critical timeout-rate threshold must be >= warning threshold.",
            )

    crm_url = updates.get("crm_webhook_url")
    if crm_url:
        from app.utils.security import UnsafeURLError, assert_safe_external_url

        try:
            assert_safe_external_url(str(crm_url), require_https=True)
        except UnsafeURLError as e:
            raise HTTPException(
                status_code=400, detail=f"Unsafe CRM webhook URL: {e}"
            ) from e


# ──────────────────────────────────────────────────────────────────────────────
# Provider Hot-Swap Endpoints
# ──────────────────────────────────────────────────────────────────────────────


@router.post("/llm/switch")
async def switch_llm_provider(
    payload: LLMProviderSwitch,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: AdminUser = Depends(require_scope("settings", edit=True)),
):
    """
    Hot-swap the active LLM provider (Groq/OpenAI/OpenRouter).
    No server restart required. Logged to audit trail.
    """
    old_provider = provider_registry.llm_name

    updates: dict[str, str] = {"active_llm_provider": payload.provider}
    if payload.model:
        updates[f"active_{payload.provider}_model"] = payload.model

    if payload.api_key:
        encrypted = _encrypt_api_key_for_storage(payload.api_key)
        updates[f"{payload.provider}_api_key_encrypted"] = encrypted

    cache_ok = await _apply_provider_switch_settings(
        db,
        updates=updates,
        label="LLM",
        updated_by=user.id,
    )

    audit_ok = await _safe_create_audit_log(
        db,
        action="switched_llm_provider",
        admin_user_id=user.id,
        entity_type="setting",
        old_value={"provider": old_provider},
        new_value={"provider": payload.provider, "model": payload.model},
        ip_address=audit_client_ip(request),
    )

    logger.info(
        f"LLM provider switched: {old_provider} -> {payload.provider} by {user.email}"
    )
    response = _add_cache_warning({
        "success": True,
        "active_provider": provider_registry.llm_name,
        "previous_provider": old_provider,
    }, cache_ok)
    return _add_audit_warning(response, audit_ok)


@router.post("/stt/switch")
async def switch_stt_provider(
    payload: STTProviderSwitch,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: AdminUser = Depends(require_scope("settings", edit=True)),
):
    """Hot-swap the active STT provider (Deepgram/Groq Whisper)."""
    old_provider = provider_registry.stt_name

    updates: dict[str, str] = {"active_stt_provider": payload.provider}
    if payload.model:
        # Store the model under the provider-specific key so a Groq model never
        # overwrites the Deepgram model (and vice versa).
        model_key = "deepgram_model" if payload.provider == "deepgram" else "groq_stt_model"
        updates[model_key] = payload.model

    if payload.api_key:
        encrypted = _encrypt_api_key_for_storage(payload.api_key)
        updates[f"{payload.provider}_api_key_encrypted"] = encrypted

    cache_ok = await _apply_provider_switch_settings(
        db,
        updates=updates,
        label="STT",
        updated_by=user.id,
    )

    audit_ok = await _safe_create_audit_log(
        db,
        action="switched_stt_provider",
        admin_user_id=user.id,
        entity_type="setting",
        old_value={"provider": old_provider},
        new_value={"provider": payload.provider},
        ip_address=audit_client_ip(request),
    )

    logger.info(
        f"STT provider switched: {old_provider} -> {payload.provider} by {user.email}"
    )
    response = _add_cache_warning({
        "success": True,
        "active_provider": provider_registry.stt_name,
        "previous_provider": old_provider,
    }, cache_ok)
    return _add_audit_warning(response, audit_ok)


@router.post("/tts/switch")
async def switch_tts_provider(
    payload: TTSProviderSwitch,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: AdminUser = Depends(require_scope("settings", edit=True)),
):
    """Hot-swap the active TTS provider (Google WaveNet/Deepgram Aura-2)."""
    old_provider = provider_registry.tts_name

    updates: dict[str, str] = {"active_tts_provider": payload.provider}
    if payload.voice:
        updates[f"tts_voice_{payload.provider}"] = payload.voice
    if payload.spanish_voice:
        key = (
            "tts_voice_deepgram_es"
            if payload.provider == "deepgram"
            else "tts_voice_google_es"
        )
        updates[key] = payload.spanish_voice
    if payload.speed is not None:
        updates["tts_speed"] = str(payload.speed)

    cache_ok = await _apply_provider_switch_settings(
        db,
        updates=updates,
        label="TTS",
        updated_by=user.id,
    )

    audit_ok = await _safe_create_audit_log(
        db,
        action="switched_tts_provider",
        admin_user_id=user.id,
        entity_type="setting",
        old_value={"provider": old_provider},
        new_value={"provider": payload.provider, "voice": payload.voice},
        ip_address=audit_client_ip(request),
    )

    logger.info(
        f"TTS provider switched: {old_provider} -> {payload.provider} by {user.email}"
    )
    response = _add_cache_warning({
        "success": True,
        "active_provider": provider_registry.tts_name,
        "previous_provider": old_provider,
    }, cache_ok)
    return _add_audit_warning(response, audit_ok)


@router.post("/api-key")
async def set_provider_api_key(
    payload: ProviderApiKeyUpdate,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: AdminUser = Depends(require_scope("settings", edit=True)),
):
    """Set or rotate the API key for ANY provider — even one that is not the
    active engine (e.g. a backup AI brain).

    The key is encrypted at rest and applied to NEW calls immediately (frozen
    into each call's provider bundle at session start). The live provider
    registry is also reloaded for admin health checks. The active provider
    selection is unchanged.
    """
    key_name = f"{payload.provider}_api_key_encrypted"
    encrypted = _encrypt_api_key_for_storage(payload.api_key)
    _, cache_ok = await crud.set_setting(db, key_name, encrypted, updated_by=user.id)

    reload_applied = True
    reload_error = None
    try:
        await provider_registry.reload_from_db(db)
    except Exception as e:
        # Persist the key even if a rebuild fails (e.g. the key is for a backup
        # provider that isn't active); it still takes effect on the next call.
        logger.warning("Registry reload after API key update failed: %s", e)
        reload_applied = False
        reload_error = "Registry reload failed"

    audit_ok = await _safe_create_audit_log(
        db,
        action="rotated_provider_api_key",
        admin_user_id=user.id,
        entity_type="setting",
        new_value={"provider": payload.provider, "key": "updated"},
        ip_address=audit_client_ip(request),
    )

    logger.info(f"API key set/rotated for {payload.provider} by {user.email}")
    response = _add_cache_warning({
        "success": True,
        "provider": payload.provider,
        "reload_applied": reload_applied,
        "reload_error": reload_error,
    }, cache_ok)
    return _add_audit_warning(response, audit_ok)


# ──────────────────────────────────────────────────────────────────────────────
# Provider Health
# ──────────────────────────────────────────────────────────────────────────────


@router.get("/health")
async def check_provider_health(
    db: AsyncSession = Depends(get_db),
    user: AdminUser = Depends(require_scope("settings")),
):
    """Ping active providers and configured backups; return health/latency info."""
    from app.core.call_settings import (
        build_call_provider_bundle,
        capture_provider_api_keys,
        load_call_settings_snapshot,
    )
    from app.providers.stt.deepgram_stt import DeepgramSTTProvider
    from app.providers.stt.groq_stt import GroqSTTProvider

    snapshot = await load_call_settings_snapshot(db)
    bundle = build_call_provider_bundle(snapshot)
    configured_keys = await capture_provider_api_keys(db)

    def _has_key(provider: str) -> bool:
        return configured_keys.configured(provider)

    async def _ping(instance, name: str, *, role: str = "primary") -> dict:
        if instance is None:
            return {"provider": name, "healthy": False, "role": role, "error": "Not configured"}
        try:
            healthy, latency = await instance.ping()
            return {
                "provider": name,
                "healthy": healthy,
                "latency_ms": latency,
                "role": role,
            }
        except Exception:
            # Avoid leaking upstream vendor exception text to admin clients.
            return {
                "provider": name,
                "healthy": False,
                "role": role,
                "error": "Provider health check failed",
            }

    results: dict = {}

    llm_primary = await _ping(bundle.llm, bundle.llm_name, role="primary")
    llm_backups = []
    for name, inst in bundle.llm_by_name.items():
        if name != bundle.llm_name:
            llm_backups.append(await _ping(inst, name, role="backup"))
    results["llm"] = llm_primary
    results["llm_backups"] = llm_backups

    stt_primary = await _ping(bundle.stt, bundle.stt_name, role="primary")
    stt_backups = []
    for name, factory_model in (
        ("deepgram", snapshot.stt_model),
        ("groq", snapshot.groq_stt_model),
    ):
        if name == bundle.stt_name:
            continue
        if name == "deepgram" and not _has_key("deepgram"):
            continue
        if name == "groq" and not _has_key("groq"):
            continue
        inst = (
            DeepgramSTTProvider(model=factory_model)
            if name == "deepgram"
            else GroqSTTProvider(model=factory_model)
        )
        stt_backups.append(await _ping(inst, name, role="backup"))
    results["stt"] = stt_primary
    results["stt_backups"] = stt_backups

    tts_primary = await _ping(bundle.tts, bundle.tts_name, role="primary")
    tts_backups = []
    for name, inst in bundle.tts_by_name.items():
        if name != bundle.tts_name:
            tts_backups.append(await _ping(inst, name, role="backup"))
    results["tts"] = tts_primary
    results["tts_backups"] = tts_backups

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Screening Questions Management
# ──────────────────────────────────────────────────────────────────────────────


@router.post("/questions/warnings")
async def preview_question_warnings(
    payload: QuestionsUpdateRequest,
    user: AdminUser = Depends(require_scope("settings", edit=True)),
):
    """Return the same warnings/runtime checks as save, without writing settings."""
    from app.core.question_flow import analyze_questions_draft

    return analyze_questions_draft([q.model_dump() for q in payload.questions])


@router.put("/questions")
async def update_questions(
    payload: QuestionsUpdateRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: AdminUser = Depends(require_scope("settings", edit=True)),
):
    """Update screening questions (add, delete, reorder, scoring)."""
    from app.core.question_flow import validate_questions_for_save

    old_questions = await crud.get_setting_value(db, "screening_questions", [])
    try:
        new_questions = validate_questions_for_save(
            [q.model_dump() for q in payload.questions]
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    from app.core.question_flow import analyze_questions_draft

    analysis = analyze_questions_draft(new_questions, validated=new_questions)
    _, cache_ok = await crud.set_setting(
        db, "screening_questions", json.dumps(new_questions), updated_by=user.id
    )

    from app.services.admin_audit_helpers import summarize_questions_audit_change

    change_summary = summarize_questions_audit_change(old_questions, new_questions)
    audit_ok = await _safe_create_audit_log(
        db,
        action="updated_screening_questions",
        admin_user_id=user.id,
        entity_type="setting",
        old_value={"count": change_summary["count_before"]},
        new_value=change_summary,
        ip_address=audit_client_ip(request),
    )

    response = _add_cache_warning({
        "success": True,
        "questions": new_questions,
        "warnings": analysis["warnings"],
        "runtime_valid": analysis["runtime_valid"],
        "runtime_errors": analysis["runtime_errors"],
    }, cache_ok)
    return _add_audit_warning(response, audit_ok)


@router.post("/questions/reset")
async def reset_questions_to_defaults(
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: AdminUser = Depends(require_scope("settings", edit=True)),
):
    """Reset screening questions to system defaults."""
    from config import DEFAULT_QUESTIONS

    _, cache_ok = await crud.set_setting(
        db, "screening_questions", json.dumps(DEFAULT_QUESTIONS), updated_by=user.id
    )

    audit_ok = await _safe_create_audit_log(
        db,
        action="reset_screening_questions",
        admin_user_id=user.id,
        entity_type="setting",
        ip_address=audit_client_ip(request),
    )
    response = _add_cache_warning(
        {"success": True, "questions": DEFAULT_QUESTIONS},
        cache_ok,
    )
    return _add_audit_warning(response, audit_ok)


@router.get("/questions/preview")
async def preview_conversation_flow(
    db: AsyncSession = Depends(get_db),
    user: AdminUser = Depends(require_scope("settings")),
    path: str | None = None,
    language: str = "en",
):
    """Simulate the conversation flow through active questions for preview."""
    from app.core.question_flow import (
        build_conversation_preview_flow,
        build_preview_sample_paths,
        canonical_language_code,
        normalize_questions,
        ordered_active_questions,
    )

    questions = normalize_questions(
        await crud.get_setting_value(db, "screening_questions", [])
    )
    property_name = await crud.get_setting_value(
        db, "property_name", "Ready Rentals Online"
    )
    greeting_message = await crud.get_setting_value(db, "greeting_message", "")
    closing_message = await crud.get_setting_value(db, "closing_message", "")
    business = (property_name or "").strip() or "Ready Rentals Online"

    paths = build_preview_sample_paths(questions)
    selected = paths[0]
    if path:
        selected = next((p for p in paths if p["id"] == path), paths[0])
    sample_data = dict(selected["data"])

    language_code = canonical_language_code(language) or "en"

    flow = build_conversation_preview_flow(
        questions,
        sample_data,
        business=business,
        greeting_message=str(greeting_message or ""),
        closing_message=str(closing_message or ""),
        language_code=language_code,
    )
    active_list = ordered_active_questions(questions, sample_data)
    return {
        "flow": flow,
        "flow_state_count": len(questions),
        "active_question_count_sample": len(active_list),
        "paths": [
            {
                "id": p["id"],
                "label": p["label"],
                "active_question_count": len(
                    ordered_active_questions(questions, p["data"])
                ),
            }
            for p in paths
        ],
        "selected_path": selected["id"],
        "selected_language": language_code,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Screening FAQs Management
# ──────────────────────────────────────────────────────────────────────────────


@router.put("/faqs")
async def update_faqs(
    payload: FaqsUpdateRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: AdminUser = Depends(require_scope("settings", edit=True)),
):
    """Update FAQ wording and match patterns while preserving canonical topics."""
    from app.core.screening_flow import validate_faqs_for_save

    old_faqs = await crud.get_setting_value(db, "screening_faqs", [])
    try:
        new_faqs = validate_faqs_for_save(
            [entry.model_dump() for entry in payload.faqs]
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    _, cache_ok = await crud.set_setting(
        db, "screening_faqs", json.dumps(new_faqs), updated_by=user.id
    )

    audit_ok = await _safe_create_audit_log(
        db,
        action="updated_screening_faqs",
        admin_user_id=user.id,
        entity_type="setting",
        old_value={"count": len(old_faqs)},
        new_value={"count": len(new_faqs)},
        ip_address=audit_client_ip(request),
    )

    response = _add_cache_warning({"success": True, "faqs": new_faqs}, cache_ok)
    return _add_audit_warning(response, audit_ok)


@router.post("/faqs/reset")
async def reset_faqs_to_defaults(
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: AdminUser = Depends(require_scope("settings", edit=True)),
):
    """Reset screening FAQs to system defaults."""
    from config import DEFAULT_FAQS

    _, cache_ok = await crud.set_setting(
        db, "screening_faqs", json.dumps(DEFAULT_FAQS), updated_by=user.id
    )

    audit_ok = await _safe_create_audit_log(
        db,
        action="reset_screening_faqs",
        admin_user_id=user.id,
        entity_type="setting",
        ip_address=audit_client_ip(request),
    )
    response = _add_cache_warning({"success": True, "faqs": DEFAULT_FAQS}, cache_ok)
    return _add_audit_warning(response, audit_ok)


@router.post("/faqs/test")
async def test_faq_phrase(
    payload: dict,
    db: AsyncSession = Depends(get_db),
    user: AdminUser = Depends(require_scope("settings")),
):
    """Server-side FAQ match preview (same logic as live calls)."""
    from app.core.conversation import _select_faq_block, match_faqs_by_pattern
    from app.core.screening_flow import normalize_faqs, validate_faqs_for_save

    phrase = str(payload.get("phrase") or "").strip()
    if not phrase:
        raise HTTPException(status_code=400, detail="phrase is required")

    raw_faqs = payload.get("faqs")
    if raw_faqs is not None:
        try:
            faqs = validate_faqs_for_save(raw_faqs)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    else:
        faqs = normalize_faqs(await crud.get_setting_value(db, "screening_faqs", []))

    active = [
        entry
        for entry in normalize_faqs(faqs)
        if entry.get("active", True) and entry.get("answer")
    ]

    matched = [
        {
            "topic": entry.get("topic", ""),
            "title": entry.get("title", ""),
            "answer": str(entry.get("answer", "")).strip(),
        }
        for entry in match_faqs_by_pattern(phrase, active)
    ]
    block, is_full = _select_faq_block(active, phrase)
    return {
        "phrase": phrase,
        "matched": matched,
        "is_full_block": is_full,
        "uses_question_heuristic": bool(not matched and is_full),
        "prompt_block_preview": block[:500],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Email Settings
# ──────────────────────────────────────────────────────────────────────────────


async def _persist_email_settings(
    db: AsyncSession,
    payload: EmailSettingsUpdate,
    user: AdminUser,
    request: Request,
) -> tuple[dict[str, str | bool], bool, bool]:
    """Shared save path for POST email settings."""
    updates = payload.model_dump(exclude_none=True)
    cache_ok = True
    if updates:
        cache_ok = await crud.set_settings_bulk(
            db,
            {key: str(value) for key, value in updates.items()},
            updated_by=user.id,
        )

    audit_ok = await _safe_create_audit_log(
        db,
        action="updated_email_settings",
        admin_user_id=user.id,
        entity_type="setting",
        new_value=updates,
        ip_address=audit_client_ip(request),
    )
    return updates, cache_ok, audit_ok


@router.post("/email/test")
async def send_test_email(
    payload: dict,
    db: AsyncSession = Depends(get_db),
    user: AdminUser = Depends(require_scope("settings", edit=True)),
):
    """Send a test email using saved templates, sender identity, and sample data."""
    import resend

    from app.services.email_service import (
        _split_emails,
        build_test_email_preview,
    )
    from config import settings as app_settings

    if not app_settings.resend_api_key:
        raise HTTPException(status_code=400, detail="RESEND_API_KEY not configured")

    email_settings = {
        key: await crud.get_setting_value(db, key, "") for key in EMAIL_SETTING_KEYS
    }

    from_name = email_settings.get("email_from_name") or app_settings.email_from_name
    from_address = email_settings.get("email_from_address") or app_settings.email_from

    test_recipient = (
        payload.get("email")
        or email_settings.get("landlord_email")
        or app_settings.default_landlord_email
    )
    if not test_recipient:
        raise HTTPException(status_code=400, detail="No recipient email provided")

    subject, html_body = build_test_email_preview(email_settings)

    try:
        resend.api_key = app_settings.resend_api_key
        send_payload: dict = {
            "from": f"{from_name} <{from_address}>",
            "to": [test_recipient],
            "subject": f"[TEST] {subject}",
            "html": (
                "<p style=\"font-size:13px;color:#64748b;margin:0 0 12px;\">"
                "This is a preview using your saved email templates and sample data."
                "</p>"
                + html_body
            ),
        }
        cc = _split_emails(email_settings.get("cc_emails"))
        bcc = _split_emails(email_settings.get("bcc_emails"))
        if cc:
            send_payload["cc"] = cc
        if bcc:
            send_payload["bcc"] = bcc
        result = resend.Emails.send(send_payload)
        await _safe_create_audit_log(
            db,
            action="sent_test_email",
            admin_user_id=user.id,
            entity_type="setting",
            new_value={"recipient": test_recipient, "subject": subject},
            ip_address=None,
        )
        return {"sent": True, "email_id": result.get("id"), "subject": subject}
    except Exception as e:
        logger.error("Test email failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail="Failed to send test email. Verify email settings and try again.",
        ) from e


@router.post("/email")
async def post_email_settings(
    payload: EmailSettingsUpdate,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: AdminUser = Depends(require_scope("settings", edit=True)),
):
    """POST alias for updating email configuration (used by the admin HTML form)."""
    updates, cache_ok, audit_ok = await _persist_email_settings(db, payload, user, request)
    response = _add_cache_warning(
        {"success": True, "updated": list(updates.keys())},
        cache_ok,
    )
    return _add_audit_warning(response, audit_ok)


EMAIL_RESET_DEFAULTS = {
    "email_from_name": "AI Tenant Screener",
    "email_from_address": "",
    "email_subject_template": "New Screening Result: {name}",
    "email_body_template": "",
    "email_notifications_enabled": "true",
    "email_qualified_only": "false",
    "email_include_transcript": "false",
    "cc_emails": "",
    "bcc_emails": "",
}


@router.post("/email/reset")
async def reset_email_settings_to_defaults(
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: AdminUser = Depends(require_scope("settings", edit=True)),
):
    """Reset email templates and delivery options (keeps recipient address)."""
    cache_ok = await crud.set_settings_bulk(
        db,
        {key: str(value) for key, value in EMAIL_RESET_DEFAULTS.items()},
        updated_by=user.id,
    )

    audit_ok = await _safe_create_audit_log(
        db,
        action="reset_email_settings",
        admin_user_id=user.id,
        entity_type="setting",
        new_value={"keys": sorted(EMAIL_RESET_DEFAULTS.keys())},
        ip_address=audit_client_ip(request),
    )
    response = _add_cache_warning(
        {"success": True, "reset": sorted(EMAIL_RESET_DEFAULTS.keys())},
        cache_ok,
    )
    return _add_audit_warning(response, audit_ok)


# ──────────────────────────────────────────────────────────────────────────────
# General Settings
# ──────────────────────────────────────────────────────────────────────────────


@router.put("/general")
async def update_general_settings(
    payload: GeneralSettingsUpdate,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: AdminUser = Depends(require_scope("settings", edit=True)),
):
    """Update general application settings (property, scoring, call config)."""
    updates = payload.model_dump(exclude_none=True)
    current = await crud.get_all_settings(db)
    _validate_general_settings_updates(updates, current)
    updates_to_persist = {key: str(value) for key, value in updates.items()}
    crm_secret = updates_to_persist.get("crm_webhook_secret")
    if crm_secret:
        from app.utils.security import is_encrypted_value

        if not is_encrypted_value(crm_secret):
            updates_to_persist["crm_webhook_secret"] = encrypt_value(crm_secret)

    cache_ok = await crud.set_settings_bulk(
        db,
        updates_to_persist,
        updated_by=user.id,
    )
    cache_failed = not cache_ok

    # Keep the in-process display timezone (used by sync template/email helpers)
    # in lock-step with the saved setting.
    if "timezone" in updates and updates["timezone"]:
        from app.utils.helpers import set_display_timezone

        set_display_timezone(str(updates["timezone"]))

    registry_keys = {
        "auto_fallback_enabled",
        "llm_fallback_provider",
        "stt_fallback_provider",
        "tts_fallback_provider",
        "property_name",
        "silence_timeout_seconds",
        "max_call_duration_seconds",
        "max_retries_per_question",
    }
    if registry_keys.intersection(updates.keys()):
        try:
            await provider_registry.reload_from_db(db)
        except Exception as e:
            logger.warning("Registry reload after general settings failed: %s", e)

    from app.utils.security import redact_for_audit

    audit_ok = await _safe_create_audit_log(
        db,
        action="updated_general_settings",
        admin_user_id=user.id,
        entity_type="setting",
        new_value=redact_for_audit(updates),
        ip_address=audit_client_ip(request),
    )
    response = _add_cache_warning(
        {"success": True, "updated": list(updates.keys())},
        not cache_failed,
    )
    return _add_audit_warning(response, audit_ok)


GENERAL_RESET_KEYS = frozenset(
    {
        "property_name",
        "timezone",
        "greeting_message",
        "closing_message",
        "provider_failure_message",
        "qualified_score_threshold",
        "review_score_threshold",
        "max_retries_per_question",
        "silence_timeout_seconds",
        "max_call_duration_seconds",
        "call_recording_enabled",
        "llm_temperature",
        "llm_max_tokens",
        "voice_latency_profile",
        "llm_streaming_enabled",
        "auto_fallback_enabled",
        "llm_fallback_provider",
        "stt_fallback_provider",
        "tts_fallback_provider",
        "retention_enabled",
        "retention_calls_days",
        "retention_recording_days",
        "retention_audit_days",
        "retention_soft_deleted_days",
        "retention_stale_call_hours",
        "crm_webhook_url",
        "crm_webhook_secret",
        "crm_notifications_enabled",
    }
)


@router.post("/general/reset")
async def reset_general_settings_to_defaults(
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: AdminUser = Depends(require_scope("settings", edit=True)),
):
    """Reset general application settings to built-in defaults (not blacklist or providers)."""
    from config import DEFAULT_SYSTEM_SETTINGS

    defaults = {
        item["key"]: item["value"]
        for item in DEFAULT_SYSTEM_SETTINGS
        if item["key"] in GENERAL_RESET_KEYS
    }
    defaults_to_persist = {key: str(value) for key, value in defaults.items()}
    crm_secret = defaults_to_persist.get("crm_webhook_secret")
    if crm_secret:
        from app.utils.security import is_encrypted_value

        if not is_encrypted_value(crm_secret):
            defaults_to_persist["crm_webhook_secret"] = encrypt_value(crm_secret)

    cache_ok = await crud.set_settings_bulk(
        db,
        defaults_to_persist,
        updated_by=user.id,
    )
    cache_failed = not cache_ok

    tz = defaults.get("timezone")
    if tz:
        from app.utils.helpers import set_display_timezone

        set_display_timezone(str(tz))

    try:
        await provider_registry.reload_from_db(db)
    except Exception as e:
        logger.warning("Registry reload after general reset failed: %s", e)

    audit_ok = await _safe_create_audit_log(
        db,
        action="reset_general_settings",
        admin_user_id=user.id,
        entity_type="setting",
        new_value={"keys": sorted(defaults.keys())},
        ip_address=audit_client_ip(request),
    )

    response = _add_cache_warning(
        {"success": True, "reset": sorted(defaults.keys())},
        not cache_failed,
    )
    return _add_audit_warning(response, audit_ok)


@router.post("/blacklist")
async def add_to_blacklist(
    payload: dict,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: AdminUser = Depends(require_scope("settings", edit=True)),
):
    """Add a phone number to the Do Not Call blacklist."""
    from app.utils.helpers import sanitize_phone_number

    phone = sanitize_phone_number(payload.get("phone_number", ""))
    if not phone:
        raise HTTPException(status_code=400, detail="Invalid phone number")

    blacklist, cache_ok = await crud.add_to_blacklist(db, phone, updated_by=user.id)

    audit_ok = await _safe_create_audit_log(
        db,
        action="added_to_blacklist",
        admin_user_id=user.id,
        entity_type="setting",
        new_value={"phone_number": phone},
        ip_address=audit_client_ip(request),
    )
    response = _add_cache_warning(
        {"success": True, "blacklist": blacklist},
        cache_ok,
    )
    return _add_audit_warning(response, audit_ok)


@router.delete("/blacklist/{phone_number}")
async def remove_from_blacklist(
    phone_number: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: AdminUser = Depends(require_scope("settings", edit=True)),
):
    """Remove a phone number from the blacklist."""
    from app.utils.helpers import sanitize_phone_number

    phone = sanitize_phone_number(phone_number)
    if not phone:
        raise HTTPException(status_code=400, detail="Invalid phone number")

    blacklist, cache_ok = await crud.remove_from_blacklist(db, phone, updated_by=user.id)

    audit_ok = await _safe_create_audit_log(
        db,
        action="removed_from_blacklist",
        admin_user_id=user.id,
        entity_type="setting",
        old_value={"phone_number": phone},
        ip_address=audit_client_ip(request),
    )
    response = _add_cache_warning(
        {"success": True, "blacklist": blacklist},
        cache_ok,
    )
    return _add_audit_warning(response, audit_ok)


@router.get("/blacklist")
async def get_blacklist(
    db: AsyncSession = Depends(get_db),
    user: AdminUser = Depends(require_scope("settings")),
):
    """Get current blacklisted numbers."""
    blacklist = await crud.get_setting_value(db, "blacklisted_numbers", [])
    return {"blacklist": blacklist}
