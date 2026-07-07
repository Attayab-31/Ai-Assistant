"""
app/api/settings.py — Provider switching API and settings management routes.

The most critical API for the platform: allows hot-swapping LLM/STT/TTS
providers from the admin panel without restarting the server.
"""

import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import crud
from app.db.database import get_db
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
from app.utils.dependencies import require_scope
from app.utils.security import encrypt_value
from config import provider_registry

logger = logging.getLogger(__name__)
router = APIRouter()

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


async def _reload_registry_or_400(db: AsyncSession, *, label: str) -> None:
    """Reload the provider registry or raise HTTP 400."""
    try:
        await provider_registry.reload_from_db(db)
    except Exception as e:
        logger.error("Failed to switch %s provider: %s", label, e)
        raise HTTPException(
            status_code=400, detail=f"Failed to switch provider: {str(e)}"
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

    await crud.set_setting(
        db, "active_llm_provider", payload.provider, updated_by=user.id
    )
    if payload.model:
        await crud.set_setting(
            db, f"active_{payload.provider}_model", payload.model, updated_by=user.id
        )

    if payload.api_key:
        encrypted = encrypt_value(payload.api_key)
        await crud.set_setting(
            db, f"{payload.provider}_api_key_encrypted", encrypted, updated_by=user.id
        )

    await _reload_registry_or_400(db, label="LLM")

    await crud.create_audit_log(
        db,
        action="switched_llm_provider",
        admin_user_id=user.id,
        entity_type="setting",
        old_value={"provider": old_provider},
        new_value={"provider": payload.provider, "model": payload.model},
        ip_address=request.client.host if request.client else None,
    )

    logger.info(
        f"LLM provider switched: {old_provider} -> {payload.provider} by {user.email}"
    )
    return {
        "success": True,
        "active_provider": provider_registry.llm_name,
        "previous_provider": old_provider,
    }


@router.post("/stt/switch")
async def switch_stt_provider(
    payload: STTProviderSwitch,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: AdminUser = Depends(require_scope("settings", edit=True)),
):
    """Hot-swap the active STT provider (Deepgram/Groq Whisper)."""
    old_provider = provider_registry.stt_name

    await crud.set_setting(
        db, "active_stt_provider", payload.provider, updated_by=user.id
    )
    if payload.model:
        # Store the model under the provider-specific key so a Groq model never
        # overwrites the Deepgram model (and vice versa).
        model_key = "deepgram_model" if payload.provider == "deepgram" else "groq_stt_model"
        await crud.set_setting(db, model_key, payload.model, updated_by=user.id)

    if payload.api_key:
        encrypted = encrypt_value(payload.api_key)
        await crud.set_setting(
            db, f"{payload.provider}_api_key_encrypted", encrypted, updated_by=user.id
        )

    await _reload_registry_or_400(db, label="STT")

    await crud.create_audit_log(
        db,
        action="switched_stt_provider",
        admin_user_id=user.id,
        entity_type="setting",
        old_value={"provider": old_provider},
        new_value={"provider": payload.provider},
        ip_address=request.client.host if request.client else None,
    )

    logger.info(
        f"STT provider switched: {old_provider} -> {payload.provider} by {user.email}"
    )
    return {
        "success": True,
        "active_provider": provider_registry.stt_name,
        "previous_provider": old_provider,
    }


@router.post("/tts/switch")
async def switch_tts_provider(
    payload: TTSProviderSwitch,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: AdminUser = Depends(require_scope("settings", edit=True)),
):
    """Hot-swap the active TTS provider (Google WaveNet/Deepgram Aura-2)."""
    old_provider = provider_registry.tts_name

    await crud.set_setting(
        db, "active_tts_provider", payload.provider, updated_by=user.id
    )
    if payload.voice:
        await crud.set_setting(
            db, f"tts_voice_{payload.provider}", payload.voice, updated_by=user.id
        )
    if payload.speed is not None:
        await crud.set_setting(db, "tts_speed", str(payload.speed), updated_by=user.id)

    await _reload_registry_or_400(db, label="TTS")

    await crud.create_audit_log(
        db,
        action="switched_tts_provider",
        admin_user_id=user.id,
        entity_type="setting",
        old_value={"provider": old_provider},
        new_value={"provider": payload.provider, "voice": payload.voice},
        ip_address=request.client.host if request.client else None,
    )

    logger.info(
        f"TTS provider switched: {old_provider} -> {payload.provider} by {user.email}"
    )
    return {
        "success": True,
        "active_provider": provider_registry.tts_name,
        "previous_provider": old_provider,
    }


@router.post("/api-key")
async def set_provider_api_key(
    payload: ProviderApiKeyUpdate,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: AdminUser = Depends(require_scope("settings", edit=True)),
):
    """Set or rotate the API key for ANY provider — even one that is not the
    active engine (e.g. a backup AI brain).

    The key is encrypted at rest and applied to NEW calls immediately (the live
    provider registry is reloaded). The active provider selection is unchanged.
    """
    key_name = f"{payload.provider}_api_key_encrypted"
    encrypted = encrypt_value(payload.api_key)
    await crud.set_setting(db, key_name, encrypted, updated_by=user.id)

    reload_applied = True
    reload_error = None
    try:
        await provider_registry.reload_from_db(db)
    except Exception as e:
        # Persist the key even if a rebuild fails (e.g. the key is for a backup
        # provider that isn't active); it still takes effect on the next call.
        logger.warning("Registry reload after API key update failed: %s", e)
        reload_applied = False
        reload_error = str(e)

    await crud.create_audit_log(
        db,
        action="rotated_provider_api_key",
        admin_user_id=user.id,
        entity_type="setting",
        new_value={"provider": payload.provider, "key": "updated"},
        ip_address=request.client.host if request.client else None,
    )

    logger.info(f"API key set/rotated for {payload.provider} by {user.email}")
    return {
        "success": True,
        "provider": payload.provider,
        "reload_applied": reload_applied,
        "reload_error": reload_error,
    }


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
        load_call_settings_snapshot,
    )
    from app.providers.stt.deepgram_stt import DeepgramSTTProvider
    from app.providers.stt.groq_stt import GroqSTTProvider
    from config import settings as env_settings

    snapshot = await load_call_settings_snapshot(db)
    bundle = build_call_provider_bundle(snapshot)

    def _has_key(provider: str) -> bool:
        env_map = {
            "groq": "groq_api_key",
            "openai": "openai_api_key",
            "openrouter": "openrouter_api_key",
            "gemini": "gemini_api_key",
            "deepgram": "deepgram_api_key",
        }
        attr = env_map.get(provider)
        if attr and (getattr(env_settings, attr, "") or "").strip():
            return True
        return False

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
        except Exception as e:
            return {"provider": name, "healthy": False, "role": role, "error": str(e)}

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

    from app.core.question_flow import question_save_warnings

    warnings = question_save_warnings(new_questions)
    await crud.set_setting(
        db, "screening_questions", json.dumps(new_questions), updated_by=user.id
    )

    await crud.create_audit_log(
        db,
        action="updated_screening_questions",
        admin_user_id=user.id,
        entity_type="setting",
        old_value={"count": len(old_questions)},
        new_value={"count": len(new_questions)},
        ip_address=request.client.host if request.client else None,
    )

    return {"success": True, "questions": new_questions, "warnings": warnings}


@router.post("/questions/reset")
async def reset_questions_to_defaults(
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: AdminUser = Depends(require_scope("settings", edit=True)),
):
    """Reset screening questions to system defaults."""
    from config import DEFAULT_QUESTIONS

    await crud.set_setting(
        db, "screening_questions", json.dumps(DEFAULT_QUESTIONS), updated_by=user.id
    )

    await crud.create_audit_log(
        db,
        action="reset_screening_questions",
        admin_user_id=user.id,
        entity_type="setting",
        ip_address=request.client.host if request.client else None,
    )
    return {"success": True, "questions": DEFAULT_QUESTIONS}


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

    await crud.set_setting(
        db, "screening_faqs", json.dumps(new_faqs), updated_by=user.id
    )

    await crud.create_audit_log(
        db,
        action="updated_screening_faqs",
        admin_user_id=user.id,
        entity_type="setting",
        old_value={"count": len(old_faqs)},
        new_value={"count": len(new_faqs)},
        ip_address=request.client.host if request.client else None,
    )

    return {"success": True, "faqs": new_faqs}


@router.post("/faqs/reset")
async def reset_faqs_to_defaults(
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: AdminUser = Depends(require_scope("settings", edit=True)),
):
    """Reset screening FAQs to system defaults."""
    from config import DEFAULT_FAQS

    await crud.set_setting(
        db, "screening_faqs", json.dumps(DEFAULT_FAQS), updated_by=user.id
    )

    await crud.create_audit_log(
        db,
        action="reset_screening_faqs",
        admin_user_id=user.id,
        entity_type="setting",
        ip_address=request.client.host if request.client else None,
    )
    return {"success": True, "faqs": DEFAULT_FAQS}


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
) -> dict[str, str | bool]:
    """Shared save path for POST email settings."""
    updates = payload.model_dump(exclude_none=True)
    for key, value in updates.items():
        await crud.set_setting(db, key, str(value), updated_by=user.id)

    await crud.create_audit_log(
        db,
        action="updated_email_settings",
        admin_user_id=user.id,
        entity_type="setting",
        new_value=updates,
        ip_address=request.client.host if request.client else None,
    )
    return updates


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
        return {"sent": True, "email_id": result.get("id"), "subject": subject}
    except Exception as e:
        logger.error("Test email failed: %s", e)
        raise HTTPException(
            status_code=500, detail=f"Failed to send test email: {str(e)}"
        ) from e


@router.post("/email")
async def post_email_settings(
    payload: EmailSettingsUpdate,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: AdminUser = Depends(require_scope("settings", edit=True)),
):
    """POST alias for updating email configuration (used by the admin HTML form)."""
    updates = await _persist_email_settings(db, payload, user, request)
    return {"success": True, "updated": list(updates.keys())}


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
    for key, value in EMAIL_RESET_DEFAULTS.items():
        await crud.set_setting(db, key, str(value), updated_by=user.id)

    await crud.create_audit_log(
        db,
        action="reset_email_settings",
        admin_user_id=user.id,
        entity_type="setting",
        new_value={"keys": sorted(EMAIL_RESET_DEFAULTS.keys())},
        ip_address=request.client.host if request.client else None,
    )
    return {"success": True, "reset": sorted(EMAIL_RESET_DEFAULTS.keys())}


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

    cache_failed = False
    for key, value in updates.items():
        _, cache_ok = await crud.set_setting(db, key, str(value), updated_by=user.id)
        if not cache_ok:
            cache_failed = True

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

    await crud.create_audit_log(
        db,
        action="updated_general_settings",
        admin_user_id=user.id,
        entity_type="setting",
        new_value=redact_for_audit(updates),
        ip_address=request.client.host if request.client else None,
    )
    response: dict = {"success": True, "updated": list(updates.keys())}
    if cache_failed:
        response["warnings"] = [_CACHE_STALE_WARNING]
    return response


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
        "retention_enabled",
        "retention_calls_days",
        "retention_recording_days",
        "retention_audit_days",
        "retention_soft_deleted_days",
        "retention_stale_call_hours",
        "crm_webhook_url",
        "crm_webhook_secret",
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

    cache_failed = False
    for key, value in defaults.items():
        _, cache_ok = await crud.set_setting(db, key, str(value), updated_by=user.id)
        if not cache_ok:
            cache_failed = True

    tz = defaults.get("timezone")
    if tz:
        from app.utils.helpers import set_display_timezone

        set_display_timezone(str(tz))

    try:
        await provider_registry.reload_from_db(db)
    except Exception as e:
        logger.warning("Registry reload after general reset failed: %s", e)

    await crud.create_audit_log(
        db,
        action="reset_general_settings",
        admin_user_id=user.id,
        entity_type="setting",
        new_value={"keys": sorted(defaults.keys())},
        ip_address=request.client.host if request.client else None,
    )

    response: dict = {"success": True, "reset": sorted(defaults.keys())}
    if cache_failed:
        response["warnings"] = [
            "Defaults restored, but the live settings cache could not be refreshed. "
            "New calls may use stale values for up to 30 seconds."
        ]
    return response


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

    blacklist = await crud.add_to_blacklist(db, phone, updated_by=user.id)

    await crud.create_audit_log(
        db,
        action="added_to_blacklist",
        admin_user_id=user.id,
        entity_type="setting",
        new_value={"phone_number": phone},
        ip_address=request.client.host if request.client else None,
    )
    return {"success": True, "blacklist": blacklist}


@router.delete("/blacklist/{phone_number}")
async def remove_from_blacklist(
    phone_number: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: AdminUser = Depends(require_scope("settings", edit=True)),
):
    """Remove a phone number from the blacklist."""
    blacklist = await crud.remove_from_blacklist(db, phone_number, updated_by=user.id)

    await crud.create_audit_log(
        db,
        action="removed_from_blacklist",
        admin_user_id=user.id,
        entity_type="setting",
        old_value={"phone_number": phone_number},
        ip_address=request.client.host if request.client else None,
    )
    return {"success": True, "blacklist": blacklist}


@router.get("/blacklist")
async def get_blacklist(
    db: AsyncSession = Depends(get_db),
    user: AdminUser = Depends(require_scope("settings")),
):
    """Get current blacklisted numbers."""
    blacklist = await crud.get_setting_value(db, "blacklisted_numbers", [])
    return {"blacklist": blacklist}
