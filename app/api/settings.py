"""
app/api/settings.py — Provider switching API and settings management routes.

The most critical API for the platform: allows hot-swapping LLM/STT/TTS
providers from the admin panel without restarting the server.
"""

import json
import logging
import time

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
    ProviderTestRequest,
    ProviderTestResponse,
    QuestionsUpdateRequest,
    STTProviderSwitch,
    TTSProviderSwitch,
)
from app.utils.dependencies import require_scope
from app.utils.security import encrypt_value
from config import provider_registry

logger = logging.getLogger(__name__)
router = APIRouter()


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

    try:
        await provider_registry.reload_from_db(db)
    except Exception as e:
        logger.error("Failed to switch LLM provider: %s", e)
        raise HTTPException(
            status_code=400, detail=f"Failed to switch provider: {str(e)}"
        ) from e

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

    try:
        await provider_registry.reload_from_db(db)
    except Exception as e:
        logger.error("Failed to switch STT provider: %s", e)
        raise HTTPException(
            status_code=400, detail=f"Failed to switch provider: {str(e)}"
        ) from e

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

    try:
        await provider_registry.reload_from_db(db)
    except Exception as e:
        logger.error("Failed to switch TTS provider: %s", e)
        raise HTTPException(
            status_code=400, detail=f"Failed to switch provider: {str(e)}"
        ) from e

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

    try:
        await provider_registry.reload_from_db(db)
    except Exception as e:
        # Persist the key even if a rebuild fails (e.g. the key is for a backup
        # provider that isn't active); it still takes effect on the next call.
        logger.warning("Registry reload after API key update failed: %s", e)

    await crud.create_audit_log(
        db,
        action="rotated_provider_api_key",
        admin_user_id=user.id,
        entity_type="setting",
        new_value={"provider": payload.provider, "key": "updated"},
        ip_address=request.client.host if request.client else None,
    )

    logger.info(f"API key set/rotated for {payload.provider} by {user.email}")
    return {"success": True, "provider": payload.provider}


# ──────────────────────────────────────────────────────────────────────────────
# Provider Testing
# ──────────────────────────────────────────────────────────────────────────────


@router.post("/test", response_model=ProviderTestResponse)
async def test_provider(
    payload: ProviderTestRequest,
    user: AdminUser = Depends(require_scope("settings")),
):
    """
    Test a provider by sending a sample request and measuring latency.
    Used by the "Test" buttons in the admin providers panel.
    """
    start = time.time()
    try:
        if payload.provider_type == "llm":
            instance = _get_llm_instance(payload.provider)
            response = await instance.get_response(
                system_prompt="You are a helpful test assistant.",
                messages=[{"role": "user", "content": payload.test_text}],
                max_tokens=50,
            )
            latency_ms = round((time.time() - start) * 1000, 1)
            return ProviderTestResponse(
                success=True, latency_ms=latency_ms, response=response
            )

        elif payload.provider_type == "tts":
            instance = _get_tts_instance(payload.provider)
            audio = await instance.synthesize(payload.test_text)
            latency_ms = round((time.time() - start) * 1000, 1)
            return ProviderTestResponse(
                success=len(audio) > 0,
                latency_ms=latency_ms,
                response=f"Generated {len(audio)} bytes of audio",
            )

        elif payload.provider_type == "stt":
            instance = _get_stt_instance(payload.provider)
            ok, latency = await instance.ping()
            return ProviderTestResponse(
                success=ok,
                latency_ms=latency,
                response="STT health check passed" if ok else None,
            )

        else:
            raise HTTPException(status_code=400, detail="Invalid provider_type")

    except Exception as e:
        latency_ms = round((time.time() - start) * 1000, 1)
        logger.error(
            f"Provider test failed ({payload.provider_type}/{payload.provider}): {e}"
        )
        return ProviderTestResponse(success=False, latency_ms=latency_ms, error=str(e))


def _get_llm_instance(provider: str):
    from app.providers.llm.gemini_llm import GeminiLLMProvider
    from app.providers.llm.groq_llm import GroqLLMProvider
    from app.providers.llm.openai_llm import OpenAILLMProvider
    from app.providers.llm.openrouter_llm import OpenRouterLLMProvider

    mapping = {
        "groq": GroqLLMProvider,
        "openai": OpenAILLMProvider,
        "openrouter": OpenRouterLLMProvider,
        "gemini": GeminiLLMProvider,
    }
    cls = mapping.get(provider)
    if not cls:
        raise ValueError(f"Unknown LLM provider: {provider}")
    return cls()


def _get_tts_instance(provider: str):
    from app.providers.tts.deepgram_tts import DeepgramTTSProvider
    from app.providers.tts.google_tts import GoogleTTSProvider

    mapping = {"google": GoogleTTSProvider, "deepgram": DeepgramTTSProvider}
    cls = mapping.get(provider)
    if not cls:
        raise ValueError(f"Unknown TTS provider: {provider}")
    return cls()


def _get_stt_instance(provider: str):
    from app.providers.stt.deepgram_stt import DeepgramSTTProvider
    from app.providers.stt.groq_stt import GroqSTTProvider

    mapping = {"deepgram": DeepgramSTTProvider, "groq": GroqSTTProvider}
    cls = mapping.get(provider)
    if not cls:
        raise ValueError(f"Unknown STT provider: {provider}")
    return cls()


# ──────────────────────────────────────────────────────────────────────────────
# Provider Status / Health
# ──────────────────────────────────────────────────────────────────────────────


@router.get("/status")
async def get_provider_status(user: AdminUser = Depends(require_scope("settings"))):
    """Get current active provider status for the admin dashboard panel."""
    return provider_registry.get_status()


@router.get("/health")
async def check_provider_health(user: AdminUser = Depends(require_scope("settings"))):
    """Ping all active providers and return health/latency info."""
    results = {}

    checks = [
        ("llm", provider_registry._llm, provider_registry.llm_name),
        ("stt", provider_registry._stt, provider_registry.stt_name),
        ("tts", provider_registry._tts, provider_registry.tts_name),
    ]

    for ptype, instance, name in checks:
        if instance:
            try:
                healthy, latency = await instance.ping()
                results[ptype] = {
                    "provider": name,
                    "healthy": healthy,
                    "latency_ms": latency,
                }
            except Exception as e:
                results[ptype] = {"provider": name, "healthy": False, "error": str(e)}
        else:
            results[ptype] = {
                "provider": name,
                "healthy": False,
                "error": "Not initialized",
            }

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Screening Questions Management
# ──────────────────────────────────────────────────────────────────────────────


@router.get("/questions")
async def get_questions(
    db: AsyncSession = Depends(get_db),
    user: AdminUser = Depends(require_scope("settings")),
):
    """Get current screening questions configuration."""
    from app.core.screening_flow import FLOW_STATE_VALUES, normalize_questions

    raw = await crud.get_setting_value(db, "screening_questions", [])
    questions = normalize_questions(raw)
    return {
        "questions": questions,
        "flow_state_count": len(FLOW_STATE_VALUES),
    }


@router.put("/questions")
async def update_questions(
    payload: QuestionsUpdateRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: AdminUser = Depends(require_scope("settings", edit=True)),
):
    """Update screening question wording while preserving the canonical flow states."""
    from app.core.screening_flow import validate_questions_for_save

    old_questions = await crud.get_setting_value(db, "screening_questions", [])
    try:
        new_questions = validate_questions_for_save(
            [q.model_dump() for q in payload.questions]
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

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

    return {"success": True, "questions": new_questions}


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
@router.post("/questions/preview")
async def preview_conversation_flow(
    db: AsyncSession = Depends(get_db),
    user: AdminUser = Depends(require_scope("settings")),
):
    """Simulate the conversation flow through active questions for preview."""
    from app.core.screening_flow import (
        FLOW_STATE_VALUES,
        build_greeting_intro,
        is_skip_state,
        normalize_questions,
    )

    questions = normalize_questions(
        await crud.get_setting_value(db, "screening_questions", [])
    )
    property_name = await crud.get_setting_value(
        db, "property_name", "Ready Rentals Online"
    )
    business = (property_name or "").strip() or "Ready Rentals Online"

    # Sample path: no pets, no eviction → 15 active of 17 flow states
    sample_data = {"has_pets": False, "has_eviction": False}

    intro = build_greeting_intro(business)

    first_q = next(
        (
            q
            for q in questions
            if q.get("state") == "Q1_FULL_NAME" and q.get("active", True)
        ),
        questions[0] if questions else None,
    )
    flow = [
        {
            "speaker": "AI",
            "text": f"{intro} {first_q['question']}" if first_q else intro,
        }
    ]

    for q in sorted(
        [q for q in questions if q.get("active", True)], key=lambda x: x.get("order", 0)
    ):
        state = q.get("state", "")
        if state == "Q1_FULL_NAME":
            continue
        if is_skip_state(state, sample_data):
            flow.append(
                {
                    "speaker": "AI",
                    "text": f"(skipped — {state} not applicable on this path)",
                }
            )
            continue
        flow.append({"speaker": "AI", "text": q["question"]})
        flow.append({"speaker": "Tenant", "text": "(tenant responds here)"})

    flow.append(
        {
            "speaker": "AI",
            "text": "Thank you so much for your time. We'll be in touch soon!",
        }
    )
    active_count = sum(
        1 for state in FLOW_STATE_VALUES if not is_skip_state(state, sample_data)
    )
    return {
        "flow": flow,
        "flow_state_count": len(FLOW_STATE_VALUES),
        "active_question_count_sample": active_count,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Screening FAQs Management
# ──────────────────────────────────────────────────────────────────────────────


@router.get("/faqs")
async def get_faqs(
    db: AsyncSession = Depends(get_db),
    user: AdminUser = Depends(require_scope("settings")),
):
    """Get current screening FAQ configuration."""
    from app.core.screening_flow import FAQ_TOPIC_VALUES, normalize_faqs

    raw = await crud.get_setting_value(db, "screening_faqs", [])
    faqs = normalize_faqs(raw)
    return {
        "faqs": faqs,
        "faq_topic_count": len(FAQ_TOPIC_VALUES),
    }


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


# ──────────────────────────────────────────────────────────────────────────────
# Email Settings
# ──────────────────────────────────────────────────────────────────────────────


@router.get("/email")
async def get_email_settings(
    db: AsyncSession = Depends(get_db),
    user: AdminUser = Depends(require_scope("settings")),
):
    """Get current email configuration."""
    keys = [
        "landlord_email",
        "email_from_name",
        "email_from_address",
        "email_subject_template",
        "email_body_template",
        "cc_emails",
        "bcc_emails",
        "email_notifications_enabled",
        "email_qualified_only",
        "email_include_transcript",
    ]
    result = {}
    for key in keys:
        result[key] = await crud.get_setting_value(db, key, "")
    return result


async def _persist_email_settings(
    db: AsyncSession,
    payload: EmailSettingsUpdate,
    user: AdminUser,
    request: Request,
) -> dict[str, str | bool]:
    """Shared save path for PUT and POST email settings."""
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


@router.put("/email")
async def update_email_settings(
    payload: EmailSettingsUpdate,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: AdminUser = Depends(require_scope("settings", edit=True)),
):
    """Update email notification settings."""
    updates = await _persist_email_settings(db, payload, user, request)
    return {"success": True, "updated": list(updates.keys())}


@router.post("/email/test")
async def send_test_email(
    payload: dict,
    db: AsyncSession = Depends(get_db),
    user: AdminUser = Depends(require_scope("settings")),
):
    """Send a test email using the SAVED sender identity (not just env)."""
    import resend

    from config import settings as app_settings

    if not app_settings.resend_api_key:
        raise HTTPException(status_code=400, detail="RESEND_API_KEY not configured")

    # Use the same saved sender identity that real screening emails use, so the
    # test actually verifies the admin's configured from-name/address.
    saved_from_name = await crud.get_setting_value(db, "email_from_name", "")
    saved_from_address = await crud.get_setting_value(db, "email_from_address", "")
    saved_landlord = await crud.get_setting_value(db, "landlord_email", "")

    from_name = saved_from_name or app_settings.email_from_name
    from_address = saved_from_address or app_settings.email_from

    test_recipient = (
        payload.get("email") or saved_landlord or app_settings.default_landlord_email
    )
    if not test_recipient:
        raise HTTPException(status_code=400, detail="No recipient email provided")

    try:
        resend.api_key = app_settings.resend_api_key
        result = resend.Emails.send(
            {
                "from": f"{from_name} <{from_address}>",
                "to": [test_recipient],
                "subject": "Test Email - AI Tenant Screener",
                "html": "<p>This is a test email from your AI Tenant Screening Platform. "
                "If you received this, your email configuration is working correctly!</p>",
            }
        )
        return {"sent": True, "email_id": result.get("id")}
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


# ──────────────────────────────────────────────────────────────────────────────
# General Settings
# ──────────────────────────────────────────────────────────────────────────────


@router.get("/general")
async def get_general_settings(
    db: AsyncSession = Depends(get_db),
    user: AdminUser = Depends(require_scope("settings")),
):
    """Get all general settings."""
    return await crud.get_all_settings(db)


@router.put("/general")
async def update_general_settings(
    payload: GeneralSettingsUpdate,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: AdminUser = Depends(require_scope("settings", edit=True)),
):
    """Update general application settings (property, scoring, call config)."""
    updates = payload.model_dump(exclude_none=True)

    weight_keys = [
        "score_weight_income",
        "score_weight_eviction",
        "score_weight_completion",
        "score_weight_move_date",
        "score_weight_rental_history",
        "score_weight_household_fit",
    ]
    if any(k in updates for k in weight_keys):
        current = await crud.get_all_settings(db)
        weights = {k: int(updates.get(k, current.get(k, 0) or 0)) for k in weight_keys}
        total = sum(weights.values())
        if total != 100:
            raise HTTPException(
                status_code=400,
                detail=f"Score weights must total 100 (currently {total}): {weights}",
            )

    # Validate qualification status cutoffs and income multiplier when present.
    if (
        "qualified_score_threshold" in updates
        or "review_score_threshold" in updates
    ):
        current = await crud.get_all_settings(db)

        def _as_int(key: str, fallback: int) -> int:
            try:
                return int(updates.get(key, current.get(key, fallback) or fallback))
            except (TypeError, ValueError):
                return fallback

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
    if "income_multiplier" in updates and updates["income_multiplier"] is not None:
        try:
            mult = float(updates["income_multiplier"])
        except (TypeError, ValueError):
            mult = 0.0
        if not (0 < mult <= 10):
            raise HTTPException(
                status_code=400,
                detail="Income multiplier must be greater than 0 and at most 10.",
            )

    # Reject an unsafe CRM webhook URL at save time (not just when it fires) so
    # an admin can't store a URL pointed at internal/loopback hosts (SSRF).
    crm_url = updates.get("crm_webhook_url")
    if crm_url:
        from app.utils.security import UnsafeURLError, assert_safe_external_url

        try:
            assert_safe_external_url(str(crm_url))
        except UnsafeURLError as e:
            raise HTTPException(
                status_code=400, detail=f"Unsafe CRM webhook URL: {e}"
            ) from e

    for key, value in updates.items():
        await crud.set_setting(db, key, str(value), updated_by=user.id)

    registry_keys = {
        "auto_fallback_enabled",
        "llm_fallback_provider",
        "stt_fallback_provider",
        "tts_fallback_provider",
        "ai_agent_name",
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

    await crud.create_audit_log(
        db,
        action="updated_general_settings",
        admin_user_id=user.id,
        entity_type="setting",
        new_value=updates,
        ip_address=request.client.host if request.client else None,
    )
    return {"success": True, "updated": list(updates.keys())}


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

    blacklist = await crud.get_setting_value(db, "blacklisted_numbers", [])
    if phone not in blacklist:
        blacklist.append(phone)
        await crud.set_setting(
            db, "blacklisted_numbers", json.dumps(blacklist), updated_by=user.id
        )

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
    blacklist = await crud.get_setting_value(db, "blacklisted_numbers", [])
    if phone_number in blacklist:
        blacklist.remove(phone_number)
        await crud.set_setting(
            db, "blacklisted_numbers", json.dumps(blacklist), updated_by=user.id
        )

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
