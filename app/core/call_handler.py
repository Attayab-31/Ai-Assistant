"""
app/core/call_handler.py — Main call orchestration and state machine execution.

Orchestrates the full lifecycle of a tenant screening call:
1. Receives audio from Telnyx WebSocket
2. Transcribes via STT provider
3. Gets LLM response
4. Synthesizes speech via TTS provider
5. Streams audio back to Telnyx
6. Saves data on call end
"""

import asyncio
import json
import logging
import random
import re
import time
from collections.abc import Awaitable, Callable, Iterable
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.call_logging import Phase, vdebug, verror, vinfo, vwarn
from app.core.call_settings import (
    CallProviderBundle,
    NotificationSettingsSnapshot,
    build_call_provider_bundle,
    capture_provider_api_keys,
    crm_notifications_active,
    load_call_settings_snapshot,
    load_notification_settings_from_db,
    notification_settings_email_dict,
    notification_settings_persist_dict,
    NOTIFICATION_SETTINGS_PERSIST_KEY,
)
from app.core.conversation import (
    CallState,
    ConversationSession,
    build_correction_readback,
    build_system_prompt,
    compose_agent_response,
    compose_spoken_display,
    confirmation_attempt_limit,
    control_flag_for_intent,
    dedupe_repeated_speech,
    field_maps_for_session,
    filter_extracted_to_allowed_fields,
    is_echo_of_agent,
    is_liveness_acknowledgment,
    is_meta_navigation_request,
    navigation_repeat_text,
    normalize_speech_parts,
    soft_callback_redirect_text,
    parse_corrected_fields,
    parse_issue,
    parse_question_complete,
    parse_relevance,
    parse_turn_intent,
    polite_redirect,
    provider_failure_message_for_session,
    reset_turn_streaming,
    streamed_audio_complete,
    strip_upcoming_question_from_ack,
    try_open_readback_confirmation,
    validate_llm_response,
)
from app.core.data_extractor import extract_tenant_data
from app.core.qualifier import calculate_qualification_score
from app.core.question_flow import (
    canonical_language_code,
    first_active_question_state,
    is_question_answered,
    is_question_required,
    localized_question_text,
    needs_readback_confirmation,
    next_unanswered_state,
    normalize_questions,
    primary_name_field,
    questions_index,
    readback_prompt_for_state,
    repair_prompt_for_state,
    resolve_language_choice,
    screening_complete,
)
from app.core.screening_flow import (
    BUSINESS_NAME,
    _is_refusal_text,
    brief_transition,
    build_greeting_intro,
    log_state_transition,
    normalize_extracted_fields,
    validate_state_transition,
)
from app.core.tenant_sanitize import sanitize_tenant_payload
from app.core.voice_language import (
    deepgram_stt_language,
    deepgram_tts_voice,
    google_tts_language_code,
    google_tts_voice,
    groq_stt_language,
)
from config import settings

# Session/audit-only keys kept in memory for scoring, email, and the CRM webhook
# but never written to the tenant row. The full transcript on the call record is
# the complete audit trail, so the DB stores only finalized structured data.
_AUDIT_ONLY_FIELDS = frozenset(
    {
        "raw_answers",
        "normalized_data",
        "answered_states",
        "refused_states",
        "faq_topics",
        "control_flags",
    }
)

logger = logging.getLogger(__name__)


def _apply_guarded_state_transition(
    session: ConversationSession,
    to_state: str,
    reason: str,
    *,
    retry_count: int = 0,
) -> bool:
    """Apply a state transition only when it passes flow validation."""
    from_state = session.current_state
    if from_state == to_state:
        return True
    is_valid = validate_state_transition(from_state, to_state, session.questions)
    # Always log for observability/counters.
    log_state_transition(
        session.call_id,
        from_state,
        to_state,
        reason,
        retry_count,
        questions=session.questions,
    )
    if not is_valid:
        session.add_error(
            "invalid_state_transition_blocked",
            f"{from_state} -> {to_state} ({reason})",
        )
        return False
    session.current_state = to_state
    return True

# Cached fallback provider instances (avoid per-utterance construction).
_llm_fallbacks: dict = {}
_tts_fallbacks: dict = {}

TTS_MIN_TIMEOUT_SECONDS = 12.0
TTS_MAX_TIMEOUT_SECONDS = 24.0
TTS_TIMEOUT_PER_CHAR_SECONDS = 0.025
TTS_PRIMARY_ATTEMPTS = 2
TTS_FALLBACK_ATTEMPTS = 1
LLM_MIN_ATTEMPT_BUDGET_SECONDS = 1.0
TTS_MIN_ATTEMPT_BUDGET_SECONDS = 1.2
TTS_CONFIRM_MIN_BUDGET_SECONDS = 3.5
VOICE_TURN_INTERNAL_BUFFER_SECONDS = 0.6

# In-memory health hints to prioritize proven-fast LLM backups (per worker).
# Shared across calls on the same worker — affects fallback order/skip only,
# never caller data. Counters decay so one bad spell does not last forever.
_llm_health_hints: dict[str, dict[str, float]] = {}
LLM_HEALTH_WINDOW_S = 300.0
LLM_HEALTH_SKIP_FAIL_THRESHOLD = 3
LLM_HEALTH_PROBE_COOLDOWN_S = 120.0
LLM_HEALTH_REDIS_TTL_S = int(LLM_HEALTH_WINDOW_S + LLM_HEALTH_PROBE_COOLDOWN_S + 120.0)


def _llm_health_cache_key(provider_name: str) -> str:
    return f"llm:health_hint:{provider_name}"


def _llm_health_now() -> float:
    """Wall-clock seconds for health-hint windows.

    Health hints are persisted to Redis and read by other workers, so the
    timestamps must be comparable across processes. ``time.monotonic()`` is
    process-local and would make cross-worker deltas meaningless (providers
    skipped forever or probed too early), so we use wall-clock time here. These
    hints only influence fallback ordering/skip — never caller data — so the
    rare backwards NTP adjustment is harmless.
    """
    return time.time()


def _normalize_llm_health_hint(hint: dict[str, float], now: float) -> None:
    """Drop stale ok/fail counts outside the health window."""
    last_fail = float(hint.get("last_fail_at", 0.0))
    last_ok = float(hint.get("last_ok_at", 0.0))
    if last_fail and now - last_fail > LLM_HEALTH_WINDOW_S:
        hint["fail"] = 0.0
    if last_ok and now - last_ok > LLM_HEALTH_WINDOW_S:
        hint["ok"] = 0.0


def _record_llm_health_success(provider_name: str, latency_ms: float) -> None:
    now = _llm_health_now()
    hint = _llm_health_hints.setdefault(provider_name, {})
    _normalize_llm_health_hint(hint, now)
    hint["ok"] = hint.get("ok", 0.0) + 1.0
    hint["fail"] = 0.0
    hint["last_ok_at"] = now
    hint["latency_ms"] = latency_ms


def _record_llm_health_failure(provider_name: str) -> None:
    now = _llm_health_now()
    hint = _llm_health_hints.setdefault(provider_name, {})
    _normalize_llm_health_hint(hint, now)
    hint["fail"] = hint.get("fail", 0.0) + 1.0
    hint["last_fail_at"] = now


async def _record_llm_health_success_shared(provider_name: str, latency_ms: float) -> None:
    """Record health hints in-process and in Redis for cross-worker consistency."""
    _record_llm_health_success(provider_name, latency_ms)
    from app.core.redis_client import cache_get_json, cache_set_json

    try:
        now = _llm_health_now()
        hint = await cache_get_json(_llm_health_cache_key(provider_name)) or {}
        if not isinstance(hint, dict):
            hint = {}
        _normalize_llm_health_hint(hint, now)
        hint["ok"] = float(hint.get("ok", 0.0)) + 1.0
        hint["fail"] = 0.0
        hint["last_ok_at"] = now
        hint["latency_ms"] = float(latency_ms)
        await cache_set_json(_llm_health_cache_key(provider_name), hint, LLM_HEALTH_REDIS_TTL_S)
    except Exception:
        pass


async def _record_llm_health_failure_shared(provider_name: str) -> None:
    """Record fallback failures in-process and in Redis for all workers."""
    _record_llm_health_failure(provider_name)
    from app.core.redis_client import cache_get_json, cache_set_json

    try:
        now = _llm_health_now()
        hint = await cache_get_json(_llm_health_cache_key(provider_name)) or {}
        if not isinstance(hint, dict):
            hint = {}
        _normalize_llm_health_hint(hint, now)
        hint["fail"] = float(hint.get("fail", 0.0)) + 1.0
        hint["last_fail_at"] = now
        await cache_set_json(_llm_health_cache_key(provider_name), hint, LLM_HEALTH_REDIS_TTL_S)
    except Exception:
        pass


async def _load_llm_health_hint_shared(provider_name: str) -> dict[str, float]:
    """Best-effort shared hint view; falls back to in-process state."""
    hint = dict(_llm_health_hints.get(provider_name, {}))
    from app.core.redis_client import cache_get_json

    try:
        shared = await cache_get_json(_llm_health_cache_key(provider_name))
        if isinstance(shared, dict):
            hint.update(shared)
    except Exception:
        pass
    now = _llm_health_now()
    _normalize_llm_health_hint(hint, now)
    return hint


async def _llm_health_rank_shared(name: str) -> tuple[float, float]:
    hint = await _load_llm_health_hint_shared(name)
    ok = float(hint.get("ok", 0.0))
    fail = float(hint.get("fail", 0.0))
    success_score = ok / max(1.0, ok + fail)
    latency = float(hint.get("latency_ms", 999999.0))
    return (-success_score, latency)


async def _llm_provider_healthy_for_fallback_shared(name: str) -> bool:
    """Skip only providers with repeated recent failures across workers."""
    hint = await _load_llm_health_hint_shared(name)
    now = _llm_health_now()
    ok = float(hint.get("ok", 0.0))
    fail = float(hint.get("fail", 0.0))
    if fail < LLM_HEALTH_SKIP_FAIL_THRESHOLD:
        return True
    if ok > 0:
        return True
    last_fail = float(hint.get("last_fail_at", 0.0))
    if last_fail and now - last_fail >= LLM_HEALTH_PROBE_COOLDOWN_S:
        return True
    return False


def _llm_health_rank(name: str) -> tuple[float, float]:
    now = _llm_health_now()
    hint = _llm_health_hints.setdefault(name, {})
    _normalize_llm_health_hint(hint, now)
    ok = float(hint.get("ok", 0.0))
    fail = float(hint.get("fail", 0.0))
    success_score = ok / max(1.0, ok + fail)
    latency = float(hint.get("latency_ms", 999999.0))
    return (-success_score, latency)


def _llm_provider_healthy_for_fallback(name: str) -> bool:
    """Skip only recent repeated failures; allow probe retry after cooldown."""
    now = _llm_health_now()
    hint = _llm_health_hints.setdefault(name, {})
    _normalize_llm_health_hint(hint, now)
    ok = float(hint.get("ok", 0.0))
    fail = float(hint.get("fail", 0.0))
    if fail < LLM_HEALTH_SKIP_FAIL_THRESHOLD:
        return True
    if ok > 0:
        return True
    last_fail = float(hint.get("last_fail_at", 0.0))
    if last_fail and now - last_fail >= LLM_HEALTH_PROBE_COOLDOWN_S:
        return True
    return False


def _is_spanish(session: ConversationSession) -> bool:
    return str(getattr(session, "call_language", "en")).strip().lower().startswith("es")


def _localize(session: ConversationSession, en: str, es: str) -> str:
    return es if _is_spanish(session) else en


def _plausibility_clarify_fallback(session: ConversationSession) -> str:
    return _localize(
        session,
        "Just to make sure I have that right — could you confirm that detail?",
        "Solo para confirmar — ¿podría repetir ese dato?",
    )


def _begin_turn_tts(session: ConversationSession) -> None:
    session.turn_interrupted = False
    session.turn_tts_tasks.clear()


def _turn_tts_suppressed(session: ConversationSession) -> bool:
    return bool(getattr(session, "turn_interrupted", False))


def _register_turn_tts_task(session: ConversationSession, task: asyncio.Task) -> None:
    session.turn_tts_tasks.append(task)

    def _discard(done: asyncio.Task) -> None:
        try:
            session.turn_tts_tasks.remove(done)
        except ValueError:
            pass

    task.add_done_callback(_discard)


def _track_turn_tts_task(
    session: ConversationSession, coro: Awaitable[bytes]
) -> asyncio.Task:
    task = asyncio.create_task(coro)
    _register_turn_tts_task(session, task)
    return task


def interrupt_turn_tts(session: ConversationSession) -> None:
    """Stop enqueueing TTS for the current turn (sync-safe for barge-in)."""
    session.turn_interrupted = True
    for task in list(session.turn_tts_tasks):
        if not task.done():
            task.cancel()


async def drain_turn_tts_tasks(session: ConversationSession) -> None:
    """Wait for in-flight turn TTS tasks after cancellation."""
    tasks = list(session.turn_tts_tasks)
    if not tasks:
        return
    await asyncio.gather(*tasks, return_exceptions=True)
    session.turn_tts_tasks.clear()


def _apply_tts_voices_for_language(session: ConversationSession, language_code: str) -> None:
    """Switch active and backup TTS voices to match call language."""
    providers = get_call_providers(session)

    from app.providers.tts.deepgram_tts import DeepgramTTSProvider
    from app.providers.tts.google_tts import GoogleTTSProvider

    for name, tts_obj in providers.tts_by_name.items():
        en_voice = str(session.tts_voices_en_by_provider.get(name) or session.tts_voice_en or "")
        if isinstance(tts_obj, DeepgramTTSProvider):
            es_voice = session.tts_voice_deepgram_es or session.tts_voice_es
            tts_obj.voice = DeepgramTTSProvider._normalize_voice(
                deepgram_tts_voice(
                    language_code=language_code,
                    english_voice=en_voice or tts_obj.voice,
                    spanish_voice=es_voice,
                )
            )
        elif isinstance(tts_obj, GoogleTTSProvider):
            es_voice = session.tts_voice_google_es or session.tts_voice_es
            tts_obj.language_code = google_tts_language_code(language_code)
            voice = google_tts_voice(
                language_code=language_code,
                english_voice=en_voice or tts_obj.voice,
                spanish_voice=es_voice,
            )
            from app.providers.tts.google_tts import AVAILABLE_VOICES

            tts_obj.voice = voice if voice in AVAILABLE_VOICES else tts_obj.voice


async def _sync_streaming_stt_language(
    session: ConversationSession, language_code: str
) -> bool:
    """Reconnect live Deepgram STT before TTS switches; keep prior socket on failure."""
    relay = getattr(session, "streaming_stt_relay", None)
    if relay is None:
        return True
    from app.core.streaming_stt import STREAMING_STT_RECONNECT_TIMEOUT_S

    try:
        await asyncio.wait_for(
            relay.reconnect(language=deepgram_stt_language(language_code)),
            timeout=STREAMING_STT_RECONNECT_TIMEOUT_S,
        )
    except Exception as exc:
        logger.warning(
            "[%s] Streaming STT language reconnect failed: %s",
            session.call_id,
            exc,
        )
        # The caller/admin language choice is the source of truth. If the live
        # socket cannot switch languages, mark it unhealthy so audio_stream's
        # normal recovery path degrades to batch STT in the selected language
        # instead of silently continuing to listen in the old one.
        try:
            await relay.close()
        except Exception as close_exc:
            logger.debug(
                "[%s] Streaming STT close after language failure: %s",
                session.call_id,
                close_exc,
            )
        return False
    if relay.lost:
        logger.warning(
            "[%s] Streaming STT language reconnect left relay lost",
            session.call_id,
        )
        return False
    return True


def _apply_batch_stt_language(session: ConversationSession, language_code: str) -> None:
    """Point buffered/batch STT at the caller's language."""
    providers = get_call_providers(session)
    stt_obj = getattr(providers, "stt", None)
    from app.providers.stt.groq_stt import GroqSTTProvider

    if isinstance(stt_obj, GroqSTTProvider):
        stt_obj.language = groq_stt_language(language_code)
    elif stt_obj is not None and hasattr(stt_obj, "language"):
        stt_obj.language = deepgram_stt_language(language_code)


async def _apply_session_language(session: ConversationSession, lang: str | None) -> None:
    resolved = canonical_language_code(lang)
    if not resolved or resolved == session.call_language:
        return
    if not await _sync_streaming_stt_language(session, resolved):
        session.add_error(
            "stt_language_sync_failed",
            f"Live STT reconnect failed for {resolved}; switching to batch STT",
        )
    session.call_language = resolved
    _apply_batch_stt_language(session, resolved)
    _apply_tts_voices_for_language(session, resolved)
    logger.info("[%s] Session language set to %s", session.call_id, resolved)


def _caller_first_name(session: ConversationSession) -> str:
    """Return the caller's first name if we captured a clean one, else ''."""
    name_field = primary_name_field(session.questions) or "full_name"
    name = session.extracted_data.get(name_field)
    if not isinstance(name, str) or not name.strip():
        return ""
    first = name.strip().split()[0].strip(" ,.")
    if 2 <= len(first) <= 20 and first.replace("'", "").replace("-", "").isalpha():
        return first[:1].upper() + first[1:]
    return ""


def human_ack(session: ConversationSession) -> str:
    """A warm acknowledgment that occasionally uses the caller's first name.

    Small, human touches like saying someone's name make the agent feel like a
    real person rather than a form-reader — without changing the deterministic
    question flow.
    """
    base = brief_transition(session.questions_answered)
    if _is_spanish(session):
        base = random.choice(
            (
                "Entendido.",
                "Perfecto.",
                "Gracias.",
                "Muy bien.",
                "Excelente.",
            )
        )
    first = _caller_first_name(session)
    if first and random.random() < 0.45:
        if _is_spanish(session):
            return random.choice(
                (
                    f"Gracias, {first}.",
                    f"Perfecto, {first}.",
                    f"Muy bien, {first}.",
                )
            )
        return random.choice(
            (f"Thanks, {first}.", f"Got it, {first}.", f"Perfect, {first}.")
        )
    return base


def question_advance_ready(
    *,
    question_complete: bool,
    deterministic_done: bool,
    understood: bool,
) -> bool:
    """Whether the current admin question is finished and the flow may advance.

    Honors the LLM ``question_complete`` flag when the caller was understood.
    A deterministic slot fill alone does not advance while the model still
    considers the current question open (e.g. vague date → ask for exact date).
    """
    llm_wants_more = understood and not question_complete
    return (question_complete or deterministic_done) and not llm_wants_more


def tts_timeout_for_text(text: str) -> float:
    """Give TTS enough time for cold starts and longer prompts."""
    text_len = len(text.strip())
    scaled_timeout = 6.0 + (text_len * TTS_TIMEOUT_PER_CHAR_SECONDS)
    return min(
        TTS_MAX_TIMEOUT_SECONDS,
        max(TTS_MIN_TIMEOUT_SECONDS, scaled_timeout),
    )


def _remaining_turn_budget_s(session: ConversationSession) -> float | None:
    deadline = getattr(session, "turn_deadline_monotonic", None)
    if deadline is None:
        return None
    return max(0.0, float(deadline) - time.monotonic())


def _split_for_tts(text: str, max_chars: int = 160) -> list[str]:
    """Split long spoken text at sentence boundaries for reliable TTS."""
    text = (text or "").strip()
    if not text or len(text) <= max_chars:
        return [text] if text else []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if not sentence:
            continue
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(sentence) <= max_chars:
            current = sentence
        else:
            # Hard-wrap an overlong sentence.
            for i in range(0, len(sentence), max_chars):
                chunks.append(sentence[i : i + max_chars])
            current = ""
    if current:
        chunks.append(current)
    return chunks or [text]


def _get_llm_fallback(name: str):
    if name not in _llm_fallbacks:
        from app.providers.llm.gemini_llm import GeminiLLMProvider
        from app.providers.llm.groq_llm import GroqLLMProvider
        from app.providers.llm.openai_llm import OpenAILLMProvider
        from app.providers.llm.openrouter_llm import OpenRouterLLMProvider

        factories = {
            "groq": GroqLLMProvider,
            "openai": OpenAILLMProvider,
            "openrouter": OpenRouterLLMProvider,
            "gemini": GeminiLLMProvider,
        }
        _llm_fallbacks[name] = factories[name]()
    return _llm_fallbacks[name]


def _get_tts_fallback(name: str):
    if name not in _tts_fallbacks:
        from app.providers.tts.deepgram_tts import DeepgramTTSProvider
        from app.providers.tts.google_tts import GoogleTTSProvider

        factories = {
            "deepgram": DeepgramTTSProvider,
            "google": GoogleTTSProvider,
        }
        _tts_fallbacks[name] = factories[name]()
    return _tts_fallbacks[name]


def _resolve_llm_provider(providers: CallProviderBundle, name: str):
    """Return a call-snapshot LLM instance when available."""
    if providers.llm_by_name and name in providers.llm_by_name:
        return providers.llm_by_name[name]
    return _get_llm_fallback(name)


def _resolve_tts_provider(providers: CallProviderBundle, name: str):
    """Return a call-snapshot TTS instance when available."""
    if providers.tts_by_name and name in providers.tts_by_name:
        return providers.tts_by_name[name]
    return _get_tts_fallback(name)


def _prewarm_fallback_clients(providers: CallProviderBundle) -> None:
    """Instantiate likely fallback providers once to reduce first-failover jitter."""
    try:
        if providers.auto_fallback_enabled:
            llm_pref = (providers.llm_fallback_provider or "auto").lower()
            if llm_pref in {"groq", "openai", "openrouter", "gemini"}:
                _resolve_llm_provider(providers, llm_pref)
            elif llm_pref == "auto":
                for name in ("groq", "openai", "openrouter", "gemini"):
                    if name != providers.llm_name:
                        _resolve_llm_provider(providers, name)
            tts_pref = (providers.tts_fallback_provider or "auto").lower()
            if tts_pref in {"deepgram", "google"} and tts_pref != providers.tts_name:
                _resolve_tts_provider(providers, tts_pref)
            elif tts_pref == "auto":
                for name in ("deepgram", "google"):
                    if name != providers.tts_name:
                        _resolve_tts_provider(providers, name)
    except Exception as exc:
        logger.debug("Fallback prewarm skipped: %s", exc)


def get_call_providers(session: ConversationSession) -> CallProviderBundle:
    """Return providers frozen at call start (never the live registry mid-call)."""
    if session.call_providers is not None:
        return session.call_providers
    raise RuntimeError(
        f"Call {session.call_id} has no frozen provider bundle — "
        "use create_session() so STT/LLM/TTS match settings captured at call start"
    )


# In-memory session store (single-worker; use Redis for multi-worker deploys)
_active_sessions: dict[str, ConversationSession] = {}
_sessions_lock = asyncio.Lock()
_session_ready_events: dict[str, asyncio.Event] = {}
# Live Telnyx/test streams register their stop_event here so hangup can wind
# the audio loop down before finalize (avoids truncating the last turn).
_stream_stop_events: dict[str, asyncio.Event] = {}


def register_stream_stop(call_id: str, stop_event: asyncio.Event) -> None:
    """Link a call's audio-stream stop_event for hangup-driven shutdown."""
    _stream_stop_events[call_id] = stop_event


def unregister_stream_stop(call_id: str) -> None:
    _stream_stop_events.pop(call_id, None)


async def _persist_stream_stop_request(call_id: str) -> None:
    """Write a DB hangup stop when the live stream is on another worker."""
    from app.db.crud import persist_stream_stop_request
    from app.db.database import AsyncSessionLocal

    try:
        async with AsyncSessionLocal() as db:
            if await persist_stream_stop_request(db, call_id):
                logger.info(
                    "[%s] Persisted DB stream-stop (cross-worker hangup)",
                    call_id,
                )
    except Exception as exc:
        logger.warning(
            "[%s] DB stream-stop persist failed: %s",
            call_id,
            exc,
        )


async def _is_stream_stop_requested_db(call_id: str) -> bool:
    from app.db.crud import is_stream_stop_requested
    from app.db.database import AsyncSessionLocal

    try:
        async with AsyncSessionLocal() as db:
            return await is_stream_stop_requested(db, call_id)
    except Exception as exc:
        logger.debug("[%s] DB stream-stop check failed: %s", call_id, exc)
        return False


async def request_stream_stop(call_id: str) -> bool:
    """Signal the audio stream to stop (local event + Redis + DB fallback).

    Returns True when a stream on this worker was registered.
    """
    stop = _stream_stop_events.get(call_id)
    local = False
    if stop is not None:
        stop.set()
        local = True
    from app.core.redis_client import ping, set_stream_stop_signal

    if await ping():
        await set_stream_stop_signal(call_id)
    if not local:
        await _persist_stream_stop_request(call_id)
    return local


async def check_stream_stop_signal(call_id: str) -> bool:
    """Poll Redis/DB for a cross-process hangup stop request."""
    from app.core.redis_client import is_stream_stop_signaled

    signaled = await is_stream_stop_signaled(call_id)
    if not signaled:
        # Redis signal is the fast path; DB remains the durable source of truth
        # for cross-worker hangup requests and can still be set even when Redis
        # is reachable but missed/evicted.
        signaled = await _is_stream_stop_requested_db(call_id)
    if signaled:
        stop = _stream_stop_events.get(call_id)
        if stop is not None:
            stop.set()
        session = get_session(call_id)
        if session is not None:
            session.pending_hangup = True
        return True
    return False


async def clear_stream_stop_signals(call_id: str) -> None:
    """Clear Redis and DB hangup stop markers after the stream ends."""
    from app.core.redis_client import clear_stream_stop_signal
    from app.db.crud import clear_stream_stop_request
    from app.db.database import AsyncSessionLocal

    await clear_stream_stop_signal(call_id)
    try:
        async with AsyncSessionLocal() as db:
            await clear_stream_stop_request(db, call_id)
    except Exception as exc:
        logger.debug("[%s] DB stream-stop clear failed: %s", call_id, exc)


async def finalize_active_session_background(call_id: str) -> None:
    """Finalize a live session in a background task (hangup / timeout fallback)."""
    session = get_session(call_id)
    if not session:
        return
    result: dict = {}
    try:
        from app.db.database import AsyncSessionLocal

        async with AsyncSessionLocal() as bg_db:
            try:
                result = await finalize_call(session, bg_db)
                if result.get("status") != "skipped":
                    logger.info(
                        "Call finalized: %s → %s (%s)",
                        call_id,
                        result.get("status"),
                        result.get("score"),
                    )
            except Exception as e:
                logger.error("Error finalizing call %s: %s", call_id, e, exc_info=True)
                from app.db.crud import merge_call_error_log, update_call

                try:
                    await update_call(
                        bg_db,
                        call_id,
                        status="failed",
                        ended_at=datetime.now(UTC),
                    )
                    await merge_call_error_log(
                        bg_db,
                        call_id,
                        {"finalize_error": str(e)},
                        commit=True,
                    )
                except Exception as db_err:
                    logger.error("Failed to mark call failed in DB: %s", db_err)
    finally:
        # Drop the in-memory session only when finalize actually ran. A
        # concurrent finalize returns "skipped" and must keep the session so
        # the winning finalize can complete.
        if get_session(call_id) is session and result.get("status") != "skipped":
            remove_session(call_id)
            unregister_stream_stop(call_id            )


STREAM_FINALIZE_GRACE_S = 28.0


async def finalize_after_stream_timeout(
    call_id: str,
    timeout: float = STREAM_FINALIZE_GRACE_S,
) -> None:
    """Fallback finalize if hangup stopped the stream but on_complete never ran."""
    await asyncio.sleep(timeout)
    session = get_session(call_id)
    if session is not None:
        hangup_requested = bool(getattr(session, "pending_hangup", False)) or (
            await check_stream_stop_signal(call_id)
        )
        if not hangup_requested:
            logger.info(
                "[%s] Stream finalize grace elapsed but no hangup signal — "
                "leaving active session alone",
                call_id,
            )
            return
        logger.warning(
            "[%s] Stream did not finalize within %.0fs after hangup — forcing",
            call_id,
            timeout,
        )
        await finalize_active_session_background(call_id)
        return
    from app.db.crud import get_call_by_call_id, mark_call_abandoned_if_active

    from app.db.database import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        call = await get_call_by_call_id(db, call_id)
        if call and call.status == "in_progress":
            from app.core.redis_client import is_finalize_inflight

            if await is_finalize_inflight(call_id):
                logger.info(
                    "[%s] Skipping abandon — finalize in progress on another worker",
                    call_id,
                )
                return
            stop_requested = await check_stream_stop_signal(call_id)
            if stop_requested:
                marked = await mark_call_abandoned_if_active(
                    db,
                    call_id,
                    error_log={
                        "finalize_timeout": (
                            f"No local session after {timeout:.0f}s hangup grace "
                            "(cross-worker safety net)"
                        )
                    },
                )
                if marked:
                    logger.warning(
                        "[%s] Marked abandoned after hangup timeout on non-owning worker",
                        call_id,
                    )
                return
            logger.warning(
                "[%s] Hangup timeout with no local session — preserving in_progress "
                "(waiting for owning worker finalize / stale-call backstop)",
                call_id,
            )


# ──────────────────────────────────────────────────────────────────────────────
# Session Management
# ──────────────────────────────────────────────────────────────────────────────

STREAM_CALL_ROW_WAIT_S = 3.0


def apply_call_phone_to_session(
    session: ConversationSession,
    call: Any | None,
) -> bool:
    """Copy phone from the DB call row when the in-memory session phone is blank."""
    if (session.phone_number or "").strip():
        return False
    if call is None:
        return False
    db_phone = (getattr(call, "phone_number", None) or "").strip()
    if not db_phone:
        return False
    session.phone_number = db_phone
    return True


async def sync_session_phone_from_db(
    session: ConversationSession,
    db: AsyncSession,
    call: Any | None = None,
) -> bool:
    """Backfill ``session.phone_number`` from the DB when the session is blank."""
    if (session.phone_number or "").strip():
        return False
    if call is None:
        from app.db.crud import get_call_by_call_id

        call = await get_call_by_call_id(db, session.call_id)
    if apply_call_phone_to_session(session, call):
        logger.info(
            "[%s] Backfilled session phone from DB call row",
            session.call_id,
        )
        return True
    return False


async def ensure_stream_session(call_id: str) -> ConversationSession | None:
    """
    Get or create the in-memory session for a Telnyx media WebSocket.

    Polls the DB briefly when the call row is not ready yet (``call.initiated``
    may still be in flight on another worker).
    """
    from app.db.crud import wait_for_call_by_call_id
    from app.db.database import AsyncSessionLocal

    session = get_session(call_id)
    if session is None:
        session = await wait_for_session(call_id, timeout=1.0)

    try:
        async with AsyncSessionLocal() as db:
            if session is None:
                call = await wait_for_call_by_call_id(
                    db,
                    call_id,
                    timeout=STREAM_CALL_ROW_WAIT_S,
                )
                if call is None:
                    logger.warning(
                        "No call row for %s after %.1fs at stream start — "
                        "proceeding with unknown phone",
                        call_id,
                        STREAM_CALL_ROW_WAIT_S,
                    )
                phone_number = (call.phone_number if call else "") or ""
                session = await create_session(
                    call_id=call_id,
                    phone_number=phone_number,
                    db=db,
                )
            elif not (session.phone_number or "").strip():
                call = await wait_for_call_by_call_id(
                    db,
                    call_id,
                    timeout=STREAM_CALL_ROW_WAIT_S,
                )
                await sync_session_phone_from_db(session, db, call=call)
    except Exception as e:
        logger.error(
            "Failed to ensure stream session for %s: %s",
            call_id,
            e,
            exc_info=True,
        )
        return None

    return session


async def create_session(
    call_id: str,
    phone_number: str,
    db: AsyncSession,
    property_name: str | None = None,
    questions: list | None = None,
    max_retries: int | None = None,
) -> ConversationSession:
    """Create and register a new call session (idempotent, race-safe)."""
    existing = _active_sessions.get(call_id)
    if existing:
        new_phone = (phone_number or "").strip()
        if new_phone and not (existing.phone_number or "").strip():
            existing.phone_number = new_phone
        return existing

    snapshot = await load_call_settings_snapshot(db)
    providers = build_call_provider_bundle(
        snapshot,
        api_keys=await capture_provider_api_keys(db),
    )
    _prewarm_fallback_clients(providers)

    async with _sessions_lock:
        existing = _active_sessions.get(call_id)
        if existing:
            logger.info("Session already exists for %s, reusing", call_id)
            new_phone = (phone_number or "").strip()
            if new_phone and not (existing.phone_number or "").strip():
                existing.phone_number = new_phone
            return existing

        session = ConversationSession(
            call_id=call_id,
            phone_number=phone_number,
            property_name=property_name or snapshot.property_name,
            greeting_message=snapshot.greeting_message,
            closing_message=snapshot.closing_message,
            provider_failure_message=snapshot.provider_failure_message,
            llm_temperature=snapshot.llm_temperature,
            llm_max_tokens=snapshot.llm_max_tokens,
            qualified_score_threshold=snapshot.qualified_score_threshold,
            review_score_threshold=snapshot.review_score_threshold,
            questions=questions
            if questions is not None
            else snapshot.questions,
            faqs=snapshot.faqs,
            max_retries=max_retries
            if max_retries is not None
            else snapshot.max_retries,
            stt_provider=providers.stt_name,
            llm_provider=providers.llm_name,
            tts_provider=providers.tts_name,
            primary_stt_provider=providers.stt_name,
            primary_llm_provider=providers.llm_name,
            primary_tts_provider=providers.tts_name,
            call_providers=providers,
            silence_timeout_seconds=snapshot.silence_timeout_seconds,
            max_call_duration_seconds=snapshot.max_call_duration_seconds,
            auto_fallback_enabled=snapshot.auto_fallback_enabled,
            settings_captured_at=snapshot.captured_at,
            voice_latency_profile=snapshot.voice_latency_profile,
            llm_streaming_enabled=snapshot.llm_streaming_enabled,
            turn_timeout_seconds=float(snapshot.turn_timeout_seconds),
            llm_timeout_voice_seconds=float(snapshot.llm_timeout_voice_seconds),
            deepgram_endpointing_ms=int(snapshot.deepgram_endpointing_ms),
            deepgram_utterance_end_ms=int(snapshot.deepgram_utterance_end_ms),
            latency_alert_turn_p95_ms=int(snapshot.latency_alert_turn_p95_ms),
            latency_alert_turn_p95_crit_ms=int(snapshot.latency_alert_turn_p95_crit_ms),
            latency_alert_timeout_rate_pct=float(snapshot.latency_alert_timeout_rate_pct),
            latency_alert_timeout_rate_crit_pct=float(
                snapshot.latency_alert_timeout_rate_crit_pct
            ),
            tts_voice_en=snapshot.tts_voice,
            tts_voice_es=(
                snapshot.tts_voice_deepgram_es
                if snapshot.tts_provider == "deepgram"
                else snapshot.tts_voice_google_es
            ),
            tts_voice_deepgram_es=snapshot.tts_voice_deepgram_es,
            tts_voice_google_es=snapshot.tts_voice_google_es,
            tts_voices_en_by_provider=dict(snapshot.tts_voices_by_provider),
            notification_settings=snapshot.notification_settings,
        )
        if snapshot.questions_runtime_fallback and questions is None:
            session.control_flags["questions_config_blocked"] = (
                snapshot.questions_runtime_fallback
            )
        _active_sessions[call_id] = session
        try:
            import asyncio

            from app.core.redis_client import upsert_monitor_session

            summary = _monitor_session_summary(session)
            loop = asyncio.get_running_loop()
            loop.create_task(upsert_monitor_session(call_id, summary))
        except Exception:
            pass
        event = _session_ready_events.setdefault(call_id, asyncio.Event())
        event.set()
        vinfo(
            logger,
            f"Session created from {phone_number}",
            session=session,
            phase=Phase.CALL_START,
            service="settings",
            primary=providers.llm_name,
            detail=(
                f"llm={providers.llm_name} stt={providers.stt_name} "
                f"tts={providers.tts_name} profile={snapshot.voice_latency_profile} "
                f"fallback={'on' if snapshot.auto_fallback_enabled else 'off'} "
                f"@ {snapshot.captured_at}"
            ),
        )
        return session


def get_session(call_id: str) -> ConversationSession | None:
    """Get an active call session by call_id."""
    return _active_sessions.get(call_id)


async def wait_for_session(
    call_id: str, timeout: float = 5.0
) -> ConversationSession | None:
    """Wait up to ``timeout`` seconds for a session to be created."""
    existing = _active_sessions.get(call_id)
    if existing:
        return existing
    event = _session_ready_events.setdefault(call_id, asyncio.Event())
    try:
        await asyncio.wait_for(event.wait(), timeout=timeout)
    except TimeoutError:
        return None
    return _active_sessions.get(call_id)


def _should_remove_session_after_stream(session: ConversationSession) -> bool:
    """Drop in-memory sessions that ended abnormally (not awaiting Test Console score)."""
    if session.control_flags.get("provider_failure"):
        return True
    if (
        session.current_state == CallState.ENDED.value
        and not session.is_screening_complete()
    ):
        return True
    return False


def finish_stream_session(call_id: str) -> None:
    """Mark a live stream as ended and auto-remove sessions that should not linger."""
    session = get_session(call_id)
    if session is None:
        return
    now = datetime.now(UTC)
    if session.stream_ended_at is None:
        session.stream_ended_at = now
    if session.current_state in (CallState.ENDED.value, CallState.WRAP_UP.value):
        session.ended_at = session.ended_at or now
    if _should_remove_session_after_stream(session):
        remove_session(call_id)
        logger.info("Session auto-removed after stream end: %s", call_id)


def remove_session(call_id: str) -> ConversationSession | None:
    """Remove and return a completed call session."""
    session = _active_sessions.pop(call_id, None)
    _session_ready_events.pop(call_id, None)
    unregister_stream_stop(call_id)
    if session:
        try:
            import asyncio

            from app.core.redis_client import remove_monitor_session

            loop = asyncio.get_running_loop()
            loop.create_task(remove_monitor_session(call_id))
        except Exception:
            pass
        vinfo(
            logger,
            "Session removed",
            call_id=call_id,
            phase=Phase.CALL_END,
            state=session.current_state,
        )
    return session


def _monitor_session_summary(session: ConversationSession) -> dict:
    return {
        "call_id": session.call_id,
        "phone_number": session.phone_number,
        "state": session.current_state,
        "duration": session.duration_seconds,
        "questions_answered": session.questions_answered,
        "active_question_count": session.active_question_count(),
        "started_at": session.started_at.isoformat(),
        "avg_turn_latency_ms": session.avg_turn_latency_ms,
        "last_turn_latency_ms": int(round(session.last_turn_latency_ms)),
        "avg_llm_latency_ms": session.avg_llm_latency_ms,
        "avg_tts_latency_ms": session.avg_tts_latency_ms,
    }


def get_active_sessions() -> list[dict]:
    """Get summary of live-stream sessions for the dashboard (stream still open)."""
    result = []
    for call_id, session in _active_sessions.items():
        if session.stream_ended_at is not None:
            continue
        result.append(_monitor_session_summary(session))
    return result


async def list_monitor_sessions() -> list[dict]:
    """Live sessions on this worker plus any published by other workers."""
    from app.core.redis_client import list_remote_monitor_sessions, upsert_monitor_session

    local = get_active_sessions()
    for row in local:
        await upsert_monitor_session(str(row["call_id"]), row)

    remote = await list_remote_monitor_sessions()
    merged = {str(row["call_id"]): row for row in remote}
    for row in local:
        merged[str(row["call_id"])] = row
    return list(merged.values())


def touch_monitor_session(call_id: str) -> None:
    """Refresh Redis live-monitor payload during an in-progress call."""
    session = _active_sessions.get(call_id)
    if session is None or session.stream_ended_at is not None:
        return
    try:
        import asyncio

        from app.core.redis_client import upsert_monitor_session

        asyncio.get_running_loop().create_task(
            upsert_monitor_session(call_id, _monitor_session_summary(session))
        )
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Greeting
# ──────────────────────────────────────────────────────────────────────────────


async def handle_call_answered(session: ConversationSession) -> list[bytes]:
    """
    Handle the initial call answer — generate and return greeting audio.

    Returns:
        Audio bytes for the greeting message
    """
    blocked = session.control_flags.get("questions_config_blocked")
    if blocked:
        reason = str(blocked)
        session.control_flags["questions_config_fallback"] = reason
        session.add_error("questions_config_blocked", reason)
        session.control_flags["provider_failure"] = {
            "service": "questions",
            "detail": reason,
        }
        session.current_state = CallState.ENDED.value
        message = provider_failure_message_for_session(session)
        session.add_transcript("AI", message)
        session.add_message("assistant", message)
        parts = await synthesize_speech_parts(message, "", session, combine=True)
        return list(parts) if parts else []

    if session.current_state != CallState.IDLE.value:
        logger.debug(f"Session {session.call_id} already past IDLE — skipping greeting")
        return b""

    session.current_state = CallState.GREETING.value
    business = (session.property_name or "").strip() or BUSINESS_NAME
    if session.greeting_message:
        intro = session.greeting_message.replace("{property_name}", business)
    else:
        intro = build_greeting_intro(business, language_code=session.call_language)
    first_state = first_active_question_state(session.questions)
    session.current_state = first_state or CallState.WRAP_UP.value
    q1 = session.get_current_question()
    question = (
        localized_question_text(
            q1, language_code=session.call_language, key="question"
        )
        if q1
        else "Let's get started with your screening."
    )
    full_greeting = f"{intro} {question}"

    session.add_transcript("AI", full_greeting)

    parts = await synthesize_speech_parts(intro, question, session, combine=True)
    if not parts:
        _, shutdown_parts, _ = await end_call_for_provider_failure(
            session, "tts", "Greeting speech synthesis failed"
        )
        return shutdown_parts
    return parts


def _strip_streamed_speech(text: str, session: ConversationSession) -> str:
    """Remove text already spoken during LLM token streaming."""
    text = (text or "").strip()
    prefix = (getattr(session, "streamed_speakable_prefix", "") or "").strip()
    if not text or not prefix:
        return text
    if text.startswith(prefix):
        return text[len(prefix) :].strip()
    return text


AudioPartCallback = Callable[[bytes, bool], Awaitable[None]]
SpeakableCallback = Callable[[str], Awaitable[None]]


async def _maybe_deliver_audio_part(
    session: ConversationSession,
    on_part_ready: AudioPartCallback | None,
    audio: bytes,
    is_last: bool,
) -> bool:
    if not audio or _turn_tts_suppressed(session):
        return False
    if on_part_ready:
        await on_part_ready(audio, is_last)
    return True


async def process_tenant_speech(
    session: ConversationSession,
    transcript: str,
    *,
    on_audio_part: AudioPartCallback | None = None,
    on_stream_text: Callable[[str], Awaitable[None]] | None = None,
) -> tuple[str, list[bytes], bool]:
    """
    Process one caller utterance with deterministic Ready Rentals flow control.

    The LLM may extract fields, but the next question is selected from the
    canonical screening flow so test console and Telnyx behave the same.
    """
    if not transcript.strip():
        return await handle_silence(session)

    # The caller hung up (or the call already ended) while this turn was queued.
    # Abort before spending an LLM call and, more importantly, before mutating
    # state on a session that is being torn down — otherwise a late-completing
    # turn could advance state or re-trigger work after the call is gone.
    if getattr(session, "pending_hangup", False) or (
        session.current_state == CallState.ENDED.value
    ):
        logger.info(
            "[%s] Ignoring speech — call ending (state=%s, pending_hangup=%s)",
            session.call_id,
            session.current_state,
            getattr(session, "pending_hangup", False),
        )
        return "", [], False

    prior_state = session.current_state

    # Technical mic guard ONLY (this is NOT "understanding the caller"): drop the
    # agent's own voice echoing back on speakerphone/hands-free so it never
    # answers itself. Everything about what the caller MEANT is decided by the LLM.
    if is_echo_of_agent(transcript, session):
        logger.info("[%s] Ignoring agent echo: %r", session.call_id, transcript)
        return "", [], False

    _begin_turn_tts(session)

    vinfo(
        logger,
        f"Tenant: {transcript}",
        session=session,
        phase=Phase.TENANT,
        service="turn",
    )
    session.add_transcript("Tenant", transcript)
    session.add_message("user", transcript)
    session.silence_count = 0
    session.streamed_speakable_prefix = ""
    session.llm_streamed_during_turn = False
    buffered_speakables: list[str] = []

    async def _on_speakable_buffer(sentence: str) -> None:
        sentence = (sentence or "").strip()
        if sentence:
            buffered_speakables.append(sentence)

    async def _release_buffered_speakables(*, discard: bool) -> None:
        if not buffered_speakables:
            return
        if discard:
            vinfo(
                logger,
                f"Discarding {len(buffered_speakables)} buffered LLM speakable(s) — read-back only",
                session=session,
                phase=Phase.TTS_DEDUP_SKIP,
                service="llm",
                detail=" ".join(s[:40] for s in buffered_speakables)[:120],
            )
            buffered_speakables.clear()
            return
        for sentence in buffered_speakables:
            await _on_speakable_during_llm(sentence)
        buffered_speakables.clear()

    async def _on_speakable_during_llm(sentence: str) -> None:
        sentence = (sentence or "").strip()
        if not sentence or _turn_tts_suppressed(session):
            return
        combined = session.append_streaming_ai_transcript(sentence)
        vinfo(
            logger,
            f"LLM speakable sentence → stream TTS: {sentence[:60]!r}",
            session=session,
            phase=Phase.STREAM_TTS,
            service="llm",
            detail=f"combined={combined[:80]!r}",
        )
        if on_stream_text:
            await on_stream_text(combined)
        if on_audio_part:
            synth_task = _track_turn_tts_task(
                session,
                synthesize_with_fallback(
                    sentence,
                    session,
                    budget_remaining_s=_remaining_turn_budget_s(session),
                ),
            )
            audio = await synth_task
            if audio:
                session.streamed_audio_sent_during_turn = True
                vinfo(
                    logger,
                    f"Stream TTS ready ({len(audio)} bytes) for sentence",
                    session=session,
                    phase=Phase.STREAM_TTS,
                    service="tts",
                    provider=session.tts_provider,
                    bytes=len(audio),
                    detail=f"sentence={sentence[:60]!r}",
                )
                await _maybe_deliver_audio_part(session, on_audio_part, audio, False)
            else:
                vwarn(
                    logger,
                    f"Stream TTS failed for sentence: {sentence[:60]!r}",
                    session=session,
                    phase=Phase.STREAM_TTS,
                    service="tts",
                    reason="empty_audio",
                )

    async def finish_turn(
        response_text: str,
        *,
        complete: bool = False,
        ack: str | None = None,
        follow_up: str = "",
        combine_audio: bool = False,
        require_speech: bool = False,
    ) -> tuple[str, list[bytes], bool]:
        if getattr(session, "pending_hangup", False):
            return "", [], False
        if require_speech:
            session.tts_confirm_priority = True
            await _release_buffered_speakables(discard=True)
        else:
            await _release_buffered_speakables(discard=False)
        try:
            if require_speech:
                reset_turn_streaming(session, full=True)
                spoken = response_text.strip()
                ack = (ack if ack is not None else response_text).strip()
                follow_up = (follow_up or "").strip()
            else:
                spoken = _strip_streamed_speech(response_text.strip(), session)
                ack = _strip_streamed_speech(
                    (ack if ack is not None else response_text).strip(), session
                )
                follow_up = _strip_streamed_speech((follow_up or "").strip(), session)
            spoken = dedupe_repeated_speech(spoken)
            ack = dedupe_repeated_speech(ack or "")
            follow_up = dedupe_repeated_speech(follow_up)
            ack, follow_up = normalize_speech_parts(ack, follow_up)
            display = compose_spoken_display(
                spoken=spoken,
                ack=ack,
                follow_up=follow_up,
                response_text=response_text,
            )
            if display:
                if session.streaming_ai_open:
                    session.close_streaming_ai_transcript(display)
                    session.add_message("assistant", display)
                else:
                    session.add_transcript("AI", display)
                    session.add_message("assistant", display)
            streamed_prefix = (session.streamed_speakable_prefix or "").strip()
            intended = (display or response_text or ack or follow_up or "").strip()
            session.turn_streaming_finalize = {
                "streamed_prefix": streamed_prefix,
                "intended": intended,
                "streamed_sent": session.streamed_audio_sent_during_turn,
                "display": display,
            }
            if streamed_audio_complete(session, intended):
                vinfo(
                    logger,
                    "Skipping batch TTS — streaming already covered full response",
                    session=session,
                    phase=Phase.TTS_DEDUP_SKIP,
                    service="tts",
                    detail=(
                        f"prefix={streamed_prefix[:80]!r} "
                        f"intended={intended[:80]!r}"
                    ),
                )
                session.llm_streamed_during_turn = False
                session.streamed_speakable_prefix = ""
                session.streamed_audio_sent_during_turn = False
                session.streaming_ai_open = False
                fin = session.turn_streaming_finalize or {}
                session.turn_streaming_finalize = {
                    **fin,
                    "streamed_prefix": intended,
                    "streamed_sent": True,
                    "live_path": True,
                }
                return display or intended, [], complete
            if not ack and not follow_up and intended:
                ack = _strip_streamed_speech(intended, session) or intended
            unsynthesized = spoken or ack or follow_up
            if not unsynthesized and response_text.strip():
                unsynthesized = _strip_streamed_speech(response_text.strip(), session)
                if unsynthesized:
                    ack = unsynthesized
            if not unsynthesized:
                vinfo(
                    logger,
                    "No unsynthesized speech left after streaming strip",
                    session=session,
                    phase=Phase.TTS_DEDUP_SKIP,
                    detail=(
                        f"display={display[:80]!r} "
                        f"prefix={streamed_prefix[:80]!r}"
                    ),
                )
                session.llm_streamed_during_turn = False
                session.streamed_speakable_prefix = ""
                session.streamed_audio_sent_during_turn = False
                session.streaming_ai_open = False
                return display or intended, [], complete
            vinfo(
                logger,
                "Batch TTS synthesize in finish_turn",
                session=session,
                phase=Phase.TTS_FINISH,
                service="tts",
                detail=(
                    f"ack={ack[:60]!r} follow_up={follow_up[:60]!r} "
                    f"prefix={streamed_prefix[:60]!r} combine={combine_audio}"
                ),
            )
            use_combine = combine_audio
            if not use_combine and ack and follow_up:
                streaming_on = getattr(session, "llm_streaming_enabled", True)
                if not streaming_on and not session.streamed_audio_sent_during_turn:
                    use_combine = True
            audio_parts = await synthesize_speech_parts(
                ack if ack is not None else spoken,
                follow_up,
                session,
                combine=use_combine,
                on_part_ready=on_audio_part,
            )
            if (
                not audio_parts
                and unsynthesized
                and not session.control_flags.get("_shutdown_tts")
                and not session.control_flags.get("provider_failure")
            ):
                return await end_call_for_provider_failure(
                    session, "tts", "All TTS providers failed"
                )
            # Voice WebSocket path: audio was already enqueued live via on_audio_part
            # during LLM streaming and/or finish_turn synthesis — never return bytes
            # for a second post-turn enqueue (that caused duplicated questions).
            if on_audio_part and audio_parts:
                fin = session.turn_streaming_finalize or {}
                session.turn_streaming_finalize = {
                    **fin,
                    "streamed_prefix": intended,
                    "streamed_sent": True,
                    "live_path": True,
                }
                audio_parts = []
            session.llm_streamed_during_turn = False
            session.streamed_speakable_prefix = ""
            session.streamed_audio_sent_during_turn = False
            session.streaming_ai_open = False
            vinfo(
                logger,
                f"finish_turn done: {len(audio_parts)} audio part(s)",
                session=session,
                phase=Phase.TTS_FINISH,
                detail=f"display={display[:80]!r}",
            )
            return display, audio_parts, complete
        finally:
            session.tts_confirm_priority = False

    async def _advance_and_ask(
        answered_state: str,
        ack_text: str | None = None,
    ) -> tuple[str, list[bytes], bool]:
        """Mark the current question answered, move on, and ask the next one."""
        session.mark_answered(answered_state)
        session.retry_count = 0
        prior_state_val = answered_state
        next_state = next_unanswered_state(
            session.extracted_data,
            session.skip_states,
            questions=session.questions,
            confirmed_fields=session.confirmed_fields,
            flow_context=session.conditional_flow_context(),
            raw_answers=session.raw_answers,
        )
        target_state = next_state or CallState.WRAP_UP.value
        if not _apply_guarded_state_transition(
            session,
            target_state,
            "Answered (confirmed)",
            retry_count=0,
        ):
            session.current_state = CallState.WRAP_UP.value
        session.refresh_progress()
        touch_monitor_session(session.call_id)
        if screening_complete(
            session.extracted_data,
            session.skip_states,
            questions=session.questions,
            confirmed_fields=session.confirmed_fields,
            flow_context=session.conditional_flow_context(),
            raw_answers=session.raw_answers,
        ):
            session.current_state = CallState.WRAP_UP.value
            session.refresh_progress()
            touch_monitor_session(session.call_id)
        text = strip_upcoming_question_from_ack(
            session, ack_text or human_ack(session)
        )
        ack, follow_up = compose_agent_response(session, text, answered_state)
        response_text = " ".join(part for part in (ack, follow_up) if part).strip()
        is_complete = session.current_state in (
            CallState.WRAP_UP.value,
            CallState.ENDED.value,
        )
        reset_turn_streaming(session)
        return await finish_turn(
            response_text,
            complete=is_complete,
            ack=ack,
            follow_up=follow_up,
        )

    async def _maybe_request_confirmation(
        answered_state: str,
    ) -> tuple[str, list[bytes], bool] | None:
        """For high-stakes fields, read the value back instead of advancing.

        Returns a finished turn (the read-back question) when confirmation is
        needed, or None when the field isn't high-stakes / has no value.
        """
        read_back = try_open_readback_confirmation(session, answered_state)
        if not read_back:
            return None
        pending = session.pending_confirmation or {}
        logger.info(
            "[%s] Read-back confirm %s=%r",
            session.call_id,
            pending.get("field"),
            pending.get("value"),
        )
        return await finish_turn(
            read_back, ack=read_back, follow_up="", require_speech=True
        )

    async def _repair_confirmation_field(
        state_obj: str,
        field: str,
    ) -> tuple[str, list[bytes], bool]:
        """Clear a failed read-back and re-ask the owning question."""
        session.reopen_question_after_failed_confirmation(state_obj, field)
        prompt = repair_prompt_for_state(
            state_obj,
            session.questions,
            language_code=session.call_language,
        )
        return await finish_turn(prompt, ack=prompt, follow_up="", require_speech=True)

    async def _reask_current(
        ack_text: str | None = None,
    ) -> tuple[str, list[bytes], bool]:
        """Acknowledge, then re-ask whatever question we were on (no advance).

        Used after resolving a mid-call correction so the caller is gently
        returned to the question they were answering before the detour.
        """
        question = session.get_current_question()
        follow_up = localized_question_text(
            question,
            language_code=session.call_language,
            key="question",
        ) or (
            "Donde ibamos? Adelante cuando este listo."
            if _is_spanish(session)
            else "Where were we — go ahead whenever you're ready."
        )
        ack = ack_text or _localize(session, "Perfect, thank you.", "Perfecto, gracias.")
        response = " ".join(part for part in (ack, follow_up) if part).strip()
        reset_turn_streaming(session)
        return await finish_turn(response, ack=ack, follow_up=follow_up)

    if is_meta_navigation_request(transcript):
        repeat = navigation_repeat_text(session)
        logger.info("[%s] Meta navigation — admin repeat", session.call_id)
        return await finish_turn(
            repeat,
            ack=repeat,
            follow_up="",
            require_speech=bool(session.pending_confirmation),
        )

    # Liveness ack right after a silence nudge — re-ask before the LLM so short
    # replies (e.g. Spanish "si") are not mis-extracted as screening answers.
    if session.silence_nudge_active:
        session.silence_nudge_active = False
        if is_liveness_acknowledgment(
            transcript, language_code=session.call_language
        ):
            logger.info(
                "[%s] Liveness ack after silence nudge — re-asking %s",
                session.call_id,
                prior_state,
            )
            question = session.get_current_question()
            prompt = localized_question_text(
                question,
                language_code=session.call_language,
                key="question",
            ) or _localize(
                session,
                "Please go ahead when you're ready.",
                "Adelante cuando este listo.",
            )
            return await finish_turn(
                prompt,
                ack=_localize(session, "Great, thanks.", "Perfecto, gracias."),
                follow_up=prompt,
            )

    # ── The single LLM brain ────────────────────────────────────────────────
    # ONE call resolves intent, FAQ answering, field extraction, and the spoken
    # reply. There is no regex router and no raw-speech parsing — the model is
    # the sole intelligence. Regex only normalizes extracted values afterwards.
    pending = session.pending_confirmation
    system_prompt = build_system_prompt(
        session,
        transcript=transcript,
        confirmation=pending,
    )
    llm_response_data = await get_llm_response_with_fallback(
        session=session,
        system_prompt=system_prompt,
        messages=session.messages,
        voice_mode=True,
        budget_remaining_s=_remaining_turn_budget_s(session),
        on_speakable_sentence=_on_speakable_buffer
        if on_audio_part and getattr(session, "llm_streaming_enabled", True)
        else None,
    )
    if getattr(session, "pending_hangup", False):
        logger.info(
            "[%s] Ignoring LLM result — hangup in progress",
            session.call_id,
        )
        await _release_buffered_speakables(discard=True)
        session.reconcile_interrupted_turn()
        return "", [], False
    if llm_response_data.get("provider_shutdown"):
        return await end_call_for_provider_failure(
            session,
            str(llm_response_data.get("provider_service") or "llm"),
            str(llm_response_data.get("provider_detail") or "unavailable"),
        )
    response_text = str(llm_response_data.get("response_text", "")).strip()
    understood = bool(llm_response_data.get("understood", False))
    intent_kind = parse_turn_intent(llm_response_data)
    faq_topic = llm_response_data.get("faq_topic")
    extracted = llm_response_data.get("extracted_data", {}) or {}
    question_complete = parse_question_complete(llm_response_data)
    relevance = parse_relevance(llm_response_data)
    field_to_state, _ = field_maps_for_session(session)
    corrected_fields = parse_corrected_fields(llm_response_data, field_to_state)
    consistency_issue = parse_issue(llm_response_data, "consistency_issue")
    plausibility_issue = parse_issue(llm_response_data, "plausibility_issue")
    logger.info(
        "[%s] LLM intent=%s faq=%s understood=%s q_complete=%s relevance=%s "
        "corrected=%s consistency=%s plausibility=%s extracted=%s",
        session.call_id,
        intent_kind,
        faq_topic,
        understood,
        question_complete,
        relevance,
        corrected_fields,
        bool(consistency_issue),
        bool(plausibility_issue),
        sorted(extracted) if isinstance(extracted, dict) else extracted,
    )

    # Flatten any {"value": x, "confidence": y} envelopes the model may emit.
    clean_extracted: dict = {}
    for key, value in (extracted.items() if isinstance(extracted, dict) else []):
        if isinstance(value, dict) and "value" in value:
            value = value.get("value")
        if value not in (None, ""):
            clean_extracted[key] = value
    extracted = clean_extracted
    question_cfg = session.get_current_question() or {}
    q_fields = {str(f) for f in (question_cfg.get("extract_fields") or [])}
    if (
        not any(k in extracted for k in ("preferred_language", "language", "call_language"))
        and q_fields.intersection({"preferred_language", "language", "call_language"})
    ):
        if str(question_cfg.get("answer_type")) == "language_choice":
            guessed_lang = resolve_language_choice(transcript, question_cfg)
        else:
            guessed_lang = canonical_language_code(transcript)
        if guessed_lang:
            extracted["preferred_language"] = guessed_lang

    # Caller's words were just our own audio echoing back with nothing new.
    if intent_kind == "echo" and not extracted:
        await _release_buffered_speakables(discard=True)
        if is_meta_navigation_request(transcript):
            repeat = navigation_repeat_text(session)
            logger.info("[%s] Echo veto — meta navigation repeat", session.call_id)
            return await finish_turn(
                repeat,
                ack=repeat,
                follow_up="",
                require_speech=bool(session.pending_confirmation),
            )
        logger.info("[%s] LLM flagged echo: %r", session.call_id, transcript)
        if session.messages and session.messages[-1].get("role") == "user":
            session.messages.pop()
        if session.transcript and session.transcript[-1].speaker == "Tenant":
            session.transcript.pop()
        return "", [], False

    # ── Pending CORRECTION read-back ─────────────────────────────────────────
    # The previous turn confirmed one or more EARLIER fields the caller changed
    # mid-call. The same LLM call classified their reply (confirm / re-correct).
    if pending and pending.get("mode") == "correction":
        cfields = pending.get("fields", []) or []
        return_state = str(
            pending.get("return_state", session.current_state)
        )

        re_changed: dict[str, str] = {}
        for c in cfields:
            f = c["field"]
            nv = extracted.get(f)
            if nv and str(nv).strip().lower() != str(c.get("value", "")).strip().lower():
                re_changed[f] = str(nv)

        if re_changed:
            session.merge_extracted_data(
                re_changed,
                raw_text=transcript,
                allow_overwrite=frozenset(re_changed),
            )
            for c in cfields:
                if c["field"] in re_changed:
                    c["value"] = str(
                        session.extracted_data.get(c["field"], re_changed[c["field"]])
                    )
            pending["attempts"] = pending.get("attempts", 1) + 1
            if pending["attempts"] > confirmation_attempt_limit(session):
                session.pending_confirmation = None
                for c in cfields:
                    session.mark_field_confirmed(c["field"])
                session.current_state = return_state
                return await _reask_current()
            rb = build_correction_readback(cfields, session=session)
            logger.info(
                "[%s] Correction re-adjusted: %s", session.call_id, sorted(re_changed)
            )
            return await finish_turn(rb, ack=rb, follow_up="", require_speech=True)

        if intent_kind in ("answer", "nothing") and understood:
            session.pending_confirmation = None
            for c in cfields:
                session.mark_field_confirmed(c["field"])
            if faq_topic and faq_topic not in session.faq_topics:
                session.faq_topics.append(faq_topic)
            session.current_state = return_state
            logger.info(
                "[%s] Caller confirmed corrected fields: %s",
                session.call_id,
                [c["field"] for c in cfields],
            )
            return await _reask_current(ack_text=response_text or None)

        if intent_kind in ("human", "stop"):
            # Hard end clears read-back; soft callback may preserve it below.
            session.pending_confirmation = None
        elif intent_kind != "callback":
            # Refusal / unclear — re-read the correction, then return to the question.
            pending["attempts"] = pending.get("attempts", 1) + 1
            if pending["attempts"] > confirmation_attempt_limit(session):
                session.pending_confirmation = None
                session.current_state = return_state
                return await _reask_current()
            again = response_text or build_correction_readback(cfields, session=session)
            return await finish_turn(again, ack=again, follow_up="", require_speech=True)
        # callback: hand off to soft redirect without clearing pending read-back.

    # ── Pending read-back confirmation ──────────────────────────────────────
    # The previous turn read a high-stakes field (name/phone/email) back to the
    # caller; the same LLM call classified their reply (confirm/correct/reject).
    if pending and not pending.get("mode"):
        field = pending["field"]
        state_obj = str(pending["state"])

        corrected = extracted.get(field)
        new_val = (
            str(corrected)
            if corrected
            and str(corrected).strip().lower() != str(pending["value"]).strip().lower()
            else None
        )

        if new_val:
            session.merge_extracted_data({field: new_val}, raw_text=transcript)
            pending["value"] = session.extracted_data.get(field, new_val)
            pending["attempts"] = pending.get("attempts", 1) + 1
            if pending["attempts"] > confirmation_attempt_limit(session):
                return await _repair_confirmation_field(state_obj, field)
            read_back = readback_prompt_for_state(
                pending["state"],
                str(pending["value"]),
                session.questions,
                language_code=session.call_language,
            )
            logger.info(
                "[%s] Caller corrected %s -> %r", session.call_id, field, new_val
            )
            return await finish_turn(
                read_back, ack=read_back, follow_up="", require_speech=True
            )

        if intent_kind in ("answer", "nothing") and understood:
            logger.info("[%s] Caller confirmed %s", session.call_id, field)
            session.pending_confirmation = None
            session.mark_field_confirmed(field)
            if faq_topic and faq_topic not in session.faq_topics:
                session.faq_topics.append(faq_topic)
            return await _advance_and_ask(
                state_obj,
                ack_text=strip_upcoming_question_from_ack(
                    session, response_text or ""
                )
                or None,
            )

        if intent_kind == "refusal":
            logger.info("[%s] Caller rejected %s; repairing", session.call_id, field)
            session.reopen_question_after_failed_confirmation(state_obj, field)
            prompt = response_text or repair_prompt_for_state(
                pending["state"],
                session.questions,
                language_code=session.call_language,
            )
            return await finish_turn(
                prompt, ack=prompt, follow_up="", require_speech=True
            )

        if intent_kind not in ("human", "callback", "stop"):
            # Unclear — re-ask the read-back; never auto-advance without confirmation.
            pending["attempts"] = pending.get("attempts", 1) + 1
            if pending["attempts"] > confirmation_attempt_limit(session):
                return await _repair_confirmation_field(state_obj, field)
            again = response_text or (
                _localize(
                    session,
                    "Sorry, I just want to be sure. ",
                    "Perdon, solo quiero asegurarme. ",
                )
                + readback_prompt_for_state(
                    pending["state"],
                    str(pending["value"]),
                    session.questions,
                    language_code=session.call_language,
                )
            )
            return await finish_turn(
                again, ack=again, follow_up="", require_speech=True
            )
        # human/callback/stop fall through to control handling below.

    # ── Call-control intent (human / callback / stop), decided by the LLM ────
    control_intent = control_flag_for_intent(intent_kind)
    if control_intent:
        if control_intent == "callback_requested" and not session.control_flags.get(
            "callback_redirect_offered"
        ):
            session.control_flags["callback_redirect_offered"] = True
            response_text = soft_callback_redirect_text(session)
            logger.info(
                "[%s] Callback redirect (soft) — staying on %s",
                session.call_id,
                session.current_state,
            )
            return await finish_turn(
                response_text,
                complete=False,
                require_speech=bool(session.pending_confirmation),
            )

        session.control_flags[control_intent] = True
        session.merge_extracted_data(
            {control_intent: True, "special_notes": transcript},
            raw_text=transcript,
        )
        prior_state_val = session.current_state
        _apply_guarded_state_transition(
            session,
            CallState.ENDED.value,
            f"Control intent: {control_intent}",
            retry_count=session.retry_count,
        )
        if control_intent == "human_requested":
            response_text = response_text or (
                _localize(
                    session,
                    "Of course. If you'd like to speak with a team member, I can forward your information and have someone reach out as soon as possible.",
                    "Claro. Si desea hablar con una persona del equipo, puedo enviar su informacion para que le contacten pronto.",
                )
            )
        elif control_intent == "callback_requested":
            response_text = response_text or (
                _localize(
                    session,
                    "No problem. I will note that you asked for a call back later.",
                    "No hay problema. Dejare anotado que prefiere que le devolvamos la llamada.",
                )
            )
        else:
            response_text = response_text or (
                _localize(
                    session,
                    "No problem. We will save what you shared so far.",
                    "No hay problema. Guardaremos lo que compartio hasta ahora.",
                )
            )
        return await finish_turn(response_text, complete=True)

    # ── Relevance gate: off-topic or unintelligible replies ─────────────────
    # Two distinct cases, handled differently:
    #   off_topic  → caller said something unrelated ("I want to go swimming").
    #                Steer back firmly; counts fully against retries.
    #   unclear    → we couldn't make sense of it: likely garbled audio, a heavy
    #                accent, or STT error — NOT evasion. Apologize and invite a
    #                repeat, and give one extra attempt before giving up so a
    #                caller genuinely trying to answer isn't cut off by bad audio.
    if relevance in ("off_topic", "unclear") and intent_kind in ("answer", "nothing"):
        session.retry_count += 1
        is_unclear = relevance == "unclear"
        # Allow one extra attempt for unclear (audio) before moving on.
        limit = session.max_retries + (1 if is_unclear else 0)
        if session.retry_count > limit:
            prior_state_val = session.current_state
            session.mark_refused(session.current_state, transcript)
            session.next_state()
            target_state = session.current_state
            session.current_state = prior_state_val
            if not _apply_guarded_state_transition(
                session,
                target_state,
                f"Relevance limit reached (relevance={relevance})",
                retry_count=session.retry_count,
            ):
                session.current_state = prior_state_val
            text = human_ack(session)
            ack, follow_up = compose_agent_response(session, text, prior_state)
            text = " ".join(part for part in (ack, follow_up) if part).strip()
            is_complete = session.current_state in (
                CallState.WRAP_UP.value,
                CallState.ENDED.value,
            )
            return await finish_turn(
                text, complete=is_complete, ack=ack, follow_up=follow_up
            )
        redirect = response_text or polite_redirect(
            session, "unclear" if is_unclear else "non_answer"
        )
        logger.info(
            "[%s] Relevance=%s — %s on %s",
            session.call_id,
            relevance,
            "asking to repeat" if is_unclear else "steering back",
            prior_state,
        )
        return await finish_turn(redirect, ack=redirect, follow_up="")

    # Record any FAQ the caller asked — the LLM already spoke the approved answer
    # inside response_text. Topic tracking feeds analytics and the summary email.
    if faq_topic and faq_topic not in session.faq_topics:
        session.faq_topics.append(faq_topic)
        logger.info("[%s] FAQ answered by LLM: %s", session.call_id, faq_topic)

    # The LLM is the sole authority on extraction. Store its values; regex only
    # normalizes formats afterwards (see normalize_extracted_fields).
    if extracted:
        if getattr(session, "pending_hangup", False):
            logger.info(
                "[%s] Ignoring LLM extraction — hangup in progress",
                session.call_id,
            )
            return "", [], False
        session.merge_extracted_data(
            extracted,
            raw_text=transcript,
            allow_overwrite=frozenset(corrected_fields),
        )
        for key in ("preferred_language", "language", "call_language"):
            if key in session.extracted_data:
                await _apply_session_language(session, session.extracted_data.get(key))
                break
        logger.info(
            "[%s] Extracted fields merged: %s", session.call_id, sorted(extracted)
        )

    # ── Mid-call correction of an EARLIER answer ─────────────────────────────
    # The caller changed a field that belongs to a previous question (e.g. fixed
    # their phone while on the income question). The new value is already merged
    # above; read every corrected value back to confirm, then return to where we
    # were. Skips fields owned by the CURRENT question (normal flow handles those).
    if corrected_fields and not session.pending_confirmation:
        to_confirm: list[dict[str, str]] = []
        field_to_state, _ = field_maps_for_session(session)
        for field_name in corrected_fields:
            owner = field_to_state.get(field_name)
            if not owner or owner == prior_state:
                continue
            value = session.extracted_data.get(field_name)
            if value in (None, ""):
                continue
            to_confirm.append({"field": field_name, "value": str(value)})
        if to_confirm:
            session.pending_confirmation = {
                "mode": "correction",
                "fields": to_confirm,
                "return_state": prior_state,
                "attempts": 1,
            }
            read_back = build_correction_readback(to_confirm, session=session)
            logger.info(
                "[%s] Caller corrected earlier field(s): %s",
                session.call_id,
                [c["field"] for c in to_confirm],
            )
            return await finish_turn(
                read_back, ack=read_back, follow_up="", require_speech=True
            )

    # ── Sanity check: consistency or plausibility issue flagged by the LLM ───
    # The model — which sees the full conversation and all extracted data — can
    # flag a value that contradicts an earlier answer or is simply implausible
    # (e.g. a monthly income that is really an hourly wage). Ask ONE friendly
    # clarifying question per issue per question; never nag about fine answers.
    issue_text = plausibility_issue or consistency_issue
    if issue_text:
        issue_kind = "plausibility" if plausibility_issue else "consistency"
        clarified_key = f"{issue_kind}_clarified_{prior_state}"
        if not session.control_flags.get(clarified_key):
            session.control_flags[clarified_key] = True
            clarify = response_text or _plausibility_clarify_fallback(session)
            logger.info(
                "[%s] %s issue on %s: %s",
                session.call_id,
                issue_kind,
                prior_state,
                issue_text,
            )
            return await finish_turn(clarify, ack=clarify, follow_up="")

    deterministic_done = is_question_answered(
        prior_state,
        session.extracted_data,
        session.skip_states,
        questions=session.questions,
        confirmed_fields=session.confirmed_fields,
        flow_context=session.conditional_flow_context(),
        raw_answers=session.raw_answers,
    )
    # The LLM is the brain: when it understood the caller but still marks the
    # current question incomplete, stay — even if a rough slot is already filled
    # (e.g. move_in_raw without an exact date). question_complete=true always
    # advances; punctuation in response_text is not consulted.
    done = question_advance_ready(
        question_complete=question_complete,
        deterministic_done=deterministic_done,
        understood=understood,
    )

    async def _finish_advance(
        ack_text: str | None = None,
        *,
        reason: str = "Question answered successfully",
    ) -> tuple[str, list[bytes], bool]:
        """Read-back high-stakes fields if needed, then advance to next question."""
        confirm_turn = await _maybe_request_confirmation(prior_state)
        if confirm_turn is not None:
            return confirm_turn
        session.mark_answered(prior_state)
        session.retry_count = 0
        prior_state_val = prior_state
        next_st = next_unanswered_state(
            session.extracted_data,
            session.skip_states,
            questions=session.questions,
            confirmed_fields=session.confirmed_fields,
            flow_context=session.conditional_flow_context(),
            raw_answers=session.raw_answers,
        )
        new_state = next_st or CallState.WRAP_UP.value
        if not _apply_guarded_state_transition(
            session,
            new_state,
            reason,
            retry_count=0,
        ):
            session.current_state = CallState.WRAP_UP.value
        session.refresh_progress()
        touch_monitor_session(session.call_id)
        text = strip_upcoming_question_from_ack(
            session, (ack_text or response_text or "").strip() or human_ack(session)
        )
        ack, follow_up = compose_agent_response(session, text, prior_state)
        spoken = " ".join(part for part in (ack, follow_up) if part).strip()
        if screening_complete(
            session.extracted_data,
            session.skip_states,
            questions=session.questions,
            confirmed_fields=session.confirmed_fields,
            flow_context=session.conditional_flow_context(),
            raw_answers=session.raw_answers,
        ):
            session.current_state = CallState.WRAP_UP.value
            session.refresh_progress()
            touch_monitor_session(session.call_id)
        is_complete = session.current_state in (
            CallState.WRAP_UP.value,
            CallState.ENDED.value,
        )
        reset_turn_streaming(session)
        return await finish_turn(
            spoken,
            complete=is_complete,
            ack=ack,
            follow_up=follow_up,
        )

    async def _refuse_and_advance(
        *,
        reason: str,
    ) -> tuple[str, list[bytes], bool]:
        """Mark the current question declined and move to the next one."""
        prior_state_val = session.current_state
        session.mark_refused(session.current_state, transcript)
        session.next_state()
        target_state = session.current_state
        session.current_state = prior_state_val
        if not _apply_guarded_state_transition(
            session,
            target_state,
            reason,
            retry_count=session.retry_count,
        ):
            session.current_state = prior_state_val
        text = human_ack(session)
        ack, follow_up = compose_agent_response(session, text, prior_state)
        spoken = " ".join(part for part in (ack, follow_up) if part).strip()
        is_complete = session.current_state in (
            CallState.WRAP_UP.value,
            CallState.ENDED.value,
        )
        return await finish_turn(
            spoken,
            complete=is_complete,
            ack=ack,
            follow_up=follow_up,
        )

    prior_q = questions_index(session.questions).get(prior_state)

    # Optional questions: caller explicitly has nothing to add.
    if intent_kind == "nothing" and understood:
        if prior_q and not is_question_required(prior_q):
            fields = prior_q.get("extract_fields") or []
            primary = str(fields[0]) if fields else ""
            if primary and not session.extracted_data.get(primary):
                session.merge_extracted_data(
                    {primary: "None disclosed"}, raw_text=transcript
                )
            session.mark_completed(prior_state)
            return await _finish_advance(
                ack_text=response_text or None,
                reason="Optional question — caller has nothing to add",
            )

    if done:
        if needs_readback_confirmation(
            prior_state,
            session.extracted_data,
            session.questions,
            session.confirmed_fields,
        ):
            confirm_turn = await _maybe_request_confirmation(prior_state)
            if confirm_turn is not None:
                return confirm_turn

        if (
            prior_q
            and is_question_required(prior_q)
            and not deterministic_done
        ):
            logger.info(
                "[%s] Required %s incomplete — blocking early advance",
                session.call_id,
                prior_state,
            )
            session.retry_count += 1
            if session.retry_count > session.max_retries:
                return await _refuse_and_advance(
                    reason="Required question incomplete after max retries",
                )
            prompt = response_text or polite_redirect(session, "non_answer")
            return await finish_turn(prompt, ack=prompt, follow_up="")

        if question_complete and not deterministic_done:
            if not prior_q or not is_question_required(prior_q):
                logger.info(
                    "[%s] LLM marked %s complete (partial slots accepted)",
                    session.call_id,
                    prior_state,
                )
                session.mark_completed(prior_state)
        return await _finish_advance()

    if intent_kind == "question":
        # Caller only asked us something — the LLM already answered it and should
        # have re-asked the current question. Do NOT count this as a failed attempt.
        if not response_text:
            question = session.get_current_question()
            response_text = (
                localized_question_text(
                    question,
                    language_code=session.call_language,
                    key="question",
                )
                if question
                else "How can I help?"
            )
        return await finish_turn(response_text, ack=response_text, follow_up="")

    if understood:
        # Partial answer / mid-build (e.g. spelling, multi-part). Ask for what's
        # still missing. Only count a retry when the caller made NO progress this
        # turn — that way genuine context-building over several turns (spelling a
        # name letter by letter) never gets cut off, while a stalled, vague caller
        # is still bounded so we eventually move on.
        made_progress = bool(extracted)
        slots_satisfied = is_question_answered(
            prior_state,
            session.extracted_data,
            session.skip_states,
            questions=session.questions,
            confirmed_fields=session.confirmed_fields,
            flow_context=session.conditional_flow_context(),
            raw_answers=session.raw_answers,
        )
        if made_progress:
            if needs_readback_confirmation(
                prior_state,
                session.extracted_data,
                session.questions,
                session.confirmed_fields,
            ):
                confirm_turn = await _maybe_request_confirmation(prior_state)
                if confirm_turn is not None:
                    return confirm_turn
            if slots_satisfied:
                session.retry_count = 0
        elif not made_progress:
            session.retry_count += 1
        if session.retry_count > session.max_retries:
            if (
                prior_q
                and is_question_required(prior_q)
                and not slots_satisfied
            ):
                logger.info(
                    "[%s] Required %s still incomplete after retries — declined",
                    session.call_id,
                    prior_state,
                )
                return await _refuse_and_advance(
                    reason="Required question incomplete after bounded follow-ups",
                )
            logger.info(
                "[%s] Bounded follow-ups exhausted on %s (no progress) — accepting "
                "partial",
                session.call_id,
                prior_state,
            )
            session.mark_completed(prior_state)
            return await _finish_advance(
                reason="Bounded follow-ups exhausted, accepting partial answer",
            )
        if not response_text:
            response_text = _localize(
                session,
                "Thanks — I just need one more detail to go with that.",
                "Gracias. Solo me falta un detalle mas para completar eso.",
            )
        return await finish_turn(response_text, ack=response_text, follow_up="")

    # Did not answer the current question (refusal or unclear) — escalate retries.
    session.retry_count += 1
    if session.retry_count > session.max_retries:
        return await _refuse_and_advance(
            reason="Max retries exceeded, marking as refused",
        )

    if not response_text:
        kind = "refusal" if intent_kind == "refusal" else "non_answer"
        response_text = polite_redirect(session, kind)
    return await finish_turn(response_text, ack=response_text, follow_up="")


PROGRESSIVE_SILENCE_PROMPTS_EN = (
    "Are you still there?",
    "Take your time, I'm here when you're ready.",
    "I haven't heard anything. We can continue now or reconnect later.",
)
PROGRESSIVE_SILENCE_PROMPTS_ES = (
    "Sigue ahi?",
    "Tome su tiempo, aqui estoy cuando este listo.",
    "No he escuchado respuesta. Podemos continuar ahora o reconectar mas tarde.",
)


async def end_call_for_provider_failure(
    session: ConversationSession,
    service: str,
    detail: str,
) -> tuple[str, list[bytes], bool]:
    """Speak the admin failure script, end the session, and flag for review."""
    if session.control_flags.get("provider_failure"):
        message = provider_failure_message_for_session(session)
        return message, [], True

    session.control_flags["provider_failure"] = {
        "service": service,
        "detail": detail,
    }
    session.add_error("provider_failure", f"{service}: {detail}")
    _apply_guarded_state_transition(
        session,
        CallState.ENDED.value,
        f"Provider failure: {service}",
        retry_count=session.retry_count,
    )
    message = provider_failure_message_for_session(session)
    session.add_transcript("AI", message)
    session.add_message("assistant", message)
    verror(
        logger,
        f"Ending call — provider failure ({service})",
        session=session,
        phase=Phase.CALL_END,
        service=service,
        reason="provider_failure",
        detail=detail,
    )
    session.control_flags["_shutdown_tts"] = True
    try:
        audio = await synthesize_with_fallback(message, session)
    finally:
        session.control_flags.pop("_shutdown_tts", None)
    parts = [audio] if audio else []
    return message, parts, True


async def handle_silence(session: ConversationSession) -> tuple[str, list[bytes], bool]:
    """Re-prompt on silence with escalating messages, then end the call.

    The caller gets ``len(PROGRESSIVE_SILENCE_PROMPTS)`` nudges; on the next
    silent turn we say goodbye and signal completion so the call hangs up.
    """
    session.silence_count += 1
    logger.warning("[%s] Silence count: %s", session.call_id, session.silence_count)

    prompts = PROGRESSIVE_SILENCE_PROMPTS_ES if _is_spanish(session) else PROGRESSIVE_SILENCE_PROMPTS_EN
    if session.silence_count > len(prompts):
        session.control_flags["ended_after_repeated_silence"] = True
        farewell = _localize(
            session,
            "It seems we may have missed each other. We will save what you shared so far and follow up if needed. Goodbye.",
            "Parece que no logramos escucharnos bien. Guardaremos lo que compartio hasta ahora y daremos seguimiento si hace falta. Adios.",
        )
        session.add_transcript("AI", farewell)
        audio = await synthesize_with_fallback(farewell, session)
        if not audio:
            # Avoid silent hangup when TTS is unavailable.
            session.add_error("silence_goodbye_tts_failed", "No goodbye audio produced")
            session.silence_count = max(0, len(prompts))
            return farewell, [], False
        return farewell, [audio], True

    prompt = prompts[session.silence_count - 1]
    session.silence_nudge_active = True
    session.add_transcript("AI", prompt)
    audio = await synthesize_with_fallback(prompt, session)
    return prompt, [audio] if audio else [], False


# ──────────────────────────────────────────────────────────────────────────────
# LLM with Fallback Chain
# ──────────────────────────────────────────────────────────────────────────────


async def _collect_streamed_llm_raw(
    provider,
    *,
    system_prompt: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    attempt_timeout: float,
    on_speakable_sentence: SpeakableCallback | None,
) -> str | None:
    """Stream LLM tokens; speak complete sentences early when possible."""
    stream_fn = getattr(provider, "stream_response", None)
    if stream_fn is None or on_speakable_sentence is None:
        return None

    from app.core.llm_streaming import drain_speakable_sentences

    buffer = ""
    spoken_through = 0
    tts_tasks: list[asyncio.Task] = []

    async def _drain_tts_tasks(*, cancel_pending: bool) -> None:
        """Wait/cancel sentence-level TTS tasks so none leak past this turn."""
        if not tts_tasks:
            return
        if cancel_pending:
            for task in tts_tasks:
                if not task.done():
                    task.cancel()
        results = await asyncio.gather(*tts_tasks, return_exceptions=True)
        for item in results:
            if isinstance(item, asyncio.CancelledError):
                continue
            if isinstance(item, Exception):
                logger.warning("Streaming TTS task failed: %s", item)

    async def _consume() -> str:
        nonlocal buffer, spoken_through
        async for delta in stream_fn(
            system_prompt=system_prompt,
            messages=messages,
            json_mode=True,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            buffer += delta
            sentences, spoken_through = drain_speakable_sentences(
                buffer, spoken_through
            )
            for sentence in sentences:
                tts_tasks.append(asyncio.create_task(on_speakable_sentence(sentence)))
        return buffer

    try:
        raw = await asyncio.wait_for(_consume(), timeout=attempt_timeout)
        await _drain_tts_tasks(cancel_pending=False)
        return raw
    except asyncio.CancelledError:
        await _drain_tts_tasks(cancel_pending=True)
        raise
    except TimeoutError:
        await _drain_tts_tasks(cancel_pending=True)
        raise
    except Exception:
        await _drain_tts_tasks(cancel_pending=True)
        return None


async def get_llm_response_with_fallback(
    session: ConversationSession,
    system_prompt: str,
    messages: list[dict],
    max_retries: int = 2,
    voice_mode: bool = False,
    budget_remaining_s: float | None = None,
    on_speakable_sentence: SpeakableCallback | None = None,
) -> dict:
    """
    Get LLM response with automatic provider fallback chain.
    Groq → OpenAI → OpenRouter → hardcoded fallback.

    Returns:
        Parsed LLM response dict
    """
    if voice_mode:
        max_retries = 1
        llm_timeout = float(getattr(session, "llm_timeout_voice_seconds", 5.5) or 5.5)
        max_tokens = 120
    else:
        llm_timeout = 5.0
        max_tokens = 300

    # Admin-tunable overrides (snapshot at call start). 0 tokens keeps the tuned
    # per-turn default above; temperature is clamped to a safe 0.0–1.0 range.
    admin_max_tokens = getattr(session, "llm_max_tokens", 0) or 0
    if admin_max_tokens > 0:
        max_tokens = admin_max_tokens
    temperature = getattr(session, "llm_temperature", 0.3)
    try:
        temperature = max(0.0, min(1.0, float(temperature)))
    except (TypeError, ValueError):
        temperature = 0.3

    llm_attempt_log: list[dict[str, Any]] = []

    async def try_provider(
        provider,
        provider_name: str,
        *,
        role: str,
    ) -> dict | None:
        """Attempt to get a valid response from a provider."""
        for attempt in range(max_retries + 1):
            remaining = _remaining_turn_budget_s(session)
            if budget_remaining_s is not None:
                remaining = min(remaining if remaining is not None else budget_remaining_s, budget_remaining_s)
            attempt_timeout = llm_timeout
            if remaining is not None:
                remaining = max(0.0, remaining - VOICE_TURN_INTERNAL_BUFFER_SECONDS)
                if remaining < LLM_MIN_ATTEMPT_BUDGET_SECONDS:
                    vwarn(
                        logger,
                        f"Skipping {provider_name} — turn budget too low ({remaining:.2f}s left)",
                        session=session,
                        phase=Phase.LLM_SKIP,
                        service="llm",
                        provider=provider_name,
                        reason="budget_exhausted",
                        budget_s=remaining,
                    )
                    info = session.add_provider_event(
                        service="llm",
                        provider=provider_name,
                        role=role,
                        outcome="skipped",
                        reason="budget_exhausted",
                        detail=f"{remaining:.2f}s left",
                    )
                    llm_attempt_log.append(info)
                    return None
                attempt_timeout = min(llm_timeout, remaining)
            try:
                start = time.time()
                raw = await _collect_streamed_llm_raw(
                    provider,
                    system_prompt=system_prompt,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    attempt_timeout=attempt_timeout,
                    on_speakable_sentence=on_speakable_sentence
                    if voice_mode
                    else None,
                )
                if raw is None:
                    raw = await asyncio.wait_for(
                        provider.get_response(
                            system_prompt=system_prompt,
                            messages=messages,
                            json_mode=True,
                            temperature=temperature,
                            max_tokens=max_tokens,
                        ),
                        timeout=attempt_timeout,
                    )
                latency = (time.time() - start) * 1000
                vinfo(
                    logger,
                    f"{provider_name} response received",
                    session=session,
                    phase=Phase.LLM_OK,
                    service="llm",
                    provider=provider_name,
                    latency_ms=int(latency),
                )

                # Parse and validate
                clean = re.sub(r"```json\s*|\s*```", "", raw).strip()
                if getattr(provider, "provider_name", "") == "gemini":
                    from app.providers.llm.gemini_llm import _extract_json

                    clean = _extract_json(clean)
                data = json.loads(clean)
                is_valid, error = validate_llm_response(data)
                if is_valid:
                    session.llm_provider = provider_name
                    session.record_llm_usage(getattr(provider, "last_usage", None))
                    session.record_llm_latency(latency)
                    await _record_llm_health_success_shared(provider_name, latency)
                    if role == "fallback":
                        session.add_provider_event(
                            service="llm",
                            provider=provider_name,
                            role=role,
                            outcome="succeeded",
                        )
                    return data
                vwarn(
                    logger,
                    f"Invalid LLM response from {provider_name} (attempt {attempt + 1}): {error}",
                    session=session,
                    phase=Phase.LLM_FAIL,
                    service="llm",
                    provider=provider_name,
                    reason="invalid_response",
                    attempt=attempt + 1,
                )
                if attempt >= max_retries:
                    info = session.add_provider_event(
                        service="llm",
                        provider=provider_name,
                        role=role,
                        outcome="failed",
                        reason="invalid_response",
                        detail=str(error or "invalid response"),
                    )
                    llm_attempt_log.append(info)

            except TimeoutError as e:
                vwarn(
                    logger,
                    f"{provider_name} timed out after {attempt_timeout:.2f}s",
                    session=session,
                    phase=Phase.LLM_FAIL,
                    service="llm",
                    provider=provider_name,
                    reason="timeout",
                    timeout_s=attempt_timeout,
                )
                info = session.add_provider_event(
                    service="llm",
                    provider=provider_name,
                    role=role,
                    outcome="failed",
                    exc=e,
                )
                llm_attempt_log.append(info)
                await _record_llm_health_failure_shared(provider_name)
            except json.JSONDecodeError as e:
                vwarn(
                    logger,
                    f"{provider_name} JSON parse failed: {e}",
                    session=session,
                    phase=Phase.LLM_FAIL,
                    service="llm",
                    provider=provider_name,
                    reason="json_parse",
                )
                if attempt >= max_retries:
                    info = session.add_provider_event(
                        service="llm",
                        provider=provider_name,
                        role=role,
                        outcome="failed",
                        reason="invalid_response",
                        detail=str(e),
                    )
                    llm_attempt_log.append(info)
            except Exception as e:
                verror(
                    logger,
                    f"{provider_name} error: {e}",
                    session=session,
                    phase=Phase.LLM_FAIL,
                    service="llm",
                    provider=provider_name,
                    reason="error",
                )
                info = session.add_provider_event(
                    service="llm",
                    provider=provider_name,
                    role=role,
                    outcome="failed",
                    exc=e,
                )
                llm_attempt_log.append(info)
                await _record_llm_health_failure_shared(provider_name)
                break  # Don't retry on auth/config errors
        return None

    providers = get_call_providers(session)

    if voice_mode:
        vdebug(
            logger,
            "LLM voice turn starting",
            session=session,
            phase=Phase.LLM_TRY,
            service="llm",
            provider=providers.llm_name,
            timeout_s=llm_timeout,
            budget_s=budget_remaining_s,
        )

    # Try primary provider
    primary = providers.llm
    vinfo(
        logger,
        f"Primary LLM attempt ({providers.llm_name})",
        session=session,
        phase=Phase.LLM_TRY,
        service="llm",
        provider=providers.llm_name,
        timeout_s=llm_timeout,
    )
    result = await try_provider(primary, providers.llm_name, role="primary")
    if result:
        return result

    # Auto-fallback (honours per-call snapshot)
    if providers.auto_fallback_enabled:
        from app.providers.llm.gemini_llm import GeminiLLMProvider
        from app.providers.llm.groq_llm import GroqLLMProvider
        from app.providers.llm.openai_llm import OpenAILLMProvider
        from app.providers.llm.openrouter_llm import OpenRouterLLMProvider

        primary_kind = (
            "groq"
            if isinstance(primary, GroqLLMProvider)
            else "openai"
            if isinstance(primary, OpenAILLMProvider)
            else "openrouter"
            if isinstance(primary, OpenRouterLLMProvider)
            else "gemini"
            if isinstance(primary, GeminiLLMProvider)
            else ""
        )
        # Candidate backups that are not the primary and have a configured key.
        api_keys = providers.api_keys
        available = [
            name
            for name in (
                "groq",
                "openai",
                "openrouter",
                "gemini",
            )
            if api_keys.configured(name) and name != primary_kind
        ]
        # Admin choice: "none" disables, "auto" tries all available, a specific
        # provider is tried FIRST then the rest remain as a safety backstop.
        pref = (getattr(providers, "llm_fallback_provider", "auto") or "auto").lower()
        if pref == "none":
            ordered: list[str] = []
        elif pref in available:
            ordered = [pref] + [n for n in available if n != pref]
        else:
            ordered = available

        # Health-aware ordering: prefer providers with better recent success/failure
        # ratio and lower observed latency.
        ranked: list[tuple[tuple[float, float], str]] = []
        for _name in ordered:
            ranked.append((await _llm_health_rank_shared(_name), _name))
        ordered = [name for _, name in sorted(ranked, key=lambda item: item[0])]

        for name in ordered:
            if not await _llm_provider_healthy_for_fallback_shared(name):
                vinfo(
                    logger,
                    f"Skipping unhealthy LLM fallback {name} (recent failures)",
                    session=session,
                    phase=Phase.LLM_SKIP,
                    service="llm",
                    provider=name,
                    reason="unhealthy",
                )
                info = session.add_provider_event(
                    service="llm",
                    provider=name,
                    role="fallback",
                    outcome="skipped",
                    reason="unhealthy",
                )
                llm_attempt_log.append(info)
                continue
            remaining = _remaining_turn_budget_s(session)
            if remaining is not None and remaining < (LLM_MIN_ATTEMPT_BUDGET_SECONDS + VOICE_TURN_INTERNAL_BUFFER_SECONDS):
                vwarn(
                    logger,
                    f"Skipping LLM fallback chain — insufficient turn budget ({remaining:.2f}s left)",
                    session=session,
                    phase=Phase.LLM_SKIP,
                    service="llm",
                    reason="budget_exhausted",
                    budget_s=remaining,
                )
                info = session.add_provider_event(
                    service="llm",
                    provider=name,
                    role="fallback",
                    outcome="skipped",
                    reason="budget_exhausted",
                    detail=f"{remaining:.2f}s left",
                )
                llm_attempt_log.append(info)
                break
            vinfo(
                logger,
                f"Falling back to LLM {name}",
                session=session,
                phase=Phase.LLM_FALLBACK,
                service="llm",
                provider=name,
            )
            result = await try_provider(
                _resolve_llm_provider(providers, name), name, role="fallback"
            )
            if result:
                return result

    # Last resort: graceful shutdown (no hardcoded re-ask loop)
    from app.core.provider_errors import summarize_provider_attempts

    summary = summarize_provider_attempts(llm_attempt_log)
    verror(
        logger,
        "All LLM providers failed — ending call for human review",
        session=session,
        phase=Phase.LLM_HARDCODED,
        service="llm",
        reason="all_providers_failed",
    )
    session.add_error(
        "all_providers_failed",
        summary,
        service="llm",
        attempts=llm_attempt_log,
    )
    return {
        "provider_shutdown": True,
        "provider_service": "llm",
        "provider_detail": "All LLM providers failed",
        "response_text": "",
    }


# ──────────────────────────────────────────────────────────────────────────────
# TTS with Fallback
# ──────────────────────────────────────────────────────────────────────────────


def join_audio_parts(parts: bytes | list[bytes]) -> bytes:
    """Concatenate mulaw segments for REST responses."""
    if isinstance(parts, list):
        return b"".join(p for p in parts if p)
    return parts or b""


async def _synthesize_chunks_pipelined(
    chunks: list[str],
    session: ConversationSession,
    *,
    on_part_ready: AudioPartCallback | None = None,
) -> list[bytes]:
    """Synthesize sentence chunks with overlap: chunk 1 starts while chunk 0 plays."""
    if not chunks or _turn_tts_suppressed(session):
        return []
    parts: list[bytes] = []
    first = await synthesize_with_fallback(chunks[0], session)
    if first and not _turn_tts_suppressed(session):
        parts.append(first)
        await _maybe_deliver_audio_part(
            session, on_part_ready, first, len(chunks) == 1
        )
    if len(chunks) == 1 or _turn_tts_suppressed(session):
        return parts

    tasks = [
        _track_turn_tts_task(session, synthesize_with_fallback(chunk, session))
        for chunk in chunks[1:]
    ]
    try:
        for idx, task in enumerate(tasks):
            if _turn_tts_suppressed(session):
                for pending in tasks[idx:]:
                    if not pending.done():
                        pending.cancel()
                await asyncio.gather(*tasks[idx:], return_exceptions=True)
                break
            audio = await task
            if audio:
                parts.append(audio)
                await _maybe_deliver_audio_part(
                    session, on_part_ready, audio, idx == len(tasks) - 1
                )
    except asyncio.CancelledError:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    return parts


async def _synthesize_ack_followup_parallel(
    ack: str,
    follow_up: str,
    session: ConversationSession,
    *,
    on_part_ready: AudioPartCallback | None = None,
) -> list[bytes]:
    """Run ack and follow-up TTS in parallel; enqueue ack before follow-up."""
    if _turn_tts_suppressed(session):
        return []
    parts: list[bytes] = []
    ack_task = _track_turn_tts_task(session, synthesize_with_fallback(ack, session))
    follow_task = _track_turn_tts_task(
        session, synthesize_with_fallback(follow_up, session)
    )
    try:
        try:
            ack_audio = await ack_task
        except asyncio.CancelledError:
            if not follow_task.done():
                follow_task.cancel()
            await asyncio.gather(follow_task, return_exceptions=True)
            if _turn_tts_suppressed(session):
                return parts
            raise
        if ack_audio and not _turn_tts_suppressed(session):
            parts.append(ack_audio)
            await _maybe_deliver_audio_part(session, on_part_ready, ack_audio, False)
        if _turn_tts_suppressed(session):
            if not follow_task.done():
                follow_task.cancel()
            await asyncio.gather(follow_task, return_exceptions=True)
            return parts
        follow_audio = await follow_task
        if follow_audio and not _turn_tts_suppressed(session):
            parts.append(follow_audio)
            await _maybe_deliver_audio_part(
                session, on_part_ready, follow_audio, True
            )
    except asyncio.CancelledError:
        for task in (ack_task, follow_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(ack_task, follow_task, return_exceptions=True)
        raise
    return parts


async def synthesize_speech_parts(
    ack: str,
    follow_up: str,
    session: ConversationSession,
    *,
    combine: bool = False,
    on_part_ready: AudioPartCallback | None = None,
) -> list[bytes]:
    """
    Synthesize speech with overlap where safe for faster time-to-first-audio.

    - ack + follow_up: parallel TTS (play ack first when both finish)
    - long/combined text: first sentence chunk first, remaining chunks overlap
    - optional ``on_part_ready`` streams each mulaw segment as it completes
    """
    ack = (ack or "").strip()
    follow_up = (follow_up or "").strip()
    parts: list[bytes] = []

    if combine and (ack or follow_up):
        full = " ".join(part for part in (ack, follow_up) if part).strip()
        parts = await _synthesize_chunks_pipelined(
            _split_for_tts(full),
            session,
            on_part_ready=on_part_ready,
        )
        if parts:
            return parts
        # Fall through to split ack/follow_up retry if combined synthesis failed.

    if ack and follow_up:
        parts = await _synthesize_ack_followup_parallel(
            ack,
            follow_up,
            session,
            on_part_ready=on_part_ready,
        )
        if parts:
            return parts
        combined = " ".join(part for part in (ack, follow_up) if part).strip()
        audio = await synthesize_with_fallback(combined, session)
        if audio:
            await _maybe_deliver_audio_part(session, on_part_ready, audio, True)
            return [audio]
        return parts

    if ack:
        ack_chunks = _split_for_tts(ack)
        if len(ack_chunks) > 1:
            return await _synthesize_chunks_pipelined(
                ack_chunks,
                session,
                on_part_ready=on_part_ready,
            )
        audio = await synthesize_with_fallback(ack, session)
        if audio:
            parts.append(audio)
            await _maybe_deliver_audio_part(session, on_part_ready, audio, not follow_up)
    if follow_up:
        follow_chunks = _split_for_tts(follow_up)
        if len(follow_chunks) > 1:
            follow_parts = await _synthesize_chunks_pipelined(
                follow_chunks,
                session,
                on_part_ready=on_part_ready,
            )
            parts.extend(follow_parts)
        else:
            audio = await synthesize_with_fallback(follow_up, session)
            if audio:
                parts.append(audio)
                await _maybe_deliver_audio_part(session, on_part_ready, audio, True)
    return parts


async def _synthesize_provider_with_retries(
    provider,
    provider_name: str,
    text: str,
    *,
    speed: float,
    attempts: int,
    timeout_cap_s: float | None = None,
) -> bytes:
    timeout = tts_timeout_for_text(text)
    if timeout_cap_s is not None:
        timeout = max(TTS_MIN_ATTEMPT_BUDGET_SECONDS, min(timeout, timeout_cap_s))
    text_len = len(text.strip())

    for attempt in range(1, attempts + 1):
        try:
            start = time.perf_counter()
            audio = await asyncio.wait_for(
                provider.synthesize(text, speed=speed),
                timeout=timeout,
            )
            latency_ms = (time.perf_counter() - start) * 1000
            if not audio:
                raise RuntimeError(f"{provider_name} TTS returned empty audio")
            logger.info(
                f"{provider_name} TTS synthesized {text_len} chars "
                f"in {latency_ms:.0f}ms ({len(audio)} bytes)"
            )
            return audio
        except TimeoutError:
            logger.warning(
                f"{provider_name} TTS timeout after {timeout:.1f}s "
                f"(attempt {attempt}/{attempts}, {text_len} chars)"
            )
            if attempt < attempts:
                await asyncio.sleep(0.25)

    raise TimeoutError(
        f"{provider_name} timed out after {attempts} attempt(s) "
        f"at {timeout:.1f}s for {text_len} chars"
    )


async def synthesize_with_fallback(
    text: str,
    session: ConversationSession,
    *,
    budget_remaining_s: float | None = None,
) -> bytes:
    """
    Synthesize speech with automatic TTS fallback.
    Google TTS → Deepgram Aura-2.
    """
    if not text.strip():
        return b""

    t0 = time.monotonic()
    providers = get_call_providers(session)
    speed = max(0.75, min(1.25, getattr(providers, "tts_speed", 1.0) or 1.0))
    remaining = _remaining_turn_budget_s(session)
    if budget_remaining_s is not None:
        remaining = min(remaining if remaining is not None else budget_remaining_s, budget_remaining_s)
    timeout_cap_s = None
    if remaining is not None:
        timeout_cap_s = max(0.0, remaining - VOICE_TURN_INTERNAL_BUFFER_SECONDS)
    if getattr(session, "tts_confirm_priority", False):
        floor = TTS_CONFIRM_MIN_BUDGET_SECONDS
        if timeout_cap_s is None:
            timeout_cap_s = floor
        else:
            timeout_cap_s = max(timeout_cap_s, floor)

    vdebug(
        logger,
        f"Primary TTS synthesize ({providers.tts_name}, {len(text)} chars)",
        session=session,
        phase=Phase.TTS_TRY,
        service="tts",
        provider=providers.tts_name,
        budget_s=timeout_cap_s,
    )

    try:
        primary_attempts = TTS_PRIMARY_ATTEMPTS
        if timeout_cap_s is not None and timeout_cap_s < (TTS_MIN_TIMEOUT_SECONDS + 0.5):
            primary_attempts = 1
        audio = await _synthesize_provider_with_retries(
            providers.tts,
            providers.tts_name,
            text,
            speed=speed,
            attempts=primary_attempts,
            timeout_cap_s=timeout_cap_s,
        )
        session.tts_provider = providers.tts_name
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        session.record_tts_latency(elapsed_ms)
        vinfo(
            logger,
            f"{providers.tts_name} TTS OK ({len(audio)} bytes)",
            session=session,
            phase=Phase.TTS_OK,
            service="tts",
            provider=providers.tts_name,
            latency_ms=elapsed_ms,
            bytes=len(audio),
        )
        return audio
    except TimeoutError as e:
        message = str(e) or f"{providers.tts_name} timed out"
        vwarn(
            logger,
            f"Primary TTS timeout: {message}",
            session=session,
            phase=Phase.TTS_FAIL,
            service="tts",
            provider=providers.tts_name,
            reason="timeout",
        )
        session.add_provider_event(
            service="tts",
            provider=providers.tts_name,
            role="primary",
            outcome="failed",
            exc=e,
        )
    except Exception as e:
        verror(
            logger,
            f"Primary TTS error: {e}",
            session=session,
            phase=Phase.TTS_FAIL,
            service="tts",
            provider=providers.tts_name,
            reason="error",
        )
        session.add_provider_event(
            service="tts",
            provider=providers.tts_name,
            role="primary",
            outcome="failed",
            exc=e,
        )

    # Fallback TTS (when auto-fallback enabled)
    if not providers.auto_fallback_enabled:
        verror(
            logger,
            "All TTS providers failed (auto-fallback disabled)",
            session=session,
            phase=Phase.TTS_FAIL,
            service="tts",
            reason="fallback_disabled",
        )
        return b""

    fallback_name = "unknown"
    try:
        from app.providers.tts.deepgram_tts import DeepgramTTSProvider
        from app.providers.tts.google_tts import GoogleTTSProvider

        api_keys = providers.api_keys
        # Which backup voices are usable and not the primary.
        deepgram_ok = (
            not isinstance(providers.tts, DeepgramTTSProvider)
            and api_keys.configured("deepgram")
        )
        google_ok = (
            not isinstance(providers.tts, GoogleTTSProvider)
            and api_keys.configured("google")
        )
        available = [
            name
            for name, ok in (("deepgram", deepgram_ok), ("google", google_ok))
            if ok
        ]
        pref = (getattr(providers, "tts_fallback_provider", "auto") or "auto").lower()
        if pref == "none":
            fallback = None
        elif pref in available:
            fallback = _resolve_tts_provider(providers, pref)
        elif available:
            fallback = _resolve_tts_provider(providers, available[0])
        else:
            fallback = None

        if fallback:
            fallback_name = fallback.provider_name
            vinfo(
                logger,
                f"Falling back to TTS {fallback.provider_name}",
                session=session,
                phase=Phase.TTS_FALLBACK,
                service="tts",
                provider=fallback.provider_name,
            )
            remaining = _remaining_turn_budget_s(session)
            fallback_timeout_cap = None
            if remaining is not None:
                fallback_timeout_cap = max(0.0, remaining - VOICE_TURN_INTERNAL_BUFFER_SECONDS)
                if fallback_timeout_cap < TTS_MIN_ATTEMPT_BUDGET_SECONDS:
                    vwarn(
                        logger,
                        f"Skipping TTS fallback — insufficient turn budget ({remaining:.2f}s left)",
                        session=session,
                        phase=Phase.TTS_SKIP,
                        service="tts",
                        reason="budget_exhausted",
                        budget_s=remaining,
                    )
                    session.add_provider_event(
                        service="tts",
                        provider=fallback.provider_name,
                        role="fallback",
                        outcome="skipped",
                        reason="budget_exhausted",
                        detail=f"{remaining:.2f}s left",
                    )
                    return b""
            audio = await _synthesize_provider_with_retries(
                fallback,
                fallback.provider_name,
                text,
                speed=speed,
                attempts=TTS_FALLBACK_ATTEMPTS,
                timeout_cap_s=fallback_timeout_cap,
            )
            session.tts_provider = fallback.provider_name
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            session.record_tts_latency(elapsed_ms)
            session.add_provider_event(
                service="tts",
                provider=fallback.provider_name,
                role="fallback",
                outcome="succeeded",
            )
            vinfo(
                logger,
                f"{fallback.provider_name} TTS fallback OK ({len(audio)} bytes)",
                session=session,
                phase=Phase.TTS_OK,
                service="tts",
                provider=fallback.provider_name,
                latency_ms=elapsed_ms,
                bytes=len(audio),
            )
            return audio
    except TimeoutError as e:
        message = str(e) or "Fallback TTS timed out"
        verror(
            logger,
            f"Fallback TTS timed out: {message}",
            session=session,
            phase=Phase.TTS_FAIL,
            service="tts",
            reason="timeout",
        )
        session.add_provider_event(
            service="tts",
            provider=fallback_name,
            role="fallback",
            outcome="failed",
            exc=e,
        )
    except Exception as e:
        verror(
            logger,
            f"Fallback TTS failed: {e}",
            session=session,
            phase=Phase.TTS_FAIL,
            service="tts",
            reason="error",
        )
        session.add_provider_event(
            service="tts",
            provider=fallback_name,
            role="fallback",
            outcome="failed",
            exc=e,
        )

    verror(
        logger,
        "All TTS providers failed",
        session=session,
        phase=Phase.TTS_FAIL,
        service="tts",
        reason="all_providers_failed",
    )
    return b""


# ──────────────────────────────────────────────────────────────────────────────
# End-of-Call Processing
# ──────────────────────────────────────────────────────────────────────────────

# Dedup concurrent finalize attempts (webhook hangup + stream complete)
_finalizing_calls: set[str] = set()
_finalize_calls_lock = asyncio.Lock()


def _has_sufficient_extraction(
    data: dict,
    questions: list[dict] | None = None,
    *,
    refused_states: Iterable[str] | None = None,
    confirmed_fields: Iterable[str] | None = None,
) -> bool:
    """True when in-call data is complete enough to skip end-of-call LLM parse.

    Requires every *required* active question to be answered, every read-back
    field to be confirmed, then at least 80% of active questions answered.
    """
    from app.core.question_flow import (
        confirm_field_for_question,
        is_question_answered_for_def,
        is_question_required,
        ordered_active_questions,
    )

    refused = set(refused_states or [])
    confirmed = set(confirmed_fields or [])
    active = ordered_active_questions(questions or [], data)
    if not active:
        return True

    for q in active:
        if not is_question_required(q):
            continue
        if not is_question_answered_for_def(
            q, data, refused, confirmed_fields=confirmed
        ):
            return False

    for q in active:
        if not q.get("requires_confirmation"):
            continue
        field = confirm_field_for_question(q)
        if field and data.get(field) not in (None, "") and field not in confirmed:
            return False

    answered = sum(
        1
        for q in active
        if is_question_answered_for_def(
            q, data, refused, confirmed_fields=confirmed
        )
    )
    return answered >= max(1, int(len(active) * 0.8))


_FINALIZE_SIDE_EFFECTS_ENQUEUE_LOCK_TTL = 120


async def _acquire_side_effects_enqueue_lock(call_uuid) -> bool:
    """Short cross-worker lock while Celery tasks are being enqueued."""
    from app.core.redis_client import acquire_once, ping

    if not await ping():
        return True
    # Fail open on Redis transport errors so a transient SET failure does not
    # drop email/CRM entirely. Per-channel DB claims still prevent duplicates.
    return await acquire_once(
        f"finalize:sideeffects:enqueue:{call_uuid}",
        _FINALIZE_SIDE_EFFECTS_ENQUEUE_LOCK_TTL,
        fail_closed=False,
    )


async def _release_side_effects_enqueue_lock(call_uuid) -> None:
    from app.core.redis_client import cache_delete

    await cache_delete(f"finalize:sideeffects:enqueue:{call_uuid}")


async def _enqueue_finalize_side_effect_channel(
    db,
    *,
    tenant_id,
    channel: str,
    redis_enqueue_lock: bool,
    enqueue,
) -> None:
    """Queue one side effect and claim only after Celery accepts the task."""
    import uuid as uuid_module

    from app.db.crud import (
        claim_finalize_side_effect_channel,
        is_finalize_side_effect_channel_claimed,
        release_finalize_side_effect_channel,
    )

    tenant_uuid = (
        tenant_id
        if isinstance(tenant_id, uuid_module.UUID)
        else uuid_module.UUID(str(tenant_id))
    )
    if await is_finalize_side_effect_channel_claimed(db, tenant_uuid, channel):
        return

    reserved = False
    if not redis_enqueue_lock:
        reserved = await claim_finalize_side_effect_channel(db, tenant_uuid, channel)
        if not reserved:
            return

    try:
        queued = await enqueue()
    except Exception:
        if reserved:
            await release_finalize_side_effect_channel(db, tenant_uuid, channel)
        raise

    if not queued:
        if reserved:
            await release_finalize_side_effect_channel(db, tenant_uuid, channel)
        return

    if redis_enqueue_lock:
        claimed = await claim_finalize_side_effect_channel(db, tenant_uuid, channel)
        if not claimed:
            logger.warning(
                "Side-effect channel %s queued but claim lost race for tenant %s",
                channel,
                tenant_uuid,
            )


async def _dispatch_finalize_side_effects(
    db,
    *,
    call_uuid,
    tenant_id,
    session: ConversationSession,
    persist_phone: str,
    merged: dict,
    score: int,
    status: str,
    reasons: list,
) -> None:
    """Enqueue post-call email/CRM/latency tasks with per-channel idempotency."""
    from app.core.redis_client import ping

    lock_acquired = await _acquire_side_effects_enqueue_lock(call_uuid)
    if not lock_acquired:
        logger.warning(
            "[%s] Skipping side effects — another worker is enqueueing",
            session.call_id,
        )
        return

    redis_enqueue_lock = await ping()
    try:
        notif = await _resolve_session_notification_settings(session, db)
        email_settings = notification_settings_email_dict(notif)
        call_id = str(call_uuid)

        if notif.email_notifications_enabled:
            await _enqueue_finalize_side_effect_channel(
                db,
                tenant_id=tenant_id,
                channel="email",
                redis_enqueue_lock=redis_enqueue_lock,
                enqueue=lambda: _trigger_email_notification(
                    db,
                    call_id=call_id,
                    phone_number=persist_phone,
                    tenant_data=merged,
                    score=score,
                    status=status,
                    reasons=reasons,
                    transcript=session.get_full_transcript(),
                    duration=session.duration_seconds,
                    providers={
                        "stt": session.stt_provider,
                        "llm": session.llm_provider,
                        "tts": session.tts_provider,
                    },
                    email_settings=email_settings,
                ),
            )

        if crm_notifications_active(notif):
            await _enqueue_finalize_side_effect_channel(
                db,
                tenant_id=tenant_id,
                channel="crm",
                redis_enqueue_lock=redis_enqueue_lock,
                enqueue=lambda: _trigger_crm_webhook(
                    db,
                    crm_url=notif.crm_webhook_url,
                    call_id=call_id,
                    phone_number=persist_phone,
                    status=status,
                    score=score,
                    tenant_data=merged,
                    app_url=settings.app_url,
                    email_settings=email_settings,
                ),
            )

        if notif.email_notifications_enabled:
            from app.services.latency_alerts import queue_latency_alert_if_needed

            async def _enqueue_latency_alert() -> bool:
                try:
                    return queue_latency_alert_if_needed(
                        session,
                        call_id=session.call_id,
                        email_settings=email_settings,
                        raise_on_error=True,
                    )
                except Exception as exc:
                    from app.services.side_effect_errors import (
                        record_side_effect_queue_failure,
                    )

                    await record_side_effect_queue_failure(
                        db, call_id, "latency_queue", str(exc)
                    )
                    return False

            await _enqueue_finalize_side_effect_channel(
                db,
                tenant_id=tenant_id,
                channel="latency",
                redis_enqueue_lock=redis_enqueue_lock,
                enqueue=_enqueue_latency_alert,
            )
    finally:
        if lock_acquired:
            await _release_side_effects_enqueue_lock(call_uuid)


def _build_call_error_log(session: ConversationSession) -> dict | None:
    """Merge session errors and per-turn latency traces for persistence."""
    payload: dict = {}
    if session.errors:
        payload["errors"] = session.errors
    if session.turn_traces:
        payload["turn_traces"] = session.turn_traces
    fallback = session.control_flags.get("questions_config_fallback")
    if fallback not in (None, ""):
        payload["questions_config_fallback"] = str(fallback)
    return payload or None


def _merge_call_error_log(
    existing: dict | None, session_log: dict | None
) -> dict | None:
    """Preserve prior error_log keys (e.g. abandon metadata) when finalizing."""
    if not existing and not session_log:
        return None
    merged = dict(existing or {})
    if session_log:
        merged.update(session_log)
    return merged or None


async def finalize_call(session: ConversationSession, db) -> dict:
    """
    Complete post-call processing:
    1. Extract structured data from transcript
    2. Calculate qualification score
    3. Save to database
    4. Trigger email notification

    Returns:
        Dict with call summary
    """
    logger.info("Finalizing call: %s", session.call_id)

    async with _finalize_calls_lock:
        if session.call_id in _finalizing_calls:
            logger.warning("Finalize already in progress: %s", session.call_id)
            return {"call_id": session.call_id, "status": "skipped"}
        _finalizing_calls.add(session.call_id)

    from app.core.redis_client import clear_finalize_inflight, set_finalize_inflight

    await set_finalize_inflight(session.call_id)
    try:
        return await _finalize_call_impl(session, db)
    finally:
        await clear_finalize_inflight(session.call_id)
        async with _finalize_calls_lock:
            _finalizing_calls.discard(session.call_id)


async def _finalize_call_impl(session: ConversationSession, db) -> dict:
    """Internal finalize implementation."""
    providers = get_call_providers(session)

    # The in-call LLM already extracts every field turn by turn into
    # session.extracted_data, and the merge below lets that in-call data win on
    # conflict. So a full end-of-call LLM re-parse of the transcript is only
    # worth its tokens when the live data is sparse (e.g. a very short or failed
    # call). Skip it when we already captured the core scoring fields — this
    # saves ~2.5-3.5k tokens on every normal call with no loss of data.
    if _has_sufficient_extraction(
        session.extracted_data,
        session.questions,
        refused_states=session.refused_states,
        confirmed_fields=session.confirmed_fields,
    ):
        logger.info(
            "[%s] Skipping end-of-call extraction — in-call data is complete",
            session.call_id,
        )
        extracted = {}
    else:
        extracted = await extract_tenant_data(
            transcript=session.get_full_transcript(),
            llm_provider=providers.llm,
            questions=session.questions,
        )
        extracted = filter_extracted_to_allowed_fields(extracted, session.questions)
        # Account the extraction call's tokens too (only when the provider
        # actually reported usage — a heuristic fallback makes no LLM call).
        ext_usage = getattr(providers.llm, "last_usage", None)
        if ext_usage:
            session.record_llm_usage(ext_usage)

    # Add call/session metadata for scoring and admin review
    extracted["questions_answered"] = session.questions_answered
    extracted["raw_answers"] = dict(session.raw_answers)
    extracted["answered_states"] = list(session.answered_states)
    extracted["refused_states"] = list(session.refused_states)
    extracted["faq_topics"] = list(session.faq_topics)
    extracted["control_flags"] = dict(session.control_flags)

    # Merge post-call extraction with in-call data. In-call values win on
    # conflict so read-back confirmed name/phone/email are not overwritten
    # by a weaker end-of-call LLM re-parse of the transcript.
    merged = {**extracted, **session.extracted_data}
    merged["questions_answered"] = session.questions_answered
    if session.faq_topics:
        merged["faq_topics"] = list(session.faq_topics)
    if session.control_flags:
        merged["control_flags"] = dict(session.control_flags)

    from app.core.data_extractor import coerce_extracted_data

    # The qualifier reads these session-state keys; keep them through coercion
    # (they are used for scoring/reasons only, not persisted to the tenant row).
    meta_keys = ("answered_states", "refused_states", "faq_topics", "control_flags")
    meta = {k: merged[k] for k in meta_keys if k in merged}
    merged = coerce_extracted_data(merged, questions=session.questions)
    merged.update(meta)
    merged["questions_answered"] = session.questions_answered
    merged = normalize_extracted_fields(merged, session.questions)
    merged.update(meta)
    merged["questions_answered"] = session.questions_answered

    for q in normalize_questions(session.questions):
        if not q.get("conditional"):
            continue
        for field in q.get("extract_fields") or []:
            val = merged.get(field)
            if isinstance(val, str) and _is_refusal_text(val):
                merged[field] = None

    # Score using thresholds frozen at call start (same snapshot as questions).
    scoring_settings = {
        "qualified_score_threshold": session.qualified_score_threshold,
        "review_score_threshold": session.review_score_threshold,
    }

    score, status, reasons = calculate_qualification_score(
        merged, scoring_settings, questions=session.questions
    )

    provider_failure = session.control_flags.get("provider_failure")
    call_status = "completed"
    call_completed = session.is_screening_complete()
    if provider_failure:
        status = "review"
        call_status = "failed"
        call_completed = False
        svc = (
            provider_failure.get("service", "unknown")
            if isinstance(provider_failure, dict)
            else "unknown"
        )
        detail = (
            provider_failure.get("detail", "")
            if isinstance(provider_failure, dict)
            else str(provider_failure)
        )
        review_reason = (
            f"Technical issue ({svc}): {detail}. Manual review required."
        )
        reasons = list(reasons or [])
        if review_reason not in reasons:
            reasons.append(review_reason)

    # Update call in DB
    from app.db.crud import (
        create_call,
        create_tenant,
        get_call_by_call_id,
        update_call,
        wait_for_call_by_call_id,
    )

    call = await get_call_by_call_id(db, session.call_id)
    if call is None and not (session.phone_number or "").strip():
        call = await wait_for_call_by_call_id(db, session.call_id, timeout=1.0)
    await sync_session_phone_from_db(session, db, call=call)
    persist_phone = (session.phone_number or "").strip() or "unknown"

    db_persisted = False
    if call is None:
        from app.core.redis_client import is_call_admin_deleted

        if await is_call_admin_deleted(session.call_id):
            logger.warning(
                "[%s] Finalize skipped — call was deleted by admin",
                session.call_id,
            )
            return {
                "status": "abandoned",
                "score": 0,
                "reasons": ["Call deleted by administrator"],
                "db_persisted": False,
            }
        # Production path expects call.initiated to create the row, but if that
        # webhook failed while the media stream still connected we would lose
        # the entire screening otherwise (test console already guards this).
        logger.warning(
            "Call record missing for %s — creating before finalize",
            session.call_id,
        )
        direction = "test" if session.call_id.startswith("test-") else "inbound"
        call = await create_call(
            db,
            call_id=session.call_id,
            phone_number=persist_phone,
            direction=direction,
            status="in_progress",
        )
    if call:
        from app.db.crud import get_tenant_by_call

        if call.status == "abandoned":
            logger.warning(
                "[%s] Finalize skipped — call is already abandoned",
                session.call_id,
            )
            return {
                "call_id": session.call_id,
                "status": "skipped",
                "reason": "abandoned",
                "score": score,
                "reasons": reasons,
                "tenant_data": merged,
                "db_persisted": False,
            }

        # Detect a re-finalize: a call already marked terminal, or one that
        # already has a tenant row, has had its side effects (email + CRM) fired
        # once. Capture this BEFORE update_call flips the status to completed so a
        # second finalize (stream end + hangup timeout + manual end) can't send a
        # duplicate email or fire the CRM webhook twice.
        existing_tenant = await get_tenant_by_call(db, call.id)
        already_finalized = (
            call.status in ("completed", "failed", "abandoned")
            or existing_tenant is not None
        )

        call_fields = {
            "status": call_status,
            "ended_at": datetime.now(UTC),
            "full_transcript": session.get_full_transcript(),
            "questions_answered": session.questions_answered,
            "call_completed": call_completed,
            "stt_provider": session.stt_provider,
            "llm_provider": session.llm_provider,
            "tts_provider": session.tts_provider,
            "duration_seconds": session.duration_seconds,
            "prompt_tokens": session.prompt_tokens,
            "completion_tokens": session.completion_tokens,
            "total_tokens": session.total_tokens,
            "llm_calls": session.llm_calls,
            "avg_llm_ms": session.avg_llm_latency_ms,
            "avg_tts_ms": session.avg_tts_latency_ms,
            "avg_turn_ms": session.avg_turn_latency_ms,
            "max_turn_ms": int(round(session.max_turn_latency_ms)),
            "turn_count": session.turn_latency_samples,
        }
        session_error_log = _build_call_error_log(session)

        from app.db.crud import merge_call_error_log, update_call_if_active

        call_persisted = await update_call_if_active(
            db, session.call_id, commit=False, **call_fields
        )
        if call_persisted and session_error_log:
            await merge_call_error_log(
                db,
                session.call_id,
                session_error_log,
                commit=False,
            )
        if not call_persisted:
            await db.rollback()
            logger.warning(
                "[%s] Finalize persist skipped — call is no longer active",
                session.call_id,
            )
            return {
                "call_id": session.call_id,
                "status": "skipped",
                "reason": "not_active",
                "score": score,
                "reasons": reasons,
                "tenant_data": merged,
                "db_persisted": False,
            }

        db_persisted = True
        # Create tenant record (skip if one already exists for this call)
        if existing_tenant is None:
            from app.models.tenant import Tenant

            all_extract_fields: set[str] = set()
            for q in session.questions:
                all_extract_fields.update(q.get("extract_fields") or [])

            custom_fields: dict[str, Any] = {}
            tenant_payload: dict[str, Any] = {}
            for k, v in merged.items():
                if k in _AUDIT_ONLY_FIELDS or k == "questions_answered":
                    continue
                if hasattr(Tenant, k):
                    tenant_payload[k] = v
                elif k in all_extract_fields:
                    custom_fields[k] = v

            # Persist the per-call question state lists to their dedicated
            # columns. They are skipped by the auto-map above (they sit in
            # _AUDIT_ONLY_FIELDS alongside non-column metadata), but the admin
            # call-detail flow table reads them to show which questions were
            # answered vs. declined — without this they all render as "—".
            tenant_payload["answered_states"] = list(session.answered_states)
            tenant_payload["refused_states"] = list(session.refused_states)

            if custom_fields:
                merged["custom_fields"] = custom_fields
            finalize_notif = await _resolve_session_notification_settings(session, db)
            normalized_data: dict[str, Any] = {
                "screening_questions": session.questions,
                "qualified_score_threshold": session.qualified_score_threshold,
                "review_score_threshold": session.review_score_threshold,
                NOTIFICATION_SETTINGS_PERSIST_KEY: notification_settings_persist_dict(
                    finalize_notif
                ),
            }
            completed = sorted(session.completed_states or set())
            if completed:
                normalized_data["completed_states"] = completed
            if custom_fields:
                normalized_data["custom_fields"] = custom_fields
            tenant_payload["normalized_data"] = normalized_data
            if merged.get("custom_question_scoring"):
                tenant_payload["qualification_details"] = {
                    "custom_questions": merged["custom_question_scoring"],
                }

            persist_overflow: dict[str, Any] = {}
            tenant_payload = sanitize_tenant_payload(
                tenant_payload,
                overflow=persist_overflow,
                log_context=session.call_id,
            )
            if persist_overflow:
                nd = dict(tenant_payload.get("normalized_data") or normalized_data)
                nd["persist_overflow"] = {
                    **nd.get("persist_overflow", {}),
                    **persist_overflow,
                }
                tenant_payload["normalized_data"] = nd

            try:
                await create_tenant(
                    db=db,
                    call_id=call.id,
                    phone_number=persist_phone,
                    qualification_score=score,
                    qualification_status=status,
                    disqualify_reasons=reasons if reasons else None,
                    commit=False,
                    **tenant_payload,
                )
                await db.commit()
            except IntegrityError:
                await db.rollback()
                raced_tenant = await get_tenant_by_call(db, call.id)
                if raced_tenant is not None:
                    logger.info(
                        "[%s] Tenant already created by a concurrent finalize — "
                        "persisting call update only",
                        session.call_id,
                    )
                    already_finalized = True
                    await update_call(db, session.call_id, commit=True, **call_fields)
                else:
                    logger.error(
                        "[%s] Tenant insert failed — persisting call as failed without tenant",
                        session.call_id,
                    )
                    fail_fields = dict(call_fields)
                    fail_fields["status"] = "failed"
                    fail_log = dict(session_error_log or {})
                    errors = list(fail_log.get("errors") or [])
                    errors.append(
                        {
                            "kind": "tenant_persist_failed",
                            "detail": "Could not save applicant profile for this call",
                        }
                    )
                    fail_log["errors"] = errors
                    await update_call_if_active(
                        db, session.call_id, commit=False, **fail_fields
                    )
                    await merge_call_error_log(
                        db,
                        session.call_id,
                        fail_log,
                        commit=False,
                    )
                    await db.commit()
                    db_persisted = True
        else:
            await db.commit()

        tenant_row = existing_tenant
        if tenant_row is None:
            tenant_row = await get_tenant_by_call(db, call.id)

        # Side effects (email + CRM) fire per channel, only when an applicant
        # profile was saved. Per-channel DB claims are set only after Celery
        # accepts each task so a queue failure can be retried on a later finalize.
        if tenant_row is not None:
            await _dispatch_finalize_side_effects(
                db,
                call_uuid=call.id,
                tenant_id=tenant_row.id,
                session=session,
                persist_phone=persist_phone,
                merged=merged,
                score=score,
                status=status,
                reasons=reasons,
            )
        elif not already_finalized:
            logger.warning(
                "[%s] Skipping email/CRM — no tenant profile saved for this call",
                session.call_id,
            )

    return {
        "call_id": session.call_id,
        "score": score,
        "status": status,
        "reasons": reasons,
        "tenant_data": merged,
        "db_persisted": db_persisted if call else False,
    }


async def _resolve_session_notification_settings(
    session: ConversationSession, db: AsyncSession
) -> NotificationSettingsSnapshot:
    """Return notification settings frozen at call start, with a safe fallback."""
    frozen = getattr(session, "notification_settings", None)
    if frozen is not None:
        return frozen
    logger.warning(
        "[%s] Missing frozen notification settings — loading live admin values",
        session.call_id,
    )
    return await load_notification_settings_from_db(db)


async def _trigger_email_notification(
    db, call_id: str, phone_number: str, **kwargs
) -> bool:
    """Fire Celery task for email notification (non-blocking)."""
    try:
        from app.services.email_service import send_screening_email_task

        send_screening_email_task.delay(
            call_id=call_id,
            phone_number=phone_number,
            **kwargs,
        )
        return True
    except Exception as e:
        logger.error("Failed to queue email task: %s", e)
        from app.services.side_effect_errors import record_side_effect_queue_failure

        await record_side_effect_queue_failure(db, call_id, "email_queue", str(e))
        return False


async def _trigger_crm_webhook(db, crm_url: str, **kwargs) -> bool:
    """Fire Celery task for CRM webhook (non-blocking)."""
    try:
        from app.services.email_service import fire_crm_webhook_task

        fire_crm_webhook_task.delay(webhook_url=crm_url, **kwargs)
        return True
    except Exception as e:
        logger.error("Failed to queue CRM webhook task: %s", e)
        call_id = str(kwargs.get("call_id") or "")
        if call_id:
            from app.services.side_effect_errors import record_side_effect_queue_failure

            await record_side_effect_queue_failure(db, call_id, "crm_queue", str(e))
        return False
