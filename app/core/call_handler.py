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
    build_call_provider_bundle,
    load_call_settings_snapshot,
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
    validate_llm_response,
)
from app.core.data_extractor import extract_tenant_data
from app.core.qualifier import calculate_qualification_score
from app.core.question_flow import (
    build_confirm_field_map,
    canonical_language_code,
    first_active_question_state,
    is_question_answered,
    is_question_required,
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
)
from app.core.tenant_sanitize import sanitize_tenant_payload
from config import provider_registry, settings

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

# In-memory health hints to prioritize proven-fast backups.
_llm_health_hints: dict[str, dict[str, float]] = {}


def _is_spanish(session: ConversationSession) -> bool:
    return str(getattr(session, "call_language", "en")).strip().lower().startswith("es")


def _localize(session: ConversationSession, en: str, es: str) -> str:
    return es if _is_spanish(session) else en


def _apply_session_language(session: ConversationSession, lang: str | None) -> None:
    resolved = canonical_language_code(lang)
    if not resolved or resolved == session.call_language:
        return
    session.call_language = resolved
    providers = get_call_providers(session)
    stt_obj = getattr(providers, "stt", None)
    if stt_obj is not None and hasattr(stt_obj, "language"):
        setattr(stt_obj, "language", "es" if resolved == "es" else "en-US")
    tts_obj = getattr(providers, "tts", None)
    if tts_obj is not None and hasattr(tts_obj, "language_code"):
        setattr(tts_obj, "language_code", "es-US" if resolved == "es" else "en-US")
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


def _asks_for_more(text: str) -> bool:
    """True when the agent's own reply is a follow-up question to the caller.

    This lets the LLM veto a premature advance: if it answered with a question
    (e.g. "What's the exact date?") we must stay on the current question, even
    when a deterministic slot already holds a rough value (e.g. move_in_raw).
    """
    t = (text or "").strip()
    return t.endswith("?") or "? " in t


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
    return CallProviderBundle(
        llm=provider_registry.llm,
        stt=provider_registry.stt,
        tts=provider_registry.tts,
        llm_name=provider_registry.llm_name,
        stt_name=provider_registry.stt_name,
        tts_name=provider_registry.tts_name,
        auto_fallback_enabled=provider_registry.auto_fallback_enabled,
        llm_fallback_provider=provider_registry.llm_fallback_provider,
        stt_fallback_provider=provider_registry.stt_fallback_provider,
        tts_fallback_provider=provider_registry.tts_fallback_provider,
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


async def request_stream_stop(call_id: str) -> bool:
    """Signal the audio stream to stop (local event + cross-process Redis).

    Returns True when a stream on this worker was registered.
    """
    stop = _stream_stop_events.get(call_id)
    if stop is not None:
        stop.set()
        return True
    from app.core.redis_client import set_stream_stop_signal

    await set_stream_stop_signal(call_id)
    return False


async def check_stream_stop_signal(call_id: str) -> bool:
    """Poll Redis for a cross-process hangup stop request."""
    from app.core.redis_client import is_stream_stop_signaled

    if await is_stream_stop_signaled(call_id):
        stop = _stream_stop_events.get(call_id)
        if stop is not None:
            stop.set()
        return True
    return False


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
                from app.db.crud import update_call

                try:
                    await update_call(
                        bg_db,
                        call_id,
                        status="failed",
                        ended_at=datetime.now(UTC),
                        error_log={"finalize_error": str(e)},
                    )
                except Exception as db_err:
                    logger.error("Failed to mark call failed in DB: %s", db_err)
    finally:
        # Always drop the in-memory session once finalize finishes (or was skipped).
        if get_session(call_id) is session:
            remove_session(call_id)
            unregister_stream_stop(call_id)


async def finalize_after_stream_timeout(
    call_id: str,
    timeout: float = 8.0,
) -> None:
    """Fallback finalize if hangup stopped the stream but on_complete never ran."""
    await asyncio.sleep(timeout)
    if get_session(call_id) is not None:
        logger.warning(
            "[%s] Stream did not finalize within %.0fs after hangup — forcing",
            call_id,
            timeout,
        )
        await finalize_active_session_background(call_id)
        return
    from app.db.crud import get_call_by_call_id

    from app.db.database import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        call = await get_call_by_call_id(db, call_id)
        if call and call.status == "in_progress":
            logger.warning(
                "[%s] Hangup with no local session — marking abandoned after %.0fs",
                call_id,
                timeout,
            )
            from app.db.crud import update_call

            await update_call(
                db,
                call_id,
                status="abandoned",
                ended_at=datetime.now(UTC),
                error_log={"hangup_no_session": "No worker held session at finalize"},
            )


# ──────────────────────────────────────────────────────────────────────────────
# Session Management
# ──────────────────────────────────────────────────────────────────────────────


async def create_session(
    call_id: str,
    phone_number: str,
    db: AsyncSession,
    property_name: str | None = None,
    questions: list | None = None,
    max_retries: int | None = None,
) -> ConversationSession:
    """Create and register a new call session (idempotent, race-safe)."""
    snapshot = await load_call_settings_snapshot(db)
    providers = build_call_provider_bundle(snapshot)
    _prewarm_fallback_clients(providers)

    async with _sessions_lock:
        existing = _active_sessions.get(call_id)
        if existing:
            logger.info("Session already exists for %s, reusing", call_id)
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
            questions=questions or snapshot.questions,
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
            latency_alert_timeout_rate_pct=float(snapshot.latency_alert_timeout_rate_pct),
        )
        _active_sessions[call_id] = session
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
        vinfo(
            logger,
            "Session removed",
            call_id=call_id,
            phase=Phase.CALL_END,
            state=session.current_state,
        )
    return session


def get_active_sessions() -> list[dict]:
    """Get summary of live-stream sessions for the dashboard (stream still open)."""
    result = []
    for call_id, session in _active_sessions.items():
        if session.stream_ended_at is not None:
            continue
        result.append(
            {
                "call_id": call_id,
                "phone_number": session.phone_number,
                "state": session.current_state,
                "duration": session.duration_seconds,
                "questions_answered": session.questions_answered,
                "started_at": session.started_at.isoformat(),
                "avg_turn_latency_ms": session.avg_turn_latency_ms,
                "last_turn_latency_ms": int(round(session.last_turn_latency_ms)),
                "avg_llm_latency_ms": session.avg_llm_latency_ms,
                "avg_tts_latency_ms": session.avg_tts_latency_ms,
            }
        )
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Greeting
# ──────────────────────────────────────────────────────────────────────────────


async def handle_call_answered(session: ConversationSession) -> list[bytes]:
    """
    Handle the initial call answer — generate and return greeting audio.

    Returns:
        Audio bytes for the greeting message
    """
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
    question = q1["question"] if q1 else "Let's get started with your screening."
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
        if not sentence:
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
            audio = await synthesize_with_fallback(
                sentence,
                session,
                budget_remaining_s=_remaining_turn_budget_s(session),
            )
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
                await on_audio_part(audio, False)
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
        )
        session.current_state = next_state or CallState.WRAP_UP.value
        log_state_transition(
            session.call_id,
            prior_state_val,
            session.current_state,
            "Answered (confirmed)",
            0,
            questions=session.questions,
        )
        session.refresh_progress()
        if screening_complete(
            session.extracted_data, session.skip_states, questions=session.questions
        ):
            session.current_state = CallState.WRAP_UP.value
            session.refresh_progress()
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
        state_value = answered_state
        confirm_map = build_confirm_field_map(session.questions)
        field = confirm_map.get(state_value)
        if not field:
            return None
        if field in session.confirmed_fields:
            return None
        value = session.extracted_data.get(field)
        if not value:
            return None
        if field == "email":
            from app.core.screening_flow import sanitize_stored_email

            validated = sanitize_stored_email(str(value))
            if not validated:
                session.extracted_data.pop(field, None)
                return None
            value = validated
            session.extracted_data[field] = validated
        session.pending_confirmation = {
            "field": field,
            "state": state_value,
            "value": str(value),
            "attempts": 1,
        }
        read_back = readback_prompt_for_state(
            state_value,
            str(value),
            session.questions,
            language_code=session.call_language,
        )
        logger.info("[%s] Read-back confirm %s=%r", session.call_id, field, value)
        return await finish_turn(
            read_back, ack=read_back, follow_up="", require_speech=True
        )

    async def _repair_confirmation_field(
        state_obj: str,
        field: str,
    ) -> tuple[str, list[bytes], bool]:
        """Clear a failed read-back and re-ask the owning question."""
        session.pending_confirmation = None
        session.extracted_data.pop(field, None)
        session.current_state = state_obj
        session.retry_count = 0
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
        follow_up = (question or {}).get("question", "") or (
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
            session.merge_extracted_data(re_changed, raw_text=transcript)
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

        if intent_kind in ("human", "callback", "stop"):
            # Hand off to the call-control handling below.
            session.pending_confirmation = None
        else:
            # Refusal / unclear — re-read the correction, then return to the question.
            pending["attempts"] = pending.get("attempts", 1) + 1
            if pending["attempts"] > confirmation_attempt_limit(session):
                session.pending_confirmation = None
                session.current_state = return_state
                return await _reask_current()
            again = response_text or build_correction_readback(cfields, session=session)
            return await finish_turn(again, ack=again, follow_up="", require_speech=True)

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
            session.pending_confirmation = None
            session.extracted_data.pop(field, None)
            session.current_state = state_obj
            session.retry_count = 0
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
            session.pending_confirmation = None
            question = session.get_current_question()
            q_text = (question or {}).get("question", "")
            if q_text:
                response_text = (
                    _localize(
                        session,
                        "No problem — we can follow up later. ",
                        "No hay problema, podemos dar seguimiento despues. ",
                    )
                    + f"{_localize(session, 'Before you go,', 'Antes de irse,')} {q_text}"
                )
            else:
                response_text = (
                    _localize(
                        session,
                        "No problem — we can follow up later. Is there anything else you need right now?",
                        "No hay problema, podemos dar seguimiento despues. Necesita algo mas ahora?",
                    )
                )
            logger.info(
                "[%s] Callback redirect (soft) — staying on %s",
                session.call_id,
                session.current_state,
            )
            return await finish_turn(response_text, complete=False)

        session.control_flags[control_intent] = True
        session.merge_extracted_data(
            {control_intent: True, "special_notes": transcript},
            raw_text=transcript,
        )
        prior_state_val = session.current_state
        session.current_state = CallState.ENDED.value
        log_state_transition(
            session.call_id,
            prior_state_val,
            CallState.ENDED.value,
            f"Control intent: {control_intent}",
            session.retry_count,
            questions=session.questions,
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

    # Liveness ack right after a silence nudge — re-ask, don't treat as an answer.
    if session.silence_nudge_active:
        session.silence_nudge_active = False
        if is_liveness_acknowledgment(transcript):
            logger.info(
                "[%s] Liveness ack after silence nudge — re-asking %s",
                session.call_id,
                prior_state,
            )
            question = session.get_current_question()
            prompt = (question or {}).get(
                "question"
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
            log_state_transition(
                session.call_id,
                prior_state_val,
                session.current_state,
                f"Relevance limit reached (relevance={relevance})",
                session.retry_count,
                questions=session.questions,
            )
            text = brief_transition(session.questions_answered)
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
        session.merge_extracted_data(extracted, raw_text=transcript)
        for key in ("preferred_language", "language", "call_language"):
            if key in session.extracted_data:
                _apply_session_language(session, session.extracted_data.get(key))
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
            read_back = build_correction_readback(to_confirm)
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
            clarify = response_text or (
                "Just to make sure I have that right — could you confirm that detail?"
            )
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
    )
    # The LLM is the brain: if it understood the caller but deliberately asked a
    # follow-up (question_complete=false AND its reply is a question), honor that
    # and DO NOT advance — even when a deterministic slot already has a rough
    # value. This is what stops "What's the exact date?" from being glued to the
    # next question. question_complete=true always wins and advances.
    llm_wants_more = (
        understood and not question_complete and _asks_for_more(response_text)
    )
    done = (question_complete or deterministic_done) and not llm_wants_more

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
        )
        new_state = next_st or CallState.WRAP_UP.value
        session.current_state = new_state
        log_state_transition(
            session.call_id,
            prior_state_val,
            new_state,
            reason,
            0,
            questions=session.questions,
        )
        session.refresh_progress()
        text = strip_upcoming_question_from_ack(
            session, (ack_text or response_text or "").strip() or human_ack(session)
        )
        ack, follow_up = compose_agent_response(session, text, prior_state)
        spoken = " ".join(part for part in (ack, follow_up) if part).strip()
        if screening_complete(
            session.extracted_data, session.skip_states, questions=session.questions
        ):
            session.current_state = CallState.WRAP_UP.value
            session.refresh_progress()
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
        log_state_transition(
            session.call_id,
            prior_state_val,
            session.current_state,
            reason,
            session.retry_count,
            questions=session.questions,
        )
        text = brief_transition(session.questions_answered)
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
                question.get("question") if question else "How can I help?"
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
            response_text = "Thanks — I just need one more detail to go with that."
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
    prior = session.current_state
    session.current_state = CallState.ENDED.value
    log_state_transition(
        session.call_id,
        prior,
        CallState.ENDED.value,
        f"Provider failure: {service}",
        session.retry_count,
        questions=session.questions,
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
        return farewell, [audio] if audio else [], True

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
        if tts_tasks:
            results = await asyncio.gather(*tts_tasks, return_exceptions=True)
            for item in results:
                if isinstance(item, Exception):
                    logger.warning("Streaming TTS task failed: %s", item)
        return raw
    except TimeoutError:
        raise
    except Exception:
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
                    hint = _llm_health_hints.setdefault(provider_name, {})
                    hint["ok"] = hint.get("ok", 0.0) + 1.0
                    hint["latency_ms"] = latency
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
                hint = _llm_health_hints.setdefault(provider_name, {})
                hint["fail"] = hint.get("fail", 0.0) + 1.0
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
                hint = _llm_health_hints.setdefault(provider_name, {})
                hint["fail"] = hint.get("fail", 0.0) + 1.0
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
        available = [
            name
            for name, has_key in (
                ("groq", bool(settings.groq_api_key)),
                ("openai", bool(settings.openai_api_key)),
                ("openrouter", bool(settings.openrouter_api_key)),
                ("gemini", bool(settings.gemini_api_key)),
            )
            if has_key and name != primary_kind
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
        def _rank(name: str) -> tuple[float, float]:
            hint = _llm_health_hints.get(name, {})
            ok = float(hint.get("ok", 0.0))
            fail = float(hint.get("fail", 0.0))
            success_score = ok / max(1.0, ok + fail)
            latency = float(hint.get("latency_ms", 999999.0))
            return (-success_score, latency)

        ordered = sorted(ordered, key=_rank)

        def _provider_healthy(name: str) -> bool:
            hint = _llm_health_hints.get(name, {})
            ok = float(hint.get("ok", 0.0))
            fail = float(hint.get("fail", 0.0))
            # Skip providers that have failed repeatedly with no recent success
            # (e.g. rate limits) so we don't burn turn budget on doomed attempts.
            if fail >= 3.0 and ok == 0.0:
                return False
            return True

        for name in ordered:
            if not _provider_healthy(name):
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
    if not chunks:
        return []
    parts: list[bytes] = []
    first = await synthesize_with_fallback(chunks[0], session)
    if first:
        parts.append(first)
        if on_part_ready:
            await on_part_ready(first, len(chunks) == 1)
    if len(chunks) == 1:
        return parts

    tasks = [
        asyncio.create_task(synthesize_with_fallback(chunk, session))
        for chunk in chunks[1:]
    ]
    for idx, task in enumerate(tasks):
        audio = await task
        if audio:
            parts.append(audio)
            if on_part_ready:
                await on_part_ready(audio, idx == len(tasks) - 1)
    return parts


async def _synthesize_ack_followup_parallel(
    ack: str,
    follow_up: str,
    session: ConversationSession,
    *,
    on_part_ready: AudioPartCallback | None = None,
) -> list[bytes]:
    """Run ack and follow-up TTS in parallel; enqueue ack before follow-up."""
    parts: list[bytes] = []
    ack_task = asyncio.create_task(synthesize_with_fallback(ack, session))
    follow_task = asyncio.create_task(synthesize_with_fallback(follow_up, session))
    ack_audio = await ack_task
    if ack_audio:
        parts.append(ack_audio)
        if on_part_ready:
            await on_part_ready(ack_audio, False)
    follow_audio = await follow_task
    if follow_audio:
        parts.append(follow_audio)
        if on_part_ready:
            await on_part_ready(follow_audio, True)
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
            if on_part_ready:
                await on_part_ready(audio, True)
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
            if on_part_ready:
                await on_part_ready(audio, not follow_up)
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
                if on_part_ready:
                    await on_part_ready(audio, True)
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

        # Which backup voices are usable and not the primary.
        deepgram_ok = (
            not isinstance(providers.tts, DeepgramTTSProvider)
            and bool(settings.deepgram_api_key)
        )
        google_ok = (
            not isinstance(providers.tts, GoogleTTSProvider)
            and bool(settings.google_application_credentials)
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


def _build_call_error_log(session: ConversationSession) -> dict | None:
    """Merge session errors and per-turn latency traces for persistence."""
    payload: dict = {}
    if session.errors:
        payload["errors"] = session.errors
    if session.turn_traces:
        payload["turn_traces"] = session.turn_traces
    return payload or None


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

    try:
        return await _finalize_call_impl(session, db)
    finally:
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
    merged = coerce_extracted_data(merged)
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
        get_all_settings,
        get_call_by_call_id,
        update_call,
    )

    call = await get_call_by_call_id(db, session.call_id)
    db_persisted = False
    if call is None:
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
            phone_number=session.phone_number or "unknown",
            direction=direction,
            status="in_progress",
        )
    if call:
        from app.db.crud import get_tenant_by_call

        # Detect a re-finalize: a call already marked terminal, or one that
        # already has a tenant row, has had its side effects (email + CRM) fired
        # once. Capture this BEFORE update_call flips the status to completed so a
        # second finalize (stream end + hangup timeout + manual end) can't send a
        # duplicate email or fire the CRM webhook twice.
        existing_tenant = await get_tenant_by_call(db, call.id)
        already_finalized = (
            call.status in ("completed", "failed") or existing_tenant is not None
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
            "error_log": _build_call_error_log(session),
        }

        await update_call(db, session.call_id, commit=False, **call_fields)

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
            normalized_data: dict[str, Any] = {
                "screening_questions": session.questions,
                "qualified_score_threshold": session.qualified_score_threshold,
                "review_score_threshold": session.review_score_threshold,
            }
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
                    phone_number=session.phone_number,
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
                else:
                    logger.warning(
                        "[%s] Tenant insert failed — persisting call without tenant",
                        session.call_id,
                    )
                    db_persisted = False
                await update_call(db, session.call_id, commit=True, **call_fields)
        else:
            await db.commit()

        tenant_row = existing_tenant
        if tenant_row is None:
            tenant_row = await get_tenant_by_call(db, call.id)

        # Side effects (email + CRM) fire exactly once, only when an applicant
        # profile was saved. Beyond the per-worker DB check above, take a short
        # cross-worker Redis lock so two workers finalizing simultaneously can't
        # both send the email / fire the CRM webhook.
        send_side_effects = not already_finalized and tenant_row is not None
        if not already_finalized and tenant_row is None:
            logger.warning(
                "[%s] Skipping email/CRM — no tenant profile saved for this call",
                session.call_id,
            )
        if send_side_effects:
            from app.core.redis_client import acquire_once

            send_side_effects = await acquire_once(
                f"finalize:sideeffects:{call.id}", 86400
            )

        if send_side_effects:
            all_settings = await get_all_settings(db)

            # Send email notification (background task)
            email_notifications = all_settings.get(
                "email_notifications_enabled", "true"
            )
            if email_notifications.lower() == "true":
                _trigger_email_notification(
                    call_id=str(call.id),
                    phone_number=session.phone_number,
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
                )

            # Fire CRM webhook if configured
            crm_url = all_settings.get("crm_webhook_url", "")
            if crm_url:
                _trigger_crm_webhook(
                    crm_url=crm_url,
                    call_id=str(call.id),
                    phone_number=session.phone_number,
                    status=status,
                    score=score,
                    tenant_data=merged,
                    app_url=settings.app_url,
                )

            if all_settings.get("email_notifications_enabled", "true").lower() == "true":
                from app.services.latency_alerts import queue_latency_alert_if_needed

                queue_latency_alert_if_needed(session, call_id=session.call_id)

    return {
        "call_id": session.call_id,
        "score": score,
        "status": status,
        "reasons": reasons,
        "tenant_data": merged,
        "db_persisted": db_persisted if call else False,
    }


def _trigger_email_notification(call_id: str, phone_number: str, **kwargs) -> None:
    """Fire Celery task for email notification (non-blocking)."""
    try:
        from app.services.email_service import send_screening_email_task

        send_screening_email_task.delay(
            call_id=call_id,
            phone_number=phone_number,
            **kwargs,
        )
    except Exception as e:
        logger.error("Failed to queue email task: %s", e)
        _record_side_effect_failure_sync(call_id, "email_queue", str(e))


def _trigger_crm_webhook(crm_url: str, **kwargs) -> None:
    """Fire Celery task for CRM webhook (non-blocking)."""
    try:
        from app.services.email_service import fire_crm_webhook_task

        fire_crm_webhook_task.delay(webhook_url=crm_url, **kwargs)
    except Exception as e:
        logger.error("Failed to queue CRM webhook task: %s", e)
        call_id = str(kwargs.get("call_id") or "")
        if call_id:
            _record_side_effect_failure_sync(call_id, "crm_queue", str(e))


def _record_side_effect_failure_sync(call_id: str, kind: str, detail: str) -> None:
    """Persist queue failures on the call row for admin visibility."""
    import asyncio
    import uuid as uuid_module

    from app.db.crud import get_call_by_call_id, get_call_by_uuid, update_call
    from app.db.database import AsyncSessionLocal

    async def _write() -> None:
        async with AsyncSessionLocal() as db:
            call = None
            try:
                call = await get_call_by_uuid(db, uuid_module.UUID(call_id))
            except (TypeError, ValueError):
                pass
            if call is None:
                call = await get_call_by_call_id(db, call_id)
            if not call:
                return
            log = dict(call.error_log or {})
            log[kind] = detail
            await update_call(db, call.call_id, error_log=log)

    try:
        asyncio.run(_write())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_write())
        finally:
            loop.close()
    except Exception as exc:
        logger.debug("Could not persist side-effect failure for %s: %s", call_id, exc)
