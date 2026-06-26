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
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.call_settings import (
    CallProviderBundle,
    build_call_provider_bundle,
    load_call_settings_snapshot,
)
from app.core.conversation import (
    FIELD_TO_STATE,
    CallState,
    ConversationSession,
    build_correction_readback,
    build_system_prompt,
    compose_agent_response,
    control_flag_for_intent,
    get_fallback_response,
    is_echo_of_agent,
    is_liveness_acknowledgment,
    parse_corrected_fields,
    parse_issue,
    parse_question_complete,
    parse_relevance,
    parse_turn_intent,
    polite_redirect,
    validate_llm_response,
)
from app.core.data_extractor import extract_tenant_data
from app.core.qualifier import calculate_qualification_score
from app.core.screening_flow import (
    BUSINESS_NAME,
    CONFIRM_FIELD_BY_STATE,
    _is_refusal_text,
    brief_transition,
    build_greeting_intro,
    build_readback,
    is_question_answered,
    log_state_transition,
    next_unanswered_state,
    repair_prompt,
    screening_complete,
)
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


def _caller_first_name(session: ConversationSession) -> str:
    """Return the caller's first name if we captured a clean one, else ''."""
    name = session.extracted_data.get("full_name")
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
    first = _caller_first_name(session)
    if first and random.random() < 0.45:
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
        from app.providers.llm.groq_llm import GroqLLMProvider
        from app.providers.llm.openai_llm import OpenAILLMProvider
        from app.providers.llm.openrouter_llm import OpenRouterLLMProvider

        factories = {
            "groq": GroqLLMProvider,
            "openai": OpenAILLMProvider,
            "openrouter": OpenRouterLLMProvider,
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


def request_stream_stop(call_id: str) -> bool:
    """Signal the audio stream to stop. Returns True if a stream was registered."""
    stop = _stream_stop_events.get(call_id)
    if stop is None:
        return False
    stop.set()
    return True


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
        # If another path already finalized (dedup lock), leave cleanup to it.
        if result.get("status") != "skipped" and get_session(call_id) is session:
            remove_session(call_id)
            unregister_stream_stop(call_id)


async def finalize_after_stream_timeout(
    call_id: str,
    timeout: float = 8.0,
) -> None:
    """Fallback finalize if hangup stopped the stream but on_complete never ran."""
    await asyncio.sleep(timeout)
    if get_session(call_id) is None:
        return
    logger.warning(
        "[%s] Stream did not finalize within %.0fs after hangup — forcing",
        call_id,
        timeout,
    )
    await finalize_active_session_background(call_id)


# ──────────────────────────────────────────────────────────────────────────────
# Session Management
# ──────────────────────────────────────────────────────────────────────────────


async def create_session(
    call_id: str,
    phone_number: str,
    db: AsyncSession,
    agent_name: str | None = None,
    property_name: str | None = None,
    questions: list | None = None,
    max_retries: int | None = None,
) -> ConversationSession:
    """Create and register a new call session (idempotent, race-safe)."""
    snapshot = await load_call_settings_snapshot(db)
    providers = build_call_provider_bundle(snapshot)

    async with _sessions_lock:
        existing = _active_sessions.get(call_id)
        if existing:
            logger.info("Session already exists for %s, reusing", call_id)
            return existing

        session = ConversationSession(
            call_id=call_id,
            phone_number=phone_number,
            agent_name=agent_name or snapshot.agent_name,
            property_name=property_name or snapshot.property_name,
            questions=questions or snapshot.questions,
            faqs=snapshot.faqs,
            max_retries=max_retries
            if max_retries is not None
            else snapshot.max_retries,
            stt_provider=providers.stt_name,
            llm_provider=providers.llm_name,
            tts_provider=providers.tts_name,
            call_providers=providers,
            silence_timeout_seconds=snapshot.silence_timeout_seconds,
            max_call_duration_seconds=snapshot.max_call_duration_seconds,
            auto_fallback_enabled=snapshot.auto_fallback_enabled,
            settings_captured_at=snapshot.captured_at,
        )
        _active_sessions[call_id] = session
        event = _session_ready_events.setdefault(call_id, asyncio.Event())
        event.set()
        logger.info(
            f"Session created: {call_id} from {phone_number} "
            f"(LLM={providers.llm_name}, STT={providers.stt_name}, "
            f"TTS={providers.tts_name}, settings@{snapshot.captured_at})"
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


def remove_session(call_id: str) -> ConversationSession | None:
    """Remove and return a completed call session."""
    session = _active_sessions.pop(call_id, None)
    _session_ready_events.pop(call_id, None)
    unregister_stream_stop(call_id)
    if session:
        logger.info("Session removed: %s", call_id)
    return session


def get_active_sessions() -> list[dict]:
    """Get summary of all active sessions for live dashboard."""
    result = []
    for call_id, session in _active_sessions.items():
        result.append(
            {
                "call_id": call_id,
                "phone_number": session.phone_number,
                "state": session.current_state.value,
                "duration": session.duration_seconds,
                "questions_answered": session.questions_answered,
                "started_at": session.started_at.isoformat(),
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
    from app.core.conversation import CallState

    if session.current_state != CallState.IDLE:
        logger.debug(f"Session {session.call_id} already past IDLE — skipping greeting")
        return b""

    session.current_state = CallState.GREETING
    business = (session.property_name or "").strip() or BUSINESS_NAME
    intro = build_greeting_intro(business)
    session.current_state = CallState.Q1_FULL_NAME
    q1 = session.get_current_question()
    question = q1["question"] if q1 else "Can I start with your full name?"
    full_greeting = f"{intro} {question}"

    session.add_transcript("AI", full_greeting)

    return await synthesize_speech_parts(intro, question, session)


async def process_tenant_speech(
    session: ConversationSession,
    transcript: str,
) -> tuple[str, list[bytes], bool]:
    """
    Process one caller utterance with deterministic Ready Rentals flow control.

    The LLM may extract fields, but the next question is selected from the
    canonical screening flow so test console and Telnyx behave the same.
    """
    if not transcript.strip():
        return await handle_silence(session)

    prior_state = session.current_state

    # Technical mic guard ONLY (this is NOT "understanding the caller"): drop the
    # agent's own voice echoing back on speakerphone/hands-free so it never
    # answers itself. Everything about what the caller MEANT is decided by the LLM.
    if is_echo_of_agent(transcript, session):
        logger.info("[%s] Ignoring agent echo: %r", session.call_id, transcript)
        return "", [], False

    logger.info("[%s] Tenant: %s", session.call_id, transcript)
    session.add_transcript("Tenant", transcript)
    session.add_message("user", transcript)
    session.silence_count = 0

    async def finish_turn(
        response_text: str,
        *,
        complete: bool = False,
        ack: str | None = None,
        follow_up: str = "",
        combine_audio: bool = False,
    ) -> tuple[str, list[bytes], bool]:
        spoken = response_text.strip()
        if spoken:
            session.add_transcript("AI", spoken)
            session.add_message("assistant", spoken)
        audio_parts = await synthesize_speech_parts(
            ack if ack is not None else spoken,
            follow_up,
            session,
            combine=combine_audio,
        )
        logger.info(
            "[%s] AI -> %s: %s... (%s audio part(s))",
            session.call_id,
            session.current_state.value,
            spoken[:80],
            len(audio_parts) if audio_parts else 0,
        )
        return spoken, audio_parts, complete

    async def _advance_and_ask(
        answered_state: CallState,
        ack_text: str | None = None,
    ) -> tuple[str, list[bytes], bool]:
        """Mark the current question answered, move on, and ask the next one."""
        session.mark_answered(answered_state)
        session.retry_count = 0
        prior_state_val = answered_state.value
        next_state = next_unanswered_state(
            session.extracted_data,
            session.skip_states,
        )
        session.current_state = (
            CallState(next_state) if next_state else CallState.WRAP_UP
        )
        log_state_transition(
            session.call_id,
            prior_state_val,
            session.current_state.value,
            "Answered (confirmed)",
            0,
        )
        session.refresh_progress()
        if screening_complete(session.extracted_data, session.skip_states):
            session.current_state = CallState.WRAP_UP
            session.refresh_progress()
        text = ack_text or human_ack(session)
        ack, follow_up = compose_agent_response(session, text, answered_state)
        response_text = " ".join(part for part in (ack, follow_up) if part).strip()
        is_complete = session.current_state in (CallState.WRAP_UP, CallState.ENDED)
        return await finish_turn(
            response_text,
            complete=is_complete,
            ack=ack,
            follow_up=follow_up,
        )

    async def _maybe_request_confirmation(
        answered_state: CallState,
    ) -> tuple[str, list[bytes], bool] | None:
        """For high-stakes fields, read the value back instead of advancing.

        Returns a finished turn (the read-back question) when confirmation is
        needed, or None when the field isn't high-stakes / has no value.
        """
        state_value = answered_state.value
        field = CONFIRM_FIELD_BY_STATE.get(state_value)
        if not field:
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
        read_back = build_readback(state_value, str(value))
        logger.info("[%s] Read-back confirm %s=%r", session.call_id, field, value)
        return await finish_turn(read_back, ack=read_back, follow_up="")

    async def _reask_current(
        ack_text: str | None = None,
    ) -> tuple[str, list[bytes], bool]:
        """Acknowledge, then re-ask whatever question we were on (no advance).

        Used after resolving a mid-call correction so the caller is gently
        returned to the question they were answering before the detour.
        """
        question = session.get_current_question()
        follow_up = (question or {}).get("question", "") or (
            "Where were we — go ahead whenever you're ready."
        )
        ack = ack_text or "Perfect, thank you."
        response = " ".join(part for part in (ack, follow_up) if part).strip()
        return await finish_turn(response, ack=ack, follow_up=follow_up)

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
    )
    response_text = str(llm_response_data.get("response_text", "")).strip()
    understood = bool(llm_response_data.get("understood", False))
    intent_kind = parse_turn_intent(llm_response_data)
    faq_topic = llm_response_data.get("faq_topic")
    extracted = llm_response_data.get("extracted_data", {}) or {}
    question_complete = parse_question_complete(llm_response_data)
    relevance = parse_relevance(llm_response_data)
    corrected_fields = parse_corrected_fields(llm_response_data)
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

    # Caller's words were just our own audio echoing back with nothing new.
    if intent_kind == "echo" and not extracted:
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
        return_state = CallState(
            pending.get("return_state", session.current_state.value)
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
            if pending["attempts"] > 3:
                session.pending_confirmation = None
                for c in cfields:
                    session.mark_field_confirmed(c["field"])
                session.current_state = return_state
                return await _reask_current()
            rb = build_correction_readback(cfields)
            logger.info(
                "[%s] Correction re-adjusted: %s", session.call_id, sorted(re_changed)
            )
            return await finish_turn(rb, ack=rb, follow_up="")

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
            # Refusal / unclear — re-read the correction a couple times, then move on.
            pending["attempts"] = pending.get("attempts", 1) + 1
            if pending["attempts"] > 3:
                session.pending_confirmation = None
                session.current_state = return_state
                return await _reask_current()
            again = response_text or build_correction_readback(cfields)
            return await finish_turn(again, ack=again, follow_up="")

    # ── Pending read-back confirmation ──────────────────────────────────────
    # The previous turn read a high-stakes field (name/phone/email) back to the
    # caller; the same LLM call classified their reply (confirm/correct/reject).
    if pending and not pending.get("mode"):
        field = pending["field"]
        state_obj = CallState(pending["state"])

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
            if pending["attempts"] > 3:
                session.pending_confirmation = None
                session.mark_field_confirmed(field)
                return await _advance_and_ask(state_obj)
            read_back = build_readback(pending["state"], str(pending["value"]))
            logger.info(
                "[%s] Caller corrected %s -> %r", session.call_id, field, new_val
            )
            return await finish_turn(read_back, ack=read_back, follow_up="")

        if intent_kind in ("answer", "nothing") and understood:
            logger.info("[%s] Caller confirmed %s", session.call_id, field)
            session.pending_confirmation = None
            session.mark_field_confirmed(field)
            if faq_topic and faq_topic not in session.faq_topics:
                session.faq_topics.append(faq_topic)
            return await _advance_and_ask(state_obj, ack_text=response_text or None)

        if intent_kind == "refusal":
            logger.info("[%s] Caller rejected %s; repairing", session.call_id, field)
            session.extracted_data.pop(field, None)
            session.pending_confirmation = None
            session.current_state = state_obj
            session.retry_count = 0
            prompt = response_text or repair_prompt(pending["state"])
            return await finish_turn(prompt, ack=prompt, follow_up="")

        if intent_kind not in ("human", "callback", "stop"):
            # Unclear — re-ask the read-back a couple of times, then accept.
            pending["attempts"] = pending.get("attempts", 1) + 1
            if pending["attempts"] > 3:
                session.pending_confirmation = None
                session.mark_field_confirmed(field)
                return await _advance_and_ask(state_obj)
            again = response_text or (
                "Sorry, I just want to be sure. "
                + build_readback(pending["state"], str(pending["value"]))
            )
            return await finish_turn(again, ack=again, follow_up="")
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
                    "No problem — we can follow up later. " f"Before you go, {q_text}"
                )
            else:
                response_text = (
                    "No problem — we can follow up later. "
                    "Is there anything else you need right now?"
                )
            logger.info(
                "[%s] Callback redirect (soft) — staying on %s",
                session.call_id,
                session.current_state.value,
            )
            return await finish_turn(response_text, complete=False)

        session.control_flags[control_intent] = True
        session.merge_extracted_data(
            {control_intent: True, "special_notes": transcript},
            raw_text=transcript,
        )
        prior_state_val = session.current_state.value
        session.current_state = CallState.ENDED
        log_state_transition(
            session.call_id,
            prior_state_val,
            CallState.ENDED.value,
            f"Control intent: {control_intent}",
            session.retry_count,
        )
        if control_intent == "human_requested":
            response_text = response_text or (
                "Of course. If you'd like to speak with a team member, I can forward "
                "your information and have someone reach out as soon as possible."
            )
        elif control_intent == "callback_requested":
            response_text = response_text or (
                "No problem. I will note that you asked for a call back later."
            )
        else:
            response_text = response_text or (
                "No problem. We will save what you shared so far."
            )
        return await finish_turn(response_text, complete=True)

    # Liveness ack right after a silence nudge — re-ask, don't treat as an answer.
    if session.silence_nudge_active:
        session.silence_nudge_active = False
        if is_liveness_acknowledgment(transcript):
            logger.info(
                "[%s] Liveness ack after silence nudge — re-asking %s",
                session.call_id,
                prior_state.value,
            )
            question = session.get_current_question()
            prompt = (question or {}).get(
                "question"
            ) or "Please go ahead when you're ready."
            return await finish_turn(prompt, ack="Great, thanks.", follow_up=prompt)

    # ── Relevance gate: off-topic or unintelligible replies ─────────────────
    # The caller said something that isn't an answer to the current question
    # (e.g. "I want to go swimming") or that we could not make sense of. Don't
    # store anything; warmly steer back. Bounded so a stuck caller still moves on.
    if relevance in ("off_topic", "unclear") and intent_kind in ("answer", "nothing"):
        session.retry_count += 1
        if session.retry_count > session.max_retries:
            prior_state_val = session.current_state.value
            session.mark_refused(session.current_state, transcript)
            session.next_state()
            log_state_transition(
                session.call_id,
                prior_state_val,
                session.current_state.value,
                f"Off-topic/unclear limit reached (relevance={relevance})",
                session.retry_count,
            )
            text = brief_transition(session.questions_answered)
            ack, follow_up = compose_agent_response(session, text, prior_state)
            text = " ".join(part for part in (ack, follow_up) if part).strip()
            is_complete = session.current_state in (CallState.WRAP_UP, CallState.ENDED)
            return await finish_turn(
                text, complete=is_complete, ack=ack, follow_up=follow_up
            )
        redirect = response_text or polite_redirect(session, "non_answer")
        logger.info(
            "[%s] Relevance=%s — steering back to %s",
            session.call_id,
            relevance,
            prior_state.value,
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
        for field_name in corrected_fields:
            owner = FIELD_TO_STATE.get(field_name)
            if not owner or owner == prior_state.value:
                continue
            value = session.extracted_data.get(field_name)
            if value in (None, ""):
                continue
            to_confirm.append({"field": field_name, "value": str(value)})
        if to_confirm:
            session.pending_confirmation = {
                "mode": "correction",
                "fields": to_confirm,
                "return_state": prior_state.value,
                "attempts": 1,
            }
            read_back = build_correction_readback(to_confirm)
            logger.info(
                "[%s] Caller corrected earlier field(s): %s",
                session.call_id,
                [c["field"] for c in to_confirm],
            )
            return await finish_turn(read_back, ack=read_back, follow_up="")

    # ── Sanity check: consistency or plausibility issue flagged by the LLM ───
    # The model — which sees the full conversation and all extracted data — can
    # flag a value that contradicts an earlier answer or is simply implausible
    # (e.g. a monthly income that is really an hourly wage). Ask ONE friendly
    # clarifying question per issue per question; never nag about fine answers.
    issue_text = plausibility_issue or consistency_issue
    if issue_text:
        issue_kind = "plausibility" if plausibility_issue else "consistency"
        clarified_key = f"{issue_kind}_clarified_{prior_state.value}"
        if not session.control_flags.get(clarified_key):
            session.control_flags[clarified_key] = True
            clarify = response_text or (
                "Just to make sure I have that right — could you confirm that detail?"
            )
            logger.info(
                "[%s] %s issue on %s: %s",
                session.call_id,
                issue_kind,
                prior_state.value,
                issue_text,
            )
            return await finish_turn(clarify, ack=clarify, follow_up="")

    deterministic_done = is_question_answered(
        prior_state.value,
        session.extracted_data,
        session.skip_states,
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
        prior_state_val = prior_state.value
        next_st = next_unanswered_state(
            session.extracted_data,
            session.skip_states,
        )
        new_state = CallState(next_st) if next_st else CallState.WRAP_UP
        session.current_state = new_state
        log_state_transition(
            session.call_id,
            prior_state_val,
            new_state.value,
            reason,
            0,
        )
        session.refresh_progress()
        text = (ack_text or response_text or "").strip() or human_ack(session)
        ack, follow_up = compose_agent_response(session, text, prior_state)
        spoken = " ".join(part for part in (ack, follow_up) if part).strip()
        if screening_complete(session.extracted_data, session.skip_states):
            session.current_state = CallState.WRAP_UP
            session.refresh_progress()
        is_complete = session.current_state in (CallState.WRAP_UP, CallState.ENDED)
        return await finish_turn(
            spoken,
            complete=is_complete,
            ack=ack,
            follow_up=follow_up,
        )

    if done:
        if question_complete and not deterministic_done:
            logger.info(
                "[%s] LLM marked %s complete (partial slots accepted)",
                session.call_id,
                prior_state.value,
            )
            session.mark_completed(prior_state.value)
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
        if made_progress:
            session.retry_count = 0
        else:
            session.retry_count += 1
        if session.retry_count > session.max_retries:
            logger.info(
                "[%s] Bounded follow-ups exhausted on %s (no progress) — accepting "
                "partial",
                session.call_id,
                prior_state.value,
            )
            session.mark_completed(prior_state.value)
            return await _finish_advance(
                reason="Bounded follow-ups exhausted, accepting partial answer",
            )
        if not response_text:
            response_text = "Thanks — I just need one more detail to go with that."
        return await finish_turn(response_text, ack=response_text, follow_up="")

    # Did not answer the current question (refusal or unclear) — escalate retries.
    session.retry_count += 1
    if session.retry_count > session.max_retries:
        prior_state_val = session.current_state.value
        session.mark_refused(session.current_state, transcript)
        session.next_state()
        log_state_transition(
            session.call_id,
            prior_state_val,
            session.current_state.value,
            "Max retries exceeded, marking as refused",
            session.retry_count,
        )
        response_text = brief_transition(session.questions_answered)
        ack, follow_up = compose_agent_response(session, response_text, prior_state)
        response_text = " ".join(part for part in (ack, follow_up) if part).strip()
        is_complete = session.current_state in (CallState.WRAP_UP, CallState.ENDED)
        return await finish_turn(
            response_text,
            complete=is_complete,
            ack=ack,
            follow_up=follow_up,
        )

    if not response_text:
        kind = "refusal" if intent_kind == "refusal" else "non_answer"
        response_text = polite_redirect(session, kind)
    return await finish_turn(response_text, ack=response_text, follow_up="")


PROGRESSIVE_SILENCE_PROMPTS = (
    "Are you still there?",
    "Take your time, I'm here when you're ready.",
    "I haven't heard anything. We can continue now or reconnect later.",
)


async def handle_silence(session: ConversationSession) -> tuple[str, list[bytes], bool]:
    """Re-prompt on silence with escalating messages, then end the call.

    The caller gets ``len(PROGRESSIVE_SILENCE_PROMPTS)`` nudges; on the next
    silent turn we say goodbye and signal completion so the call hangs up.
    """
    session.silence_count += 1
    logger.warning("[%s] Silence count: %s", session.call_id, session.silence_count)

    if session.silence_count > len(PROGRESSIVE_SILENCE_PROMPTS):
        session.control_flags["ended_after_repeated_silence"] = True
        farewell = (
            "It seems we may have missed each other. We will save what you "
            "shared so far and follow up if needed. Goodbye."
        )
        session.add_transcript("AI", farewell)
        audio = await synthesize_with_fallback(farewell, session)
        return farewell, [audio] if audio else [], True

    prompt = PROGRESSIVE_SILENCE_PROMPTS[session.silence_count - 1]
    session.silence_nudge_active = True
    session.add_transcript("AI", prompt)
    audio = await synthesize_with_fallback(prompt, session)
    return prompt, [audio] if audio else [], False


# ──────────────────────────────────────────────────────────────────────────────
# LLM with Fallback Chain
# ──────────────────────────────────────────────────────────────────────────────


async def get_llm_response_with_fallback(
    session: ConversationSession,
    system_prompt: str,
    messages: list[dict],
    max_retries: int = 2,
    voice_mode: bool = False,
) -> dict:
    """
    Get LLM response with automatic provider fallback chain.
    Groq → OpenAI → OpenRouter → hardcoded fallback.

    Returns:
        Parsed LLM response dict
    """
    import time

    if voice_mode:
        max_retries = 1
        llm_timeout = 5.5
        max_tokens = 200
    else:
        llm_timeout = 5.0
        max_tokens = 300

    async def try_provider(provider, provider_name: str) -> dict | None:
        """Attempt to get a valid response from a provider."""
        for attempt in range(max_retries + 1):
            try:
                start = time.time()
                raw = await asyncio.wait_for(
                    provider.get_response(
                        system_prompt=system_prompt,
                        messages=messages,
                        json_mode=True,
                        temperature=0.3,
                        max_tokens=max_tokens,
                    ),
                    timeout=llm_timeout,
                )
                latency = (time.time() - start) * 1000
                logger.info("%s voice response in %.0fms", provider_name, latency)

                # Parse and validate
                import re

                clean = re.sub(r"```json\s*|\s*```", "", raw).strip()
                data = json.loads(clean)
                is_valid, error = validate_llm_response(data)
                if is_valid:
                    session.llm_provider = provider_name
                    return data
                else:
                    logger.warning(
                        f"Invalid LLM response (attempt {attempt+1}): {error}"
                    )

            except TimeoutError:
                logger.warning("%s timeout after %ss", provider_name, llm_timeout)
                session.add_error("llm_timeout", f"{provider_name} timed out")
            except json.JSONDecodeError as e:
                logger.warning("%s JSON parse failed: %s", provider_name, e)
            except Exception as e:
                logger.error("%s error: %s", provider_name, e)
                session.add_error("llm_error", str(e))
                break  # Don't retry on auth/config errors
        return None

    providers = get_call_providers(session)

    # Try primary provider
    primary = providers.llm
    result = await try_provider(primary, providers.llm_name)
    if result:
        return result

    # Auto-fallback (honours per-call snapshot)
    if providers.auto_fallback_enabled:
        from app.providers.llm.groq_llm import GroqLLMProvider
        from app.providers.llm.openai_llm import OpenAILLMProvider
        from app.providers.llm.openrouter_llm import OpenRouterLLMProvider

        fallback_chain = []
        if not isinstance(primary, GroqLLMProvider) and settings.groq_api_key:
            fallback_chain.append(("groq", "groq"))
        if not isinstance(primary, OpenAILLMProvider) and settings.openai_api_key:
            fallback_chain.append(("openai", "openai"))
        if (
            not isinstance(primary, OpenRouterLLMProvider)
            and settings.openrouter_api_key
        ):
            fallback_chain.append(("openrouter", "openrouter"))

        for factory_name, name in fallback_chain:
            logger.info("Falling back to %s", name)
            result = await try_provider(_get_llm_fallback(factory_name), name)
            if result:
                return result

    # Last resort: hardcoded fallback
    logger.error("All LLM providers failed — using hardcoded fallback response")
    session.add_error("all_providers_failed", "All LLM providers failed")
    return get_fallback_response(session.current_state)


# ──────────────────────────────────────────────────────────────────────────────
# TTS with Fallback
# ──────────────────────────────────────────────────────────────────────────────


def join_audio_parts(parts: bytes | list[bytes]) -> bytes:
    """Concatenate mulaw segments for REST responses."""
    if isinstance(parts, list):
        return b"".join(p for p in parts if p)
    return parts or b""


async def synthesize_speech_parts(
    ack: str,
    follow_up: str,
    session: ConversationSession,
    *,
    combine: bool = False,
) -> list[bytes]:
    """
    Synthesize ack first, then follow-up — enables faster time-to-first-audio.

    When ``combine`` is True (FAQ answers), synthesize as one or more sentence
    chunks so a failed first segment does not drop the FAQ voice entirely.
    """
    ack = (ack or "").strip()
    follow_up = (follow_up or "").strip()
    parts: list[bytes] = []

    if combine and (ack or follow_up):
        full = " ".join(part for part in (ack, follow_up) if part).strip()
        for chunk in _split_for_tts(full):
            audio = await synthesize_with_fallback(chunk, session)
            if audio:
                parts.append(audio)
        if parts:
            return parts
        # Fall through to split ack/follow_up retry if combined synthesis failed.

    if ack:
        audio = await synthesize_with_fallback(ack, session)
        if audio:
            parts.append(audio)
        elif follow_up:
            # First segment failed — try the full utterance as one blob.
            combined = " ".join(part for part in (ack, follow_up) if part).strip()
            audio = await synthesize_with_fallback(combined, session)
            if audio:
                return [audio]
    if follow_up:
        audio = await synthesize_with_fallback(follow_up, session)
        if audio:
            parts.append(audio)
    return parts


async def _synthesize_provider_with_retries(
    provider,
    provider_name: str,
    text: str,
    *,
    speed: float,
    attempts: int,
) -> bytes:
    timeout = tts_timeout_for_text(text)
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


async def synthesize_with_fallback(text: str, session: ConversationSession) -> bytes:
    """
    Synthesize speech with automatic TTS fallback.
    Google TTS → Deepgram Aura-2.
    """
    if not text.strip():
        return b""

    providers = get_call_providers(session)
    speed = max(0.75, min(1.25, getattr(providers, "tts_speed", 1.0) or 1.0))

    try:
        audio = await _synthesize_provider_with_retries(
            providers.tts,
            providers.tts_name,
            text,
            speed=speed,
            attempts=TTS_PRIMARY_ATTEMPTS,
        )
        session.tts_provider = providers.tts_name
        return audio
    except TimeoutError as e:
        message = str(e) or f"{providers.tts_name} timed out"
        logger.warning("Primary TTS timeout: %s", message)
        session.add_error("tts_timeout", message)
    except Exception as e:
        logger.error("Primary TTS error: %s", e)
        session.add_error("tts_error", str(e))

    # Fallback TTS (when auto-fallback enabled)
    if not providers.auto_fallback_enabled:
        logger.error("All TTS providers failed")
        return b""

    try:
        from app.providers.tts.deepgram_tts import DeepgramTTSProvider
        from app.providers.tts.google_tts import GoogleTTSProvider

        if (
            not isinstance(providers.tts, DeepgramTTSProvider)
            and settings.deepgram_api_key
        ):
            fallback = _get_tts_fallback("deepgram")
        elif (
            not isinstance(providers.tts, GoogleTTSProvider)
            and settings.google_application_credentials
        ):
            fallback = _get_tts_fallback("google")
        else:
            fallback = None

        if fallback:
            logger.info("Falling back to %s TTS", fallback.provider_name)
            audio = await _synthesize_provider_with_retries(
                fallback,
                fallback.provider_name,
                text,
                speed=speed,
                attempts=TTS_FALLBACK_ATTEMPTS,
            )
            session.tts_provider = fallback.provider_name
            return audio
    except TimeoutError as e:
        message = str(e) or "Fallback TTS timed out"
        logger.error("Fallback TTS timed out: %s", message)
        session.add_error("tts_timeout", message)
    except Exception as e:
        logger.error("Fallback TTS also failed: %s", e)
        session.add_error("tts_error", str(e))

    logger.error("All TTS providers failed")
    return b""


# ──────────────────────────────────────────────────────────────────────────────
# End-of-Call Processing
# ──────────────────────────────────────────────────────────────────────────────

# Dedup concurrent finalize attempts (webhook hangup + stream complete)
_finalizing_calls: set[str] = set()
_finalize_calls_lock = asyncio.Lock()


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

    # Extract structured data
    extracted = await extract_tenant_data(
        transcript=session.get_full_transcript(),
        llm_provider=providers.llm,
    )

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

    # A caller can decline to give eviction details ("I won't disclose that").
    # That refusal is a valid, accepted answer — but it must not be stored as the
    # eviction circumstances themselves. Keep has_eviction/eviction_raw; clear the
    # circumstances so the record doesn't read as if they explained anything.
    circumstances = merged.get("eviction_circumstances")
    if isinstance(circumstances, str) and _is_refusal_text(circumstances):
        merged["eviction_circumstances"] = None

    # Get scoring settings from DB
    from app.db.crud import get_all_settings

    all_settings = await get_all_settings(db)

    score, status, reasons = calculate_qualification_score(merged, all_settings)

    # Update call in DB
    from app.db.crud import create_call, create_tenant, get_call_by_call_id, update_call

    call = await get_call_by_call_id(db, session.call_id)
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

        await update_call(
            db,
            session.call_id,
            status="completed",
            ended_at=datetime.now(UTC),
            full_transcript=session.get_full_transcript(),
            questions_answered=session.questions_answered,
            call_completed=session.is_screening_complete(),
            stt_provider=session.stt_provider,
            llm_provider=session.llm_provider,
            tts_provider=session.tts_provider,
            duration_seconds=session.duration_seconds,
            error_log={"errors": session.errors} if session.errors else None,
        )

        # Create tenant record (skip if one already exists for this call)
        if existing_tenant is None:
            await create_tenant(
                db=db,
                call_id=call.id,
                phone_number=session.phone_number,
                qualification_score=score,
                qualification_status=status,
                disqualify_reasons=reasons if reasons else None,
                **{
                    k: v
                    for k, v in merged.items()
                    if k != "questions_answered"
                    and k not in _AUDIT_ONLY_FIELDS
                    and hasattr(
                        __import__("app.models.tenant", fromlist=["Tenant"]).Tenant, k
                    )
                },
            )

        # Side effects (email + CRM) fire exactly once, on the first finalize.
        if not already_finalized:
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

    return {
        "call_id": session.call_id,
        "score": score,
        "status": status,
        "reasons": reasons,
        "tenant_data": merged,
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


def _trigger_crm_webhook(crm_url: str, **kwargs) -> None:
    """Fire Celery task for CRM webhook (non-blocking)."""
    try:
        from app.services.email_service import fire_crm_webhook_task

        fire_crm_webhook_task.delay(webhook_url=crm_url, **kwargs)
    except Exception as e:
        logger.error("Failed to queue CRM webhook task: %s", e)
