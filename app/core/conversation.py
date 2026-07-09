"""Conversation state, validation, prompts, and response shaping."""

from __future__ import annotations

import copy
import json
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from enum import Enum
from typing import Any

from app.core.call_settings import NotificationSettingsSnapshot
from app.core.question_flow import (
    active_extract_fields,
    build_field_maps,
    build_question_slot_config,
    confirm_field_for_question,
    ConditionalFlowContext,
    count_active_questions,
    count_answered_questions,
    field_answer_types_from_questions,
    flow_states_in_order,
    inactive_flow_states,
    localized_question_text,
    next_unanswered_state,
    needs_readback_confirmation,
    normalize_questions,
    prompt_fields_catalog,
    readback_prompt_for_state,
    prompt_flow_stats,
    prompt_screening_flow_outline,
    questions_index,
    retry_prompt_for_count,
    screening_complete,
    slot_fill_examples_for_question,
    understanding_guide_for_question,
    validation_hint_for_question,
)
from app.core.screening_flow import (
    BUSINESS_NAME,
    normalize_extracted_fields,
    normalize_faqs,
)

# Meta states only — screening question states are dynamic strings from admin config.
META_STATES = frozenset({"IDLE", "GREETING", "WRAP_UP", "ENDED"})


class CallState(str, Enum):
    """Non-question session phases (question states are dynamic strings)."""

    IDLE = "IDLE"
    GREETING = "GREETING"
    WRAP_UP = "WRAP_UP"
    ENDED = "ENDED"


def is_meta_state(state: str) -> bool:
    return state in META_STATES


def is_question_state(state: str, questions: list[dict] | None = None) -> bool:
    if is_meta_state(state):
        return False
    return state in set(flow_states_in_order(questions))


@dataclass
class TranscriptEntry:
    """A single turn in the conversation transcript."""

    speaker: str
    text: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(UTC).strftime("%H:%M:%S")
    )
    state: str = ""


# Tenant columns call_handler sets via merge_extracted_data (not admin questions).
_RUNTIME_MERGE_FIELDS = frozenset(
    {
        "human_requested",
        "callback_requested",
        "stop_requested",
        "special_notes",
    }
)


def allowed_extract_fields(questions: list[dict] | None) -> frozenset[str]:
    """Admin-configured extract fields plus runtime control columns."""
    return frozenset(active_extract_fields(questions)) | _RUNTIME_MERGE_FIELDS


def filter_extracted_to_allowed_fields(
    data: dict[str, Any] | None,
    questions: list[dict] | None,
) -> dict[str, Any]:
    """Drop keys the LLM invented that are not in the call's question snapshot."""
    allowed = allowed_extract_fields(questions)
    return {k: v for k, v in (data or {}).items() if k in allowed}


@dataclass
class ConversationSession:
    """State for one live call or test-console session."""

    call_id: str
    phone_number: str
    property_name: str = BUSINESS_NAME
    # Admin-editable scripts. Empty string = use the built-in default text.
    greeting_message: str = ""
    closing_message: str = ""
    provider_failure_message: str = ""
    # Caller-selected language for spoken responses. Defaults to English.
    call_language: str = "en"
    # Frozen per-call TTS voices for English / Spanish switching.
    tts_voice_en: str = ""
    tts_voice_es: str = ""
    tts_voice_deepgram_es: str = ""
    tts_voice_google_es: str = ""
    tts_voices_en_by_provider: dict = field(default_factory=dict)
    # Set by the live audio stream when Deepgram streaming STT is active.
    streaming_stt_relay: Any = None
    # Admin-tunable LLM behavior. 0 tokens = use the per-turn tuned default.
    llm_temperature: float = 0.3
    llm_max_tokens: int = 0
    qualified_score_threshold: int = 75
    review_score_threshold: int = 40

    current_state: str = CallState.IDLE.value
    retry_count: int = 0
    max_retries: int = 2
    silence_count: int = 0
    questions_answered: int = 0
    # Set True when a hangup arrives before the audio stream has registered its
    # stop_event. The stream checks this on startup and winds down immediately,
    # closing the race where an early hangup would otherwise finalize a call that
    # is still spinning up its WebSocket.
    pending_hangup: bool = False

    extracted_data: dict = field(default_factory=dict)
    raw_answers: dict = field(default_factory=dict)
    answered_states: list[str] = field(default_factory=list)
    refused_states: list[str] = field(default_factory=list)
    # States the LLM marked complete (including partial accept after bounded follow-ups).
    # Kept separate from refused_states so qualification scoring is unaffected.
    completed_states: set[str] = field(default_factory=set)
    faq_topics: list[str] = field(default_factory=list)
    control_flags: dict = field(default_factory=dict)

    messages: list[dict] = field(default_factory=list)
    transcript: list[TranscriptEntry] = field(default_factory=list)
    questions: list[dict] = field(default_factory=list)
    faqs: list[dict] = field(default_factory=list)

    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    # Set when the live WebSocket stream closes; hides the call from Monitor and
    # freezes duration. The session may remain for Test Console "End & score".
    stream_ended_at: datetime | None = None
    ended_at: datetime | None = None

    stt_provider: str = ""
    llm_provider: str = ""
    tts_provider: str = ""
    # Frozen at call start — never overwritten when fallback succeeds mid-call.
    primary_stt_provider: str = ""
    primary_llm_provider: str = ""
    primary_tts_provider: str = ""

    # Frozen at call start — post-call email/CRM use these, not live admin settings.
    notification_settings: NotificationSettingsSnapshot | None = None
    call_providers: Any = None
    silence_timeout_seconds: int = 12
    max_call_duration_seconds: int = 600
    auto_fallback_enabled: bool = True
    settings_captured_at: str = ""
    # Voice latency (frozen from admin profile at call start).
    voice_latency_profile: str = "balanced"
    llm_streaming_enabled: bool = True
    turn_timeout_seconds: float = 15.0
    llm_timeout_voice_seconds: float = 5.5
    deepgram_endpointing_ms: int = 900
    deepgram_utterance_end_ms: int = 1000
    latency_alert_turn_p95_ms: int = 1200
    latency_alert_turn_p95_crit_ms: int = 1800
    latency_alert_timeout_rate_pct: float = 2.0
    latency_alert_timeout_rate_crit_pct: float = 5.0

    errors: list[dict] = field(default_factory=list)
    interruption_count: int = 0
    # Set when Deepgram hears caller words while the agent still owns the turn
    # (incl. streaming TTS gaps). Keeps finalized transcripts out of the bridge
    # drop path until the worker consumes them.
    caller_speech_pending: bool = False
    # Consecutive empty STT results before graceful provider-failure shutdown.
    stt_empty_strikes: int = 0
    # Real LLM token accounting for this call, summed across every LLM call
    # (per-turn brain + any end-of-call extraction). Populated from each
    # provider's reported usage; persisted to the Call row at finalize.
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    # Real latency accounting (milliseconds), summed across the call so the
    # admin can see WHERE response time is spent: the LLM brain, TTS voice
    # synthesis, and the full turn (transcript-in → audio-ready). "Other" is
    # derived as turn − llm − tts (STT assembly, normalization, queueing).
    llm_latency_ms_total: float = 0.0
    tts_latency_ms_total: float = 0.0
    turn_latency_ms_total: float = 0.0
    turn_latency_samples: int = 0
    max_turn_latency_ms: float = 0.0
    last_turn_latency_ms: float = 0.0
    # Per-turn latency snapshots for post-call analysis (lean Phase 1 telemetry).
    turn_traces: list[dict] = field(default_factory=list)
    # Fields that passed read-back confirmation. These are locked: the LLM/local
    # extractors may not overwrite them on later turns unless the caller issues
    # an explicit correction ("wait, the name is wrong"). Belt-and-suspenders
    # protection against LLM drift on high-stakes data.
    confirmed_fields: set[str] = field(default_factory=set)
    # Pending read-back confirmation for a high-stakes field, e.g.
    # {"field": "contact_phone", "state": "Q2_PHONE", "value": "+1...", "attempts": 1}
    pending_confirmation: dict | None = None
    # Absolute monotonic deadline for the current caller turn (set by audio loop).
    turn_deadline_monotonic: float | None = None
    # Text/audio already streamed to the caller during an in-flight LLM response.
    streamed_speakable_prefix: str = ""
    llm_streamed_during_turn: bool = False
    streamed_audio_sent_during_turn: bool = False
    # Set after a silence nudge ("Are you still there?") — next short ack is
    # liveness, not an answer to the current screening question.
    silence_nudge_active: bool = False
    # Monotonic time when timeout/TTS recovery last played — suppresses stacked nudges.
    last_recovery_at_monotonic: float = 0.0
    # Populated by finish_turn so the audio loop can synth remainders / flush turn_end.
    turn_streaming_finalize: dict | None = None
    # Elevates TTS attempt budget for read-back / confirm lines within a turn.
    tts_confirm_priority: bool = False
    # True while LLM streaming sentences are updating the last AI transcript line.
    streaming_ai_open: bool = False
    # Set on barge-in so in-flight TTS skips enqueueing stale audio.
    turn_interrupted: bool = False
    # Parallel/pipelined TTS tasks for the current caller turn (cleared each turn).
    turn_tts_tasks: list = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self.questions = normalize_questions(self.questions)
        self.faqs = normalize_faqs(self.faqs)

    def get_current_question(self) -> dict | None:
        if is_meta_state(self.current_state):
            return None
        for question in self.questions:
            if (
                question.get("state") == self.current_state
                and question.get("active", True)
            ):
                return question
        return questions_index(self.questions).get(self.current_state)

    def conditional_flow_context(self) -> ConditionalFlowContext:
        return ConditionalFlowContext(
            answered_states=frozenset(self.answered_states or []),
            refused_states=frozenset(self.refused_states or []),
            completed_states=frozenset(self.completed_states or set()),
            questions=tuple(self.questions or []),
        )

    def next_state(self) -> str:
        flow_ctx = self.conditional_flow_context()
        next_st = next_unanswered_state(
            self.extracted_data,
            self.skip_states,
            questions=self.questions,
            confirmed_fields=self.confirmed_fields,
            flow_context=flow_ctx,
            raw_answers=self.raw_answers,
        )
        if next_st:
            self.current_state = next_st
        else:
            self.current_state = CallState.WRAP_UP.value
        self.retry_count = 0
        self.refresh_progress()
        return self.current_state

    @property
    def skip_states(self) -> set[str]:
        """States to skip when walking the question order.

        Combines refused questions, LLM-completed questions, and questions the
        admin switched off (active=False). Inactive states are treated exactly
        like answered ones so the flow never lands on a disabled question.
        """
        return (
            set(self.refused_states)
            | set(self.completed_states)
            | inactive_flow_states(self.questions)
        )

    def refresh_progress(self) -> None:
        flow_ctx = self.conditional_flow_context()
        self.questions_answered = count_answered_questions(
            self.extracted_data,
            self.skip_states,
            questions=self.questions,
            confirmed_fields=self.confirmed_fields,
            flow_context=flow_ctx,
            raw_answers=self.raw_answers,
        )
        for state in flow_states_in_order(self.questions):
            from app.core.question_flow import (
                is_question_answered_for_def,
                should_skip_question,
            )

            if state in self.refused_states:
                continue
            q = questions_index(self.questions).get(state)
            if not q or should_skip_question(
                q, self.extracted_data, flow_context=flow_ctx
            ):
                continue
            if is_question_answered_for_def(
                q,
                self.extracted_data,
                self.skip_states,
                confirmed_fields=self.confirmed_fields,
                flow_context=flow_ctx,
                raw_answers=self.raw_answers,
            ):
                self.mark_answered(state)

    def mark_answered(self, state: str | CallState) -> None:
        state_value = state.value if isinstance(state, CallState) else str(state)
        if is_question_state(state_value, self.questions) and state_value not in self.answered_states:
            self.answered_states.append(state_value)

    def mark_refused(self, state: str | CallState, raw_answer: str = "") -> None:
        state_value = state.value if isinstance(state, CallState) else state
        if state_value not in self.refused_states:
            self.refused_states.append(state_value)
        if raw_answer:
            self.raw_answers[state_value] = raw_answer
        self.refresh_progress()

    def mark_completed(self, state: str | CallState) -> None:
        """Mark a question complete (LLM satisfied or bounded follow-ups exhausted)."""
        state_value = state.value if isinstance(state, CallState) else str(state)
        if is_question_state(state_value, self.questions):
            self.completed_states.add(state_value)
        self.refresh_progress()

    def reopen_question_after_failed_confirmation(
        self,
        state: str | CallState,
        field: str | None = None,
    ) -> None:
        """Re-ask an admin-configured question after read-back repair.

        Clears skip markers so ``next_unanswered_state`` can reach this step again.
        Admin question flags (required, requires_confirmation) remain the source of truth.
        """
        state_value = state.value if isinstance(state, CallState) else str(state)
        self.pending_confirmation = None
        if field:
            self.extracted_data.pop(field, None)
        self.completed_states.discard(state_value)
        if state_value in self.answered_states:
            self.answered_states.remove(state_value)
        self.current_state = state_value
        self.retry_count = 0
        self.refresh_progress()

    def merge_extracted_data(
        self,
        data: dict[str, Any],
        *,
        raw_text: str = "",
        allow_overwrite: frozenset[str] | None = None,
    ) -> None:
        """Merge LLM-extracted values allowed by the admin question snapshot.

        Fields in ``confirmed_fields`` (read-back confirmed) are skipped unless
        their key is listed in *allow_overwrite* (explicit caller correction).
        Works for any admin-defined extract field — not hardcoded to built-ins.
        """
        overwrite = allow_overwrite or frozenset()
        clean: dict[str, Any] = {}
        for key, value in filter_extracted_to_allowed_fields(data, self.questions).items():
            value = _unwrap_confidence(value)
            if value not in (None, ""):
                clean[key] = value
        clean = normalize_extracted_fields(clean, questions=self.questions)
        if not clean:
            return
        to_apply: dict[str, Any] = {}
        for key, value in clean.items():
            if key in self.confirmed_fields and key not in overwrite:
                continue
            to_apply[key] = value
        if not to_apply:
            return
        self.extracted_data.update(to_apply)
        if raw_text and is_question_state(self.current_state, self.questions):
            self.raw_answers[self.current_state] = raw_text
        self.refresh_progress()

    def mark_field_confirmed(self, field_name: str) -> None:
        """Lock a field after it passes read-back confirmation."""
        if field_name:
            self.confirmed_fields.add(field_name)

    def is_screening_complete(self) -> bool:
        return screening_complete(
            self.extracted_data,
            self.skip_states,
            questions=self.questions,
            confirmed_fields=self.confirmed_fields,
            flow_context=self.conditional_flow_context(),
            raw_answers=self.raw_answers,
        )

    def active_question_count(self) -> int:
        return count_active_questions(
            self.extracted_data, self.skip_states, questions=self.questions
        )

    def add_transcript(self, speaker: str, text: str) -> None:
        entry = TranscriptEntry(
            speaker=speaker,
            text=text,
            state=self.current_state,
        )
        self.transcript.append(entry)
        if len(self.transcript) > _MAX_TRANSCRIPT_ENTRIES:
            self.transcript = self.transcript[-_MAX_TRANSCRIPT_ENTRIES:]

    def append_streaming_ai_transcript(self, sentence: str) -> str:
        """Update transcript as LLM streams speakable sentences to TTS."""
        sentence = (sentence or "").strip()
        if not sentence:
            return (self.streamed_speakable_prefix or "").strip()
        if self.streamed_speakable_prefix:
            self.streamed_speakable_prefix += " " + sentence
        else:
            self.streamed_speakable_prefix = sentence
        self.llm_streamed_during_turn = True
        combined = self.streamed_speakable_prefix.strip()
        if (
            self.transcript
            and self.transcript[-1].speaker == "AI"
            and self.streaming_ai_open
        ):
            self.transcript[-1].text = combined
        else:
            self.add_transcript("AI", sentence)
            self.streaming_ai_open = True
        return combined

    def close_streaming_ai_transcript(self, final_text: str | None = None) -> None:
        """Finalize the open streaming AI line (or leave transcript unchanged)."""
        text = (final_text or self.streamed_speakable_prefix or "").strip()
        if (
            text
            and self.transcript
            and self.transcript[-1].speaker == "AI"
            and self.streaming_ai_open
        ):
            self.transcript[-1].text = text
        self.streaming_ai_open = False

    def add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        # Keep only the most recent exchanges. The structured ``extracted_data``
        # in the system prompt already carries long-term memory (every captured
        # field), so the verbatim history only needs the last few turns for
        # natural conversational flow. A smaller window cuts per-turn tokens
        # substantially on longer calls without losing intelligence.
        if len(self.messages) > 12:
            self.messages = self.messages[-12:]

    def reconcile_interrupted_turn(self) -> None:
        """Repair conversation history after a turn is cancelled mid-flight.

        A caller turn appends the ``user`` message and ``Tenant`` transcript line
        up front (before the LLM runs), and streaming may open a partial ``AI``
        transcript line. When barge-in cancels the turn before ``finish_turn``
        commits the matching ``assistant`` message, the history is left
        inconsistent: a dangling user turn and possibly a half-written AI line.

        This reconciles both logs so the next LLM call and the saved transcript
        stay coherent:

        - If the agent actually streamed speech, finalize the open AI line to
          that partial text and pair the user turn with a matching assistant
          message (nothing spoken is fabricated — only what was streamed).
        - If the agent produced no speech at all, drop the empty AI line and the
          unanswered trailing user turn so no orphan user message survives.

        Screening data is untouched on barge-in when the LLM never ran. Turn
        timeout uses :func:`handle_turn_timeout` to restore a pre-turn snapshot
        after reconcile when the turn did not finish cleanly.
        """
        partial = (self.streamed_speakable_prefix or "").strip()

        if self.streaming_ai_open:
            if partial:
                self.close_streaming_ai_transcript(partial)
            else:
                if self.transcript and self.transcript[-1].speaker == "AI":
                    self.transcript.pop()
                self.streaming_ai_open = False

        if self.messages and self.messages[-1].get("role") == "user":
            if partial:
                self.messages.append({"role": "assistant", "content": partial})
            else:
                self.messages.pop()
                if self.transcript and self.transcript[-1].speaker == "Tenant":
                    self.transcript.pop()

        self.streamed_speakable_prefix = ""
        self.streamed_audio_sent_during_turn = False
        self.llm_streamed_during_turn = False
        self.streaming_ai_open = False

    def get_full_transcript(self) -> str:
        text = "\n".join(
            f"[{entry.timestamp}] {entry.speaker}: {entry.text}"
            for entry in self.transcript
        )
        if len(text) > _MAX_TRANSCRIPT_CHARS:
            return text[-_MAX_TRANSCRIPT_CHARS:]
        return text

    def add_error(self, error_type: str, message: str, **extra: Any) -> None:
        entry: dict[str, Any] = {
            "type": error_type,
            "message": message,
            "timestamp": datetime.now(UTC).isoformat(),
            "state": self.current_state,
        }
        if extra:
            entry.update(extra)
        self.errors.append(entry)
        # Bound growth on a pathological call (e.g. a provider failing every turn)
        # so a single long call can't accumulate unbounded error entries.
        if len(self.errors) > 50:
            self.errors = self.errors[-50:]

    def add_provider_event(
        self,
        *,
        service: str,
        provider: str,
        role: str = "primary",
        outcome: str,
        exc: BaseException | None = None,
        reason: str | None = None,
        detail: str | None = None,
        http_status: int | None = None,
        provider_message: str | None = None,
        provider_code: str | None = None,
    ) -> dict[str, Any]:
        """Record a structured provider attempt (failure, success, or skip)."""
        from app.core.provider_errors import (
            build_provider_message,
            classify_provider_failure,
            legacy_error_type,
        )

        info: dict[str, Any] = {
            "service": (service or "").lower(),
            "provider": (provider or "").lower(),
            "role": (role or "primary").lower(),
            "outcome": outcome,
        }
        if outcome == "failed" and exc is not None:
            classified = classify_provider_failure(exc, service=service, provider=provider)
            info.update(classified)
        elif reason:
            info["reason"] = reason
        if http_status is not None:
            info["http_status"] = http_status
        if provider_message:
            info["provider_message"] = provider_message
        if provider_code:
            info["provider_code"] = provider_code
        if detail:
            info["detail"] = detail

        message = build_provider_message(
            service=info["service"],
            provider=info["provider"],
            role=info["role"],
            outcome=outcome,
            reason=info.get("reason"),
            http_status=info.get("http_status"),
            provider_message=info.get("provider_message"),
            detail=detail,
        )

        if outcome == "failed":
            error_type = legacy_error_type(info["service"], info.get("reason"))
        elif outcome == "succeeded":
            error_type = "provider_ok"
        else:
            error_type = "provider_skipped"

        self.add_error(error_type, message, **info)
        return info

    def record_llm_usage(self, usage: dict | None) -> None:
        """Add one LLM call's token usage to the running totals.

        Always counts the call; adds token counts only when the provider
        reported usage (some may omit it). Never raises.
        """
        self.llm_calls += 1
        if not usage:
            return
        try:
            self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
            self.completion_tokens += int(usage.get("completion_tokens") or 0)
        except (TypeError, ValueError, AttributeError):
            pass

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def record_llm_latency(self, ms: float | None) -> None:
        """Record one LLM brain call's latency (ms). Never raises."""
        try:
            ms = float(ms or 0)
        except (TypeError, ValueError):
            return
        if ms <= 0:
            return
        self.llm_latency_ms_total += ms

    def record_tts_latency(self, ms: float | None) -> None:
        """Record one TTS synthesis latency (ms). Never raises."""
        try:
            ms = float(ms or 0)
        except (TypeError, ValueError):
            return
        if ms <= 0:
            return
        self.tts_latency_ms_total += ms

    def record_turn_latency(self, ms: float | None) -> None:
        """Record one full turn's latency (transcript-in → audio-ready, ms)."""
        try:
            ms = float(ms or 0)
        except (TypeError, ValueError):
            return
        if ms <= 0:
            return
        self.turn_latency_ms_total += ms
        self.turn_latency_samples += 1
        self.last_turn_latency_ms = ms
        if ms > self.max_turn_latency_ms:
            self.max_turn_latency_ms = ms

    def record_turn_trace(self, trace: dict) -> None:
        """Append one per-turn latency snapshot (bounded). Never raises."""
        if not trace:
            return
        try:
            self.turn_traces.append(dict(trace))
        except (TypeError, ValueError):
            return
        if len(self.turn_traces) > 200:
            self.turn_traces = self.turn_traces[-200:]

    @staticmethod
    def _avg(total: float, samples: int) -> int:
        return int(round(total / samples)) if samples else 0

    # NOTE: the per-stage averages below are divided by the TURN count, not by
    # the per-stage call count. One turn typically makes 1 LLM call but 2+ TTS
    # calls (a short ack, then the next question, sometimes FAQ chunks). Dividing
    # TTS by its own call count would understate its real per-turn cost and dump
    # the difference into the derived "other" bucket — making the breakdown lie
    # about where time goes. Per-turn denominators keep llm + tts + other = turn.
    @property
    def avg_llm_latency_ms(self) -> int:
        return self._avg(self.llm_latency_ms_total, self.turn_latency_samples)

    @property
    def avg_tts_latency_ms(self) -> int:
        return self._avg(self.tts_latency_ms_total, self.turn_latency_samples)

    @property
    def avg_turn_latency_ms(self) -> int:
        return self._avg(self.turn_latency_ms_total, self.turn_latency_samples)

    @property
    def duration_seconds(self) -> int:
        end = self.ended_at or self.stream_ended_at or datetime.now(UTC)
        return int((end - self.started_at).total_seconds())

    def to_dict(self) -> dict:
        return {
            "call_id": self.call_id,
            "phone_number": self.phone_number,
            "current_state": self.current_state,
            "state": self.current_state,
            "is_screening_complete": self.is_screening_complete(),
            "questions_answered": self.questions_answered,
            "active_question_count": self.active_question_count(),
            "extracted_data": self.extracted_data,
            "raw_answers": self.raw_answers,
            "answered_states": self.answered_states,
            "refused_states": self.refused_states,
            "completed_states": sorted(self.completed_states),
            "confirmed_fields": sorted(self.confirmed_fields),
            "faq_topics": self.faq_topics,
            "control_flags": self.control_flags,
            "transcript": self.get_full_transcript(),
            "duration_seconds": self.duration_seconds,
            "errors": self.errors,
            "stt_provider": self.stt_provider,
            "llm_provider": self.llm_provider,
            "tts_provider": self.tts_provider,
            "settings_captured_at": self.settings_captured_at,
            "auto_fallback_enabled": self.auto_fallback_enabled,
            "avg_turn_latency_ms": self.avg_turn_latency_ms,
            "last_turn_latency_ms": int(round(self.last_turn_latency_ms)),
            "avg_llm_latency_ms": self.avg_llm_latency_ms,
            "avg_tts_latency_ms": self.avg_tts_latency_ms,
        }


_ECHO_PHRASES = frozenset(
    {
        "thank you",
        "thanks",
        "thank you very much",
        "thank you so much",
        "ok thank you",
        "okay thank you",
        "great thank you",
        "thanks so much",
        "ok",
        "okay",
        "sure",
        "alright",
        "got it",
        "perfect",
        "great",
    }
)

_LIVENESS_ACKS_EN = frozenset(
    {
        "yes",
        "yeah",
        "yep",
        "yup",
        "im here",
        "i am here",
        "still here",
        "here",
        "hello",
        "hi",
        "go ahead",
        "continue",
        "ready",
        "ok",
        "okay",
        "yes i am",
        "yes im here",
    }
)

_LIVENESS_ACKS_ES = frozenset(
    {
        "si",
        "sí",
        "aqui",
        "aquí",
        "estoy aqui",
        "estoy aquí",
        "sigo aqui",
        "aqui estoy",
        "aquí estoy",
        "presente",
        "listo",
        "lista",
        "vale",
        "bueno",
        "adelante",
        "hola",
        "continua",
        "continúe",
        "si estoy",
        "sí estoy",
        "sigo",
        "estoy",
        "ok",
        "okay",
    }
)

# Back-compat alias (English set).
_LIVENESS_ACKS = _LIVENESS_ACKS_EN

_LIVENESS_ACK_RE_EN = re.compile(
    r"\b(still here|i'?m here|yes i'?m|here yes|i am)\b",
    re.I,
)
_LIVENESS_ACK_RE_ES = re.compile(
    r"\b("
    r"estoy aqui|estoy aquí|sigo aqui|aqui estoy|aquí estoy|si estoy|sí estoy|"
    r"presente|aqui sigo"
    r")\b",
    re.I,
)


def _normalize_speech(text: str) -> str:
    cleaned = re.sub(r"[^\w\s']", " ", text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _last_ai_text(session: ConversationSession) -> str:
    for entry in reversed(session.transcript):
        if entry.speaker == "AI":
            return entry.text
    return ""


def is_echo_of_agent(transcript: str, session: ConversationSession) -> bool:
    norm = _normalize_speech(transcript)
    if not norm:
        return True
    if norm in _ECHO_PHRASES:
        return True
    if len(norm) <= 14 and "thank" in norm:
        return True
    last_ai = _normalize_speech(_last_ai_text(session))
    if last_ai and norm in last_ai:
        return True
    return False


# Caller asks to repeat the current step — answer from admin config, not LLM echo.
_META_NAV_RE = re.compile(
    r"\b("
    r"what(?:'?s| is) (?:your |the )?next question"
    r"|what (?:do you|did you) (?:want to )?(?:ask|need)"
    r"|(?:repeat|say) (?:that|it) again"
    r"|what (?:was|is) (?:the |your )?question"
    r"|what did you (?:say|ask)"
    r"|can you repeat"
    r"|go ahead(?: with the next question)?"
    r")\b",
    re.I,
)


def is_meta_navigation_request(transcript: str) -> bool:
    """True when the caller wants the current admin question repeated."""
    norm = _normalize_speech(transcript)
    if not norm:
        return False
    return bool(_META_NAV_RE.search(norm))


def navigation_repeat_text(session: ConversationSession) -> str:
    """Admin-respecting text for meta-navigation (repeat current step)."""
    from app.core.question_flow import readback_prompt_for_state

    pending = session.pending_confirmation
    if pending and pending.get("mode") == "correction":
        fields = pending.get("fields") or []
        if fields:
            return build_correction_readback(fields, session=session)
    if pending and not pending.get("mode"):
        rb = readback_prompt_for_state(
            pending["state"],
            str(pending.get("value", "")),
            session.questions,
            language_code=session.call_language,
        )
        if rb:
            return rb
    question_cfg = session.get_current_question()
    if not question_cfg:
        if str(session.call_language).lower().startswith("es"):
            return "Adelante cuando este listo."
        return "Go ahead whenever you're ready."
    return retry_prompt_for_count(
        question_cfg,
        session.retry_count,
        language_code=session.call_language,
    )


def soft_callback_redirect_text(session: ConversationSession) -> str:
    """First callback redirect: note callback, then repeat the current step.

    When read-back confirmation is in progress, repeats the read-back instead of
    dropping ``pending_confirmation`` and re-asking the original question.
    """
    is_es = str(session.call_language).lower().startswith("es")
    pause = (
        "No hay problema, podemos dar seguimiento despues. "
        if is_es
        else "No problem — we can follow up later. "
    )
    if session.pending_confirmation:
        follow_up = navigation_repeat_text(session)
        if follow_up:
            session.control_flags["callback_soft_noted"] = True
            return f"{pause}{follow_up}"

    question_cfg = session.get_current_question()
    q_text = localized_question_text(
        question_cfg,
        language_code=session.call_language,
        key="question",
    )
    if q_text:
        before = "Antes de irse," if is_es else "Before you go,"
        return f"{pause}{before} {q_text}"
    return pause + (
        "Necesita algo mas ahora?"
        if is_es
        else "Is there anything else you need right now?"
    )


def needs_extended_turn_budget(session: ConversationSession) -> bool:
    """Read-back confirm in progress needs more wall time than a simple extract+ack."""
    return session.pending_confirmation is not None


def turn_timeout_recovery_text(
    session: ConversationSession, transcript: str = ""
) -> str:
    """Admin retry prompt after orchestration timeout — never blame the caller for slow AI."""
    heard = (transcript or "").strip()
    prompt = navigation_repeat_text(session)
    pause = (
        "Perdon por la pausa."
        if str(session.call_language).lower().startswith("es")
        else "Sorry for the pause."
    )
    if session.streamed_speakable_prefix or session.streamed_audio_sent_during_turn:
        return f"{pause} {prompt}" if heard else prompt
    if heard:
        return f"{pause} {prompt}"
    return polite_redirect(session, "unclear")


def try_open_readback_confirmation(
    session: ConversationSession,
    state_value: str,
    *,
    require_caller_spoke: bool = False,
    transcript: str = "",
) -> str | None:
    """Open ``pending_confirmation`` when admin read-back is due for ``state_value``.

    Single entry for normal question advance and turn-timeout recovery.
    Returns the read-back prompt to speak, or ``None`` when confirmation is
    not needed or already open.
    """
    if session.pending_confirmation:
        return None
    if require_caller_spoke and not (transcript or "").strip():
        return None
    if not needs_readback_confirmation(
        str(state_value),
        session.extracted_data,
        session.questions,
        session.confirmed_fields,
    ):
        return None

    q = questions_index(session.questions).get(str(state_value))
    field = confirm_field_for_question(q) if q else None
    if not field:
        return None

    value = session.extracted_data.get(field)
    if value in (None, ""):
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
        "state": str(state_value),
        "value": str(value),
        "attempts": 1,
    }
    return readback_prompt_for_state(
        str(state_value),
        str(value),
        session.questions,
        language_code=session.call_language,
    )


@dataclass(frozen=True)
class TurnStateSnapshot:
    """Screening state captured before one caller turn begins."""

    current_state: str
    extracted_data: dict[str, Any]
    raw_answers: dict[str, Any]
    pending_confirmation: dict[str, Any] | None
    confirmed_fields: frozenset[str]
    retry_count: int
    answered_states: tuple[str, ...]
    refused_states: tuple[str, ...]
    completed_states: frozenset[str]
    questions_answered: int


def capture_turn_snapshot(session: ConversationSession) -> TurnStateSnapshot:
    """Snapshot screening state before ``process_tenant_speech`` mutates it."""
    pending = session.pending_confirmation
    return TurnStateSnapshot(
        current_state=str(session.current_state),
        extracted_data=copy.deepcopy(session.extracted_data),
        raw_answers=copy.deepcopy(session.raw_answers),
        pending_confirmation=copy.deepcopy(pending) if pending else None,
        confirmed_fields=frozenset(session.confirmed_fields),
        retry_count=int(session.retry_count),
        answered_states=tuple(session.answered_states or []),
        refused_states=tuple(session.refused_states or []),
        completed_states=frozenset(session.completed_states or set()),
        questions_answered=int(session.questions_answered),
    )


def restore_turn_snapshot(
    session: ConversationSession, snapshot: TurnStateSnapshot
) -> None:
    """Restore screening fields from a pre-turn snapshot."""
    session.current_state = snapshot.current_state
    session.extracted_data = copy.deepcopy(snapshot.extracted_data)
    session.raw_answers = copy.deepcopy(snapshot.raw_answers)
    session.pending_confirmation = (
        copy.deepcopy(snapshot.pending_confirmation)
        if snapshot.pending_confirmation
        else None
    )
    session.confirmed_fields = set(snapshot.confirmed_fields)
    session.retry_count = snapshot.retry_count
    session.answered_states = list(snapshot.answered_states)
    session.refused_states = list(snapshot.refused_states)
    session.completed_states = set(snapshot.completed_states)
    session.questions_answered = snapshot.questions_answered
    session.refresh_progress()


def handle_turn_timeout(
    session: ConversationSession,
    transcript: str,
    snapshot: TurnStateSnapshot,
) -> str:
    """Repair history and roll back partial turn work after a turn budget expires."""
    session.reconcile_interrupted_turn()
    restore_turn_snapshot(session, snapshot)
    return plan_turn_timeout_recovery(
        session,
        transcript,
        answered_state=snapshot.current_state,
    )


def handle_hangup_cancelled_turn(
    session: ConversationSession,
    snapshot: TurnStateSnapshot,
) -> None:
    """Undo partial turn work when the caller hung up before the turn finished."""
    session.reconcile_interrupted_turn()
    restore_turn_snapshot(session, snapshot)


def plan_turn_timeout_recovery(
    session: ConversationSession,
    transcript: str,
    *,
    answered_state: str | None = None,
) -> str:
    """Pick recovery speech after timeout; resume read-back when data was already captured."""
    prefix = (session.streamed_speakable_prefix or "").strip()
    if prefix:
        session.close_streaming_ai_transcript(prefix)

    recovery_state = str(answered_state or session.current_state)
    read_back = try_open_readback_confirmation(
        session,
        recovery_state,
        require_caller_spoke=True,
        transcript=transcript,
    )
    if read_back:
        return read_back

    return turn_timeout_recovery_text(session, transcript)


TURN_CONFIRM_BUDGET_EXTENSION_SECONDS = 5.0


def turn_budget_seconds(session: ConversationSession) -> float:
    """Per-turn outer budget; extended for admin confirm/read-back steps."""
    base = float(getattr(session, "turn_timeout_seconds", None) or 15.0)
    if needs_extended_turn_budget(session):
        return base + TURN_CONFIRM_BUDGET_EXTENSION_SECONDS
    return base


def unsynthesized_speech_remainder(
    response_text: str, session: ConversationSession
) -> str:
    """Text the caller should hear but was not yet synthesized to audio."""
    fin = getattr(session, "turn_streaming_finalize", None) or {}
    intended = (fin.get("intended") or response_text or fin.get("display") or "").strip()
    prefix = (fin.get("streamed_prefix") or "").strip()
    if not intended:
        return ""
    if prefix and intended.startswith(prefix):
        return intended[len(prefix) :].strip()
    return intended.strip()


def streamed_audio_complete(session: ConversationSession, intended: str) -> bool:
    """True when streaming TTS already covered everything we would synthesize."""
    if not getattr(session, "streamed_audio_sent_during_turn", False):
        return False
    prefix = (getattr(session, "streamed_speakable_prefix", "") or "").strip()
    intended = (intended or "").strip()
    if not prefix or not intended:
        return False
    if intended.startswith(prefix):
        return not intended[len(prefix) :].strip()
    return False


def mark_recovery_played(session: ConversationSession) -> None:
    session.last_recovery_at_monotonic = time.monotonic()


def should_suppress_silence_nudge(session: ConversationSession, *, now: float | None = None) -> bool:
    """Avoid stacking a silence prompt right after timeout/TTS recovery."""
    last = float(getattr(session, "last_recovery_at_monotonic", 0.0) or 0.0)
    if last <= 0:
        return False
    ts = now if now is not None else time.monotonic()
    return (ts - last) < 4.0


def is_liveness_acknowledgment(
    transcript: str,
    *,
    language_code: str = "en",
) -> bool:
    """Short reply to a silence nudge — not an answer to the screening question."""
    norm = _normalize_speech(transcript)
    if not norm:
        return False
    is_es = str(language_code or "en").strip().lower().startswith("es")
    if is_es:
        if norm in _LIVENESS_ACKS_ES or norm in _LIVENESS_ACKS_EN:
            return True
        if len(norm) <= 32 and _LIVENESS_ACK_RE_ES.search(norm):
            return True
        return False
    if norm in _LIVENESS_ACKS_EN:
        return True
    if len(norm) <= 24 and _LIVENESS_ACK_RE_EN.search(norm):
        return True
    return False


def _unwrap_confidence(value: Any) -> Any:
    """Flatten the LLM's ``{"value": x, "confidence": y}`` envelope to ``x``.

    The conversational LLM wraps every extracted field in a confidence object;
    storing that verbatim leaks ``{'value': ...}`` into read-backs and the
    inspector. We keep only the value so downstream code sees plain scalars.
    """
    if (
        isinstance(value, dict)
        and "value" in value
        and set(value.keys()) <= {"value", "confidence"}
    ):
        return value["value"]
    return value


_REDIRECT_LEADINS = (
    "Sorry, I didn't quite catch that.",
    "I want to make sure I get this right.",
    "Let's try that once more.",
)


def polite_redirect(session: ConversationSession, kind: str) -> str:
    """Re-ask the current question with an escalating, non-repetitive prompt.

    Real agents never read the exact same sentence twice — they rephrase and
    offer more guidance each attempt, then make it clear it's okay.
    """
    is_es = str(session.call_language).lower().startswith("es")
    question_cfg = session.get_current_question()
    if not question_cfg:
        if is_es:
            return "No hay problema. Continue cuando este listo."
        return "No problem. Please continue when you are ready."

    retry_count = session.retry_count
    prompt = retry_prompt_for_count(
        question_cfg,
        retry_count,
        language_code=session.call_language,
    )
    if not prompt:
        prompt = localized_question_text(
            question_cfg, language_code=session.call_language, key="question"
        )

    if kind == "refusal":
        if is_es:
            return (
                "Esta bien. Hacemos las mismas preguntas a todos y es confidencial. "
                f"{prompt}"
            )
        return (
            "That's completely okay — we ask everyone the same questions and it "
            f"stays confidential. {prompt}"
        )

    if kind == "unclear":
        # Likely garbled audio / STT trouble, not evasion — apologize for our
        # end and invite a repeat rather than implying they went off topic.
        leadins = (
            "Sorry, I didn't quite catch that.",
            "Apologies, the line broke up a little.",
            "I want to be sure I get this right.",
        )
        if is_es:
            leadins = (
                "Perdon, no lo alcance a escuchar bien.",
                "Disculpe, la linea se corto un poco.",
                "Quiero asegurarme de anotarlo bien.",
            )
        lead = leadins[min(retry_count, len(leadins) - 1)]
        return f"{lead} {prompt}"

    lead = _REDIRECT_LEADINS[min(retry_count, len(_REDIRECT_LEADINS) - 1)]
    if is_es:
        leadins_es = (
            "Perdon, no lo entendi bien.",
            "Quiero asegurarme de que quede correcto.",
            "Intentemos una vez mas.",
        )
        lead = leadins_es[min(retry_count, len(leadins_es) - 1)]
    return f"{lead} {prompt}"


def reset_turn_streaming(session: ConversationSession, *, full: bool = False) -> None:
    """Clear per-turn LLM streaming transcript state.

    By default keeps ``streamed_speakable_prefix`` and ``streamed_audio_sent_during_turn``
    so ``finish_turn`` can skip re-synthesizing audio the caller already heard.
    Pass ``full=True`` to wipe everything (read-backs, timeout recovery).
    """
    session.llm_streamed_during_turn = False
    session.streaming_ai_open = False
    if full:
        session.streamed_speakable_prefix = ""
        session.streamed_audio_sent_during_turn = False


def confirmation_attempt_limit(session: ConversationSession) -> int:
    """How many read-back re-prompts before giving up (admin max_retries aware)."""
    return max(int(session.max_retries or 1) + 2, 3)


def strip_upcoming_question_from_ack(
    session: ConversationSession, ack_text: str
) -> str:
    """Remove the next question text from an LLM ack when compose will add it."""
    ack = (ack_text or "").strip()
    question = session.get_current_question()
    if not ack or not question:
        return ack
    prompt = localized_question_text(
        question,
        language_code=session.call_language,
        key="question",
    )
    if not prompt:
        return ack
    lower_ack = ack.lower()
    lower_prompt = prompt.lower()
    if lower_prompt not in lower_ack:
        return ack
    # Collapse repeated prompt echoes the LLM may emit in one string:
    # "Question? Question?" -> "Question?"
    repeated = re.compile(rf"(?:{re.escape(prompt)}\s*){{2,}}", re.IGNORECASE)
    ack = repeated.sub(f"{prompt} ", ack).strip()
    lower_ack = ack.lower()
    idx = lower_ack.rfind(lower_prompt)
    if idx < 0:
        return ack
    trimmed = (ack[:idx] + ack[idx + len(prompt) :]).strip(" ,.-")
    return trimmed or ack


def normalize_speech_parts(ack: str, follow_up: str) -> tuple[str, str]:
    """Drop redundant follow-up when ack already contains the same prompt."""
    ack = (ack or "").strip()
    follow_up = (follow_up or "").strip()
    if not ack or not follow_up:
        return ack, follow_up
    if follow_up.lower() in ack.lower():
        return ack, ""
    if ack.lower() in follow_up.lower():
        return "", follow_up
    return ack, follow_up


def dedupe_repeated_speech(text: str) -> str:
    """Collapse adjacent duplicate sentences/questions in spoken output."""
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if not cleaned:
        return ""
    parts = [
        p.strip()
        for p in re.findall(r"[^?.!]+[?.!]?", cleaned)
        if p and p.strip()
    ]
    if not parts:
        return cleaned
    out: list[str] = []
    for part in parts:
        if out and part.casefold() == out[-1].casefold():
            continue
        out.append(part)
    return " ".join(out).strip()


def dedupe_repeated_block(text: str) -> str:
    """Collapse an exact doubled block (e.g. 'ABC ABC' -> 'ABC')."""
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if len(cleaned) < 2:
        return cleaned
    if len(cleaned) % 2 == 0:
        mid = len(cleaned) // 2
        first, second = cleaned[:mid].strip(), cleaned[mid:].strip()
        if first and first == second:
            return first
    words = cleaned.split()
    if len(words) >= 2 and len(words) % 2 == 0:
        half = len(words) // 2
        first = " ".join(words[:half])
        second = " ".join(words[half:])
        if first == second:
            return first
    return cleaned


def compose_spoken_display(
    *,
    spoken: str = "",
    ack: str = "",
    follow_up: str = "",
    response_text: str = "",
) -> str:
    """Build one transcript line without doubling response_text + ack + follow_up."""
    ack = (ack or "").strip()
    follow_up = (follow_up or "").strip()
    spoken = (spoken or "").strip()
    if ack or follow_up:
        display = " ".join(part for part in (ack, follow_up) if part).strip()
    else:
        display = spoken or (response_text or "").strip()
    return dedupe_repeated_block(dedupe_repeated_speech(display))


def compose_agent_response(
    session: ConversationSession,
    acknowledgment: str,
    prior_state: str,
) -> tuple[str, str]:
    """Return spoken acknowledgment and the next deterministic prompt."""
    ack = (acknowledgment or "").strip()
    current = session.current_state

    if is_question_state(current, session.questions):
        question = session.get_current_question()
        if not question:
            return ack, ""

        if current == prior_state and session.retry_count > 0:
            prompt = retry_prompt_for_count(
                question,
                session.retry_count,
                language_code=session.call_language,
            )
        else:
            prompt = localized_question_text(
                question,
                language_code=session.call_language,
                key="question",
            )

        if not ack:
            return prompt, ""
        if prompt.lower() not in ack.lower():
            return normalize_speech_parts(ack, prompt)
        return ack, ""

    if current == CallState.WRAP_UP.value:
        business = (session.property_name or "").strip() or BUSINESS_NAME
        if session.closing_message:
            closing = session.closing_message.replace("{property_name}", business)
        else:
            if str(session.call_language).lower().startswith("es"):
                closing = (
                    "Gracias. Un especialista de alquiler revisara su informacion "
                    "y se comunicara pronto."
                )
            else:
                closing = (
                    "Thank you. A leasing specialist will review your information "
                    "and follow up soon."
                )
        return (ack, closing) if ack else (closing, "")

    if current == CallState.ENDED.value and not ack:
        business = (session.property_name or "").strip() or BUSINESS_NAME
        if str(session.call_language).lower().startswith("es"):
            return f"Gracias por llamar a {business}. Adios.", ""
        return f"Thank you for calling {business}. Goodbye.", ""

    return ack, ""


def field_maps_for_session(session: ConversationSession) -> tuple[dict[str, str], dict[str, str]]:
    return build_field_maps(session.questions)


def _short_label(field: str, session: ConversationSession | None = None) -> str:
    """A concise spoken label for a field (used in correction read-backs)."""
    if session:
        _, labels = field_maps_for_session(session)
        raw = labels.get(field, field.replace("_", " "))
    else:
        raw = field.replace("_", " ")
    return raw.split("(")[0].strip()


def build_correction_readback(
    fields: list[dict[str, str]], session: ConversationSession | None = None
) -> str:
    """Combined read-back confirming one or more EARLIER fields the caller just
    corrected. ``fields`` is a list of {"field", "value"} dicts."""
    is_es = bool(session and str(session.call_language).lower().startswith("es"))
    parts: list[str] = []
    for item in fields:
        label = _short_label(item.get("field", ""), session)
        value = str(item.get("value", "")).strip()
        if not value:
            continue
        if is_es:
            parts.append(f"su {label} a {value}")
        else:
            parts.append(f"your {label} to {value}")
    if not parts:
        if is_es:
            return "Permita confirmar los datos actualizados. Es correcto?"
        return "Let me make sure I have your updated details right. Is that correct?"
    if len(parts) == 1:
        joined = parts[0]
    else:
        joined = ", ".join(parts[:-1]) + ", and " + parts[-1]
    if is_es:
        return f"Confirmo antes de continuar: actualice {joined}. Es correcto?"
    return f"Quick check before we move on — I've updated {joined}. Did I get that right?"


def _slot_value_present(
    data: dict[str, Any],
    field: str,
    *,
    answer_type: str | None = None,
) -> bool:
    val = data.get(field)
    if val is None or val == "" or val == []:
        return False
    if answer_type == "yes_no" or field.startswith(("has_", "is_")):
        return val is True or val is False
    return True


def _render_question_slots(
    state: str, data: dict[str, Any], questions: list[dict] | None = None
) -> str:
    """Build filled vs missing slot lines for the current question."""
    q = questions_index(questions).get(state) if questions else None
    if not q:
        return "No slot schema for this state."
    cfg = build_question_slot_config(q)
    answer_type = str(q.get("answer_type") or "text")

    labels = cfg.get("labels") or {}
    required_any = cfg.get("required_any", False)
    required = cfg.get("required") or ()
    optional = cfg.get("optional") or ()

    filled: list[str] = []
    missing: list[str] = []

    if required_any:
        any_filled = any(
            _slot_value_present(data, f, answer_type=answer_type) for f in required
        )
        for field_key in required:
            if _slot_value_present(data, field_key, answer_type=answer_type):
                filled.append(
                    f'{labels.get(field_key, field_key)}="{data.get(field_key)}"'
                )
        if not any_filled:
            missing.append(
                " or ".join(labels.get(f, f) for f in required)
                + " (at least one with a clear value)"
            )
    else:
        for field_key in required:
            label = labels.get(field_key, field_key)
            if _slot_value_present(data, field_key, answer_type=answer_type):
                filled.append(f'{label}="{data.get(field_key)}"')
            else:
                missing.append(label)

    for field_key in optional:
        if _slot_value_present(data, field_key, answer_type=answer_type):
            filled.append(
                f'{labels.get(field_key, field_key)}="{data.get(field_key)}" (optional)'
            )

    lines = []
    if filled:
        lines.append("Already captured for THIS question: " + "; ".join(filled))
    else:
        lines.append("Already captured for THIS question: nothing yet")
    if missing:
        lines.append("Still needed BEFORE question_complete=true: " + "; ".join(missing))
    else:
        lines.append("Still needed BEFORE question_complete=true: none (you may set question_complete=true if the answer is good)")
    hint = cfg.get("complete_hint")
    if hint:
        lines.append(f"Completeness note: {hint}")
    return "\n".join(lines)


_PROMPT_EXTRACTED_BUDGET = 3500
_MAX_TRANSCRIPT_ENTRIES = 400
_MAX_TRANSCRIPT_CHARS = 120_000

# Interrogative markers used to decide if a caller turn is likely a question and
# therefore needs the FULL approved FAQ answers in the prompt as a safety net.
_QUESTION_MARKERS_RE = re.compile(
    r"\?|\b(what|how|why|when|where|who|which|do you|does|are you|is there|"
    r"can i|can you|could you|would you|will you|should i|tell me|"
    r"i have a question|wondering|curious)\b",
    re.I,
)


def _faq_topic_index(active_faqs: list[dict[str, Any]]) -> str:
    """Compact one-line-per-FAQ index of topics (no answers).

    Tells the model which topics it can address without spending the full
    answer text on every turn. ~1 line each vs. a full paragraph.
    """
    lines = [
        f'- {entry.get("topic", "")}: {entry.get("title", "")}'.rstrip(": ").strip()
        for entry in active_faqs
    ]
    return "\n".join(line for line in lines if line) or "None configured."


def match_faqs_by_pattern(
    text: str, active_faqs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return FAQ entries whose regex pattern matches *text*."""
    matched: list[dict[str, Any]] = []
    for entry in active_faqs:
        pattern = str(entry.get("pattern") or "").strip()
        if not pattern:
            continue
        try:
            if re.search(pattern, text, re.I):
                matched.append(entry)
        except re.error:
            continue
    return matched


_FAQ_UNMATCHED_TOP_K = 3


def _faq_relevance_score(text: str, entry: dict[str, Any]) -> int:
    """Rough keyword overlap between caller text and an FAQ entry."""
    from app.core.screening_flow import normalize_text

    norm = normalize_text(text)
    words = {w for w in re.findall(r"[a-z0-9]+", norm) if len(w) >= 3}
    if not words:
        return 0
    score = 0
    for field, weight in (("topic", 3), ("title", 3), ("pattern", 1), ("answer", 1)):
        blob = normalize_text(str(entry.get(field) or ""))
        for word in words:
            if word in blob:
                score += weight
    return score


def _top_faqs_for_unmatched_question(
    text: str, active_faqs: list[dict[str, Any]], *, limit: int = _FAQ_UNMATCHED_TOP_K
) -> list[dict[str, Any]]:
    """Pick the most relevant admin FAQ entries when no regex pattern matched."""
    ranked = sorted(
        ((_faq_relevance_score(text, entry), entry) for entry in active_faqs),
        key=lambda pair: pair[0],
        reverse=True,
    )
    top = [entry for score, entry in ranked if score > 0][:limit]
    return top or active_faqs[:limit]


def _select_faq_block(
    active_faqs: list[dict[str, Any]], transcript: str
) -> tuple[str, bool]:
    """Choose the FAQ text to embed this turn (token-aware).

    Returns ``(block_text, is_full)``. To keep the per-turn prompt small, the
    full approved answers are embedded ONLY when the caller's turn actually
    needs them:
      - their words match one or more FAQ patterns → embed just those answers;
      - their words look like a question but match nothing → embed the top few
        relevant admin FAQ answers (not the entire list);
      - otherwise (a normal screening answer) → embed only the compact topic
        index, which is a fraction of the size.
    The model can still set faq_topic from the always-present topic index.
    """
    if not active_faqs:
        return "None configured.", False

    text = (transcript or "").strip()
    matched = match_faqs_by_pattern(text, active_faqs)

    def _full(entries: list[dict[str, Any]]) -> str:
        return (
            "\n".join(
                f'- topic "{e.get("topic", "")}": {str(e.get("answer", "")).strip()}'
                for e in entries
            )
            or "None configured."
        )

    if matched:
        return _full(matched), True
    if _QUESTION_MARKERS_RE.search(text):
        relevant = _top_faqs_for_unmatched_question(text, active_faqs)
        return _full(relevant), True
    # Normal answer turn: just the cheap topic index.
    return _faq_topic_index(active_faqs), False


def _compact_extracted_for_prompt(data: dict[str, Any]) -> str:
    """Serialize extracted_data for the system prompt within a char budget.

    On long calls (many fields, custom questions, verbose raw text) the full
    dump can balloon the per-turn prompt and risk hitting the model's context
    limit. We keep every field when small; when over budget we shed the bulky
    free-text ``*_raw`` fields first (their normalized counterparts remain),
    then hard-cap as a last resort. Scalar answers are always preserved.
    """
    full = json.dumps(data or {}, default=str)
    if len(full) <= _PROMPT_EXTRACTED_BUDGET:
        return full

    trimmed = {
        k: v
        for k, v in (data or {}).items()
        if not (str(k).endswith("_raw") and isinstance(v, str) and len(v) > 40)
    }
    compact = json.dumps(trimmed, default=str)
    if len(compact) <= _PROMPT_EXTRACTED_BUDGET:
        return compact

    # Still too large: keep scalar values only (drop nested/long strings).
    scalars = {
        k: v
        for k, v in trimmed.items()
        if isinstance(v, (bool, int, float))
        or (isinstance(v, str) and len(v) <= 60)
    }
    return json.dumps(scalars, default=str)[: _PROMPT_EXTRACTED_BUDGET]


def build_system_prompt(
    session: ConversationSession,
    *,
    transcript: str = "",
    confirmation: dict[str, Any] | None = None,
) -> str:
    question = session.get_current_question()
    question_text = (
        localized_question_text(
            question,
            language_code=session.call_language,
            key="question",
        )
        if question
        else "Close the call politely."
    )
    retry_prompt = (
        retry_prompt_for_count(
            question,
            session.retry_count,
            language_code=session.call_language,
        )
        if question
        else question_text
    )
    # Use the admin-configured property/business name in the agent's identity so
    # it introduces itself correctly; fall back to the built-in constant.
    business = (session.property_name or "").strip() or BUSINESS_NAME
    state_value = session.current_state
    question_guide = (
        understanding_guide_for_question(question)
        if question
        else "Extract the field that matches the current screening question."
    )
    validation_hint = (
        validation_hint_for_question(question) if question else ""
    )

    active_faqs = [
        entry
        for entry in session.faqs
        if entry.get("active", True) and entry.get("answer")
    ]
    faq_topic_keys = ", ".join(
        str(entry.get("topic", "")) for entry in active_faqs
    ) or "none"
    # Token-aware FAQ embedding: full approved answers only when the caller's
    # turn needs them (a question / pattern match); otherwise a cheap topic
    # index. This cuts ~600-800 tokens on the ~80% of turns that are plain
    # answers, with no loss of answer quality on the turns that ARE questions.
    faq_text, faq_is_full = _select_faq_block(active_faqs, transcript)
    if faq_is_full:
        faq_section = (
            "# APPROVED FAQ ANSWERS (never invent policy — use only these)\n"
            + faq_text
        )
    else:
        faq_section = (
            "# FAQ TOPICS YOU CAN ANSWER (caller didn't ask one this turn). If "
            "they ask about one, answer ONLY from approved text; if it's not "
            "listed, say a specialist will confirm — never invent policy.\n"
            + faq_text
        )

    extracted_json = _compact_extracted_for_prompt(session.extracted_data)
    caller_line = transcript.strip()
    slots_block = _render_question_slots(
        state_value, session.extracted_data, session.questions
    )
    retry_note = ""
    if session.retry_count > 0 and question:
        retry_note = (
            f"\n- Retry #{session.retry_count}/{session.max_retries}: ask only "
            f"still-missing slot(s); vary wording (e.g. \"{retry_prompt}\")."
        )

    fields_catalog = prompt_fields_catalog(session.questions)
    flow_outline = prompt_screening_flow_outline(
        session.questions, current_state=state_value
    )
    flow_stats = prompt_flow_stats(session.questions)
    slot_examples = slot_fill_examples_for_question(question)
    confirmed_line = ""
    if session.confirmed_fields:
        confirmed_line = (
            "\n- Confirmed (do not change unless caller explicitly corrects): "
            + ", ".join(sorted(session.confirmed_fields))
        )
    yes_no_fields = [
        field
        for field, answer_type in field_answer_types_from_questions(session.questions).items()
        if answer_type == "yes_no"
    ]
    yes_no_hint = (
        f" Set {', '.join(yes_no_fields)} as true/false."
        if yes_no_fields
        else ""
    )
    optional_notes_states = [
        str(q["state"])
        for q in normalize_questions(session.questions)
        if q.get("active", True)
        and q.get("required") is False
        and str(q.get("answer_type") or "") == "long_text"
    ]
    nothing_intent_hint = (
        ' Use intent "nothing" only when the caller confirms they have nothing to add '
        f"on optional note questions ({', '.join(optional_notes_states)})."
        if optional_notes_states
        else ""
    )
    language_label = (
        "Spanish"
        if str(getattr(session, "call_language", "en")).lower().startswith("es")
        else "English"
    )

    if confirmation and confirmation.get("mode") == "correction":
        cfields = confirmation.get("fields", []) or []
        readback_list = "; ".join(
            f'{c.get("field")}="{c.get("value")}"' for c in cfields
        )
        return f"""# ROLE
You are the conversational intelligence engine for "{business}", an AI voice agent on a live tenant-screening call. The caller corrected one or more details they gave EARLIER, and you just read the updated values back to confirm them. Decide what they meant.

# CONFIRMATION CONTEXT
- You read these updated values back: {readback_list}
- Caller just replied: "{caller_line}"
- Extracted data so far: {extracted_json}

{faq_section}

# HOW TO CLASSIFY (set "intent")
- "answer": they CONFIRMED the updated values are correct (e.g. "yes", "that's right"). Leave extracted_data empty and corrected_fields empty.
- "answer" WITH a further correction: they changed a value again — put the new value in extracted_data under its field name and list it in corrected_fields.
- "refusal": they say it's still wrong but won't give the correction.
- "question": they asked us something — answer from approved FAQ in response_text (set faq_topic).
- "human" / "callback" / "stop": they want a person, a callback, or to end.
Set understood=true when they clearly confirmed or corrected; false otherwise.

# VOICE UX
Keep response_text under 18 words, natural, no markdown. A brief "Perfect, thank you." is enough when they confirm.
Always write response_text in {language_label}.

# OUTPUT FORMAT
Respond with ONE JSON object only — no markdown, no code fences. response_text MUST be first.
{{
  "response_text": "short spoken reply",
  "intent": "answer",
  "faq_topic": null,
  "understood": true,
  "relevance": "on_topic",
  "corrected_fields": [],
  "extracted_data": {{}},
  "call_complete": false
}}"""

    if confirmation:
        cfield = confirmation.get("field", "")
        cvalue = confirmation.get("value", "")
        return f"""# ROLE
You are the conversational intelligence engine for "{business}", an AI voice agent on a live tenant-screening call. You just READ A VALUE BACK to the caller to confirm it, and they replied. Decide what they meant.

# CONFIRMATION CONTEXT
- You read back: {cfield} = "{cvalue}"
- Caller just replied: "{caller_line}"
- Extracted data so far: {extracted_json}

{faq_section}

# HOW TO CLASSIFY (set "intent")
- "answer": they CONFIRMED the value is correct (e.g. "yes", "that's right", "correct"). Leave extracted_data empty.
- "answer" WITH a corrected value: they restated it differently (e.g. "no, it's John with an h"). Put the corrected value in extracted_data under "{cfield}".
- "refusal": they reject the value but won't give a correction.
- "question": they asked us something instead — answer it from the approved FAQ answers in response_text (set faq_topic), and keep response_text short.
- "human" / "callback" / "stop": they want a person, a callback, or to end.
Set understood=true when they clearly confirmed or corrected; false otherwise.

# KEEP BUILDING CONTEXT (you can see the whole conversation)
- If they SPELL the value letter by letter (now or across turns), assemble the letters into the corrected value and put the assembled result in extracted_data under "{cfield}".
- If they also VOLUNTEER other details while confirming (e.g. mention their email or employer), extract those into extracted_data too — never drop information.
- Use everything said earlier; don't ask for anything they already provided.

# VOICE UX
Keep response_text under 18 words, natural, no markdown. Vary your wording — don't always say the same phrase. If they confirmed, a brief, warm acknowledgment is enough (the next question is added by the system).
Always write response_text in {language_label}.

# OUTPUT FORMAT
Respond with ONE JSON object only — no markdown, no code fences. response_text MUST be first.
{{
  "response_text": "short spoken reply",
  "intent": "answer",
  "faq_topic": null,
  "understood": true,
  "extracted_data": {{}},
  "call_complete": false
}}"""

    return f"""# ROLE
You are the conversational intelligence for "{business}", screening tenants on a live call. Understand casual, partial, or mixed answers; extract JSON; reply warmly in under 20 words.

# RULES (stable — same for every turn on this call)
- Use SLOTS + captured data; never re-ask filled fields. question_complete=true only when CURRENT question is fully answered; false while you are asking a follow-up.
- Vague dates/amounts: ask once for precision; accept if they cannot be more specific. For dates: if the year is before today, ask whether they meant a future date before accepting.
- Cross-fill: if they volunteer later-step info, extract it, acknowledge briefly, stay on CURRENT.
- Spoken text only — no markdown, bullets, or lists. Vary acknowledgments; answer interruptions warmly first, then resume CURRENT.
- Always write response_text in {language_label}.
- Extract only configured fields below, including future volunteered fields. *_raw = caller's exact words as a string (never true/false or bare numbers). Dates: ISO YYYY-MM-DD when clear. Phone/email/money: raw value; system normalizes.
- Do not overwrite confirmed fields unless corrected this turn. understood=true if they answered CURRENT (even partial); false if off-topic, gibberish, only asked us, declined, or human/callback/stop.
Fields:
{fields_catalog}

# INTENT (exactly one)
- answer: answered CURRENT.{yes_no_hint}
- question: they asked us — use approved FAQ text (faq_topic), then re-ask CURRENT. Never invent policy.
- refusal | human | callback | stop | echo | nothing (optional notes only).{nothing_intent_hint}
faq_topic keys: {faq_topic_keys} or null. Answer+question in one breath → prefer answer, still set faq_topic.

# SIGNALS
relevance: on_topic|off_topic|unclear. corrected_fields: earlier fields they changed ([] if none). consistency_issue/plausibility_issue: short phrase + clarifying response_text, or null (at most one, only when genuinely unsure).

# OUTPUT — one JSON object, response_text first, no fences
{{
  "response_text": "short spoken reply",
  "intent": "answer",
  "faq_topic": null,
  "understood": true,
  "question_complete": false,
  "relevance": "on_topic",
  "corrected_fields": [],
  "consistency_issue": null,
  "plausibility_issue": null,
  "extracted_data": {{}},
  "call_complete": false
}}

# CALL CONTEXT (changes each turn)
- Today: {date.today().isoformat()} (use for all date questions — clarify past dates)
- Property: {business} | {flow_stats}
- Flow order (states; full wording is on CURRENT):
{flow_outline}
- Current: {state_value} — "{question_text}"
- Guide: {question_guide}
- Expect: {validation_hint or "complete answer for current question"}
- Slots this question:
{slots_block}
- Captured so far: {extracted_json}
- Caller: "{caller_line}"{retry_note}{confirmed_line}
{slot_examples}

{faq_section}"""


def validate_llm_response(response_data: dict) -> tuple[bool, str]:
    required_fields = ["response_text", "understood"]
    for field_name in required_fields:
        if field_name not in response_data:
            return False, f"Missing required field: {field_name}"
    if not isinstance(response_data.get("understood"), bool):
        return False, "understood must be a boolean"
    if not str(response_data.get("response_text", "")).strip():
        return False, "response_text is empty"
    return True, ""


# The full set of intents the conversational LLM may return for a turn. This is
# the single source of truth for caller intent — there is no regex router.
TURN_INTENTS = frozenset(
    {
        "answer",
        "question",
        "refusal",
        "human",
        "callback",
        "stop",
        "echo",
        "nothing",
    }
)

_CONTROL_INTENT_MAP = {
    "human": "human_requested",
    "callback": "callback_requested",
    "stop": "stop_requested",
}


def parse_turn_intent(response_data: dict) -> str:
    """Normalize the LLM's ``intent`` field to a known value (default 'answer')."""
    raw = str(response_data.get("intent", "") or "").strip().lower()
    if raw in TURN_INTENTS:
        return raw
    # Tolerate close variants the model might emit.
    if raw.startswith("quest") or raw == "faq":
        return "question"
    if raw.startswith("refus") or raw in {"decline", "declined"}:
        return "refusal"
    if raw in {"agent", "representative", "rep", "person"}:
        return "human"
    if raw in {"hangup", "hang_up", "end", "quit", "cancel"}:
        return "stop"
    if raw in {"none", "nothing_to_add", "no_more"}:
        return "nothing"
    return "answer"


def control_flag_for_intent(intent: str) -> str | None:
    """Map a turn intent to a control-flag name, or None if not a control intent."""
    return _CONTROL_INTENT_MAP.get(intent)


def parse_question_complete(response_data: dict) -> bool:
    """Whether the LLM considers the CURRENT question fully answered."""
    val = response_data.get("question_complete")
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in {"true", "yes", "1"}
    return False


_RELEVANCE_VALUES = frozenset({"on_topic", "off_topic", "unclear"})


def parse_relevance(response_data: dict) -> str:
    """Normalize the LLM's ``relevance`` signal (default 'on_topic')."""
    raw = str(response_data.get("relevance", "") or "").strip().lower()
    raw = raw.replace("-", "_").replace(" ", "_")
    if raw in _RELEVANCE_VALUES:
        return raw
    if raw in {"offtopic", "off", "irrelevant", "unrelated"}:
        return "off_topic"
    if raw in {"unintelligible", "gibberish", "garbled", "ambiguous", "confusing"}:
        return "unclear"
    return "on_topic"


def parse_corrected_fields(
    response_data: dict,
    field_to_state: dict[str, str] | None = None,
) -> list[str]:
    """Earlier fields the caller changed this turn (validated against the schema)."""
    raw = response_data.get("corrected_fields")
    if not isinstance(raw, (list, tuple)):
        return []
    valid = field_to_state or {}
    out: list[str] = []
    for item in raw:
        name = str(item or "").strip()
        if name in valid and name not in out:
            out.append(name)
    return out


def parse_issue(response_data: dict, key: str) -> str:
    """Return a non-empty consistency/plausibility issue description, else ''."""
    val = response_data.get(key)
    if not val or not isinstance(val, str):
        return ""
    text = val.strip()
    if text.lower() in {"null", "none", "false", "n/a", "no"}:
        return ""
    return text


DEFAULT_PROVIDER_FAILURE_MESSAGE = (
    "I'm sorry — we're having a technical issue on our end and can't continue "
    "this call right now. Someone from {property_name} will review what we "
    "have and follow up with you soon. Goodbye."
)

STT_EMPTY_STRIKE_LIMIT = 3


def provider_failure_message_for_session(session: ConversationSession) -> str:
    """Admin-configured script when voice providers are unavailable."""
    business = (session.property_name or "").strip() or BUSINESS_NAME
    template = (getattr(session, "provider_failure_message", "") or "").strip()
    if not template:
        if str(getattr(session, "call_language", "en")).lower().startswith("es"):
            template = (
                "Lo siento, estamos teniendo un problema tecnico y no podemos "
                "continuar esta llamada ahora. Alguien de {property_name} revisara "
                "lo compartido y le dara seguimiento pronto. Adios."
            )
        else:
            template = DEFAULT_PROVIDER_FAILURE_MESSAGE
    return template.replace("{property_name}", business)
