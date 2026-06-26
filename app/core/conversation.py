"""Conversation state, validation, prompts, and response shaping."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from app.core.screening_flow import (
    BUSINESS_NAME,
    DEFAULT_FAQ_ENTRIES,
    DEFAULT_SCREENING_QUESTIONS,
    FLOW_STATE_VALUES,
    QUESTION_BY_STATE,
    count_active_questions,
    count_answered_questions,
    inactive_flow_states,
    is_question_answered,
    next_unanswered_state,
    normalize_extracted_fields,
    normalize_faqs,
    normalize_questions,
    screening_complete,
)

logger = logging.getLogger(__name__)


class CallState(str, Enum):
    """All possible states in the Ready Rentals phone screening."""

    IDLE = "IDLE"
    GREETING = "GREETING"
    Q1_FULL_NAME = "Q1_FULL_NAME"
    Q2_PHONE = "Q2_PHONE"
    Q3_EMAIL = "Q3_EMAIL"
    Q4_MOVE_IN_DATE = "Q4_MOVE_IN_DATE"
    Q5_OCCUPANTS = "Q5_OCCUPANTS"
    Q6_PETS = "Q6_PETS"
    Q6A_PET_DETAILS = "Q6A_PET_DETAILS"
    Q7_CURRENT_RESIDENCE = "Q7_CURRENT_RESIDENCE"
    Q8_RESIDENCE_DURATION = "Q8_RESIDENCE_DURATION"
    Q9_MOVE_REASON = "Q9_MOVE_REASON"
    Q10_MOVE_TIMING = "Q10_MOVE_TIMING"
    Q11_EVICTION = "Q11_EVICTION"
    Q11A_EVICTION_DETAILS = "Q11A_EVICTION_DETAILS"
    Q12_INCOME = "Q12_INCOME"
    Q13_EMPLOYER = "Q13_EMPLOYER"
    Q14_EMPLOYMENT_DURATION = "Q14_EMPLOYMENT_DURATION"
    Q15_GENERAL_NOTES = "Q15_GENERAL_NOTES"
    WRAP_UP = "WRAP_UP"
    ENDED = "ENDED"


STATE_TO_QUESTION_ID = {
    CallState(defn.state): defn.id for defn in QUESTION_BY_STATE.values()
}


@dataclass
class TranscriptEntry:
    """A single turn in the conversation transcript."""

    speaker: str
    text: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(UTC).strftime("%H:%M:%S")
    )
    state: str = ""


@dataclass
class ConversationSession:
    """State for one live call or test-console session."""

    call_id: str
    phone_number: str
    agent_name: str = "Ready Rentals assistant"
    property_name: str = BUSINESS_NAME

    current_state: CallState = CallState.IDLE
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
    questions: list[dict] = field(
        default_factory=lambda: list(DEFAULT_SCREENING_QUESTIONS)
    )
    faqs: list[dict] = field(default_factory=lambda: list(DEFAULT_FAQ_ENTRIES))

    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_activity: datetime = field(default_factory=lambda: datetime.now(UTC))

    stt_provider: str = ""
    llm_provider: str = ""
    tts_provider: str = ""

    call_providers: Any = None
    silence_timeout_seconds: int = 12
    max_call_duration_seconds: int = 600
    auto_fallback_enabled: bool = True
    settings_captured_at: str = ""

    errors: list[dict] = field(default_factory=list)
    interruption_count: int = 0
    # Fields that passed read-back confirmation. These are locked: the LLM/local
    # extractors may not overwrite them on later turns unless the caller issues
    # an explicit correction ("wait, the name is wrong"). Belt-and-suspenders
    # protection against LLM drift on high-stakes data.
    confirmed_fields: set[str] = field(default_factory=set)
    # Pending read-back confirmation for a high-stakes field, e.g.
    # {"field": "contact_phone", "state": "Q2_PHONE", "value": "+1...", "attempts": 1}
    pending_confirmation: dict | None = None
    # Set after a silence nudge ("Are you still there?") — next short ack is
    # liveness, not an answer to the current screening question.
    silence_nudge_active: bool = False

    def __post_init__(self) -> None:
        self.questions = normalize_questions(self.questions)
        self.faqs = normalize_faqs(self.faqs)

    def get_current_question(self) -> dict | None:
        question_id = STATE_TO_QUESTION_ID.get(self.current_state)
        if not question_id:
            return None
        for question in self.questions:
            if question.get("id") == question_id and question.get("active", True):
                return question
        fallback = QUESTION_BY_STATE.get(self.current_state.value)
        return fallback.as_config() if fallback else None

    def next_state(self) -> CallState:
        next_state = next_unanswered_state(self.extracted_data, self.skip_states)
        if next_state:
            self.current_state = CallState(next_state)
        else:
            self.current_state = CallState.WRAP_UP
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
        self.questions_answered = count_answered_questions(
            self.extracted_data,
            self.skip_states,
        )
        for state in FLOW_STATE_VALUES:
            if is_question_answered(state, self.extracted_data, self.skip_states):
                self.mark_answered(state)

    def mark_answered(self, state: str | CallState) -> None:
        state_value = state.value if isinstance(state, CallState) else state
        if state_value in FLOW_STATE_VALUES and state_value not in self.answered_states:
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
        state_value = state.value if isinstance(state, CallState) else state
        if state_value in FLOW_STATE_VALUES:
            self.completed_states.add(state_value)
        self.refresh_progress()

    def merge_extracted_data(self, data: dict[str, Any], *, raw_text: str = "") -> None:
        clean: dict[str, Any] = {}
        for key, value in (data or {}).items():
            value = _unwrap_confidence(value)
            if value not in (None, ""):
                clean[key] = value
        # Regex normalization layer: format the LLM's structured output for
        # storage. Regex shapes data only here, never on raw caller speech.
        clean = normalize_extracted_fields(clean)
        if not clean:
            return
        self.extracted_data.update(clean)
        if raw_text and self.current_state.value in FLOW_STATE_VALUES:
            self.raw_answers[self.current_state.value] = raw_text
        self.refresh_progress()

    def mark_field_confirmed(self, field_name: str) -> None:
        """Lock a field after it passes read-back confirmation."""
        if field_name:
            self.confirmed_fields.add(field_name)

    def is_screening_complete(self) -> bool:
        return screening_complete(self.extracted_data, self.skip_states)

    def active_question_count(self) -> int:
        return count_active_questions(self.extracted_data, self.skip_states)

    def add_transcript(self, speaker: str, text: str) -> None:
        entry = TranscriptEntry(
            speaker=speaker,
            text=text,
            state=self.current_state.value,
        )
        self.transcript.append(entry)
        self.last_activity = datetime.now(UTC)

    def add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > 24:
            self.messages = self.messages[-24:]

    def get_full_transcript(self) -> str:
        return "\n".join(
            f"[{entry.timestamp}] {entry.speaker}: {entry.text}"
            for entry in self.transcript
        )

    def add_error(self, error_type: str, message: str) -> None:
        self.errors.append(
            {
                "type": error_type,
                "message": message,
                "timestamp": datetime.now(UTC).isoformat(),
                "state": self.current_state.value,
            }
        )
        # Bound growth on a pathological call (e.g. a provider failing every turn)
        # so a single long call can't accumulate unbounded error entries.
        if len(self.errors) > 50:
            self.errors = self.errors[-50:]

    @property
    def duration_seconds(self) -> int:
        return int((datetime.now(UTC) - self.started_at).total_seconds())

    def to_dict(self) -> dict:
        return {
            "call_id": self.call_id,
            "phone_number": self.phone_number,
            "current_state": self.current_state.value,
            "state": self.current_state.value,
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

_LIVENESS_ACKS = frozenset(
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


def is_liveness_acknowledgment(transcript: str) -> bool:
    """Short reply to a silence nudge — not an answer to the screening question."""
    norm = _normalize_speech(transcript)
    if not norm:
        return False
    if norm in _LIVENESS_ACKS:
        return True
    if len(norm) <= 24 and re.search(
        r"\b(still here|i'?m here|yes i'?m|here yes|i am)\b", norm
    ):
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
    question_cfg = session.get_current_question()
    if not question_cfg:
        return "No problem. Please continue when you are ready."

    retry_count = session.retry_count
    if question_cfg.get("retry_prompt_3") and retry_count >= 2:
        prompt = question_cfg["retry_prompt_3"]
    elif question_cfg.get("retry_prompt_2") and retry_count >= 1:
        prompt = question_cfg["retry_prompt_2"]
    else:
        prompt = question_cfg.get("retry_prompt") or question_cfg["question"]

    if kind == "refusal":
        return (
            "That's completely okay — we ask everyone the same questions and it "
            f"stays confidential. {prompt}"
        )

    lead = _REDIRECT_LEADINS[min(retry_count, len(_REDIRECT_LEADINS) - 1)]
    return f"{lead} {prompt}"


def compose_agent_response(
    session: ConversationSession,
    acknowledgment: str,
    prior_state: CallState,
) -> tuple[str, str]:
    """Return spoken acknowledgment and the next deterministic prompt."""
    ack = (acknowledgment or "").strip()
    current = session.current_state

    if current.value in FLOW_STATE_VALUES:
        question = session.get_current_question()
        if not question:
            return ack, ""

        # Use retry prompt variations based on attempt count
        if current == prior_state and session.retry_count > 0:
            # Check if question has retry_prompt_2 or retry_prompt_3
            if question.get("retry_prompt_3") and session.retry_count >= 3:
                prompt = question["retry_prompt_3"]
            elif question.get("retry_prompt_2") and session.retry_count >= 2:
                prompt = question["retry_prompt_2"]
            else:
                prompt = question.get("retry_prompt", question["question"])
        else:
            prompt = question["question"]

        if not ack:
            return prompt, ""
        if prompt.lower() not in ack.lower():
            return ack, prompt
        return ack, ""

    if current == CallState.WRAP_UP:
        closing = (
            "Thank you. A leasing specialist will review your information "
            "and follow up soon."
        )
        return (ack, closing) if ack else (closing, "")

    if current == CallState.ENDED and not ack:
        return "Thank you for calling Ready Rentals Online. Goodbye.", ""

    return ack, ""


# Per-question slot schema: what sub-details the LLM must gather before marking
# question_complete=true. Single-slot questions are trivially complete once filled.
_QUESTION_SLOTS: dict[str, dict[str, Any]] = {
    "Q1_FULL_NAME": {
        "required": ("full_name",),
        "optional": (),
        "labels": {"full_name": "full legal name (first and last)"},
        "complete_hint": "Both first and last name required; spelled letters must be assembled.",
    },
    "Q2_PHONE": {
        "required": ("contact_phone",),
        "optional": (),
        "labels": {"contact_phone": "phone number"},
    },
    "Q3_EMAIL": {
        "required": ("email",),
        "optional": (),
        "labels": {"email": "email address"},
        "complete_hint": "Local part and domain required; assemble spelled letters.",
    },
    "Q4_MOVE_IN_DATE": {
        "required": ("move_in_date", "move_in_raw"),
        "required_any": True,
        "optional": (),
        "labels": {
            "move_in_date": "move-in date (ISO if clear)",
            "move_in_raw": "move-in timeframe wording",
        },
    },
    "Q5_OCCUPANTS": {
        "required": ("occupants_count",),
        "optional": ("adults_count", "children_count"),
        "labels": {
            "occupants_count": "total occupants",
            "adults_count": "adults",
            "children_count": "children",
        },
    },
    "Q6_PETS": {
        "required": ("has_pets",),
        "optional": ("pets_raw",),
        "labels": {"has_pets": "yes/no pets", "pets_raw": "pet description"},
    },
    "Q6A_PET_DETAILS": {
        "required": ("pet_type", "pet_breed", "pet_weight"),
        "optional": ("pets_raw",),
        "labels": {
            "pet_type": "pet type (dog, cat, etc.)",
            "pet_breed": "breed",
            "pet_weight": "approximate weight",
        },
        "complete_hint": "All three required before question_complete=true.",
    },
    "Q7_CURRENT_RESIDENCE": {
        "required": ("current_residence",),
        "optional": (),
        "labels": {"current_residence": "current address or area"},
    },
    "Q8_RESIDENCE_DURATION": {
        "required": ("residence_duration",),
        "optional": (),
        "labels": {"residence_duration": "how long at current home"},
    },
    "Q9_MOVE_REASON": {
        "required": ("move_reason",),
        "optional": (),
        "labels": {"move_reason": "reason for moving"},
    },
    "Q10_MOVE_TIMING": {
        "required": ("move_timing",),
        "optional": (),
        "labels": {"move_timing": "when they plan to leave current place"},
    },
    "Q11_EVICTION": {
        "required": ("has_eviction",),
        "optional": ("eviction_raw",),
        "labels": {
            "has_eviction": "yes/no eviction history",
            "eviction_raw": "brief eviction mention",
        },
    },
    "Q11A_EVICTION_DETAILS": {
        "required": ("eviction_circumstances",),
        "optional": ("eviction_raw",),
        "labels": {"eviction_circumstances": "eviction circumstances"},
    },
    "Q12_INCOME": {
        "required": ("monthly_income", "income_raw"),
        "required_any": True,
        "optional": (),
        "labels": {
            "monthly_income": "monthly household income before taxes",
            "income_raw": "income wording (amount + hourly/monthly/annual)",
        },
        "complete_hint": "Need a clear amount AND period (monthly/hourly/annual).",
    },
    "Q13_EMPLOYER": {
        "required": ("employer",),
        "optional": (),
        "labels": {"employer": "employer or income source"},
    },
    "Q14_EMPLOYMENT_DURATION": {
        "required": ("employment_duration",),
        "optional": (),
        "labels": {"employment_duration": "time at current job"},
    },
    "Q15_GENERAL_NOTES": {
        "required": ("general_notes",),
        "optional": (),
        "labels": {"general_notes": "final notes or 'None disclosed'"},
    },
}

# Derived from the slot schema: which question owns each field, and a human
# label for reading a corrected value back to the caller.
FIELD_TO_STATE: dict[str, str] = {}
FIELD_LABELS: dict[str, str] = {}
for _state, _cfg in _QUESTION_SLOTS.items():
    _labels = _cfg.get("labels") or {}
    for _field in tuple(_cfg.get("required") or ()) + tuple(_cfg.get("optional") or ()):
        FIELD_TO_STATE.setdefault(_field, _state)
        FIELD_LABELS.setdefault(_field, _labels.get(_field, _field.replace("_", " ")))


def _short_label(field: str) -> str:
    """A concise spoken label for a field (used in correction read-backs)."""
    raw = FIELD_LABELS.get(field, field.replace("_", " "))
    # Trim parenthetical hints like "move-in date (ISO if clear)".
    return raw.split("(")[0].strip()


def build_correction_readback(fields: list[dict[str, str]]) -> str:
    """Combined read-back confirming one or more EARLIER fields the caller just
    corrected. ``fields`` is a list of {"field", "value"} dicts."""
    parts: list[str] = []
    for item in fields:
        label = _short_label(item.get("field", ""))
        value = str(item.get("value", "")).strip()
        if not value:
            continue
        parts.append(f"your {label} to {value}")
    if not parts:
        return "Let me make sure I have your updated details right. Is that correct?"
    if len(parts) == 1:
        joined = parts[0]
    else:
        joined = ", ".join(parts[:-1]) + ", and " + parts[-1]
    return f"Quick check before we move on — I've updated {joined}. Did I get that right?"


_SLOT_FILL_EXAMPLES = """
# HUMAN SLOT-FILLING EXAMPLES (follow this pattern)
- Pets: Caller "I have a dog." → extract pet_type=dog, question_complete=false, response "A dog, lovely. What breed is it, and roughly how much does it weigh?"
- Offer to spell: Caller "Let me spell my name for you." → question_complete=false, response "Of course — go right ahead." Then assemble letters on the next turns.
- Name spelled: Caller "J-o-h-n Smith" → assemble full_name="John Smith", question_complete=true only when first AND last are clear.
- Vague date: Caller "I'd move this Sunday." → extract move_in_raw="this Sunday", question_complete=FALSE, response "Great — and what's the exact date that lands on?" (do NOT advance).
- Cross-fill: Caller gives email while on the phone question → extract email too, acknowledge briefly, stay on the current question until it's complete.
"""


def _slot_value_present(data: dict[str, Any], field: str) -> bool:
    val = data.get(field)
    if val is None or val == "" or val == []:
        return False
    if field == "has_pets" or field == "has_eviction":
        return val is True or val is False
    return True


def _render_question_slots(state: str, data: dict[str, Any]) -> str:
    """Build filled vs missing slot lines for the current question."""
    cfg = _QUESTION_SLOTS.get(state)
    if not cfg:
        return "No slot schema for this state."

    labels = cfg.get("labels") or {}
    required_any = cfg.get("required_any", False)
    required = cfg.get("required") or ()
    optional = cfg.get("optional") or ()

    filled: list[str] = []
    missing: list[str] = []

    if required_any:
        any_filled = any(_slot_value_present(data, f) for f in required)
        for field in required:
            if _slot_value_present(data, field):
                filled.append(f'{labels.get(field, field)}="{data.get(field)}"')
        if not any_filled:
            missing.append(
                " or ".join(labels.get(f, f) for f in required)
                + " (at least one with a clear value)"
            )
    else:
        for field in required:
            label = labels.get(field, field)
            if _slot_value_present(data, field):
                filled.append(f'{label}="{data.get(field)}"')
            else:
                missing.append(label)

    for field in optional:
        if _slot_value_present(data, field):
            filled.append(f'{labels.get(field, field)}="{data.get(field)}" (optional)')

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


_QUESTION_UNDERSTANDING_GUIDE: dict[str, str] = {
    "Q1_FULL_NAME": (
        "Listen for full legal name even if given casually ('I'm John Smith', 'this is Maria'). "
        "Strip filler ('my name is', 'I said'). Spelled letters are a correction — assemble them. "
        "A single first name alone is NOT enough — set understood=false and ask for full name."
    ),
    "Q2_PHONE": (
        "Accept any format: spoken digits, grouped numbers, 'my cell is...'. "
        "Normalize to digits. Partial numbers mean understood=false."
    ),
    "Q3_EMAIL": (
        "Accept spoken email ('john at gmail dot com'), spelled local parts, corrections "
        "('no it's j-o-h-n at...'). Must have a local part before @. "
        "If only a domain is given, set understood=false."
    ),
    "Q4_MOVE_IN_DATE": (
        "Accept relative dates ('next month', 'ASAP', 'July 15'), numeric dates (MM/DD/YYYY), "
        "and vague windows ('this summer'). Store ISO date when clear; otherwise move_in_raw."
    ),
    "Q5_OCCUPANTS": (
        "Count everyone: 'me and my brother' = 2, 'family of four' = 4, 'just me' = 1. "
        "Include roommates, partners, parents living with them. Split adults vs children when stated."
    ),
    "Q6_PETS": (
        "Boolean only. 'No pets', 'pet-free', 'I have a dog' → has_pets true/false. "
        "If they describe pets without yes/no, infer has_pets=true."
    ),
    "Q6A_PET_DETAILS": (
        "Extract type, breed, weight. Accept casual phrasing ('small terrier about 15 pounds')."
    ),
    "Q7_CURRENT_RESIDENCE": (
        "Free-text address or area. 'I live in Dallas on Oak Street' is valid. "
        "Do NOT treat employer/job descriptions as an address."
    ),
    "Q8_RESIDENCE_DURATION": (
        "How long at current home. Accept 'about 3 years', 'since 2020', 'a couple months'."
    ),
    "Q9_MOVE_REASON": (
        "Why relocating. Accept any honest reason: job, family, lease ending, need more space."
    ),
    "Q10_MOVE_TIMING": (
        "When they plan to leave current place — may differ from move-in date. "
        "Accept relative timing ('end of lease', 'next month')."
    ),
    "Q11_EVICTION": (
        "Boolean. 'Never', 'clean record' → has_eviction=false. "
        "'Yes once', 'had an eviction' → has_eviction=true."
    ),
    "Q11A_EVICTION_DETAILS": (
        "Circumstances only — when, why, outcome. A bare 'yes' is NOT enough; ask for details."
    ),
    "Q12_INCOME": (
        "Gross monthly household income before taxes. If they give an hourly rate, "
        "estimate monthly as hourly times 160 (full-time) and keep income_raw. "
        "If they give an annual salary, divide by 12. If the result is below "
        "$100/month, leave monthly_income unextracted so the state machine can "
        "clarify; always preserve their exact wording in income_raw."
    ),
    "Q13_EMPLOYER": (
        "Employer name or self-employed description. Job duties ARE valid ('I work for myself "
        "cleaning houses'). Do NOT confuse with FAQ questions."
    ),
    "Q14_EMPLOYMENT_DURATION": (
        "How long at current job. Accept '6 months', 'since January', 'about two years'."
    ),
    "Q15_GENERAL_NOTES": (
        "Optional final notes. 'Nothing else', 'that's all', 'I'm good' → general_notes='None disclosed'. "
        "A bare 'yes' means they want to add something — set understood=false and ask what."
    ),
}


def build_system_prompt(
    session: ConversationSession,
    *,
    transcript: str = "",
    local_hints: dict[str, Any] | None = None,
    faq_context: str | None = None,
    confirmation: dict[str, Any] | None = None,
) -> str:
    question = session.get_current_question()
    question_text = question["question"] if question else "Close the call politely."
    retry_prompt = (
        question.get("retry_prompt", question_text) if question else question_text
    )
    # Use the admin-configured property/business name in the agent's identity so
    # it introduces itself correctly; fall back to the built-in constant.
    business = (session.property_name or "").strip() or BUSINESS_NAME
    state_value = session.current_state.value
    question_guide = _QUESTION_UNDERSTANDING_GUIDE.get(
        state_value,
        "Extract the field that matches the current screening question.",
    )

    active_faqs = [
        entry
        for entry in session.faqs
        if entry.get("active", True) and entry.get("answer")
    ]
    faq_answers = (
        "\n".join(
            f'- topic "{entry.get("topic", "")}": {entry.get("answer", "").strip()}'
            for entry in active_faqs
        )
        or "None configured."
    )
    faq_topic_keys = ", ".join(
        str(entry.get("topic", "")) for entry in active_faqs
    ) or "none"

    retry_line = ""
    if session.retry_count > 0 and question:
        retry_line = (
            f"\n- Retry #{session.retry_count} of {session.max_retries} on this "
            f"question. If you must re-ask, vary the wording; you may use: "
            f'"{retry_prompt}".'
        )

    extracted_json = json.dumps(session.extracted_data, default=str)
    hints_json = json.dumps(local_hints or {}, default=str)
    caller_line = transcript.strip()
    faq_block = faq_context.strip() if faq_context else "None"
    slots_block = _render_question_slots(state_value, session.extracted_data)
    follow_up_note = ""
    if session.retry_count > 0:
        follow_up_note = (
            f"\n- Follow-up #{session.retry_count} of {session.max_retries} on this "
            "question. Ask ONLY for the still-missing slot(s) above — do not re-read "
            "the whole question verbatim."
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

# APPROVED FAQ ANSWERS (never invent policy — use only these)
{faq_answers}

# HOW TO CLASSIFY (set "intent")
- "answer": they CONFIRMED the updated values are correct (e.g. "yes", "that's right"). Leave extracted_data empty and corrected_fields empty.
- "answer" WITH a further correction: they changed a value again — put the new value in extracted_data under its field name and list it in corrected_fields.
- "refusal": they say it's still wrong but won't give the correction.
- "question": they asked us something — answer from approved FAQ in response_text (set faq_topic).
- "human" / "callback" / "stop": they want a person, a callback, or to end.
Set understood=true when they clearly confirmed or corrected; false otherwise.

# VOICE UX
Keep response_text under 18 words, natural, no markdown. A brief "Perfect, thank you." is enough when they confirm.

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

# APPROVED FAQ ANSWERS (never invent policy — use only these)
{faq_answers}

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
You are the conversational intelligence engine for "{business}", an advanced AI voice agent screening prospective tenants on a live phone call. Listen to the caller, extract their information into structured JSON, and generate a warm, ultra-concise, natural spoken response. You are the PRIMARY intelligence: the caller may answer in any way — casual, partial, corrected, rambling, or mixed with a question — and you must understand them like an experienced human leasing agent.

# CONTEXT
- Current state / question: {state_value} — "{question_text}"
- Question intelligence guide: {question_guide}
- CURRENT QUESTION SLOTS (your memory for this question only):
{slots_block}
- All extracted data so far (including future questions they volunteered): {extracted_json}
- Relevant FAQ data for this turn: {faq_block}
- Local fallback hints (deterministic parse — verify, correct, or override): {hints_json}
- Caller just said: "{caller_line}"{retry_line}{follow_up_note}

# SLOT-FILLING RULES (sound like a real human agent)
1. REMEMBER what you already captured for THIS question (see SLOTS above) and everything earlier in the call. Build on it — never ask for something the caller already gave.
2. Ask ONLY for still-missing slot(s). Never re-read the entire multi-part question if you already have part of the answer.
3. Set question_complete=true ONLY when the CURRENT question has a complete, good final answer (all required slots satisfied for this question).
4. Set question_complete=false when you still need a sub-detail (e.g. they said "dog" but breed/weight missing) OR when you are asking a clarifying follow-up. If your response_text is a question to the caller, you MUST set question_complete=false.
5. PRECISION: If the caller gives a vague or relative value (e.g. "Sunday", "next month", "a while ago"), keep question_complete=false and ask ONCE for the specific detail (the exact date, an approximate number). If they truly can't be more precise, accept their best answer and set question_complete=true — don't badger.
6. SPELLING: If the caller offers to spell something ("let me spell it", "I'll spell my name"), warmly invite them ("Of course — go ahead") and set question_complete=false. Then assemble the letters/digits across the next turns, confirming the assembled value only once it's complete.
7. If they volunteer info for LATER questions (email, employer, etc.), extract it into extracted_data and acknowledge naturally — those questions will be skipped later.
{_SLOT_FILL_EXAMPLES}

# APPROVED FAQ ANSWERS (never invent policy — use only these)
{faq_answers}

# VOICE UX CONSTRAINTS
1. BREVITY: Keep response_text under 20 words whenever possible. Long replies cause awkward silences and sound robotic.
2. NO MARKDOWN: Never use asterisks, bullets, or dashes. Write exactly what should be spoken aloud.
3. VARIETY (sound human, not scripted): Do NOT begin every reply with "Thank you" or "Got it". React to the SPECIFICS the caller gave ("A golden retriever, those are great", "Downtown, nice area"). Vary your wording every turn the way a real person would.
4. WELCOMING & CONTINUITY: If the caller interrupts with a question, comment, or worry, answer it warmly FIRST, then naturally pick up exactly where you left off ("To your question — ... Now, back to ..."). Never sound annoyed or robotic about being interrupted.
5. FAQ SMOOTHING: If "Relevant FAQ data for this turn" is provided, blend that answer naturally into response_text, then return to the current question.
6. MID-CALL CORRECTIONS: If the caller corrects an earlier field, update extracted_data immediately and say something like "No problem, I've updated that."
7. MEMORY: You can see the whole conversation — reference earlier things the caller mentioned when it feels natural ("Since it's just you and your partner moving in, ...") so it feels like one continuous, attentive conversation.

# EXTRACTION RULES
- Extract ALL fields present in the utterance, even if they belong to future questions (e.g. if they give name and phone together, extract both).
- Never overwrite an already-confirmed field in "Extracted data so far" unless the caller explicitly corrects it this turn.
- Set understood=true if the caller gave any valid, relevant piece of information for the CURRENT question (even if incomplete); set it to false only when they did not answer the current question (off-topic, unintelligible, only asked a question of their own, declined, or asked for a human/callback/stop).
- Set question_complete=true ONLY when the CURRENT question has a complete final answer (see CURRENT QUESTION SLOTS). Set question_complete=false when you still need a sub-detail.
- Ask exactly one question at a time. Evictions are reviewed individually; credit alone is not an automatic disqualifier; Section 8 and housing vouchers are accepted.

# EXTRACTION FIELDS
full_name, contact_phone, email, move_in_date, move_in_raw, occupants_count,
adults_count, children_count, has_pets, pets_raw, pet_type, pet_breed, pet_weight,
current_residence, residence_duration, move_reason, move_timing, has_eviction,
eviction_raw, eviction_circumstances, monthly_income, income_raw, employer,
employment_duration, general_notes.
Use ISO YYYY-MM-DD for dates when clear and keep the caller's exact wording in the matching *_raw field. Do not pre-format phone/email/money — store the caller's value; the system normalizes it after you.

# INTENT — classify the caller's latest message as exactly ONE of:
- "answer": they answered the current screening question (this includes a yes/no answer to a yes/no question — also fill has_pets/has_eviction etc. in extracted_data).
- "question": they asked US something. If it matches an approved FAQ topic, set faq_topic to that topic key and ANSWER it inside response_text using ONLY the approved answer text, then warmly re-ask the current question. If it's a question with no approved answer, briefly say a leasing specialist will confirm, then re-ask. Never invent policy.
- "refusal": they declined to answer the current question.
- "human": they want to speak with a real person / agent / representative.
- "callback": they want us to call them back later, or now isn't a good time.
- "stop": they want to stop, cancel, hang up, or quit now.
- "echo": the message is just our own words echoed back / empty filler with no content.
- "nothing": (only valid on the final notes question) they confirmed they have nothing to add.
Valid faq_topic keys: {faq_topic_keys}. Use null when intent is not "question" or no topic matches.
A caller can both answer AND ask in one breath — prefer intent "answer" and still set faq_topic + blend the FAQ answer into response_text.

# EDGE-CASE INTELLIGENCE (you have the FULL conversation + all extracted data — act like a sharp human agent)
Set these four extra signals on every turn:
- "relevance": one of "on_topic" | "off_topic" | "unclear".
    * "on_topic": they answered the current question, corrected something, or asked a relevant question.
    * "off_topic": the reply has nothing to do with screening (e.g. "I want to go swimming", random chit-chat, a joke). Do NOT put anything in extracted_data. In response_text, warmly acknowledge and steer back to the current question.
    * "unclear": gibberish, a single stray word, or garbled speech you cannot interpret. In response_text, gently ask them to repeat.
- "corrected_fields": a list of field names from EARLIER questions that the caller is changing THIS turn (e.g. while on the income question they say "actually, my number is 555-1234" → ["contact_phone"]). Always ALSO put the new value in extracted_data. Use [] when nothing earlier changed. Do NOT list a field that simply belongs to the CURRENT question.
- "consistency_issue": if the caller's latest answer CONTRADICTS something already in extracted data (e.g. earlier occupants_count=2 but now they mention "my three kids"; a date or count that conflicts), describe the conflict in a short phrase and make response_text a friendly clarifying question that reconciles it. Otherwise null.
- "plausibility_issue": if a value is implausible or likely a misunderstanding (income far too low for a monthly figure, an impossible occupant count, a move-in date in the past, an absurd pet weight), describe it briefly and make response_text a friendly clarifying question. Otherwise null.
Only raise ONE of consistency_issue / plausibility_issue per turn, and only when you are genuinely unsure — never nag about clearly-fine answers.

# OUTPUT FORMAT
Respond with ONE JSON object only — no markdown, no code fences. response_text MUST be the first key so it can stream to the voice engine instantly. Do NOT include next_state; the state machine controls which question comes next.
{{
  "response_text": "short, conversational, completely unformatted spoken reply",
  "intent": "answer",
  "faq_topic": null,
  "understood": true,
  "question_complete": false,
  "relevance": "on_topic",
  "corrected_fields": [],
  "consistency_issue": null,
  "plausibility_issue": null,
  "extracted_data": {{"field_name": "extracted raw value"}},
  "call_complete": false
}}"""


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


def parse_corrected_fields(response_data: dict) -> list[str]:
    """Earlier fields the caller changed this turn (validated against the schema)."""
    raw = response_data.get("corrected_fields")
    if not isinstance(raw, (list, tuple)):
        return []
    out: list[str] = []
    for item in raw:
        name = str(item or "").strip()
        if name in FIELD_TO_STATE and name not in out:
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


def get_fallback_response(state: CallState) -> dict:
    question = QUESTION_BY_STATE.get(state.value)
    if state == CallState.WRAP_UP:
        return {
            "response_text": (
                "Thank you. A leasing specialist will review your information "
                "and follow up soon."
            ),
            "understood": True,
            "extracted_data": {},
            "next_state": "ENDED",
            "call_complete": True,
        }
    if question:
        return {
            "response_text": question.retry_prompt,
            "understood": False,
            "question_complete": False,
            "extracted_data": {},
            "next_state": state.value,
            "call_complete": False,
        }
    return {
        "response_text": "Thank you for calling Ready Rentals Online. Goodbye.",
        "understood": True,
        "extracted_data": {},
        "next_state": "ENDED",
        "call_complete": True,
    }
