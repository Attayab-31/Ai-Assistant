"""Ready Rentals Online screening flow and policy helpers.

This module is intentionally deterministic. The LLM may help extract values,
but the ordered question flow, FAQ answers, skips, and handoff behavior live
here so test console and Telnyx calls follow the same business rules.
"""

from __future__ import annotations

import logging
import random
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any

logger = logging.getLogger(__name__)


BUSINESS_NAME = "Ready Rentals Online"


def build_greeting_intro(business: str = BUSINESS_NAME) -> str:
    """Ready Rentals approved opening script (before the first screening question)."""
    name = (business or "").strip() or BUSINESS_NAME
    return (
        f"Hello and thank you for calling {name}! I'm your virtual assistant. "
        "We've been helping people find their perfect homes for over 30 years. "
        "I'm here to listen to what makes you unique, because we're not just looking "
        "at numbers, we're looking at you as an individual. Let's get started!"
    )


@dataclass(frozen=True)
class ScreeningQuestionDef:
    id: str
    state: str
    question: str
    extract_fields: tuple[str, ...]
    validation: str
    retry_prompt: str
    order: int
    active: bool = True
    retry_prompt_2: str = ""  # Second attempt clarification
    retry_prompt_3: str = ""  # Third attempt more guided

    def as_config(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "state": self.state,
            "question": self.question,
            "extract_fields": list(self.extract_fields),
            "validation": self.validation,
            "retry_prompt": self.retry_prompt,
            "retry_prompt_2": self.retry_prompt_2,
            "retry_prompt_3": self.retry_prompt_3,
            "active": self.active,
            "order": self.order,
        }


QUESTION_DEFS: tuple[ScreeningQuestionDef, ...] = (
    ScreeningQuestionDef(
        id="Q1",
        state="Q1_FULL_NAME",
        question="Can I start with your full name?",
        extract_fields=("full_name",),
        validation="full name or preferred legal name",
        retry_prompt="What is your full name?",
        retry_prompt_2="Could you tell me your first and last name?",
        retry_prompt_3="Just your first and last name is perfect.",
        order=1,
    ),
    ScreeningQuestionDef(
        id="Q2",
        state="Q2_PHONE",
        question="What is the best phone number for you?",
        extract_fields=("contact_phone",),
        validation="phone number",
        retry_prompt="What phone number should the leasing team use?",
        retry_prompt_2="Could you say your number one digit at a time, including the area code?",
        retry_prompt_3="No rush — just read the digits slowly and I'll take them down.",
        order=2,
    ),
    ScreeningQuestionDef(
        id="Q3",
        state="Q3_EMAIL",
        question="What email address should we use?",
        extract_fields=("email",),
        validation="email address",
        retry_prompt="Could you spell or say your email address?",
        retry_prompt_2="You can spell it out — for example, j-o-h-n, at, gmail, dot, com.",
        retry_prompt_3="Just say the letters, and use 'at' for the @ sign and 'dot' for the period.",
        order=3,
    ),
    ScreeningQuestionDef(
        id="Q4",
        state="Q4_MOVE_IN_DATE",
        question="What move-in date are you hoping for?",
        extract_fields=("move_in_date", "move_in_raw"),
        validation="date or timeframe",
        retry_prompt="What date or timeframe are you hoping to move in?",
        order=4,
    ),
    ScreeningQuestionDef(
        id="Q5",
        state="Q5_OCCUPANTS",
        question="How many people would live in the home?",
        extract_fields=("occupants_count", "adults_count", "children_count"),
        validation="number of occupants",
        retry_prompt="How many total occupants would be living there?",
        order=5,
    ),
    ScreeningQuestionDef(
        id="Q6",
        state="Q6_PETS",
        question="Do you have any pets?",
        extract_fields=("has_pets", "pets_raw"),
        validation="yes or no",
        retry_prompt="Do you have any pets, yes or no?",
        order=6,
    ),
    ScreeningQuestionDef(
        id="Q6A",
        state="Q6A_PET_DETAILS",
        question="What type, breed, and approximate weight are they?",
        extract_fields=(
            "pet_type",
            "pet_breed",
            "pet_weight",
            "pets_raw",
        ),
        validation="pet type, breed, and weight",
        retry_prompt="Could you give me the pet type, breed, and weight?",
        order=7,
    ),
    ScreeningQuestionDef(
        id="Q7",
        state="Q7_CURRENT_RESIDENCE",
        question="Where are you currently living?",
        extract_fields=("current_residence",),
        validation="current city/address or housing situation",
        retry_prompt="Where do you currently live?",
        order=8,
    ),
    ScreeningQuestionDef(
        id="Q8",
        state="Q8_RESIDENCE_DURATION",
        question="How long have you lived there?",
        extract_fields=("residence_duration",),
        validation="length of residence",
        retry_prompt="About how long have you lived at your current place?",
        order=9,
    ),
    ScreeningQuestionDef(
        id="Q9",
        state="Q9_MOVE_REASON",
        question="Why are you moving?",
        extract_fields=("move_reason",),
        validation="reason for moving",
        retry_prompt="Could you briefly share why you are moving?",
        order=10,
    ),
    ScreeningQuestionDef(
        id="Q10",
        state="Q10_MOVE_TIMING",
        question="When are you looking to move?",
        extract_fields=("move_timing",),
        validation="move timeframe",
        retry_prompt="When are you looking to move?",
        order=11,
    ),
    ScreeningQuestionDef(
        id="Q11",
        state="Q11_EVICTION",
        question=(
            "Have you ever experienced an eviction or landlord-tenant " "court filing?"
        ),
        extract_fields=("has_eviction", "eviction_raw"),
        validation="yes or no, reviewed individually",
        retry_prompt=(
            "Just to confirm, have you ever had an eviction or "
            "landlord-tenant court filing?"
        ),
        order=12,
    ),
    ScreeningQuestionDef(
        id="Q11A",
        state="Q11A_EVICTION_DETAILS",
        question="Thanks for sharing that. Could you briefly explain the circumstances?",
        extract_fields=("eviction_circumstances", "eviction_raw"),
        validation="short explanation",
        retry_prompt="Could you briefly explain what happened with that filing?",
        order=13,
    ),
    ScreeningQuestionDef(
        id="Q12",
        state="Q12_INCOME",
        question="What is your monthly household income before taxes?",
        extract_fields=("monthly_income", "income_raw"),
        validation="monthly household income before taxes",
        retry_prompt="About how much is your household income each month before taxes?",
        order=14,
    ),
    ScreeningQuestionDef(
        id="Q13",
        state="Q13_EMPLOYER",
        question="Where do you work?",
        extract_fields=("employer",),
        validation="employer or income source",
        retry_prompt="Where are you currently employed, or what is your income source?",
        order=15,
    ),
    ScreeningQuestionDef(
        id="Q14",
        state="Q14_EMPLOYMENT_DURATION",
        question="How long have you been employed there?",
        extract_fields=("employment_duration",),
        validation="employment duration",
        retry_prompt="How long have you been with that employer or income source?",
        order=16,
    ),
    ScreeningQuestionDef(
        id="Q15",
        state="Q15_GENERAL_NOTES",
        question=(
            "Is there anything in your rental, credit, or background history "
            "you want the team to know before proceeding?"
        ),
        extract_fields=("general_notes",),
        validation="any final note, or no",
        retry_prompt=(
            "Anything about your rental, credit, or background history "
            "you want the team to know?"
        ),
        order=17,
    ),
)

DEFAULT_SCREENING_QUESTIONS = [q.as_config() for q in QUESTION_DEFS]
QUESTION_BY_STATE = {q.state: q for q in QUESTION_DEFS}
FLOW_STATE_VALUES = tuple(q.state for q in QUESTION_DEFS)

# Identity questions can never be disabled in the admin UI — without a name,
# phone, and email the screening result is unusable and the read-back/confirm
# flow has nothing to anchor on. validate_questions_for_save forces these active.
PROTECTED_QUESTION_STATES = frozenset(
    {"Q1_FULL_NAME", "Q2_PHONE", "Q3_EMAIL"}
)


def inactive_flow_states(questions: list[dict[str, Any]] | None) -> set[str]:
    """Flow states the admin has switched off (active=False).

    Protected identity states are never returned, even if a stale record marks
    them inactive. Conditional follow-ups (Q6A/Q11A) are governed by their
    parent yes/no, not by this toggle, so callers fold this into the skip set.
    """
    result: set[str] = set()
    for q in questions or []:
        state = str(q.get("state") or "")
        if (
            state in FLOW_STATE_VALUES
            and state not in PROTECTED_QUESTION_STATES
            and not q.get("active", True)
        ):
            result.add(state)
    return result


@dataclass(frozen=True)
class FaqEntryDef:
    """Approved FAQ topic shown during live calls when callers ask questions."""

    id: str
    topic: str
    title: str
    pattern: str
    answer: str
    order: int
    active: bool = True

    def as_config(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "topic": self.topic,
            "title": self.title,
            "pattern": self.pattern,
            "answer": self.answer,
            "active": self.active,
            "order": self.order,
        }


FAQ_DEFS: tuple[FaqEntryDef, ...] = (
    FaqEntryDef(
        id="FAQ1",
        topic="available_properties",
        title="Available properties",
        pattern=r"\b(available|availability|listings?|homes?|properties?)\b",
        answer=(
            "We have several homes available and new listings coming soon. "
            "I can help match you with the best options based on your move-in date, "
            "budget, and household needs."
        ),
        order=1,
    ),
    FaqEntryDef(
        id="FAQ2",
        topic="viewing_scheduling",
        title="Schedule a viewing",
        pattern=r"\b(view|tour|showing|schedule|appointment|see it)\b",
        answer=(
            "Once you complete our quick pre-screening questions, I'll send your "
            "information to our leasing specialist. They will then find the best "
            "possible options to fit your needs."
        ),
        order=2,
    ),
    FaqEntryDef(
        id="FAQ3",
        topic="income_requirements",
        title="Income requirements",
        pattern=r"\b(3x|three times|income requirement|income required|qualify income|min(imum)? income)\b",
        answer=(
            "Our standard requirement is a household income of at least three times "
            "the monthly rent. We also talk with you to figure and fine-tune the "
            "numbers to make sure you get the best possible home to meet your needs."
        ),
        order=3,
    ),
    FaqEntryDef(
        id="FAQ4",
        topic="section_8_vouchers",
        title="Section 8 / housing vouchers",
        pattern=r"\b(section 8|voucher|vouchers|housing assistance)\b",
        answer=(
            "Absolutely, we are here and have been part of the community for over "
            "30 years."
        ),
        order=4,
    ),
    FaqEntryDef(
        id="FAQ5",
        topic="pet_policy",
        title="Pet policy",
        pattern=r"\b(pet[- ]?friendly|allow pets|pet policy|pets allowed)\b",
        answer=(
            "Some properties are pet friendly. I'll just need to know the type, "
            "breed, and weight of your pet to confirm availability. You would need "
            "to meet a few requirements to make sure it's a safe living environment "
            "for everyone."
        ),
        order=5,
    ),
    FaqEntryDef(
        id="FAQ6",
        topic="eviction_policy",
        title="Eviction history",
        pattern=(
            r"\b(eviction|evicted|landlord[- ]tenant|court filing)\b.*\b(disqualify|deny|"
            r"problem|automatic|allowed|review)\b|\b(disqualify|deny|problem|automatic|"
            r"allowed|review)\b.*\b(eviction|evicted|landlord[- ]tenant|court filing)\b"
        ),
        answer=(
            "We do ask about eviction history, but we review each situation "
            "individually. We look at you as a whole person, not just your past."
        ),
        order=6,
    ),
    FaqEntryDef(
        id="FAQ7",
        topic="application_fee",
        title="Application fee",
        pattern=r"\b(application fee|app fee|fee|cost to apply|apply cost)\b",
        answer=(
            "Our application fee is typically around fifty dollars, depending on the "
            "property. This covers credit, background, and verification checks. It "
            "is paid to an independent third-party processor."
        ),
        order=7,
    ),
    FaqEntryDef(
        id="FAQ8",
        topic="documents",
        title="Required documents",
        pattern=r"\b(documents?|paperwork|proof of income|photo id|id needed)\b",
        answer=(
            "You'll need a photo ID, proof of income, and your rental history. "
            "I can send you the application link when you're ready."
        ),
        order=8,
    ),
    FaqEntryDef(
        id="FAQ9",
        topic="approval_time",
        title="Approval timeline",
        pattern=r"\b(approval|approved|how long|24|48|turnaround)\b",
        answer=(
            "Most applications are processed within 24 to 48 hours once all "
            "documents are submitted."
        ),
        order=9,
    ),
    FaqEntryDef(
        id="FAQ10",
        topic="bad_credit",
        title="Bad credit",
        pattern=r"\b(bad credit|credit score|poor credit|credit alone|low credit)\b",
        answer=(
            "We consider the full picture — income, rental history, stability, and "
            "more. Credit alone does not automatically disqualify you."
        ),
        order=10,
    ),
    FaqEntryDef(
        id="FAQ11",
        topic="maintenance",
        title="Emergency maintenance",
        pattern=r"\b(maintenance|repair|emergency)\b",
        answer=(
            "Yes. For emergencies like flooding or safety hazards, we respond "
            "immediately. For non-emergencies, we create a maintenance ticket right "
            "away."
        ),
        order=11,
    ),
    FaqEntryDef(
        id="FAQ12",
        topic="rent_payment",
        title="Rent payment methods",
        pattern=r"\b(pay rent|rent payment|payment methods?|money order|mobile app|pickup)\b",
        answer=(
            "Rent can be paid in several ways: online, check, money order, mobile "
            "apps — we even offer a pickup service."
        ),
        order=12,
    ),
    FaqEntryDef(
        id="FAQ13",
        topic="home_sales",
        title="Home sales",
        pattern=r"\b(home sale|buy a home|sell my home|sales?|purchase)\b",
        answer=(
            "Yes, we offer home sale solutions and can connect you with a specialist "
            "if you're interested in buying or selling."
        ),
        order=13,
    ),
    FaqEntryDef(
        id="FAQ14",
        topic="why_ai",
        title="Why an AI assistant",
        pattern=r"\b(why.*ai|robot|virtual assistant|are you ai|automation)\b",
        answer=(
            "Our virtual assistant helps you quickly get the information you need and "
            "ensures your details reach the right team member. We've been in this "
            "business for over 30 years, and this system helps us serve you faster "
            "and more accurately."
        ),
        order=14,
    ),
    FaqEntryDef(
        id="FAQ15",
        topic="speak_to_person",
        title="Speak to a real person",
        pattern=(
            r"\b(real person|speak to (a )?(person|human|team member|someone)|"
            r"talk to (a )?(person|human|agent|team member|someone))\b"
        ),
        answer=(
            "Of course. If you'd like to speak with a team member, I can forward "
            "your information and have someone reach out as soon as possible."
        ),
        order=15,
    ),
)

DEFAULT_FAQ_ENTRIES = [entry.as_config() for entry in FAQ_DEFS]
FAQ_TOPIC_VALUES = tuple(entry.topic for entry in FAQ_DEFS)


PHONE_RE = re.compile(r"(?:\+?1[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}")
EMAIL_RE = re.compile(r"[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}", re.I)
MONEY_RE = re.compile(
    r"(?P<prefix>\$)?\b(?P<num>\d{2,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(?P<suffix>k|thousand|grand)?\b",
    re.I,
)
WEIGHT_RE = re.compile(r"\b(\d{1,3})\s*(pounds?|lbs?|lb)\b", re.I)
OCCUPANT_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}
PET_TYPES = (
    "dog",
    "cat",
    "bird",
    "fish",
    "rabbit",
    "hamster",
    "reptile",
    "snake",
    "lizard",
)


def normalize_questions(questions: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Use DB questions only when they contain the Ready Rentals flow."""
    if not questions:
        return list(DEFAULT_SCREENING_QUESTIONS)
    states = {str(q.get("state") or "") for q in questions}
    if not set(FLOW_STATE_VALUES).issubset(states):
        return list(DEFAULT_SCREENING_QUESTIONS)
    by_state = {str(q.get("state")): dict(q) for q in questions}
    normalized = []
    for default in DEFAULT_SCREENING_QUESTIONS:
        merged = dict(default)
        merged.update(
            {k: v for k, v in by_state[default["state"]].items() if v is not None}
        )
        normalized.append(merged)
    return normalized


def validate_questions_for_save(
    questions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Validate admin question updates and merge with canonical defaults."""
    if not questions:
        raise ValueError("At least one question is required")
    states = [str(q.get("state") or "") for q in questions]
    if len(states) != len(set(states)):
        raise ValueError("Duplicate question states are not allowed")
    required = set(FLOW_STATE_VALUES)
    got = set(states)
    if got != required:
        missing = sorted(required - got)
        extra = sorted(got - required)
        parts = []
        if missing:
            parts.append(f"missing states: {', '.join(missing)}")
        if extra:
            parts.append(f"unknown states: {', '.join(extra)}")
        raise ValueError("; ".join(parts))
    by_state = {str(q["state"]): dict(q) for q in questions}
    ordered = [by_state[state] for state in FLOW_STATE_VALUES]
    # Identity questions are mandatory — never let them be saved as inactive.
    for q in ordered:
        if str(q.get("state")) in PROTECTED_QUESTION_STATES:
            q["active"] = True
    return normalize_questions(ordered)


def normalize_text(text: str) -> str:
    cleaned = re.sub(r"[^\w\s@.+$'-]", " ", (text or "").lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def normalize_faqs(faqs: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Use DB FAQs only when they contain the full Ready Rentals topic set."""
    if not faqs:
        return list(DEFAULT_FAQ_ENTRIES)
    topics = {str(f.get("topic") or "") for f in faqs}
    if not set(FAQ_TOPIC_VALUES).issubset(topics):
        return list(DEFAULT_FAQ_ENTRIES)
    by_topic = {str(f.get("topic")): dict(f) for f in faqs}
    normalized = []
    for default in DEFAULT_FAQ_ENTRIES:
        merged = dict(default)
        merged.update(
            {k: v for k, v in by_topic[default["topic"]].items() if v is not None}
        )
        normalized.append(merged)
    return normalized


def validate_faqs_for_save(faqs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate admin FAQ updates and merge with canonical defaults."""
    if not faqs:
        raise ValueError("At least one FAQ entry is required")
    topics = [str(f.get("topic") or "") for f in faqs]
    if len(topics) != len(set(topics)):
        raise ValueError("Duplicate FAQ topics are not allowed")
    required = set(FAQ_TOPIC_VALUES)
    got = set(topics)
    if got != required:
        missing = sorted(required - got)
        extra = sorted(got - required)
        parts = []
        if missing:
            parts.append(f"missing topics: {', '.join(missing)}")
        if extra:
            parts.append(f"unknown topics: {', '.join(extra)}")
        raise ValueError("; ".join(parts))
    for entry in faqs:
        pattern = str(entry.get("pattern") or "").strip()
        answer = str(entry.get("answer") or "").strip()
        if not pattern:
            raise ValueError(f"FAQ {entry.get('topic')} is missing a match pattern")
        if not answer:
            raise ValueError(f"FAQ {entry.get('topic')} is missing an answer")
        try:
            re.compile(pattern, re.I)
        except re.error as exc:
            raise ValueError(
                f"FAQ {entry.get('topic')} has invalid regex: {exc}"
            ) from exc
    by_topic = {str(f["topic"]): dict(f) for f in faqs}
    ordered = [by_topic[topic] for topic in FAQ_TOPIC_VALUES]
    return normalize_faqs(ordered)


def parse_yes_no(text: str, *, domain: str = "generic") -> bool | None:
    """Delegate to hybrid intent parser (implicit + correction-aware yes/no)."""
    from app.core.intent import parse_yes_no as _parse_yes_no

    return _parse_yes_no(text, domain=domain)


_SPOKEN_DIGITS = {
    "zero": "0",
    "oh": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}
_DIGIT_REPEAT_WORDS = {"double": 2, "triple": 3}


def _spoken_to_digits(value: str) -> str:
    """Convert spoken number words into digit characters.

    Handles plain words ("zero three one"), embedded digits, and the
    "double"/"triple" repeat patterns ("double five" -> "55").
    """
    tokens = re.split(r"[\s.\-]+", value.lower())
    digits: list[str] = []
    repeat = 1
    for token in tokens:
        if not token:
            continue
        if token in _DIGIT_REPEAT_WORDS:
            repeat = _DIGIT_REPEAT_WORDS[token]
            continue
        digit = _SPOKEN_DIGITS.get(token)
        if digit is None and token.isdigit():
            digit = token
        if digit is not None:
            digits.append(digit * repeat)
        repeat = 1
    return "".join(digits)


def normalize_phone(value: str) -> str | None:
    """Normalize a phone number from typed or spoken digits.

    Accepts US numbers and international/variable-length numbers (E.164 allows
    up to 15 digits). We must never reject a real number the caller gives us —
    that traps the call in an endless "what's your number?" loop.
    """
    if not value:
        return None

    has_plus = "+" in value
    # Prefer the literal digits the caller gave; fall back to spoken digits
    # ("zero three one...") when there are no numeric characters. Using the full
    # digit string (rather than a US-only regex substring) keeps international
    # country codes intact.
    raw_digits = re.sub(r"\D", "", value)
    digits = raw_digits if raw_digits else _spoken_to_digits(value)
    if not digits:
        return None

    # "00" is the international dialing prefix — treat it like a leading "+".
    if digits.startswith("00"):
        digits = digits[2:]
        has_plus = True

    # US: 11 digits with country code "1", or a 10-digit NANP number. NANP
    # area codes and exchanges always start with 2-9 — a 10-digit number that
    # starts with 0 or 1 is a national-format international number, not US, so
    # we must not prepend +1 (that produced bogus values like +10317...).
    if len(digits) == 11 and digits.startswith("1"):
        return f"+1{digits[1:]}"
    if len(digits) == 10 and not has_plus and digits[0] in "23456789":
        return f"+1{digits}"

    # Anything else in the valid E.164 length range: keep it as the caller said.
    if 7 <= len(digits) <= 15:
        return f"+{digits}" if has_plus else digits

    return None


_EMAIL_LEADIN_TOKENS = frozenset(
    {
        "my",
        "email",
        "e-mail",
        "is",
        "it",
        "its",
        "it's",
        "the",
        "address",
        "please",
        "so",
        "well",
        "um",
        "uh",
        "name",
        "contact",
        "use",
        "that's",
        "that",
        "sure",
        "yes",
        "no",
        "actually",
        "sorry",
        "id",
    }
)
_EMAIL_SYMBOL_WORDS = {
    "at": "@",
    "dot": ".",
    "period": ".",
    "underscore": "_",
    "dash": "-",
    "hyphen": "-",
}


_EMAIL_CORRECTION_RE = re.compile(
    r"\b(?:no|nope|nah|actually|sorry|i mean|it'?s|it is|that'?s|that is|"
    r"not right|incorrect|wrong|should be)\b",
    re.I,
)


def parse_spoken_email(value: str) -> str | None:
    """Assemble an email from speech, including spelled-out local parts.

    Handles "a t t a y a b b c at gmail dot com" -> "attayabbc@gmail.com",
    "john dot smith at gmail dot com", partially-typed "A,tt,ayabbc@Gmail.com",
    correction lead-ins ("No, it's not right, X@gmail.com" -> "x@gmail.com"),
    and the "at the rate" / "therate" artifact STT emits for the @ sign.
    """
    if not value:
        return None

    raw = value.lower().strip()

    # "at the rate" is a common spoken form of "@"; Deepgram frequently glues it
    # into "therate". Normalize both so the @ is neither duplicated nor buried.
    raw = re.sub(r"\bat the rate\b", " at ", raw)
    raw = raw.replace("therate", "")

    # If the caller restated after a correction ("no, it's not right, X@…"),
    # keep only the part after the LAST correction marker — but only when that
    # tail actually carries email content, so we never drop a real local part.
    matches = list(_EMAIL_CORRECTION_RE.finditer(raw))
    if matches:
        tail = raw[matches[-1].end() :]
        if "@" in tail or re.search(
            r"\b(at|dot|gmail|yahoo|hotmail|outlook|icloud|proton)\b", tail
        ):
            raw = tail

    # Tokenize on whitespace and commas (commas are STT spelling separators).
    tokens = [t for t in re.split(r"[\s,]+", raw) if t]

    # Strip leading filler ("my email is …"), ignoring punctuation on tokens.
    while tokens and tokens[0].strip(".'\"-_") in _EMAIL_LEADIN_TOKENS:
        tokens.pop(0)

    buf = ""
    for tok in tokens:
        mapped = _EMAIL_SYMBOL_WORDS.get(tok, tok)
        is_symbol = mapped in ("@", ".", "_", "-")
        is_emailish = is_symbol or ("@" in mapped) or ("." in mapped)
        is_single = len(mapped) == 1 and mapped.isalnum()
        is_word = mapped.isalnum() and len(mapped) > 1

        after_at = buf.split("@", 1)[1] if "@" in buf else ""
        if (
            is_word
            and mapped.isalpha()
            and "@" in buf
            and "." in after_at
            and not after_at.endswith(".")
        ):
            # Domain TLD already present (dot after @) — this is trailing prose, stop.
            break
        if is_emailish or is_single or is_word:
            buf += mapped

    # Drop stray leading/trailing separators ("...com." / ".attayab...").
    match = EMAIL_RE.search(buf.strip("._-"))
    if match:
        return match.group(0).lower()
    return None


def normalize_email(value: str) -> str | None:
    """Normalize email from speech patterns like 'at', 'dot', spelled letters."""
    if not value:
        return None
    return parse_spoken_email(value)


def _email_has_local_part(email: str) -> bool:
    """Reject '@gmail.com' and other domain-only fragments."""
    if "@" not in email:
        return False
    local = email.split("@", 1)[0].strip("._-+")
    return len(local) >= 1


def sanitize_stored_email(value: Any) -> str | None:
    """Validate and normalize an email before persisting or read-back."""
    if value in (None, ""):
        return None
    raw = str(value).strip()
    if not raw:
        return None
    parsed = parse_spoken_email(raw)
    if parsed and _email_has_local_part(parsed):
        return parsed.lower()
    if EMAIL_RE.fullmatch(raw) and _email_has_local_part(raw):
        return raw.lower()
    return None


def _looks_iso_date(value: str) -> bool:
    return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", value.strip()))


def normalize_extracted_fields(data: dict[str, Any]) -> dict[str, Any]:
    """Format already-extracted structured values for storage.

    This is the *only* place regex should shape user data: it runs on the
    structured values the LLM (or a fallback parser) has already produced —
    NOT on raw caller speech. Following the standard cascaded pipeline
    (STT -> LLM understanding -> regex normalization -> DB), the LLM decides
    *what* the caller meant; this layer decides *how it is stored*.

    Conservative by design: when a normalizer can't improve a value we keep the
    original rather than dropping it, so we never lose a usable answer. The one
    exception is email, where an unparseable value is rejected outright (a
    malformed address is worse than none for read-back and follow-up).
    """
    if not data:
        return {}
    out = dict(data)

    phone = out.get("contact_phone")
    if phone not in (None, ""):
        normalized = normalize_phone(str(phone))
        if normalized:
            out["contact_phone"] = normalized

    if "email" in out and out["email"] not in (None, ""):
        validated = sanitize_stored_email(out["email"])
        if validated:
            out["email"] = validated
        else:
            out.pop("email")

    income = out.get("monthly_income")
    if income not in (None, "") and not isinstance(income, (int, float, Decimal)):
        normalized = normalize_money(income)
        if normalized is not None:
            out["monthly_income"] = normalized

    move_in = out.get("move_in_date")
    if isinstance(move_in, str) and move_in.strip():
        if not _looks_iso_date(move_in):
            parsed, _raw = parse_relative_date(move_in)
            if parsed is not None:
                out["move_in_date"] = parsed.isoformat()
        else:
            # ISO date, but the LLM sometimes guesses a year in the past for a
            # bare "July 24". Callers never plan a move-in in the past, so roll
            # forward — preferring a re-parse of the caller's exact wording, then
            # falling back to bumping the year to the next future occurrence.
            try:
                parsed_iso = date.fromisoformat(move_in)
            except ValueError:
                parsed_iso = None
            if parsed_iso is not None and parsed_iso < date.today():
                raw_hint = out.get("move_in_raw")
                reparsed = parse_relative_date(raw_hint)[0] if raw_hint else None
                if reparsed is not None and reparsed >= date.today():
                    out["move_in_date"] = reparsed.isoformat()
                else:
                    while parsed_iso < date.today():
                        try:
                            parsed_iso = parsed_iso.replace(year=parsed_iso.year + 1)
                        except ValueError:  # Feb 29 in a non-leap year
                            parsed_iso = parsed_iso.replace(
                                month=2, day=28, year=parsed_iso.year + 1
                            )
                    out["move_in_date"] = parsed_iso.isoformat()

    for count_field in (
        "occupants_count",
        "adults_count",
        "children_count",
        "pet_weight",
    ):
        value = out.get(count_field)
        if value not in (None, "") and not isinstance(value, int):
            coerced = _coerce_count(value)
            if coerced is not None:
                out[count_field] = coerced

    # Pet weight: standardize to pounds. The caller's exact wording in pets_raw is
    # authoritative for the unit (the LLM tends to return the bare number and drop
    # "kg"), so when pets_raw states an explicit weight we re-derive from it. This
    # is idempotent — always computed from the immutable raw text, never from the
    # current pet_weight — so repeated normalization can't double-convert.
    pets_raw = out.get("pets_raw")
    if isinstance(pets_raw, str) and pets_raw.strip():
        pounds = parse_pet_weight_lbs(pets_raw)
        if pounds is not None:
            out["pet_weight"] = pounds

    return out


_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}


def _coerce_count(value: Any) -> int | None:
    """Turn a count value ('2', 'two', 'two people') into an int."""
    match = re.search(r"\d+", str(value))
    if match:
        return int(match.group(0))
    for word in re.findall(r"[a-z]+", str(value).lower()):
        if word in _NUMBER_WORDS:
            return _NUMBER_WORDS[word]
    return None


# ---------------------------------------------------------------------------
# Read-back confirmation for high-stakes fields (name / phone / email)
#
# Speech-to-text is unreliable on proper nouns and number sequences, so the
# standard pattern is to read these fields back and let the caller confirm or
# correct them before moving on. Soft fields (move reason, timing, etc.) are
# trusted as-is to avoid making the call feel like a form.
# ---------------------------------------------------------------------------
CONFIRM_FIELD_BY_STATE: dict[str, str] = {
    "Q1_FULL_NAME": "full_name",
    "Q2_PHONE": "contact_phone",
    "Q3_EMAIL": "email",
}

# Filler/affirmation words that carry no value on their own. Used to tell a
# bare "yes" apart from "Right, it's <a different value>" (a correction that
# happens to start with an affirmation word).
_PURE_AFFIRM_STRIP_RE = re.compile(
    r"\b(yes|yeah|yep|yup|correct|right|sure|exactly|perfect|absolutely|"
    r"definitely|affirmative|ok|okay|good|fine|great|thanks?|thank you|"
    r"that'?s|thats|it'?s|its|it is|that is|this is|is|sounds good|"
    r"you got it|uh[ -]?huh|mm[ -]?hmm|mhm)\b",
    re.I,
)


def is_pure_affirmation(text: str) -> bool:
    """True if the reply is only an affirmation, with no extra content words.

    'Yes.' / 'Yes, that's right.' -> True; 'Right, it's Atayab Ashraf' -> False
    (the trailing name is a correction, not a confirmation).
    """
    t = (text or "").strip().lower()
    if not t:
        return False
    t = _PURE_AFFIRM_STRIP_RE.sub(" ", t)
    t = re.sub(r"[^a-z0-9]+", " ", t).strip()
    return t == ""


def _digits_spaced(value: str) -> str:
    """Render digits for digit-by-digit TTS read-back, grouped 3-3-4."""
    digits = re.sub(r"\D", "", value or "")
    if not digits:
        return value or ""
    # Drop the US country code so a 10-digit number reads back cleanly as
    # 3-3-4 instead of an awkward "1 3 1 7..." with the leading 1.
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    groups: list[str] = []
    i, n = 0, len(digits)
    while i < n:
        size = 3 if (n - i) > 4 else (n - i)
        groups.append(" ".join(digits[i : i + size]))
        i += size
    return ", ".join(groups)


def build_readback(state_value: str, value: str) -> str:
    """Natural read-back prompt for a captured high-stakes field."""
    if state_value == "Q1_FULL_NAME":
        return f"Just to confirm, I have your name as {value}. Did I get that right?"
    if state_value == "Q2_PHONE":
        return (
            "Let me read that back to make sure I have it right — "
            f"{_digits_spaced(value)}. Is that correct?"
        )
    if state_value == "Q3_EMAIL":
        return f"I have your email as {value}. Is that right?"
    return f"I have {value}. Is that correct?"


def repair_prompt(state_value: str) -> str:
    """Prompt used when the caller says the read-back was wrong."""
    if state_value == "Q1_FULL_NAME":
        return "No problem — could you say your full name again, nice and clearly?"
    if state_value == "Q2_PHONE":
        return "No problem — please say your phone number again, one digit at a time."
    if state_value == "Q3_EMAIL":
        return (
            "No problem — could you say your email again slowly? Feel free to spell it."
        )
    return "No problem — could you say that again?"


def normalize_money(value: Any) -> Decimal | None:
    if value in (None, ""):
        return None
    if isinstance(value, Decimal):
        return value.quantize(Decimal("0.01"))
    raw = str(value).replace(",", "").strip().lower()
    multiplier = Decimal("1")
    if raw.endswith("k"):
        multiplier = Decimal("1000")
        raw = raw[:-1]
    elif raw.endswith("thousand") or raw.endswith("grand"):
        multiplier = Decimal("1000")
        raw = re.sub(r"(thousand|grand)$", "", raw).strip()
    raw = raw.replace("$", "").strip()
    try:
        return (Decimal(raw) * multiplier).quantize(Decimal("0.01"))
    except (InvalidOperation, ValueError):
        return None


def extract_money_from_text(text: str) -> tuple[Decimal | None, str | None]:
    match = MONEY_RE.search(text or "")
    if not match:
        return None, None
    amount = match.group("num")
    suffix = match.group("suffix") or ""
    raw = match.group(0)
    normalized = normalize_money(f"{amount}{'k' if suffix.lower() == 'k' else ''}")
    if suffix.lower() in {"thousand", "grand"}:
        normalized = normalize_money(f"{amount} thousand")
    return normalized, raw


def parse_relative_date(
    text: str, today: date | None = None
) -> tuple[date | None, str | None]:
    today = today or date.today()
    raw = (text or "").strip()
    norm = normalize_text(raw)
    if not norm:
        return None, None
    if re.search(r"\b(asap|immediately|right away|now)\b", norm):
        return today, raw
    if "tomorrow" in norm:
        return today + timedelta(days=1), raw
    if "next week" in norm:
        return today + timedelta(days=7), raw
    if "next month" in norm:
        year = today.year + (1 if today.month == 12 else 0)
        month = 1 if today.month == 12 else today.month + 1
        return date(year, month, min(today.day, 28)), raw
    # Numeric dates: MM/DD/YYYY, M-D-YY, MM.DD.YYYY (assume US month-first;
    # if the first field is > 12 it must be day-first). Search the RAW text —
    # normalize_text() turns the "/" separators into spaces.
    numeric = re.search(r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})\b", raw)
    if numeric:
        a, b, c = (
            int(numeric.group(1)),
            int(numeric.group(2)),
            int(numeric.group(3)),
        )
        year = c + 2000 if c < 100 else c
        month, day = (a, b) if a <= 12 else (b, a)
        try:
            return date(year, month, day), raw
        except ValueError:
            return None, raw
    iso = re.search(r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b", norm)
    if iso:
        try:
            return date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3))), raw
        except ValueError:
            return None, raw
    month = re.search(
        r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
        r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
        r"dec(?:ember)?)\.?\s+(\d{1,2})(?:st|nd|rd|th)?(?:,\s*(20\d{2}))?\b",
        norm,
    )
    if month:
        month_names = {
            "jan": 1,
            "feb": 2,
            "mar": 3,
            "apr": 4,
            "may": 5,
            "jun": 6,
            "jul": 7,
            "aug": 8,
            "sep": 9,
            "oct": 10,
            "nov": 11,
            "dec": 12,
        }
        month_num = month_names[month.group(1)[:3]]
        year = int(month.group(3) or today.year)
        try:
            parsed = date(year, month_num, int(month.group(2)))
            if parsed < today and month.group(3) is None:
                parsed = date(year + 1, month_num, int(month.group(2)))
            return parsed, raw
        except ValueError:
            return None, raw
    if re.search(
        r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|spring|summer|fall|winter|month|week)\b",
        norm,
    ):
        return None, raw
    return None, None


def _word_or_digit_count(text: str) -> int | None:
    digit = re.search(r"\b(\d{1,2})\b", text)
    if digit:
        return int(digit.group(1))
    norm = normalize_text(text)
    for word, value in OCCUPANT_WORDS.items():
        if re.search(rf"\b{word}\b", norm):
            return value
    return None


_PARTNER_RE = re.compile(
    r"\b(wife|husband|partner|spouse|girlfriend|boyfriend|fiance|fiancee|"
    r"significant other)\b",
    re.I,
)
_KID_NOUN_RE = re.compile(
    r"\b(son|sons|daughter|daughters|baby|babies|infant|toddler|kid|kids|"
    r"child|children)\b",
    re.I,
)
# Other adult co-occupants people commonly name (counted as additional adults).
_OTHER_ADULT_RE = re.compile(
    r"\b(brother|brothers|sister|sisters|sibling|siblings|"
    r"roommate|roommates|housemate|housemates|flatmate|flatmates|"
    r"friend|friends|mother|father|mom|mum|dad|parent|parents|"
    r"grandmother|grandfather|grandma|grandpa|"
    r"uncle|aunt|cousin|cousins|nephew|niece|"
    r"colleague|coworker|co-worker)\b",
    re.I,
)


def _count_word(token: str) -> int | None:
    if token.isdigit():
        return int(token)
    return OCCUPANT_WORDS.get(token)


def extract_occupants(text: str) -> dict[str, int]:
    """Parse the natural ways people describe a household.

    Handles "just me" (1), "me and my wife" (2), "me, my husband and two kids"
    (4), "family of 4", "there are 3 of us", "five people". Falls back to a bare
    number when no relational phrasing is present.
    """
    norm = normalize_text(text)
    if not norm:
        return {}
    out: dict[str, int] = {}
    num = r"(\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"

    # Explicit children count ("2 kids", "three children").
    m_children = re.search(
        rf"\b{num}\s+(?:children|kids|child|sons?|daughters?)\b", norm
    )
    if m_children:
        c = _count_word(m_children.group(1))
        if c is not None:
            out["children_count"] = c

    # Explicit adults count ("2 adults").
    m_adults = re.search(rf"\b{num}\s+adults?\b", norm)
    if m_adults:
        a = _count_word(m_adults.group(1))
        if a is not None:
            out["adults_count"] = a

    # Explicit total ("family of 4", "3 of us", "5 people/occupants").
    m_total = re.search(rf"\bfamily of\s+{num}\b", norm)
    if not m_total:
        m_total = re.search(
            rf"\b{num}\s+(?:of us|people|persons?|occupants?|total)\b", norm
        )
    if m_total:
        t = _count_word(m_total.group(1))
        if t is not None:
            out["occupants_count"] = t

    # Self + partner + named kids when no explicit adult count was given.
    has_me = bool(
        re.search(r"\b(just me|only me|myself|me and|me my|i live|it'?s me)\b", norm)
        or re.fullmatch(r"(just |only )?me", norm)
    )
    partners = len(_PARTNER_RE.findall(norm))
    # Count other adults, honoring an explicit number before the noun
    # ("two roommates" -> 2, "my brother" -> 1).
    others = 0
    for m in _OTHER_ADULT_RE.finditer(norm):
        preceding = norm[: m.start()].split()
        n = _count_word(preceding[-1]) if preceding else None
        others += n if n else 1
    if "adults_count" not in out:
        adults = (1 if has_me else 0) + partners + others
        if adults:
            out["adults_count"] = adults

    # Named children with no number ("my son", "our daughter and baby").
    if "children_count" not in out:
        kid_hits = len(_KID_NOUN_RE.findall(norm))
        if kid_hits:
            out["children_count"] = kid_hits

    # Derive a total from the parts when none was stated outright.
    if "occupants_count" not in out and (
        "adults_count" in out or "children_count" in out
    ):
        out["occupants_count"] = out.get("adults_count", 0) + out.get(
            "children_count", 0
        )

    return out


# Weight units we accept from speech, mapped to a multiplier that converts the
# stated amount into POUNDS — the single unit the rest of the app and the admin
# UI use. Callers say kg/grams/ounces too, so we normalize everything to lbs and
# keep the caller's exact wording in pets_raw.
_WEIGHT_UNIT_TO_LBS: dict[str, float] = {
    "lb": 1.0,
    "lbs": 1.0,
    "pound": 1.0,
    "pounds": 1.0,
    "kg": 2.2046226218,
    "kgs": 2.2046226218,
    "kilo": 2.2046226218,
    "kilos": 2.2046226218,
    "kilogram": 2.2046226218,
    "kilograms": 2.2046226218,
    "g": 0.0022046226,
    "gram": 0.0022046226,
    "grams": 0.0022046226,
    "oz": 0.0625,
    "ounce": 0.0625,
    "ounces": 0.0625,
}

# Tens words so "forty pounds" / "forty five pounds" parse without digits.
_WEIGHT_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}

_WEIGHT_UNIT_GROUP = (
    r"kilograms?|kilos?|kgs?|kg|grams?|g|ounces?|oz|pounds?|lbs?|lb"
)
_WEIGHT_DIGIT_RE = re.compile(
    rf"(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>{_WEIGHT_UNIT_GROUP})\b", re.I
)
_WEIGHT_WORD_RE = re.compile(
    rf"(?P<words>(?:[a-z]+[\s-]?){{1,3}}?)\s*(?P<unit>{_WEIGHT_UNIT_GROUP})\b", re.I
)


def _weight_words_to_number(phrase: str) -> int | None:
    """Resolve trailing number words ('forty five', 'two') to an int."""
    total = 0
    found = False
    for tok in re.findall(r"[a-z]+", phrase.lower()):
        if tok in _WEIGHT_TENS:
            total += _WEIGHT_TENS[tok]
            found = True
        elif tok in _NUMBER_WORDS:
            total += _NUMBER_WORDS[tok]
            found = True
    return total if found else None


def parse_pet_weight_lbs(text: str) -> int | None:
    """Parse a spoken pet weight and return it in POUNDS, rounded.

    Handles "15 pounds", "2 kg", "about five kg", "forty pounds", "16 oz".
    A bare number with no unit is treated as already in pounds and is NOT parsed
    here (the caller/LLM value is kept as-is by the caller of this function).
    """
    if not text:
        return None
    norm = normalize_text(text)
    amount: float | None = None
    unit: str | None = None

    digit = _WEIGHT_DIGIT_RE.search(norm)
    if digit:
        amount = float(digit.group("num"))
        unit = digit.group("unit").lower()
    else:
        word = _WEIGHT_WORD_RE.search(norm)
        if word:
            amount = _weight_words_to_number(word.group("words"))
            unit = word.group("unit").lower()

    if amount is None or unit is None:
        return None
    factor = _WEIGHT_UNIT_TO_LBS.get(unit)
    if factor is None:
        return None
    pounds = round(amount * factor)
    return pounds if pounds > 0 else None


def extract_pet_fields(text: str) -> dict[str, Any]:
    norm = normalize_text(text)
    out: dict[str, Any] = {}
    for pet_type in PET_TYPES:
        if re.search(rf"\b{pet_type}s?\b", norm):
            out["pet_type"] = pet_type
            break
    pounds = parse_pet_weight_lbs(text)
    if pounds is not None:
        out["pet_weight"] = pounds
    if out:
        out["pets_raw"] = text.strip()
    return out


_BARE_ACK_WORDS = frozenset(
    {
        "yes",
        "yeah",
        "yep",
        "yup",
        "no",
        "nope",
        "correct",
        "right",
        "wrong",
    }
)
_DURATION_HINT_RE = re.compile(
    r"\d|year|month|week|since|ago|long|couple|few|while",
    re.I,
)


def _is_refusal_text(text: str) -> bool:
    from app.core.intent import detect_refusal

    return detect_refusal(text)


def _is_bare_ack(text: str) -> bool:
    return normalize_text(text) in _BARE_ACK_WORDS


def _looks_like_duration(text: str) -> bool:
    return bool(_DURATION_HINT_RE.search(normalize_text(text)))


def _looks_like_residence(text: str) -> bool:
    norm = normalize_text(text)
    if not norm or norm in _BARE_ACK_WORDS:
        return False
    if _is_refusal_text(text):
        return False
    if re.search(
        r"\b(section eight|section 8|handle section|do you accept|faq)\b",
        norm,
    ):
        return False
    if "?" in text and not re.search(
        r"\b(street|avenue|road|drive|lane|apt|apartment|city|town|"
        r"live in|living in|address|house|home)\b",
        norm,
    ):
        return False
    return len(norm) > 2


# Correction / lead-in markers. When a caller restates a value
# ("..., no, it's X" / "actually my name is X"), the real answer is whatever
# follows the LAST marker — everything before it is correction noise.
_NAME_CORRECTION_SPLIT_RE = re.compile(
    r"\b(?:no|nope|nah|actually|sorry|it'?s|it\s+is|that'?s|that\s+is|"
    r"my\s+name\s+is|the\s+name\s+is|full\s+name\s+is|name\s+is|"
    # "this is" only counts as a restatement when a name plausibly follows —
    # not "this is my phone line" / "this is the number".
    r"this\s+is(?!\s+(?:my|a|an|the|not|just|our|your|his|her)\b)|" r"i'?m|i\s+am)\b",
    re.I,
)

# Multi-character filler tokens to drop from a parsed name. Includes words that
# never appear in a real name in this context ("phone", "line", "number") so a
# frustrated aside like "this is my phone line" doesn't become a name.
_NAME_FILLER_TOKENS = frozenset(
    {
        "the",
        "an",
        "is",
        "it",
        "its",
        "that",
        "name",
        "my",
        "this",
        "am",
        "so",
        "well",
        "um",
        "uh",
        "hmm",
        "actually",
        "yes",
        "no",
        "yeah",
        "yep",
        "nope",
        "nah",
        "ok",
        "okay",
        "right",
        "correct",
        "wrong",
        "sure",
        "please",
        "like",
        "sorry",
        "and",
        "said",
        "say",
        "phone",
        "line",
        "number",
        "mobile",
        "cell",
        "call",
        "calling",
    }
)


def _assemble_spelled_letters(tokens: list[str]) -> list[str]:
    """Merge runs of >=3 single letters into one spelled word.

    "a t t a y a b" -> "attayab"; supports "double t" / "triple a". Runs of
    1-2 single letters are kept as initials (e.g. a middle initial).
    """
    out: list[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        is_single = len(tok) == 1 and tok.isalpha()
        is_repeat = tok.lower() in ("double", "triple")
        if is_single or (
            is_repeat
            and i + 1 < n
            and len(tokens[i + 1]) == 1
            and tokens[i + 1].isalpha()
        ):
            run: list[str] = []
            j = i
            while j < n:
                t = tokens[j]
                if (
                    t.lower() in ("double", "triple")
                    and j + 1 < n
                    and (len(tokens[j + 1]) == 1 and tokens[j + 1].isalpha())
                ):
                    rep = 2 if t.lower() == "double" else 3
                    run.append(tokens[j + 1] * rep)
                    j += 2
                    continue
                if len(t) == 1 and t.isalpha():
                    run.append(t)
                    j += 1
                    continue
                break
            letters = "".join(run)
            if len(letters) >= 3:
                out.append(letters)
                i = j
                continue
        out.append(tok)
        i += 1
    return out


def parse_spoken_name(text: str) -> str:
    """Robustly parse a spoken full name.

    Handles correction lead-ins ("no, it's…"), filler ("my name is…"),
    STT question-mark artifacts, and spelled-out letters
    ("Mohammed a-t-t-a-y-a-b" -> "Mohammed Attayab").
    """
    if not text:
        return ""
    s = text.replace("?", " ").strip()
    if not s:
        return ""

    # If the caller restated, keep the part after the LAST correction marker.
    matches = list(_NAME_CORRECTION_SPLIT_RE.finditer(s))
    if matches:
        tail = s[matches[-1].end() :]
        if tail.strip():
            s = tail

    # Keep letters, apostrophes and hyphens; turn everything else into spaces
    # so "a, t, t" becomes individual letter tokens we can reassemble.
    cleaned = re.sub(r"[^A-Za-z'\-\s]", " ", s)
    raw_tokens = [t for t in re.split(r"\s+", cleaned) if t]
    # Expand hyphen-spelled tokens ("a-t-t-a-y-a-b") into individual letters,
    # but leave real hyphenated names ("Jean-Pierre") intact.
    tokens: list[str] = []
    for t in raw_tokens:
        if re.fullmatch(r"[A-Za-z](?:-[A-Za-z])+", t):
            tokens.extend(t.split("-"))
        else:
            tokens.append(t)
    tokens = _assemble_spelled_letters(tokens)

    result: list[str] = []
    for tok in tokens:
        low = tok.lower()
        if low in _NAME_FILLER_TOKENS:
            continue
        if len(tok) == 1 and tok.isalpha():
            # Leftover single letter: drop articles/pronoun, keep real initials.
            if low in ("a", "i"):
                continue
            result.append(tok.upper())
            continue
        result.append(tok[:1].upper() + tok[1:].lower())

    return " ".join(result).strip()


def extract_fields_from_text(
    text: str,
    current_state: str,
    existing_data: dict[str, Any] | None = None,
    *,
    intent: Any | None = None,
) -> dict[str, Any]:
    """Best-effort deterministic extraction for common voice answers."""
    existing_data = existing_data or {}
    out: dict[str, Any] = {}
    stripped = (text or "").strip()
    norm = normalize_text(stripped)
    if not stripped:
        return out

    if current_state == "Q3_EMAIL":
        # On the email question, always run the spoken-email parser first so a
        # spelled local part ("a t t a y a b b c at gmail dot com") or a
        # comma-broken transcript ("A,tt,ayabbc@gmail.com") is assembled in full
        # rather than a bare EMAIL_RE grab dropping the spelled prefix.
        spoken_email = normalize_email(stripped)
        if spoken_email:
            out["email"] = spoken_email
    else:
        email = EMAIL_RE.search(stripped)
        if email:
            out["email"] = email.group(0).lower()

    # Only mine for a phone number on the phone question (or when a clearly
    # phone-shaped token appears); otherwise dates/amounts/IDs in other answers
    # could be mistaken for a phone number.
    if current_state == "Q2_PHONE" or PHONE_RE.search(stripped):
        phone = normalize_phone(stripped)
        if phone:
            out["contact_phone"] = phone

    if current_state == "Q1_FULL_NAME":
        if not EMAIL_RE.search(stripped):
            name = parse_spoken_name(stripped)
            # Require at least two parts for a full name, OR a single spelled-out
            # word long enough to be a real name (e.g. just a surname correction).
            parts = name.split()
            if len(parts) >= 2 or (len(parts) == 1 and len(parts[0]) >= 3):
                out["full_name"] = name

    if current_state == "Q4_MOVE_IN_DATE":
        parsed, raw = parse_relative_date(stripped)
        if raw:
            out["move_in_raw"] = raw
        if parsed:
            out["move_in_date"] = parsed.isoformat()
        # Flexible / undecided move-in is still a valid answer ("not sure yet",
        # "haven't planned", "as soon as possible"). Keep the raw text so the
        # question counts as answered instead of looping on a retry.
        if not raw and not _is_refusal_text(stripped) and len(stripped.split()) >= 2:
            out["move_in_raw"] = stripped

    if current_state == "Q5_OCCUPANTS":
        natural = extract_occupants(stripped)
        out.update(natural)
        # Only fall back to a bare number when the natural parser found nothing,
        # so "me and my two kids" counts as 3 rather than being read as 2.
        if not natural:
            count = _word_or_digit_count(stripped)
            if count is not None:
                out["occupants_count"] = count

    if current_state == "Q6_PETS":
        domain = "pets"
        yn = (
            intent.yes_no
            if intent is not None and intent.yes_no is not None
            else parse_yes_no(stripped, domain=domain)
        )
        if yn is not None:
            out["has_pets"] = yn
            out["pets_raw"] = stripped
        if yn is True or any(p in norm for p in PET_TYPES):
            out.update(extract_pet_fields(stripped))

    if current_state == "Q6A_PET_DETAILS":
        if not _is_bare_ack(stripped) and not _is_refusal_text(stripped):
            out.update(extract_pet_fields(stripped))
            if stripped:
                out.setdefault("pets_raw", stripped)

    if current_state == "Q7_CURRENT_RESIDENCE":
        if _looks_like_residence(stripped):
            out["current_residence"] = stripped

    if current_state == "Q8_RESIDENCE_DURATION":
        if (
            not _is_bare_ack(stripped)
            and not _is_refusal_text(stripped)
            and _looks_like_duration(stripped)
        ):
            out["residence_duration"] = stripped

    if current_state == "Q9_MOVE_REASON":
        if not _is_bare_ack(stripped) and not _is_refusal_text(stripped):
            if not re.search(r"\b(section eight|section 8|faq)\b", norm):
                out["move_reason"] = stripped

    if current_state == "Q10_MOVE_TIMING":
        if not _is_bare_ack(stripped) and not _is_refusal_text(stripped):
            parsed, raw = parse_relative_date(stripped)
            out["move_timing"] = raw or stripped

    if current_state == "Q11_EVICTION":
        domain = "eviction"
        yn = (
            intent.yes_no
            if intent is not None and intent.yes_no is not None
            else parse_yes_no(stripped, domain=domain)
        )
        if yn is not None:
            out["has_eviction"] = yn
            out["eviction_raw"] = stripped

    if current_state == "Q11A_EVICTION_DETAILS":
        if (
            not _is_bare_ack(stripped)
            and not _is_refusal_text(stripped)
            and len(norm) > 8
        ):
            out["eviction_circumstances"] = stripped
            out["eviction_raw"] = stripped

    if current_state == "Q12_INCOME":
        monthly, raw = extract_money_from_text(stripped)
        is_hourly = bool(re.search(r"\b(hour|hourly|per hour|an hour|/hr|hr)\b", norm))
        if raw:
            out["income_raw"] = stripped
        if monthly is not None and not is_hourly:
            out["monthly_income"] = monthly
        # A forthcoming caller who states pay in words ("about three thousand a
        # month", "twenty an hour") or describes it without a clean number still
        # answered the question. Preserve the wording so it counts as answered
        # (post-call parsing fills monthly_income) instead of looping on income.
        if "income_raw" not in out and (
            is_hourly
            or re.search(
                r"\b(thousand|grand|salary|wage|income|make|makes|earn|earns|"
                r"paid|pay|annually|annual|year|yearly|month|monthly|week|"
                r"weekly|biweekly|bi-?weekly|hundred|\bk\b)\b",
                norm,
            )
        ):
            out["income_raw"] = stripped

    if current_state == "Q13_EMPLOYER":
        employer = re.sub(
            r"\b(i work at|i work for|work at|work for|employer is)\b",
            "",
            stripped,
            flags=re.I,
        )
        out["employer"] = employer.strip(" .") or stripped

    if current_state == "Q14_EMPLOYMENT_DURATION":
        if (
            not _is_bare_ack(stripped)
            and not _is_refusal_text(stripped)
            and _looks_like_duration(stripped)
        ):
            out["employment_duration"] = stripped

    if current_state == "Q15_GENERAL_NOTES":
        yn = (
            intent.yes_no
            if intent is not None and intent.yes_no is not None
            else parse_yes_no(stripped)
        )
        # Natural ways people signal "I have nothing to add" — treat the same as
        # a plain "no" so they aren't bounced to the LLM or asked again.
        done = bool(
            re.search(
                r"\b(nothing else|nothing more|nothing to add|that'?s all|"
                r"that'?s it|that is all|that is it|that'?s everything|"
                r"that'?s about it|all good|i'?m good|we'?re good|"
                r"no that'?s (it|all|everything)|i think that'?s it)\b",
                norm,
            )
        )
        if yn is False or done:
            out["general_notes"] = "None disclosed"
        elif yn is True and is_pure_affirmation(stripped):
            # "Yes" means they DO have something to share but haven't said what
            # yet — leave unanswered so the flow asks them to elaborate.
            pass
        else:
            out["general_notes"] = stripped

    return {k: v for k, v in out.items() if v not in (None, "")}


def is_skip_state(state: str, data: dict[str, Any]) -> bool:
    # Detail follow-ups only apply when the parent yes/no was affirmatively yes.
    if state == "Q6A_PET_DETAILS":
        return data.get("has_pets") is not True
    if state == "Q11A_EVICTION_DETAILS":
        return data.get("has_eviction") is not True
    return False


def _has_value(data: dict[str, Any], *fields: str) -> bool:
    return any(data.get(field) not in (None, "", []) for field in fields)


def is_question_answered(
    state: str, data: dict[str, Any], refused_states: Iterable[str] | None = None
) -> bool:
    refused = set(refused_states or [])
    if state in refused:
        return True
    if is_skip_state(state, data):
        return True
    if state == "Q1_FULL_NAME":
        return _has_value(data, "full_name")
    if state == "Q2_PHONE":
        return _has_value(data, "contact_phone")
    if state == "Q3_EMAIL":
        return _has_value(data, "email")
    if state == "Q4_MOVE_IN_DATE":
        return _has_value(data, "move_in_date", "move_in_raw")
    if state == "Q5_OCCUPANTS":
        return _has_value(data, "occupants_count", "adults_count")
    if state == "Q6_PETS":
        return data.get("has_pets") in (True, False)
    if state == "Q6A_PET_DETAILS":
        return (
            _has_value(data, "pet_type")
            and _has_value(data, "pet_breed", "pets_raw")
            and _has_value(data, "pet_weight", "pets_raw")
        )
    if state == "Q7_CURRENT_RESIDENCE":
        return _has_value(data, "current_residence")
    if state == "Q8_RESIDENCE_DURATION":
        return _has_value(data, "residence_duration")
    if state == "Q9_MOVE_REASON":
        return _has_value(data, "move_reason")
    if state == "Q10_MOVE_TIMING":
        return _has_value(data, "move_timing")
    if state == "Q11_EVICTION":
        return data.get("has_eviction") in (True, False)
    if state == "Q11A_EVICTION_DETAILS":
        # Must have the circumstances specifically — the parent Q11 also writes
        # eviction_raw, so sharing it here would skip this follow-up entirely.
        return _has_value(data, "eviction_circumstances")
    if state == "Q12_INCOME":
        return _has_value(data, "monthly_income", "income_raw")
    if state == "Q13_EMPLOYER":
        return _has_value(data, "employer")
    if state == "Q14_EMPLOYMENT_DURATION":
        return _has_value(data, "employment_duration")
    if state == "Q15_GENERAL_NOTES":
        return _has_value(data, "general_notes")
    return False


def next_unanswered_state(
    data: dict[str, Any],
    refused_states: Iterable[str] | None = None,
) -> str | None:
    for state in FLOW_STATE_VALUES:
        if not is_question_answered(state, data, refused_states):
            return state
    return None


def count_answered_questions(
    data: dict[str, Any],
    refused_states: Iterable[str] | None = None,
) -> int:
    return sum(
        1
        for state in FLOW_STATE_VALUES
        if not is_skip_state(state, data)
        and is_question_answered(state, data, refused_states)
    )


def count_active_questions(
    data: dict[str, Any], skip_states: Iterable[str] | None = None
) -> int:
    """Count questions that are actually in play for this call.

    ``skip_states`` (refused / admin-disabled / LLM-completed) are excluded so
    the progress denominator matches the questions the caller is really asked.
    """
    skip = set(skip_states or [])
    return sum(
        1
        for state in FLOW_STATE_VALUES
        if not is_skip_state(state, data) and state not in skip
    )


def screening_complete(
    data: dict[str, Any],
    refused_states: Iterable[str] | None = None,
) -> bool:
    return next_unanswered_state(data, refused_states) is None


_TRANSITIONS = (
    "Got it.",
    "Thanks, that helps.",
    "Perfect.",
    "Thanks for sharing that.",
    "Great, thank you.",
    "Okay, noted.",
    "Wonderful.",
    "Alright.",
    "Sounds good.",
)


def brief_transition(_answered_count: int = 0) -> str:
    """Return a short, warm acknowledgment.

    Picked at random from a small pool so the agent doesn't repeat the same
    phrase every turn — small touches like this are what make it feel human.
    The deterministic question flow is unaffected (only the ack wording varies).
    """
    return random.choice(_TRANSITIONS)


# State Machine Transition Map
STATE_TRANSITIONS = {
    "IDLE": ["GREETING"],
    "GREETING": ["Q1_FULL_NAME"],
    "Q1_FULL_NAME": ["Q2_PHONE"],
    "Q2_PHONE": ["Q3_EMAIL"],
    "Q3_EMAIL": ["Q4_MOVE_IN_DATE"],
    "Q4_MOVE_IN_DATE": ["Q5_OCCUPANTS"],
    "Q5_OCCUPANTS": ["Q6_PETS"],
    "Q6_PETS": ["Q6A_PET_DETAILS", "Q7_CURRENT_RESIDENCE"],
    "Q6A_PET_DETAILS": ["Q7_CURRENT_RESIDENCE"],
    "Q7_CURRENT_RESIDENCE": ["Q8_RESIDENCE_DURATION"],
    "Q8_RESIDENCE_DURATION": ["Q9_MOVE_REASON"],
    "Q9_MOVE_REASON": ["Q10_MOVE_TIMING"],
    "Q10_MOVE_TIMING": ["Q11_EVICTION"],
    "Q11_EVICTION": ["Q11A_EVICTION_DETAILS", "Q12_INCOME"],
    "Q11A_EVICTION_DETAILS": ["Q12_INCOME"],
    "Q12_INCOME": ["Q13_EMPLOYER"],
    "Q13_EMPLOYER": ["Q14_EMPLOYMENT_DURATION"],
    "Q14_EMPLOYMENT_DURATION": ["Q15_GENERAL_NOTES"],
    "Q15_GENERAL_NOTES": ["WRAP_UP"],
    "WRAP_UP": ["ENDED"],
    "ENDED": [],
}


def validate_state_transition(from_state: str, to_state: str) -> bool:
    """Validate that a state transition is allowed."""
    if from_state not in STATE_TRANSITIONS:
        return False
    return to_state in STATE_TRANSITIONS[from_state]


def log_state_transition(
    call_id: str,
    from_state: str,
    to_state: str,
    reason: str,
    retry_count: int = 0,
) -> None:
    """Log state transition with context."""
    import logging

    logger = logging.getLogger(__name__)

    is_valid = validate_state_transition(from_state, to_state)
    status = "VALID" if is_valid else "INVALID"

    logger.info(
        f"[{call_id}] STATE CHANGE: {from_state} → {to_state} "
        f"| Status: {status} | Reason: {reason} | Retry: {retry_count}"
    )

    if not is_valid:
        logger.warning(
            f"[{call_id}] INVALID STATE TRANSITION DETECTED: "
            f"{from_state} → {to_state} not in allowed transitions"
        )
