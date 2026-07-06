"""Speech parsing and field-normalization helpers for live screening calls.

Question order, FAQ content, and scoring rules come from admin settings in the
database (frozen per call). This module only provides shared parsers (phone,
email, money, pets, etc.).
"""

from __future__ import annotations

import logging
import random
import re
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any

logger = logging.getLogger(__name__)


BUSINESS_NAME = "Ready Rentals Online"


def build_greeting_intro(business: str = BUSINESS_NAME, *, language_code: str = "en") -> str:
    """Short default opening when admin has not set a custom greeting."""
    name = (business or "").strip() or BUSINESS_NAME
    if str(language_code).lower().startswith("es"):
        return (
            f"Hola y gracias por llamar a {name}. "
            "Soy su asistente virtual. Empecemos."
        )
    return (
        f"Hello and thank you for calling {name}! "
        "I'm your virtual assistant. Let's get started!"
    )


PHONE_RE = re.compile(r"(?:\+?1[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}")
EMAIL_RE = re.compile(r"[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}", re.I)
MONEY_RE = re.compile(
    r"(?P<prefix>\$)?\b(?P<num>\d{2,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(?P<suffix>k|thousand|grand)?\b",
    re.I,
)
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


def normalize_text(text: str) -> str:
    cleaned = re.sub(r"[^\w\s@.+$'-]", " ", (text or "").lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def normalize_faqs(faqs: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Normalize admin FAQ entries from the database."""
    if not faqs:
        return []
    normalized: list[dict[str, Any]] = []
    for entry in faqs:
        item = dict(entry)
        item.setdefault("active", True)
        item.setdefault("order", len(normalized) + 1)
        normalized.append(item)
    normalized.sort(key=lambda x: int(x.get("order") or 0))
    for idx, item in enumerate(normalized, start=1):
        item["order"] = idx
    return normalized


def validate_faqs_for_save(faqs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate admin FAQ updates."""
    if not faqs:
        raise ValueError("At least one FAQ entry is required")
    topics = [str(f.get("topic") or "").strip() for f in faqs]
    if any(not topic for topic in topics):
        raise ValueError("Each FAQ entry needs a topic id")
    if len(topics) != len(set(topics)):
        raise ValueError("Duplicate FAQ topics are not allowed")
    for entry in faqs:
        pattern = str(entry.get("pattern") or "").strip()
        answer = str(entry.get("answer") or "").strip()
        title = str(entry.get("title") or "").strip()
        if not title:
            raise ValueError(f"FAQ {entry.get('topic')} is missing a title")
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
    return normalize_faqs(faqs)


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


def roll_future_date(d: date | None, today: date | None = None) -> date | None:
    """Roll a past screening date forward to the next sensible future occurrence.

    Callers planning a move rarely intend a date in the past; small LLMs often
    guess a wrong year. Applies to any admin ``answer_type: date`` field.
    """
    if d is None:
        return None
    today = today or date.today()
    if d >= today:
        return d
    for year in (today.year, today.year + 1):
        try:
            candidate = d.replace(year=year)
        except ValueError:
            candidate = d.replace(year=year, month=2, day=28)
        if candidate >= today:
            return candidate
    return d


def _normalize_date_field_value(
    field: str,
    value: Any,
    out: dict[str, Any],
    *,
    today: date | None = None,
) -> None:
    """Parse, store, and roll-forward a single date-type field."""
    if value in (None, ""):
        return
    today = today or date.today()
    if isinstance(value, str) and value.strip():
        if not _looks_iso_date(value):
            parsed, _raw = parse_relative_date(value)
            if parsed is not None:
                out[field] = roll_future_date(parsed, today).isoformat()
                return
        try:
            parsed_iso = date.fromisoformat(str(value)[:10])
        except ValueError:
            out[field] = str(value).strip()
            return
        rolled = roll_future_date(parsed_iso, today)
        if rolled is not None:
            out[field] = rolled.isoformat()
    elif isinstance(value, date) and not isinstance(value, datetime):
        rolled = roll_future_date(value, today)
        if rolled is not None:
            out[field] = rolled.isoformat()


def normalize_extracted_fields(
    data: dict[str, Any],
    questions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
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
    if out.get("monthly_income") in (None, ""):
        raw_income = out.get("income_raw")
        if isinstance(raw_income, str) and raw_income.strip():
            inferred = infer_monthly_income_from_raw(raw_income)
            if inferred is not None:
                out["monthly_income"] = inferred

    # Coerce known yes/no fields to real booleans. The LLM sometimes emits the
    # strings "yes"/"no"/"true" instead of a JSON boolean; left as strings they
    # break boolean checks ("no" is truthy in Python) used by skip/conditional
    # logic and scoring. Any field prefixed has_/is_ is treated as a flag.
    for key in list(out.keys()):
        if (key.startswith("has_") or key.startswith("is_")) and not key.endswith(
            "_raw"
        ):
            coerced = _coerce_bool(out.get(key))
            if coerced is not None:
                out[key] = coerced

    # *_raw columns store verbatim caller wording (VARCHAR). Drop bool copies of
    # yes/no flags; coerce numeric duplicates to plain strings.
    for key in list(out.keys()):
        if not str(key).endswith("_raw"):
            continue
        val = out.get(key)
        if val in (None, ""):
            continue
        if isinstance(val, bool):
            out.pop(key, None)
        elif isinstance(val, (int, float, Decimal)):
            out[key] = str(val)
        elif not isinstance(val, str):
            text = str(val).strip()
            if text:
                out[key] = text
            else:
                out.pop(key, None)

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

    if questions:
        from app.core.question_flow import (
            field_answer_types_from_questions,
            normalize_questions,
            speech_mode_for_question,
        )

        for q in normalize_questions(questions):
            if speech_mode_for_question(q) != "pet_bundle":
                continue
            fields = [str(f) for f in (q.get("extract_fields") or [])]
            raw_key = next((f for f in fields if f.endswith("_raw")), None)
            weight_key = next(
                (f for f in fields if "weight" in f.lower() and not f.endswith("_raw")),
                None,
            )
            if not raw_key or not weight_key:
                continue
            raw_text = out.get(raw_key)
            if isinstance(raw_text, str) and raw_text.strip():
                pounds = parse_pet_weight_lbs(raw_text)
                if pounds is not None:
                    out[weight_key] = pounds

        for q in normalize_questions(questions):
            if str(q.get("answer_type") or "") != "currency":
                continue
            fields = [str(f) for f in (q.get("extract_fields") or [])]
            primary = next((f for f in fields if not f.endswith("_raw")), None)
            raw_key = next((f for f in fields if f.endswith("_raw")), None)
            if not primary or not raw_key:
                continue
            if out.get(primary) not in (None, ""):
                continue
            raw_val = out.get(raw_key)
            if isinstance(raw_val, str) and raw_val.strip():
                inferred = infer_monthly_income_from_raw(raw_val)
                if inferred is not None:
                    out[primary] = inferred

        for field, answer_type in field_answer_types_from_questions(questions).items():
            if str(field).endswith("_raw"):
                continue
            value = out.get(field)
            if value in (None, ""):
                continue
            if answer_type == "phone":
                normalized = normalize_phone(str(value))
                if normalized:
                    out[field] = normalized
            elif answer_type == "email":
                validated = sanitize_stored_email(value)
                if validated:
                    out[field] = validated
                else:
                    out.pop(field, None)
            elif answer_type == "currency" and not isinstance(
                value, (int, float, Decimal)
            ):
                normalized = normalize_money(value)
                if normalized is not None:
                    out[field] = normalized
            elif answer_type == "number" and not isinstance(value, int):
                coerced = _coerce_count(value)
                if coerced is not None:
                    out[field] = coerced
            elif answer_type == "yes_no" or field.startswith(("has_", "is_")):
                coerced = _coerce_bool(value)
                if coerced is not None:
                    out[field] = coerced
            elif answer_type == "date":
                _normalize_date_field_value(field, value, out)
    else:
        for field in ("move_in_date", "move_timing"):
            if field in out:
                _normalize_date_field_value(field, out.get(field), out)

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

_OCCUPANT_WORD_KEYS = (
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
)
OCCUPANT_WORDS = {word: _NUMBER_WORDS[word] for word in _OCCUPANT_WORD_KEYS}


def _coerce_count(value: Any) -> int | None:
    """Turn a count value ('2', 'two', 'two people') into an int."""
    match = re.search(r"\d+", str(value))
    if match:
        return int(match.group(0))
    for word in re.findall(r"[a-z]+", str(value).lower()):
        if word in _NUMBER_WORDS:
            return _NUMBER_WORDS[word]
    return None


def _coerce_bool(value: Any) -> bool | None:
    """Normalize a yes/no flag value to a real boolean.

    Returns None when the value carries no clear affirmative/negative signal so
    the caller can decide whether to leave it untouched.
    """
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return None
    text = str(value).strip().lower()
    if text in {"true", "yes", "y", "1", "yeah", "yep", "correct", "affirmative"}:
        return True
    if text in {"false", "no", "n", "0", "nope", "none", "negative"}:
        return False
    parsed = parse_yes_no(text)
    return parsed


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


# STT often turns "100k" into "100,000 k" — treat that pattern as hundreds + k.
_STT_HUNDREDS_K_RE = re.compile(r"\b(\d{1,3}),\d{3}\s*k\b", re.I)
_ANNUAL_INCOME_RE = re.compile(
    r"\b(per\s+year|yearly|annually|annual(?:ly)?|a\s+year)\b", re.I
)
_MONTHLY_INCOME_RE = re.compile(
    r"\b(per\s+month|monthly|a\s+month|/mo|each\s+month)\b", re.I
)


def _clean_income_raw_for_parse(text: str) -> str:
    """Strip common STT noise before money extraction."""
    cleaned = (text or "").strip().lower()
    cleaned = _STT_HUNDREDS_K_RE.sub(lambda m: f"{m.group(1)}k", cleaned)
    cleaned = re.sub(r"\bmeans\b", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def infer_monthly_income_from_raw(income_raw: str) -> Decimal | None:
    """Derive monthly_income when the LLM captured wording but not a number."""
    cleaned = _clean_income_raw_for_parse(income_raw)
    if not cleaned:
        return None
    amount, _ = extract_money_from_text(cleaned)
    if amount is None:
        amount = normalize_money(cleaned)
    if amount is None:
        return None
    if _ANNUAL_INCOME_RE.search(cleaned):
        return (amount / Decimal("12")).quantize(Decimal("0.01"))
    if _MONTHLY_INCOME_RE.search(cleaned):
        return amount
    # No period stated — treat as monthly (matches in-call extraction guidance).
    return amount


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


def _is_refusal_text(text: str) -> bool:
    from app.core.intent import detect_refusal

    return detect_refusal(text)


def _is_bare_ack(text: str) -> bool:
    return normalize_text(text) in _BARE_ACK_WORDS


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


def _has_value(data: dict[str, Any], *fields: str) -> bool:
    return any(data.get(field) not in (None, "", []) for field in fields)


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


# Meta-state transitions only — screening question order is admin-driven.
_META_STATE_TRANSITIONS = {
    "IDLE": frozenset({"GREETING"}),
    "WRAP_UP": frozenset({"ENDED"}),
    "ENDED": frozenset({"ENDED"}),
}


def validate_state_transition(
    from_state: str,
    to_state: str,
    questions: list[dict[str, Any]] | None = None,
) -> bool:
    """Validate a state transition against meta states and admin question flow."""
    if from_state == "ENDED" and to_state != "ENDED":
        return False
    allowed = _META_STATE_TRANSITIONS.get(from_state)
    if allowed is not None:
        return to_state in allowed
    if from_state == "GREETING":
        from app.core.question_flow import (
            first_active_question_state,
            flow_states_in_order,
        )

        first = first_active_question_state(questions)
        valid = set(flow_states_in_order(questions)) | {"WRAP_UP", "ENDED"}
        if first:
            valid.add(first)
        return to_state in valid
    return True


def log_state_transition(
    call_id: str,
    from_state: str,
    to_state: str,
    reason: str,
    retry_count: int = 0,
    *,
    questions: list[dict[str, Any]] | None = None,
) -> None:
    """Log state transition with context."""
    is_valid = validate_state_transition(from_state, to_state, questions)
    status = "VALID" if is_valid else "INVALID"

    logger.info(
        f"[{call_id}] STATE CHANGE: {from_state} → {to_state} "
        f"| Status: {status} | Reason: {reason} | Retry: {retry_count}"
    )

    if not is_valid:
        logger.warning(
            f"[{call_id}] INVALID STATE TRANSITION DETECTED: "
            f"{from_state} → {to_state}"
        )
