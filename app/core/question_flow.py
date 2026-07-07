"""Dynamic screening question flow (schema v2).

Admins may add, delete, reorder, and score questions. Navigation, completion,
and skip logic are driven by the stored question list instead of a fixed
17-state machine.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Iterable
from decimal import Decimal, InvalidOperation
from typing import Any

from app.core.screening_flow import (
    _has_value,
    normalize_email,
    normalize_phone,
)

SCHEMA_VERSION = 2

ANSWER_TYPES = frozenset(
    {
        "text",
        "long_text",
        "yes_no",
        "number",
        "currency",
        "date",
        "phone",
        "email",
        "language_choice",
    }
)

DEFAULT_LANGUAGE_OPTIONS: tuple[dict[str, Any], ...] = (
    {
        "value": "en",
        "label": "English",
        "aliases": ["english", "inglés", "ingles", "en"],
    },
    {
        "value": "es",
        "label": "Español",
        "aliases": ["spanish", "español", "espanol", "es"],
    },
)


def default_language_options() -> list[dict[str, Any]]:
    return [dict(o) for o in DEFAULT_LANGUAGE_OPTIONS]


def canonical_language_code(value: Any) -> str | None:
    """Normalize admin / caller language tokens to ``en`` or ``es``."""
    raw = str(value or "").strip().lower()
    if not raw:
        return None
    cleaned = re.sub(r"[^a-z-]", "", raw)
    if cleaned.startswith("es") or cleaned in {"spanish", "espanol", "español", "spa"}:
        return "es"
    if cleaned.startswith("en") or cleaned in {"english", "ingles"}:
        return "en"
    return None


def normalize_language_options(raw: Any) -> list[dict[str, Any]]:
    """Return at least two unique language options for a language_choice question."""
    if not isinstance(raw, list) or not raw:
        return default_language_options()
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        code = canonical_language_code(item.get("value")) or str(
            item.get("value") or ""
        ).strip().lower()
        if not code or code in seen:
            continue
        seen.add(code)
        aliases = [
            str(alias).strip()
            for alias in (item.get("aliases") or [])
            if str(alias).strip()
        ]
        label = str(item.get("label") or code).strip() or code
        out.append({"value": code, "label": label, "aliases": aliases})
    return out if len(out) >= 2 else default_language_options()


def normalize_question_locales(raw: Any) -> dict[str, dict[str, str]]:
    """Normalize optional per-language question wording overrides."""
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict[str, str]] = {}
    for lang, entries in raw.items():
        code = canonical_language_code(lang) or str(lang or "").strip().lower()
        if not code or not isinstance(entries, dict):
            continue
        cleaned: dict[str, str] = {}
        for key in ("question", "retry_prompt", "retry_prompt_2", "retry_prompt_3"):
            val = str(entries.get(key) or "").strip()
            if val:
                cleaned[key] = val
        if cleaned:
            out[code] = cleaned
    return out


def localized_question_text(
    question: dict[str, Any] | None,
    *,
    language_code: str = "en",
    key: str = "question",
) -> str:
    """Return admin-defined localized text, falling back to base question fields."""
    if not question:
        return ""
    code = canonical_language_code(language_code) or str(language_code or "en").lower()
    locales = normalize_question_locales(question.get("locales"))
    if code and code in locales:
        localized = str((locales.get(code) or {}).get(key) or "").strip()
        if localized:
            return localized
    return str(question.get(key) or "").strip()


def resolve_language_choice(
    transcript: str,
    question: dict[str, Any] | None = None,
) -> str | None:
    """Match caller speech to a configured language code (``en`` / ``es``)."""
    from app.core.screening_flow import normalize_text

    text = normalize_text(transcript or "")
    if not text:
        return None
    if question and str(question.get("answer_type")) == "language_choice":
        for opt in normalize_language_options(question.get("language_options")):
            code = canonical_language_code(opt.get("value"))
            if not code:
                continue
            candidates = {code, normalize_text(opt.get("label") or "")}
            for alias in opt.get("aliases") or []:
                candidates.add(normalize_text(str(alias)))
            for cand in candidates:
                if not cand:
                    continue
                if cand == text or re.search(rf"\b{re.escape(cand)}\b", text):
                    return code
        return None
    return canonical_language_code(transcript)

SCORING_RULE_TYPES = frozenset(
    {
        "any_answer",
        "yes_no",
        "numeric_range",
        "date_within",
        "required_field",
    }
)

CONDITIONAL_OPERATORS = frozenset({"eq", "ne", "truthy", "falsy"})

SPEECH_MODES = frozenset(
    {
        "default",
        "spoken_name",
        "pet_bundle",
        "employer",
        "optional_notes",
        "occupants",
    }
)

# Built-in state metadata for v1→v2 migration, default parsers, and the
# DEFAULT per-question scoring. There is no separate "bucket" weighting system:
# the default questions carry their own scoring (summing to 100), exactly like
# any admin-added question. Deleting a question removes its score with it.
_LEGACY_STATE_META: dict[str, dict[str, Any]] = {
    "Q1_FULL_NAME": {
        "answer_type": "text",
        "extract_fields": ("full_name",),
        "requires_confirmation": True,
        "speech_mode": "spoken_name",
        "field_labels": {"full_name": "full legal name (first and last)"},
        "understanding_guide": (
            "Listen for full legal name even if given casually. "
            "Spelled letters are corrections — assemble them."
        ),
    },
    "Q2_PHONE": {
        "answer_type": "phone",
        "extract_fields": ("contact_phone",),
        "requires_confirmation": True,
        "field_labels": {"contact_phone": "phone number"},
        "understanding_guide": "Accept any phone format; normalize to digits.",
    },
    "Q3_EMAIL": {
        "answer_type": "email",
        "extract_fields": ("email",),
        "requires_confirmation": True,
        "field_labels": {"email": "email address"},
        "understanding_guide": "Accept spoken email; assemble spelled local parts.",
    },
    "Q4_MOVE_IN_DATE": {
        "answer_type": "date",
        "extract_fields": ("move_in_date", "move_in_raw"),
        "field_labels": {
            "move_in_date": "move-in date (ISO if clear)",
            "move_in_raw": "move-in timeframe wording",
        },
        "understanding_guide": "Accept relative dates and vague windows.",
        "scoring": {
            "max_points": 15,
            "rule_type": "date_within",
            "pass_config": {"max_days_ahead": 120},
        },
    },
    "Q5_OCCUPANTS": {
        "answer_type": "number",
        "extract_fields": ("occupants_count", "adults_count", "children_count"),
        "speech_mode": "occupants",
        "field_labels": {
            "occupants_count": "total occupants",
            "adults_count": "adults",
            "children_count": "children",
        },
        "understanding_guide": "Count everyone living in the home.",
        "scoring": {"max_points": 10, "rule_type": "any_answer", "pass_config": {}},
    },
    "Q6_PETS": {
        "answer_type": "yes_no",
        "extract_fields": ("has_pets", "pets_raw"),
        "field_labels": {"has_pets": "yes/no pets", "pets_raw": "pet description"},
        "understanding_guide": "Boolean only for pets.",
        "scoring": {"max_points": 5, "rule_type": "any_answer", "pass_config": {}},
    },
    "Q6A_PET_DETAILS": {
        "answer_type": "text",
        "extract_fields": ("pet_type", "pet_breed", "pet_weight", "pets_raw"),
        "conditional": {"field": "has_pets", "operator": "eq", "value": True},
        "require_all_extract_fields": True,
        "speech_mode": "pet_bundle",
        "field_labels": {
            "pet_type": "pet type",
            "pet_breed": "breed",
            "pet_weight": "approximate weight",
        },
        "understanding_guide": "Extract type, breed, and weight.",
    },
    "Q7_CURRENT_RESIDENCE": {
        "answer_type": "text",
        "extract_fields": ("current_residence",),
        "field_labels": {"current_residence": "current address or area"},
        "scoring": {"max_points": 10, "rule_type": "any_answer", "pass_config": {}},
    },
    "Q8_RESIDENCE_DURATION": {
        "answer_type": "text",
        "extract_fields": ("residence_duration",),
        "field_labels": {"residence_duration": "how long at current home"},
        "scoring": {"max_points": 5, "rule_type": "any_answer", "pass_config": {}},
    },
    "Q9_MOVE_REASON": {
        "answer_type": "long_text",
        "extract_fields": ("move_reason",),
        "field_labels": {"move_reason": "reason for moving"},
        "scoring": {"max_points": 5, "rule_type": "any_answer", "pass_config": {}},
    },
    "Q10_MOVE_TIMING": {
        "answer_type": "date",
        "extract_fields": ("move_timing",),
        "field_labels": {"move_timing": "when they plan to leave current place"},
    },
    "Q11_EVICTION": {
        "answer_type": "yes_no",
        "extract_fields": ("has_eviction", "eviction_raw"),
        "field_labels": {
            "has_eviction": "yes/no eviction history",
            "eviction_raw": "brief eviction mention",
        },
        "scoring": {
            "max_points": 15,
            "rule_type": "yes_no",
            "pass_config": {"yes": 0, "no": 15},
        },
    },
    "Q11A_EVICTION_DETAILS": {
        "answer_type": "long_text",
        "extract_fields": ("eviction_circumstances", "eviction_raw"),
        "conditional": {"field": "has_eviction", "operator": "eq", "value": True},
        "field_labels": {"eviction_circumstances": "eviction circumstances"},
    },
    "Q12_INCOME": {
        "answer_type": "currency",
        "extract_fields": ("monthly_income", "income_raw"),
        "field_labels": {
            "monthly_income": "monthly household income before taxes",
            "income_raw": "income wording",
        },
        "scoring": {"max_points": 35, "rule_type": "any_answer", "pass_config": {}},
    },
    "Q13_EMPLOYER": {
        "answer_type": "text",
        "extract_fields": ("employer",),
        "speech_mode": "employer",
        "field_labels": {"employer": "employer or income source"},
    },
    "Q14_EMPLOYMENT_DURATION": {
        "answer_type": "text",
        "extract_fields": ("employment_duration",),
        "field_labels": {"employment_duration": "time at current job"},
    },
    "Q15_GENERAL_NOTES": {
        "answer_type": "long_text",
        "extract_fields": ("general_notes",),
        "required": False,
        "speech_mode": "optional_notes",
        "field_labels": {"general_notes": "final notes or 'None disclosed'"},
    },
}


def _default_field_labels(extract_fields: list[str]) -> dict[str, str]:
    return {f: f.replace("_", " ") for f in extract_fields}


def _infer_answer_type(q: dict[str, Any]) -> str:
    if q.get("answer_type") in ANSWER_TYPES:
        return str(q["answer_type"])
    state = str(q.get("state") or "")
    meta = _LEGACY_STATE_META.get(state, {})
    if meta.get("answer_type"):
        return str(meta["answer_type"])
    fields = q.get("extract_fields") or []
    if fields:
        name = str(fields[0]).lower()
        if name.startswith("has_") or name.endswith("_flag"):
            return "yes_no"
        if "email" in name:
            return "email"
        if "phone" in name:
            return "phone"
        if "date" in name or "timing" in name:
            return "date"
        if "income" in name or "amount" in name:
            return "currency"
        if "count" in name or name.endswith("_num"):
            return "number"
    validation = str(q.get("validation") or "").lower()
    if "email" in validation:
        return "email"
    if "phone" in validation:
        return "phone"
    if "date" in validation or "timeframe" in validation:
        return "date"
    if "yes" in validation or "no" in validation:
        return "yes_no"
    if "number" in validation or "occupant" in validation:
        return "number"
    return "text"


def _normalize_speech_mode(
    explicit: Any,
    meta_default: Any,
    state: str,
) -> str:
    for candidate in (explicit, meta_default):
        mode = str(candidate or "").strip().lower()
        if mode in SPEECH_MODES:
            return mode
    return "default"


def speech_mode_for_question(q: dict[str, Any] | None) -> str:
    if not q:
        return "default"
    mode = str(q.get("speech_mode") or "default").strip().lower()
    return mode if mode in SPEECH_MODES else "default"


def migrate_question_to_v2(q: dict[str, Any]) -> dict[str, Any]:
    """Upgrade a single question dict to schema v2."""
    state = str(q.get("state") or "")
    meta = _LEGACY_STATE_META.get(state, {})
    extract_fields = list(q.get("extract_fields") or [])
    if not extract_fields:
        meta_fields = meta.get("extract_fields")
        if meta_fields:
            extract_fields = list(meta_fields)

    field_labels = dict(q.get("field_labels") or meta.get("field_labels") or {})
    if not field_labels and extract_fields:
        field_labels = _default_field_labels(extract_fields)

    scoring = q.get("scoring")
    if scoring is None:
        # Only built-in defaults carry meta scoring; admin-saved questions always
        # include an explicit scoring dict, so this never re-enables a score the
        # admin turned off.
        meta_scoring = meta.get("scoring")
        if meta_scoring:
            scoring = {
                "enabled": True,
                "max_points": int(meta_scoring.get("max_points") or 0),
                "rule_type": meta_scoring.get("rule_type") or "any_answer",
                "pass_config": dict(meta_scoring.get("pass_config") or {}),
            }
        else:
            scoring = {
                "enabled": False,
                "max_points": 0,
                "rule_type": "any_answer",
                "pass_config": {},
            }
    elif isinstance(scoring, dict) and "enabled" not in scoring:
        scoring = {
            "enabled": bool(scoring.get("max_points")),
            "max_points": int(scoring.get("max_points") or 0),
            "rule_type": scoring.get("rule_type") or "any_answer",
            "pass_config": dict(scoring.get("pass_config") or {}),
        }

    conditional = q.get("conditional")
    if conditional is None and meta.get("conditional"):
        conditional = dict(meta["conditional"])

    answer_type = _infer_answer_type(q)
    result = {
        "schema_version": SCHEMA_VERSION,
        "id": str(q.get("id") or f"Q_{state}"),
        "state": state or f"CUSTOM_{uuid.uuid4().hex[:8].upper()}",
        "question": str(q.get("question") or "Please answer this question."),
        "answer_type": answer_type,
        "extract_fields": extract_fields or [f"field_{q.get('id', 'custom')}"],
        "field_labels": field_labels,
        "validation": q.get("validation"),
        "retry_prompt": q.get("retry_prompt"),
        "retry_prompt_2": q.get("retry_prompt_2") or "",
        "retry_prompt_3": q.get("retry_prompt_3") or "",
        "active": bool(q.get("active", True)),
        "order": int(q.get("order") or 0),
        "required": bool(q.get("required", meta.get("required", True))),
        "requires_confirmation": bool(
            q.get("requires_confirmation", meta.get("requires_confirmation", False))
        ),
        "conditional": conditional,
        "scoring": scoring,
        "understanding_guide": q.get(
            "understanding_guide", meta.get("understanding_guide", "")
        ),
        "require_all_extract_fields": bool(
            q.get(
                "require_all_extract_fields",
                meta.get("require_all_extract_fields", False),
            )
        ),
        "speech_mode": _normalize_speech_mode(
            q.get("speech_mode"), meta.get("speech_mode"), state
        ),
        "locales": normalize_question_locales(q.get("locales")),
    }
    if answer_type == "language_choice":
        result["extract_fields"] = ["preferred_language"]
        result["field_labels"] = dict(
            q.get("field_labels") or {"preferred_language": "preferred language"}
        )
        result["requires_confirmation"] = False
        result["language_options"] = normalize_language_options(q.get("language_options"))
        if not str(result.get("understanding_guide") or "").strip():
            labels = ", ".join(o["label"] for o in result["language_options"])
            codes = ", ".join(o["value"] for o in result["language_options"])
            result["understanding_guide"] = (
                f"Extract preferred_language as one of: {codes}. "
                f"Caller may say: {labels}."
            )
        lang_scoring = result.get("scoring") or {}
        if lang_scoring.get("enabled"):
            result["scoring"] = {
                **lang_scoring,
                "enabled": False,
                "max_points": 0,
            }
    return result


def migrate_questions_to_v2(questions: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if not questions:
        return []
    return [migrate_question_to_v2(dict(q)) for q in questions]


def default_questions_v2() -> list[dict[str, Any]]:
    """Install-time default questions (seed JSON). Not a live-call fallback."""
    import copy

    from app.core.seed_data import load_seed_questions

    return copy.deepcopy(load_seed_questions())


def is_v2_question(q: dict[str, Any]) -> bool:
    return bool(q.get("schema_version") == SCHEMA_VERSION or q.get("answer_type"))


def normalize_questions(questions: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Normalize admin questions from the database."""
    if not questions:
        return []
    if not any(is_v2_question(q) for q in questions):
        # v1 list — migrate wholesale
        return migrate_questions_to_v2(questions)
    normalized = [migrate_question_to_v2(dict(q)) for q in questions]
    normalized.sort(key=lambda x: int(x.get("order") or 0))
    for idx, q in enumerate(normalized, start=1):
        q["order"] = idx
    return normalized


def questions_index(questions: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    return {str(q["state"]): q for q in normalize_questions(questions)}


def flow_states_in_order(questions: list[dict[str, Any]] | None) -> list[str]:
    return [str(q["state"]) for q in normalize_questions(questions)]


def ordered_active_questions(
    questions: list[dict[str, Any]] | None,
    data: dict[str, Any] | None = None,
    *,
    skip_states: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Active questions in order, respecting conditionals and skip sets."""
    data = data or {}
    skip = set(skip_states or [])
    result: list[dict[str, Any]] = []
    for q in normalize_questions(questions):
        if not q.get("active", True):
            continue
        state = str(q["state"])
        if state in skip:
            continue
        if should_skip_question(q, data):
            continue
        result.append(q)
    return result


def evaluate_conditional(
    conditional: dict[str, Any] | None, data: dict[str, Any]
) -> bool:
    """Return True when the question should be asked (condition met)."""
    if not conditional:
        return True
    field = str(conditional.get("field") or "")
    if not field:
        return True
    value = data.get(field)
    op = str(conditional.get("operator") or "truthy")
    expected = conditional.get("value")
    if op == "eq":
        return _conditional_equals(value, expected)
    if op == "ne":
        return not _conditional_equals(value, expected)
    if op == "falsy":
        return _truthy_value(value) is False
    if op == "truthy":
        return _truthy_value(value) is True
    return True


def _truthy_value(value: Any) -> bool:
    """Semantic truthiness for conditional rules.

    Critically, the strings "no"/"false"/"0" are treated as FALSE even though
    they are truthy in Python — otherwise a yes/no answer stored as text would
    wrongly trigger a follow-up question.
    """
    from app.core.screening_flow import _coerce_bool

    if isinstance(value, bool):
        return value
    if value in (None, "", [], {}):
        return False
    coerced = _coerce_bool(value)
    if coerced is not None:
        return coerced
    return bool(value)


def _conditional_equals(value: Any, expected: Any) -> bool:
    """Equality that understands booleans expressed as strings."""
    if isinstance(expected, bool) or str(expected).strip().lower() in {
        "true",
        "false",
        "yes",
        "no",
    }:
        from app.core.screening_flow import _coerce_bool

        exp_bool = _coerce_bool(expected)
        val_bool = _coerce_bool(value)
        if exp_bool is not None:
            return val_bool == exp_bool
    return str(value).strip().lower() == str(expected).strip().lower()


def should_skip_question(q: dict[str, Any], data: dict[str, Any]) -> bool:
    conditional = q.get("conditional")
    if conditional and not evaluate_conditional(conditional, data):
        return True
    return False


def confirm_field_for_question(q: dict[str, Any]) -> str | None:
    if not q.get("requires_confirmation"):
        return None
    fields = q.get("extract_fields") or []
    return str(fields[0]) if fields else None


def build_confirm_field_map(questions: list[dict[str, Any]] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for q in normalize_questions(questions):
        field = confirm_field_for_question(q)
        if field:
            out[str(q["state"])] = field
    return out


def needs_readback_confirmation(
    state: str,
    data: dict[str, Any],
    questions: list[dict[str, Any]] | None,
    confirmed_fields: Iterable[str] | None = None,
) -> bool:
    """True when a captured value still needs spoken read-back confirmation."""
    q = questions_index(questions).get(str(state))
    if not q:
        return False
    field = confirm_field_for_question(q)
    if not field or field in set(confirmed_fields or ()):
        return False
    value = data.get(field)
    return value not in (None, "")


def _primary_field(q: dict[str, Any]) -> str:
    fields = q.get("extract_fields") or []
    return str(fields[0]) if fields else str(q.get("state") or "value")


def _is_valid_phone(value: Any) -> bool:
    if not value:
        return False
    return bool(normalize_phone(str(value)))


def _is_valid_email(value: Any) -> bool:
    if not value:
        return False
    return bool(normalize_email(str(value)))


def _parse_number(value: Any) -> Decimal | None:
    if value in (None, ""):
        return None
    try:
        cleaned = re.sub(r"[^\d.\-]", "", str(value))
        if not cleaned:
            return None
        return Decimal(cleaned)
    except (InvalidOperation, ValueError):
        return None


def is_question_required(q: dict[str, Any] | None) -> bool:
    """False only when the admin explicitly marked the question optional."""
    if not q:
        return True
    return q.get("required", True) is not False


def require_all_extract_fields(q: dict[str, Any]) -> bool:
    """When true, every non-raw extract field must be captured before advancing."""
    return bool(q.get("require_all_extract_fields"))


def non_raw_extract_fields(q: dict[str, Any]) -> list[str]:
    return [
        str(f)
        for f in (q.get("extract_fields") or [])
        if not str(f).endswith("_raw")
    ]


def _extract_fields_satisfied(
    q: dict[str, Any],
    data: dict[str, Any],
    *,
    answer_type: str,
) -> bool:
    """True when configured extract slots for this question are filled."""
    from app.core.screening_flow import _has_value

    if require_all_extract_fields(q):
        for field in non_raw_extract_fields(q):
            if field.startswith(("has_", "is_")):
                if data.get(field) not in (True, False):
                    return False
            elif not _has_value(data, field):
                return False
        return True

    primary = _primary_field(q)
    if answer_type == "yes_no":
        return data.get(primary) in (True, False)
    if answer_type in ("number", "currency"):
        if _parse_number(data.get(primary)) is not None:
            return True
        return _has_value(data, primary)
    if answer_type == "date":
        raw_field = f"{primary}_raw" if primary else ""
        date_fields = [primary]
        if raw_field and raw_field not in date_fields:
            date_fields.append(raw_field)
        return _has_value(data, *date_fields)
    if answer_type == "phone":
        return _is_valid_phone(data.get(primary))
    if answer_type == "email":
        return _is_valid_email(data.get(primary))
    if answer_type == "language_choice":
        return canonical_language_code(data.get(primary)) is not None
    if answer_type in ("text", "long_text"):
        if _has_value(data, primary):
            return True
        state = str(q.get("state") or "")
        return bool((data.get("raw_answers") or {}).get(state))
    return _has_value(data, primary)


def is_question_answered_for_def(
    q: dict[str, Any],
    data: dict[str, Any],
    refused_states: Iterable[str] | None = None,
    *,
    confirmed_fields: Iterable[str] | None = None,
) -> bool:
    """True when the caller genuinely answered this question (not merely skipped)."""
    state = str(q.get("state") or "")
    refused = set(refused_states or [])
    if state in refused:
        return True
    # Conditional follow-ups that do not apply are skipped, not answered.
    if should_skip_question(q, data):
        return False

    confirm_field = confirm_field_for_question(q)
    if confirm_field and confirm_field not in set(confirmed_fields or ()):
        return False

    answer_type = str(q.get("answer_type") or "text")
    return _extract_fields_satisfied(q, data, answer_type=answer_type)


def _is_spanish_language(language_code: str | None) -> bool:
    code = str(language_code or "").strip().lower()
    return code.startswith("es")


def readback_prompt_for_question(
    q: dict[str, Any], value: str, *, language_code: str = "en"
) -> str:
    """Spoken read-back for any admin-configured question with confirmation."""
    answer_type = str(q.get("answer_type") or "text")
    primary = _primary_field(q)
    labels = q.get("field_labels") or {}
    label = str(labels.get(primary) or primary.replace("_", " ")).strip()

    is_es = _is_spanish_language(language_code)

    if answer_type == "phone":
        from app.core.screening_flow import _digits_spaced

        if is_es:
            return (
                "Permiteme repetirlo para asegurarme de tenerlo bien: "
                f"{_digits_spaced(str(value))}. Es correcto?"
            )
        return (
            "Let me read that back to make sure I have it right — "
            f"{_digits_spaced(str(value))}. Is that correct?"
        )
    if answer_type == "email":
        if is_es:
            return f"Tengo su {label} como {value}. Esta bien?"
        return f"I have your {label} as {value}. Is that right?"
    if "name" in primary.lower() or "name" in label.lower():
        if is_es:
            return f"Solo para confirmar, tengo su {label} como {value}. Es correcto?"
        return f"Just to confirm, I have your {label} as {value}. Did I get that right?"
    if is_es:
        return f"Solo para confirmar, tengo {label} como {value}. Es correcto?"
    return f"Just to confirm, I have {label} as {value}. Is that correct?"


def repair_prompt_for_question(q: dict[str, Any], *, language_code: str = "en") -> str:
    """Re-ask prompt after the caller rejects a read-back."""
    answer_type = str(q.get("answer_type") or "text")
    primary = _primary_field(q)
    question_text = localized_question_text(
        q,
        language_code=language_code,
        key="question",
    )
    is_es = _is_spanish_language(language_code)
    if answer_type == "phone":
        if is_es:
            return "No hay problema. Diga su numero de telefono otra vez, digito por digito."
        return "No problem — please say your phone number again, one digit at a time."
    if answer_type == "email":
        if is_es:
            return (
                "No hay problema. Podria decir su correo otra vez despacio? "
                "Puede deletrearlo."
            )
        return (
            "No problem — could you say your email again slowly? "
            "Feel free to spell it."
        )
    if "name" in primary.lower():
        if is_es:
            return "No hay problema. Podria decir su nombre completo otra vez, claramente?"
        return "No problem — could you say your full name again, nice and clearly?"
    if question_text:
        if is_es:
            return f"No hay problema. {question_text}"
        return f"No problem — {question_text}"
    if is_es:
        return "No hay problema. Podria repetirlo?"
    return "No problem — could you say that again?"


def readback_prompt_for_state(
    state_value: str,
    value: str,
    questions: list[dict[str, Any]] | None = None,
    *,
    language_code: str = "en",
) -> str:
    q = questions_index(questions).get(str(state_value))
    if q:
        return readback_prompt_for_question(q, value, language_code=language_code)
    if _is_spanish_language(language_code):
        return f"Solo para confirmar, tengo {value}. Es correcto?"
    return f"Just to confirm, I have {value}. Is that correct?"


def repair_prompt_for_state(
    state_value: str,
    questions: list[dict[str, Any]] | None = None,
    *,
    language_code: str = "en",
) -> str:
    q = questions_index(questions).get(str(state_value))
    if q:
        return repair_prompt_for_question(q, language_code=language_code)
    if _is_spanish_language(language_code):
        return "No hay problema. Podria repetirlo?"
    return "No problem — could you say that again?"


def extract_fields_from_speech(
    text: str,
    question: dict[str, Any],
    existing_data: dict[str, Any] | None = None,
    *,
    intent: Any | None = None,
) -> dict[str, Any]:
    """Best-effort deterministic extraction for one admin-configured question."""
    existing_data = existing_data or {}
    out: dict[str, Any] = {}
    stripped = (text or "").strip()
    if not stripped:
        return out

    from app.core.screening_flow import (
        PHONE_RE,
        _is_bare_ack,
        _is_refusal_text,
        _word_or_digit_count,
        extract_money_from_text,
        extract_occupants,
        extract_pet_fields,
        is_pure_affirmation,
        normalize_email,
        normalize_phone,
        normalize_text,
        parse_relative_date,
        parse_spoken_name,
        parse_yes_no,
    )

    answer_type = str(question.get("answer_type") or "text")
    fields = list(question.get("extract_fields") or [])
    primary = str(fields[0]) if fields else ""
    if not primary:
        return out

    raw_field = next((f for f in fields if str(f).endswith("_raw")), None)

    if answer_type == "email":
        spoken = normalize_email(stripped)
        if spoken:
            out[primary] = spoken
    elif answer_type == "phone":
        if PHONE_RE.search(stripped) or answer_type == "phone":
            phone = normalize_phone(stripped)
            if phone:
                out[primary] = phone
    elif answer_type == "yes_no":
        domain = primary.replace("has_", "").replace("is_", "")
        yn = (
            intent.yes_no
            if intent is not None and getattr(intent, "yes_no", None) is not None
            else parse_yes_no(stripped, domain=domain)
        )
        if yn is not None:
            out[primary] = yn
            if raw_field:
                out[raw_field] = stripped
    elif answer_type == "date":
        parsed, raw = parse_relative_date(stripped)
        if raw and raw_field:
            out[raw_field] = raw
        if parsed:
            out[primary] = parsed.isoformat()
        elif not _is_refusal_text(stripped) and len(stripped.split()) >= 2:
            target = raw_field or primary
            out[target] = stripped
        elif not parsed and stripped and not _is_bare_ack(stripped):
            out[primary] = stripped
    elif answer_type == "currency":
        monthly, _raw = extract_money_from_text(stripped)
        norm = normalize_text(stripped)
        is_hourly = bool(re.search(r"\b(hour|hourly|per hour|an hour|/hr|hr)\b", norm))
        if raw_field:
            out[raw_field] = stripped
        if monthly is not None and not is_hourly:
            out[primary] = monthly
        elif raw_field is None and (
            is_hourly
            or re.search(
                r"\b(thousand|grand|salary|wage|income|make|earn|paid|pay|"
                r"month|monthly|week|weekly|hundred|\bk\b)\b",
                norm,
            )
        ):
            out[primary] = stripped
    elif answer_type == "number":
        mode = speech_mode_for_question(question)
        if mode == "occupants" or len([f for f in fields if not str(f).endswith("_raw")]) > 1:
            natural = extract_occupants(stripped)
            for key, value in natural.items():
                if key in fields:
                    out[key] = value
        if not out:
            count = _word_or_digit_count(stripped)
            if count is not None:
                out[primary] = count
    elif answer_type == "language_choice":
        resolved = resolve_language_choice(stripped, question)
        if resolved:
            out[primary] = resolved
    elif answer_type in ("text", "long_text"):
        mode = speech_mode_for_question(question)
        if mode == "spoken_name":
            name = parse_spoken_name(stripped)
            parts = name.split()
            if len(parts) >= 2 or (len(parts) == 1 and len(parts[0]) >= 3):
                out[primary] = name
        elif mode == "pet_bundle":
            if not _is_bare_ack(stripped) and not _is_refusal_text(stripped):
                out.update(extract_pet_fields(stripped))
                if stripped:
                    out.setdefault(raw_field or "pets_raw", stripped)
        elif mode == "employer":
            employer = re.sub(
                r"\b(i work at|i work for|work at|work for|employer is)\b",
                "",
                stripped,
                flags=re.I,
            )
            cleaned = employer.strip(" .") or stripped
            if cleaned and not _is_bare_ack(stripped):
                out[primary] = cleaned
        elif mode == "optional_notes":
            yn = (
                intent.yes_no
                if intent is not None and getattr(intent, "yes_no", None) is not None
                else parse_yes_no(stripped)
            )
            done = bool(
                re.search(
                    r"\b(nothing else|nothing more|nothing to add|that'?s all|"
                    r"that'?s it|that is all|i'?m good|we'?re good)\b",
                    normalize_text(stripped),
                )
            )
            if yn is False or done:
                out[primary] = "None disclosed"
            elif yn is True and is_pure_affirmation(stripped):
                pass
            elif not _is_bare_ack(stripped) and not _is_refusal_text(stripped):
                out[primary] = stripped
        elif not _is_bare_ack(stripped) and not _is_refusal_text(stripped):
            out[primary] = stripped

    return {k: v for k, v in out.items() if v not in (None, "")}


def build_flow_rows(
    snapshot: list[dict[str, Any]],
    answered_states: Iterable[str] | None,
    refused_states: Iterable[str] | None,
    scoring_data: dict[str, Any] | None = None,
    *,
    confirmed_fields: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Build the per-question flow rows used by the call/tenant detail pages.

    Marks each active question Answered / Declined / Skipped / "—". When the
    tenant has no recorded answered/refused state lists (older calls finalized
    before those were persisted) and ``scoring_data`` is supplied, falls back to
    inferring "answered" from whether the question's field was filled.
    """
    answered = set(answered_states or [])
    refused = set(refused_states or [])
    have_states = bool(answered or refused)
    data = scoring_data or {}
    rows: list[dict[str, Any]] = []
    for q in sorted(snapshot, key=lambda x: int(x.get("order") or 0)):
        if not q.get("active", True):
            continue
        state = str(q.get("state") or "")
        if q.get("conditional") and should_skip_question(q, data):
            status_label = "Skipped"
        elif state in refused:
            status_label = "Declined"
        elif state in answered:
            status_label = "Answered"
        elif (
            not have_states
            and scoring_data is not None
            and is_question_answered_for_def(
                q,
                scoring_data,
                refused_states=refused,
                confirmed_fields=confirmed_fields,
            )
        ):
            status_label = "Answered"
        else:
            status_label = "—"
        rows.append(
            {
                "order": q.get("order"),
                "question": q.get("question"),
                "state": state,
                "status": status_label,
            }
        )
    return rows


def is_question_answered(
    state: str,
    data: dict[str, Any],
    refused_states: Iterable[str] | None = None,
    *,
    questions: list[dict[str, Any]] | None = None,
    confirmed_fields: Iterable[str] | None = None,
) -> bool:
    idx = questions_index(questions)
    if state not in idx:
        return False
    return is_question_answered_for_def(
        idx[state],
        data,
        refused_states,
        confirmed_fields=confirmed_fields,
    )


def next_unanswered_state(
    data: dict[str, Any],
    skip_states: Iterable[str] | None = None,
    *,
    questions: list[dict[str, Any]] | None = None,
    confirmed_fields: Iterable[str] | None = None,
) -> str | None:
    skip = set(skip_states or [])
    qs = normalize_questions(questions)
    for q in qs:
        state = str(q["state"])
        if state in skip:
            continue
        if not q.get("active", True):
            continue
        if should_skip_question(q, data):
            continue
        if not is_question_answered_for_def(
            q, data, skip_states, confirmed_fields=confirmed_fields
        ):
            return state
    return None


def first_active_question_state(questions: list[dict[str, Any]] | None) -> str | None:
    active = ordered_active_questions(questions, {})
    return str(active[0]["state"]) if active else None


def count_answered_questions(
    data: dict[str, Any],
    skip_states: Iterable[str] | None = None,
    *,
    questions: list[dict[str, Any]] | None = None,
    confirmed_fields: Iterable[str] | None = None,
) -> int:
    skip = set(skip_states or [])
    total = 0
    for q in normalize_questions(questions):
        state = str(q["state"])
        if state in skip or not q.get("active", True):
            continue
        if should_skip_question(q, data):
            continue
        if is_question_answered_for_def(
            q, data, skip_states, confirmed_fields=confirmed_fields
        ):
            total += 1
    return total


def count_active_questions(
    data: dict[str, Any],
    skip_states: Iterable[str] | None = None,
    *,
    questions: list[dict[str, Any]] | None = None,
) -> int:
    skip = set(skip_states or [])
    return len(ordered_active_questions(questions, data, skip_states=skip))


def screening_complete(
    data: dict[str, Any],
    skip_states: Iterable[str] | None = None,
    *,
    questions: list[dict[str, Any]] | None = None,
) -> bool:
    return next_unanswered_state(data, skip_states, questions=questions) is None


def inactive_flow_states(questions: list[dict[str, Any]] | None) -> set[str]:
    result: set[str] = set()
    for q in normalize_questions(questions):
        if not q.get("active", True):
            result.add(str(q["state"]))
    return result


def validation_hint_for_question(q: dict[str, Any]) -> str:
    """Admin ``validation`` text, or a sensible default derived from answer_type."""
    explicit = str(q.get("validation") or "").strip()
    if explicit:
        return explicit
    answer_type = str(q.get("answer_type") or "text")
    labels = q.get("field_labels") or {}
    primary = _primary_field(q)
    label = labels.get(primary, primary.replace("_", " "))
    defaults = {
        "email": "valid email address",
        "phone": "valid phone number",
        "date": "date or timeframe",
        "yes_no": "clear yes or no",
        "number": "numeric count",
        "currency": "money amount; preserve exact wording in any *_raw field",
        "long_text": f"complete answer for {label}",
        "text": f"clear answer for {label}",
        "language_choice": "one of the configured language options",
    }
    return defaults.get(answer_type, defaults["text"])


def retry_prompt_for_count(
    question: dict[str, Any] | None,
    retry_count: int,
    *,
    language_code: str = "en",
) -> str:
    """Pick the admin retry prompt for the current retry attempt."""
    if not question:
        return ""
    if retry_count >= 2 and question.get("retry_prompt_3"):
        return localized_question_text(
            question, language_code=language_code, key="retry_prompt_3"
        ) or str(question["retry_prompt_3"])
    if retry_count >= 1 and question.get("retry_prompt_2"):
        return localized_question_text(
            question, language_code=language_code, key="retry_prompt_2"
        ) or str(question["retry_prompt_2"])
    if retry_count > 0 and question.get("retry_prompt"):
        return localized_question_text(
            question, language_code=language_code, key="retry_prompt"
        ) or str(question["retry_prompt"])
    return localized_question_text(question, language_code=language_code, key="question")


def build_question_slot_config(q: dict[str, Any]) -> dict[str, Any]:
    fields = list(q.get("extract_fields") or [])
    labels = dict(q.get("field_labels") or _default_field_labels(fields))
    answer_type = str(q.get("answer_type") or "text")
    non_raw = [f for f in fields if not str(f).endswith("_raw")]
    if require_all_extract_fields(q) and non_raw:
        required = non_raw
        optional = [f for f in fields if f not in non_raw]
    else:
        required = fields[:1] if fields else []
        optional = fields[1:]
    cfg: dict[str, Any] = {
        "required": tuple(required),
        "optional": tuple(optional),
        "labels": labels,
        "complete_hint": validation_hint_for_question(q),
    }
    if answer_type == "date" and len(fields) > 1:
        cfg["required_any"] = True
    if answer_type == "currency" and len(fields) > 1:
        cfg["required_any"] = True
    return cfg


def is_custom_question_state(state: str | None) -> bool:
    """True for admin-added questions (editable primary field key)."""
    return str(state or "").startswith("CUSTOM_")


def locked_primary_field_for_state(state: str | None) -> str | None:
    """Primary extract field for built-in questions; None when admin may edit."""
    if is_custom_question_state(state):
        return None
    meta = _LEGACY_STATE_META.get(str(state or ""))
    fields = (meta or {}).get("extract_fields") or ()
    return str(fields[0]) if fields else None


def understanding_guide_for_question(q: dict[str, Any]) -> str:
    guide = (q.get("understanding_guide") or "").strip()
    if guide:
        return guide
    answer_type = str(q.get("answer_type") or "text")
    labels = q.get("field_labels") or {}
    primary = _primary_field(q)
    label = labels.get(primary, primary.replace("_", " "))
    hints = {
        "yes_no": f"Extract a clear yes or no for {label}.",
        "number": f"Extract a numeric value for {label}.",
        "currency": f"Extract an amount for {label}; preserve exact wording in raw fields.",
        "date": f"Extract a date or timeframe for {label}.",
        "phone": f"Extract and normalize a phone number for {label}.",
        "email": f"Extract and normalize an email for {label}.",
        "long_text": f"Capture a complete answer for {label}.",
        "text": f"Extract {label} from the caller's reply.",
        "language_choice": (
            "Map the caller's reply to preferred_language using only the "
            "configured language options (codes en/es)."
        ),
    }
    return hints.get(answer_type, hints["text"])


def build_field_maps(
    questions: list[dict[str, Any]] | None,
) -> tuple[dict[str, str], dict[str, str]]:
    field_to_state: dict[str, str] = {}
    field_labels: dict[str, str] = {}
    for q in normalize_questions(questions):
        state = str(q["state"])
        cfg = build_question_slot_config(q)
        labels = cfg.get("labels") or {}
        for field in tuple(cfg.get("required") or ()) + tuple(cfg.get("optional") or ()):
            field_to_state.setdefault(field, state)
            field_labels.setdefault(field, labels.get(field, field.replace("_", " ")))
    return field_to_state, field_labels


_RESERVED_TENANT_COLUMNS: frozenset[str] | None = None

# Explicit list of directly-mapped Tenant answer/meta columns that a custom
# question field must not reuse. Kept as a hardcoded fallback so the guard works
# even if the ORM model can't be imported in a given context (e.g. tooling).
# The live model columns are unioned in at runtime to catch any future drift.
_RESERVED_TENANT_FALLBACK: frozenset[str] = frozenset(
    {
        "phone_number",
        "full_name",
        "contact_phone",
        "email",
        "adults_count",
        "children_count",
        "occupants_count",
        "monthly_income",
        "income_raw",
        "has_pets",
        "pets_raw",
        "pet_type",
        "pet_breed",
        "pet_weight",
        "has_eviction",
        "eviction_raw",
        "eviction_circumstances",
        "move_in_date",
        "move_in_raw",
        "current_residence",
        "residence_duration",
        "move_reason",
        "move_timing",
        "employer",
        "employment_duration",
        "general_notes",
        "special_notes",
        "human_requested",
        "callback_requested",
        "stop_requested",
        "qualification_score",
        "qualification_status",
        "disqualify_reasons",
        "notes",
        "email_sent",
        "email_sent_at",
        "reviewed_by_admin",
        "reviewed_at",
        "is_blacklisted",
    }
)

# Container/meta columns that are never written from extract_fields — excluded
# so a legitimate custom field is not falsely flagged.
_RESERVED_TENANT_EXCLUDE: frozenset[str] = frozenset(
    {
        "id",
        "call_id",
        "normalized_data",
        "raw_answers",
        "answered_states",
        "refused_states",
        "faq_topics",
        "control_flags",
        "qualification_details",
        "created_at",
        "updated_at",
    }
)


def _reserved_tenant_columns() -> frozenset[str]:
    """Tenant column names that custom question fields may not reuse.

    Unions a hardcoded fallback (so the guard never silently disables) with the
    live ORM columns (so new columns are covered automatically), minus the
    JSON/meta container columns that aren't answer targets.
    """
    global _RESERVED_TENANT_COLUMNS
    if _RESERVED_TENANT_COLUMNS is not None:
        return _RESERVED_TENANT_COLUMNS
    cols = set(_RESERVED_TENANT_FALLBACK)
    try:
        from app.models.tenant import Tenant

        cols |= set(Tenant.__table__.columns.keys())
    except Exception:
        pass
    cols -= _RESERVED_TENANT_EXCLUDE
    _RESERVED_TENANT_COLUMNS = frozenset(cols)
    return _RESERVED_TENANT_COLUMNS


def validate_questions_for_save(
    questions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not questions:
        raise ValueError("At least one question is required")

    for raw_q in questions:
        if not isinstance(raw_q, dict):
            continue
        if str(raw_q.get("answer_type")) == "language_choice":
            if (raw_q.get("scoring") or {}).get("enabled"):
                raise ValueError(
                    f"Language choice question {raw_q.get('id')} cannot affect score"
                )

    normalized = [migrate_question_to_v2(dict(q)) for q in questions]
    ids = [str(q["id"]) for q in normalized]
    states = [str(q["state"]) for q in normalized]
    if len(ids) != len(set(ids)):
        raise ValueError("Question IDs must be unique")
    if len(states) != len(set(states)):
        raise ValueError("Duplicate question states are not allowed")

    active_count = sum(1 for q in normalized if q.get("active", True))
    if active_count < 1:
        raise ValueError("At least one active question is required")

    normalized.sort(key=lambda x: int(x.get("order") or 0))

    active_language_choice = 0
    known_fields: set[str] = set()
    primary_fields: set[str] = set()
    for q in normalized:
        if str(q.get("answer_type") or "") not in ANSWER_TYPES:
            raise ValueError(f"Invalid answer_type on {q.get('id')}")
        mode = str(q.get("speech_mode") or "default")
        if mode not in SPEECH_MODES:
            raise ValueError(f"Invalid speech_mode on {q.get('id')}")
        fields = q.get("extract_fields") or []
        if not fields:
            raise ValueError(f"Question {q.get('id')} needs at least one extract field")
        conditional = q.get("conditional")
        if conditional:
            ref = str(conditional.get("field") or "")
            if ref and ref not in known_fields:
                raise ValueError(
                    f"Conditional on {q.get('id')} references field '{ref}' "
                    "from an earlier question"
                )
            op = str(conditional.get("operator") or "")
            if op and op not in CONDITIONAL_OPERATORS:
                raise ValueError(f"Invalid conditional operator '{op}' on {q.get('id')}")
        # Only the primary (first) field must be unique — it drives completion
        # and scoring. Secondary/raw fields (e.g. pets_raw, eviction_raw) may be
        # shared between a question and its conditional follow-up.
        primary = str(fields[0])

        if str(q.get("answer_type")) == "language_choice":
            if "preferred_language" in primary_fields:
                raise ValueError("Only one language choice question is allowed")
            if primary != "preferred_language":
                raise ValueError(
                    f"Language choice question {q.get('id')} must use "
                    "preferred_language as its primary field"
                )
            opts = normalize_language_options(q.get("language_options"))
            codes = [canonical_language_code(o["value"]) for o in opts]
            if len(opts) < 2 or any(c is None for c in codes):
                raise ValueError(
                    f"Language choice question {q.get('id')} needs at least "
                    "two valid language options"
                )
            if len(set(codes)) != len(codes):
                raise ValueError(
                    f"Duplicate language codes on {q.get('id')}"
                )
            q["language_options"] = opts
            q["requires_confirmation"] = False
            if q.get("active", True):
                active_language_choice += 1

        locked_primary = locked_primary_field_for_state(str(q.get("state") or ""))
        if locked_primary is not None and primary != locked_primary:
            raise ValueError(
                f"Cannot change the primary field for built-in question "
                f"{q.get('id')!r} (expected '{locked_primary}')"
            )
        if primary in primary_fields:
            raise ValueError(
                f"Duplicate primary field '{primary}' on {q.get('id')}"
            )
        primary_fields.add(primary)

        # Custom admin-added questions must not reuse a Tenant column name.
        if str(q.get("state", "")).startswith("CUSTOM_"):
            reserved = _reserved_tenant_columns()
            for field in fields:
                if str(field) in reserved:
                    raise ValueError(
                        f"Question {q.get('id')} uses reserved field name "
                        f"'{field}'. Pick a different field key for custom "
                        "questions so it doesn't collide with a stored column."
                    )

        for field in fields:
            known_fields.add(str(field))

    if active_language_choice > 1:
        raise ValueError("Only one active language choice question is allowed")

    for idx, q in enumerate(normalized, start=1):
        q["order"] = idx
        q["schema_version"] = SCHEMA_VERSION
        scoring = q.get("scoring") or {}
        if scoring.get("enabled") and str(scoring.get("rule_type")) not in SCORING_RULE_TYPES:
            raise ValueError(f"Invalid scoring rule on {q.get('id')}")

    if first_active_question_state(normalized) is None:
        raise ValueError(
            "No question can be asked at the start of a call — every active "
            "question is conditional. Add at least one question that is always "
            "asked (no “only when” rule), or turn off the condition on your "
            "first question."
        )

    return normalized


def coerce_questions_for_runtime(
    questions: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Validate admin questions at call start; fall back to defaults if corrupt."""
    import logging

    log = logging.getLogger(__name__)
    if not questions:
        return default_questions_v2()
    try:
        return validate_questions_for_save(
            [dict(q) for q in questions if isinstance(q, dict)]
        )
    except ValueError as exc:
        log.error(
            "Invalid screening_questions in settings (%s) — using system defaults",
            exc,
        )
        return default_questions_v2()


def new_language_question(
    *,
    question: str = (
        "Which language would you like to use for this call — English or Spanish?"
    ),
    order: int = 1,
    active: bool = True,
) -> dict[str, Any]:
    """Admin helper: first-call language picker (order 1, before screening)."""
    uid = uuid.uuid4().hex[:8]
    return migrate_question_to_v2(
        {
            "id": f"Q0_LANGUAGE_{uid.upper()}",
            "state": f"Q0_LANGUAGE_{uid.upper()}",
            "question": question,
            "answer_type": "language_choice",
            "extract_fields": ["preferred_language"],
            "field_labels": {"preferred_language": "preferred language"},
            "retry_prompt": "You can say English or Spanish.",
            "language_options": default_language_options(),
            "order": order,
            "active": active,
        }
    )


def new_custom_question(
    *,
    question: str = "New screening question",
    answer_type: str = "text",
    order: int | None = None,
) -> dict[str, Any]:
    uid = uuid.uuid4().hex[:8]
    field = f"custom_{uid}"
    return migrate_question_to_v2(
        {
            "id": f"CUSTOM_{uid.upper()}",
            "state": f"CUSTOM_{uid.upper()}",
            "question": question,
            "answer_type": answer_type,
            "extract_fields": [field],
            "field_labels": {field: "answer"},
            "order": order or 999,
            "active": True,
        }
    )


# ── Admin / validation helpers ───────────────────────────────────────────────

CONTACT_FIELD_LABELS: dict[str, str] = {
    "full_name": "full name",
    "contact_phone": "contact phone",
    "email": "email address",
}


def active_extract_fields(questions: list[dict[str, Any]] | None) -> set[str]:
    fields: set[str] = set()
    for q in normalize_questions(questions):
        if not q.get("active", True):
            continue
        for field in q.get("extract_fields") or []:
            fields.add(str(field))
    return fields


def missing_contact_fields(questions: list[dict[str, Any]] | None) -> list[str]:
    """Contact fields not covered by any active question."""
    present = active_extract_fields(questions)
    missing: list[str] = []
    for key, label in CONTACT_FIELD_LABELS.items():
        if key in present:
            continue
        if _contact_field_covered_by_questions(questions, key):
            continue
        missing.append(label)
    return missing


def _contact_field_covered_by_questions(
    questions: list[dict[str, Any]] | None,
    contact_key: str,
) -> bool:
    """True when an active question collects the same contact info another way."""
    type_map = {
        "full_name": "text",
        "contact_phone": "phone",
        "email": "email",
    }
    expected_type = type_map.get(contact_key)
    if not expected_type:
        return False
    for q in normalize_questions(questions):
        if not q.get("active", True):
            continue
        if str(q.get("answer_type") or "") != expected_type:
            continue
        fields = q.get("extract_fields") or []
        if not fields:
            continue
        if contact_key == "full_name":
            if speech_mode_for_question(q) == "spoken_name":
                return True
            primary = str(fields[0])
            labels = q.get("field_labels") or {}
            label = str(labels.get(primary, "")).lower()
            if "name" in primary.lower() or "name" in label:
                return True
            continue
        return True
    return False


def total_enabled_scoring_points(questions: list[dict[str, Any]] | None) -> int:
    total = 0
    for q in normalize_questions(questions):
        if not q.get("active", True):
            continue
        scoring = q.get("scoring") or {}
        if scoring.get("enabled"):
            total += int(scoring.get("max_points") or 0)
    return total


def question_save_warnings(questions: list[dict[str, Any]] | None) -> list[str]:
    warnings: list[str] = []
    missing = missing_contact_fields(questions)
    if missing:
        warnings.append(
            "No active question collects "
            + ", ".join(missing)
            + ". Result emails and CRM may lack contact info."
        )
    total = total_enabled_scoring_points(questions)
    if total > 100:
        warnings.append(
            f"Enabled scoring points total {total} exceeds 100; "
            "the server normalizes scores at runtime."
        )

    normalized = normalize_questions(questions)

    # A flow where no active question is reachable on the opening turn (every
    # active question is gated by a conditional that is false on empty data)
    # would jump straight to wrap-up and end the call without asking anything.
    if normalized and first_active_question_state(normalized) is None:
        warnings.append(
            "No question can be asked at the start of the call (every active "
            "question is conditional). The call may end before asking anything."
        )

    # Scoring is per-question only. With none enabled, the system can't score
    # applicants and every call routes to manual review.
    if total == 0:
        warnings.append(
            "No question has scoring enabled, so applicants can't be scored "
            "automatically — every call will be marked for manual review. "
            "Enable scoring on at least one question to qualify applicants."
        )
    return warnings


def questions_snapshot_from_tenant(tenant: Any | None) -> list[dict[str, Any]] | None:
    """Return the question list frozen at call finalize, if stored."""
    if tenant is None or not isinstance(getattr(tenant, "normalized_data", None), dict):
        return None
    stored = tenant.normalized_data.get("screening_questions")
    if isinstance(stored, list) and stored:
        return normalize_questions(stored)
    return None


def scoring_thresholds_from_tenant(
    tenant: Any | None,
    *,
    fallback_settings: dict | None = None,
) -> dict[str, int]:
    """Return score cutoffs frozen at call finalize, or from fallback settings."""
    defaults = {"qualified_score_threshold": 75, "review_score_threshold": 40}
    if fallback_settings:
        try:
            defaults["qualified_score_threshold"] = int(
                fallback_settings.get("qualified_score_threshold", 75)
            )
        except (TypeError, ValueError):
            pass
        try:
            defaults["review_score_threshold"] = int(
                fallback_settings.get("review_score_threshold", 40)
            )
        except (TypeError, ValueError):
            pass

    if tenant is None or not isinstance(getattr(tenant, "normalized_data", None), dict):
        return defaults

    nd = tenant.normalized_data
    for key in ("qualified_score_threshold", "review_score_threshold"):
        if key in nd:
            try:
                defaults[key] = int(nd[key])
            except (TypeError, ValueError):
                pass
    return defaults


def field_labels_from_questions(
    questions: list[dict[str, Any]] | None,
) -> dict[str, str]:
    labels: dict[str, str] = {}
    for q in normalize_questions(questions):
        for field, label in (q.get("field_labels") or {}).items():
            labels.setdefault(str(field), str(label))
    return labels


def field_answer_types_from_questions(
    questions: list[dict[str, Any]] | None,
) -> dict[str, str]:
    """Map each extract field to its admin-configured answer_type."""
    out: dict[str, str] = {}
    for q in normalize_questions(questions):
        answer_type = str(q.get("answer_type") or "text")
        for field in q.get("extract_fields") or []:
            field = str(field)
            # Verbatim caller wording — never inherit currency/yes_no from parent Q.
            if field.endswith("_raw"):
                out.setdefault(field, "text")
            else:
                out.setdefault(field, answer_type)
    return out


def _field_question_meta(
    questions: list[dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    """Map each extract field to its owning active question metadata."""
    out: dict[str, dict[str, Any]] = {}
    for q in normalize_questions(questions):
        if not q.get("active", True):
            continue
        for field in q.get("extract_fields") or []:
            out.setdefault(str(field), q)
    return out


def prompt_fields_catalog(questions: list[dict[str, Any]] | None) -> str:
    """Build the LLM extraction field list from the admin question snapshot."""
    labels = field_labels_from_questions(questions)
    types = field_answer_types_from_questions(questions)
    meta_by_field = _field_question_meta(questions)
    lines: list[str] = []
    seen: set[str] = set()
    for q in normalize_questions(questions):
        if not q.get("active", True):
            continue
        for field in q.get("extract_fields") or []:
            field = str(field)
            if field in seen:
                continue
            seen.add(field)
            label = labels.get(field, field.replace("_", " "))
            answer_type = types.get(field, "text")
            owner = meta_by_field.get(field) or q
            tags: list[str] = []
            if field.endswith("_raw"):
                tags.append("verbatim text")
            if owner.get("required", True) is False:
                tags.append("opt")
            if confirm_field_for_question(owner) == field:
                tags.append("confirm")
            suffix = f" [{', '.join(tags)}]" if tags else ""
            lines.append(f"- {field}: {label} ({answer_type}){suffix}")
    return "\n".join(lines) if lines else "- (no active extract fields configured)"


def _flow_outline_line(q: dict[str, Any], *, current_state: str | None = None) -> str:
    tags: list[str] = []
    cond = q.get("conditional")
    if cond:
        ref = cond.get("field", "")
        op = cond.get("operator", "")
        val = cond.get("value", "")
        val_bit = f" {val!r}" if val not in (None, "") else ""
        tags.append(f"if {ref}{op}{val_bit}")
    if q.get("required", True) is False:
        tags.append("opt")
    if q.get("requires_confirmation"):
        tags.append("confirm")
    tag_str = f" [{', '.join(tags)}]" if tags else ""
    state = str(q.get("state") or "")
    marker = " ← CURRENT" if current_state and state == current_state else ""
    return f"  {state}{tag_str}{marker}"


def prompt_screening_flow_outline(
    questions: list[dict[str, Any]] | None,
    *,
    current_state: str | None = None,
    window_after: int = 3,
) -> str:
    """Compact ordered flow — states + skip/confirm flags (full wording is on CURRENT).

    Short flows (≤8 active steps) list every state. Longer admin flows send only a
    small window around *current_state* so per-turn LLM input stays bounded while
    the full question list remains the admin source of truth at call start.
    """
    active = [q for q in normalize_questions(questions) if q.get("active", True)]
    if not active:
        return "  (no active questions configured)"
    if len(active) <= 8 or not current_state:
        return "\n".join(
            _flow_outline_line(q, current_state=current_state) for q in active
        )

    states = [str(q.get("state") or "") for q in active]
    try:
        idx = states.index(current_state)
    except ValueError:
        idx = 0

    start = max(0, idx - 1)
    end = min(len(active), idx + window_after + 1)
    lines: list[str] = []
    if start > 0:
        lines.append(f"  ... ({start} earlier step(s))")
    for q in active[start:end]:
        lines.append(_flow_outline_line(q, current_state=current_state))
    if end < len(active):
        lines.append(f"  ... ({len(active) - end} more step(s))")
    return "\n".join(lines)


def prompt_flow_stats(questions: list[dict[str, Any]] | None) -> str:
    """One-line active question counts (replaces repeating full question text)."""
    required = optional = 0
    for q in normalize_questions(questions):
        if not q.get("active", True):
            continue
        if q.get("required", True):
            required += 1
        else:
            optional += 1
    total = required + optional
    if not total:
        return "No active questions configured."
    return f"{total} active steps ({required} required, {optional} optional)"


def prompt_confirmation_fields(questions: list[dict[str, Any]] | None) -> str:
    """Fields the admin marked for spoken read-back confirmation."""
    fields: list[str] = []
    labels = field_labels_from_questions(questions)
    for q in normalize_questions(questions):
        if not q.get("active", True):
            continue
        field = confirm_field_for_question(q)
        if field:
            fields.append(f"{field} ({labels.get(field, field.replace('_', ' '))})")
    return ", ".join(fields) if fields else "none configured"


def prompt_required_questions_summary(questions: list[dict[str, Any]] | None) -> str:
    """Short required vs optional summary from admin flags."""
    required: list[str] = []
    optional: list[str] = []
    for q in normalize_questions(questions):
        if not q.get("active", True):
            continue
        label = str(q.get("question") or q.get("state") or "question")
        if q.get("required", True):
            required.append(label)
        else:
            optional.append(label)
    parts: list[str] = []
    if required:
        parts.append("Required: " + "; ".join(required))
    if optional:
        parts.append("Optional: " + "; ".join(optional))
    return " ".join(parts) if parts else "No active questions configured."


def prompt_extraction_rules(
    questions: list[dict[str, Any]] | None,
    *,
    today: str | None = None,
) -> str:
    """Build end-of-call extraction rules from the active admin question list."""
    from datetime import date as _date

    today = today or _date.today().isoformat()
    normalized = normalize_questions(questions)
    active_fields = active_extract_fields(normalized)
    types = field_answer_types_from_questions(normalized)
    lines = ["Rules:"]

    if any(str(f).endswith("_raw") for f in active_fields):
        lines.append("- Preserve raw caller wording in *_raw fields.")

    currency_fields = [
        f
        for f in active_fields
        if types.get(f) == "currency" and not str(f).endswith("_raw")
    ]
    if "monthly_income" in active_fields or currency_fields:
        income_fields = sorted({"monthly_income", *currency_fields})
        lines.append(
            "- For money fields "
            f"({', '.join(income_fields)}): respect the period the caller states. "
            "If monthly or no period is given, store as monthly. "
            "Divide by 12 only when clearly yearly. "
            "Put caller wording in matching *_raw fields as text, not numbers."
        )

    if "has_eviction" in active_fields or "eviction_raw" in active_fields:
        lines.append(
            "- Eviction means an eviction or landlord-tenant court filing. "
            "If unclear, leave has_eviction null and keep caller wording in eviction_raw."
        )

    notes_fields = [
        f
        for f in active_fields
        if "notes" in str(f).lower() or types.get(f) == "long_text"
    ]
    if notes_fields:
        lines.append(
            "- For open-ended note fields "
            f"({', '.join(sorted(notes_fields))}): capture disclosures the caller wants reviewed."
        )

    if any(types.get(f) == "date" for f in active_fields):
        lines.append(
            "- For date fields: use ISO YYYY-MM-DD when clear; keep vague wording "
            "in *_raw fields. Today's date is the reference — if a date is in the "
            "past, ask whether they meant a future date before accepting it."
        )

    if any(types.get(f) in ("phone", "email") for f in active_fields):
        lines.append(
            "- For phone/email fields: extract the value; formatting is normalized after extraction."
        )

    lines.append("- Return JSON only, no markdown.")
    lines.append(f"Today's date: {today}")
    return "\n".join(lines)


def primary_name_field(questions: list[dict[str, Any]] | None) -> str | None:
    """Return the primary field used for a caller's name, if configured."""
    for q in normalize_questions(questions):
        if not q.get("active", True):
            continue
        fields = q.get("extract_fields") or []
        if not fields:
            continue
        primary = str(fields[0])
        labels = q.get("field_labels") or {}
        label = str(labels.get(primary, "")).lower()
        if primary == "full_name" or "name" in primary.lower() or "name" in label:
            return primary
    return None


def slot_fill_examples_for_question(q: dict[str, Any] | None) -> str:
    """Short, question-aware slot-filling hint for the live LLM prompt."""
    if not q:
        return "- Stay on CURRENT; extract volunteered later-step fields."
    answer_type = str(q.get("answer_type") or "text")
    fields = list(q.get("extract_fields") or [])
    primary = str(fields[0]) if fields else "value"
    labels = q.get("field_labels") or {}
    label = str(labels.get(primary, primary.replace("_", " ")))

    if answer_type == "yes_no" and len(fields) > 1:
        secondary = ", ".join(str(f) for f in fields[1:])
        return (
            f'- Yes/no + detail: "yes" → {primary}=true, question_complete=false '
            f"until {secondary} captured."
        )
    if answer_type == "date":
        from datetime import date as _date

        today = _date.today().isoformat()
        raw_field = next((str(f) for f in fields if str(f).endswith("_raw")), f"{primary}_raw")
        return (
            f'- Today is {today}. Vague date → {raw_field}=their words, '
            "question_complete=false, ask once for exact calendar date. "
            "If they give a past year, clarify before accepting."
        )
    if answer_type == "number" and len(fields) > 1:
        return (
            f"- Multi-part: fill {', '.join(str(f) for f in fields)} before "
            "question_complete=true."
        )
    return (
        f"- Partial {label}: fill {primary}, question_complete=false until done."
    )


def _tenant_field_value(tenant: Any, field: str, custom_fields: dict[str, Any]) -> Any:
    if tenant is None:
        return None
    if hasattr(tenant, field):
        value = getattr(tenant, field, None)
        if value not in (None, ""):
            return value
    return custom_fields.get(field)


def _format_summary_display_value(
    field: str,
    value: Any,
    *,
    answer_type: str,
) -> str:
    if value is None or value == "":
        return "-"
    if answer_type == "yes_no":
        if value is True:
            return "Yes"
        if value is False:
            return "No"
    if answer_type == "currency":
        from app.utils.helpers import format_currency

        return format_currency(value)
    if answer_type == "phone":
        from app.utils.helpers import format_phone_display

        return format_phone_display(str(value))
    return str(value)


def build_applicant_summary_rows(
    tenant: Any,
    questions: list[dict[str, Any]] | None,
) -> list[dict[str, str]]:
    """Ordered applicant summary rows driven by the admin question snapshot."""
    custom_fields: dict[str, Any] = {}
    nd = getattr(tenant, "normalized_data", None) if tenant is not None else None
    if isinstance(nd, dict):
        cf = nd.get("custom_fields")
        if isinstance(cf, dict):
            custom_fields = cf

    labels = field_labels_from_questions(questions)
    types = field_answer_types_from_questions(questions)
    rows: list[dict[str, str]] = []
    seen: set[str] = set()

    for q in normalize_questions(questions):
        if not q.get("active", True):
            continue
        answer_type = str(q.get("answer_type") or "text")
        for field in q.get("extract_fields") or []:
            field = str(field)
            if field in seen:
                continue
            seen.add(field)
            label = labels.get(field, field.replace("_", " "))
            value = _tenant_field_value(tenant, field, custom_fields)
            rows.append(
                {
                    "label": label,
                    "value": _format_summary_display_value(
                        field, value, answer_type=types.get(field, answer_type)
                    ),
                }
            )
    return rows


def build_preview_sample_paths(
    questions: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Labeled sample data paths for admin conversation preview."""
    normalized = normalize_questions(questions)
    gate_fields: dict[str, list[Any]] = {}
    for q in normalized:
        if not q.get("active", True):
            continue
        cond = q.get("conditional")
        if not cond:
            continue
        field = str(cond.get("field") or "")
        if not field:
            continue
        op = str(cond.get("operator") or "truthy")
        expected = cond.get("value")
        if op == "eq":
            gate_fields.setdefault(field, []).append(expected)
        elif op == "ne":
            gate_fields.setdefault(field, []).append(not expected)
        elif op == "truthy":
            gate_fields.setdefault(field, []).append(True)
        elif op == "falsy":
            gate_fields.setdefault(field, []).append(False)

    base: dict[str, Any] = {}
    for field in gate_fields:
        samples = gate_fields[field]
        if False in samples:
            base[field] = False
        elif any(s is False for s in samples):
            base[field] = False
        else:
            base[field] = False

    paths: list[dict[str, Any]] = [
        {
            "id": "default",
            "label": "Default (follow-ups hidden)",
            "data": dict(base),
        }
    ]

    for field, values in sorted(gate_fields.items()):
        unique_vals: list[Any] = []
        for val in values:
            if val not in unique_vals:
                unique_vals.append(val)
        for val in unique_vals:
            if val in (False, None, ""):
                continue
            data = dict(base)
            data[field] = val
            paths.append(
                {
                    "id": f"{field}__{val}".lower().replace(" ", "_"),
                    "label": f"{field.replace('_', ' ')} = {val}",
                    "data": data,
                }
            )

    if len(gate_fields) >= 2:
        combo = dict(base)
        for field, values in gate_fields.items():
            positive = next((v for v in values if v not in (False, None, "")), True)
            combo[field] = positive
        paths.append(
            {
                "id": "all_followups",
                "label": "All follow-up questions shown",
                "data": combo,
            }
        )

    return paths


def build_conversation_preview_flow(
    questions: list[dict[str, Any]] | None,
    sample_data: dict[str, Any],
    *,
    business: str,
    greeting_message: str = "",
    closing_message: str = "",
    language_code: str = "en",
) -> list[dict[str, str]]:
    """Simulate assistant turns for one preview sample path."""
    from app.core.screening_flow import build_greeting_intro

    normalized = normalize_questions(questions)
    if (greeting_message or "").strip():
        intro = str(greeting_message).replace("{property_name}", business).strip()
    else:
        intro = build_greeting_intro(business, language_code=language_code)

    first_state = first_active_question_state(normalized)
    first_q = next(
        (q for q in normalized if q.get("state") == first_state),
        normalized[0] if normalized else None,
    )
    flow: list[dict[str, str]] = [
        {
            "speaker": "AI",
            "text": (
                f"{intro} "
                f"{localized_question_text(first_q, language_code=language_code, key='question')}"
                if first_q
                else intro
            ),
        }
    ]

    for q in ordered_active_questions(normalized, sample_data):
        state = q.get("state", "")
        if first_q and state == first_q.get("state"):
            continue
        if should_skip_question(q, sample_data):
            flow.append(
                {
                    "speaker": "AI",
                    "text": f"(skipped — {state} not applicable on this path)",
                }
            )
            continue
        flow.append(
            {
                "speaker": "AI",
                "text": localized_question_text(
                    q,
                    language_code=language_code,
                    key="question",
                ),
            }
        )
        flow.append({"speaker": "Tenant", "text": "(tenant responds here)"})

    closing = (closing_message or "").strip()
    if closing:
        closing = closing.replace("{property_name}", business)
    else:
        if str(language_code or "en").lower().startswith("es"):
            closing = (
                "Gracias. Un especialista de alquiler revisara su informacion "
                "y se comunicara pronto."
            )
        else:
            closing = (
                "Thank you. A leasing specialist will review your information "
                "and follow up soon."
            )
    flow.append({"speaker": "AI", "text": closing})
    return flow


def build_tenant_edit_fields(
    tenant: Any,
    questions: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Editable applicant fields driven by the per-call question snapshot."""
    custom_fields: dict[str, Any] = {}
    nd = getattr(tenant, "normalized_data", None) if tenant is not None else None
    if isinstance(nd, dict):
        cf = nd.get("custom_fields")
        if isinstance(cf, dict):
            custom_fields = cf

    labels = field_labels_from_questions(questions)
    types = field_answer_types_from_questions(questions)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    for q in normalize_questions(questions):
        if not q.get("active", True):
            continue
        answer_type = str(q.get("answer_type") or "text")
        for field in q.get("extract_fields") or []:
            field = str(field)
            if field.endswith("_raw") or field in seen:
                continue
            seen.add(field)
            value = _tenant_field_value(tenant, field, custom_fields)
            rows.append(
                {
                    "field": field,
                    "label": labels.get(field, field.replace("_", " ")),
                    "answer_type": types.get(field, answer_type),
                    "value": value if value is not None else "",
                    "is_custom": field.startswith("custom_"),
                }
            )
    return rows
