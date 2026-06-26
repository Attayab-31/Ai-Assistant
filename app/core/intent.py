"""
Deterministic utterance helpers used by the screening flow.

These are the cheap regex fast-paths for yes/no parsing, refusal detection, and
a "is the caller asking us a question?" check. All LLM-based intent
classification was removed when the call flow became LLM-first: a single
conversational model call (see ``conversation.build_system_prompt`` and
``call_handler.process_tenant_speech``) now resolves intent, FAQ answering, and
field extraction for every turn.
"""

from __future__ import annotations

import re

from app.core.screening_flow import normalize_text

# ── Yes / no lexicons (expanded beyond bare "yes"/"no") ─────────────────────

_YES_RE = re.compile(
    r"\b(yes|yeah|yep|yup|sure|correct|right|affirmative|absolutely|definitely|"
    r"uh[ -]?huh|mm[ -]?hmm|mhm|i do|we do|have one|have a|have two|of course|"
    r"you bet|indeed|that's right|thats right)\b",
    re.I,
)
_NO_RE = re.compile(
    r"\b(no|nope|nah|none|not any|don't have|do not have|never|negative|"
    r"not at all|no way|no sir|no ma'am|no maam|i don't|we don't|"
    r"don't|do not|cannot|can't|not really)\b",
    re.I,
)

# Implicit answers — no literal "yes"/"no" word required.
_IMPLICIT_PETS_NO = re.compile(
    r"\b(pet.?free|no pets|no animals|animal.?free|without pets|"
    r"don't have any pets|do not have any pets|zero pets|nothing like that|"
    r"not at the moment|not right now|allergic)\b",
    re.I,
)
_IMPLICIT_PETS_YES = re.compile(
    r"\b(we have a|got a|have a|have two|have three|yes we|dog|cat|puppy|"
    r"kitten|bird|hamster|rabbit|reptile|snake|lizard|ferret)\b",
    re.I,
)
_IMPLICIT_EVICTION_NO = re.compile(
    r"\b(clean (rental )?history|never evicted|no evictions?|always paid|"
    r"never had trouble|good standing|paid on time|never been evicted|"
    r"no landlord issues|spotless record|never filed)\b",
    re.I,
)
_IMPLICIT_EVICTION_YES = re.compile(
    r"\b(was evicted|got evicted|had an eviction|eviction on|eviction in|"
    r"landlord.?tenant court|unlawful detainer|court filing|evicted from)\b",
    re.I,
)

# ── Refusal lexicon (expanded) ───────────────────────────────────────────────

_REFUSAL_RE = re.compile(
    r"(won't tell|will not tell|won't say|will not say|won't explain|"
    r"will not explain|don't want to (say|tell|share|explain|answer)|"
    r"do not want to|not going to tell|prefer not to|rather not|"
    r"can't share|cannot share|none of your business|won't answer|"
    r"will not answer|i won't|i will not tell|"
    r"that's private|that is private|that's personal|that is personal|"
    r"none of your|mind your own|why do you need|why would you need|"
    r"skip that|pass on that|don't feel comfortable|not comfortable sharing|"
    r"rather keep that|rather not say|rather not share|no comment|"
    r"not answering that|i'd rather not|i would rather not|"
    r"\bnot tell\b|\bnot say\b|\bnot explain\b|\bnot answer\b)",
    re.I,
)

def detect_refusal(text: str) -> bool:
    """Regex refusal detector — expanded lexicon."""
    return bool(_REFUSAL_RE.search(text or ""))


def _implicit_yes_no(domain: str, norm: str) -> bool | None:
    if domain == "pets":
        if _IMPLICIT_PETS_NO.search(norm):
            return False
        if _IMPLICIT_PETS_YES.search(norm) and not re.search(
            r"\b(don't|do not|no pets|no animals|pet.?free)\b", norm
        ):
            return True
    if domain == "eviction":
        if _IMPLICIT_EVICTION_NO.search(norm):
            return False
        if _IMPLICIT_EVICTION_YES.search(norm):
            return True
    return None


def parse_yes_no(text: str, *, domain: str = "generic") -> bool | None:
    """Resolve yes/no with implicit domain answers and correction-aware ordering.

    When both yes and no signals appear ("no wait, yes I do"), the LAST signal
    wins — matching how humans correct themselves mid-sentence.
    """
    norm = normalize_text(text)
    if not norm:
        return None

    implicit = _implicit_yes_no(domain, norm)
    if implicit is not None:
        return implicit

    yes_hits = list(_YES_RE.finditer(norm))
    no_hits = list(_NO_RE.finditer(norm))

    if yes_hits and no_hits:
        last_yes = yes_hits[-1].start()
        last_no = no_hits[-1].start()
        return last_yes > last_no

    if no_hits and not yes_hits:
        return False
    if yes_hits and not no_hits:
        return True
    return None
