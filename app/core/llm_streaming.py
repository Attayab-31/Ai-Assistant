"""Extract speakable sentences from streaming JSON LLM output (voice calls)."""

from __future__ import annotations

import json
import re

_SPEAKABLE_KEYS = ("response_text", "ack")

# Avoid flushing "Dr." / "Mr." / "3." as complete sentences mid-stream.
_ABBREV_TAIL_RE = re.compile(
    r"\b(?:Dr|Mr|Mrs|Ms|Miss|Prof|Sr|Jr|vs|etc|St|U\.S)\.\s*$",
    re.I,
)
_DECIMAL_TAIL_RE = re.compile(r"\d\.\s*$")


def _premature_sentence_end(sentence: str, *, trailing: str) -> bool:
    """True when punctuation looks like an abbreviation or decimal, not a full stop."""
    s = sentence.rstrip()
    if not trailing:
        return False
    return bool(_ABBREV_TAIL_RE.search(s) or _DECIMAL_TAIL_RE.search(s))


def partial_json_string_value(buffer: str, key: str) -> str | None:
    """Read a JSON string value for ``key`` even when the object is incomplete."""
    match = re.search(rf'"{re.escape(key)}"\s*:\s*"', buffer)
    if not match:
        return None
    i = match.end()
    chars: list[str] = []
    while i < len(buffer):
        ch = buffer[i]
        if ch == '"':
            break
        if ch == "\\" and i + 1 < len(buffer):
            chars.append(buffer[i + 1])
            i += 2
            continue
        chars.append(ch)
        i += 1
    raw = "".join(chars)
    try:
        return json.loads(f'"{raw}"')
    except json.JSONDecodeError:
        return raw


def _json_string_closed(buffer: str, key: str) -> bool:
    match = re.search(rf'"{re.escape(key)}"\s*:\s*"', buffer)
    if not match:
        return False
    i = match.end()
    while i < len(buffer):
        if buffer[i] == "\\" and i + 1 < len(buffer):
            i += 2
            continue
        if buffer[i] == '"':
            return True
        i += 1
    return False


def drain_speakable_sentences(
    partial_json: str,
    spoken_through: int,
) -> tuple[list[str], int]:
    """
    Return newly completed sentences from ``response_text`` / ``ack`` in a
    partial JSON buffer. ``spoken_through`` is a char offset in the combined
    speakable text we've already sent to TTS.
    """
    combined = ""
    for key in _SPEAKABLE_KEYS:
        value = partial_json_string_value(partial_json, key)
        if value:
            combined = f"{combined} {value}".strip() if combined else value.strip()
    if not combined or spoken_through >= len(combined):
        return [], spoken_through

    json_closed = any(_json_string_closed(partial_json, key) for key in _SPEAKABLE_KEYS)

    remainder = combined[spoken_through:].lstrip()
    if not remainder:
        return [], spoken_through

    sentences: list[str] = []
    offset = spoken_through + (len(combined[spoken_through:]) - len(remainder))
    deferred_prefix = ""
    while remainder and not sentences:
        match = re.match(r"^(.+?[.!?])(?:\s+|$)", remainder, re.DOTALL)
        if not match:
            break
        sentence = match.group(1).strip()
        consumed = len(match.group(0))
        trailing = remainder[consumed:].strip()
        if sentence and (trailing or json_closed):
            if _premature_sentence_end(sentence, trailing=trailing):
                deferred_prefix = f"{deferred_prefix}{sentence} ".strip() + " "
                offset += len(match.group(1))
                remainder = remainder[len(match.group(1)) :].lstrip()
                continue
            if deferred_prefix:
                sentence = f"{deferred_prefix.strip()} {sentence}".strip()
                deferred_prefix = ""
            sentences.append(sentence)
        else:
            break
        offset += consumed
        remainder = remainder[consumed:]

    # Voice TTFA: emit a phrase chunk before sentence punctuation when the
    # JSON stream is still open but we already have enough speakable text.
    if not sentences and remainder and not json_closed and len(remainder) >= 28:
        cut = remainder[:56].rfind(" ")
        if cut >= 20:
            phrase = remainder[:cut].strip()
            if phrase:
                sentences.append(phrase)
                offset = spoken_through + cut

    return sentences, offset
