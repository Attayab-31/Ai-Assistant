"""Map caller language choice to Deepgram / Groq STT and TTS voice settings."""

from __future__ import annotations

from app.core.question_flow import canonical_language_code

DEFAULT_DEEPGRAM_STT_LANG_EN = "en-US"
DEFAULT_DEEPGRAM_STT_LANG_ES = "es"
DEFAULT_DEEPGRAM_TTS_VOICE_ES = "aura-2-estrella-es"
DEFAULT_GOOGLE_TTS_VOICE_ES = "es-US-Neural2-A"


def is_spanish_code(language_code: str | None) -> bool:
    code = canonical_language_code(language_code) or str(language_code or "").strip().lower()
    return code.startswith("es")


def deepgram_stt_language(language_code: str | None) -> str:
    """Deepgram live / batch STT language token."""
    return DEFAULT_DEEPGRAM_STT_LANG_ES if is_spanish_code(language_code) else DEFAULT_DEEPGRAM_STT_LANG_EN


def groq_stt_language(language_code: str | None) -> str:
    """Groq Whisper language token."""
    return "es" if is_spanish_code(language_code) else "en"


def deepgram_tts_voice(
    *,
    language_code: str | None,
    english_voice: str,
    spanish_voice: str,
) -> str:
    if is_spanish_code(language_code):
        return (spanish_voice or "").strip() or DEFAULT_DEEPGRAM_TTS_VOICE_ES
    return english_voice


def google_tts_voice(
    *,
    language_code: str | None,
    english_voice: str,
    spanish_voice: str,
) -> str:
    if is_spanish_code(language_code):
        return (spanish_voice or "").strip() or DEFAULT_GOOGLE_TTS_VOICE_ES
    return english_voice


def google_tts_language_code(language_code: str | None) -> str:
    return "es-US" if is_spanish_code(language_code) else "en-US"
