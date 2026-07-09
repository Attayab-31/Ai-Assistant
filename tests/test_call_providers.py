"""Frozen call provider bundle — no live registry fallback mid-call."""

import pytest

from app.core.call_settings import CallProviderBundle
from app.core.conversation import ConversationSession


def _minimal_bundle() -> CallProviderBundle:
    return CallProviderBundle(
        llm=object(),
        stt=object(),
        tts=object(),
        llm_name="groq",
        stt_name="deepgram",
        tts_name="deepgram",
        auto_fallback_enabled=True,
    )


def test_get_call_providers_returns_attached_bundle():
    from app.core.call_handler import get_call_providers

    bundle = _minimal_bundle()
    session = ConversationSession(
        call_id="call-1",
        phone_number="+15551234567",
        call_providers=bundle,
    )
    assert get_call_providers(session) is bundle


def test_get_call_providers_raises_when_bundle_missing():
    from app.core.call_handler import get_call_providers

    session = ConversationSession(call_id="call-2", phone_number="+15551234567")
    assert session.call_providers is None

    with pytest.raises(RuntimeError, match="frozen provider bundle"):
        get_call_providers(session)


@pytest.mark.asyncio
async def test_transcribe_buffer_requires_frozen_bundle():
    from app.core.audio_stream import transcribe_buffer

    session = ConversationSession(call_id="call-3", phone_number="+15551234567")

    with pytest.raises(RuntimeError, match="frozen provider bundle"):
        await transcribe_buffer(b"\xff" * 320, session=session)
