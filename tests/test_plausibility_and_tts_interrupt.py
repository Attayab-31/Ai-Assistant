"""Plausibility clarify localization and turn TTS interruption."""

import asyncio

import pytest


@pytest.mark.asyncio
async def test_plausibility_clarify_fallback_uses_spanish(monkeypatch):
    from app.core.call_handler import process_tenant_speech
    from app.core.conversation import ConversationSession

    session = ConversationSession(
        call_id="plaus-es",
        phone_number="+1",
        call_language="es",
        current_state="Q_INCOME",
        questions=[
            {
                "state": "Q_INCOME",
                "question": "What is your monthly income?",
                "extract_fields": ["monthly_income"],
                "active": True,
                "order": 1,
            }
        ],
    )

    async def _fake_llm(*_args, **_kwargs):
        return {
            "response_text": "",
            "understood": True,
            "intent": "answer",
            "relevance": "on_topic",
            "question_complete": False,
            "extracted_data": {"monthly_income": "500"},
            "plausibility_issue": "hourly wage not monthly",
            "consistency_issue": None,
        }

    async def _fake_synthesize(*_args, **_kwargs):
        return [b"ok"]

    monkeypatch.setattr(
        "app.core.call_handler.get_llm_response_with_fallback", _fake_llm
    )
    monkeypatch.setattr(
        "app.core.call_handler.synthesize_speech_parts", _fake_synthesize
    )

    response_text, _audio_parts, complete = await process_tenant_speech(
        session, "quinientos por hora"
    )

    assert complete is False
    assert "Solo para confirmar" in response_text
    assert "Just to make sure" not in response_text


@pytest.mark.asyncio
async def test_parallel_tts_skips_follow_up_after_interrupt(monkeypatch):
    from app.core.call_handler import (
        _synthesize_ack_followup_parallel,
        interrupt_turn_tts,
    )
    from app.core.conversation import ConversationSession

    session = ConversationSession(call_id="tts-int", phone_number="+1")
    enqueued: list[tuple[bytes, bool]] = []

    async def _slow_synth(text, _session, **_kwargs):
        if text == "follow":
            await asyncio.sleep(0.05)
        else:
            await asyncio.sleep(0.01)
        return f"audio-{text}".encode()

    async def _on_part(audio: bytes, is_last: bool) -> None:
        enqueued.append((audio, is_last))
        if audio == b"audio-ack":
            interrupt_turn_tts(session)

    monkeypatch.setattr(
        "app.core.call_handler.synthesize_with_fallback", _slow_synth
    )

    parts = await _synthesize_ack_followup_parallel(
        "ack",
        "follow",
        session,
        on_part_ready=_on_part,
    )

    assert len(parts) == 1
    assert len(enqueued) == 1
    assert enqueued[0][0] == b"audio-ack"
    assert enqueued[0][1] is False


@pytest.mark.asyncio
async def test_pipelined_tts_skips_remaining_chunks_after_interrupt(monkeypatch):
    from app.core.call_handler import (
        _synthesize_chunks_pipelined,
        interrupt_turn_tts,
    )
    from app.core.conversation import ConversationSession

    session = ConversationSession(call_id="pipe-int", phone_number="+1")
    enqueued: list[bytes] = []

    async def _slow_synth(text, _session, **_kwargs):
        await asyncio.sleep(0.01)
        return f"audio-{text}".encode()

    async def _on_part(audio: bytes, _is_last: bool) -> None:
        enqueued.append(audio)
        if audio == b"audio-chunk-0":
            interrupt_turn_tts(session)

    monkeypatch.setattr(
        "app.core.call_handler.synthesize_with_fallback", _slow_synth
    )

    parts = await _synthesize_chunks_pipelined(
        ["chunk-0", "chunk-1", "chunk-2"],
        session,
        on_part_ready=_on_part,
    )

    assert len(parts) == 1
    assert enqueued == [b"audio-chunk-0"]
