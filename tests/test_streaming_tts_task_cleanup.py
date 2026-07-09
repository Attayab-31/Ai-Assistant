"""Ensure sentence-level streaming TTS tasks are cleaned on failures."""

import asyncio

import pytest


@pytest.mark.asyncio
async def test_collect_streamed_llm_raw_cancels_tts_tasks_on_timeout(monkeypatch):
    from app.core.call_handler import _collect_streamed_llm_raw

    cancelled = asyncio.Event()
    emitted = {"done": False}

    class Provider:
        async def stream_response(self, **_kwargs):
            yield "Hello there."
            await asyncio.sleep(0.2)
            yield "never"

    async def on_speakable_sentence(_sentence: str):
        try:
            await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    def _drain(_buffer: str, spoken_through: int):
        if not emitted["done"]:
            emitted["done"] = True
            return (["Hello there."], spoken_through + 1)
        return ([], spoken_through)

    monkeypatch.setattr("app.core.llm_streaming.drain_speakable_sentences", _drain)

    with pytest.raises(TimeoutError):
        await _collect_streamed_llm_raw(
            Provider(),
            system_prompt="sys",
            messages=[],
            temperature=0.0,
            max_tokens=16,
            attempt_timeout=0.05,
            on_speakable_sentence=on_speakable_sentence,
        )

    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_collect_streamed_llm_raw_cancels_tts_tasks_on_stream_error(monkeypatch):
    from app.core.call_handler import _collect_streamed_llm_raw

    cancelled = asyncio.Event()
    emitted = {"done": False}

    class Provider:
        async def stream_response(self, **_kwargs):
            yield "Thanks."
            await asyncio.sleep(0.01)
            raise RuntimeError("boom")

    async def on_speakable_sentence(_sentence: str):
        try:
            await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    def _drain(_buffer: str, spoken_through: int):
        if not emitted["done"]:
            emitted["done"] = True
            return (["Thanks."], spoken_through + 1)
        return ([], spoken_through)

    monkeypatch.setattr("app.core.llm_streaming.drain_speakable_sentences", _drain)

    raw = await _collect_streamed_llm_raw(
        Provider(),
        system_prompt="sys",
        messages=[],
        temperature=0.0,
        max_tokens=16,
        attempt_timeout=0.2,
        on_speakable_sentence=on_speakable_sentence,
    )
    assert raw is None
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_collect_streamed_llm_raw_cancels_tts_tasks_on_cancelled_error(monkeypatch):
    from app.core.call_handler import _collect_streamed_llm_raw

    cancelled = asyncio.Event()
    emitted = {"done": False}

    class Provider:
        async def stream_response(self, **_kwargs):
            yield "Hello."
            await asyncio.sleep(1.0)

    async def on_speakable_sentence(_sentence: str):
        try:
            await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    def _drain(_buffer: str, spoken_through: int):
        if not emitted["done"]:
            emitted["done"] = True
            return (["Hello."], spoken_through + 1)
        return ([], spoken_through)

    monkeypatch.setattr("app.core.llm_streaming.drain_speakable_sentences", _drain)

    task = asyncio.create_task(
        _collect_streamed_llm_raw(
            Provider(),
            system_prompt="sys",
            messages=[],
            temperature=0.0,
            max_tokens=16,
            attempt_timeout=3.0,
            on_speakable_sentence=on_speakable_sentence,
        )
    )
    await asyncio.sleep(0.05)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert cancelled.is_set()
