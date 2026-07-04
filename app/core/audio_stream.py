"""
app/core/audio_stream.py — Shared bidirectional audio WebSocket handler.

Used by both the production Telnyx stream (/telnyx/stream) and the test
console stream (/test/api/stream) so both paths exercise identical logic:
reader → STT → LLM → TTS → sender, with silence timeout and max duration.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import re
import struct
import time
import wave
from collections.abc import Awaitable, Callable

import orjson
from fastapi import WebSocket, WebSocketDisconnect

from app.core import call_handler
from app.core.call_logging import Phase, vdebug, verror, vinfo, vwarn
from app.core.call_settings import CallProviderBundle
from app.core.conversation import (
    STT_EMPTY_STRIKE_LIMIT,
    ConversationSession,
    is_echo_of_agent,
    mark_recovery_played,
    plan_turn_timeout_recovery,
    reset_turn_streaming,
    should_suppress_silence_nudge,
    unsynthesized_speech_remainder,
)
from app.core.streaming_stt import DeepgramStreamingSession
from app.utils.audio import (
    any_audio_to_mulaw,
    chunk_audio,
    decode_telnyx_payload,
    encode_telnyx_payload,
    is_silence,
    mulaw_to_wav,
)
from config import settings

logger = logging.getLogger(__name__)


def _json_default(obj):
    """Serialize values orjson can't handle natively (Decimal, etc.)."""
    from decimal import Decimal

    if isinstance(obj, Decimal):
        # Whole numbers as int, otherwise float — keeps the inspector readable.
        return int(obj) if obj == obj.to_integral_value() else float(obj)
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return str(obj)


# Defaults match production Telnyx tuning. Deepgram utterance_end handles
# turn detection on the streaming path; these buffer thresholds apply only
# when streaming STT is unavailable (non-Deepgram fallback).
SILENCE_THRESHOLD_CHUNKS = 35  # ~700 ms trailing silence ends utterance
MIN_UTTERANCE_BYTES = 1600  # ~200 ms of mulaw at 8 kHz
# ~100 ms of non-silence before an utterance counts
MIN_SPEECH_CHUNKS = 5
# flush every ~2 s of continuous speech (no pause needed)
FORCE_FLUSH_BYTES = 16000 * 2
MAX_BUFFER_BYTES = 16000 * 10  # ~10 s safety cap
TURN_PAUSE_SECONDS = 0.18  # natural pause between ack and next question


def _turn_budget_seconds(session: ConversationSession) -> float:
    from app.core.conversation import turn_budget_seconds

    return turn_budget_seconds(session)

# Interruption/barge-in support
BARGE_IN_ENABLED = True
# Require this many consecutive non-silent chunks while the AI is speaking
# before treating it as a real barge-in (filters out line noise / clicks).
BARGE_IN_MIN_SPEECH_CHUNKS = 3

# Short answers that may legitimately repeat across consecutive questions.
_DEDUP_SHORT_ANSWERS = frozenset(
    {
        "yes",
        "no",
        "yeah",
        "nope",
        "yep",
        "nah",
        "ok",
        "okay",
        "sure",
        "correct",
        "right",
        "wrong",
        "none",
        "zero",
    }
)

# outbound queue item: (mulaw_bytes, turn_end)
OutboundItem = tuple[bytes, bool]

# utterance queue item: (format, audio_bytes)  format = "mulaw" | "wav"
UtteranceItem = tuple[str, bytes]

# Streaming STT: finalized transcript strings from Deepgram live endpointing
TranscriptItem = str

_groq_stt_fallback = None


def _use_streaming_stt(session: ConversationSession) -> bool:
    """True when this call should use Deepgram live streaming for turn detection."""
    from app.core.call_handler import get_call_providers
    from app.providers.stt.deepgram_stt import DeepgramSTTProvider

    providers = get_call_providers(session)
    return isinstance(providers.stt, DeepgramSTTProvider)


async def _enqueue_utterance(
    queue: asyncio.Queue[UtteranceItem],
    audio: bytes,
    call_id: str,
    *,
    audio_format: str = "mulaw",
) -> None:
    """Enqueue without blocking the WebSocket reader task."""
    try:
        await queue.put((audio_format, audio))
    except Exception as e:
        logger.error("[%s] Failed to enqueue utterance: %s", call_id, e)


def _get_groq_stt_fallback():
    global _groq_stt_fallback
    if _groq_stt_fallback is None:
        from app.providers.stt.groq_stt import GroqSTTProvider

        _groq_stt_fallback = GroqSTTProvider()
    return _groq_stt_fallback


def audio_duration_seconds(data: bytes, fmt: str = "mulaw") -> float:
    """Estimate audio duration for STT timeout scaling."""
    if not data:
        return 0.0
    if fmt == "mulaw":
        return len(data) / 8000.0
    try:
        with wave.open(io.BytesIO(data), "rb") as wav:
            return wav.getnframes() / float(wav.getframerate())
    except (wave.Error, struct.error, EOFError) as e:
        logger.debug("Could not read WAV duration, estimating: %s", e)
        return len(data) / 32000.0


def stt_timeout_for_duration(duration_sec: float) -> float:
    """Scale STT timeout with clip length — long answers need more time."""
    return max(6.0, min(30.0, duration_sec * 1.5 + 3.0))


async def transcribe_buffer(
    audio_bytes: bytes,
    *,
    input_format: str = "mulaw",
    session: ConversationSession | None = None,
) -> str:
    """Transcribe audio using the call's STT provider (+ Groq fallback)."""
    from app.core.call_handler import get_call_providers
    from config import provider_registry

    if not audio_bytes:
        return ""

    if session is not None:
        providers = get_call_providers(session)
    else:
        providers = CallProviderBundle(
            llm=provider_registry.llm,
            stt=provider_registry.stt,
            tts=provider_registry.tts,
            llm_name=provider_registry.llm_name,
            stt_name=provider_registry.stt_name,
            tts_name=provider_registry.tts_name,
            auto_fallback_enabled=provider_registry.auto_fallback_enabled,
            stt_fallback_provider=provider_registry.stt_fallback_provider,
        )

    duration_sec = audio_duration_seconds(audio_bytes, input_format)
    timeout = stt_timeout_for_duration(duration_sec)
    vinfo(
        logger,
        f"STT input {len(audio_bytes)} bytes {input_format} (~{duration_sec:.1f}s)",
        session=session,
        phase=Phase.STT,
        service="stt",
        provider=providers.stt_name,
        bytes=len(audio_bytes),
        duration_s=round(duration_sec, 2),
        timeout_s=timeout,
    )

    async def _primary() -> str:
        if input_format == "wav":
            from app.providers.stt.deepgram_stt import DeepgramSTTProvider

            if isinstance(providers.stt, DeepgramSTTProvider):
                return await providers.stt.transcribe_wav_chunk(audio_bytes)
            return await providers.stt.transcribe_chunk(any_audio_to_mulaw(audio_bytes))
        return await providers.stt.transcribe_chunk(audio_bytes)

    try:
        transcript = await asyncio.wait_for(_primary(), timeout=timeout)
        if transcript.strip():
            vinfo(
                logger,
                f"STT result: {transcript[:80]!r}",
                session=session,
                phase=Phase.STT,
                service="stt",
                provider=providers.stt_name,
            )
            if session is not None:
                session.stt_provider = providers.stt_name
            return transcript
        vwarn(
            logger,
            "Primary STT returned empty transcript",
            session=session,
            phase=Phase.STT,
            service="stt",
            provider=providers.stt_name,
            reason="empty",
        )
    except TimeoutError:
        vwarn(
            logger,
            f"STT timeout after {timeout:.0f}s — trying fallback",
            session=session,
            phase=Phase.STT,
            service="stt",
            provider=providers.stt_name,
            reason="timeout",
            timeout_s=timeout,
        )
    except Exception as e:
        verror(
            logger,
            f"Primary STT error: {e}",
            session=session,
            phase=Phase.STT,
            service="stt",
            provider=providers.stt_name,
            reason="error",
        )

    if not providers.auto_fallback_enabled:
        return ""

    # Admin-chosen backup ears: "none" disables, "auto" uses the other provider,
    # or a specific one is forced. Only Groq and Deepgram exist for STT.
    pref = (getattr(providers, "stt_fallback_provider", "auto") or "auto").lower()
    if pref == "none":
        return ""

    try:
        from app.providers.stt.deepgram_stt import DeepgramSTTProvider
        from app.providers.stt.groq_stt import GroqSTTProvider

        primary_is_groq = isinstance(providers.stt, GroqSTTProvider)
        groq_ok = not primary_is_groq and bool(settings.groq_api_key)
        deepgram_ok = not isinstance(providers.stt, DeepgramSTTProvider) and bool(
            settings.deepgram_api_key
        )

        if pref == "groq" and groq_ok:
            chosen = "groq"
        elif pref == "deepgram" and deepgram_ok:
            chosen = "deepgram"
        elif groq_ok:  # "auto" default prefers Groq Whisper for buffered audio
            chosen = "groq"
        elif deepgram_ok:
            chosen = "deepgram"
        else:
            chosen = ""

        if chosen:
            mulaw = (
                audio_bytes
                if input_format == "mulaw"
                else any_audio_to_mulaw(audio_bytes)
            )
            if chosen == "groq":
                fallback_stt = _get_groq_stt_fallback()
            else:
                fallback_stt = DeepgramSTTProvider(model=settings.deepgram_model)
            transcript = await asyncio.wait_for(
                fallback_stt.transcribe_chunk(mulaw),
                timeout=timeout,
            )
            if transcript.strip():
                vinfo(
                    logger,
                    f"STT fallback ({chosen}): {transcript[:80]!r}",
                    session=session,
                    phase=Phase.STT_FALLBACK,
                    service="stt",
                    provider=chosen,
                )
                if session is not None:
                    session.stt_provider = chosen
                return transcript
    except TimeoutError:
        verror(
            logger,
            "STT fallback timed out",
            session=session,
            phase=Phase.STT_FALLBACK,
            service="stt",
            reason="timeout",
        )
    except Exception as e:
        verror(
            logger,
            f"STT fallback failed: {e}",
            session=session,
            phase=Phase.STT_FALLBACK,
            service="stt",
            reason="error",
        )

    return ""


async def run_bidirectional_audio_stream(
    websocket: WebSocket,
    call_id: str,
    session: ConversationSession,
    *,
    hangup_on_complete: bool = True,
    emit_debug_events: bool = False,
    on_complete: Callable[[], Awaitable[None]] | None = None,
) -> None:
    """
    Run the production-grade reader / worker / sender / watchdog loop.

    Args:
        hangup_on_complete: Call Telnyx hangup when the conversation ends.
        emit_debug_events: Send transcript/state/complete events (test console).
        on_complete: Optional async callback after the stream ends.
    """
    silence_timeout = getattr(session, "silence_timeout_seconds", 12) or 12
    max_call_duration = getattr(session, "max_call_duration_seconds", 600) or 600

    # Honor the admin-configured silence timeout exactly. A small hard floor of
    # 3s only guards against a pathological 0/1 that would cut callers off
    # mid-breath; any sensible configured value is used as-is.
    listen_timeout = max(silence_timeout, 3)

    streaming_stt_enabled = _use_streaming_stt(session)
    utterance_queue: asyncio.Queue[UtteranceItem] = asyncio.Queue(maxsize=8)
    transcript_queue: asyncio.Queue[TranscriptItem | None] = asyncio.Queue(maxsize=8)
    audio_feed_queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=256)
    outbound_queue: asyncio.Queue[OutboundItem | bytes] = asyncio.Queue()
    ai_speaking = asyncio.Event()
    tenant_may_speak = asyncio.Event()
    listen_active = asyncio.Event()
    stop_event = asyncio.Event()
    turn_in_progress = asyncio.Event()
    current_turn_task: list[asyncio.Task | None] = [None]
    call_handler.register_stream_stop(call_id, stop_event)
    # A hangup may have landed while this WebSocket was still connecting (before
    # the stop_event existed). If so, wind down immediately instead of running a
    # full turn against a caller who is already gone.
    if getattr(session, "pending_hangup", False):
        logger.info("[%s] Hangup arrived before stream start — stopping", call_id)
        stop_event.set()
    streaming_stt: DeepgramStreamingSession | None = None
    # Set by the reader on a real barge-in so the sender aborts mid-utterance.
    interrupt_event = asyncio.Event()
    # Transcript-gated barge-in: ignore speech detected within this monotonic
    # deadline. Covers the echo tail right after the agent stops (speaker decay
    # re-entering the mic on hands-free) so it can't self-trigger a barge-in.
    barge_in_cooldown_until = 0.0
    BARGE_IN_COOLDOWN_S = 0.2
    # Telnyx: bidirectional RTP is not ready until the stream "start" (or first
    # media) arrives. Greeting audio sent before that can clip the opening.
    stream_ready = asyncio.Event()
    if emit_debug_events:
        stream_ready.set()

    async def enqueue_audio(
        audio: bytes | list[bytes],
        *,
        turn_end: bool = True,
    ) -> None:
        """Queue one or more mulaw segments; only the last marks end-of-turn."""
        if isinstance(audio, list):
            parts = [p for p in audio if p]
            if not parts:
                if turn_end:
                    vdebug(
                        logger,
                        "Enqueue empty flush (turn_end)",
                        session=session,
                        call_id=call_id,
                        phase=Phase.AUDIO_ENQUEUE,
                        detail="turn_end=True bytes=0",
                    )
                    await outbound_queue.put((b"", True))
                return
            for i, part in enumerate(parts):
                is_last = turn_end and i == len(parts) - 1
                vdebug(
                    logger,
                    f"Enqueue audio part {i + 1}/{len(parts)} ({len(part)} bytes)",
                    session=session,
                    call_id=call_id,
                    phase=Phase.AUDIO_ENQUEUE,
                    bytes=len(part),
                    detail=f"turn_end={is_last}",
                )
                await outbound_queue.put((part, is_last))
                if not is_last:
                    await asyncio.sleep(TURN_PAUSE_SECONDS)
        elif audio:
            vdebug(
                logger,
                f"Enqueue audio ({len(audio)} bytes)",
                session=session,
                call_id=call_id,
                phase=Phase.AUDIO_ENQUEUE,
                bytes=len(audio),
                detail=f"turn_end={turn_end}",
            )
            await outbound_queue.put((audio, turn_end))
        elif turn_end:
            vdebug(
                logger,
                "Enqueue empty flush (turn_end)",
                session=session,
                call_id=call_id,
                phase=Phase.AUDIO_ENQUEUE,
                detail="turn_end=True bytes=0",
            )
            await outbound_queue.put((b"", True))

    async def _await_outbound_playback_done(*, timeout: float = 120.0) -> None:
        """Wait until all queued outbound audio has finished playing."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if outbound_queue.empty() and not ai_speaking.is_set():
                # One tick so the sender can finish its finally block.
                await asyncio.sleep(0.05)
                if outbound_queue.empty() and not ai_speaking.is_set():
                    return
            await asyncio.sleep(0.05)
        logger.warning(
            "[%s] Outbound playback wait timed out after %.0fs",
            call_id,
            timeout,
        )

    def _drain_pending_input() -> int:
        """Discard stale STT/utterance input from the previous turn."""
        dropped = 0
        queues: tuple[asyncio.Queue, ...] = (
            (transcript_queue,) if streaming_stt_enabled else (utterance_queue,)
        )
        for q in queues:
            while not q.empty():
                try:
                    q.get_nowait()
                    dropped += 1
                except asyncio.QueueEmpty:
                    break
        if dropped:
            logger.debug("[%s] Dropped %s stale input item(s)", call_id, dropped)
        return dropped

    def _apply_barge_in() -> None:
        """Stop current playback and hand the turn back to the caller.

        Barge-in always means "stop talking and listen" — we discard any queued
        outbound audio and release the listen gate so the worker processes the
        caller's utterance as their answer. We deliberately do NOT auto re-ask
        or drop the caller's input: doing so created a re-ask feedback loop and
        swallowed real answers.
        """
        interrupt_event.set()
        listen_active.set()
        while not outbound_queue.empty():
            try:
                outbound_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        ai_speaking.clear()
        tenant_may_speak.set()
        task = current_turn_task[0]
        if task is not None and not task.done():
            task.cancel()
        # Tell the test-console browser to stop its local playback immediately
        # (the server is the barge-in authority; the browser just obeys).
        if emit_debug_events:
            asyncio.create_task(_emit("interrupt"))

    async def _emit(event: str, **payload) -> None:
        if not emit_debug_events:
            return
        try:
            # orjson + a default hook so non-JSON-native values from extracted
            # data (e.g. Decimal income, datetime) don't break the inspector.
            data = orjson.dumps(
                {"event": event, **payload}, default=_json_default
            ).decode()
            await websocket.send_text(data)
        except Exception as e:
            logger.debug("[%s] debug emit failed: %s", call_id, e)

    def _should_feed_stt() -> bool:
        # Keep feeding live STT during LLM think time so transcript-gated
        # barge-in can fire as soon as the caller speaks. Stale finalized
        # transcripts are dropped in stt_bridge / drained at turn start.
        return True

    async def _on_speech_started(text: str) -> None:
        """Transcript-gated barge-in: Deepgram heard real words from the caller.

        This replaces raw audio-energy VAD for streaming sessions. Because the
        speech model only emits words for actual speech (not HVAC, clicks, or
        silence), this all but eliminates the false barge-ins that energy VAD
        produced on noisy / hands-free setups.
        """
        nonlocal barge_in_cooldown_until
        if not BARGE_IN_ENABLED or not ai_speaking.is_set():
            return
        if time.monotonic() < barge_in_cooldown_until:
            return
        # Guard against the agent's own (echoed) speech being transcribed back
        # as a "caller" interruption on speakerphone / hands-free setups.
        if is_echo_of_agent(text, session):
            logger.debug(
                "[%s] Ignoring agent echo for barge-in: %r", call_id, text[:40]
            )
            return
        session.interruption_count += 1
        logger.info(
            "[%s] Barge-in (speech detected: %r, interruption #%s)",
            call_id,
            text[:40],
            session.interruption_count,
        )
        _apply_barge_in()

    async def reader() -> None:
        audio_buffer = bytearray()
        silence_chunks = 0
        speech_chunks = 0
        barge_in_speech_chunks = 0
        try:
            while not stop_event.is_set():
                try:
                    raw_message = await asyncio.wait_for(
                        websocket.receive_text(), timeout=1.0
                    )
                except TimeoutError:
                    if await call_handler.check_stream_stop_signal(call_id):
                        stop_event.set()
                        break
                    continue
                try:
                    message = orjson.loads(raw_message)
                except orjson.JSONDecodeError:
                    continue
                event = message.get("event", "")

                if event == "media":
                    if not emit_debug_events and not stream_ready.is_set():
                        stream_ready.set()
                    payload_b64 = message.get("media", {}).get("payload", "")
                    chunk = decode_telnyx_payload(payload_b64)
                    if not chunk:
                        continue
                    chunk_is_silence = (
                        is_silence(chunk)
                        if (
                            message.get("media", {}).get("encoding", "mulaw") == "mulaw"
                        )
                        else False
                    )

                    # Energy-based barge-in is a FALLBACK only — used when the
                    # session has no live transcript stream to gate on. When
                    # streaming STT is active (the normal case) barge-in is
                    # transcript-gated in _on_speech_started instead, which does
                    # not false-trigger on noise / echo / silence.
                    if (
                        BARGE_IN_ENABLED
                        and not streaming_stt_enabled
                        and ai_speaking.is_set()
                    ):
                        if chunk_is_silence:
                            barge_in_speech_chunks = max(0, barge_in_speech_chunks - 1)
                            continue
                        barge_in_speech_chunks += 1
                        if barge_in_speech_chunks < BARGE_IN_MIN_SPEECH_CHUNKS:
                            continue
                        session.interruption_count += 1
                        logger.info(
                            "[%s] Barge-in detected (interruption #%s)",
                            call_id,
                            session.interruption_count,
                        )
                        _apply_barge_in()
                        barge_in_speech_chunks = 0

                    if streaming_stt_enabled:
                        if _should_feed_stt():
                            try:
                                audio_feed_queue.put_nowait(chunk)
                            except asyncio.QueueFull:
                                logger.debug(
                                    "[%s] audio feed queue full, dropping chunk",
                                    call_id,
                                )
                        continue

                    audio_buffer.extend(chunk)

                    if chunk_is_silence:
                        silence_chunks += 1
                    else:
                        silence_chunks = 0
                        speech_chunks += 1

                    should_flush = (
                        (
                            silence_chunks >= SILENCE_THRESHOLD_CHUNKS
                            and len(audio_buffer) >= MIN_UTTERANCE_BYTES
                            and speech_chunks >= MIN_SPEECH_CHUNKS
                        )
                        or (
                            len(audio_buffer) >= FORCE_FLUSH_BYTES
                            and speech_chunks >= MIN_SPEECH_CHUNKS
                        )
                        or len(audio_buffer) >= MAX_BUFFER_BYTES
                    )
                    if should_flush:
                        if speech_chunks >= MIN_SPEECH_CHUNKS:
                            buf_len = len(audio_buffer)
                            utterance = bytes(audio_buffer)
                            logger.info(
                                f"[{call_id}] Utterance detected ({buf_len} bytes)"
                            )
                            asyncio.create_task(
                                _enqueue_utterance(utterance_queue, utterance, call_id)
                            )
                            if emit_debug_events:
                                await _emit(
                                    "debug",
                                    message=f"Utterance queued ({buf_len} bytes)",
                                )
                        else:
                            logger.debug(
                                f"[{call_id}] Discarding silence-only buffer "
                                f"({len(audio_buffer)} bytes)"
                            )
                        audio_buffer.clear()
                        silence_chunks = 0
                        speech_chunks = 0
                    elif (
                        silence_chunks >= SILENCE_THRESHOLD_CHUNKS
                        and len(audio_buffer) >= MIN_UTTERANCE_BYTES
                    ):
                        audio_buffer.clear()
                        silence_chunks = 0
                        speech_chunks = 0
                elif event in ("stop", "end"):
                    logger.info("[%s] Stream stop event", call_id)
                    stop_event.set()
                    return
                elif event == "start":
                    logger.debug("[%s] Stream start event", call_id)
                    if not emit_debug_events:
                        stream_ready.set()
        except WebSocketDisconnect:
            logger.info("[%s] WS disconnect in reader", call_id)
        except Exception as e:
            logger.error("[%s] reader error: %s", call_id, e, exc_info=True)
            session.add_error("ws_reader_error", str(e))
        finally:
            stop_event.set()

    async def audio_pump() -> None:
        """Forward inbound audio chunks to the live Deepgram session."""
        try:
            while not stop_event.is_set():
                chunk = await audio_feed_queue.get()
                if chunk is None:
                    break
                if streaming_stt is not None:
                    await streaming_stt.feed(chunk)
        except Exception as e:
            logger.error("[%s] audio_pump error: %s", call_id, e, exc_info=True)

    async def stt_bridge() -> None:
        """Deepgram live endpointing → finalized transcript queue."""
        if streaming_stt is None:
            return
        try:
            async for transcript in streaming_stt.transcripts():
                if stop_event.is_set():
                    break
                # We feed Deepgram continuously (incl. while the agent speaks)
                # so transcript-gated barge-in works. But a turn finalized while
                # the agent is still speaking and the listen gate is closed is
                # almost always echo/cross-talk — a genuine interruption already
                # opened the mic via _on_speech_started. Drop those stray turns
                # so they can't be replayed later as a stale "answer".
                if transcript and ai_speaking.is_set() and not listen_active.is_set():
                    logger.debug(
                        "[%s] Dropping turn finalized during agent speech: %r",
                        call_id,
                        transcript[:40],
                    )
                    continue
                await transcript_queue.put(transcript)
        except Exception as e:
            logger.error("[%s] stt_bridge error: %s", call_id, e, exc_info=True)
        finally:
            await transcript_queue.put(None)

    if streaming_stt_enabled:
        from app.core.call_handler import get_call_providers

        _providers = get_call_providers(session)
        _model = getattr(_providers.stt, "model", settings.deepgram_model)
        _stt_encoding = "linear16" if emit_debug_events else "mulaw"

        async def _on_interim(text: str) -> None:
            await _emit("debug", message=f"…{text[-40:]}" if len(text) > 40 else text)

        streaming_stt = DeepgramStreamingSession(
            model=_model,
            encoding=_stt_encoding,
            sample_rate=8000,
            endpointing_ms=int(getattr(session, "deepgram_endpointing_ms", 900) or 900),
            utterance_end_ms=int(getattr(session, "deepgram_utterance_end_ms", 1000) or 1000),
            on_interim=_on_interim if emit_debug_events else None,
            on_speech_started=_on_speech_started,
        )
        await streaming_stt.start()
        session.stt_provider = _providers.stt_name
        vinfo(
            logger,
            f"Streaming STT active ({_stt_encoding})",
            session=session,
            call_id=call_id,
            phase=Phase.STT_STREAM,
            service="stt",
            provider=_providers.stt_name,
        )

    async def worker() -> None:
        try:
            try:
                if not emit_debug_events:
                    try:
                        await asyncio.wait_for(stream_ready.wait(), timeout=5.0)
                        logger.debug("[%s] Telnyx stream ready — greeting", call_id)
                    except TimeoutError:
                        logger.warning(
                            "[%s] Stream ready timeout — playing greeting anyway",
                            call_id,
                        )
                greeting_audio = await call_handler.handle_call_answered(session)
                if session.control_flags.get("provider_failure"):
                    if greeting_audio:
                        await enqueue_audio(greeting_audio, turn_end=True)
                    await _await_outbound_playback_done()
                    stop_event.set()
                    return
                if greeting_audio:
                    await enqueue_audio(greeting_audio, turn_end=True)
                    if emit_debug_events:
                        greeting_text = next(
                            (
                                t.text
                                for t in reversed(session.transcript)
                                if t.speaker == "AI"
                            ),
                            "",
                        )
                        await _emit(
                            "greeting",
                            text=greeting_text,
                            session=session.to_dict(),
                        )
                else:
                    tenant_may_speak.set()
            except Exception as e:
                logger.error("[%s] greeting failed: %s", call_id, e)
                session.add_error("greeting_failed", str(e))
                tenant_may_speak.set()

            def _drain_pending() -> int:
                return _drain_pending_input()

            # Fingerprint of the last transcript that was actually ACCEPTED
            # (advanced state / stored data / triggered a read-back). We only
            # suppress an identical repeat of an accepted answer — a caller who
            # repeats themselves after a failed attempt must never be blocked.
            last_accepted_norm = ""
            last_accepted_at = 0.0
            last_accepted_state = ""

            while not stop_event.is_set():
                # Do not start the silence timer until AI finished the current turn.
                await tenant_may_speak.wait()
                tenant_may_speak.clear()

                if emit_debug_events:
                    await _emit("debug", message="Your turn — speak when ready.")

                listen_active.set()

                try:
                    if streaming_stt_enabled:
                        transcript = await asyncio.wait_for(
                            transcript_queue.get(), timeout=listen_timeout
                        )
                        audio_format = "stream"
                        utterance = b""
                    else:
                        audio_format, utterance = await asyncio.wait_for(
                            utterance_queue.get(), timeout=listen_timeout
                        )
                        transcript = None
                except TimeoutError:
                    listen_active.clear()
                    if ai_speaking.is_set() or not outbound_queue.empty():
                        logger.debug(
                            f"[{call_id}] Silence skipped — agent still speaking "
                            f"or audio queued"
                        )
                        tenant_may_speak.set()
                        continue
                    logger.info(f"[{call_id}] No speech within {listen_timeout}s")
                    if should_suppress_silence_nudge(session):
                        logger.debug(
                            "[%s] Silence nudge suppressed — recent recovery",
                            call_id,
                        )
                        tenant_may_speak.set()
                        continue
                    (
                        silence_text,
                        audio_parts,
                        is_complete,
                    ) = await call_handler.handle_silence(session)
                    if silence_text:
                        await _emit(
                            "response",
                            text=silence_text,
                            speaker="AI",
                            session=session.to_dict(),
                        )
                    if audio_parts:
                        await enqueue_audio(audio_parts, turn_end=True)
                    elif not is_complete:
                        tenant_may_speak.set()
                    if is_complete:
                        await _await_outbound_playback_done()
                        stop_event.set()
                        break
                    continue

                listen_active.clear()
                session.silence_count = 0
                session.stt_empty_strikes = 0

                if streaming_stt_enabled:
                    if transcript is None:
                        if stop_event.is_set():
                            break
                        # Deepgram live socket closed (error or hangup). Do not
                        # end the call abruptly — fall through to silence
                        # timeouts until the stream stop event fires.
                        session.add_error(
                            "stt_streaming_lost",
                            "Deepgram live session ended unexpectedly",
                        )
                        logger.warning(
                            "[%s] Streaming STT session ended — using silence handling",
                            call_id,
                        )
                        tenant_may_speak.set()
                        continue
                else:
                    if emit_debug_events:
                        await _emit("debug", message="Transcribing…")
                    t0 = time.monotonic()
                    transcript = await transcribe_buffer(
                        utterance, input_format=audio_format, session=session
                    )
                    stt_ms = (time.monotonic() - t0) * 1000
                    if not transcript.strip():
                        logger.info(
                            f"[{call_id}] STT empty "
                            f"({len(utterance)} bytes {audio_format}, {stt_ms:.0f}ms)"
                        )
                        session.stt_empty_strikes += 1
                        if session.stt_empty_strikes >= STT_EMPTY_STRIKE_LIMIT:
                            (
                                fail_text,
                                fail_parts,
                                is_complete,
                            ) = await call_handler.end_call_for_provider_failure(
                                session,
                                "stt",
                                (
                                    f"{session.stt_empty_strikes} consecutive "
                                    "empty transcriptions"
                                ),
                            )
                            if fail_text:
                                await _emit(
                                    "response",
                                    text=fail_text,
                                    speaker="AI",
                                    session=session.to_dict(),
                                )
                            if fail_parts:
                                await enqueue_audio(fail_parts, turn_end=True)
                            if is_complete:
                                await _await_outbound_playback_done()
                                stop_event.set()
                                break
                            continue
                        retry_text = (
                            "Sorry, I didn't catch that. Could you repeat that for me?"
                        )
                        session.add_transcript("AI", retry_text)
                        await _emit(
                            "response",
                            text=retry_text,
                            speaker="AI",
                            session=session.to_dict(),
                        )
                        retry_audio = await call_handler.synthesize_with_fallback(
                            retry_text, session
                        )
                        if retry_audio:
                            await enqueue_audio([retry_audio], turn_end=True)
                        else:
                            tenant_may_speak.set()
                        continue
                    logger.info("[%s] STT %.0fms: %r", call_id, stt_ms, transcript[:60])

                if not (transcript or "").strip():
                    tenant_may_speak.set()
                    continue

                vinfo(
                    logger,
                    f"Turn transcript: {transcript[:80]!r}",
                    session=session,
                    call_id=call_id,
                    phase=Phase.TURN_START,
                    service="turn",
                    budget_s=round(_turn_budget_seconds(session), 1),
                )

                # Drop a duplicate STT result only when it repeats an answer we
                # already ACCEPTED on the same question within a few seconds (a
                # doubled transcription answering twice). A repeat after a failed
                # attempt must fall through so the caller can re-answer.
                norm = re.sub(r"[^a-z0-9]+", " ", transcript.lower()).strip()
                now = time.monotonic()
                current_state = session.current_state
                is_short_answer = norm in _DEDUP_SHORT_ANSWERS
                if (
                    norm
                    and not is_short_answer
                    and norm == last_accepted_norm
                    and current_state == last_accepted_state
                    and (now - last_accepted_at) < 6.0
                ):
                    vinfo(
                        logger,
                        f"Ignoring duplicate transcript: {transcript[:80]!r}",
                        session=session,
                        call_id=call_id,
                        phase=Phase.ECHO,
                        reason="duplicate",
                    )
                    tenant_may_speak.set()
                    continue

                # Technical mic guard only: drop the agent's own voice echoing
                # back. All understanding of the caller happens in the LLM.
                if is_echo_of_agent(transcript, session):
                    vinfo(
                        logger,
                        f"Ignoring agent echo: {transcript[:80]!r}",
                        session=session,
                        call_id=call_id,
                        phase=Phase.ECHO,
                        reason="agent_echo",
                    )
                    await _emit("debug", message="Still listening…")
                    tenant_may_speak.set()
                    continue

                await _emit("transcript", text=transcript, speaker="Tenant")

                # Snapshot state so we can tell whether this turn was accepted
                # (advanced the flow, stored data, or opened a read-back). Only
                # an accepted answer arms duplicate suppression.
                pre_state = session.current_state
                pre_data_keys = len(session.extracted_data)
                pre_pending = session.pending_confirmation is not None
                turn_llm_before = session.llm_latency_ms_total
                turn_tts_before = session.tts_latency_ms_total

                t1 = time.monotonic()
                first_audio_ms: float | None = None
                audio_streamed = False
                stream_turn_end_sent = False

                async def _stream_audio_part(
                    audio: bytes, is_last: bool, _turn_start: float = t1
                ) -> None:
                    nonlocal first_audio_ms, audio_streamed, stream_turn_end_sent
                    if first_audio_ms is None:
                        first_audio_ms = (time.monotonic() - _turn_start) * 1000
                    audio_streamed = True
                    if is_last:
                        stream_turn_end_sent = True
                    _prefix_preview = repr((session.streamed_speakable_prefix or "")[:60])
                    vinfo(
                        logger,
                        f"Streaming TTS chunk ({len(audio)} bytes, is_last={is_last})",
                        session=session,
                        call_id=call_id,
                        phase=Phase.STREAM_TTS,
                        service="tts",
                        bytes=len(audio),
                        detail=f"prefix={_prefix_preview}",
                    )
                    await enqueue_audio([audio], turn_end=is_last)

                async def _stream_text_update(text: str) -> None:
                    if not (text or "").strip():
                        return
                    vinfo(
                        logger,
                        f"UI stream text update: {text[:80]!r}",
                        session=session,
                        call_id=call_id,
                        phase=Phase.UI_STREAM,
                        detail=f"len={len(text.strip())}",
                    )
                    await _emit(
                        "response",
                        text=text.strip(),
                        speaker="AI",
                        streaming=True,
                        session=session.to_dict(),
                    )

                # Internal budget marker used by call_handler to avoid spending
                # the entire 15s outer guard on deep fallback chains.
                turn_budget = _turn_budget_seconds(session)
                session.turn_deadline_monotonic = t1 + turn_budget
                turn_in_progress.set()
                _drain_pending_input()
                current_turn_task[0] = asyncio.create_task(
                    asyncio.wait_for(
                        call_handler.process_tenant_speech(
                            session,
                            transcript,
                            on_audio_part=_stream_audio_part,
                            on_stream_text=_stream_text_update,
                        ),
                        timeout=turn_budget,
                    )
                )
                try:
                    (
                        response_text,
                        audio_parts,
                        is_complete,
                    ) = await current_turn_task[0]
                except asyncio.CancelledError:
                    vinfo(
                        logger,
                        "Turn cancelled (barge-in or hangup)",
                        session=session,
                        call_id=call_id,
                        phase=Phase.BARGE_IN,
                    )
                    tenant_may_speak.set()
                    continue
                except TimeoutError:
                    vwarn(
                        logger,
                        f"Turn timed out after {turn_budget:.0f}s",
                        session=session,
                        call_id=call_id,
                        phase=Phase.TURN_TIMEOUT,
                        service="turn",
                        timeout_s=turn_budget,
                        detail=f"llm={session.llm_provider} tts={session.tts_provider}",
                    )
                    task = current_turn_task[0]
                    if task is not None and not task.done():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                    session.add_error(
                        "turn_timeout",
                        f"Exceeded {turn_budget:.0f}s turn budget",
                    )
                    session.record_turn_trace(
                        {
                            "state": pre_state,
                            "turn_ms": int(turn_budget * 1000),
                            "llm_ms": int(
                                session.llm_latency_ms_total - turn_llm_before
                            ),
                            "tts_ms": int(
                                session.tts_latency_ms_total - turn_tts_before
                            ),
                            "timed_out": True,
                        }
                    )
                    retry_text = plan_turn_timeout_recovery(session, transcript)
                    vinfo(
                        logger,
                        f"Turn timeout recovery: {retry_text[:80]!r}",
                        session=session,
                        call_id=call_id,
                        phase=Phase.TURN_RECOVERY,
                        service="turn",
                    )
                    if (
                        session.transcript
                        and session.transcript[-1].speaker == "AI"
                        and session.transcript[-1].text.strip() == retry_text.strip()
                    ):
                        pass
                    else:
                        session.add_transcript("AI", retry_text)
                    await _emit(
                        "response",
                        text=retry_text,
                        speaker="AI",
                        session=session.to_dict(),
                    )
                    mark_recovery_played(session)
                    reset_turn_streaming(session, full=True)
                    session.turn_streaming_finalize = None
                    retry_audio = await call_handler.synthesize_with_fallback(
                        retry_text, session
                    )
                    if retry_audio:
                        await enqueue_audio([retry_audio], turn_end=True)
                    else:
                        tenant_may_speak.set()
                    continue
                finally:
                    turn_in_progress.clear()
                    current_turn_task[0] = None
                    session.turn_deadline_monotonic = None
                turn_ms = (time.monotonic() - t1) * 1000
                session.record_turn_latency(turn_ms)
                session.record_turn_trace(
                    {
                        "state": pre_state,
                        "turn_ms": int(round(turn_ms)),
                        "llm_ms": int(session.llm_latency_ms_total - turn_llm_before),
                        "tts_ms": int(session.tts_latency_ms_total - turn_tts_before),
                        "ttfa_ms": int(round(first_audio_ms))
                        if first_audio_ms is not None
                        else None,
                        "audio_parts": len(audio_parts) if audio_parts else 0,
                        "streamed": audio_streamed,
                        "llm_streamed": getattr(session, "llm_streamed_during_turn", False),
                        "complete": is_complete,
                    }
                )

                accepted = (
                    session.current_state != pre_state
                    or len(session.extracted_data) != pre_data_keys
                    or (session.pending_confirmation is not None and not pre_pending)
                    or is_complete
                )
                if accepted and norm and not is_short_answer:
                    last_accepted_norm = norm
                    last_accepted_at = now
                    last_accepted_state = pre_state
                vinfo(
                    logger,
                    f"Turn complete in {turn_ms:.0f}ms",
                    session=session,
                    call_id=call_id,
                    phase=Phase.TURN_END,
                    service="turn",
                    latency_ms=int(turn_ms),
                    detail=(
                        f"llm={session.llm_provider} tts={session.tts_provider} "
                        f"accepted={accepted} complete={is_complete}"
                    ),
                )
                vdebug(
                    logger,
                    f"Response text: {response_text!r}, audio_parts={len(audio_parts) if audio_parts else 0}",
                    session=session,
                    call_id=call_id,
                    phase=Phase.TURN_END,
                )

                if not response_text and not audio_parts:
                    logger.debug("[%s] No response (likely echo) — listening", call_id)
                    if emit_debug_events:
                        await _emit("debug", message="Still listening…")
                    tenant_may_speak.set()
                    continue

                if response_text:
                    last_ai = ""
                    if session.transcript and session.transcript[-1].speaker == "AI":
                        last_ai = (session.transcript[-1].text or "").strip()
                    final_text = response_text.strip()
                    ui_duplicate = bool(last_ai and last_ai == final_text)
                    vinfo(
                        logger,
                        f"UI final response emit (duplicate_line={ui_duplicate})",
                        session=session,
                        call_id=call_id,
                        phase=Phase.UI_FINAL,
                        reason="duplicate_append" if ui_duplicate else "new_line",
                        detail=(
                            f"text={final_text[:80]!r} "
                            f"streamed={audio_streamed} "
                            f"audio_parts={len(audio_parts) if audio_parts else 0}"
                        ),
                    )
                    await _emit(
                        "response",
                        text=final_text,
                        speaker="AI",
                        streaming=ui_duplicate,
                        session=session.to_dict(),
                    )
                elif emit_debug_events:
                    logger.debug("[%s] Response text empty but has audio", call_id)
                    await _emit("debug", message="Processing response…")

                # The next question is about to play — discard any audio that
                # arrived during the turn we just processed (split-answer tails,
                # echoes) so it can't be applied to the upcoming question.
                if not is_complete:
                    _drain_pending()

                if stop_event.is_set():
                    break

                if audio_streamed:
                    fin = getattr(session, "turn_streaming_finalize", None) or {}
                    if stream_turn_end_sent:
                        vinfo(
                            logger,
                            "Skipping remainder TTS — batch audio already ended turn",
                            session=session,
                            call_id=call_id,
                            phase=Phase.TTS_DEDUP_SKIP,
                            detail=f"intended={(fin.get('intended') or response_text or '')[:80]!r}",
                        )
                    else:
                        remainder = unsynthesized_speech_remainder(
                            response_text, session
                        )
                        vinfo(
                            logger,
                            f"Post-stream audio path remainder={remainder[:60]!r}"
                            if remainder
                            else "Post-stream audio path (no remainder)",
                            session=session,
                            call_id=call_id,
                            phase=Phase.TTS_REMAINDER if remainder else Phase.TTS_DEDUP_SKIP,
                            detail=(
                                f"streamed_prefix={fin.get('streamed_prefix', '')[:60]!r} "
                                f"intended={fin.get('intended', '')[:60]!r} "
                                f"streamed_sent={fin.get('streamed_sent')}"
                            ),
                        )
                        if remainder:
                            retry_audio = await call_handler.synthesize_with_fallback(
                                remainder, session
                            )
                            if retry_audio:
                                vinfo(
                                    logger,
                                    f"Remainder TTS synthesized ({len(retry_audio)} bytes)",
                                    session=session,
                                    call_id=call_id,
                                    phase=Phase.TTS_REMAINDER,
                                    bytes=len(retry_audio),
                                    detail=f"text={remainder[:80]!r}",
                                )
                                await enqueue_audio([retry_audio], turn_end=True)
                            else:
                                await enqueue_audio([], turn_end=True)
                        else:
                            await enqueue_audio([], turn_end=True)
                    session.turn_streaming_finalize = None
                elif audio_parts:
                    vinfo(
                        logger,
                        f"Batch TTS enqueue {len(audio_parts)} part(s)",
                        session=session,
                        call_id=call_id,
                        phase=Phase.TTS_FINISH,
                        detail=f"total_bytes={sum(len(p) for p in audio_parts)}",
                    )
                    await enqueue_audio(audio_parts, turn_end=True)
                elif response_text and not is_complete:
                    # Last-mile guard: if turn produced text but no audio chunks,
                    # synthesize once more so state does not drift silently.
                    retry_audio = await call_handler.synthesize_with_fallback(
                        response_text.strip(), session
                    )
                    if retry_audio:
                        await enqueue_audio([retry_audio], turn_end=True)
                    else:
                        fail_text, fail_parts, is_complete = (
                            await call_handler.end_call_for_provider_failure(
                                session,
                                "tts",
                                "Response speech synthesis failed",
                            )
                        )
                        if fail_text:
                            await _emit(
                                "response",
                                text=fail_text,
                                speaker="AI",
                                session=session.to_dict(),
                            )
                        if fail_parts:
                            await enqueue_audio(fail_parts, turn_end=True)
                    session.turn_streaming_finalize = None
                elif not is_complete:
                    # All TTS providers failed — release the listen gate so
                    # the call does not hang until max_call_duration.
                    tenant_may_speak.set()
                if is_complete:
                    await _await_outbound_playback_done()
                    stop_event.set()
                    break
        except Exception as e:
            logger.error("[%s] worker error: %s", call_id, e, exc_info=True)
            session.add_error("ws_worker_error", str(e))
        finally:
            await outbound_queue.put(b"")

    async def sender() -> None:
        nonlocal barge_in_cooldown_until
        try:
            while True:
                item = await outbound_queue.get()
                if not item:
                    if stop_event.is_set():
                        return
                    continue

                if isinstance(item, tuple):
                    audio, turn_end = item
                else:
                    audio, turn_end = item, True

                if not audio:
                    if turn_end:
                        tenant_may_speak.set()
                        logger.debug("[%s] Agent turn complete — listening", call_id)
                        if emit_debug_events:
                            await _emit("agent_done")
                    continue

                tenant_may_speak.clear()
                ai_speaking.set()
                interrupt_event.clear()
                try:
                    if emit_debug_events:
                        wav = mulaw_to_wav(audio)
                        await _emit(
                            "play_wav",
                            audio_wav_b64=base64.b64encode(wav).decode("ascii"),
                            duration_ms=int(len(audio) / 8000 * 1000),
                            turn_end=turn_end,
                        )
                        # Interruptible playback window: a browser barge-in sets
                        # interrupt_event so we stop "speaking" immediately.
                        play_seconds = len(audio) / 8000.0
                        waited = 0.0
                        while waited < play_seconds:
                            if stop_event.is_set() or interrupt_event.is_set():
                                logger.debug(
                                    "[%s] Playback interrupted (barge-in)", call_id
                                )
                                break
                            await asyncio.sleep(0.05)
                            waited += 0.05
                    else:
                        for chunk in chunk_audio(audio, chunk_size=160):
                            if stop_event.is_set() or interrupt_event.is_set():
                                break
                            payload_b64 = encode_telnyx_payload(chunk)
                            try:
                                await websocket.send_text(
                                    orjson.dumps(
                                        {
                                            "event": "media",
                                            "media": {"payload": payload_b64},
                                        }
                                    ).decode()
                                )
                            except Exception as e:
                                logger.warning(f"[{call_id}] send chunk failed: {e}")
                                stop_event.set()
                                return
                            await asyncio.sleep(0.02)
                finally:
                    ai_speaking.clear()
                    # Brief cooldown so the speaker's decaying echo tail (on
                    # hands-free) can't be transcribed into a self barge-in.
                    barge_in_cooldown_until = time.monotonic() + BARGE_IN_COOLDOWN_S
                    # On a normal turn end, hand the mic back. On barge-in the
                    # reader already released the listen gate, so don't fight it.
                    if turn_end and not interrupt_event.is_set():
                        tenant_may_speak.set()
                        logger.debug("[%s] Agent turn complete — listening", call_id)
                    if emit_debug_events and turn_end:
                        await _emit("agent_done")
        except Exception as e:
            logger.error("[%s] sender error: %s", call_id, e, exc_info=True)

    async def watchdog() -> None:
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=max_call_duration)
        except TimeoutError:
            vwarn(
                logger,
                "Max call duration reached — ending call",
                session=session,
                call_id=call_id,
                phase=Phase.CALL_END,
                reason="max_duration",
            )
            session.add_error("max_duration", f"Exceeded {max_call_duration}s")
            stop_event.set()

    tasks = [
        asyncio.create_task(reader(), name=f"ws-reader-{call_id}"),
        asyncio.create_task(worker(), name=f"ws-worker-{call_id}"),
        asyncio.create_task(sender(), name=f"ws-sender-{call_id}"),
        asyncio.create_task(watchdog(), name=f"ws-watchdog-{call_id}"),
    ]
    if streaming_stt_enabled:
        tasks.append(asyncio.create_task(audio_pump(), name=f"ws-audio-pump-{call_id}"))
        tasks.append(asyncio.create_task(stt_bridge(), name=f"ws-stt-bridge-{call_id}"))

    try:
        await stop_event.wait()
    finally:
        call_handler.unregister_stream_stop(call_id)
        from app.core.redis_client import clear_stream_stop_signal

        await clear_stream_stop_signal(call_id)
        try:
            audio_feed_queue.put_nowait(None)
        except asyncio.QueueFull:
            pass
        if streaming_stt is not None:
            await streaming_stt.close()
        for t in tasks:
            if not t.done():
                t.cancel()
        for t in tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass

        if emit_debug_events:
            await _emit("complete", session=session.to_dict())

        if hangup_on_complete:
            try:
                from app.services.telnyx_service import telnyx_service

                await telnyx_service.hangup_call(call_id)
            except Exception as e:
                logger.debug("[%s] hangup (may already be over): %s", call_id, e)

        if on_complete:
            try:
                await on_complete()
            except Exception as e:
                logger.error("[%s] on_complete callback failed: %s", call_id, e)

        call_handler.finish_stream_session(call_id)

        try:
            await websocket.close()
        except (asyncio.CancelledError, RuntimeError) as e:
            logger.debug("WebSocket close error (may already be closed): %s", e)
        logger.info("WebSocket closed for call: %s", call_id)
