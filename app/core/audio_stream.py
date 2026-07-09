"""
app/core/audio_stream.py — Shared bidirectional audio WebSocket handler.

Used by both the production Telnyx stream (/telnyx/stream) and the test
console stream (/test/api/stream) so both paths exercise identical logic:
reader → STT → LLM → TTS → sender, with silence timeout and max duration.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
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
from app.core.call_settings import stt_model_for_provider
from app.core.conversation import (
    STT_EMPTY_STRIKE_LIMIT,
    ConversationSession,
    capture_turn_snapshot,
    handle_hangup_cancelled_turn,
    handle_turn_timeout,
    is_echo_of_agent,
    mark_recovery_played,
    reset_turn_streaming,
    should_suppress_silence_nudge,
    turn_budget_seconds,
)
from app.core.streaming_stt import DeepgramStreamingSession, StreamingSttRelay
from app.core.voice_language import deepgram_stt_language
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


async def _await_turn_or_hangup(
    turn_task: asyncio.Task,
    stop_event: asyncio.Event,
    *,
    session: ConversationSession,
) -> tuple[str, list[bytes], bool] | None:
    """Wait for an in-flight turn, or cancel it when hangup sets ``stop_event``.

    Returns the turn result, or ``None`` when hangup ended the turn early.
    Raises ``asyncio.CancelledError`` when barge-in cancels the task.
    """
    hangup_watch = asyncio.create_task(stop_event.wait())
    try:
        finished, _ = await asyncio.wait(
            {turn_task, hangup_watch},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if turn_task.done() and not turn_task.cancelled():
            return await turn_task
        if hangup_watch in finished or stop_event.is_set():
            if not turn_task.done():
                call_handler.interrupt_turn_tts(session)
                turn_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await turn_task
            return None
        return await turn_task
    finally:
        hangup_watch.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await hangup_watch


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

# One reconnect attempt before degrading live Deepgram to batch transcription.
from app.core.streaming_stt import STREAMING_STT_RECONNECT_TIMEOUT_S
STT_STREAM_RECONNECT_FLAG = "stt_stream_reconnects"
# Keep only a bounded recent window of caller audio for live->batch handoff.
STREAMING_TO_BATCH_CARRYOVER_MAX_BYTES = MAX_BUFFER_BYTES


class _StreamStopRequested(Exception):
    """Raised when the caller hung up while a listen queue was waiting."""


async def _queue_get_with_stop(
    queue: asyncio.Queue,
    stop_event: asyncio.Event,
    *,
    timeout: float,
):
    """Wait for a queue item but return promptly when hangup sets ``stop_event``."""
    if stop_event.is_set():
        raise _StreamStopRequested()
    getter = asyncio.create_task(queue.get())
    stopper = asyncio.create_task(stop_event.wait())
    try:
        done, pending = await asyncio.wait(
            {getter, stopper},
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        if stopper in done:
            raise _StreamStopRequested()
        if getter in done:
            return getter.result()
        raise TimeoutError()
    finally:
        if not getter.done():
            getter.cancel()
        if not stopper.done():
            stopper.cancel()


def agent_turn_in_progress(
    *,
    ai_speaking: bool,
    outbound_pending: bool,
    turn_task_active: bool,
) -> bool:
    """True while agent audio or the in-flight LLM/TTS turn is still active."""
    return ai_speaking or outbound_pending or turn_task_active


def should_drop_streaming_transcript(
    *,
    transcript: str,
    listen_active: bool,
    caller_speech_pending: bool,
    agent_turn_in_progress: bool,
) -> bool:
    """Drop stray echo finalizations during agent speech unless capture opened."""
    if not (transcript or "").strip():
        return False
    if listen_active or caller_speech_pending:
        return False
    return agent_turn_in_progress


def should_preserve_pending_transcripts(
    *,
    caller_speech_pending: bool,
    queued_transcripts: int,
) -> bool:
    """Keep queued STT turns when caller speech arrived during an agent turn."""
    return caller_speech_pending or queued_transcripts > 0


def append_streaming_carryover_audio(
    buffer: bytearray,
    chunk: bytes,
    *,
    max_bytes: int = STREAMING_TO_BATCH_CARRYOVER_MAX_BYTES,
) -> None:
    """Append recent caller audio for potential live->batch STT failover."""
    if not chunk:
        return
    buffer.extend(chunk)
    overflow = len(buffer) - max_bytes
    if overflow > 0:
        del buffer[:overflow]

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
    session: ConversationSession,
) -> str:
    """Transcribe audio using the call's STT provider (+ Groq fallback)."""
    from app.core.call_handler import get_call_providers

    if not audio_bytes:
        return ""

    providers = get_call_providers(session)

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
        api_keys = providers.api_keys
        groq_ok = not primary_is_groq and api_keys.configured("groq")
        deepgram_ok = not isinstance(providers.stt, DeepgramSTTProvider) and api_keys.configured(
            "deepgram"
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
                from app.core.voice_language import groq_stt_language

                lang = groq_stt_language(
                    session.call_language if session is not None else "en"
                )
                fallback_stt = GroqSTTProvider(
                    model=stt_model_for_provider(providers, "groq"),
                    language=lang,
                    api_key=api_keys.groq,
                )
            else:
                from app.core.voice_language import deepgram_stt_language

                lang = deepgram_stt_language(
                    session.call_language if session is not None else "en"
                )
                fallback_stt = DeepgramSTTProvider(
                    model=stt_model_for_provider(providers, "deepgram"),
                    language=lang,
                    api_key=api_keys.deepgram,
                )
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


STT_EMPTY_RETRY_TEXT_EN = (
    "Sorry, I didn't catch that. Could you repeat that for me?"
)
STT_EMPTY_RETRY_TEXT_ES = (
    "Perdon, no escuche bien. Podria repetirlo, por favor?"
)
# Back-compat alias for tests and any external imports.
STT_EMPTY_RETRY_TEXT = STT_EMPTY_RETRY_TEXT_EN


def stt_empty_retry_text(session: ConversationSession) -> str:
    """Localized retry when STT returns no words (matches silence-nudge language)."""
    if str(getattr(session, "call_language", "en")).strip().lower().startswith("es"):
        return STT_EMPTY_RETRY_TEXT_ES
    return STT_EMPTY_RETRY_TEXT_EN


async def handle_stt_empty_transcript(
    session: ConversationSession,
    *,
    call_id: str,
    log_detail: str = "",
    emit_response: Callable[[str], Awaitable[None]] | None = None,
    enqueue_retry: Callable[[list[bytes]], Awaitable[None]] | None = None,
    await_playback_done: Callable[[], Awaitable[None]] | None = None,
) -> tuple[bool, bool]:
    """Shared empty-STT handling for live streaming and batch paths.

    Returns (should_end_call, retry_audio_queued).
    """
    if log_detail:
        logger.info("[%s] STT empty %s", call_id, log_detail)
    else:
        logger.info("[%s] STT empty", call_id)

    session.stt_empty_strikes += 1
    if session.stt_empty_strikes >= STT_EMPTY_STRIKE_LIMIT:
        fail_text, fail_parts, is_complete = (
            await call_handler.end_call_for_provider_failure(
                session,
                "stt",
                (
                    f"{session.stt_empty_strikes} consecutive "
                    "empty transcriptions"
                ),
            )
        )
        if fail_text and emit_response is not None:
            await emit_response(fail_text)
        if fail_parts and enqueue_retry is not None:
            await enqueue_retry(fail_parts)
        if is_complete and await_playback_done is not None:
            await await_playback_done()
        return True, False

    retry_text = stt_empty_retry_text(session)
    session.add_transcript("AI", retry_text)
    if emit_response is not None:
        await emit_response(retry_text)
    retry_audio = await call_handler.synthesize_with_fallback(
        retry_text, session
    )
    retry_queued = False
    if retry_audio and enqueue_retry is not None:
        await enqueue_retry([retry_audio])
        retry_queued = True
    return False, retry_queued


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
    transcript_queue: asyncio.Queue[TranscriptItem | None] = asyncio.Queue(maxsize=32)
    audio_feed_queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=512)
    # Recent caller audio captured while live STT is active. If live STT fails
    # and we switch to batch mode, we replay this window so failover does not
    # drop in-flight speech around the handoff boundary.
    streaming_carryover_buffer = bytearray()
    outbound_queue: asyncio.Queue[OutboundItem | bytes] = asyncio.Queue()
    ai_speaking = asyncio.Event()
    tenant_may_speak = asyncio.Event()
    listen_active = asyncio.Event()
    stop_event = asyncio.Event()
    current_turn_task: list[asyncio.Task | None] = [None]
    enqueue_tasks: set[asyncio.Task] = set()
    caller_speech_pending_since: list[float | None] = [None]
    call_handler.register_stream_stop(call_id, stop_event)
    # A hangup may have landed while this WebSocket was still connecting (before
    # the stop_event existed). If so, wind down immediately instead of running a
    # full turn against a caller who is already gone.
    if getattr(session, "pending_hangup", False):
        logger.info("[%s] Hangup arrived before stream start — stopping", call_id)
        stop_event.set()
    elif await call_handler.check_stream_stop_signal(call_id):
        logger.info(
            "[%s] Cross-worker hangup stop detected at stream start — stopping",
            call_id,
        )
        session.pending_hangup = True
        stop_event.set()
    streaming_stt: DeepgramStreamingSession | StreamingSttRelay | None = None
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

    def _agent_turn_active() -> bool:
        task = current_turn_task[0]
        return agent_turn_in_progress(
            ai_speaking=ai_speaking.is_set(),
            outbound_pending=not outbound_queue.empty(),
            turn_task_active=task is not None and not task.done(),
        )

    def _caller_has_listen_floor() -> bool:
        return listen_active.is_set() and not _agent_turn_active()

    def _apply_barge_in() -> None:
        """Stop current playback and hand the turn back to the caller.

        Barge-in always means "stop talking and listen" — we discard any queued
        outbound audio and release the listen gate so the worker processes the
        caller's utterance as their answer. We deliberately do NOT auto re-ask
        or drop the caller's input: doing so created a re-ask feedback loop and
        swallowed real answers.
        """
        session.caller_speech_pending = True
        call_handler.interrupt_turn_tts(session)
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

    def _streaming_is_lost() -> bool:
        relay = streaming_stt
        if relay is None:
            return True
        return bool(getattr(relay, "lost", False))

    def _drain_transcript_queue() -> int:
        dropped = 0
        while not transcript_queue.empty():
            try:
                transcript_queue.get_nowait()
                dropped += 1
            except asyncio.QueueEmpty:
                break
        return dropped

    def _enqueue_audio_feed_chunk(chunk: bytes) -> None:
        """Best-effort enqueue for live STT without silently losing newest audio."""
        try:
            audio_feed_queue.put_nowait(chunk)
            return
        except asyncio.QueueFull:
            pass
        # Drop one oldest chunk to preserve current speech (newest audio is
        # usually more relevant for turn-finalization than stale buffered audio).
        try:
            audio_feed_queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        try:
            audio_feed_queue.put_nowait(chunk)
        except asyncio.QueueFull:
            logger.warning(
                "[%s] audio feed queue saturated, dropping chunk after eviction",
                call_id,
            )

    async def _try_recover_streaming_stt() -> bool:
        """Try one live-socket restart; returns True when streaming is healthy again."""
        nonlocal streaming_stt
        if streaming_stt is None:
            return False
        attempts = int(session.control_flags.get(STT_STREAM_RECONNECT_FLAG, 0) or 0)
        if attempts >= 1:
            return False
        session.control_flags[STT_STREAM_RECONNECT_FLAG] = attempts + 1
        try:
            await asyncio.wait_for(
                streaming_stt.restart(),
                timeout=STREAMING_STT_RECONNECT_TIMEOUT_S,
            )
        except Exception as exc:
            logger.warning(
                "[%s] Streaming STT reconnect failed: %s",
                call_id,
                exc,
            )
            return False
        if _streaming_is_lost():
            return False
        session.add_provider_event(
            service="stt",
            provider=session.stt_provider or "deepgram",
            role="primary",
            outcome="recovered",
            detail="Live streaming STT restarted after disconnect",
        )
        logger.info("[%s] Streaming STT reconnected — resuming live mode", call_id)
        return True

    async def _degrade_to_batch_stt(*, reason: str) -> None:
        """Stop live Deepgram and use buffered batch transcription for this call."""
        nonlocal streaming_stt_enabled, streaming_stt
        if not streaming_stt_enabled:
            return
        streaming_stt_enabled = False
        session.add_error("stt_streaming_lost", reason)
        session.add_provider_event(
            service="stt",
            provider=session.stt_provider or "deepgram",
            role="primary",
            outcome="degraded",
            detail="Falling back to batch transcription for remainder of call",
        )
        logger.warning(
            "[%s] Streaming STT lost — switching to batch transcription: %s",
            call_id,
            reason,
        )
        while not audio_feed_queue.empty():
            try:
                pending = audio_feed_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if pending:
                append_streaming_carryover_audio(streaming_carryover_buffer, pending)
        try:
            audio_feed_queue.put_nowait(None)
        except asyncio.QueueFull:
            pass
        if len(streaming_carryover_buffer) >= MIN_UTTERANCE_BYTES:
            carryover = bytes(streaming_carryover_buffer)
            streaming_carryover_buffer.clear()
            await _enqueue_utterance(utterance_queue, carryover, call_id)
            if emit_debug_events:
                await _emit(
                    "debug",
                    message=(
                        f"Recovered {len(carryover)} bytes of in-flight audio "
                        "after live STT disconnect"
                    ),
                )
        relay = streaming_stt
        streaming_stt = None
        session.streaming_stt_relay = None
        if relay is not None:
            try:
                await relay.close()
            except Exception as exc:
                logger.debug("[%s] streaming STT close after degrade: %s", call_id, exc)
        # Preserve already-finalized caller turns collected during the failure
        # window; the worker should process them after we switch to batch mode.

    async def _handle_streaming_stt_failure(*, reason: str) -> None:
        """Reconnect live STT once, else degrade to batch mode for the rest of the call."""
        if await _try_recover_streaming_stt():
            return
        await _degrade_to_batch_stt(reason=reason)

    async def _on_speech_started(text: str) -> None:
        """Transcript-gated barge-in: Deepgram heard real words from the caller.

        Capture and interrupt are split: caller words during an agent turn always
        open capture (so stt_bridge won't drop the finalize), while playback is
        only stopped when the speech is not agent echo.
        """
        nonlocal barge_in_cooldown_until
        if _caller_has_listen_floor():
            return

        session.caller_speech_pending = True
        if caller_speech_pending_since[0] is None:
            caller_speech_pending_since[0] = time.monotonic()
        listen_active.set()

        if is_echo_of_agent(text, session):
            logger.debug(
                "[%s] Caller speech captured; skipping barge-in (echo): %r",
                call_id,
                text[:40],
            )
            session.caller_speech_pending = False
            caller_speech_pending_since[0] = None
            listen_active.clear()
            return

        if not _agent_turn_active():
            if time.monotonic() < barge_in_cooldown_until:
                return
            return

        # Cooldown only blocks echo tails after the agent finished — never during
        # active TTS or while more outbound audio is still queued.
        if (
            not ai_speaking.is_set()
            and outbound_queue.empty()
            and time.monotonic() < barge_in_cooldown_until
        ):
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
                    if not streaming_stt_enabled and _agent_turn_active():
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
                        append_streaming_carryover_audio(
                            streaming_carryover_buffer, chunk
                        )
                        _enqueue_audio_feed_chunk(chunk)
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
                            task = asyncio.create_task(
                                _enqueue_utterance(utterance_queue, utterance, call_id)
                            )
                            enqueue_tasks.add(task)
                            task.add_done_callback(enqueue_tasks.discard)
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
                # so transcript-gated barge-in works. Drop stray echo finals only
                # when the agent still owns the turn and capture was not opened.
                if should_drop_streaming_transcript(
                    transcript=transcript,
                    listen_active=listen_active.is_set(),
                    caller_speech_pending=session.caller_speech_pending,
                    agent_turn_in_progress=_agent_turn_active(),
                ):
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

        streaming_stt = StreamingSttRelay(
            model=_model,
            encoding=_stt_encoding,
            sample_rate=8000,
            language=deepgram_stt_language(session.call_language),
            endpointing_ms=int(getattr(session, "deepgram_endpointing_ms", 900) or 900),
            utterance_end_ms=int(getattr(session, "deepgram_utterance_end_ms", 1000) or 1000),
            on_interim=_on_interim if emit_debug_events else None,
            on_speech_started=_on_speech_started,
            api_key=_providers.api_keys.deepgram,
        )
        session.streaming_stt_relay = streaming_stt
        try:
            await streaming_stt.start()
        except Exception as e:
            logger.warning(
                "[%s] Deepgram streaming STT failed — falling back to batch: %s",
                call_id,
                e,
            )
            session.add_provider_event(
                service="stt",
                provider=_providers.stt_name,
                role="primary",
                outcome="failed",
                exc=e,
                detail="Falling back to batch transcription",
            )
            streaming_stt_enabled = False
            streaming_stt = None
            session.streaming_stt_relay = None
        else:
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
                        transcript = await _queue_get_with_stop(
                            transcript_queue,
                            stop_event,
                            timeout=listen_timeout,
                        )
                        audio_format = "stream"
                        utterance = b""
                    else:
                        audio_format, utterance = await _queue_get_with_stop(
                            utterance_queue,
                            stop_event,
                            timeout=listen_timeout,
                        )
                        transcript = None
                except _StreamStopRequested:
                    listen_active.clear()
                    break
                except TimeoutError:
                    listen_active.clear()
                    if ai_speaking.is_set() or not outbound_queue.empty():
                        logger.debug(
                            f"[{call_id}] Silence skipped — agent still speaking "
                            f"or audio queued"
                        )
                        tenant_may_speak.set()
                        continue
                    if streaming_stt_enabled and session.caller_speech_pending:
                        pending_for = time.monotonic() - (
                            caller_speech_pending_since[0] or time.monotonic()
                        )
                        max_pending_s = max(float(listen_timeout) * 2.0, 30.0)
                        if pending_for < max_pending_s:
                            logger.debug(
                                "[%s] Extending listen timeout — caller speech "
                                "pending for %.1fs",
                                call_id,
                                pending_for,
                            )
                            tenant_may_speak.set()
                            continue
                        logger.warning(
                            "[%s] Caller speech pending exceeded %.1fs without "
                            "final transcript",
                            call_id,
                            max_pending_s,
                        )
                        session.caller_speech_pending = False
                        caller_speech_pending_since[0] = None
                    if streaming_stt_enabled and _streaming_is_lost():
                        await _handle_streaming_stt_failure(
                            reason="Deepgram live session stopped producing transcripts",
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
                stt_ms = 0.0

                if streaming_stt_enabled and (transcript or "").strip():
                    session.caller_speech_pending = False
                    caller_speech_pending_since[0] = None
                    streaming_carryover_buffer.clear()

                if streaming_stt_enabled:
                    if transcript is None:
                        if stop_event.is_set():
                            break
                        await _handle_streaming_stt_failure(
                            reason="Deepgram live session ended unexpectedly",
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
                    if (transcript or "").strip():
                        logger.info(
                            "[%s] STT %.0fms: %r", call_id, stt_ms, transcript[:60]
                        )

                if not (transcript or "").strip():
                    log_detail = (
                        f"({len(utterance)} bytes {audio_format}, {stt_ms:.0f}ms)"
                        if not streaming_stt_enabled
                        else "(streaming)"
                    )

                    async def _emit_stt_response(text: str) -> None:
                        await _emit(
                            "response",
                            text=text,
                            speaker="AI",
                            session=session.to_dict(),
                        )

                    async def _enqueue_stt_retry(parts: list[bytes]) -> None:
                        await enqueue_audio(parts, turn_end=True)

                    should_end, retry_queued = await handle_stt_empty_transcript(
                        session,
                        call_id=call_id,
                        log_detail=log_detail,
                        emit_response=_emit_stt_response if emit_debug_events else None,
                        enqueue_retry=_enqueue_stt_retry,
                        await_playback_done=_await_outbound_playback_done,
                    )
                    if should_end:
                        stop_event.set()
                        break
                    if not retry_queued:
                        tenant_may_speak.set()
                    continue

                session.stt_empty_strikes = 0
                session.control_flags.pop(STT_STREAM_RECONNECT_FLAG, None)

                vinfo(
                    logger,
                    f"Turn transcript: {transcript[:80]!r}",
                    session=session,
                    call_id=call_id,
                    phase=Phase.TURN_START,
                    service="turn",
                    budget_s=round(turn_budget_seconds(session), 1),
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
                    session.caller_speech_pending = False
                    caller_speech_pending_since[0] = None
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
                turn_snapshot = capture_turn_snapshot(session)
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
                    if session.turn_interrupted:
                        return
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
                turn_budget = turn_budget_seconds(session)
                session.turn_deadline_monotonic = t1 + turn_budget
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
                    turn_result = await _await_turn_or_hangup(
                        current_turn_task[0],
                        stop_event,
                        session=session,
                    )
                    if turn_result is None:
                        vinfo(
                            logger,
                            "Turn cancelled — caller hung up",
                            session=session,
                            call_id=call_id,
                            phase=Phase.CALL_END,
                            reason="hangup_mid_turn",
                        )
                        handle_hangup_cancelled_turn(session, turn_snapshot)
                        await call_handler.drain_turn_tts_tasks(session)
                        break
                    (
                        response_text,
                        audio_parts,
                        is_complete,
                    ) = turn_result
                except asyncio.CancelledError:
                    vinfo(
                        logger,
                        "Turn cancelled (barge-in)",
                        session=session,
                        call_id=call_id,
                        phase=Phase.BARGE_IN,
                    )
                    # Repair history so the cancelled turn doesn't leave a
                    # dangling user message (no matching assistant) or a
                    # half-written streaming AI transcript line.
                    session.reconcile_interrupted_turn()
                    await call_handler.drain_turn_tts_tasks(session)
                    if stop_event.is_set() or getattr(
                        session, "pending_hangup", False
                    ):
                        handle_hangup_cancelled_turn(session, turn_snapshot)
                        break
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
                        call_handler.interrupt_turn_tts(session)
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                    await call_handler.drain_turn_tts_tasks(session)
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
                    retry_text = handle_turn_timeout(
                        session, transcript, turn_snapshot
                    )
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

                # The next question is about to play — discard stale input unless
                # caller speech arrived during this agent turn (incl. TTS gaps).
                if not is_complete and not should_preserve_pending_transcripts(
                    caller_speech_pending=session.caller_speech_pending,
                    queued_transcripts=transcript_queue.qsize(),
                ):
                    _drain_pending()

                if stop_event.is_set():
                    break

                if audio_streamed:
                    # Low-latency path: all TTS for this turn was enqueued live via
                    # on_audio_part during process_tenant_speech. Do not run post-turn
                    # remainder/batch synthesis — that duplicated questions/read-backs.
                    fin = getattr(session, "turn_streaming_finalize", None) or {}
                    vinfo(
                        logger,
                        "Low-latency turn audio complete — no post-turn TTS",
                        session=session,
                        call_id=call_id,
                        phase=Phase.TTS_DEDUP_SKIP,
                        detail=(
                            f"intended={(fin.get('intended') or response_text or '')[:80]!r} "
                            f"stream_turn_end_sent={stream_turn_end_sent}"
                        ),
                    )
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
            stop_event.set()
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
                            # Browser queue may have live-streamed chunks tagged
                            # turn_end=false; signal completion without extra audio.
                            await _emit(
                                "play_wav",
                                audio_wav_b64="",
                                duration_ms=0,
                                turn_end=True,
                            )
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
        await call_handler.clear_stream_stop_signals(call_id)
        try:
            audio_feed_queue.put_nowait(None)
        except asyncio.QueueFull:
            pass
        if streaming_stt is not None:
            await streaming_stt.close()
        session.streaming_stt_relay = None
        for t in tasks:
            if not t.done():
                t.cancel()
        for t in tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        for t in list(enqueue_tasks):
            if not t.done():
                t.cancel()
        for t in list(enqueue_tasks):
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
