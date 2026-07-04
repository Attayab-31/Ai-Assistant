"""
app/core/streaming_stt.py — Deepgram live STT with semantic turn detection.

Replaces silence-only client/server VAD for turn boundaries. Deepgram combines
interim results, endpointing (speech_final), and utterance_end_ms so mid-thought
pauses ("I will move… next month") stay one turn instead of splitting.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable

from deepgram import DeepgramClient, LiveOptions, LiveTranscriptionEvents

from config import settings

logger = logging.getLogger(__name__)

# Tuned for natural phone conversation. endpointing is deliberately generous:
# at short values (~300-450ms) Deepgram closes the segment on a normal pause
# between words (e.g. a first and last name, "Muhammad … Attayab"), and needs
# ~100-200ms to re-lock when speech resumes — which clips the next word. A
# ~1000ms window keeps ordinary inter-word/inter-sentence pauses (300-800ms)
# inside a single segment, so long answers aren't truncated. utterance_end_ms
# (word-timing based, ignores background noise) is the turn boundary; 1000ms is
# Deepgram's documented floor and gives a natural, not-rushed turn handoff.
DEFAULT_ENDPOINTING_MS = 1000
DEFAULT_UTTERANCE_END_MS = 1000

# Keyterm prompting (nova-3, English only) biases recognition toward known,
# frequently-mangled vocabulary without hurting general accuracy. We boost the
# email/spelling vocabulary callers use most ("at", "dot", common providers)
# and screening-specific phrases that drive flow decisions / FAQ matching.
DEFAULT_KEYTERMS: tuple[str, ...] = (
    "gmail",
    "yahoo",
    "hotmail",
    "outlook",
    "icloud",
    "proton",
    "dot com",
    "dot net",
    "dot org",
    "Section 8",
    "voucher",
    "co-signer",
    "guarantor",
    "eviction",
    "lease",
    "deposit",
    "credit score",
    "move-in",
)

OnInterim = Callable[[str], Awaitable[None]]
# Fired with the first real transcribed words of a caller turn. Used for
# transcript-gated barge-in: the agent only stops talking when the speech model
# returns actual words, not on raw audio energy (noise / echo / silence).
OnSpeechStarted = Callable[[str], Awaitable[None]]


class DeepgramStreamingSession:
    """One live Deepgram socket per call; feed audio continuously, get turns."""

    def __init__(
        self,
        *,
        model: str = "nova-3",
        language: str = "en-US",
        encoding: str = "mulaw",
        sample_rate: int = 8000,
        endpointing_ms: int = DEFAULT_ENDPOINTING_MS,
        utterance_end_ms: int = DEFAULT_UTTERANCE_END_MS,
        keyterms: tuple[str, ...] | None = DEFAULT_KEYTERMS,
        on_interim: OnInterim | None = None,
        on_speech_started: OnSpeechStarted | None = None,
    ) -> None:
        self.model = model
        self.language = language
        self.encoding = encoding
        self.sample_rate = sample_rate
        from app.core.voice_latency import _clamp_utterance_end_ms

        self.endpointing_ms = endpointing_ms
        self.utterance_end_ms = _clamp_utterance_end_ms(utterance_end_ms)
        self.keyterms = keyterms or ()
        self.on_interim = on_interim
        self.on_speech_started = on_speech_started

        self._client: DeepgramClient | None = None
        self._connection = None
        self._final_parts: list[str] = []
        self._transcript_queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._started = False
        self._closed = False
        self._last_emitted = ""
        self._last_emit_at = 0.0
        self._last_feed_at = 0.0
        # True once on_speech_started has fired for the current turn, so we only
        # signal barge-in on the first words, not every interim packet.
        self._speech_started_fired = False
        self._keepalive_task: asyncio.Task | None = None
        self._turn_emit_task: asyncio.Task | None = None

    async def start(self) -> None:
        if self._started:
            return
        if not settings.deepgram_api_key:
            raise ValueError("DEEPGRAM_API_KEY not set")

        self._client = DeepgramClient(settings.deepgram_api_key)
        options = LiveOptions(
            model=self.model,
            language=self.language,
            encoding=self.encoding,
            sample_rate=self.sample_rate,
            channels=1,
            punctuate=True,
            smart_format=True,
            interim_results=True,
            utterance_end_ms=str(self.utterance_end_ms),
            endpointing=str(self.endpointing_ms),
            vad_events=True,
        )

        # Keyterm prompting is nova-3 + English only; attach defensively so an
        # older SDK or non-nova-3 model never breaks session startup.
        if (
            self.keyterms
            and self.model.startswith("nova-3")
            and self.language.startswith("en")
        ):
            try:
                options.keyterm = list(self.keyterms)
            except Exception as exc:  # pragma: no cover - SDK compatibility guard
                logger.debug(
                    "Keyterm prompting unavailable, continuing without: %s", exc
                )

        connection = self._client.listen.asynclive.v("1")

        async def on_transcript(_conn, *, result, **_kwargs) -> None:
            try:
                alt = result.channel.alternatives[0]
                text = (alt.transcript or "").strip()
                if not text:
                    return
                # Transcript-gated barge-in: the first real words of a turn are
                # the signal that the caller is genuinely speaking. Fire once
                # per turn, before anything else, so playback can stop fast.
                if not self._speech_started_fired and self.on_speech_started:
                    self._speech_started_fired = True
                    try:
                        await self.on_speech_started(text)
                    except Exception as exc:
                        logger.debug("on_speech_started error: %s", exc)
                if result.is_final:
                    self._final_parts.append(text)
                elif self.on_interim:
                    await self.on_interim(text)
                if result.speech_final:
                    # Safety net only — delay past utterance_end_ms so utterance_end
                    # wins and joins multi-segment answers into one turn.
                    delay = (self.utterance_end_ms / 1000.0) + 0.4
                    await self._schedule_turn_emit("speech_final", delay_s=delay)
            except Exception as exc:
                logger.warning("Streaming STT transcript parse error: %s", exc)

        async def on_utterance_end(_conn, **_kwargs) -> None:
            await self._cancel_turn_emit()
            await self._emit_turn("utterance_end")

        async def on_error(_conn, *, error, **_kwargs) -> None:
            logger.error("Deepgram streaming error: %s", error)
            await self._transcript_queue.put(None)

        connection.on(LiveTranscriptionEvents.Transcript, on_transcript)
        connection.on(LiveTranscriptionEvents.UtteranceEnd, on_utterance_end)
        connection.on(LiveTranscriptionEvents.Error, on_error)

        await connection.start(options)
        self._connection = connection
        self._started = True
        # Deepgram closes the socket (1011) after ~10 s without audio. During
        # the greeting / agent turns we feed no audio, so a periodic KeepAlive
        # text message keeps the live session open between caller turns.
        self._keepalive_task = asyncio.create_task(self._keepalive_loop())
        logger.info(
            "Deepgram streaming STT started (model=%s, encoding=%s, "
            "endpointing=%sms, utterance_end=%sms)",
            self.model,
            self.encoding,
            self.endpointing_ms,
            self.utterance_end_ms,
        )

    async def feed(self, chunk: bytes) -> None:
        if not chunk or self._closed or not self._connection:
            return
        self._last_feed_at = time.monotonic()
        try:
            await self._connection.send(chunk)
        except Exception as exc:
            logger.warning("Deepgram feed error: %s", exc)

    async def _keepalive_loop(self) -> None:
        """Send KeepAlive pings while the caller isn't actively speaking."""
        try:
            while not self._closed:
                await asyncio.sleep(3.0)
                if self._closed or self._connection is None:
                    break
                # Only ping when audio has been idle — avoids redundant traffic
                # mid-utterance while still covering greeting/agent-speech gaps.
                if (time.monotonic() - self._last_feed_at) > 2.0:
                    try:
                        await self._connection.send(json.dumps({"type": "KeepAlive"}))
                    except Exception as exc:
                        logger.debug("KeepAlive send failed: %s", exc)
        except asyncio.CancelledError:
            pass

    async def _cancel_turn_emit(self) -> None:
        if self._turn_emit_task is not None:
            self._turn_emit_task.cancel()
            self._turn_emit_task = None

    async def _schedule_turn_emit(self, reason: str, *, delay_s: float) -> None:
        await self._cancel_turn_emit()

        async def _delayed() -> None:
            try:
                await asyncio.sleep(delay_s)
                await self._emit_turn(reason)
            except asyncio.CancelledError:
                pass
            finally:
                self._turn_emit_task = None

        self._turn_emit_task = asyncio.create_task(_delayed())

    async def _emit_turn(self, reason: str) -> None:
        text = " ".join(p for p in self._final_parts if p).strip()
        self._final_parts.clear()
        # Re-arm barge-in detection for the next turn.
        self._speech_started_fired = False
        if not text:
            return
        now = time.monotonic()
        if text == self._last_emitted and (now - self._last_emit_at) < 2.0:
            return
        self._last_emitted = text
        self._last_emit_at = now
        logger.debug("Streaming turn (%s): %r", reason, text[:80])
        await self._transcript_queue.put(text)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._keepalive_task is not None:
            self._keepalive_task.cancel()
        await self._cancel_turn_emit()
        if self._connection:
            try:
                await self._connection.finish()
            except Exception as exc:
                logger.debug("Deepgram finish: %s", exc)
        await self._transcript_queue.put(None)

    async def transcripts(self):
        """Yield finalized caller turns until the session closes."""
        while True:
            item = await self._transcript_queue.get()
            if item is None:
                break
            yield item
