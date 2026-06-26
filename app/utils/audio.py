"""Helpers for Telnyx mulaw media payloads."""

import base64
import io
import logging
import struct
import wave
from collections.abc import Iterable

import audioop

logger = logging.getLogger(__name__)


def decode_telnyx_payload(payload: str) -> bytes:
    """Decode a Telnyx base64 media payload into raw audio bytes."""
    if not payload:
        return b""
    return base64.b64decode(payload)


def encode_telnyx_payload(audio_bytes: bytes) -> str:
    """Encode raw audio bytes for a Telnyx media event."""
    if not audio_bytes:
        return ""
    return base64.b64encode(audio_bytes).decode("ascii")


def chunk_audio(audio_bytes: bytes, chunk_size: int = 160) -> Iterable[bytes]:
    """Yield fixed-size audio chunks for real-time playback pacing."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    for index in range(0, len(audio_bytes), chunk_size):
        yield audio_bytes[index : index + chunk_size]


def is_silence(audio_bytes: bytes, threshold: int = 40) -> bool:
    """Detect likely silence in 8-bit mulaw audio chunks."""
    if not audio_bytes:
        return True
    try:
        linear = audioop.ulaw2lin(audio_bytes, 2)
        return audioop.rms(linear, 2) < threshold
    except audioop.error:
        return False


def mulaw_to_wav(mulaw_bytes: bytes, sample_rate: int = 8000) -> bytes:
    """Convert mulaw 8kHz bytes to a browser-playable PCM16 WAV file."""
    if not mulaw_bytes:
        return b""
    pcm = audioop.ulaw2lin(mulaw_bytes, 2)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buf.getvalue()


def wav_to_mulaw(wav_bytes: bytes, target_rate: int = 8000) -> bytes:
    """Convert an arbitrary WAV blob (typically from the browser) to mulaw 8kHz."""
    if not wav_bytes:
        return b""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())

    if channels == 2:
        frames = audioop.tomono(frames, sample_width, 1, 1)
    if sample_width != 2:
        frames = audioop.lin2lin(frames, sample_width, 2)
        sample_width = 2
    if rate != target_rate:
        frames, _ = audioop.ratecv(frames, sample_width, 1, rate, target_rate, None)
    return audioop.lin2ulaw(frames, sample_width)


def any_audio_to_mulaw(audio_bytes: bytes, target_rate: int = 8000) -> bytes:
    """
    Convert browser-uploaded audio (WAV, WebM, OGG, MP3, etc.) to mulaw 8kHz.

    Tries native WAV parsing first, then falls back to pydub/ffmpeg.
    """
    if not audio_bytes:
        return b""

    # Fast path: valid WAV from browsers that support audio/wav recording.
    try:
        return wav_to_mulaw(audio_bytes, target_rate)
    except (wave.Error, struct.error, EOFError) as e:
        logger.debug("WAV to mulaw conversion failed, trying pydub: %s", e)

    try:
        from pydub import AudioSegment

        segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
        segment = segment.set_channels(1).set_frame_rate(target_rate)
        pcm = segment.raw_data  # 16-bit little-endian mono
        return audioop.lin2ulaw(pcm, 2)
    except Exception as e:
        raise ValueError(
            f"Could not decode audio ({len(audio_bytes)} bytes). "
            "Supported formats: WAV, WebM, OGG, MP3. "
            "Install ffmpeg for WebM/Opus support."
        ) from e
