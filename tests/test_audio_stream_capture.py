"""Streaming STT capture during agent speech — regression tests."""

from app.core.audio_stream import (
    STREAMING_TO_BATCH_CARRYOVER_MAX_BYTES,
    agent_turn_in_progress,
    append_streaming_carryover_audio,
    should_drop_streaming_transcript,
    should_preserve_pending_transcripts,
)


def test_agent_turn_in_progress_any_signal():
    assert agent_turn_in_progress(
        ai_speaking=True, outbound_pending=False, turn_task_active=False
    )
    assert agent_turn_in_progress(
        ai_speaking=False, outbound_pending=True, turn_task_active=False
    )
    assert agent_turn_in_progress(
        ai_speaking=False, outbound_pending=False, turn_task_active=True
    )
    assert not agent_turn_in_progress(
        ai_speaking=False, outbound_pending=False, turn_task_active=False
    )


def test_should_drop_streaming_transcript_echo_without_capture():
    assert should_drop_streaming_transcript(
        transcript="yes I have a job",
        listen_active=False,
        caller_speech_pending=False,
        agent_turn_in_progress=True,
    )


def test_should_keep_transcript_when_capture_opened():
    assert not should_drop_streaming_transcript(
        transcript="yes I have a job",
        listen_active=False,
        caller_speech_pending=True,
        agent_turn_in_progress=True,
    )


def test_should_keep_transcript_when_listen_active():
    assert not should_drop_streaming_transcript(
        transcript="yes I have a job",
        listen_active=True,
        caller_speech_pending=False,
        agent_turn_in_progress=True,
    )


def test_should_keep_transcript_after_agent_turn():
    assert not should_drop_streaming_transcript(
        transcript="yes I have a job",
        listen_active=False,
        caller_speech_pending=False,
        agent_turn_in_progress=False,
    )


def test_should_preserve_pending_transcripts_when_flag_set():
    assert should_preserve_pending_transcripts(
        caller_speech_pending=True,
        queued_transcripts=0,
    )


def test_should_preserve_pending_transcripts_when_queue_nonempty():
    assert should_preserve_pending_transcripts(
        caller_speech_pending=False,
        queued_transcripts=1,
    )


def test_should_drain_when_no_pending_speech():
    assert not should_preserve_pending_transcripts(
        caller_speech_pending=False,
        queued_transcripts=0,
    )


def test_append_streaming_carryover_audio_keeps_recent_window():
    buf = bytearray()
    append_streaming_carryover_audio(buf, b"a" * 6, max_bytes=10)
    append_streaming_carryover_audio(buf, b"b" * 8, max_bytes=10)
    assert bytes(buf) == b"aabbbbbbbb"


def test_append_streaming_carryover_audio_default_limit_bounds_memory():
    buf = bytearray()
    append_streaming_carryover_audio(
        buf,
        b"x" * (STREAMING_TO_BATCH_CARRYOVER_MAX_BYTES + 123),
    )
    assert len(buf) == STREAMING_TO_BATCH_CARRYOVER_MAX_BYTES
