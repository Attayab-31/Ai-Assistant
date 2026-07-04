"""Streaming TTS deduplication — no double-synth or doubled transcript lines."""

from app.core.conversation import (
    ConversationSession,
    compose_spoken_display,
    dedupe_repeated_block,
    reset_turn_streaming,
    streamed_audio_complete,
)


def test_streamed_audio_complete_when_fully_played():
    session = ConversationSession(call_id="t", phone_number="+1")
    session.streamed_speakable_prefix = "What is your full name?"
    session.streamed_audio_sent_during_turn = True
    assert streamed_audio_complete(session, "What is your full name?")


def test_streamed_audio_incomplete_when_remainder_left():
    session = ConversationSession(call_id="t", phone_number="+1")
    session.streamed_speakable_prefix = "Thank you."
    session.streamed_audio_sent_during_turn = True
    assert not streamed_audio_complete(
        session, "Thank you. What is your full name?"
    )


def test_reset_turn_streaming_preserves_audio_tracking_by_default():
    session = ConversationSession(call_id="t", phone_number="+1")
    session.streamed_speakable_prefix = "Excellent, thank you."
    session.streamed_audio_sent_during_turn = True
    session.streaming_ai_open = True
    reset_turn_streaming(session)
    assert session.streamed_speakable_prefix == "Excellent, thank you."
    assert session.streamed_audio_sent_during_turn is True
    assert session.streaming_ai_open is False


def test_reset_turn_streaming_full_clears_audio_tracking():
    session = ConversationSession(call_id="t", phone_number="+1")
    session.streamed_speakable_prefix = "Excellent, thank you."
    session.streamed_audio_sent_during_turn = True
    reset_turn_streaming(session, full=True)
    assert session.streamed_speakable_prefix == ""
    assert session.streamed_audio_sent_during_turn is False


def test_compose_spoken_display_readback_not_doubled():
    read_back = (
        "Just to confirm, I have your full legal name (first and last) as Dawn Smith. "
        "Did I get that right?"
    )
    display = compose_spoken_display(
        spoken=read_back,
        ack=read_back,
        follow_up="",
        response_text=read_back,
    )
    assert display == read_back
    assert display.count("Did I get that right?") == 1


def test_compose_spoken_display_ack_and_follow_up_not_tripled():
    ack = "Excellent, thank you for confirming."
    follow_up = "What is the best phone number for you?"
    spoken = f"{ack} {follow_up}"
    display = compose_spoken_display(
        spoken=spoken,
        ack=ack,
        follow_up=follow_up,
        response_text=spoken,
    )
    assert display == spoken
    assert display.count("Excellent") == 1
    assert display.count("phone number") == 1


def test_dedupe_repeated_block():
    text = "Hello there. Hello there."
    assert dedupe_repeated_block(text) == "Hello there."
