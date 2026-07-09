"""Invalid state transition observability counter."""

from types import SimpleNamespace

from app.core.call_handler import _apply_guarded_state_transition
from app.core.screening_flow import (
    invalid_state_transition_count,
    log_state_transition,
    reset_invalid_state_transition_count,
)


def test_invalid_state_transition_increments_counter():
    reset_invalid_state_transition_count()
    log_state_transition("call-1", "IDLE", "ENDED", "test jump", questions=[])
    assert invalid_state_transition_count() == 1
    log_state_transition("call-1", "WRAP_UP", "ENDED", "normal finish", questions=[])
    assert invalid_state_transition_count() == 1


def test_guarded_transition_blocks_invalid_jump():
    reset_invalid_state_transition_count()
    session = SimpleNamespace(
        current_state="IDLE",
        call_id="call-2",
        questions=[],
        errors=[],
        add_error=lambda key, detail: session.errors.append((key, detail)),
    )
    applied = _apply_guarded_state_transition(
        session,
        "ENDED",
        "forced jump",
    )
    assert applied is False
    assert session.current_state == "IDLE"
    assert invalid_state_transition_count() == 1
