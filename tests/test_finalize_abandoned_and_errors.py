"""Finalize guards for abandoned calls and side-effect error recording."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.call_handler import (
    _merge_call_error_log,
    _trigger_email_notification,
)
from app.core.conversation import ConversationSession


def test_merge_call_error_log_preserves_abandon_metadata():
    merged = _merge_call_error_log(
        {"hangup_no_session": "No worker held session at finalize"},
        {"errors": ["stt timeout"]},
    )
    assert merged == {
        "hangup_no_session": "No worker held session at finalize",
        "errors": ["stt timeout"],
    }


@pytest.mark.asyncio
async def test_trigger_email_notification_records_queue_failure_on_running_loop():
    db = AsyncMock()
    call_id = str(uuid.uuid4())

    with patch(
        "app.services.email_service.send_screening_email_task"
    ) as task:
        task.delay.side_effect = RuntimeError("broker down")
        with patch(
            "app.services.side_effect_errors.record_side_effect_queue_failure",
            AsyncMock(),
        ) as record:
            ok = await _trigger_email_notification(
                db,
                call_id=call_id,
                phone_number="+15551234567",
            )

    assert ok is False
    record.assert_awaited_once_with(db, call_id, "email_queue", "broker down")


@pytest.mark.asyncio
async def test_finalize_impl_skips_abandoned_call():
    from app.core.call_handler import _finalize_call_impl

    session = ConversationSession(
        call_id="v3:abandoned",
        phone_number="+15551234567",
        questions=[],
    )
    call = MagicMock()
    call.id = uuid.uuid4()
    call.status = "abandoned"
    call.error_log = {"hangup_no_session": "gone"}

    db = AsyncMock()
    with patch(
        "app.db.crud.get_call_by_call_id",
        AsyncMock(return_value=call),
    ):
        with patch(
            "app.db.crud.update_call_if_active",
            AsyncMock(return_value=True),
        ) as update_active:
            with patch(
                "app.core.call_handler.get_call_providers",
                MagicMock(),
            ):
                with patch(
                    "app.core.call_handler._has_sufficient_extraction",
                    return_value=True,
                ):
                    with patch(
                        "app.core.call_handler.calculate_qualification_score",
                        return_value=(0, "review", []),
                    ):
                        result = await _finalize_call_impl(session, db)

    assert result["status"] == "skipped"
    assert result["reason"] == "abandoned"
    update_active.assert_not_awaited()


@pytest.mark.asyncio
async def test_finalize_impl_skips_persist_when_call_no_longer_active():
    from app.core.call_handler import _finalize_call_impl

    session = ConversationSession(
        call_id="v3:race",
        phone_number="+15551234567",
        questions=[],
    )
    call = MagicMock()
    call.id = uuid.uuid4()
    call.status = "in_progress"
    call.error_log = None

    db = AsyncMock()
    db.rollback = AsyncMock()
    with patch(
        "app.db.crud.get_call_by_call_id",
        AsyncMock(return_value=call),
    ):
        with patch(
            "app.db.crud.update_call_if_active",
            AsyncMock(return_value=False),
        ) as update_active:
            with patch(
                "app.core.call_handler.get_call_providers",
                MagicMock(),
            ):
                with patch(
                    "app.core.call_handler._has_sufficient_extraction",
                    return_value=True,
                ):
                    with patch(
                        "app.core.call_handler.calculate_qualification_score",
                        return_value=(50, "review", []),
                    ):
                        with patch(
                            "app.db.crud.get_tenant_by_call",
                            AsyncMock(return_value=None),
                        ):
                            result = await _finalize_call_impl(session, db)

    assert result["status"] == "skipped"
    assert result["reason"] == "not_active"
    update_active.assert_awaited_once()
    db.rollback.assert_awaited_once()


def test_record_side_effect_delivery_failure_sync_delegates_to_runner():
    from app.services.side_effect_errors import (
        record_side_effect_delivery_failure_sync,
    )

    with patch(
        "app.services.side_effect_errors.run_side_effect_db_write",
    ) as runner:
        record_side_effect_delivery_failure_sync(
            "v3:call-1",
            key="email_delivery",
            detail="smtp error",
        )
        runner.assert_called_once()
