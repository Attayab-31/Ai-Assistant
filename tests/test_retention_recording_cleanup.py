"""Recording cleanup — retries, DB pointer safety, orphan queue."""

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.db import crud
from app.services.recording_cleanup import (
    RecordingRemovalResult,
    enqueue_orphaned_recording,
    is_managed_recording_path,
    remove_recording,
    removal_ok_for_db_clear,
    retry_pending_recording_deletes,
)


def _batch(rows):
    result = MagicMock()
    result.all.return_value = rows
    return result


def _delete_result(rowcount):
    result = MagicMock()
    result.rowcount = rowcount
    return result


def test_is_managed_recording_path():
    assert is_managed_recording_path("recordings/call-1.mp3")
    assert not is_managed_recording_path("https://telnyx.example/rec.mp3")
    assert not is_managed_recording_path(None)


def test_removal_ok_for_db_clear():
    assert removal_ok_for_db_clear(RecordingRemovalResult.REMOVED)
    assert removal_ok_for_db_clear(RecordingRemovalResult.EXTERNAL)
    assert not removal_ok_for_db_clear(RecordingRemovalResult.FAILED)


@pytest.mark.asyncio
async def test_remove_recording_external_url_skips_storage():
    result = await remove_recording("https://example.com/rec.mp3")
    assert result == RecordingRemovalResult.EXTERNAL


@pytest.mark.asyncio
async def test_remove_recording_retries_then_succeeds(monkeypatch):
    delete = AsyncMock(side_effect=[False, False, True])
    monkeypatch.setattr(
        "app.services.storage_service.storage_service.delete_recording",
        delete,
    )
    sleep = AsyncMock()
    monkeypatch.setattr("app.services.recording_cleanup.asyncio.sleep", sleep)

    result = await remove_recording("recordings/x.mp3", retries=3)

    assert result == RecordingRemovalResult.REMOVED
    assert delete.await_count == 3


@pytest.mark.asyncio
async def test_remove_recording_failed_after_retries(monkeypatch):
    delete = AsyncMock(return_value=False)
    monkeypatch.setattr(
        "app.services.storage_service.storage_service.delete_recording",
        delete,
    )
    monkeypatch.setattr("app.services.recording_cleanup.asyncio.sleep", AsyncMock())

    result = await remove_recording("recordings/x.mp3", retries=2)

    assert result == RecordingRemovalResult.FAILED
    assert delete.await_count == 2


@pytest.mark.asyncio
async def test_enqueue_orphaned_recording_uses_redis():
    redis = AsyncMock()
    redis.sadd = AsyncMock()
    redis.expire = AsyncMock()
    with patch("app.core.redis_client.get_redis", return_value=redis):
        await enqueue_orphaned_recording("recordings/orphan.mp3")
    redis.sadd.assert_awaited_once()
    redis.expire.assert_awaited_once()


@pytest.mark.asyncio
async def test_retry_pending_recording_deletes():
    redis = AsyncMock()
    redis.smembers = AsyncMock(return_value={"recordings/a.mp3", "recordings/b.mp3"})
    redis.srem = AsyncMock()
    redis.scard = AsyncMock(return_value=1)

    with patch("app.core.redis_client.get_redis", return_value=redis):
        with patch(
            "app.services.recording_cleanup.remove_recording",
            AsyncMock(
                side_effect=[
                    RecordingRemovalResult.REMOVED,
                    RecordingRemovalResult.FAILED,
                ]
            ),
        ) as remove:
            summary = await retry_pending_recording_deletes(limit=10)

    assert summary["retried"] == 2
    assert summary["removed"] == 1
    assert summary["remaining"] == 1
    remove.assert_awaited()
    redis.srem.assert_awaited_once()


@pytest.mark.asyncio
async def test_purge_calls_before_deletes_recordings(monkeypatch):
    cid_with = uuid.uuid4()
    cid_without = uuid.uuid4()
    rows = [(cid_with, "recordings/a.mp3"), (cid_without, None)]

    db = AsyncMock()
    db.execute = AsyncMock(side_effect=[_batch(rows), _delete_result(2)])

    remove = AsyncMock(return_value=RecordingRemovalResult.REMOVED)
    monkeypatch.setattr(
        "app.services.recording_cleanup.remove_recording", remove
    )
    enqueue = AsyncMock()
    monkeypatch.setattr(
        "app.services.recording_cleanup.enqueue_orphaned_recording", enqueue
    )

    total = await crud.purge_calls_before(db, datetime.now(UTC), batch_size=500)

    assert total == 2
    remove.assert_awaited_once_with("recordings/a.mp3")
    enqueue.assert_not_awaited()
    db.commit.assert_awaited()


@pytest.mark.asyncio
async def test_purge_calls_before_enqueues_when_storage_delete_fails(monkeypatch):
    cid = uuid.uuid4()
    rows = [(cid, "recordings/fail.mp3")]

    db = AsyncMock()
    db.execute = AsyncMock(side_effect=[_batch(rows), _delete_result(1)])

    remove = AsyncMock(return_value=RecordingRemovalResult.FAILED)
    monkeypatch.setattr(
        "app.services.recording_cleanup.remove_recording", remove
    )
    enqueue = AsyncMock()
    monkeypatch.setattr(
        "app.services.recording_cleanup.enqueue_orphaned_recording", enqueue
    )

    total = await crud.purge_calls_before(db, datetime.now(UTC), batch_size=500)

    assert total == 1
    enqueue.assert_awaited_once_with("recordings/fail.mp3")


@pytest.mark.asyncio
async def test_purge_calls_before_no_recordings(monkeypatch):
    rows = [(uuid.uuid4(), None), (uuid.uuid4(), None)]

    db = AsyncMock()
    db.execute = AsyncMock(side_effect=[_batch(rows), _delete_result(2)])

    remove = AsyncMock()
    monkeypatch.setattr(
        "app.services.recording_cleanup.remove_recording", remove
    )

    total = await crud.purge_calls_before(db, datetime.now(UTC), batch_size=500)

    assert total == 2
    remove.assert_not_awaited()


@pytest.mark.asyncio
async def test_purge_soft_deleted_calls_before_deletes_recordings(monkeypatch):
    cid = uuid.uuid4()
    rows = [(cid, "recordings/legacy.mp3")]

    db = AsyncMock()
    db.execute = AsyncMock(side_effect=[_batch(rows), _delete_result(1)])

    remove = AsyncMock(return_value=RecordingRemovalResult.REMOVED)
    monkeypatch.setattr(
        "app.services.recording_cleanup.remove_recording", remove
    )

    total = await crud.purge_soft_deleted_calls_before(
        db, datetime.now(UTC), batch_size=500
    )

    assert total == 1
    remove.assert_awaited_once_with("recordings/legacy.mp3")


@pytest.mark.asyncio
async def test_retention_recording_purge_skips_db_clear_on_failure(monkeypatch):
    from app.services import retention_service

    call_id = uuid.uuid4()
    db = AsyncMock()
    settings = {
        "retention_enabled": "true",
        "retention_recording_days": "90",
        "retention_calls_days": "0",
        "retention_audit_days": "0",
        "retention_soft_deleted_days": "0",
        "retention_stale_call_hours": "0",
    }

    class _SessionCtx:
        async def __aenter__(self):
            return db

        async def __aexit__(self, *args):
            return False

    monkeypatch.setattr(
        "app.db.database.AsyncSessionLocal", lambda: _SessionCtx()
    )
    monkeypatch.setattr(
        "app.db.crud.get_all_settings", AsyncMock(return_value=settings)
    )
    monkeypatch.setattr(
        "app.core.redis_client.acquire_once", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(
        "app.services.recording_cleanup.retry_pending_recording_deletes",
        AsyncMock(return_value={"retried": 0, "removed": 0, "remaining": 0}),
    )
    monkeypatch.setattr(
        "app.db.crud.get_recordings_before",
        AsyncMock(
            side_effect=[
                [(call_id, "recordings/old.mp3", datetime.now(UTC))],
            ]
        ),
    )
    clear = AsyncMock()
    monkeypatch.setattr("app.db.crud.clear_recording_url", clear)
    enqueue = AsyncMock()
    monkeypatch.setattr(
        "app.services.recording_cleanup.enqueue_orphaned_recording", enqueue
    )
    monkeypatch.setattr(
        "app.services.recording_cleanup.remove_recording",
        AsyncMock(return_value=RecordingRemovalResult.FAILED),
    )

    summary = await retention_service._run_retention()

    assert summary["recording_delete_failures"] == 1
    assert summary["recordings"] == 0
    clear.assert_not_awaited()
    enqueue.assert_awaited_once_with("recordings/old.mp3")


@pytest.mark.asyncio
async def test_retention_recording_purge_clears_db_on_success(monkeypatch):
    from app.services import retention_service

    call_id = uuid.uuid4()
    db = AsyncMock()
    settings = {
        "retention_enabled": "true",
        "retention_recording_days": "90",
        "retention_calls_days": "0",
        "retention_audit_days": "0",
        "retention_soft_deleted_days": "0",
        "retention_stale_call_hours": "0",
    }

    class _SessionCtx:
        async def __aenter__(self):
            return db

        async def __aexit__(self, *args):
            return False

    monkeypatch.setattr(
        "app.db.database.AsyncSessionLocal", lambda: _SessionCtx()
    )
    monkeypatch.setattr(
        "app.db.crud.get_all_settings", AsyncMock(return_value=settings)
    )
    monkeypatch.setattr(
        "app.core.redis_client.acquire_once", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(
        "app.services.recording_cleanup.retry_pending_recording_deletes",
        AsyncMock(return_value={"retried": 0, "removed": 0, "remaining": 0}),
    )
    monkeypatch.setattr(
        "app.db.crud.get_recordings_before",
        AsyncMock(
            side_effect=[
                [(call_id, "recordings/old.mp3", datetime.now(UTC))],
                [],
            ]
        ),
    )
    clear = AsyncMock()
    monkeypatch.setattr("app.db.crud.clear_recording_url", clear)
    monkeypatch.setattr(
        "app.services.recording_cleanup.remove_recording",
        AsyncMock(return_value=RecordingRemovalResult.REMOVED),
    )

    summary = await retention_service._run_retention()

    assert summary["recordings"] == 1
    assert summary["recording_delete_failures"] == 0
    clear.assert_awaited_once_with(db, call_id)


@pytest.mark.asyncio
async def test_retention_recording_purge_uses_keyset_cursor_when_mutating_rows(
    monkeypatch,
):
    """Clearing recording_url in-loop must not skip later eligible rows."""
    from app.services import retention_service

    db = AsyncMock()
    settings = {
        "retention_enabled": "true",
        "retention_recording_days": "90",
        "retention_calls_days": "0",
        "retention_audit_days": "0",
        "retention_soft_deleted_days": "0",
        "retention_stale_call_hours": "0",
    }

    class _SessionCtx:
        async def __aenter__(self):
            return db

        async def __aexit__(self, *args):
            return False

    after_markers: list[tuple[datetime | None, uuid.UUID | None]] = []
    base = datetime(2026, 1, 1, tzinfo=UTC)
    rows = [
        (uuid.uuid4(), "recordings/a.mp3", base),
        (uuid.uuid4(), "recordings/b.mp3", base),
        (uuid.uuid4(), "recordings/c.mp3", base),
    ]

    async def _get_recordings(
        _db,
        _cutoff,
        *,
        limit=200,
        after_created_at=None,
        after_id=None,
    ):
        after_markers.append((after_created_at, after_id))
        if after_created_at is None:
            return rows[:2]
        if after_created_at == base and after_id == rows[1][0]:
            return [rows[2]]
        return []

    monkeypatch.setattr(
        "app.db.database.AsyncSessionLocal", lambda: _SessionCtx()
    )
    monkeypatch.setattr(
        "app.db.crud.get_all_settings", AsyncMock(return_value=settings)
    )
    monkeypatch.setattr(
        "app.core.redis_client.acquire_once", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(
        "app.services.recording_cleanup.retry_pending_recording_deletes",
        AsyncMock(return_value={"retried": 0, "removed": 0, "remaining": 0}),
    )
    monkeypatch.setattr(retention_service, "RECORDING_BATCH_SIZE", 2)
    monkeypatch.setattr("app.db.crud.get_recordings_before", _get_recordings)
    monkeypatch.setattr("app.db.crud.clear_recording_url", AsyncMock())
    enqueue = AsyncMock()
    monkeypatch.setattr(
        "app.services.recording_cleanup.enqueue_orphaned_recording", enqueue
    )
    monkeypatch.setattr(
        "app.services.recording_cleanup.remove_recording",
        AsyncMock(return_value=RecordingRemovalResult.REMOVED),
    )

    summary = await retention_service._run_retention()

    assert after_markers[0] == (None, None)
    assert after_markers[1] == (base, rows[1][0])
    assert summary["recordings"] == 3
    assert summary["recording_delete_failures"] == 0
    enqueue.assert_not_awaited()


@pytest.mark.asyncio
async def test_admin_delete_call_warns_when_recording_delete_fails(monkeypatch):
    from types import SimpleNamespace

    from app.api import admin as admin_api

    call = SimpleNamespace(
        call_id="call-1",
        phone_number="+15550001",
        recording_url="recordings/x.mp3",
    )
    monkeypatch.setattr(admin_api.crud, "get_call_by_uuid", AsyncMock(return_value=call))
    monkeypatch.setattr(admin_api.crud, "hard_delete_call", AsyncMock())
    monkeypatch.setattr(
        admin_api,
        "_safe_create_audit_log",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        "app.services.recording_cleanup.remove_recording",
        AsyncMock(return_value=RecordingRemovalResult.FAILED),
    )
    enqueue = AsyncMock()
    monkeypatch.setattr(
        "app.services.recording_cleanup.enqueue_orphaned_recording", enqueue
    )

    request = SimpleNamespace(client=SimpleNamespace(host="127.0.0.1"))
    user = SimpleNamespace(id="admin-1")

    result = await admin_api.api_delete_call(
        call_id="00000000-0000-0000-0000-000000000001",
        request=request,
        db=object(),
        user=user,
    )

    assert result["deleted"] is True
    assert result["recording_delete_failed"] is True
    assert any("recording" in w.lower() for w in result.get("warnings", []))
    enqueue.assert_awaited_once_with("recordings/x.mp3")


@pytest.mark.asyncio
async def test_admin_delete_marks_call_tombstone(monkeypatch):
    from types import SimpleNamespace

    from app.api import admin as admin_api

    call = SimpleNamespace(
        call_id="call-tombstone",
        phone_number="+15550001",
        recording_url=None,
    )
    monkeypatch.setattr(admin_api.crud, "get_call_by_uuid", AsyncMock(return_value=call))
    monkeypatch.setattr(admin_api.crud, "hard_delete_call", AsyncMock())
    monkeypatch.setattr(admin_api, "_safe_create_audit_log", AsyncMock(return_value=True))
    mark_deleted = AsyncMock()
    monkeypatch.setattr(
        "app.core.redis_client.mark_call_admin_deleted",
        mark_deleted,
    )

    await admin_api.api_delete_call(
        call_id="00000000-0000-0000-0000-000000000002",
        request=SimpleNamespace(client=SimpleNamespace(host="127.0.0.1")),
        db=object(),
        user=SimpleNamespace(id="admin-1"),
    )
    mark_deleted.assert_awaited_once_with("call-tombstone")

