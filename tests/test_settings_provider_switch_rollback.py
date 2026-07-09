"""Provider-switch rollback safety tests."""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException


@pytest.mark.asyncio
async def test_apply_provider_switch_settings_success(monkeypatch):
    from app.api import settings as settings_api

    monkeypatch.setattr(
        settings_api,
        "_snapshot_settings",
        AsyncMock(
            return_value={
                "active_llm_provider": settings_api._SettingSnapshotEntry(
                    True, "groq", False
                )
            }
        ),
    )
    restore = AsyncMock()
    monkeypatch.setattr(settings_api, "_restore_settings", restore)
    set_settings_bulk = AsyncMock(return_value=True)
    monkeypatch.setattr(settings_api.crud, "set_settings_bulk", set_settings_bulk)
    reload_db = AsyncMock(return_value=None)
    monkeypatch.setattr(settings_api.provider_registry, "reload_from_db", reload_db)
    monkeypatch.setattr(
        "app.core.redis_client.acquire_once",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        "app.core.redis_client.cache_delete",
        AsyncMock(),
    )

    await settings_api._apply_provider_switch_settings(
        db=object(),
        updates={"active_llm_provider": "openai", "active_openai_model": "gpt-4.1"},
        label="LLM",
        updated_by="admin-id",
    )

    set_settings_bulk.assert_awaited_once()
    restore.assert_not_called()
    assert reload_db.await_count == 1


@pytest.mark.asyncio
async def test_apply_provider_switch_settings_rolls_back_on_reload_failure(monkeypatch):
    from app.api import settings as settings_api

    snapshot = {
        "active_stt_provider": settings_api._SettingSnapshotEntry(
            True, "deepgram", False
        )
    }
    monkeypatch.setattr(
        settings_api,
        "_snapshot_settings",
        AsyncMock(return_value=snapshot),
    )
    restore = AsyncMock()
    monkeypatch.setattr(settings_api, "_restore_settings", restore)
    set_settings_bulk = AsyncMock(return_value=True)
    monkeypatch.setattr(settings_api.crud, "set_settings_bulk", set_settings_bulk)
    reload_db = AsyncMock(side_effect=[RuntimeError("reload failed"), None])
    monkeypatch.setattr(settings_api.provider_registry, "reload_from_db", reload_db)
    monkeypatch.setattr(
        "app.core.redis_client.acquire_once",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        "app.core.redis_client.cache_delete",
        AsyncMock(),
    )

    updates = {"active_stt_provider": "groq"}
    with pytest.raises(HTTPException, match="Failed to switch provider"):
        await settings_api._apply_provider_switch_settings(
            db=object(),
            updates=updates,
            label="STT",
            updated_by="admin-id",
        )

    restore.assert_awaited_once()
    assert restore.await_args.kwargs["written_values"] == updates
    assert reload_db.await_count == 2


@pytest.mark.asyncio
async def test_apply_provider_switch_settings_rejects_concurrent_switch(monkeypatch):
    from app.api import settings as settings_api

    monkeypatch.setattr(
        "app.core.redis_client.acquire_once",
        AsyncMock(return_value=False),
    )

    with pytest.raises(HTTPException, match="in progress"):
        await settings_api._apply_provider_switch_settings(
            db=object(),
            updates={"active_llm_provider": "openai"},
            label="LLM",
            updated_by="admin-id",
        )


@pytest.mark.asyncio
async def test_restore_settings_recreates_previous_state(monkeypatch):
    from app.api import settings as settings_api

    invalidate = AsyncMock()
    monkeypatch.setattr("app.services.settings_cache.invalidate_settings_cache", invalidate)

    active_llm = SimpleNamespace(
        key="active_llm_provider",
        value="openai",
        is_sensitive=False,
        updated_at=datetime.now(UTC),
    )
    stale_row = SimpleNamespace(
        key="active_tts_provider",
        value="deepgram",
        is_sensitive=False,
        updated_at=datetime.now(UTC),
    )

    class FakeDB:
        def __init__(self) -> None:
            self.deleted = []
            self.commit_count = 0
            self.added = []

        async def execute(self, _query):
            class _Result:
                def scalars(self_inner):
                    class _Scalars:
                        def __iter__(self_scalars):
                            return iter([active_llm, stale_row])

                    return _Scalars()

            return _Result()

        async def delete(self, row):
            self.deleted.append(row)

        def add(self, row):
            self.added.append(row)

        async def commit(self):
            self.commit_count += 1

    db = FakeDB()
    snapshot = {
        "active_llm_provider": settings_api._SettingSnapshotEntry(True, "groq", False),
        "active_tts_provider": settings_api._SettingSnapshotEntry(False, "", False),
    }

    await settings_api._restore_settings(
        db=db,
        snapshot=snapshot,
        keys=["active_llm_provider", "active_tts_provider"],
    )

    assert active_llm.value == "groq"
    assert active_llm.is_sensitive is False
    assert db.deleted == [stale_row]
    assert db.commit_count == 1
    assert db.added == []
    invalidate.assert_awaited_once()


@pytest.mark.asyncio
async def test_restore_settings_skips_stale_rollback_when_newer_value(monkeypatch):
    from app.api import settings as settings_api

    invalidate = AsyncMock()
    monkeypatch.setattr("app.services.settings_cache.invalidate_settings_cache", invalidate)

    active_llm = SimpleNamespace(
        key="active_llm_provider",
        value="gemini",
        is_sensitive=False,
        updated_at=datetime.now(UTC),
    )

    class FakeDB:
        def __init__(self) -> None:
            self.commit_count = 0

        async def execute(self, _query):
            class _Result:
                def scalars(self_inner):
                    class _Scalars:
                        def __iter__(self_scalars):
                            return iter([active_llm])

                    return _Scalars()

            return _Result()

        async def commit(self):
            self.commit_count += 1

    db = FakeDB()
    snapshot = {
        "active_llm_provider": settings_api._SettingSnapshotEntry(True, "groq", False),
    }

    await settings_api._restore_settings(
        db=db,
        snapshot=snapshot,
        keys=["active_llm_provider"],
        written_values={"active_llm_provider": "openai"},
    )

    assert active_llm.value == "gemini"
    assert db.commit_count == 0
    invalidate.assert_not_awaited()
