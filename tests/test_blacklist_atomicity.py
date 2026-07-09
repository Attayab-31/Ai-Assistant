"""Blacklist update safety tests."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
async def test_mutate_blacklist_adds_with_single_commit(monkeypatch):
    from app.db import crud

    invalidate = AsyncMock()
    monkeypatch.setattr("app.services.settings_cache.invalidate_settings_cache", invalidate)

    row = SimpleNamespace(
        key="blacklisted_numbers",
        value=json.dumps(["+15550001"]),
        value_type="json",
        is_sensitive=False,
        updated_by=None,
        updated_at=None,
    )

    class FakeResult:
        def scalar_one_or_none(self):
            return row

    class FakeDB:
        def __init__(self):
            self.commits = 0

        async def execute(self, _query):
            return FakeResult()

        async def delete(self, _row):
            raise AssertionError("delete should not be called")

        def add(self, _row):
            raise AssertionError("add should not be called")

        async def commit(self):
            self.commits += 1

    db = FakeDB()
    updated, cache_ok = await crud._mutate_blacklist_numbers(
        db,
        updated_by=None,
        add_phone="+15550002",
    )

    assert updated == ["+15550001", "+15550002"]
    assert cache_ok is True
    assert json.loads(row.value) == ["+15550001", "+15550002"]
    assert row.value_type == "json"
    assert row.is_sensitive is False
    assert db.commits == 1
    invalidate.assert_awaited_once()


@pytest.mark.asyncio
async def test_mutate_blacklist_creates_row_when_missing(monkeypatch):
    from app.db import crud

    invalidate = AsyncMock()
    monkeypatch.setattr("app.services.settings_cache.invalidate_settings_cache", invalidate)

    class FakeResult:
        def scalar_one_or_none(self):
            return None

    class FakeDB:
        def __init__(self):
            self.commits = 0
            self.added = []

        async def execute(self, _query):
            return FakeResult()

        async def delete(self, _row):
            raise AssertionError("delete should not be called")

        def add(self, row):
            self.added.append(row)

        async def commit(self):
            self.commits += 1

    db = FakeDB()
    updated, cache_ok = await crud._mutate_blacklist_numbers(
        db,
        updated_by=None,
        add_phone="+15550003",
    )

    assert updated == ["+15550003"]
    assert cache_ok is True
    assert len(db.added) == 1
    created = db.added[0]
    assert created.key == "blacklisted_numbers"
    assert created.value == json.dumps(["+15550003"])
    assert created.value_type == "json"
    assert created.is_sensitive is False
    assert db.commits == 1
    invalidate.assert_awaited_once()


@pytest.mark.asyncio
async def test_mutate_blacklist_normalizes_legacy_entries(monkeypatch):
    from app.db import crud

    invalidate = AsyncMock()
    monkeypatch.setattr("app.services.settings_cache.invalidate_settings_cache", invalidate)

    row = SimpleNamespace(
        key="blacklisted_numbers",
        value=json.dumps(["(555) 123-4567"]),
        value_type="json",
        is_sensitive=False,
        updated_by=None,
        updated_at=None,
    )

    class FakeResult:
        def scalar_one_or_none(self):
            return row

    class FakeDB:
        async def execute(self, _query):
            return FakeResult()

        async def commit(self):
            return None

    db = FakeDB()
    updated, cache_ok = await crud._mutate_blacklist_numbers(
        db,
        updated_by=None,
        add_phone="+15551234567",
    )

    assert updated == ["+15551234567"]
    assert cache_ok is True
    assert row.value == json.dumps(["+15551234567"])


@pytest.mark.asyncio
async def test_admin_blacklist_tenant_sanitizes_phone(monkeypatch):
    from app.api import admin as admin_api

    tenant = SimpleNamespace(
        id="tenant-1",
        phone_number="(555) 123-4567",
    )
    monkeypatch.setattr(admin_api.crud, "get_tenant_by_id", AsyncMock(return_value=tenant))
    added: list[str] = []

    async def _add(_db, phone, **kwargs):
        added.append(phone)
        return (["+15551234567"], True)

    monkeypatch.setattr(admin_api.crud, "add_to_blacklist", _add)
    monkeypatch.setattr(admin_api.crud, "update_tenant", AsyncMock())
    monkeypatch.setattr(admin_api, "_safe_create_audit_log", AsyncMock(return_value=True))

    request = SimpleNamespace(client=SimpleNamespace(host="127.0.0.1"))
    user = SimpleNamespace(id="admin-1")

    result = await admin_api.api_blacklist_tenant(
        tenant_id="tenant-1",
        request=request,
        db=object(),
        user=user,
    )

    assert added == ["+15551234567"]
    assert result["blacklisted"] is True
