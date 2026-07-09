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
