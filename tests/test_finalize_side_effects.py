"""Post-call email/CRM idempotency and enqueue-after-claim behavior."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.db.crud import (
    SIDE_EFFECTS_CHANNELS_KEY,
    SIDE_EFFECTS_CLAIM_KEY,
    claim_finalize_side_effect_channel,
    claim_finalize_side_effects,
    is_finalize_side_effect_channel_claimed,
    is_finalize_side_effects_claimed,
    release_finalize_side_effect_channel,
)


@pytest.mark.asyncio
async def test_claim_finalize_side_effects_first_claim_wins():
    tenant_id = uuid.uuid4()
    tenant = MagicMock()
    tenant.id = tenant_id
    tenant.normalized_data = {}

    db = AsyncMock()
    db.execute = AsyncMock(return_value=MagicMock(scalar_one_or_none=lambda: tenant))

    assert await claim_finalize_side_effects(db, tenant_id) is True
    assert tenant.normalized_data[SIDE_EFFECTS_CLAIM_KEY] is True
    assert tenant.normalized_data[SIDE_EFFECTS_CHANNELS_KEY]
    db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_claim_finalize_side_effects_rejects_duplicate():
    tenant_id = uuid.uuid4()
    tenant = MagicMock()
    tenant.id = tenant_id
    tenant.normalized_data = {SIDE_EFFECTS_CLAIM_KEY: True}

    db = AsyncMock()
    db.execute = AsyncMock(return_value=MagicMock(scalar_one_or_none=lambda: tenant))

    assert await claim_finalize_side_effects(db, tenant_id) is False
    db.rollback.assert_not_awaited()
    db.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_is_finalize_side_effects_claimed_reads_tenant_flag():
    tenant_id = uuid.uuid4()
    tenant = MagicMock()
    tenant.normalized_data = {SIDE_EFFECTS_CLAIM_KEY: True}

    db = AsyncMock()
    with patch("app.db.crud.get_tenant_by_id", AsyncMock(return_value=tenant)):
        assert await is_finalize_side_effects_claimed(db, tenant_id) is True


@pytest.mark.asyncio
async def test_claim_finalize_side_effect_channel_records_only_that_channel():
    tenant_id = uuid.uuid4()
    tenant = MagicMock()
    tenant.id = tenant_id
    tenant.normalized_data = {}

    db = AsyncMock()
    db.execute = AsyncMock(return_value=MagicMock(scalar_one_or_none=lambda: tenant))

    assert await claim_finalize_side_effect_channel(db, tenant_id, "email") is True
    assert "email" in tenant.normalized_data[SIDE_EFFECTS_CHANNELS_KEY]
    assert "crm" not in tenant.normalized_data[SIDE_EFFECTS_CHANNELS_KEY]


@pytest.mark.asyncio
async def test_release_finalize_side_effect_channel_clears_failed_enqueue_claim():
    tenant_id = uuid.uuid4()
    tenant = MagicMock()
    tenant.id = tenant_id
    tenant.normalized_data = {
        SIDE_EFFECTS_CHANNELS_KEY: {"email": "2026-01-01T00:00:00+00:00"}
    }

    db = AsyncMock()
    db.execute = AsyncMock(return_value=MagicMock(scalar_one_or_none=lambda: tenant))

    await release_finalize_side_effect_channel(db, tenant_id, "email")
    assert tenant.normalized_data == {}


@pytest.mark.asyncio
async def test_is_finalize_side_effect_channel_claimed_honors_legacy_flag():
    tenant_id = uuid.uuid4()
    tenant = MagicMock()
    tenant.normalized_data = {SIDE_EFFECTS_CLAIM_KEY: True}

    db = AsyncMock()
    with patch("app.db.crud.get_tenant_by_id", AsyncMock(return_value=tenant)):
        assert await is_finalize_side_effect_channel_claimed(db, tenant_id, "crm") is True


@pytest.mark.asyncio
async def test_claim_finalize_side_effect_channel_invalid_channel_does_not_rollback():
    tenant_id = uuid.uuid4()
    db = AsyncMock()

    assert await claim_finalize_side_effect_channel(db, tenant_id, "unknown") is False
    db.rollback.assert_not_awaited()
    db.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_enqueue_side_effect_claims_only_after_successful_delay():
    from app.core.call_handler import _enqueue_finalize_side_effect_channel

    tenant_id = uuid.uuid4()
    db = AsyncMock()

    async def _enqueue() -> bool:
        return True

    with patch(
        "app.db.crud.is_finalize_side_effect_channel_claimed",
        AsyncMock(return_value=False),
    ):
        with patch(
            "app.db.crud.claim_finalize_side_effect_channel",
            AsyncMock(return_value=True),
        ) as claim:
            with patch(
                "app.db.crud.release_finalize_side_effect_channel",
                AsyncMock(),
            ) as release:
                await _enqueue_finalize_side_effect_channel(
                    db,
                    tenant_id=tenant_id,
                    channel="email",
                    redis_enqueue_lock=True,
                    enqueue=_enqueue,
                )

    claim.assert_awaited_once_with(db, tenant_id, "email")
    release.assert_not_awaited()


@pytest.mark.asyncio
async def test_enqueue_side_effect_does_not_claim_when_delay_fails():
    from app.core.call_handler import _enqueue_finalize_side_effect_channel

    tenant_id = uuid.uuid4()
    db = AsyncMock()

    async def _enqueue() -> bool:
        return False

    with patch(
        "app.db.crud.is_finalize_side_effect_channel_claimed",
        AsyncMock(return_value=False),
    ):
        with patch(
            "app.db.crud.claim_finalize_side_effect_channel",
            AsyncMock(),
        ) as claim:
            with patch(
                "app.db.crud.release_finalize_side_effect_channel",
                AsyncMock(),
            ) as release:
                await _enqueue_finalize_side_effect_channel(
                    db,
                    tenant_id=tenant_id,
                    channel="crm",
                    redis_enqueue_lock=True,
                    enqueue=_enqueue,
                )

    claim.assert_not_awaited()
    release.assert_not_awaited()


@pytest.mark.asyncio
async def test_enqueue_side_effect_releases_db_reserve_when_redis_down_and_delay_fails():
    from app.core.call_handler import _enqueue_finalize_side_effect_channel

    tenant_id = uuid.uuid4()
    db = AsyncMock()

    async def _enqueue() -> bool:
        return False

    with patch(
        "app.db.crud.is_finalize_side_effect_channel_claimed",
        AsyncMock(return_value=False),
    ):
        with patch(
            "app.db.crud.claim_finalize_side_effect_channel",
            AsyncMock(return_value=True),
        ) as claim:
            with patch(
                "app.db.crud.release_finalize_side_effect_channel",
                AsyncMock(),
            ) as release:
                await _enqueue_finalize_side_effect_channel(
                    db,
                    tenant_id=tenant_id,
                    channel="email",
                    redis_enqueue_lock=False,
                    enqueue=_enqueue,
                )

    claim.assert_awaited_once_with(db, tenant_id, "email")
    release.assert_awaited_once_with(db, tenant_id, "email")


@pytest.mark.asyncio
async def test_dispatch_finalize_side_effects_skips_when_enqueue_lock_held():
    from app.core.call_handler import _dispatch_finalize_side_effects

    session = MagicMock()
    session.call_id = "call-123"

    with patch(
        "app.core.call_handler._acquire_side_effects_enqueue_lock",
        AsyncMock(return_value=False),
    ):
        with patch(
            "app.core.call_handler._enqueue_finalize_side_effect_channel",
            AsyncMock(),
        ) as enqueue:
            await _dispatch_finalize_side_effects(
                AsyncMock(),
                call_uuid=uuid.uuid4(),
                tenant_id=uuid.uuid4(),
                session=session,
                persist_phone="+15551234567",
                merged={},
                score=80,
                status="qualified",
                reasons=[],
            )

    enqueue.assert_not_awaited()


@pytest.mark.asyncio
async def test_acquire_side_effects_enqueue_lock_fails_open_on_redis_error():
    from app.core.call_handler import _acquire_side_effects_enqueue_lock

    call_uuid = uuid.uuid4()
    with patch("app.core.redis_client.ping", AsyncMock(return_value=True)):
        with patch(
            "app.core.redis_client.acquire_once",
            AsyncMock(return_value=True),
        ) as acquire:
            assert await _acquire_side_effects_enqueue_lock(call_uuid) is True

    acquire.assert_awaited_once()
    assert acquire.await_args.kwargs["fail_closed"] is False


@pytest.mark.asyncio
async def test_dispatch_finalize_side_effects_proceeds_when_enqueue_lock_fails_open():
    from app.core.call_handler import _dispatch_finalize_side_effects

    session = MagicMock()
    session.call_id = "call-456"
    session.get_full_transcript.return_value = "AI: hello"
    session.stt_provider = "deepgram"
    session.llm_provider = "groq"
    session.tts_provider = "deepgram"
    session.duration_seconds = 12

    notif = MagicMock()
    notif.email_notifications_enabled = True
    notif.crm_webhook_url = ""

    with patch("app.core.redis_client.ping", AsyncMock(return_value=True)):
        with patch(
            "app.core.redis_client.acquire_once",
            AsyncMock(return_value=True),
        ):
            with patch(
                "app.core.call_handler._resolve_session_notification_settings",
                AsyncMock(return_value=notif),
            ):
                with patch(
                    "app.core.call_handler.notification_settings_email_dict",
                    return_value={},
                ):
                    with patch(
                        "app.core.call_handler._enqueue_finalize_side_effect_channel",
                        AsyncMock(),
                    ) as enqueue:
                        await _dispatch_finalize_side_effects(
                            AsyncMock(),
                            call_uuid=uuid.uuid4(),
                            tenant_id=uuid.uuid4(),
                            session=session,
                            persist_phone="+15551234567",
                            merged={},
                            score=80,
                            status="qualified",
                            reasons=[],
                        )

    assert enqueue.await_count >= 1
