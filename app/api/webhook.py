"""
app/api/webhook.py — Telnyx call webhook handlers and WebSocket audio streaming.

Handles:
- POST /telnyx/webhook — call.initiated, call.answered, call.hangup events
- WS /telnyx/stream/{call_id} — bidirectional real-time audio streaming

This is the core entry point for the voice AI pipeline.
"""

import asyncio
import json
import logging
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import call_handler
from app.core.ratelimit import limiter
from app.db.crud import (
    create_call,
    get_call_by_call_id,
    get_setting_value,
    is_number_blacklisted,
)
from app.db.database import get_db
from app.services.telnyx_service import telnyx_service, verify_telnyx_webhook_signature
from app.utils.helpers import sanitize_phone_number
from app.utils.security import mask_phone
from config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Reject webhooks whose signed timestamp is older/newer than this many seconds.
# Telnyx signs (timestamp | body); a valid signature on a stale timestamp means a
# captured request is being replayed, so we drop it even though the crypto checks out.
WEBHOOK_TIMESTAMP_TOLERANCE_S = 300
_RETRYABLE_WEBHOOK_EVENTS = {
    "call.initiated",
    "call.answered",
    "call.hangup",
    "call.recording.saved",
}
# Events whose handling is meaningless without a call_control_id.
_ID_REQUIRED_WEBHOOK_EVENTS = {
    "call.initiated",
    "call.answered",
    "call.hangup",
}


def _webhook_dedupe_key(event_type: str, call_payload: dict) -> str | None:
    """Return the Redis dedupe key used by the handler for this Telnyx event."""
    call_control_id = call_payload.get("call_control_id", "")
    if not call_control_id:
        return None
    if event_type == "call.initiated":
        return f"webhook:initiated:{call_control_id}"
    if event_type == "call.answered":
        return f"webhook:answered:{call_control_id}"
    if event_type == "call.hangup":
        return f"webhook:hangup:{call_control_id}"
    if event_type == "call.recording.saved":
        recording_id = call_payload.get("recording_id") or call_control_id
        return f"webhook:recording:{recording_id}"
    return None


async def _dedupe_webhook_event(
    call_control_id: str,
    lock_key: str,
    *,
    log_label: str,
) -> bool:
    """Return True when this delivery is the first for *lock_key* (process it)."""
    if not call_control_id:
        return True
    from app.core.redis_client import acquire_once

    # Keep lifecycle webhooks available even when Redis is down. If dedupe storage
    # is unavailable we prefer processing (idempotent handlers) over dropping
    # call.initiated/answered/hangup events entirely.
    if await acquire_once(
        lock_key,
        3600,
        fail_closed=False,
    ):
        return True
    logger.info("Duplicate %s for %s — ignoring", log_label, call_control_id)
    return False


# ──────────────────────────────────────────────────────────────────────────────
# Webhook signature verification dependency
# ──────────────────────────────────────────────────────────────────────────────


async def verify_webhook(request: Request) -> bytes:
    """
    Verify Telnyx webhook signature before processing.
    Returns raw body bytes if valid, raises 401 otherwise.
    """
    body = await request.body()

    if not settings.telnyx_public_key:
        if settings.is_production:
            # Never accept unsigned webhooks in production — without the public
            # key anyone could POST forged call events.
            logger.error("TELNYX_PUBLIC_KEY not set in production — rejecting webhook")
            raise HTTPException(
                status_code=503,
                detail="Webhook verification not configured",
            )
        if not settings.allow_unsigned_webhooks_in_dev:
            logger.error(
                "TELNYX_PUBLIC_KEY not set and unsigned webhooks disabled in development"
            )
            raise HTTPException(
                status_code=503,
                detail="Webhook verification not configured",
            )
        logger.warning(
            "TELNYX_PUBLIC_KEY not set — skipping signature verification (DEV ONLY)"
        )
        return body

    signature = request.headers.get("telnyx-signature-ed25519", "")
    timestamp = request.headers.get("telnyx-timestamp", "")

    if not signature or not timestamp:
        raise HTTPException(status_code=401, detail="Missing webhook signature headers")

    # Replay protection: reject signed payloads whose timestamp is outside the
    # tolerance window. Without this, a captured (validly-signed) webhook could be
    # replayed indefinitely to re-trigger call handling.
    try:
        ts_age = abs(datetime.now(UTC).timestamp() - float(timestamp))
    except (TypeError, ValueError):
        raise HTTPException(status_code=401, detail="Invalid webhook timestamp") from None
    if ts_age > WEBHOOK_TIMESTAMP_TOLERANCE_S:
        logger.warning("Rejecting stale webhook (age %.0fs)", ts_age)
        raise HTTPException(status_code=401, detail="Stale webhook timestamp")

    is_valid = verify_telnyx_webhook_signature(
        payload=body,
        signature=signature,
        timestamp=timestamp,
        public_key=settings.telnyx_public_key,
    )

    if not is_valid:
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    return body


# ──────────────────────────────────────────────────────────────────────────────
# Main Webhook Endpoint
# ──────────────────────────────────────────────────────────────────────────────


@router.post("/webhook")
@limiter.limit("120/minute")
async def telnyx_webhook(request: Request, db: AsyncSession = Depends(get_db)):
    """
    Main Telnyx Call Control webhook endpoint.
    Handles: call.initiated, call.answered, call.hangup, streaming events.
    """
    body = await verify_webhook(request)

    try:
        payload = json.loads(body)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail="Invalid JSON payload") from e

    event_data = payload.get("data", {})
    event_type = event_data.get("event_type", "")
    call_payload = event_data.get("payload", {})
    call_control_id = call_payload.get("call_control_id", "")

    logger.info(f"Telnyx webhook: {event_type} | call_control_id={call_control_id}")

    # Lifecycle events are keyed entirely by call_control_id (DB lookups, dedupe
    # locks, hangup handling). An empty id bypasses dedupe (see
    # _dedupe_webhook_event) and would let handlers act on an unidentifiable call,
    # so reject it up front rather than processing a malformed delivery.
    if event_type in _ID_REQUIRED_WEBHOOK_EVENTS and not call_control_id:
        logger.warning("Rejecting %s webhook with empty call_control_id", event_type)
        raise HTTPException(status_code=400, detail="Missing call_control_id")
    if (
        event_type == "call.recording.saved"
        and not call_control_id
        and not call_payload.get("recording_id")
    ):
        logger.warning("Rejecting call.recording.saved webhook with no call identifier")
        raise HTTPException(status_code=400, detail="Missing call identifier")

    try:
        if event_type == "call.initiated":
            await handle_call_initiated(db, call_payload)
        elif event_type == "call.answered":
            await handle_call_answered_event(db, call_payload)
        elif event_type == "call.hangup":
            await handle_call_hangup(db, call_payload)
        elif event_type == "call.recording.saved":
            await handle_recording_saved(db, call_payload)
        elif event_type == "streaming.started":
            logger.info("Streaming started for %s", call_control_id)
        elif event_type == "streaming.stopped":
            logger.info("Streaming stopped for %s", call_control_id)
        else:
            logger.debug("Unhandled event type: %s", event_type)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error handling %s: %s", event_type, e, exc_info=True)
        if event_type in _RETRYABLE_WEBHOOK_EVENTS:
            dedupe_key = _webhook_dedupe_key(event_type, call_payload)
            if dedupe_key:
                from app.core.redis_client import cache_delete

                await cache_delete(dedupe_key)
            raise HTTPException(
                status_code=500,
                detail="Webhook processing failed",
            ) from e

    return {"received": True}


# ──────────────────────────────────────────────────────────────────────────────
# Event Handlers
# ──────────────────────────────────────────────────────────────────────────────


async def handle_call_initiated(db: AsyncSession, call_payload: dict) -> None:
    """Handle call.initiated — create DB record, check blacklist, answer if OK."""
    call_control_id = call_payload.get("call_control_id", "")
    raw_phone = call_payload.get("from", "")
    direction = call_payload.get("direction", "incoming")
    phone_number = sanitize_phone_number(raw_phone)

    if not await _dedupe_webhook_event(
        call_control_id,
        f"webhook:initiated:{call_control_id}",
        log_label="call.initiated",
    ):
        return

    # Check blacklist (same rule as test console — admin DNC list is source of truth).
    if await is_number_blacklisted(db, phone_number):
        logger.warning("Rejecting blacklisted number: %s", mask_phone(phone_number))
        await telnyx_service.reject_call(call_control_id, cause="CALL_REJECTED")
        return

    # Create call record (recording flag frozen here — not re-read at answer).
    recording_enabled = await get_setting_value(db, "call_recording_enabled", False)
    try:
        await create_call(
            db,
            call_id=call_control_id,
            phone_number=phone_number,
            direction="inbound" if direction == "incoming" else "outbound",
            status="initiated",
            recording_requested=bool(recording_enabled),
        )
    except IntegrityError:
        await db.rollback()
        existing = await get_call_by_call_id(db, call_control_id)
        if existing is not None:
            logger.info(
                "Duplicate call.initiated for %s — treating as idempotent",
                call_control_id,
            )
            return
        raise

    # Answer the call
    if direction == "incoming":
        try:
            await telnyx_service.answer_call(call_control_id)
            logger.info("Answered call from %s", mask_phone(phone_number))
        except Exception:
            # If answering fails the media stream never connects, so nothing
            # would ever finalize this row — mark it failed instead of leaving
            # it stuck in "initiated" forever.
            from app.db.crud import update_call

            await update_call(
                db,
                call_control_id,
                status="failed",
                ended_at=datetime.now(UTC),
            )
            raise


async def handle_call_answered_event(db: AsyncSession, call_payload: dict) -> None:
    """Handle call.answered — start recording and audio streaming.

    The ConversationSession is created lazily by the WebSocket handler when
    Telnyx connects the media stream (see ``telnyx_audio_stream``). Creating it
    there — rather than here — keeps the live session on the same worker that
    owns the WebSocket, so the agent works correctly behind multiple workers.
    """
    call_control_id = call_payload.get("call_control_id", "")

    if not await _dedupe_webhook_event(
        call_control_id,
        f"webhook:answered:{call_control_id}",
        log_label="call.answered",
    ):
        return

    from app.db.crud import get_call_by_call_id, update_call

    call = await get_call_by_call_id(db, call_control_id)
    # Use the flag frozen at call.initiated; fall back to live setting only if
    # the row is missing (webhook ordering race).
    if call is not None:
        recording_enabled = bool(call.recording_requested)
    else:
        recording_enabled = await get_setting_value(db, "call_recording_enabled", False)
    if recording_enabled:
        try:
            await telnyx_service.start_recording(call_control_id)
        except Exception as e:
            logger.warning("Failed to start recording: %s", e)

    # Start bidirectional audio streaming to our WebSocket. The URL carries a
    # short-lived signed token so the media WebSocket can't be driven by anyone
    # who merely knows/guesses the call_control_id.
    from app.utils.helpers import generate_stream_token

    base = settings.app_url.rstrip("/")
    ws_base = base.replace("https://", "wss://").replace("http://", "ws://")
    stream_token = generate_stream_token(call_control_id, settings.secret_key)
    stream_url = f"{ws_base}/telnyx/stream/{call_control_id}?token={stream_token}"

    # Never log the full stream URL — it carries a short-lived bearer token that
    # would let anyone with log access hijack the live call's audio.
    try:
        await telnyx_service.start_streaming(call_control_id, stream_url)
        logger.info("Streaming started for call %s", call_control_id)
        await update_call(db, call_control_id, status="in_progress")
    except Exception as e:
        # Duplicate call.answered deliveries can happen (especially while dedupe
        # is fail-open), and Telnyx may reject a second start_streaming for a
        # call that's already active. Treat that as idempotent success rather
        # than failing/hanging up a healthy in-progress call.
        latest = await get_call_by_call_id(db, call_control_id)
        if latest is not None and latest.status == "in_progress":
            logger.info(
                "Streaming already active for %s; treating duplicate answered as idempotent",
                call_control_id,
            )
            return
        # If streaming really can't start, the caller would otherwise sit in
        # silence on an active call. Mark it failed and hang up cleanly.
        logger.error("Failed to start streaming for call %s: %s", call_control_id, e)
        await update_call(db, call_control_id, status="failed")
        try:
            await telnyx_service.hangup_call(call_control_id)
        except Exception as hangup_err:
            logger.warning(
                "Failed to hang up call %s after streaming error: %s",
                call_control_id,
                hangup_err,
            )


async def handle_call_hangup(db: AsyncSession, call_payload: dict) -> None:
    """Handle call.hangup — stop the audio stream and finalize after it ends."""
    call_control_id = call_payload.get("call_control_id", "")

    if not await _dedupe_webhook_event(
        call_control_id,
        f"webhook:hangup:{call_control_id}",
        log_label="call.hangup",
    ):
        return

    session = call_handler.get_session(call_control_id)
    if session:
        # Mark the session so a stream that is still connecting stops as soon as it
        # registers (closes the early-hangup race), then stop any live stream.
        session.pending_hangup = True

        # Stop the live WebSocket loop first so the worker can finish (or abort)
        # its current turn, then finalize from on_complete with a fuller transcript.
        # If the stream hasn't registered yet, finalize_after_stream_timeout still
        # waits a grace period — by then the stream has either started (and stopped
        # itself via pending_hangup, finalizing via on_complete) or never connected
        # (the timeout finalizes it). Either way we avoid finalizing mid-startup.
        await call_handler.request_stream_stop(call_control_id)
        asyncio.create_task(
            call_handler.finalize_after_stream_timeout(call_control_id)
        )
        return
    else:
        # Session may live on another worker — signal stream stop via Redis and
        # schedule a safety-net finalize if nothing completes the call.
        await call_handler.request_stream_stop(call_control_id)
        asyncio.create_task(
            call_handler.finalize_after_stream_timeout(call_control_id)
        )
        from app.db.crud import get_call_by_call_id, mark_call_abandoned_if_active

        existing = await get_call_by_call_id(db, call_control_id)
        if existing and existing.status == "in_progress":
            logger.info(
                "Hangup for in_progress call %s — stream stop signaled "
                "(session on another worker; Redis + DB fallback); finalize timeout scheduled",
                call_control_id,
            )
            return
        if existing and existing.status not in ("completed", "failed", "abandoned"):
            marked = await mark_call_abandoned_if_active(db, call_control_id)
            if marked:
                logger.info("Call marked abandoned: %s", call_control_id)


async def handle_recording_saved(db: AsyncSession, call_payload: dict) -> None:
    """Download Telnyx recording and store in Supabase when configured."""
    call_control_id = call_payload.get("call_control_id", "")
    recording_id = call_payload.get("recording_id") or call_control_id
    if not await _dedupe_webhook_event(
        recording_id,
        f"webhook:recording:{recording_id}",
        log_label="call.recording.saved",
    ):
        return

    recording_urls = call_payload.get("recording_urls") or {}
    mp3_url = recording_urls.get("mp3") or recording_urls.get("wav")

    if not mp3_url:
        logger.warning("No recording URL for %s", call_control_id)
        return

    from app.db.crud import update_call
    from app.services.storage_service import storage_service
    from app.utils.security import UnsafeURLError, assert_safe_external_url

    # The recording URL comes from the webhook payload — validate it points at a
    # public host before fetching so a forged/redirected URL can't be used to
    # probe internal services (SSRF).
    try:
        assert_safe_external_url(mp3_url, require_https=True)
    except UnsafeURLError as e:
        logger.error("Refusing unsafe recording URL for %s: %s", call_control_id, e)
        return

    try:
        import httpx

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.get(mp3_url)
            resp.raise_for_status()
            audio_bytes = resp.content

        # Stores the private Supabase object path when configured (served later
        # via a signed URL), otherwise falls back to the raw Telnyx URL.
        object_path = await storage_service.upload_recording(
            call_control_id, audio_bytes
        )
        await update_call(
            db,
            call_control_id,
            recording_url=object_path or mp3_url,
        )
        logger.info("Recording saved for %s", call_control_id)
    except Exception as e:
        logger.error("Recording processing failed for %s: %s", call_control_id, e)
        try:
            await update_call(db, call_control_id, recording_url=mp3_url)
        except (SQLAlchemyError, IntegrityError) as db_err:
            logger.error("Failed to update recording URL in DB: %s", db_err)


# ──────────────────────────────────────────────────────────────────────────────
# WebSocket Audio Stream Handler
# ──────────────────────────────────────────────────────────────────────────────


@router.websocket("/stream/{call_id}")
async def telnyx_audio_stream(websocket: WebSocket, call_id: str):
    """
    WebSocket endpoint that Telnyx connects to for bidirectional audio.
    Delegates to the shared production audio stream handler.

    Requires the signed token issued in the stream URL (see
    ``handle_call_answered_event``). Connections without a valid token are
    rejected so the call audio can't be hijacked.
    """
    from app.utils.helpers import verify_stream_token

    token = websocket.query_params.get("token", "")
    if not verify_stream_token(call_id, token, settings.secret_key):
        logger.warning(
            "Rejecting media WebSocket for %s — missing/invalid stream token", call_id
        )
        await websocket.close(code=1008)
        return

    await websocket.accept()
    logger.info("WebSocket connected for call: %s", call_id)

    # Session is created lazily on the WebSocket worker; polls the DB briefly if
    # call.initiated has not committed the row yet (multi-worker / ordering race).
    session = await call_handler.ensure_stream_session(call_id)
    if session is None:
        logger.error("No session for call %s — closing WS", call_id)
        await websocket.close(code=1008)
        return

    async def _finalize_on_stream_end() -> None:
        """Finalize on the worker that owns the live session (multi-worker safe)."""
        await call_handler.finalize_active_session_background(call_id)

    from app.core.audio_stream import run_bidirectional_audio_stream

    await run_bidirectional_audio_stream(
        websocket,
        call_id,
        session,
        hangup_on_complete=True,
        emit_debug_events=False,
        on_complete=_finalize_on_stream_end,
    )
