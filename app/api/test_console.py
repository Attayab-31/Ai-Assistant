"""
app/api/test_console.py — Browser-based test interface for the voice agent.

Lets a developer exercise the full STT → LLM → TTS pipeline without
provisioning a real Telnyx phone number. Three flows are supported:

1. Text chat   (POST /test/api/start, POST /test/api/say)
   - Skips STT entirely. Pipes text straight to the LLM, returns TTS audio.
2. Audio upload (POST /test/api/say-audio)
   - Browser records a short WAV blob, server converts to mulaw 8kHz and
     runs the real STT provider before continuing the same flow.
3. Live WebSocket (WS /test/api/stream/{call_id})
   - Browser streams mic audio as Telnyx-compatible `media` events (linear16).
   - Deepgram live STT on the server detects end-of-turn; barge-in is transcript-gated.
   - Downlink uses `play_wav` for clear browser playback; production Telnyx uses mulaw.

All flows reuse the production ConversationSession + call_handler, so any
fix here is automatically a fix for real Telnyx calls.
"""

from __future__ import annotations

import base64
import logging
import uuid
from pathlib import Path

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
)
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import call_handler
from app.core.conversation import CallState
from app.db.crud import get_user_by_id
from app.db.database import AsyncSessionLocal, get_db
from app.utils.audio import (
    any_audio_to_mulaw,
    mulaw_to_wav,
)
from app.utils.dependencies import ACCESS_TOKEN_COOKIE_NAME, get_current_user
from app.utils.helpers import friendly_provider_name
from config import settings

logger = logging.getLogger(__name__)
router = APIRouter()
__test__ = False

TEMPLATES_DIR = Path(__file__).parent.parent / "admin" / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# ──────────────────────────────────────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────────────────────────────────────


class StartCallRequest(BaseModel):
    phone_number: str = "+15555550123"
    property_name: str | None = None
    # "text" | "voice" — voice skips greeting (WS delivers it)
    mode: str = "text"


class SayRequest(BaseModel):
    call_id: str
    text: str


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _audio_payload(audio_mulaw) -> dict:
    """Wrap raw mulaw bytes (or list of segments) as browser-playable WAV."""
    from app.core.call_handler import join_audio_parts

    audio_mulaw = join_audio_parts(audio_mulaw)
    if not audio_mulaw:
        return {"audio_wav_b64": "", "audio_mulaw_b64": "", "duration_ms": 0}
    wav = mulaw_to_wav(audio_mulaw)
    duration_ms = int(len(audio_mulaw) / 8000 * 1000)
    return {
        "audio_wav_b64": base64.b64encode(wav).decode("ascii"),
        "audio_mulaw_b64": base64.b64encode(audio_mulaw).decode("ascii"),
        "duration_ms": duration_ms,
    }


def _session_snapshot(session) -> dict:
    return {
        "call_id": session.call_id,
        "phone_number": session.phone_number,
        "state": session.current_state,
        "questions_answered": session.questions_answered,
        "active_question_count": session.active_question_count(),
        "is_screening_complete": session.is_screening_complete(),
        "duration_seconds": session.duration_seconds,
        "extracted_data": {
            k: (str(v) if v is not None else None)
            for k, v in session.extracted_data.items()
        },
        "errors": session.errors,
        "providers": {
            "stt": session.stt_provider,
            "llm": session.llm_provider,
            "tts": session.tts_provider,
        },
        "settings_captured_at": session.settings_captured_at,
        "auto_fallback_enabled": session.auto_fallback_enabled,
    }


_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost", "::ffff:127.0.0.1"}


def _is_loopback(client_host: str | None) -> bool:
    """True when the request originates from the local machine."""
    return bool(client_host) and client_host in _LOOPBACK_HOSTS


def _dev_loopback_exempt(client_host: str | None) -> bool:
    """Allow unauthenticated local QA on loopback in non-production."""
    return not settings.is_production and _is_loopback(client_host)


async def require_test_console_access(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> None:
    """Require Settings edit access to use the test console.

    Always enforced in production. In non-production it is also enforced for
    any non-loopback client, so a dev instance exposed on a network can't be
    driven anonymously; only same-machine localhost QA is allowed without login.
    """
    client_host = request.client.host if request.client else None
    if _dev_loopback_exempt(client_host):
        return
    user = await get_current_user(request, db)
    if not user.can("settings"):
        raise HTTPException(
            status_code=403,
            detail="Test console requires Settings access.",
        )
    if not user.can_edit:
        raise HTTPException(
            status_code=403,
            detail="Your account is read-only.",
        )


async def _verify_ws_auth(websocket: WebSocket) -> bool:
    """Verify admin cookie and Settings edit access on WebSocket connect.

    Always enforced in production. In non-production, loopback connections are
    allowed without auth for local QA; non-loopback connections must present a
    valid admin token with Settings edit permission.
    """
    client_host = websocket.client.host if websocket.client else None
    if _dev_loopback_exempt(client_host):
        return True

    from app.utils.security import decode_access_token

    token = websocket.cookies.get(ACCESS_TOKEN_COOKIE_NAME)
    if not token:
        return False
    payload = decode_access_token(token)
    if not payload or not payload.get("sub"):
        return False

    try:
        user_id = uuid.UUID(str(payload["sub"]))
    except ValueError:
        return False

    async with AsyncSessionLocal() as db:
        user = await get_user_by_id(db, user_id)
        if not user or not user.is_active:
            return False
        return user.can("settings") and user.can_edit


# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────


@router.get("", response_class=HTMLResponse, include_in_schema=False)
@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def test_console_page(
    request: Request,
    _auth: None = Depends(require_test_console_access),
):
    """Render the test interface."""
    from config import provider_registry

    raw = provider_registry.get_status()
    provider_labels = {
        "llm": friendly_provider_name(raw.get("llm"), "llm"),
        "stt": friendly_provider_name(raw.get("stt"), "stt"),
        "tts": friendly_provider_name(raw.get("tts"), "tts"),
    }

    return templates.TemplateResponse(
        "test_console.html",
        {
            "request": request,
            "providers": raw,
            "provider_labels": provider_labels,
            "app_url": settings.app_url,
        },
    )


@router.get("/call", response_class=HTMLResponse, include_in_schema=False)
async def client_call_page(
    request: Request,
    _auth: None = Depends(require_test_console_access),
):
    """Render the minimal, client-friendly single-button call interface.

    Uses the exact same backend as the full console (start → WebSocket audio
    stream → end); only the UI is stripped down to one round call button.
    """
    return templates.TemplateResponse(
        "client_call.html",
        {"request": request, "app_url": settings.app_url},
    )


# ──────────────────────────────────────────────────────────────────────────────
# REST: start / say / end / inspect
# ──────────────────────────────────────────────────────────────────────────────


@router.post("/api/start")
async def start_test_call(
    request: Request,
    payload: StartCallRequest,
    db: AsyncSession = Depends(get_db),
    _auth: None = Depends(require_test_console_access),
):
    """
    Create a new in-memory test session and return the greeting (text + audio).
    No DB call record is created — this is a sandbox.
    """
    call_id = f"test-{uuid.uuid4().hex[:12]}"

    session = await call_handler.create_session(
        call_id=call_id,
        phone_number=payload.phone_number,
        db=db,
        property_name=payload.property_name,
    )

    ws_path = f"/test/api/stream/{call_id}"
    ws_scheme = "wss" if request.url.scheme == "https" else "ws"
    ws_url = f"{ws_scheme}://{request.url.netloc}{ws_path}"

    # Voice mode: session only — greeting arrives over the WebSocket (same as Telnyx).
    if payload.mode == "voice":
        return {
            "call_id": call_id,
            "mode": "voice",
            "ws_url": ws_url,
            "greeting_text": "",
            "audio_wav_b64": "",
            "audio_mulaw_b64": "",
            "duration_ms": 0,
            "session": _session_snapshot(session),
        }

    try:
        greeting_audio = await call_handler.handle_call_answered(session)
    except Exception as e:
        logger.error("Test greeting failed: %s", e, exc_info=True)
        greeting_audio = b""

    greeting_text = next(
        (t.text for t in reversed(session.transcript) if t.speaker == "AI"), ""
    )

    return {
        "call_id": call_id,
        "mode": "text",
        "ws_url": ws_url,
        "greeting_text": greeting_text,
        **_audio_payload(greeting_audio),
        "session": _session_snapshot(session),
    }


@router.post("/api/say")
async def say(
    payload: SayRequest,
    _auth: None = Depends(require_test_console_access),
):
    """
    Send text as if it were a transcribed tenant utterance. Returns the AI's
    text reply, TTS audio, and updated session state.
    """
    session = call_handler.get_session(payload.call_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    (
        response_text,
        response_audio,
        is_complete,
    ) = await call_handler.process_tenant_speech(session, payload.text)

    logger.debug(
        "Test console /say response: text=%r, audio_len=%s",
        response_text,
        len(response_audio) if response_audio else 0,
    )

    return {
        "call_id": payload.call_id,
        "response_text": response_text if response_text else "",
        "is_complete": is_complete,
        **_audio_payload(response_audio),
        "session": _session_snapshot(session),
    }


@router.post("/api/say-audio")
async def say_audio(
    call_id: str = Form(...),
    audio: UploadFile = File(...),
    _auth: None = Depends(require_test_console_access),
):
    """
    Upload a WAV blob (e.g. from MediaRecorder), transcribe it with the
    active STT provider, then run the same flow as /api/say.
    """
    session = call_handler.get_session(call_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    raw = await audio.read()
    try:
        mulaw = any_audio_to_mulaw(raw)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unsupported audio: {e}") from e

    from app.core.audio_stream import transcribe_buffer

    transcript = await transcribe_buffer(mulaw, session=session)

    (
        response_text,
        response_audio,
        is_complete,
    ) = await call_handler.process_tenant_speech(session, transcript)

    return {
        "call_id": call_id,
        "transcript": transcript,
        "response_text": response_text,
        "is_complete": is_complete,
        **_audio_payload(response_audio),
        "session": _session_snapshot(session),
    }


class EndCallRequest(BaseModel):
    call_id: str


@router.post("/api/end")
async def end_test_call(
    payload: EndCallRequest,
    _auth: None = Depends(require_test_console_access),
):
    """Finalize the test call: run extraction + scoring, then drop the session."""
    cid = payload.call_id
    session = call_handler.get_session(cid)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Force state to ENDED for clean scoring
    if session.current_state not in (CallState.WRAP_UP.value, CallState.ENDED.value):
        session.current_state = CallState.ENDED.value

    # Build a fake call row so finalize_call can write results to DB
    summary = {
        "call_id": cid,
        "snapshot": _session_snapshot(session),
        "transcript": session.get_full_transcript(),
    }

    try:
        async with AsyncSessionLocal() as bg_db:
            from app.db.crud import create_call, get_call_by_call_id

            if not await get_call_by_call_id(bg_db, cid):
                await create_call(
                    bg_db,
                    call_id=cid,
                    phone_number=session.phone_number,
                    direction="test",
                    status="in_progress",
                    commit=False,
                )
            result = await call_handler.finalize_call(session, bg_db)
        summary.update(result)
        if result.get("db_persisted") is False:
            summary["finalize_warning"] = (
                "Score computed but tenant record was not saved — check server logs."
            )
    except Exception as e:
        logger.error("Test finalize failed: %s", e, exc_info=True)
        summary["finalize_error"] = str(e)
    finally:
        call_handler.remove_session(cid)

    return summary


@router.get("/api/session/{call_id}")
async def get_session_state(
    call_id: str,
    _auth: None = Depends(require_test_console_access),
):
    """Inspect the current state of a test session."""
    session = call_handler.get_session(call_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session": _session_snapshot(session),
        "transcript": [
            {
                "speaker": t.speaker,
                "text": t.text,
                "state": t.state,
                "timestamp": t.timestamp,
            }
            for t in session.transcript
        ],
    }


@router.get("/api/sessions")
async def list_test_sessions(
    _auth: None = Depends(require_test_console_access),
):
    """List all in-memory test sessions."""
    return {"sessions": call_handler.get_active_sessions()}


# ──────────────────────────────────────────────────────────────────────────────
# Live WebSocket — mirrors Telnyx Media Streaming protocol
# ──────────────────────────────────────────────────────────────────────────────


@router.websocket("/api/stream/{call_id}")
async def test_stream(websocket: WebSocket, call_id: str):
    """
    Telnyx-compatible WebSocket: same protocol and handler as production.
    Emits debug events (transcript, response, complete) for the test UI.
    """
    await websocket.accept()
    if not await _verify_ws_auth(websocket):
        await websocket.send_json(
            {"event": "error", "message": "authentication required"}
        )
        await websocket.close(code=1008)
        return

    session = await call_handler.wait_for_session(call_id, timeout=5.0)
    if not session:
        await websocket.send_json({"event": "error", "message": "session not found"})
        await websocket.close(code=1008)
        return

    from app.core.audio_stream import run_bidirectional_audio_stream

    await run_bidirectional_audio_stream(
        websocket,
        call_id,
        session,
        hangup_on_complete=False,
        emit_debug_events=True,
    )
