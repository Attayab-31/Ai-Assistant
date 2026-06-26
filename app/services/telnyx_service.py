"""
app/services/telnyx_service.py — Telnyx telephony integration.

Handles:
- Call Control API actions (answer, hangup, speak, transfer)
- Webhook signature verification
- Outbound SMS (for follow-ups)
- Phone number management
"""

import base64
import logging

import httpx
from nacl.exceptions import BadSignatureError
from nacl.signing import VerifyKey

from config import settings

logger = logging.getLogger(__name__)

TELNYX_API_BASE = "https://api.telnyx.com/v2"


class TelnyxService:
    """Wraps Telnyx Call Control API for call management."""

    def __init__(self) -> None:
        self.api_key = settings.telnyx_api_key
        self._client: httpx.AsyncClient | None = None

    @property
    def headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def _client_or_new(self) -> httpx.AsyncClient:
        """Reuse a single HTTP client for lower latency on repeated API calls."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=10.0,
                headers=self.headers,
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def answer_call(self, call_control_id: str) -> dict:
        """Answer an inbound call."""
        url = f"{TELNYX_API_BASE}/calls/{call_control_id}/actions/answer"
        client = await self._client_or_new()
        response = await client.post(url, json={})
        response.raise_for_status()
        return response.json()

    async def hangup_call(self, call_control_id: str) -> dict:
        """Hang up an active call."""
        url = f"{TELNYX_API_BASE}/calls/{call_control_id}/actions/hangup"
        client = await self._client_or_new()
        response = await client.post(url, json={})
        response.raise_for_status()
        return response.json()

    async def start_streaming(
        self,
        call_control_id: str,
        stream_url: str,
        track: str = "inbound_track",
    ) -> dict:
        """Start bidirectional audio streaming to our WebSocket endpoint.

        - ``inbound_track`` so we only receive the caller's audio for STT
          (``both_tracks`` would feed the agent's own TTS back into STT,
          causing echo loops and false barge-in).
        - ``stream_bidirectional_mode="rtp"`` + ``PCMU`` codec so the audio we
          send back over the same WebSocket (μ-law 8 kHz frames) is played to
          the caller. Without this, Telnyx only forks audio one-way and the
          caller never hears the agent.
        """
        url = f"{TELNYX_API_BASE}/calls/{call_control_id}/actions/streaming_start"
        payload = {
            "stream_url": stream_url,
            "stream_track": track,
            "stream_bidirectional_mode": "rtp",
            "stream_bidirectional_codec": "PCMU",
        }
        client = await self._client_or_new()
        response = await client.post(url, json=payload)
        response.raise_for_status()
        return response.json()

    async def start_recording(self, call_control_id: str) -> dict:
        """Start call recording."""
        url = f"{TELNYX_API_BASE}/calls/{call_control_id}/actions/record_start"
        payload = {"format": "mp3", "channels": "dual"}
        client = await self._client_or_new()
        response = await client.post(url, json=payload)
        response.raise_for_status()
        return response.json()

    async def reject_call(
        self, call_control_id: str, cause: str = "CALL_REJECTED"
    ) -> dict:
        """Reject a call (e.g., for blacklisted numbers)."""
        url = f"{TELNYX_API_BASE}/calls/{call_control_id}/actions/reject"
        payload = {"cause": cause}
        client = await self._client_or_new()
        response = await client.post(url, json=payload)
        response.raise_for_status()
        return response.json()


def verify_telnyx_webhook_signature(
    payload: bytes,
    signature: str,
    timestamp: str,
    public_key: str,
) -> bool:
    """
    Verify Telnyx webhook signature using Ed25519 public key.
    Telnyx signs webhooks with: signature header + timestamp + payload.
    """
    try:
        signed_payload = f"{timestamp}|{payload.decode('utf-8')}".encode()
        verify_key = VerifyKey(base64.b64decode(public_key))
        verify_key.verify(signed_payload, base64.b64decode(signature))
        return True
    except (BadSignatureError, Exception) as e:
        logger.warning(f"Telnyx webhook signature verification failed: {e}")
        return False


telnyx_service = TelnyxService()
