"""
app/services/storage_service.py — Supabase Storage integration for call recordings.

Optional feature. Uploads recordings from Telnyx to Supabase Storage bucket
and returns public/signed URLs for playback in the admin dashboard.
"""

import logging

import httpx

from config import settings

logger = logging.getLogger(__name__)


class StorageService:
    """Handles uploading and retrieving call recordings via Supabase Storage."""

    def __init__(self) -> None:
        self.bucket = settings.supabase_storage_bucket
        self.base_url = self._derive_supabase_url()
        self.secret_key = settings.supabase_secret_key

    def _derive_supabase_url(self) -> str | None:
        """Return the configured Supabase project URL."""
        if settings.supabase_url:
            return settings.supabase_url.rstrip("/")
        return None

    def _auth_headers(self, content_type: str | None = None) -> dict[str, str]:
        """Build Supabase Storage headers for privileged server-side writes."""
        headers = {
            "apikey": self.secret_key,
            "Authorization": f"Bearer {self.secret_key}",
        }
        if content_type:
            headers["Content-Type"] = content_type
        return headers

    async def upload_recording(
        self,
        call_id: str,
        audio_bytes: bytes,
        content_type: str = "audio/mpeg",
    ) -> str | None:
        """
        Upload a call recording to Supabase Storage.

        The object is stored privately — we return the object *path* (e.g.
        ``recordings/<call_id>.mp3``) rather than a public URL, so access is only
        granted through short-lived signed URLs (see ``create_signed_url``). This
        avoids exposing recordings via a guessable, permanently-public link.

        Returns:
            The storage object path, or None on failure.
        """
        if not self.base_url or not self.secret_key:
            logger.warning("Supabase Storage not configured — recording not uploaded")
            return None

        object_path = f"{self.bucket}/{call_id}.mp3"
        url = f"{self.base_url}/storage/v1/object/{object_path}"

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    url,
                    content=audio_bytes,
                    headers={
                        **self._auth_headers(content_type),
                        "x-upsert": "true",
                    },
                )
                response.raise_for_status()
                logger.info(f"Recording uploaded for call {call_id}")
                return object_path
        except Exception as e:
            logger.error(f"Recording upload failed for call {call_id}: {e}")
            return None

    async def create_signed_url(
        self,
        object_path: str,
        expires_in: int = 3600,
    ) -> str | None:
        """Return a short-lived signed URL for a stored object path.

        ``object_path`` is the ``{bucket}/{name}`` value returned by
        ``upload_recording``. Returns None if storage isn't configured or signing
        fails.
        """
        if not self.base_url or not self.secret_key:
            return None

        sign_url = f"{self.base_url}/storage/v1/object/sign/{object_path}"

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    sign_url,
                    json={"expiresIn": expires_in},
                    headers=self._auth_headers("application/json"),
                )
                response.raise_for_status()
                signed_path = response.json().get("signedURL")
                if not signed_path:
                    return None
                return f"{self.base_url}/storage/v1{signed_path}"
        except Exception as e:
            logger.error(f"Failed to sign recording URL for {object_path}: {e}")
            return None


    async def delete_recording(self, object_path: str) -> bool:
        """Delete a stored recording object by its ``{bucket}/{name}`` path.

        Returns True on successful delete. Returns False when storage is not
        configured, the path is empty, the path is an external URL, or the
        delete request fails.
        """
        if not self.base_url or not self.secret_key or not object_path:
            return False
        if object_path.startswith(("http://", "https://")):
            # Legacy public URLs aren't deletable by path; skip silently.
            return False
        url = f"{self.base_url}/storage/v1/object/{object_path}"
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.delete(url, headers=self._auth_headers())
                response.raise_for_status()
                logger.info(f"Recording deleted: {object_path}")
                return True
        except Exception as e:
            logger.error(f"Failed to delete recording {object_path}: {e}")
            return False


storage_service = StorageService()
