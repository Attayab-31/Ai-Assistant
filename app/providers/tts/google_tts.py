"""
app/providers/tts/google_tts.py — Google Cloud Text-to-Speech (WaveNet/Neural2).

Primary TTS provider. 1M chars/month free on Google Cloud.
Returns mulaw 8kHz audio for Telnyx phone compatibility.
"""

import logging

from app.providers.base import BaseTTSProvider
from config import settings

logger = logging.getLogger(__name__)

AVAILABLE_VOICES = {
    "en-US-Wavenet-D": {"gender": "MALE", "name": "en-US-Wavenet-D"},
    "en-US-Wavenet-F": {"gender": "FEMALE", "name": "en-US-Wavenet-F"},
    "en-US-Neural2-A": {"gender": "MALE", "name": "en-US-Neural2-A"},
    "en-US-Neural2-C": {"gender": "FEMALE", "name": "en-US-Neural2-C"},
    "en-US-Neural2-D": {"gender": "MALE", "name": "en-US-Neural2-D"},
    "en-US-Neural2-F": {"gender": "FEMALE", "name": "en-US-Neural2-F"},
    "en-US-Neural2-H": {"gender": "FEMALE", "name": "en-US-Neural2-H"},
    "en-US-Neural2-J": {"gender": "MALE", "name": "en-US-Neural2-J"},
}


class GoogleTTSProvider(BaseTTSProvider):
    """
    Google Cloud WaveNet/Neural2 TTS provider.
    Produces high-quality, natural-sounding speech for phone calls.
    """

    provider_name = "google"

    def __init__(
        self,
        voice: str = "en-US-Wavenet-D",
        language_code: str = "en-US",
    ) -> None:
        self.voice = voice if voice in AVAILABLE_VOICES else "en-US-Wavenet-D"
        self.language_code = language_code
        self._client = None
        logger.info("GoogleTTSProvider initialized: voice=%s", self.voice)

    @property
    def client(self):
        """Lazy-initialize Google TTS client."""
        if self._client is None:
            try:
                from google.cloud import texttospeech

                if settings.google_application_credentials:
                    import os

                    os.environ[
                        "GOOGLE_APPLICATION_CREDENTIALS"
                    ] = settings.google_application_credentials

                self._client = texttospeech.TextToSpeechAsyncClient()
            except ImportError as e:
                raise ImportError("google-cloud-texttospeech not installed") from e
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize Google TTS client: {e}"
                ) from e
        return self._client

    async def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
    ) -> bytes:
        """
        Convert text to mulaw 8kHz audio for Telnyx phone calls.

        Args:
            text: Text to synthesize (max 5000 chars per call)
            voice: Voice name override
            speed: Speaking rate (0.75-1.25)

        Returns:
            Raw mulaw 8kHz audio bytes
        """
        from google.cloud import texttospeech

        active_voice = voice or self.voice
        voice_config = AVAILABLE_VOICES.get(
            active_voice, AVAILABLE_VOICES["en-US-Wavenet-D"]
        )

        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice_params = texttospeech.VoiceSelectionParams(
            language_code=self.language_code,
            name=voice_config["name"],
            ssml_gender=getattr(
                texttospeech.SsmlVoiceGender,
                voice_config["gender"],
                texttospeech.SsmlVoiceGender.NEUTRAL,
            ),
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MULAW,
            sample_rate_hertz=8000,
            speaking_rate=max(0.75, min(1.25, speed)),
        )

        try:
            response = await self.client.synthesize_speech(
                input=synthesis_input,
                voice=voice_params,
                audio_config=audio_config,
            )
            logger.debug(
                "Google TTS synthesized %d chars → %d bytes",
                len(text),
                len(response.audio_content),
            )
            return response.audio_content
        except Exception as e:
            logger.error("Google TTS synthesis error: %s", e)
            raise
