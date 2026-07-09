"""Regressions for provider credential isolation and per-credential caching.

Covers two audit fixes:
- Google TTS must not mutate the process-global GOOGLE_APPLICATION_CREDENTIALS.
- Vendor balance lookups must be cached per credential, not per provider name.
"""

import os
from types import SimpleNamespace

import pytest


class _FakeAsyncClient:
    """Minimal stand-in for TextToSpeechAsyncClient."""

    def __init__(self, *, credentials=None):
        self.credentials = credentials
        self.from_file_path = None

    @classmethod
    def from_service_account_file(cls, path):
        obj = cls()
        obj.from_file_path = path
        return obj


class _FakeTexttospeech(SimpleNamespace):
    pass


def _make_module(monkeypatch):
    return _FakeTexttospeech(TextToSpeechAsyncClient=_FakeAsyncClient)


def test_google_tts_does_not_mutate_environment(monkeypatch):
    from app.providers.tts.google_tts import GoogleTTSProvider

    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    ts = _make_module(monkeypatch)

    client = GoogleTTSProvider._build_client(ts, "/secrets/sa.json")

    assert client.from_file_path == "/secrets/sa.json"
    assert "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ


def test_google_tts_uses_adc_when_no_credential(monkeypatch):
    from app.providers.tts.google_tts import GoogleTTSProvider

    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    ts = _make_module(monkeypatch)

    client = GoogleTTSProvider._build_client(ts, "")

    assert isinstance(client, _FakeAsyncClient)
    assert client.from_file_path is None
    assert client.credentials is None
    assert "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ


def test_google_tts_accepts_inline_json_credential(monkeypatch):
    from app.providers.tts.google_tts import GoogleTTSProvider

    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    ts = _make_module(monkeypatch)
    sentinel = object()
    fake_sa = SimpleNamespace(
        Credentials=SimpleNamespace(
            from_service_account_info=lambda info: sentinel
        )
    )
    # google.oauth2.service_account is imported lazily inside _build_client.
    import sys

    monkeypatch.setitem(
        sys.modules,
        "google.oauth2.service_account",
        fake_sa,
    )
    monkeypatch.setitem(sys.modules, "google.oauth2", SimpleNamespace(service_account=fake_sa))

    client = GoogleTTSProvider._build_client(ts, '{"type": "service_account"}')

    assert client.credentials is sentinel
    assert "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ


def test_balance_cache_keyed_by_credential():
    from app.services import provider_usage

    provider_usage._cache.clear()
    key_a = "sk-aaaaaaaaaaaaaaaa"
    key_b = "sk-bbbbbbbbbbbbbbbb"

    provider_usage._cache_set(
        f"openrouter_balance:{provider_usage._key_fingerprint(key_a)}",
        {"available": True, "remaining_label": "$10.00"},
    )

    # A different key must not hit the first key's cached balance.
    assert (
        provider_usage._cache_get(
            f"openrouter_balance:{provider_usage._key_fingerprint(key_b)}"
        )
        is None
    )
    assert (
        provider_usage._cache_get(
            f"openrouter_balance:{provider_usage._key_fingerprint(key_a)}"
        )
        == {"available": True, "remaining_label": "$10.00"}
    )


@pytest.mark.asyncio
async def test_openrouter_balance_refetches_after_key_change(monkeypatch):
    from app.services import provider_usage

    provider_usage._cache.clear()

    calls = []

    class _Resp:
        status_code = 200

        def __init__(self, remaining):
            self._remaining = remaining

        def json(self):
            return {"data": {"total_credits": self._remaining, "total_usage": 0}}

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, headers=None):
            calls.append(headers["Authorization"])
            token = headers["Authorization"]
            return _Resp(100.0 if token.endswith("key-A") else 5.0)

    monkeypatch.setattr(provider_usage.httpx, "AsyncClient", _Client)

    first = await provider_usage._openrouter_balance(api_key="key-A")
    second = await provider_usage._openrouter_balance(api_key="key-B")

    assert first["remaining_label"] == "$100.00"
    assert second["remaining_label"] == "$5.00"
    # Both keys triggered real lookups; the second was not served stale cache.
    assert calls == ["Bearer key-A", "Bearer key-B"]
