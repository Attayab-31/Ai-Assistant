"""Render deployment configuration tests."""

from __future__ import annotations

from unittest.mock import patch


def test_resolve_app_url_prefers_https_over_render_internal(monkeypatch):
    from config import Settings

    monkeypatch.delenv("APP_URL", raising=False)
    monkeypatch.delenv("RENDER_EXTERNAL_URL", raising=False)
    monkeypatch.setenv("RENDER_INTERNAL_URL", "http://ai-screener-web:10000")

    settings = Settings()
    assert settings.app_url == "http://localhost:8000"


def test_resolve_app_url_uses_render_external_when_placeholder(monkeypatch):
    from config import Settings

    monkeypatch.delenv("APP_URL", raising=False)
    monkeypatch.setenv(
        "RENDER_EXTERNAL_URL", "https://ai-screener-web.onrender.com"
    )
    monkeypatch.setenv("RENDER_INTERNAL_URL", "http://ai-screener-web:10000")

    settings = Settings()
    assert settings.app_url == "https://ai-screener-web.onrender.com"


def test_resolve_app_url_explicit_https_wins(monkeypatch):
    from config import Settings

    monkeypatch.setenv("APP_URL", "https://ai-screener-web.onrender.com")
    monkeypatch.setenv("RENDER_INTERNAL_URL", "http://ai-screener-web:10000")

    settings = Settings()
    assert settings.app_url == "https://ai-screener-web.onrender.com"


def test_resolve_app_url_strips_trailing_slash(monkeypatch):
    from config import Settings

    monkeypatch.setenv("APP_URL", "https://ai-screener-web.onrender.com/")

    settings = Settings()
    assert settings.app_url == "https://ai-screener-web.onrender.com"


def test_validate_runtime_secrets_allows_bootstrap_without_telnyx():
    from config import Settings

    settings = Settings(
        environment="production",
        secret_key="x" * 32,
        encryption_key="dGVzdC1rZXktdGVzdC1rZXktdGVzdC1rZXk=",
        app_url="https://example.com",
        telnyx_public_key="",
        telnyx_api_key="",
        debug=False,
        admin_password="Admin123!",
        web_workers=1,
        trusted_proxy_ips="10.0.0.0/8",
        bootstrap_deploy=True,
        redis_url="redis://red-abc:6379",
        celery_broker_url="redis://red-abc:6379",
        celery_result_backend="redis://red-abc:6379",
    )
    with patch("cryptography.fernet.Fernet"):
        errors = settings.validate_runtime_secrets()
    assert errors == []


def test_validate_runtime_secrets_rejects_localhost_redis(monkeypatch):
    from config import Settings

    for key in ("REDIS_URL", "CELERY_BROKER_URL", "CELERY_RESULT_BACKEND"):
        monkeypatch.delenv(key, raising=False)

    settings = Settings(
        environment="production",
        secret_key="x" * 32,
        encryption_key="dGVzdC1rZXktdGVzdC1rZXktdGVzdC1rZXk=",
        app_url="https://example.com",
        telnyx_public_key="pub",
        telnyx_api_key="telnyx",
        debug=False,
        admin_password="strong-password-here",
        web_workers=1,
        trusted_proxy_ips="10.0.0.0/8",
        redis_url="redis://localhost:6379/0",
        celery_broker_url="redis://localhost:6379/1",
        celery_result_backend="redis://localhost:6379/2",
    )
    with patch("cryptography.fernet.Fernet"):
        errors = settings.validate_runtime_secrets()
    assert any("localhost" in e for e in errors)


def test_validate_celery_runtime_secrets_rejects_localhost_redis(monkeypatch):
    from config import Settings

    for key in ("REDIS_URL", "CELERY_BROKER_URL", "CELERY_RESULT_BACKEND"):
        monkeypatch.delenv(key, raising=False)

    settings = Settings(
        environment="production",
        secret_key="x" * 32,
        app_url="https://ai-screener-web.onrender.com",
        debug=False,
        redis_url="redis://localhost:6379/0",
        celery_broker_url="redis://localhost:6379/1",
        celery_result_backend="redis://localhost:6379/2",
    )
    errors = settings.validate_celery_runtime_secrets(require_encryption=False)
    assert any("localhost" in e for e in errors)
