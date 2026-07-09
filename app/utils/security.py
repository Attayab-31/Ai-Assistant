"""Password hashing, JWT auth, and small secret-handling helpers."""

import base64
import hashlib
import ipaddress
import logging
import socket
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlparse

import bcrypt
from cryptography.fernet import Fernet
from jose import JWTError, jwt

from config import settings

logger = logging.getLogger(__name__)

ALGORITHM = "HS256"
# 11 rounds (~2x faster than 12) keeps logins snappy while staying well above
# the OWASP-recommended minimum. Hashes embed their own cost, so old 12-round
# hashes still verify; password_needs_rehash() lets us migrate them on login.
BCRYPT_ROUNDS = 11
MAX_BCRYPT_PASSWORD_BYTES = 72


def _password_bytes(password: str) -> bytes:
    data = password.encode("utf-8")
    if len(data) > MAX_BCRYPT_PASSWORD_BYTES:
        raise ValueError(
            f"bcrypt passwords must be {MAX_BCRYPT_PASSWORD_BYTES} bytes or fewer"
        )
    return data


def hash_password(password: str) -> str:
    """Hash a plaintext password for storage."""
    return bcrypt.hashpw(
        _password_bytes(password),
        bcrypt.gensalt(rounds=BCRYPT_ROUNDS),
    ).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against a stored hash."""
    if not hashed_password:
        return False

    try:
        return bcrypt.checkpw(
            _password_bytes(plain_password),
            hashed_password.encode("utf-8"),
        )
    except (TypeError, ValueError):
        return False


def password_needs_rehash(hashed_password: str) -> bool:
    """True when a stored hash uses a different bcrypt cost than the current one."""
    try:
        cost = int(hashed_password.split("$")[2])
    except (IndexError, ValueError, AttributeError):
        return True
    return cost != BCRYPT_ROUNDS


def create_access_token(
    data: dict[str, Any], expires_delta: timedelta | None = None
) -> str:
    """Create a signed JWT access token."""
    expire = datetime.now(UTC) + (expires_delta or timedelta(hours=8))
    payload = data.copy()
    payload.update({"exp": expire})
    return jwt.encode(payload, settings.secret_key, algorithm=ALGORITHM)


def decode_access_token(token: str) -> dict[str, Any] | None:
    """Decode a JWT token, returning None when invalid or expired."""
    if not isinstance(token, str) or not token:
        return None

    try:
        return jwt.decode(token, settings.secret_key, algorithms=[ALGORITHM])
    except JWTError:
        return None


def _fernet() -> Fernet:
    raw_key = settings.encryption_key.strip()
    if raw_key:
        try:
            return Fernet(raw_key.encode())
        except ValueError:
            logger.error(
                "ENCRYPTION_KEY is set but invalid — must be a Fernet urlsafe key"
            )
            if settings.is_production:
                raise RuntimeError(
                    "ENCRYPTION_KEY is invalid; fix the configured key"
                ) from None
            logger.warning(
                "Falling back to secret_key-derived encryption key (dev only)"
            )

    digest = hashlib.sha256(settings.secret_key.encode()).digest()
    return Fernet(base64.urlsafe_b64encode(digest))


def is_encrypted_value(value: str) -> bool:
    """True when ``value`` is ciphertext that decrypts with the active Fernet key."""
    val = (value or "").strip()
    if not val:
        return False
    try:
        _fernet().decrypt(val.encode())
        return True
    except Exception:
        return False


def encrypt_value(value: str) -> str:
    """Encrypt a sensitive string for storage in settings."""
    return _fernet().encrypt(value.encode()).decode()


def decrypt_value(value: str) -> str:
    """Decrypt a value encrypted with encrypt_value."""
    return _fernet().decrypt(value.encode()).decode()


_SENSITIVE_SETTING_KEYS = frozenset({"crm_webhook_secret"})


def is_sensitive_setting_key(key: str) -> bool:
    """True for secrets that must be masked in API responses and audit logs."""
    return key in _SENSITIVE_SETTING_KEYS or key.endswith("_api_key_encrypted")


def redact_for_audit(data: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``data`` with sensitive values replaced for audit storage."""
    return {
        k: ("***" if is_sensitive_setting_key(k) else v) for k, v in data.items()
    }


# ──────────────────────────────────────────────────────────────────────────────
# PII redaction (for logs)
# ──────────────────────────────────────────────────────────────────────────────


def mask_phone(phone: str | None) -> str:
    """Mask a phone number for logging, keeping only the last 4 digits."""
    if not phone:
        return "<none>"
    digits = [c for c in phone if c.isdigit()]
    if len(digits) < 4:
        return "***"
    return "***" + "".join(digits[-4:])


def mask_email(email: str | None) -> str:
    """Mask an email for logging (e.g. ``ab***@example.com``)."""
    if not email or "@" not in email:
        return "<none>"
    local, _, domain = email.partition("@")
    visible = local[:2]
    return f"{visible}***@{domain}"


# ──────────────────────────────────────────────────────────────────────────────
# SSRF protection for outbound HTTP to externally-influenced URLs
# ──────────────────────────────────────────────────────────────────────────────


class UnsafeURLError(ValueError):
    """Raised when a URL targets a non-public / internal address."""


def _is_blocked_ip(ip: str) -> bool:
    """True for loopback/private/link-local/reserved/multicast addresses."""
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return True
    return (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_multicast
        or addr.is_unspecified
    )


def assert_safe_external_url(url: str, *, require_https: bool = False) -> str:
    """Validate that ``url`` points at a public host before fetching/posting.

    Guards against SSRF when the URL is influenced by an external party (a
    webhook payload) or an admin-configured destination. Resolves the host and
    rejects any that maps to a private/loopback/link-local/reserved address.

    Returns the URL unchanged when safe; raises ``UnsafeURLError`` otherwise.
    """
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    allowed_schemes = {"https"} if require_https else {"http", "https"}
    if scheme not in allowed_schemes:
        raise UnsafeURLError(f"URL scheme not allowed: {scheme or '<none>'}")

    host = parsed.hostname
    if not host:
        raise UnsafeURLError("URL has no host")

    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if scheme == "https" else 80))
    except socket.gaierror as e:
        raise UnsafeURLError(f"Could not resolve host: {host}") from e

    for info in infos:
        ip = info[4][0]
        if _is_blocked_ip(ip):
            raise UnsafeURLError(f"URL resolves to a non-public address: {host} → {ip}")

    return url
