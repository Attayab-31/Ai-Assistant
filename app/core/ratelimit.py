"""Shared rate limiter plus an in-memory auth brute-force backstop.

A single Limiter instance is kept for the app's RateLimitExceeded exception
handler. It uses *in-memory* storage rather than Redis: the project's Redis user
is read-only (Upstash ``default_ro``), and the ``limits`` library implements its
Redis counters with a Lua script (EVALSHA). A read-only user cannot run EVALSHA,
which raises ``NoPermissionError`` and, because slowapi's ``swallow_errors`` does
not reliably catch storage failures from the decorator path, surfaced as
intermittent HTTP 500s on ``/auth/login``. In-memory storage avoids Redis
entirely for rate limiting and never fails closed.

Auth brute-force protection is provided by the functions below, NOT by a slowapi
decorator. Critically, this backstop counts only FAILED attempts and is cleared
on a successful login. A correct password therefore never contributes to a
lockout — only repeated bad attempts do — so a legitimate admin can log in (and
log in again) as many times as they like without ever tripping the limit.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from threading import Lock

from fastapi import HTTPException, Request, status
from slowapi import Limiter
from slowapi.util import get_remote_address

# In-memory storage: never touches the read-only Redis (see module docstring).
# A single shared instance is kept for the app's RateLimitExceeded handler.
limiter = Limiter(
    key_func=get_remote_address,
    storage_uri="memory://",
    swallow_errors=True,
)

# In-memory brute-force backstop for auth endpoints. Tracks recent *failed*
# attempts per client IP; successful logins clear the bucket.
_AUTH_FALLBACK_WINDOW_S = 15 * 60
_AUTH_FALLBACK_MAX_FAILURES = 10
_auth_failures: dict[str, deque[float]] = defaultdict(deque)
_auth_failures_lock = Lock()


def _prune(bucket: deque[float], cutoff: float) -> None:
    while bucket and bucket[0] < cutoff:
        bucket.popleft()


def check_auth_rate_limit(request: Request) -> None:
    """Raise HTTP 429 if this IP has too many recent *failed* auth attempts.

    Does not itself record an attempt — call ``record_auth_failure`` on a bad
    login and ``reset_auth_failures`` on a successful one. This way a valid
    password is never counted and can never cause a lockout.
    """
    ip = get_remote_address(request)
    cutoff = time.monotonic() - _AUTH_FALLBACK_WINDOW_S
    with _auth_failures_lock:
        bucket = _auth_failures.get(ip)
        if bucket is None:
            return
        _prune(bucket, cutoff)
        if len(bucket) >= _AUTH_FALLBACK_MAX_FAILURES:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many failed attempts. Please try again later.",
            )


def record_auth_failure(request: Request) -> None:
    """Record a failed auth attempt for this IP (drives the 429 backstop)."""
    ip = get_remote_address(request)
    now = time.monotonic()
    cutoff = now - _AUTH_FALLBACK_WINDOW_S
    with _auth_failures_lock:
        bucket = _auth_failures[ip]
        _prune(bucket, cutoff)
        bucket.append(now)

        # Drop empty buckets for IPs that have aged out so the dict can't grow
        # without bound under churn.
        if len(_auth_failures) > 1024:
            for key in [k for k, v in _auth_failures.items() if not v]:
                del _auth_failures[key]


def reset_auth_failures(request: Request) -> None:
    """Clear recorded failures for this IP after a successful login."""
    ip = get_remote_address(request)
    with _auth_failures_lock:
        _auth_failures.pop(ip, None)
