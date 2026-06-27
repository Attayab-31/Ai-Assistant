"""
app/api/auth.py — Authentication routes: login, logout, current user.

Uses JWT tokens stored in httpOnly cookies. Rate-limited login endpoint
(5 attempts then 15-min lockout) protects against brute force.
"""

import asyncio
import logging
import uuid
from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, EmailStr
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.ratelimit import (
    check_auth_rate_limit,
    record_auth_failure,
    reset_auth_failures,
)
from app.db import crud
from app.db.crud import create_audit_log, get_user_by_email, update_last_login
from app.db.database import AsyncSessionLocal, get_db
from app.utils.security import (
    MAX_BCRYPT_PASSWORD_BYTES,
    create_access_token,
    hash_password,
    mask_email,
    password_needs_rehash,
    verify_password,
)
from config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

COOKIE_NAME = "access_token"
MIN_PASSWORD_LENGTH = 8


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict


def _cookie_secure(request: Request) -> bool:
    """Whether the auth cookie should be marked Secure.

    Production is always HTTPS (enforced at startup), so we require Secure
    there. Behind a TLS-terminating proxy the direct request scheme is "http",
    so we also honor the X-Forwarded-Proto header set by the proxy.
    """
    if settings.is_production:
        return True
    forwarded = request.headers.get("x-forwarded-proto", "")
    if forwarded.split(",")[0].strip().lower() == "https":
        return True
    return request.url.scheme == "https"


def _validate_password(password: str) -> None:
    if len(password) < MIN_PASSWORD_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Password must be at least {MIN_PASSWORD_LENGTH} characters",
        )
    if len(password.encode("utf-8")) > MAX_BCRYPT_PASSWORD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Password must be {MAX_BCRYPT_PASSWORD_BYTES} bytes or fewer",
        )


async def _post_login_tasks(
    user_id: uuid.UUID,
    *,
    ip_address: str | None,
    user_agent: str | None,
    plain_password: str,
    stored_hash: str,
) -> None:
    """Non-blocking after-login bookkeeping in its own DB session.

    Runs the last-login timestamp, audit log, and (if needed) a bcrypt rehash
    off the request path so the login response returns as fast as possible.
    """
    try:
        async with AsyncSessionLocal() as db:
            await update_last_login(db, user_id)
            await create_audit_log(
                db,
                action="admin_login",
                admin_user_id=user_id,
                entity_type="auth",
                ip_address=ip_address,
                user_agent=user_agent,
            )
            if password_needs_rehash(stored_hash):
                await crud.update_user_password(
                    db, user_id, hash_password(plain_password)
                )
    except Exception as e:
        logger.warning("Post-login bookkeeping failed for %s: %s", user_id, e)


@router.post("/login", response_model=LoginResponse)
async def login(
    request: Request,
    response: Response,
    credentials: LoginRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Authenticate admin user and issue JWT token.

    Brute-force protection counts only failed attempts (per IP) and is cleared on
    success, so a correct password is never throttled. This in-process backstop is
    used instead of a Redis-backed limiter, which is unusable on a read-only Redis
    user (the limits library needs EVALSHA, which read-only users cannot run).
    """
    check_auth_rate_limit(request)

    user = await get_user_by_email(db, credentials.email)

    if not user or not verify_password(credentials.password, user.hashed_password):
        record_auth_failure(request)
        logger.warning("Failed login attempt for: %s", mask_email(credentials.email))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated",
        )

    # Valid credentials — clear any recorded failures so this IP starts fresh.
    reset_auth_failures(request)

    token = create_access_token(
        data={"sub": str(user.id), "email": user.email, "role": user.role},
        expires_delta=timedelta(hours=8),
    )

    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        httponly=True,
        secure=_cookie_secure(request),
        samesite="lax",
        max_age=8 * 3600,
    )

    # Bookkeeping (last login, audit, rehash) runs off the response path.
    asyncio.create_task(
        _post_login_tasks(
            user.id,
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            plain_password=credentials.password,
            stored_hash=user.hashed_password,
        )
    )

    logger.info("Login successful: %s", mask_email(user.email))

    return LoginResponse(
        access_token=token,
        user={
            "id": str(user.id),
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role,
        },
    )


@router.post("/logout")
async def logout(response: Response):
    """Clear the auth cookie."""
    response.delete_cookie(COOKIE_NAME)
    return {"message": "Logged out successfully"}


@router.get("/me")
async def get_current_user_info(request: Request, db: AsyncSession = Depends(get_db)):
    """Get current authenticated user info."""
    from app.utils.dependencies import get_current_user

    user = await get_current_user(request, db)
    return {
        "id": str(user.id),
        "email": user.email,
        "full_name": user.full_name,
        "role": user.role,
        "last_login": user.last_login.isoformat() if user.last_login else None,
    }
