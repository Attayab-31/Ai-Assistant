"""FastAPI dependency functions for auth and RBAC."""

import time
import uuid

from fastapi import Cookie, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.crud import get_user_by_id
from app.db.database import get_db
from app.models.user import AdminUser
from app.utils.security import decode_access_token
from config import settings

ACCESS_TOKEN_COOKIE_NAME = "access_token"

# Short-lived cache of authenticated users so back-to-back admin requests don't
# each hit the database. Disabled in production so role/deactivation changes
# take effect immediately.
USER_CACHE_TTL_SECONDS = 0.0 if settings.is_production else 30.0
_user_cache: dict[uuid.UUID, tuple[float, AdminUser]] = {}


def _resolve_access_token(request: Request, access_token: str | None) -> str | None:
    """Return a JWT from cookie injection, request cookies, or bearer auth."""
    if isinstance(access_token, str) and access_token:
        return access_token

    cookie_token = request.cookies.get(ACCESS_TOKEN_COOKIE_NAME)
    if cookie_token:
        return cookie_token

    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header.split(" ", 1)[1]

    return None


async def get_current_user(
    request: Request,
    db: AsyncSession = Depends(get_db),
    access_token: str | None = Cookie(default=None),
) -> AdminUser:
    """
    FastAPI dependency: get the currently authenticated admin user.
    Raises 401 if not authenticated or token invalid/expired.
    """
    token = _resolve_access_token(request, access_token)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )

    from app.core.redis_client import is_token_revoked

    if await is_token_revoked(token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session expired",
        )

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload"
        )

    try:
        user_uuid = uuid.UUID(user_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid user ID in token"
        ) from e

    cached = _user_cache.get(user_uuid)
    if cached and cached[0] > time.monotonic():
        user = cached[1]
    else:
        user = await get_user_by_id(db, user_uuid)
        if user and user.is_active:
            _user_cache[user_uuid] = (time.monotonic() + USER_CACHE_TTL_SECONDS, user)

    if not user or not user.is_active:
        _user_cache.pop(user_uuid, None)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
        )

    return user


def require_role(*allowed_roles: str):
    """
    Dependency factory: require the current user to have one of the given roles.
    Usage: Depends(require_role("super_admin", "admin"))
    """

    async def _check_role(user: AdminUser = Depends(get_current_user)) -> AdminUser:
        if user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires one of roles: {', '.join(allowed_roles)}",
            )
        return user

    return _check_role


# Staff = anyone allowed to mutate data. The "viewer" role is read-only and is
# intentionally excluded so viewers cannot edit notes, override qualification,
# resend emails, change settings, etc. Use ``require_scope(..., edit=True)`` on
# write endpoints instead of a separate staff dependency.


def require_scope(scope: str, *, edit: bool = False):
    """Dependency factory: require access to a feature-area scope.

    Usage:
        Depends(require_scope("calls"))             # any access to Calls
        Depends(require_scope("calls", edit=True))  # must also be able to edit

    super_admin / admin always have every scope. staff / viewer are limited to
    the scopes granted on their account. When edit=True, read-only viewers are
    rejected even if they can see the area.
    """

    async def _check_scope(user: AdminUser = Depends(get_current_user)) -> AdminUser:
        if not user.can(scope):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Your account doesn't have access to {scope}.",
            )
        if edit and not user.can_edit:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Your account is read-only.",
            )
        return user

    return _check_scope


def invalidate_user_cache(user_id: uuid.UUID) -> None:
    """Drop a cached user row so role/scope changes take effect immediately."""
    _user_cache.pop(user_id, None)


async def get_current_user_optional(
    request: Request,
    db: AsyncSession = Depends(get_db),
    access_token: str | None = Cookie(default=None),
) -> AdminUser | None:
    """Like get_current_user but returns None instead of raising on failure."""
    try:
        return await get_current_user(request, db, access_token)
    except HTTPException:
        return None
