"""
main.py — FastAPI application entry point.

Bootstraps the AI Tenant Screening Platform:
- Registers all routers
- Sets up middleware (CORS, security headers, rate limiting)
- Initializes DB, Redis, and ProviderRegistry on startup
- Serves static files and admin templates (via app/api/admin.py)
- Provides /health endpoint
"""

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.api.admin import router as admin_router
from app.api.auth import router as auth_router
from app.api.settings import router as settings_router
from app.api.test_console import router as test_console_router
from app.api.webhook import router as webhook_router

# ──────────────────────────────────────────────────────────────────────────────
# Logging configuration
# ──────────────────────────────────────────────────────────────────────────────
from app.core.logging_config import setup_logging
from app.core.ratelimit import limiter
from app.utils.dependencies import ACCESS_TOKEN_COOKIE_NAME
from config import provider_registry, settings

setup_logging()
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Application startup / shutdown
# ──────────────────────────────────────────────────────────────────────────────

APP_START_TIME = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup → yield → shutdown."""
    # Re-apply after uvicorn configures logging so voice trace handler stays active.
    setup_logging()
    logger.info("🚀 AI Tenant Screener starting up...")

    # Fail fast on insecure production configuration so we never serve real
    # traffic with development defaults or unverifiable webhooks.
    config_errors = settings.validate_runtime_secrets()
    if config_errors:
        for err in config_errors:
            logger.critical("Production configuration error: %s", err)
        raise RuntimeError(
            "Refusing to start in production with insecure configuration: "
            + "; ".join(config_errors)
        )

    if settings.web_workers != 1:
        logger.warning(
            "WEB_WORKERS=%s — live call sessions are in-process only; "
            "use WEB_WORKERS=1 until shared session store exists",
            settings.web_workers,
        )

    # Database
    try:
        from app.db.database import init_db

        await init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.error("Database startup failed: %s", e, exc_info=True)
        raise

    # Seed initial data
    try:
        from app.db.crud import seed_defaults
        from app.db.database import AsyncSessionLocal

        async with AsyncSessionLocal() as db:
            await seed_defaults(db)
        logger.info("Default data verified")
    except Exception as e:
        logger.error("Database seed failed: %s", e, exc_info=True)
        raise

    # Provider Registry
    try:
        await provider_registry.initialize()
        logger.info("✅ Provider registry initialized")
    except Exception as e:
        logger.warning("Provider registry init partial: %s", e)

    # Seed the in-process display timezone from the admin setting so synchronous
    # template/email helpers localize timestamps without a DB round-trip.
    try:
        from app.db.crud import get_setting_value
        from app.db.database import AsyncSessionLocal
        from app.utils.helpers import set_display_timezone

        async with AsyncSessionLocal() as db:
            set_display_timezone(await get_setting_value(db, "timezone", ""))
    except Exception as e:
        logger.warning("Could not load display timezone: %s", e)

    logger.info("✅ AI Tenant Screener ready!")
    yield

    # Shutdown
    logger.info("👋 AI Tenant Screener shutting down...")
    try:
        from app.services.telnyx_service import telnyx_service

        await telnyx_service.close()
    except Exception as e:
        logger.warning("Error closing Telnyx service: %s", e)
    try:
        from app.db.database import engine

        await engine.dispose()
    except Exception as e:
        logger.warning("Error disposing database engine: %s", e)
    try:
        from app.core.redis_client import close_redis

        await close_redis()
    except Exception as e:
        logger.warning("Error closing Redis client: %s", e)


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Tenant Screener",
    description="Production-grade AI-powered tenant screening platform",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs" if not settings.is_production else None,
    redoc_url="/api/redoc" if not settings.is_production else None,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ──────────────────────────────────────────────────────────────────────────────
# Middleware
# ──────────────────────────────────────────────────────────────────────────────

_CSRF_PROTECTED_PREFIXES = ("/admin/api/", "/api/settings/")
_CSRF_PROTECTED_PATHS = frozenset({"/auth/logout"})

# With cookie-based auth we need credentialed CORS, which browsers reject
# alongside a "*" origin. Use explicit origins (credentials on) in production
# and a permissive, non-credentialed policy in development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.app_url] if settings.is_production else ["*"],
    allow_credentials=settings.is_production,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def csrf_middleware(request: Request, call_next):
    """Require a custom header on cookie-authenticated mutating requests."""
    if request.method not in ("POST", "PUT", "PATCH", "DELETE"):
        return await call_next(request)

    path = request.url.path
    protected = path in _CSRF_PROTECTED_PATHS or path.startswith(_CSRF_PROTECTED_PREFIXES)
    if not protected or not request.cookies.get(ACCESS_TOKEN_COOKIE_NAME):
        return await call_next(request)

    # Bearer-authenticated API clients are not vulnerable to cookie CSRF.
    if request.headers.get("Authorization", "").startswith("Bearer "):
        return await call_next(request)

    if request.headers.get("X-Requested-With") != "XMLHttpRequest":
        return JSONResponse({"detail": "Missing CSRF header"}, status_code=403)

    return await call_next(request)


@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    """Add security headers to every response."""
    response: Response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    if settings.is_production:
        response.headers[
            "Strict-Transport-Security"
        ] = "max-age=31536000; includeSubDomains"
    return response


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """Log all incoming requests with timing."""
    start = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start) * 1000
    logger.info(
        "%s %s → %s (%.1fms)",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


# ──────────────────────────────────────────────────────────────────────────────
# Static files
# ──────────────────────────────────────────────────────────────────────────────

STATIC_DIR = Path(__file__).parent / "app" / "admin" / "static"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ──────────────────────────────────────────────────────────────────────────────
# Routers
# ──────────────────────────────────────────────────────────────────────────────


app.include_router(webhook_router, prefix="/telnyx", tags=["Telnyx Webhooks"])
app.include_router(auth_router, prefix="/auth", tags=["Authentication"])
app.include_router(admin_router, prefix="/admin", tags=["Admin Dashboard"])
app.include_router(settings_router, prefix="/api/settings", tags=["Settings API"])
if settings.allow_test_console:
    app.include_router(test_console_router, prefix="/test", tags=["Test Console"])
else:
    logger.info("Test console disabled (production default)")


# ──────────────────────────────────────────────────────────────────────────────
# Health endpoint
# ──────────────────────────────────────────────────────────────────────────────


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint. Returns app, DB, Redis, provider status."""
    uptime = time.time() - APP_START_TIME
    db_ok = False
    redis_ok = False

    # Check DB
    try:
        from app.db.database import engine

        async with engine.connect() as conn:
            await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
        db_ok = True
    except Exception as e:
        logger.warning("DB health check failed: %s", e)

    # Check Redis via the shared pooled client (no per-call connections).
    try:
        from app.core.redis_client import ping as redis_ping

        redis_ok = await redis_ping()
    except Exception as e:
        logger.warning("Redis health check failed: %s", e)

    providers = provider_registry.get_status()

    # The database is the only hard dependency for serving requests, so a DB
    # failure returns HTTP 503 to keep load balancers from routing to this
    # instance. Redis powers async jobs/cache and is reported but not fatal.
    healthy = db_ok
    return JSONResponse(
        {
            "status": "healthy" if healthy else "degraded",
            "app": settings.app_name,
            "version": "1.0.0",
            "environment": settings.environment,
            "uptime_seconds": round(uptime, 1),
            "database": "connected" if db_ok else "disconnected",
            "redis": "connected" if redis_ok else "disconnected",
            "providers": providers,
        },
        status_code=200 if healthy else 503,
    )


@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to admin dashboard."""
    from fastapi.responses import RedirectResponse

    return RedirectResponse(url="/admin/dashboard")
