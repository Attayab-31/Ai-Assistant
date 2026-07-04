"""
app/db/database.py — Async SQLAlchemy engine, session factory, and DB init.

Uses async SQLAlchemy for PostgreSQL/Supabase.
Provides:
- engine: async engine singleton
- AsyncSessionLocal: session factory
- get_db(): FastAPI dependency that yields a session
- init_db(): Runs or validates Alembic migrations on startup
"""

import logging
import socket
from collections.abc import AsyncGenerator
from uuid import uuid4

from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import IntegrityError, OperationalError, SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool

from config import settings

logger = logging.getLogger(__name__)


class DatabaseInitializationError(RuntimeError):
    """Raised when the configured database cannot be initialized."""

    pass


# ──────────────────────────────────────────────────────────────────────────────
# Declarative base for all SQLAlchemy models
# ──────────────────────────────────────────────────────────────────────────────


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass


# ──────────────────────────────────────────────────────────────────────────────
# Engine & session factory
# ──────────────────────────────────────────────────────────────────────────────

VALID_SSL_MODES = {"disable", "allow", "prefer", "require", "verify-ca", "verify-full"}
VALID_POOL_MODES = {"auto", "direct", "session", "transaction"}


def _is_supabase_host(host: str | None) -> bool:
    """Return True when a database host belongs to Supabase."""
    if not host:
        return False
    return host.endswith(".supabase.co") or host.endswith(".pooler.supabase.com")


def _normalize_database_url(raw_url: str) -> tuple[URL, str | None]:
    """
    Accept Supabase dashboard URLs and convert them for SQLAlchemy asyncpg.

    Supabase commonly shows postgres:// or postgresql:// URLs with
    sslmode=require. SQLAlchemy's asyncpg dialect needs postgresql+asyncpg://
    and receives SSL as an asyncpg connect argument instead of sslmode.
    """
    url = make_url(raw_url)

    if url.drivername in {"postgres", "postgresql"}:
        url = url.set(drivername="postgresql+asyncpg")

    query = dict(url.query)
    sslmode = query.pop("sslmode", None)
    if sslmode:
        sslmode = str(sslmode).lower()
        if sslmode not in VALID_SSL_MODES:
            raise ValueError(
                f"Unsupported DATABASE_URL sslmode '{sslmode}'. "
                f"Use one of: {', '.join(sorted(VALID_SSL_MODES))}."
            )
        url = url.set(query=query)

    return url, sslmode


def _database_ssl_mode(url: URL, sslmode_from_url: str | None) -> str | None:
    configured = settings.database_ssl.strip().lower()
    if sslmode_from_url:
        return sslmode_from_url
    if configured and configured != "auto":
        if configured not in VALID_SSL_MODES:
            raise ValueError(
                f"Unsupported DATABASE_SSL '{configured}'. "
                f"Use auto or one of: {', '.join(sorted(VALID_SSL_MODES))}."
            )
        return configured
    if _is_supabase_host(url.host):
        return "require"
    return None


def _database_pool_mode(url: URL) -> str:
    configured = settings.database_pool_mode.strip().lower()
    if configured not in VALID_POOL_MODES:
        raise ValueError(
            f"Unsupported DATABASE_POOL_MODE '{configured}'. "
            f"Use one of: {', '.join(sorted(VALID_POOL_MODES))}."
        )
    if configured != "auto":
        return configured
    if url.get_backend_name() != "postgresql":
        return "direct"
    if url.port == 6543:
        return "transaction"
    if url.host and url.host.endswith(".pooler.supabase.com"):
        return "session"
    return "direct"


def _is_supabase_direct_host(url: URL) -> bool:
    return bool(
        url.host
        and url.host.startswith("db.")
        and url.host.endswith(".supabase.co")
        and (url.port is None or url.port == 5432)
    )


def _supabase_project_ref(url: URL) -> str | None:
    if not _is_supabase_direct_host(url) or not url.host:
        return None
    return url.host.removeprefix("db.").removesuffix(".supabase.co")


database_url, database_ssl_mode = _normalize_database_url(settings.database_url)
database_pool_mode = _database_pool_mode(database_url)
database_resolved_ssl_mode = (
    _database_ssl_mode(database_url, database_ssl_mode)
    if database_url.get_backend_name() == "postgresql"
    and "asyncpg" in database_url.drivername
    else None
)


def describe_database_target() -> str:
    """Return a password-safe summary of the configured database target."""
    return (
        f"driver={database_url.drivername}, "
        f"user={database_url.username or '<none>'}, "
        f"host={database_url.host or '<none>'}, "
        f"port={database_url.port or '<default>'}, "
        f"database={database_url.database or '<none>'}, "
        f"pool_mode={database_pool_mode}, "
        f"ssl={database_resolved_ssl_mode or 'default'}"
    )


def _walk_exception_chain(exc: BaseException):
    seen: set[int] = set()
    stack: list[BaseException] = [exc]

    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current

        if isinstance(current, BaseExceptionGroup):
            stack.extend(current.exceptions)
        if current.__cause__ is not None:
            stack.append(current.__cause__)
        if current.__context__ is not None:
            stack.append(current.__context__)


def _contains_exception(exc: BaseException, exc_type: type[BaseException]) -> bool:
    return any(isinstance(item, exc_type) for item in _walk_exception_chain(exc))


def format_database_connection_error(exc: BaseException) -> str:
    """Build a clear, password-safe database startup error."""
    details = [
        f"Could not initialize database at {describe_database_target()}.",
        f"Original error: {exc}",
    ]

    if _contains_exception(exc, socket.gaierror):
        details.append(
            "The configured database host could not be resolved by DNS from this "
            "runtime."
        )

    if _is_supabase_direct_host(database_url):
        project_ref = _supabase_project_ref(database_url) or "<project-ref>"
        details.append(
            "This DATABASE_URL uses Supabase's direct host. Supabase direct "
            "connections require IPv6 unless the project has the IPv4 add-on. "
            "On IPv4-only networks, use the Session pooler connection string "
            "from Supabase Dashboard > Connect instead, for example: "
            f"postgresql://postgres.{project_ref}:<password>"
            "@aws-0-<region>.pooler.supabase.com:5432/postgres?sslmode=require"
        )

    return " ".join(details)


engine_kwargs = {
    "echo": settings.debug,
    "pool_pre_ping": True,
}
connect_args = {}

if database_url.get_backend_name() == "postgresql":
    if database_pool_mode == "transaction":
        engine_kwargs["poolclass"] = NullPool
    else:
        engine_kwargs.update(
            {
                "pool_size": settings.database_pool_size,
                "max_overflow": settings.database_max_overflow,
                "pool_recycle": settings.database_pool_recycle_seconds,
                "pool_timeout": 30,
            }
        )

    if "asyncpg" in database_url.drivername:
        connect_args = {
            "server_settings": {"jit": "off"},
            "timeout": settings.database_connect_timeout_seconds,
        }
        if database_resolved_ssl_mode:
            connect_args["ssl"] = database_resolved_ssl_mode
        if database_pool_mode == "transaction":
            connect_args.update(
                {
                    "statement_cache_size": 0,
                    "prepared_statement_cache_size": 0,
                    "prepared_statement_name_func": lambda: f"__asyncpg_{uuid4()}__",
                }
            )

if connect_args:
    engine_kwargs["connect_args"] = connect_args

engine = create_async_engine(database_url, **engine_kwargs)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI dependency
# ──────────────────────────────────────────────────────────────────────────────


async def get_db() -> AsyncGenerator[AsyncSession]:
    """
    FastAPI dependency that provides a database session.
    Automatically commits on success, rolls back on error, and closes on exit.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except (SQLAlchemyError, IntegrityError, OperationalError) as e:
            await session.rollback()
            logger.error("Database error during transaction: %s", e)
            raise
        except Exception as e:
            await session.rollback()
            logger.critical(
                "Unexpected error during database transaction: %s", e, exc_info=True
            )
            raise
        finally:
            await session.close()


# ──────────────────────────────────────────────────────────────────────────────
# Database initialization
# ──────────────────────────────────────────────────────────────────────────────


def _register_orm_models() -> None:
    """Import all ORM models so SQLAlchemy registers them with Base.metadata."""
    from app.models.audit_log import AuditLog  # noqa: F401
    from app.models.call import Call  # noqa: F401
    from app.models.settings import SystemSetting  # noqa: F401
    from app.models.tenant import Tenant  # noqa: F401
    from app.models.user import AdminUser  # noqa: F401


async def init_db() -> None:
    """
    Initialize the database schema using Alembic migrations.

    In development, the default migration mode upgrades to the latest migration.
    In production, the default migration mode validates that migrations were
    already applied before the app starts.
    """
    from app.db.migrations import initialize_database_schema

    _register_orm_models()
    try:
        await initialize_database_schema()
    except Exception as exc:
        raise DatabaseInitializationError(
            format_database_connection_error(exc)
        ) from exc

    logger.info("Database schema initialized")
