"""Alembic migration helpers used during application startup."""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import inspect, text

from alembic import command
from app.db.database import (
    Base,
    DatabaseInitializationError,
    _register_orm_models,
    engine,
)
from config import settings

logger = logging.getLogger(__name__)

CORE_TABLES = {"admin_users", "audit_logs", "calls", "system_settings", "tenants"}
LEGACY_SCHEMA_REVISION = "20260619_0001"
VALID_MIGRATION_MODES = {"auto", "upgrade", "check", "create_all", "skip"}


@dataclass(frozen=True)
class SchemaState:
    has_alembic_version: bool
    existing_core_tables: set[str]

    @property
    def has_legacy_schema(self) -> bool:
        return self.existing_core_tables == CORE_TABLES

    @property
    def has_partial_schema(self) -> bool:
        return bool(self.existing_core_tables) and not self.has_legacy_schema


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_alembic_config() -> Config:
    config = Config(str(_project_root() / "alembic.ini"))
    config.set_main_option("script_location", str(_project_root() / "alembic"))
    return config


def _head_revision() -> str:
    script = ScriptDirectory.from_config(get_alembic_config())
    return script.get_current_head()


async def _schema_state() -> SchemaState:
    def inspect_schema(sync_conn) -> SchemaState:
        inspector = inspect(sync_conn)
        tables = set(inspector.get_table_names())
        return SchemaState(
            has_alembic_version="alembic_version" in tables,
            existing_core_tables=tables.intersection(CORE_TABLES),
        )

    async with engine.connect() as conn:
        return await conn.run_sync(inspect_schema)


async def _current_revision() -> str | None:
    async with engine.connect() as conn:
        has_version_table = await conn.run_sync(
            lambda sync_conn: inspect(sync_conn).has_table("alembic_version")
        )
        if not has_version_table:
            return None

        rows = (
            (await conn.execute(text("SELECT version_num FROM alembic_version")))
            .scalars()
            .all()
        )
        if not rows:
            return None
        if len(rows) > 1:
            raise DatabaseInitializationError(
                "Database has multiple Alembic heads. Merge migrations before startup."
            )
        return rows[0]


async def _run_alembic_upgrade() -> None:
    await asyncio.to_thread(command.upgrade, get_alembic_config(), "head")


async def _stamp_alembic_revision(revision: str) -> None:
    await asyncio.to_thread(command.stamp, get_alembic_config(), revision)


async def _create_all_development() -> None:
    if settings.is_production:
        raise DatabaseInitializationError(
            "DATABASE_MIGRATION_MODE=create_all is not allowed in production."
        )

    logger.warning(
        "Creating tables with SQLAlchemy metadata because "
        "DATABASE_MIGRATION_MODE=create_all. Prefer Alembic migrations."
    )
    _register_orm_models()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def _prepare_legacy_schema_for_alembic(*, allow_stamp: bool) -> None:
    state = await _schema_state()
    if state.has_alembic_version:
        return

    if state.has_legacy_schema:
        if not allow_stamp:
            raise DatabaseInitializationError(
                "Existing tables were found without alembic_version. Run "
                f"`alembic stamp {LEGACY_SCHEMA_REVISION}` once after verifying "
                "the schema matches the initial migration, then run "
                "`alembic upgrade head` before starting the app again."
            )
        logger.warning(
            "Existing tables were found without alembic_version; stamping the "
            "current schema as the initial Alembic revision. Pending migrations "
            "will run next."
        )
        await _stamp_alembic_revision(LEGACY_SCHEMA_REVISION)
        return

    if state.has_partial_schema:
        raise DatabaseInitializationError(
            "Partial database schema exists without alembic_version. Back up the "
            "database, reconcile the schema manually, then run the appropriate "
            "`alembic stamp <revision>` or `alembic upgrade head` command."
        )


def _resolved_migration_mode() -> str:
    mode = settings.database_migration_mode.strip().lower()
    if mode not in VALID_MIGRATION_MODES:
        raise DatabaseInitializationError(
            "DATABASE_MIGRATION_MODE must be one of: "
            f"{', '.join(sorted(VALID_MIGRATION_MODES))}."
        )
    if mode == "auto":
        return "check" if settings.is_production else "upgrade"
    return mode


async def initialize_database_schema() -> None:
    """Initialize or validate the database schema using Alembic."""
    mode = _resolved_migration_mode()

    if mode == "skip":
        logger.warning("Skipping database migration checks by configuration.")
        return

    if mode == "create_all":
        await _create_all_development()
        return

    if mode == "upgrade":
        await _prepare_legacy_schema_for_alembic(allow_stamp=True)
        await _run_alembic_upgrade()
        logger.info("Database migrations are at Alembic head")
        return

    await _prepare_legacy_schema_for_alembic(allow_stamp=False)
    current_revision = await _current_revision()
    head_revision = _head_revision()
    if current_revision != head_revision:
        raise DatabaseInitializationError(
            "Database schema is not at the Alembic head "
            f"(current={current_revision or '<none>'}, head={head_revision}). "
            "Run `alembic upgrade head` before starting the production app, or set "
            "DATABASE_MIGRATION_MODE=upgrade if startup migrations are intentional."
        )

    logger.info("Database schema is at Alembic head")
