"""
Wipe all application data and re-seed defaults (admin + settings).

Preserves ``alembic_version`` so the schema/migration history stays intact.
Run: python scripts/reset_database.py
"""

import asyncio
import logging
import sys
from pathlib import Path

# Allow running from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text

from app.db.crud import seed_defaults
from app.db.database import AsyncSessionLocal, engine

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("reset_database")

APP_TABLES = (
    "audit_logs",
    "tenants",
    "calls",
    "system_settings",
    "admin_users",
)


async def reset_database() -> None:
    """Truncate all app tables and re-seed defaults."""
    from app.services.settings_cache import invalidate_settings_cache

    logger.info("Truncating application tables: %s", ", ".join(APP_TABLES))
    async with AsyncSessionLocal() as db:
        await db.execute(
            text(
                f"TRUNCATE TABLE {', '.join(APP_TABLES)} "
                "RESTART IDENTITY CASCADE"
            )
        )
        await db.commit()

    logger.info("Re-seeding defaults (admin user + system settings)...")
    async with AsyncSessionLocal() as db:
        await seed_defaults(db)

    await invalidate_settings_cache()
    await engine.dispose()
    logger.info("Database reset complete — fresh install ready.")


if __name__ == "__main__":
    asyncio.run(reset_database())
