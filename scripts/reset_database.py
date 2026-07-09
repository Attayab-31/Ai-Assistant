"""
Wipe all application data and re-seed defaults (admin + settings).

Preserves ``alembic_version`` so the schema/migration history stays intact.
Run: python scripts/reset_database.py

Refuses to run against a production ``ENVIRONMENT`` unless ``--force`` is passed.
"""

import argparse
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Wipe app data and re-seed defaults")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Required when ENVIRONMENT=production",
    )
    args = parser.parse_args()

    from config import settings

    if settings.is_production and not args.force:
        logger.error(
            "Refusing to reset database in production. "
            "Pass --force if you really intend to wipe all data."
        )
        raise SystemExit(1)
    if settings.is_production:
        logger.warning("DESTRUCTIVE: wiping production database tables.")

    asyncio.run(reset_database())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
