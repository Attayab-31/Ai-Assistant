"""
Reset application database data.

Scopes:
  operational — wipe calls, applicants (tenants), audit logs, and analytics cache.
                Keeps admin users, system settings, and schema/migrations.
  all         — wipe everything and re-seed defaults (admin + settings).

Preserves ``alembic_version`` so the schema/migration history stays intact.

Examples:
  python scripts/reset_database.py --scope operational
  python scripts/reset_database.py --scope all
  python scripts/reset_database.py --scope operational --force   # production

Refuses to run against a production ``ENVIRONMENT`` unless ``--force`` is passed.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

# Allow running from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text

from app.db.crud import seed_defaults
from app.db.database import AsyncSessionLocal, engine

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("reset_database")

ALL_APP_TABLES = (
    "audit_logs",
    "tenants",
    "calls",
    "system_settings",
    "admin_users",
)

OPERATIONAL_TABLES = (
    "audit_logs",
    "calls",  # CASCADE removes linked tenants (applicants)
)


async def invalidate_analytics_cache() -> None:
    """Clear cached analytics snapshots (analytics are computed from calls)."""
    from app.core.redis_client import cache_delete, close_redis

    today = datetime.now(UTC).date()
    keys = [f"analytics:{days}:{today}" for days in (7, 30, 90)]
    await cache_delete(*keys)
    await close_redis()
    logger.info("Cleared analytics cache keys: %s", ", ".join(keys))


async def reset_operational_data() -> None:
    """Remove screening history while keeping admin accounts and settings."""
    await invalidate_analytics_cache()

    logger.info(
        "Truncating operational tables: %s",
        ", ".join(OPERATIONAL_TABLES),
    )
    async with AsyncSessionLocal() as db:
        await db.execute(
            text(
                f"TRUNCATE TABLE {', '.join(OPERATIONAL_TABLES)} "
                "RESTART IDENTITY CASCADE"
            )
        )
        await db.commit()

    await engine.dispose()
    logger.info(
        "Operational reset complete — calls, applicants, audits, and analytics cleared."
    )


async def reset_all_data() -> None:
    """Truncate all app tables and re-seed defaults."""
    from app.services.settings_cache import invalidate_settings_cache

    await invalidate_analytics_cache()

    logger.info("Truncating application tables: %s", ", ".join(ALL_APP_TABLES))
    async with AsyncSessionLocal() as db:
        await db.execute(
            text(
                f"TRUNCATE TABLE {', '.join(ALL_APP_TABLES)} "
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
    parser = argparse.ArgumentParser(description="Reset app database data")
    parser.add_argument(
        "--scope",
        choices=("operational", "all"),
        default="all",
        help=(
            "operational = calls, applicants, audit logs, analytics cache only; "
            "all = wipe everything and re-seed defaults (default)"
        ),
    )
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
            "Pass --force if you really intend to wipe data."
        )
        raise SystemExit(1)
    if settings.is_production:
        if args.scope == "operational":
            logger.warning(
                "DESTRUCTIVE: wiping production calls, applicants, audits, and analytics."
            )
        else:
            logger.warning("DESTRUCTIVE: wiping all production database tables.")

    if args.scope == "operational":
        asyncio.run(reset_operational_data())
    else:
        asyncio.run(reset_all_data())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
