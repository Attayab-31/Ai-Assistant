#!/usr/bin/env python3
"""Create performance indexes on critical columns."""

import asyncio
import sys

from sqlalchemy import text

from app.db.database import engine


async def create_indexes():
    indexes = [
        (
            "idx_calls_created_at",
            "CREATE INDEX IF NOT EXISTS idx_calls_created_at ON calls(created_at) WHERE is_deleted = false",
        ),
        (
            "idx_calls_created_status",
            "CREATE INDEX IF NOT EXISTS idx_calls_created_status ON calls(created_at, status) WHERE is_deleted = false",
        ),
    ]

    async with engine.begin() as conn:
        for name, sql in indexes:
            try:
                await conn.execute(text(sql))
                print(f"Created index: {name}")
            except Exception as e:
                if "already exists" in str(e).lower():
                    print(f"Index already exists: {name}")
                else:
                    print(f"Error creating {name}: {e}")
                    return False

    print("\nAll performance indexes created successfully!")
    return True


if __name__ == "__main__":
    success = asyncio.run(create_indexes())
    sys.exit(0 if success else 1)
