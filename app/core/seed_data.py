"""Install-time seed data loaded from JSON — not used as a live-call fallback.

Fresh databases and the admin "reset to defaults" actions read these files.
Runtime call flow always uses ``screening_questions`` / ``screening_faqs`` from
the database (frozen per call at session start).
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _read_json(name: str) -> list[dict[str, Any]]:
    path = _DATA_DIR / name
    if not path.is_file():
        logger.error("Seed file missing: %s", path)
        return []
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"{name} must contain a JSON array")
    return data


@lru_cache(maxsize=1)
def load_seed_questions() -> list[dict[str, Any]]:
    """Default screening questions for DB seed / admin reset only."""
    from app.core.question_flow import validate_questions_for_save

    raw = _read_json("seed_questions.json")
    if not raw:
        return []
    return validate_questions_for_save(raw)


@lru_cache(maxsize=1)
def load_seed_faqs() -> list[dict[str, Any]]:
    """Default FAQ entries for DB seed / admin reset only."""
    from app.core.screening_flow import validate_faqs_for_save

    raw = _read_json("seed_faqs.json")
    if not raw:
        return []
    return validate_faqs_for_save(raw)
