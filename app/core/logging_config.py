"""
app/core/logging_config.py — Production-ready logging configuration.

Provides structured logging setup with:
- Per-module log level control
- Suppressed verbose library logs
- JSON formatting for log aggregation
- Request ID correlation tracking
"""

import json
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from config import settings


class JsonFormatter(logging.Formatter):
    """Format logs as JSON for production log aggregation."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        if hasattr(record, "request_id"):
            log_obj["request_id"] = record.request_id

        return json.dumps(log_obj)


class TextFormatter(logging.Formatter):
    """Clean text format for console output (development)."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        reset = self.COLORS["RESET"]

        # Format: TIMESTAMP | LEVEL | LOGGER | MESSAGE
        base = (
            f"{self.formatTime(record, '%Y-%m-%d %H:%M:%S')} | "
            f"{color}{record.levelname:8}{reset} | "
            f"{record.name} | "
            f"{record.getMessage()}"
        )

        if record.exc_info:
            base += f"\n{self.formatException(record.exc_info)}"

        return base


def setup_logging() -> None:
    """Configure production-ready logging."""
    # Root logger
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Remove existing handlers
    root.handlers.clear()

    # Determine if production
    is_production = settings.is_production
    use_json = is_production

    # 1. Console handler (always)
    # Force UTF-8 so non-ASCII log lines (em-dashes, arrows, etc.) never crash
    # the handler on consoles that default to a legacy codepage (e.g. Windows cp1252).
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    except (AttributeError, ValueError):
        pass
    console_handler = logging.StreamHandler(sys.stdout)
    console_level = logging.INFO if is_production else logging.DEBUG
    console_handler.setLevel(console_level)
    console_handler.setFormatter(JsonFormatter() if use_json else TextFormatter())
    root.addHandler(console_handler)

    # 2. File handler (production only, with rotation)
    if is_production:
        log_dir = Path(
            settings.log_dir if hasattr(settings, "log_dir") else "/var/log/ai-screener"
        )
        log_dir.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            filename=log_dir / "app.log",
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(JsonFormatter())
        root.addHandler(file_handler)

    # 3. Per-library log levels (suppress verbose libraries)
    library_levels = {
        # FastAPI/Starlette
        "fastapi": logging.WARNING if is_production else logging.INFO,
        "starlette": logging.WARNING if is_production else logging.INFO,
        "uvicorn": logging.INFO if is_production else logging.DEBUG,
        "uvicorn.access": logging.WARNING,
        # SQLAlchemy - CRITICAL: suppress verbose logs
        "sqlalchemy": logging.WARNING,
        "sqlalchemy.engine": logging.WARNING,  # Suppresses SQL query logging
        "sqlalchemy.engine.Engine": logging.WARNING,
        "sqlalchemy.pool": logging.WARNING,
        "sqlalchemy.orm": logging.WARNING,
        # Alembic migrations
        "alembic": logging.INFO,
        "alembic.runtime": logging.INFO,
        "alembic.runtime.migration": logging.INFO,
        # HTTP clients
        "httpx": logging.WARNING,
        "httpcore": logging.WARNING,
        # Websockets
        "websockets": logging.WARNING,
        "websocket": logging.WARNING,
        # Redis
        "redis": logging.WARNING,
        "aioredis": logging.WARNING,
        # Other utilities
        "urllib3": logging.WARNING,
        "requests": logging.WARNING,
        "boto3": logging.WARNING,
        "botocore": logging.WARNING,
    }

    for logger_name, level in library_levels.items():
        logging.getLogger(logger_name).setLevel(level)

    # 4. Application loggers (should remain at DEBUG/INFO)
    app_loggers = [
        "app",
        "app.core",
        "app.api",
        "app.db",
        "app.providers",
        "app.services",
        "app.models",
        "app.utils",
        "config",
        "main",
    ]

    for logger_name in app_loggers:
        logging.getLogger(logger_name).setLevel(logging.DEBUG)

    # Log startup info
    logger = logging.getLogger("app")
    logger.info(
        "Logging configured - environment=%s, json=%s", settings.environment, use_json
    )
