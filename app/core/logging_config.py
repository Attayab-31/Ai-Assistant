"""
app/core/logging_config.py — Production-ready logging configuration.

Provides structured logging setup with:
- Per-module log level control
- Suppressed verbose library logs
- JSON formatting for log aggregation
- Voice-call context (call_id, phase, provider) in dev and prod
"""

import json
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.core.call_logging import (
    VoiceTraceFilter,
    format_call_prefix,
    voice_context_from_record,
)
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

        voice_ctx = voice_context_from_record(record)
        if voice_ctx:
            log_obj["voice"] = voice_ctx

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
    DIM = "\033[2m"
    PHASE = "\033[96m"  # Bright cyan for voice context

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        reset = self.COLORS["RESET"]

        prefix = format_call_prefix(record)
        prefix_fmt = ""
        if prefix:
            prefix_fmt = f"{self.PHASE}{prefix}{reset}"

        # Format: TIMESTAMP | LEVEL | LOGGER | [call|phase] MESSAGE
        base = (
            f"{self.formatTime(record, '%Y-%m-%d %H:%M:%S')} | "
            f"{color}{record.levelname:8}{reset} | "
            f"{record.name} | "
            f"{prefix_fmt}{record.getMessage()}"
        )

        # Append compact detail fields when present (latency, reason, etc.)
        detail_parts: list[str] = []
        for key in ("latency_ms", "reason", "timeout_s", "budget_s", "detail"):
            val = getattr(record, key, None)
            if val is not None and val != "":
                detail_parts.append(f"{key}={val}")
        if detail_parts:
            base += f" {self.DIM}({', '.join(detail_parts)}){reset}"

        if record.exc_info:
            base += f"\n{self.formatException(record.exc_info)}"

        return base


class VoiceTraceFormatter(logging.Formatter):
    """Plain trace file — no ANSI colors, includes voice prefix."""

    def format(self, record: logging.LogRecord) -> str:
        prefix = format_call_prefix(record)
        base = (
            f"{self.formatTime(record, '%Y-%m-%d %H:%M:%S')} | "
            f"{record.levelname:8} | "
            f"{record.name} | "
            f"{prefix}{record.getMessage()}"
        )
        if record.exc_info:
            base += f"\n{self.formatException(record.exc_info)}"
        return base


def setup_logging() -> None:
    """Configure production-ready logging."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    is_production = settings.is_production
    use_json = is_production

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    except (AttributeError, ValueError):
        pass

    console_handler = logging.StreamHandler(sys.stdout)
    console_level = logging.INFO if is_production else logging.DEBUG
    console_handler.setLevel(console_level)
    console_handler.setFormatter(JsonFormatter() if use_json else TextFormatter())
    root.addHandler(console_handler)

    log_dir = Path(settings.log_dir)

    if is_production:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            filename=log_dir / "app.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(JsonFormatter())
        root.addHandler(file_handler)

    # Dev voice trace: grep-friendly file with only call pipeline lines.
    if not is_production and settings.log_voice_trace:
        log_dir.mkdir(parents=True, exist_ok=True)
        trace_handler = RotatingFileHandler(
            filename=log_dir / "voice.trace.log",
            maxBytes=20 * 1024 * 1024,
            backupCount=3,
        )
        trace_handler.setLevel(logging.DEBUG)
        trace_handler.setFormatter(VoiceTraceFormatter())
        trace_handler.addFilter(VoiceTraceFilter())
        root.addHandler(trace_handler)

    library_levels = {
        "fastapi": logging.WARNING if is_production else logging.INFO,
        "starlette": logging.WARNING if is_production else logging.INFO,
        "uvicorn": logging.INFO if is_production else logging.DEBUG,
        "uvicorn.access": logging.WARNING,
        "sqlalchemy": logging.WARNING,
        "sqlalchemy.engine": logging.WARNING,
        "sqlalchemy.engine.Engine": logging.WARNING,
        "sqlalchemy.pool": logging.WARNING,
        "sqlalchemy.orm": logging.WARNING,
        "alembic": logging.INFO,
        "alembic.runtime": logging.INFO,
        "alembic.runtime.migration": logging.INFO,
        "httpx": logging.WARNING,
        "httpcore": logging.WARNING,
        "websockets": logging.WARNING,
        "websocket": logging.WARNING,
        "redis": logging.WARNING,
        "aioredis": logging.WARNING,
        "urllib3": logging.WARNING,
        "requests": logging.WARNING,
        "boto3": logging.WARNING,
        "botocore": logging.WARNING,
    }

    for logger_name, level in library_levels.items():
        logging.getLogger(logger_name).setLevel(level)

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

    logger = logging.getLogger("app")
    logger.info(
        "Logging configured — environment=%s json=%s voice_trace=%s log_dir=%s",
        settings.environment,
        use_json,
        settings.log_voice_trace and not is_production,
        log_dir,
    )
