"""Celery application configuration for background tasks."""

from celery import Celery
from celery.schedules import crontab

from config import celery_redis_ssl_options, settings

celery_app = Celery(
    "ai_tenant_screener",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=[
        "app.services.email_service",
        "app.services.retention_service",
    ],
)

_ssl = celery_redis_ssl_options()
_celery_conf = dict(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    broker_connection_retry_on_startup=True,
    # ── Memory hygiene (keeps Redis well under small plans like 256 MB) ──
    # These tasks are fire-and-forget (dispatched with .delay(), never awaited),
    # so we don't persist results at all. Anything that does slip through
    # expires quickly. Without this the result backend (Redis DB /2) would
    # accumulate one entry per task for a full day by default.
    task_ignore_result=True,
    result_expires=3600,
    # Cap how long an unacked task stays "invisible" before redelivery and
    # recycle workers periodically so memory can't creep.
    broker_transport_options={"visibility_timeout": 3600},
    worker_max_tasks_per_child=200,
    beat_schedule={
        "provider-health-check-every-5-minutes": {
            "task": "app.services.email_service.provider_health_check_task",
            "schedule": 300.0,
        },
        "daily-screening-digest": {
            "task": "app.services.email_service.send_daily_digest_task",
            "schedule": 86400.0,
        },
        "daily-data-retention-cleanup": {
            "task": "app.services.retention_service.purge_expired_data_task",
            "schedule": crontab(hour=3, minute=0),
        },
    },
)
if _ssl:
    _celery_conf["broker_use_ssl"] = _ssl
    _celery_conf["redis_backend_use_ssl"] = _ssl

celery_app.conf.update(_celery_conf)
