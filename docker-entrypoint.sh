#!/bin/sh
set -e

# Single dispatch point for all container roles so one image serves the web
# app, the Celery worker, and the Celery beat scheduler.
ROLE="${1:-web}"

case "$ROLE" in
  web)
    echo "[entrypoint] Starting web server (schema init runs via app lifespan)..."
    # Single uvicorn worker: live call sessions are held in-memory per process.
    # To scale out, add a shared session store and a sticky load balancer.
    # Bind to $PORT when present (Render/Heroku inject it); default to 8000.
    exec uvicorn main:app --host 0.0.0.0 --port "${PORT:-8000}" \
      --proxy-headers --forwarded-allow-ips='*'
    ;;
  worker)
    echo "[entrypoint] Starting Celery worker (solo pool for async DB safety)..."
    exec celery -A app.core.celery_app.celery_app worker --pool=solo --loglevel=info
    ;;
  beat)
    echo "[entrypoint] Starting Celery beat scheduler..."
    exec celery -A app.core.celery_app.celery_app beat --loglevel=info
    ;;
  migrate)
    echo "[entrypoint] Running migrations only..."
    exec alembic upgrade head
    ;;
  *)
    # Fall through to an arbitrary command for debugging.
    exec "$@"
    ;;
esac
