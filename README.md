# AI Tenant Screener

An AI-powered phone screening platform for rental applicants. It answers calls
over Telnyx, runs a deterministic screening conversation (speech-to-text →
LLM → text-to-speech), scores applicants, stores results, and notifies the
leasing team. It ships with an admin dashboard for configuration, live call
monitoring, analytics, and a browser-based test console.

- **Web framework:** FastAPI
- **Database:** PostgreSQL / Supabase (async SQLAlchemy + Alembic)
- **Background jobs:** Celery + Redis
- **Voice:** Telnyx (PCMU/RTP), Deepgram (STT), Groq/OpenAI/OpenRouter (LLM),
  Deepgram/Google (TTS)

---

## Architecture

The system runs as **three processes** that share the same codebase:

| Process | Command (role) | Purpose |
|---------|----------------|---------|
| Web | `uvicorn main:app` | HTTP API, admin UI, Telnyx webhooks + media WebSocket |
| Celery worker | `celery -A app.core.celery_app.celery_app worker` | Emails, CRM webhooks, daily digest |
| Celery beat | `celery -A app.core.celery_app.celery_app beat` | Scheduled jobs (provider health, digest) |

> **Single web worker:** live call sessions are held in memory per process.
> Run **one** uvicorn worker (or use a sticky load balancer) until a shared
> session store is added. See `app/core/call_handler.py`.

Redis uses three logical databases: `/0` cache, `/1` Celery broker,
`/2` Celery results.

### Redis usage & memory

Live calls do **not** use Redis — sessions, audio, and Deepgram sockets are
in-process. Redis is used only for small, TTL-bounded things, so a 256 MB
instance is far more than enough (real usage is single-digit MB):

| Use | DB | Bound |
|-----|----|-------|
| Rate-limit counters (login/signup) | `/0` | tiny, short TTL |
| Per-call settings snapshot cache | `/0` | 1 key, 30 s TTL |
| Analytics cache | `/0` | few keys, 5 min TTL |
| Webhook idempotency (`call.initiated`) | `/0` | 1 key/call, 1 h TTL |
| Celery broker (task queue) | `/1` | transient (drained by worker) |
| Celery results | `/2` | disabled (`task_ignore_result`) |

**Required eviction policy:** because `/1` is a task broker, do **not** let
Redis evict broker messages. Configure the instance with `noeviction`
(recommended) or `volatile-lru` (evicts only keys that already have a TTL —
all of ours do, broker messages don't). Avoid `allkeys-lru`/`allkeys-random`,
which can silently drop queued emails/CRM webhooks.

```bash
# self-hosted redis.conf
maxmemory 256mb
maxmemory-policy noeviction
```

On managed Redis (Upstash, Redis Cloud, etc.), set the eviction policy to
`noeviction` in the dashboard.

---

## Quick start (local, without Docker)

Requirements: Python 3.13, Redis, a PostgreSQL/Supabase database, and `ffmpeg`
on your PATH (for audio conversion).

```bash
python -m venv venv
# Windows:  .\venv\Scripts\activate
# Unix:     source venv/bin/activate

pip install -r requirements.txt

cp .env.example .env        # then edit .env with real values

alembic upgrade head        # apply database migrations
```

Run each process in its own terminal:

```bash
# 1) Web
uvicorn main:app --reload --host 127.0.0.1 --port 8000

# 2) Celery worker
celery -A app.core.celery_app.celery_app worker --loglevel=info

# 3) Celery beat
celery -A app.core.celery_app.celery_app beat --loglevel=info
```

Open the admin panel at <http://127.0.0.1:8000/admin/dashboard>.

On a fresh database, a `super_admin` is seeded from `ADMIN_EMAIL` /
`ADMIN_PASSWORD`. In **development** you may instead create the first account
via the signup page; in **production** signup always requires an existing
super admin.

---

## Quick start (Docker Compose)

This brings up the web app, Celery worker, Celery beat, and Redis. Your
`DATABASE_URL` must point to a reachable Postgres/Supabase instance (Compose
does not run a database).

```bash
cp .env.example .env        # fill in real values
docker compose up --build
```

The web container runs `alembic upgrade head` automatically on start.

---

## Configuration

All settings load from environment variables (see `.env.example` for the full
list and inline guidance). Key variables:

| Variable | Notes |
|----------|-------|
| `SECRET_KEY` | Signs JWTs and stream tokens. **Required strong value in prod.** |
| `ENCRYPTION_KEY` | Fernet key for encrypting stored API keys. **Required in prod.** Set before saving any keys in the admin UI. |
| `ADMIN_EMAIL` / `ADMIN_PASSWORD` | Seeds the first super admin. Change the password before deploying. |
| `APP_URL` | Public base URL. **Must be `https://` in prod.** Used for CORS and the Telnyx media WebSocket URL. |
| `ENVIRONMENT` | `development` or `production`. |
| `DATABASE_URL` | Postgres/Supabase connection string (URL-encode the password). |
| `DATABASE_MIGRATION_MODE` | Prod defaults to `check` (run `alembic upgrade head` first). |
| `REDIS_URL`, `CELERY_BROKER_URL`, `CELERY_RESULT_BACKEND` | Redis DBs `/0`, `/1`, `/2`. |
| `TELNYX_API_KEY`, `TELNYX_PUBLIC_KEY`, `TELNYX_PHONE_NUMBER` | `TELNYX_PUBLIC_KEY` is **required in prod** to verify webhooks. |

### Generating secrets

```bash
# SECRET_KEY
python -c "import secrets; print(secrets.token_urlsafe(48))"

# ENCRYPTION_KEY (valid Fernet key)
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

### Production startup guard

When `ENVIRONMENT=production`, the app refuses to start if `SECRET_KEY`,
`ADMIN_PASSWORD`, `ENCRYPTION_KEY`, `APP_URL`, or `TELNYX_PUBLIC_KEY` are
missing, default, or invalid. See `Settings.validate_runtime_secrets()` in
`config.py`.

---

## Database migrations

```bash
alembic upgrade head            # apply latest
alembic revision -m "message"   # create a new migration
alembic downgrade -1            # roll back one
```

---

## Health check

`GET /health` returns:

- **200** when the database is reachable
- **503** when the database is down (so load balancers stop routing traffic)

The body also reports Redis and provider status.

---

## Telnyx setup

1. Buy/assign a number and point its **Voice webhook** to
   `https://your-domain.com/telnyx/webhook` (HTTPS required).
2. Set `TELNYX_API_KEY`, `TELNYX_PHONE_NUMBER`, and `TELNYX_PUBLIC_KEY`
   (the public key verifies webhook signatures).
3. Ensure `APP_URL` is your public HTTPS URL — the media stream WebSocket URL
   is derived from it.

---

## Go-live checklist

Before the first real call:

- [ ] `ENVIRONMENT=production` and `APP_URL=https://your-domain.com`
- [ ] Strong `SECRET_KEY` and valid `ENCRYPTION_KEY` set
- [ ] `ADMIN_PASSWORD` changed from the default
- [ ] `TELNYX_PUBLIC_KEY` set; webhook URL configured in Telnyx
- [ ] Provider API keys set (Deepgram, an LLM, a TTS provider, Resend)
- [ ] `alembic upgrade head` run against the production database
- [ ] All three processes running (web, worker, beat) with Redis reachable
- [ ] `GET /health` returns 200
- [ ] One end-to-end test call: transcript saved, applicant scored, email queued

---

## Project layout

```
main.py                  FastAPI app, middleware, /health, startup guard
config.py                Typed settings + production validation
alembic/                 Migrations
app/
  api/                   Routers: auth, admin, webhook, settings, test_console
  core/                  Call pipeline, conversation flow, Celery app, logging
  db/                    SQLAlchemy models access, CRUD, seeding, migrations
  models/ schemas/       ORM models and Pydantic schemas
  providers/             LLM / STT / TTS provider implementations
  services/              Telnyx, email, storage, CRM webhook tasks
  admin/templates/       Jinja2 admin UI
```

---

## Notes & known limitations

- Live call sessions are in-memory (single web worker — see above).
- Tests: `pytest` is included in `requirements.txt` but the suite is not yet
  populated.
