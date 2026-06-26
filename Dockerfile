# syntax=docker/dockerfile:1
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# ffmpeg: required by pydub for non-WAV audio conversion.
# curl: used by the container healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

RUN chmod +x /app/docker-entrypoint.sh

EXPOSE 8000

# The entrypoint dispatches on the first argument: web | worker | beat | migrate.
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["web"]
