# ─────────────────────────────────────────────────────────────
# Multi-stage Dockerfile for the Fraud Detection FastAPI service
# ─────────────────────────────────────────────────────────────

# Stage 1: Builder ─ install dependencies
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --upgrade pip setuptools wheel \
    && pip install --prefix=/install .

# Stage 2: Runtime ─ slim production image
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application source and required artifacts
COPY src ./src
COPY configs ./configs
COPY models ./models
COPY reports ./reports
COPY pyproject.toml ./

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "fraud_detection.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
