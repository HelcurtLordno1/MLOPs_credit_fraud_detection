FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip \
    && pip install mlflow psycopg2-binary

COPY deployment/mlflow/entrypoint.sh /app/deployment/mlflow/entrypoint.sh
RUN chmod +x /app/deployment/mlflow/entrypoint.sh

EXPOSE 5000

CMD ["/app/deployment/mlflow/entrypoint.sh"]
