"""FastAPI application for the Fraud Detection system.

Endpoints:
- GET  /health          — Liveness / readiness probe
- GET  /api/v1/model    — Model metadata
- GET  /api/v1/drift    — Latest drift report
- POST /api/v1/predict  — Score a single transaction
- POST /api/v1/predict/batch — Score multiple transactions
- GET  /metrics         — Prometheus metrics
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
)
from starlette.responses import Response

from fraud_detection.api.schemas import (
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    Transaction,
)
from fraud_detection.api.service import FraudDetectionService

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
PREDICTION_COUNT = Counter(
    "fraud_api_predictions_total",
    "Total number of predictions served",
    ["outcome"],
)
PREDICTION_LATENCY = Histogram(
    "fraud_api_prediction_latency_seconds",
    "Prediction latency in seconds",
)
MODEL_LOADED = Counter(
    "fraud_api_model_loaded",
    "1 if model is loaded successfully",
)

# ---------------------------------------------------------------------------
# Global service instance
# ---------------------------------------------------------------------------
service = FraudDetectionService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup."""
    try:
        service.load_model()
        MODEL_LOADED.inc()
        logger.info("Model loaded successfully at startup")
    except Exception:
        logger.exception("Failed to load model at startup")
    yield


app = FastAPI(
    title="Fraud Detection API",
    description=(
        "REST API for credit card fraud detection. "
        "Serves the champion model from the MLOps pipeline."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Health & metadata
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    """Liveness / readiness check for Kubernetes probes."""
    return HealthResponse(
        status="ok" if service.model is not None else "degraded",
        model_loaded=service.model is not None,
        model_name=service.model_name,
        version=service.model_version,
    )


@app.get("/api/v1/model", response_model=ModelInfoResponse, tags=["metadata"])
async def model_info() -> ModelInfoResponse:
    """Return metadata about the currently loaded model."""
    info = service.get_model_info()
    return ModelInfoResponse(**info)


@app.get("/api/v1/drift", tags=["monitoring"])
async def drift_report():
    """Return the latest drift detection report."""
    return service.get_drift_report()


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


@app.post("/api/v1/predict", response_model=PredictionResponse, tags=["prediction"])
async def predict(transaction: Transaction) -> PredictionResponse:
    """Score a single transaction for fraud probability."""
    if service.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.perf_counter()
    try:
        result = service.predict(transaction.model_dump())
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        PREDICTION_LATENCY.observe(time.perf_counter() - start)

    outcome = "fraud" if result["is_fraud"] else "legit"
    PREDICTION_COUNT.labels(outcome=outcome).inc()

    return PredictionResponse(**result)


@app.post(
    "/api/v1/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["prediction"],
)
async def predict_batch(
    transactions: list[Transaction],
) -> BatchPredictionResponse:
    """Score a batch of transactions."""
    if service.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    results = service.predict_batch([t.model_dump() for t in transactions])
    predictions = [PredictionResponse(**r) for r in results]
    return BatchPredictionResponse(predictions=predictions, total=len(predictions))


# ---------------------------------------------------------------------------
# Prometheus
# ---------------------------------------------------------------------------


@app.get("/metrics", tags=["ops"])
async def metrics():
    """Expose Prometheus metrics."""
    return Response(content=generate_latest(), media_type="text/plain")
