"""FastAPI application for Credit Card Fraud Detection.

Serves fraud predictions with Prometheus metrics instrumentation.
Loads model from local artifact with configuration from configs/inference.yaml.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

from src.app.schemas import (
    BatchPredictionResponse,
    BatchTransactionRequest,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    TransactionRequest,
)

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_model = None
_inference_cfg: dict = {}

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

REQUEST_COUNT = Counter(
    "fraud_api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status"],
)

PREDICTION_COUNT = Counter(
    "fraud_api_predictions_total",
    "Total predictions made",
    ["result"],
)

REQUEST_LATENCY = Histogram(
    "fraud_api_request_duration_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

MODEL_LOADED = Gauge(
    "fraud_api_model_loaded",
    "Whether a model is currently loaded (1=yes, 0=no)",
)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _load_inference_config() -> dict:
    """Load inference config from YAML."""
    cfg_path = PROJECT_ROOT / "configs" / "inference.yaml"
    if not cfg_path.exists():
        return {
            "threshold": 0.5,
            "model_name": "unknown",
            "model_path": "models/trained/latest.joblib",
            "feature_columns": [f"V{i}" for i in range(1, 29)] + ["Amount"],
            "selected_at": "N/A",
        }
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_model() -> None:
    """Load model from local artifact path specified in inference config."""
    global _model, _inference_cfg

    _inference_cfg = _load_inference_config()
    model_path = PROJECT_ROOT / str(
        _inference_cfg.get("model_path", "models/trained/latest.joblib")
    )

    if not model_path.exists():
        # Fallback to root-level model.joblib
        model_path = PROJECT_ROOT / "model.joblib"

    if not model_path.exists():
        print(f"WARNING: No model found at {model_path}")
        _model = None
        MODEL_LOADED.set(0)
        return

    _model = joblib.load(model_path)
    MODEL_LOADED.set(1)
    print(f"Model loaded from {model_path}")


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    _load_model()
    yield


app = FastAPI(
    title="Credit Card Fraud Detection API",
    description=(
        "Real-time fraud detection API for credit card transactions. "
        "Part of the MLOps Credit Fraud Detection project."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check service health and model availability."""
    REQUEST_COUNT.labels(method="GET", endpoint="/health", status="200").inc()
    return HealthResponse(
        status="healthy",
        model_loaded=_model is not None,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(transaction: TransactionRequest):
    """Predict fraud probability for a single transaction.

    Expects exactly 29 features: V1-V28 + Amount.
    """
    start_time = time.perf_counter()

    if _model is None:
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", status="503").inc()
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Call /reload or ensure model artifact exists.",
        )

    try:
        features_array = np.array(transaction.features).reshape(1, -1)
        proba = float(_model.predict_proba(features_array)[0, 1])
        threshold = float(_inference_cfg.get("threshold", 0.5))
        is_fraud = proba >= threshold

        # Track prediction result
        result_label = "fraud" if is_fraud else "legitimate"
        PREDICTION_COUNT.labels(result=result_label).inc()
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", status="200").inc()

        duration = time.perf_counter() - start_time
        REQUEST_LATENCY.labels(endpoint="/predict").observe(duration)

        return PredictionResponse(
            fraud_probability=round(proba, 6),
            is_fraud=is_fraud,
            threshold=threshold,
        )

    except Exception as e:
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", status="500").inc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Prediction"],
)
async def predict_batch(batch: BatchTransactionRequest):
    """Predict fraud probability for a batch of transactions (max 1000)."""
    start_time = time.perf_counter()

    if _model is None:
        REQUEST_COUNT.labels(method="POST", endpoint="/predict/batch", status="503").inc()
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Call /reload or ensure model artifact exists.",
        )

    try:
        features_matrix = np.array([tx.features for tx in batch.transactions])
        probas = _model.predict_proba(features_matrix)[:, 1]
        threshold = float(_inference_cfg.get("threshold", 0.5))

        predictions = []
        for proba in probas:
            is_fraud = float(proba) >= threshold
            result_label = "fraud" if is_fraud else "legitimate"
            PREDICTION_COUNT.labels(result=result_label).inc()
            predictions.append(
                PredictionResponse(
                    fraud_probability=round(float(proba), 6),
                    is_fraud=is_fraud,
                    threshold=threshold,
                )
            )

        REQUEST_COUNT.labels(method="POST", endpoint="/predict/batch", status="200").inc()
        duration = time.perf_counter() - start_time
        REQUEST_LATENCY.labels(endpoint="/predict/batch").observe(duration)

        return BatchPredictionResponse(
            predictions=predictions,
            count=len(predictions),
        )

    except Exception as e:
        REQUEST_COUNT.labels(method="POST", endpoint="/predict/batch", status="500").inc()
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.get("/model_info", response_model=ModelInfoResponse, tags=["System"])
async def model_info():
    """Return metadata about the currently loaded model."""
    REQUEST_COUNT.labels(method="GET", endpoint="/model_info", status="200").inc()

    if not _inference_cfg:
        raise HTTPException(status_code=503, detail="No model configuration loaded.")

    return ModelInfoResponse(
        model_name=str(_inference_cfg.get("model_name", "unknown")),
        model_path=str(_inference_cfg.get("model_path", "N/A")),
        threshold=float(_inference_cfg.get("threshold", 0.5)),
        feature_columns=_inference_cfg.get(
            "feature_columns",
            [f"V{i}" for i in range(1, 29)] + ["Amount"],
        ),
        selected_at=str(_inference_cfg.get("selected_at", "N/A")),
    )


@app.post("/reload", tags=["System"])
async def reload_model():
    """Reload the model from disk (hot-reload without restart)."""
    REQUEST_COUNT.labels(method="POST", endpoint="/reload", status="200").inc()
    try:
        _load_model()
        return {"status": "reloaded", "model_loaded": _model is not None}
    except Exception as e:
        REQUEST_COUNT.labels(method="POST", endpoint="/reload", status="500").inc()
        raise HTTPException(status_code=500, detail=f"Reload failed: {str(e)}")


@app.get("/metrics", tags=["System"])
async def metrics():
    """Expose Prometheus-compatible metrics."""
    return PlainTextResponse(
        content=generate_latest().decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )
