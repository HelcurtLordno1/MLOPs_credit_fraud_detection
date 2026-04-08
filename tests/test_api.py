"""Tests for the FastAPI fraud detection API.

Uses FastAPI TestClient for synchronous endpoint testing.
Tests cover health check, prediction, batch prediction, model info,
reload, metrics, and error handling.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_model():
    """Create a mock sklearn model that returns deterministic probabilities."""
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.7, 0.3]])
    return model


@pytest.fixture()
def client(mock_model):
    """Create a TestClient with a mocked model pre-loaded."""
    # Patch the model and config before importing the app
    import src.app.api as api_module

    api_module._model = mock_model
    api_module._inference_cfg = {
        "threshold": 0.5,
        "model_name": "test_model",
        "model_path": "models/trained/latest.joblib",
        "feature_columns": [f"V{i}" for i in range(1, 29)] + ["Amount"],
        "selected_at": "2026-01-01T00:00:00+00:00",
    }

    from src.app.api import app

    with TestClient(app, raise_server_exceptions=False) as tc:
        yield tc

    # Clean up
    api_module._model = None
    api_module._inference_cfg = {}


@pytest.fixture()
def client_no_model():
    """Create a TestClient with no model loaded."""
    import src.app.api as api_module

    api_module._model = None
    api_module._inference_cfg = {}

    from src.app.api import app

    with TestClient(app, raise_server_exceptions=False) as tc:
        yield tc


@pytest.fixture()
def sample_features():
    """29 features matching V1-V28 + Amount."""
    return [0.0] * 29


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert data["status"] == "healthy"

    def test_health_model_loaded_true(self, client):
        data = client.get("/health").json()
        assert data["model_loaded"] is True

    def test_health_model_not_loaded(self, client_no_model):
        data = client_no_model.get("/health").json()
        assert data["model_loaded"] is False


# ---------------------------------------------------------------------------
# Predict endpoint
# ---------------------------------------------------------------------------


class TestPredictEndpoint:
    def test_predict_returns_200(self, client, sample_features):
        response = client.post("/predict", json={"features": sample_features})
        assert response.status_code == 200

    def test_predict_response_structure(self, client, sample_features):
        data = client.post("/predict", json={"features": sample_features}).json()
        assert "fraud_probability" in data
        assert "is_fraud" in data
        assert "threshold" in data

    def test_predict_probability_range(self, client, sample_features):
        data = client.post("/predict", json={"features": sample_features}).json()
        assert 0.0 <= data["fraud_probability"] <= 1.0

    def test_predict_threshold_applied(self, client, mock_model, sample_features):
        # Mock returns [0.7, 0.3], so fraud_probability = 0.3
        # With threshold 0.5, is_fraud should be False
        data = client.post("/predict", json={"features": sample_features}).json()
        assert data["is_fraud"] is False  # 0.3 < 0.5

    def test_predict_fraud_detected(self, client, mock_model, sample_features):
        # Override mock to return high fraud probability
        mock_model.predict_proba.return_value = np.array([[0.1, 0.9]])
        data = client.post("/predict", json={"features": sample_features}).json()
        assert data["is_fraud"] is True  # 0.9 >= 0.5

    def test_predict_wrong_feature_count_too_few(self, client):
        response = client.post("/predict", json={"features": [0.0] * 10})
        assert response.status_code == 422

    def test_predict_wrong_feature_count_too_many(self, client):
        response = client.post("/predict", json={"features": [0.0] * 50})
        assert response.status_code == 422

    def test_predict_empty_features(self, client):
        response = client.post("/predict", json={"features": []})
        assert response.status_code == 422

    def test_predict_missing_body(self, client):
        response = client.post("/predict")
        assert response.status_code == 422

    def test_predict_no_model_returns_503(self, client_no_model, sample_features):
        response = client_no_model.post("/predict", json={"features": sample_features})
        assert response.status_code == 503


# ---------------------------------------------------------------------------
# Batch predict endpoint
# ---------------------------------------------------------------------------


class TestBatchPredictEndpoint:
    def test_batch_predict_single(self, client, mock_model, sample_features):
        mock_model.predict_proba.return_value = np.array([[0.7, 0.3]])
        response = client.post(
            "/predict/batch",
            json={"transactions": [{"features": sample_features}]},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        assert len(data["predictions"]) == 1

    def test_batch_predict_multiple(self, client, mock_model, sample_features):
        mock_model.predict_proba.return_value = np.array(
            [[0.7, 0.3], [0.2, 0.8], [0.5, 0.5]]
        )
        response = client.post(
            "/predict/batch",
            json={"transactions": [{"features": sample_features}] * 3},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 3

    def test_batch_predict_no_model_returns_503(self, client_no_model, sample_features):
        response = client_no_model.post(
            "/predict/batch",
            json={"transactions": [{"features": sample_features}]},
        )
        assert response.status_code == 503


# ---------------------------------------------------------------------------
# Model info endpoint
# ---------------------------------------------------------------------------


class TestModelInfoEndpoint:
    def test_model_info_returns_200(self, client):
        response = client.get("/model_info")
        assert response.status_code == 200

    def test_model_info_structure(self, client):
        data = client.get("/model_info").json()
        assert "model_name" in data
        assert "model_path" in data
        assert "threshold" in data
        assert "feature_columns" in data
        assert "selected_at" in data

    def test_model_info_feature_count(self, client):
        data = client.get("/model_info").json()
        assert len(data["feature_columns"]) == 29


# ---------------------------------------------------------------------------
# Metrics endpoint
# ---------------------------------------------------------------------------


class TestMetricsEndpoint:
    def test_metrics_returns_200(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_contains_prometheus_format(self, client):
        response = client.get("/metrics")
        text = response.text
        # Prometheus format contains metric names
        assert "fraud_api_requests_total" in text or "HELP" in text

    def test_metrics_content_type(self, client):
        response = client.get("/metrics")
        assert "text/plain" in response.headers.get("content-type", "")
