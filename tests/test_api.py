"""Tests for the Fraud Detection API endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from fraud_detection.api.main import app, service
from fraud_detection.api.schemas import Transaction
from fraud_detection.ui_helpers import default_transaction_payload


@pytest.fixture(autouse=True)
def _load_model_if_available():
    """Attempt to load the model before tests; skip predict tests if unavailable."""
    try:
        if service.model is None:
            service.load_model()
    except Exception:
        pass
    yield


@pytest.fixture()
def client():
    """FastAPI TestClient."""
    return TestClient(app)


@pytest.fixture()
def sample_transaction() -> dict:
    """A valid sample transaction payload."""
    return default_transaction_payload()


class TestHealthEndpoint:
    def test_health_returns_200(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data

    def test_health_model_status(self, client: TestClient) -> None:
        response = client.get("/health")
        data = response.json()
        assert isinstance(data["model_loaded"], bool)


class TestModelEndpoint:
    def test_model_info_returns_200(self, client: TestClient) -> None:
        response = client.get("/api/v1/model")
        assert response.status_code == 200
        data = response.json()
        assert "model_name" in data
        assert "threshold" in data


class TestDriftEndpoint:
    def test_drift_returns_200(self, client: TestClient) -> None:
        response = client.get("/api/v1/drift")
        assert response.status_code == 200


class TestPredictEndpoint:
    def test_predict_returns_200(self, client: TestClient, sample_transaction: dict) -> None:
        if service.model is None:
            pytest.skip("Model not loaded")
        response = client.post("/api/v1/predict", json=sample_transaction)
        assert response.status_code == 200
        data = response.json()
        assert "fraud_probability" in data
        assert "is_fraud" in data
        assert "threshold" in data
        assert 0.0 <= data["fraud_probability"] <= 1.0

    def test_predict_batch_returns_200(self, client: TestClient, sample_transaction: dict) -> None:
        if service.model is None:
            pytest.skip("Model not loaded")
        response = client.post("/api/v1/predict/batch", json=[sample_transaction])
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert data["total"] == 1

    def test_predict_invalid_gender_returns_422(
        self, client: TestClient, sample_transaction: dict
    ) -> None:
        sample_transaction["gender"] = "X"
        response = client.post("/api/v1/predict", json=sample_transaction)
        assert response.status_code == 422


class TestMetricsEndpoint:
    def test_metrics_returns_200(self, client: TestClient) -> None:
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "fraud_api" in response.text


class TestTransactionSchema:
    def test_valid_transaction(self) -> None:
        payload = default_transaction_payload()
        txn = Transaction(**payload)
        assert txn.amt == payload["amt"]

    def test_invalid_state_rejected(self) -> None:
        payload = default_transaction_payload()
        payload["state"] = "XYZ"
        with pytest.raises(Exception):
            Transaction(**payload)
