"""Pydantic schemas for the Fraud Detection API.

Defines request/response models aligned with the Kaggle Credit Card
Fraud Detection dataset (fraudTrain.csv / fraudTest.csv).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Transaction(BaseModel):
    """A single credit-card transaction submitted for fraud scoring.

    Field names mirror the raw CSV columns **except** ``is_fraud``
    (the target the API predicts) and ``Unnamed: 0`` (row index).
    """

    cc_num: str = Field(..., description="Credit card number (string)")
    merchant: str = Field(..., description="Merchant name")
    category: str = Field(..., description="Transaction category")
    amt: float = Field(..., ge=0, description="Transaction amount in USD")
    first: str = Field(..., description="Cardholder first name")
    last: str = Field(..., description="Cardholder last name")
    gender: str = Field(..., pattern=r"^[FM]$", description="Gender: F or M")
    street: str = Field(..., description="Street address")
    city: str = Field(..., description="City name")
    state: str = Field(..., pattern=r"^[A-Z]{2}$", description="US state code")
    zip: str = Field(..., description="ZIP code")
    lat: float = Field(..., ge=-90, le=90, description="Cardholder latitude")
    long: float = Field(..., ge=-180, le=180, description="Cardholder longitude")
    city_pop: int = Field(..., ge=0, description="City population")
    job: str = Field(..., description="Cardholder job title")
    dob: str = Field(..., description="Date of birth (YYYY-MM-DD)")
    trans_num: str = Field(..., description="Unique transaction hash")
    unix_time: int = Field(..., ge=0, description="Transaction UNIX timestamp")
    merch_lat: float = Field(..., ge=-90, le=90, description="Merchant latitude")
    merch_long: float = Field(..., ge=-180, le=180, description="Merchant longitude")


class PredictionResponse(BaseModel):
    """Response returned by the ``/predict`` endpoint."""

    fraud_probability: float = Field(..., description="Probability of fraud (0-1)")
    is_fraud: bool = Field(..., description="Binary fraud decision at threshold")
    threshold: float = Field(..., description="Decision threshold used")


class BatchPredictionResponse(BaseModel):
    """Wrapper for batch prediction results."""

    predictions: list[PredictionResponse]
    total: int


class HealthResponse(BaseModel):
    """Response from the ``/health`` liveness probe."""

    status: str = Field(..., description="Service status (ok / degraded)")
    model_loaded: bool = Field(..., description="Whether a model is loaded")
    model_name: str | None = Field(None, description="Name of loaded model")
    version: str | None = Field(None, description="Model version")


class ModelInfoResponse(BaseModel):
    """Metadata about the currently loaded model."""

    model_name: str
    version: str
    threshold: float
    feature_count: int
    trained_at: str | None = None
