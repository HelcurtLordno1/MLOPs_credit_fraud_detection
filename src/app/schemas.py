"""Pydantic request/response schemas for the Fraud Detection API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    """Single transaction for fraud prediction.

    The `features` list must contain exactly 29 float values corresponding to
    the model's feature columns: V1-V28 and Amount (in that order).
    """

    features: list[float] = Field(
        ...,
        min_length=29,
        max_length=29,
        description="29 feature values: V1-V28 + Amount",
        json_schema_extra={"example": [0.0] * 29},
    )


class PredictionResponse(BaseModel):
    """Response from the /predict endpoint."""

    fraud_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Predicted probability of fraud"
    )
    is_fraud: bool = Field(
        ..., description="Whether the transaction is classified as fraud at the operating threshold"
    )
    threshold: float = Field(
        ..., description="Operating threshold used for classification"
    )


class HealthResponse(BaseModel):
    """Response from the /health endpoint."""

    status: str = Field(..., description="Service health status")
    model_loaded: bool = Field(..., description="Whether a model is currently loaded")


class ModelInfoResponse(BaseModel):
    """Response from the /model_info endpoint."""

    model_name: str = Field(..., description="Name of the currently loaded model")
    model_path: str = Field(..., description="Path to the model artifact")
    threshold: float = Field(..., description="Operating threshold")
    feature_columns: list[str] = Field(..., description="Expected feature column names")
    selected_at: str = Field(..., description="Timestamp when the model was selected")


class BatchTransactionRequest(BaseModel):
    """Batch of transactions for fraud prediction."""

    transactions: list[TransactionRequest] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of transactions to predict (max 1000)",
    )


class BatchPredictionResponse(BaseModel):
    """Response from the /predict/batch endpoint."""

    predictions: list[PredictionResponse] = Field(
        ..., description="Prediction results for each transaction"
    )
    count: int = Field(..., description="Number of transactions processed")
