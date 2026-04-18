"""Fraud Detection API package."""

from fraud_detection.api.main import app
from fraud_detection.api.service import FraudDetectionService

__all__ = ["app", "FraudDetectionService"]
