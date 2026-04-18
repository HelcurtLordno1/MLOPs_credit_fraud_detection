"""Fraud detection model serving service.

Handles model loading, feature preprocessing for inference, and prediction.
The preprocessing must produce the **exact same numeric feature set** that
the training pipeline (``features.py`` → ``train.py:prepare_features()``)
produces, but from a single raw transaction JSON rather than a full DataFrame.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from fraud_detection.data.features import (
    engineer_amount_features,
    engineer_distance_features,
    engineer_temporal_features,
)
from fraud_detection.utils.paths import find_project_root

logger = logging.getLogger(__name__)


class FraudDetectionService:
    """Singleton-style service that loads the champion model and serves predictions."""

    def __init__(self) -> None:
        self.model: Any = None
        self.model_name: str = "unknown"
        self.model_version: str = "0.0.0"
        self.threshold: float = 0.5
        self.feature_count: int = 0
        self.trained_at: str | None = None

        # Aggregate statistics computed from training data for single-row inference.
        self._customer_defaults: dict[str, float] = {}
        self._merchant_defaults: dict[str, float] = {}
        self._category_defaults: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load_model(self, model_path: Path | None = None) -> None:
        """Load the trained model and associated metadata."""
        project_root = find_project_root()

        if model_path is None:
            serve_cfg = self._load_serve_config(project_root)
            model_path = project_root / serve_cfg.get(
                "model_path", "models/trained/best_model.joblib"
            )
            self.threshold = float(serve_cfg.get("threshold", 0.5))
            self.model_name = serve_cfg.get("name", "fraud-detector")
            self.model_version = serve_cfg.get("version", "1.0.0")

        if not model_path.exists():
            logger.warning("Model file not found at %s", model_path)
            return

        self.model = joblib.load(model_path)
        logger.info("Loaded model from %s", model_path)

        # Probe feature count from training data if available.
        processed_dir = project_root / "data" / "processed"
        train_path = processed_dir / "train.parquet"
        if train_path.exists():
            sample = pd.read_parquet(train_path, columns=None).head(5)
            numeric_cols = sample.select_dtypes(include=[np.number]).columns.tolist()
            for drop_col in ["is_fraud", "row_id", "Unnamed: 0", "index", "id"]:
                if drop_col in numeric_cols:
                    numeric_cols.remove(drop_col)
            self.feature_count = len(numeric_cols)
            self._compute_aggregate_defaults(processed_dir)

        # Read training timestamp from metrics if available.
        metrics_path = project_root / "reports" / "metrics" / "train_metrics.json"
        if metrics_path.exists():
            with open(metrics_path, encoding="utf-8") as f:
                metrics = json.load(f)
            self.trained_at = metrics.get("best_run_id")

    def _load_serve_config(self, project_root: Path) -> dict[str, Any]:
        """Load serving configuration from ``configs/serve.yaml``."""
        config_path = project_root / "configs" / "serve.yaml"
        if not config_path.exists():
            return {}
        with open(config_path, encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}

        flat: dict[str, Any] = {}
        for section in payload.values():
            if isinstance(section, dict):
                flat.update(section)
        return flat

    def _compute_aggregate_defaults(self, processed_dir: Path) -> None:
        """Pre-compute aggregate feature defaults from the training parquet."""
        train_path = processed_dir / "train.parquet"
        if not train_path.exists():
            return

        df = pd.read_parquet(train_path)

        # Customer defaults (global averages across all customers).
        self._customer_defaults = {
            "customer_txn_count": float(df["customer_txn_count"].median())
            if "customer_txn_count" in df.columns
            else 1.0,
            "customer_avg_amt": float(df["customer_avg_amt"].median())
            if "customer_avg_amt" in df.columns
            else float(df["amt"].mean()),
            "customer_std_amt": float(df["customer_std_amt"].median())
            if "customer_std_amt" in df.columns
            else float(df["amt"].std()),
            "customer_fraud_rate": float(df["customer_fraud_rate"].median())
            if "customer_fraud_rate" in df.columns
            else 0.0,
        }

        # Merchant defaults.
        self._merchant_defaults = {
            "merchant_fraud_rate": float(df["merchant_fraud_rate"].median())
            if "merchant_fraud_rate" in df.columns
            else 0.0,
            "merchant_txn_count": float(df["merchant_txn_count"].median())
            if "merchant_txn_count" in df.columns
            else 1.0,
            "merchant_avg_amt": float(df["merchant_avg_amt"].median())
            if "merchant_avg_amt" in df.columns
            else float(df["amt"].mean()),
            "merchant_std_amt": float(df["merchant_std_amt"].median())
            if "merchant_std_amt" in df.columns
            else float(df["amt"].std()),
        }

        # Category defaults.
        self._category_defaults = {
            "category_fraud_rate": float(df["category_fraud_rate"].median())
            if "category_fraud_rate" in df.columns
            else 0.0,
            "category_txn_count": float(df["category_txn_count"].median())
            if "category_txn_count" in df.columns
            else 1.0,
            "category_avg_amt": float(df["category_avg_amt"].median())
            if "category_avg_amt" in df.columns
            else float(df["amt"].mean()),
        }

        logger.info(
            "Computed aggregate defaults from %d training rows",
            len(df),
        )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def preprocess_transaction(self, data: dict[str, Any]) -> pd.DataFrame:
        """Transform a raw transaction dict into a model-ready numeric DataFrame.

        This replicates the feature engineering pipeline used during training
        but is designed to work with a **single row** (no group-by possible).
        Aggregate features use pre-computed defaults from the training set.
        """
        # Build a single-row DataFrame with the raw fields.
        raw = pd.DataFrame([data])

        #  Temporal features -----------------------------------------------
        # Supply trans_date_trans_time from unix_time so temporal helpers work.
        if "trans_date_trans_time" not in raw.columns and "unix_time" in raw.columns:
            raw["trans_date_trans_time"] = pd.to_datetime(raw["unix_time"], unit="s").dt.strftime(
                "%Y-%m-%d %H:%M:%S"
            )

        raw = engineer_temporal_features(raw)

        #  Customer velocity (defaults) ------------------------------------
        for col, default in self._customer_defaults.items():
            raw[col] = default

        #  Merchant risk (defaults) ----------------------------------------
        for col, default in self._merchant_defaults.items():
            raw[col] = default

        #  Category (defaults) ---------------------------------------------
        for col, default in self._category_defaults.items():
            raw[col] = default

        #  Distance --------------------------------------------------------
        raw = engineer_distance_features(raw)

        #  Amount anomaly --------------------------------------------------
        raw = engineer_amount_features(raw)

        #  Select numeric columns only (same as train.py:prepare_features) -
        drop_cols = {"is_fraud", "row_id", "Unnamed: 0", "index", "id"}
        numeric = raw.select_dtypes(include=[np.number])
        numeric = numeric.drop(
            columns=[c for c in drop_cols if c in numeric.columns],
            errors="ignore",
        )

        return numeric

    def predict(self, transaction: dict[str, Any]) -> dict[str, Any]:
        """Return fraud probability and decision for a single transaction."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        features = self.preprocess_transaction(transaction)
        probability = float(self.model.predict_proba(features)[:, 1][0])

        return {
            "fraud_probability": round(probability, 6),
            "is_fraud": probability >= self.threshold,
            "threshold": self.threshold,
        }

    def predict_batch(self, transactions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Score a list of transactions."""
        return [self.predict(txn) for txn in transactions]

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------

    def get_drift_report(self) -> dict[str, Any]:
        """Load the latest drift report from disk."""
        project_root = find_project_root()
        drift_path = project_root / "reports" / "drift" / "drift_report.json"
        if not drift_path.exists():
            return {"error": "No drift report available"}
        with open(drift_path, encoding="utf-8") as f:
            return json.load(f)

    def get_model_info(self) -> dict[str, Any]:
        """Return metadata about the loaded model."""
        return {
            "model_name": self.model_name,
            "version": self.model_version,
            "threshold": self.threshold,
            "feature_count": self.feature_count,
            "trained_at": self.trained_at,
        }
