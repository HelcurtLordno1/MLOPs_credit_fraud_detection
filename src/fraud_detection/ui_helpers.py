"""UI helper functions for the Streamlit dashboard.

Provides default payloads, local report loading, and CSV parsing
for the Fraud Detection Operations Dashboard.
"""
from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd


def _project_root() -> Path:
    """Best-effort project root lookup for the Streamlit process."""
    # When running via ``streamlit run streamlit_app/app.py`` from repo root
    # the CWD is typically the repo root.
    cwd = Path.cwd()
    if (cwd / "pyproject.toml").exists():
        return cwd
    # Fallback: walk up from this file.
    current = Path(__file__).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return cwd


def default_transaction_payload() -> dict[str, Any]:
    """Return a sample transaction dict matching the Kaggle schema."""
    return {
        "cc_num": "4263981246278992",
        "merchant": "fraud_Kirlin and Sons",
        "category": "personal_care",
        "amt": 2.86,
        "first": "Jeff",
        "last": "Elliott",
        "gender": "M",
        "street": "351 Darlene Green",
        "city": "Columbia",
        "state": "SC",
        "zip": "29209",
        "lat": 33.9659,
        "long": -80.9355,
        "city_pop": 333497,
        "job": "Mechanical engineer",
        "dob": "1968-03-19",
        "trans_num": "2da90c7d74bd46f29b8f41a0aa06a240",
        "unix_time": 1325376018,
        "merch_lat": 33.986391,
        "merch_long": -81.200714,
    }


def load_local_status() -> dict[str, Any]:
    """Load local pipeline status files for the Retraining tab."""
    root = _project_root()
    status: dict[str, Any] = {}

    # Training metrics
    train_path = root / "reports" / "metrics" / "train_metrics.json"
    if train_path.exists():
        with open(train_path, encoding="utf-8") as f:
            status["training"] = json.load(f)

    # Test / evaluation metrics
    test_path = root / "reports" / "metrics" / "test_metrics.json"
    if test_path.exists():
        with open(test_path, encoding="utf-8") as f:
            status["candidate"] = json.load(f)

    # Promotion decision
    promo_path = root / "models" / "registry" / "last_promotion.json"
    if promo_path.exists():
        with open(promo_path, encoding="utf-8") as f:
            status["promotion"] = json.load(f)

    # Champion = promoted model info
    status["champion"] = status.get("promotion", {})

    # Drift report
    drift_path = root / "reports" / "drift" / "drift_report.json"
    if drift_path.exists():
        with open(drift_path, encoding="utf-8") as f:
            status["drift"] = json.load(f)

    return status


def parse_batch_csv(raw_bytes: bytes) -> pd.DataFrame:
    """Parse an uploaded CSV file into a DataFrame for batch scoring.

    Expects column names matching the ``Transaction`` schema.
    """
    df = pd.read_csv(BytesIO(raw_bytes))
    # Normalise column names (strip whitespace, lowercase).
    df.columns = [c.strip() for c in df.columns]
    return df
