from pathlib import Path

import joblib
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_training_config_exists():
    assert (PROJECT_ROOT / "configs" / "training.yaml").exists()


def test_processed_data_has_target_column():
    test_path = PROJECT_ROOT / "data" / "processed" / "test.parquet"
    if not test_path.exists():
        pytest.skip("Processed test data not found. Run data prep first.")

    df = pd.read_parquet(test_path)
    assert "Class" in df.columns
    assert set(df["Class"].unique()).issubset({0, 1})


def test_model_predict_proba_range_if_model_exists():
    model_path = PROJECT_ROOT / "models" / "trained" / "latest.joblib"
    data_path = PROJECT_ROOT / "data" / "processed" / "test.parquet"

    if not model_path.exists() or not data_path.exists():
        pytest.skip("Model/data artifact missing. Run training pipeline first.")

    model = joblib.load(model_path)
    df = pd.read_parquet(data_path)
    X = df.drop(columns=["Class", "Time"], errors="ignore")

    scores = model.predict_proba(X)[:, 1]
    assert (scores >= 0).all()
    assert (scores <= 1).all()
