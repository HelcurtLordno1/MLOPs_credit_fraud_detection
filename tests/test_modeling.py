from __future__ import annotations

import pandas as pd

from fraud_detection.modeling.train import (
    prepare_features,
    train_lightgbm,
    train_logistic_regression,
)


def _sample_frame(rows: int = 120) -> pd.DataFrame:
    payload = {
        "amt": [float(i % 50) for i in range(rows)],
        "city_pop": [1000 + i for i in range(rows)],
        "unix_time": [1546300800 + i for i in range(rows)],
        "is_fraud": [1 if i % 15 == 0 else 0 for i in range(rows)],
    }
    return pd.DataFrame(payload)


def test_prepare_features_selects_numeric_predictors() -> None:
    frame = _sample_frame()
    frame["merchant"] = "m1"
    x, y = prepare_features(frame, target_col="is_fraud")

    assert "merchant" not in x.columns
    assert "is_fraud" not in x.columns
    assert len(x.columns) == 3
    assert y.nunique() == 2


def test_training_helpers_fit_and_predict_probabilities() -> None:
    frame = _sample_frame()
    x, y = prepare_features(frame, target_col="is_fraud")

    logistic = train_logistic_regression(x, y, {"class_weight": "balanced", "max_iter": 200})
    logistic_prob = logistic.predict_proba(x)[:, 1]
    assert len(logistic_prob) == len(frame)

    lightgbm = train_lightgbm(x, y, {"n_estimators": 30, "num_leaves": 15})
    lgbm_prob = lightgbm.predict_proba(x)[:, 1]
    assert len(lgbm_prob) == len(frame)
    assert float(lightgbm.get_params().get("scale_pos_weight", 0.0)) > 1.0
