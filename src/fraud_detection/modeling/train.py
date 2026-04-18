"""Model training for fraud detection."""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict[str, Any],
) -> Pipeline:
    """
    Train Logistic Regression baseline model.

    Args:
        X_train: Training features
        y_train: Training target
        params: Model parameters

    Returns:
        Fitted Pipeline with StandardScaler and LogisticRegression
    """
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=params.get("C", 1.0),
                    max_iter=params.get("max_iter", 1000),
                    solver=params.get("solver", "lbfgs"),
                    class_weight=params.get("class_weight", "balanced"),
                    random_state=params.get("random_state", 42),
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    return model


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict[str, Any],
) -> lgb.LGBMClassifier:
    """
    Train LightGBM gradient boosting model.

    Args:
        X_train: Training features
        y_train: Training target
        params: Model parameters

    Returns:
        Fitted LGBMClassifier
    """
    # Calculate scale_pos_weight for imbalance handling
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = neg_count / max(pos_count, 1)

    model = lgb.LGBMClassifier(
        n_estimators=params.get("n_estimators", 400),
        learning_rate=params.get("learning_rate", 0.05),
        num_leaves=params.get("num_leaves", 63),
        subsample=params.get("subsample", 0.9),
        colsample_bytree=params.get("colsample_bytree", 0.9),
        reg_lambda=params.get("reg_lambda", 1.0),
        reg_alpha=params.get("reg_alpha", 0.0),
        min_child_samples=params.get("min_child_samples", 50),
        scale_pos_weight=scale_pos_weight,
        objective=params.get("objective", "binary"),
        metric=params.get("metric", "auc"),
        random_state=params.get("random_state", 42),
        n_jobs=params.get("n_jobs", -1),
        verbose=params.get("verbose", -1),
    )

    model.fit(X_train, y_train)
    return model


def get_feature_names(df: pd.DataFrame) -> list[str]:
    """Extract numeric feature names from dataframe."""
    # Drop ID-like and non-numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    for col in ["Unnamed: 0", "index", "id", "row_id"]:
        if col in numeric_df.columns:
            numeric_df = numeric_df.drop(columns=[col])
    return numeric_df.columns.tolist()


def prepare_features(
    df: pd.DataFrame,
    target_col: str = "is_fraud",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare X and y from dataframe.

    Args:
        df: Input dataframe
        target_col: Name of target column

    Returns:
        Tuple of (X, y)
    """
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])

    # Drop ID-like columns
    for col in ["Unnamed: 0", "index", "id", "row_id"]:
        if col in X.columns:
            X = X.drop(columns=[col])

    # Keep only numeric features
    X = X.select_dtypes(include=[np.number])

    return X, y
