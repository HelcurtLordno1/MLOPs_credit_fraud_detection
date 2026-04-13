"""Model evaluation for fraud detection."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: pd.Series,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Compute evaluation metrics for binary classification.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities for positive class
        threshold: Decision threshold
        
    Returns:
        Dictionary of metrics
    """
    y_pred = (y_prob >= threshold).astype(int)
    
    return {
        "auc_roc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def evaluate_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> dict[str, dict[str, float]]:
    """
    Evaluate model on train, validation, and test sets.
    
    Args:
        model: Trained model with predict_proba
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        threshold: Decision threshold
        
    Returns:
        Dictionary mapping split names to metrics
    """
    results = {}
    
    for split_name, X, y in [
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        y_prob = model.predict_proba(X)[:, 1]
        results[split_name] = compute_metrics(y, y_prob, threshold)
    
    return results


def analyze_bias_variance(
    results: dict[str, dict[str, float]],
    metric: str = "auprc",
) -> dict[str, Any]:
    """
    Analyze bias-variance tradeoff.
    
    Args:
        results: Dictionary from evaluate_model
        metric: Metric to analyze (default: auprc - best for imbalanced)
        
    Returns:
        Dictionary with bias-variance analysis
    """
    train_val_gap = abs(results["train"][metric] - results["val"][metric])
    train_test_gap = abs(results["train"][metric] - results["test"][metric])
    val_test_gap = abs(results["val"][metric] - results["test"][metric])
    
    # Diagnosis
    if results["train"][metric] < 0.5:
        diagnosis = "HIGH BIAS (Underfitting)"
    elif train_val_gap > 0.15:
        diagnosis = "HIGH VARIANCE (Overfitting)"
    elif train_val_gap > 0.05:
        diagnosis = "MODERATE VARIANCE"
    else:
        diagnosis = "GOOD FIT"
    
    return {
        f"train_{metric}": results["train"][metric],
        f"val_{metric}": results["val"][metric],
        f"test_{metric}": results["test"][metric],
        f"train_val_gap_{metric}": train_val_gap,
        f"train_test_gap_{metric}": train_test_gap,
        f"val_test_gap_{metric}": val_test_gap,
        "diagnosis": diagnosis,
    }


def get_classification_report(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> str:
    """
    Get detailed classification report.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        threshold: Decision threshold
        
    Returns:
        Classification report as string
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    
    return classification_report(
        y_test,
        y_pred,
        target_names=["Legit (0)", "Fraud (1)"],
        digits=4,
    )


def get_confusion_matrix(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Get confusion matrix.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        threshold: Decision threshold
        
    Returns:
        Confusion matrix as numpy array
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    
    return confusion_matrix(y_test, y_pred)
