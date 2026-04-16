"""Drift detection for fraud detection models."""
from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage()
        }
        return json.dumps(log_record)

logger = logging.getLogger("drift_monitor")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)


def detect_distribution_drift(
    reference: pd.Series,
    current: pd.Series,
    method: str = "ks",
    threshold: float = 0.05,
) -> dict[str, Any]:
    """
    Detect distribution drift using Kolmogorov-Smirnov or Chi-Square test.
    
    Args:
        reference: Reference distribution (e.g., training data)
        current: Current distribution (e.g., recent production data)
        method: "ks" for continuous or "chi2" for categorical
        threshold: Statistical significance threshold (p-value)
        
    Returns:
        Dictionary with drift detection results
    """
    if method == "ks":
        statistic, p_value = stats.ks_2samp(reference, current)
        drifted = p_value < threshold
        test_name = "Kolmogorov-Smirnov"
    elif method == "chi2":
        # For categorical: convert to frequency tables
        ref_counts = reference.value_counts()
        curr_counts = current.value_counts()
        
        # Align categories
        all_categories = set(ref_counts.index) | set(curr_counts.index)
        ref_freq = [ref_counts.get(cat, 0) for cat in all_categories]
        curr_freq = [curr_counts.get(cat, 0) for cat in all_categories]
        
        statistic, p_value = stats.chisquare(curr_freq, ref_freq)
        drifted = p_value < threshold
        test_name = "Chi-Square"
    else:
        raise ValueError(f"Unknown drift detection method: {method}")
    
    return {
        "test": test_name,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "drifted": bool(drifted),
        "threshold": threshold,
    }


def detect_feature_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    numeric_threshold: float = 0.05,
    categorical_threshold: float = 0.05,
) -> dict[str, Any]:
    """
    Detect feature drift across multiple features.
    
    Args:
        reference_df: Reference data (development/training)
        current_df: Current data (production)
        numeric_threshold: KS test p-value threshold
        categorical_threshold: Chi-square test p-value threshold
        
    Returns:
        Dictionary mapping feature names to drift results
    """
    drift_results = {}
    
    numeric_cols = reference_df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if col not in current_df.columns:
            continue
        
        ref_data = reference_df[col].dropna()
        curr_data = current_df[col].dropna()
        
        if len(ref_data) > 0 and len(curr_data) > 0:
            drift_results[col] = detect_distribution_drift(
                ref_data, curr_data, method="ks", threshold=numeric_threshold
            )
    
    return drift_results


def detect_target_drift(
    reference_target: pd.Series,
    current_target: pd.Series,
    threshold: float = 0.05,
) -> dict[str, Any]:
    """
    Detect drift in target variable (fraud rate).
    
    Args:
        reference_target: Reference target (training data)
        current_target: Current target (production data)
        threshold: Statistical significance threshold
        
    Returns:
        Dictionary with target drift results
    """
    ref_fraud_rate = reference_target.mean()
    curr_fraud_rate = current_target.mean()
    
    # Use binomial test for proportions
    n_fraud = int(current_target.sum())
    n_total = len(current_target)
    expected_fraud = int(n_total * ref_fraud_rate)

    # scipy.stats.binom_test was removed in newer SciPy versions.
    if hasattr(stats, "binomtest"):
        p_value = stats.binomtest(
            n_fraud,
            n_total,
            ref_fraud_rate,
            alternative="two-sided",
        ).pvalue
    else:
        p_value = stats.binom_test(
            n_fraud,
            n_total,
            ref_fraud_rate,
            alternative="two-sided",
        )
    drifted = p_value < threshold
    
    return {
        "reference_fraud_rate": float(ref_fraud_rate),
        "current_fraud_rate": float(curr_fraud_rate),
        "fraud_rate_change": float(curr_fraud_rate - ref_fraud_rate),
        "p_value": float(p_value),
        "drifted": bool(drifted),
        "threshold": threshold,
    }


def summarize_drift(
    feature_drift_results: dict[str, Any],
) -> dict[str, Any]:
    """
    Summarize drift detection results.
    
    Args:
        feature_drift_results: Results from detect_feature_drift
        
    Returns:
        Summary dictionary
    """
    drifted_features = [f for f, r in feature_drift_results.items() if r["drifted"]]
    
    return {
        "total_features_checked": len(feature_drift_results),
        "features_with_drift": len(drifted_features),
        "drifted_feature_names": drifted_features,
        "overall_drift_detected": len(drifted_features) > 0,
    }
