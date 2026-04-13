"""
Day 2 — Member 2: The Model Trainer
====================================
Trains Logistic Regression (baseline) + LightGBM on the processed fraud dataset.

Features:
- class_weight='balanced' for Logistic Regression
- scale_pos_weight for LightGBM (handles severe class imbalance ~0.58%)
- Bias-Variance analysis: logs train/val/test gaps to MLflow
- MLflow nested runs: one parent run + one child run per model
- Quality gate: warns if metrics fall below thresholds in configs/training.yaml
- Saves best model (by val AUPRC) → model.joblib
- Saves full metrics report → reports/train_metrics.json

Usage:
    python -m src.ml.train          # normal run
    dvc repro train                 # via DVC pipeline
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ─── Paths ───────────────────────────────────────────────────────────────────
CONFIG_PATH   = Path("configs/training.yaml")
PROCESSED_DIR = Path("data/processed")
REPORT_PATH   = Path("reports/train_metrics.json")
MODEL_PATH    = Path("model.joblib")


# ─── Helpers ─────────────────────────────────────────────────────────────────

def load_config() -> dict[str, Any]:
    """Load training configuration from YAML."""
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_split(split: str) -> pd.DataFrame:
    """
    Load a processed parquet split (train | val | test).

    Args:
        split: One of 'train', 'val', 'test'.

    Returns:
        DataFrame loaded from data/processed/<split>.parquet.

    Raises:
        FileNotFoundError: If the parquet file does not exist.
    """
    path = PROCESSED_DIR / (split + ".parquet")
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset split: {path}")
    return pd.read_parquet(path)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate features (X) and target (y) from a DataFrame.

    Target column must be 'is_fraud'. Only numeric columns are kept as features.

    Args:
        df: Full DataFrame including target column.

    Returns:
        Tuple of (X, y).

    Raises:
        ValueError: If 'is_fraud' column is missing or no numeric features exist.
    """
    if "is_fraud" not in df.columns:
        raise ValueError("Expected target column is_fraud not found in DataFrame")

    y = df["is_fraud"].astype(int)
    X = df.drop(columns=["is_fraud"]).select_dtypes(include=[np.number])

    # LightGBM rejects some characters in feature names; normalize once here.
    rename_map = {
        col: re.sub(r"[^0-9a-zA-Z_]+", "_", col).strip("_") or "feature"
        for col in X.columns
    }
    X = X.rename(columns=rename_map)

    if X.shape[1] == 0:
        raise ValueError("No numeric features available for training")

    return X, y


def build_logistic(cfg: dict[str, Any]) -> Pipeline:
    """
    Build a Logistic Regression pipeline with StandardScaler.

    Config keys read from cfg['model']['logistic']:
        C, max_iter, solver (default: lbfgs), class_weight (default: balanced)

    Args:
        cfg: Full training config dict.

    Returns:
        sklearn Pipeline: [StandardScaler → LogisticRegression]
    """
    lr_cfg = cfg["model"].get("logistic", {})
    return Pipeline([
        ("scaler",     StandardScaler()),
        ("classifier", LogisticRegression(
            C            = lr_cfg.get("C", 1.0),
            max_iter     = lr_cfg.get("max_iter", 1000),
            solver       = lr_cfg.get("solver", "lbfgs"),
            class_weight = lr_cfg.get("class_weight", "balanced"),
        )),
    ])


def build_lightgbm(cfg: dict[str, Any], scale_pos_weight: float) -> lgb.LGBMClassifier:
    """
    Build a LightGBM classifier configured for imbalanced fraud detection.

    scale_pos_weight = n_negatives / n_positives.

    Config keys read from cfg['model']['lightgbm']:
        n_estimators, learning_rate, num_leaves, subsample,
        colsample_bytree, reg_lambda, min_child_samples

    Args:
        cfg:              Full training config dict.
        scale_pos_weight: Ratio of negatives to positives in the training set.

    Returns:
        LGBMClassifier instance.
    """
    lgb_cfg = cfg["model"].get("lightgbm", {})
    return lgb.LGBMClassifier(
        n_estimators      = lgb_cfg.get("n_estimators", 400),
        learning_rate     = lgb_cfg.get("learning_rate", 0.05),
        num_leaves        = lgb_cfg.get("num_leaves", 63),
        subsample         = lgb_cfg.get("subsample", 0.9),
        colsample_bytree  = lgb_cfg.get("colsample_bytree", 0.9),
        reg_lambda        = lgb_cfg.get("reg_lambda", 1.0),
        min_child_samples = lgb_cfg.get("min_child_samples", 50),
        scale_pos_weight  = scale_pos_weight,
        verbose           = -1,
    )


def score(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict[str, float]:
    """
    Compute classification metrics for a fitted model on a given split.

    Metrics returned:
        roc_auc, auprc, recall, precision, fpr (false positive rate)

    Args:
        model: Any fitted sklearn-compatible model with predict_proba().
        X:     Feature matrix.
        y:     True binary labels.

    Returns:
        Dict mapping metric name → float value.
    """
    proba = model.predict_proba(X)[:, 1]
    y_pred = (proba >= 0.5).astype(int)

    tp = float(((y_pred == 1) & (y == 1)).sum())
    fp = float(((y_pred == 1) & (y == 0)).sum())
    tn = float(((y_pred == 0) & (y == 0)).sum())

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "roc_auc":   float(roc_auc_score(y, proba)),
        "auprc":     float(average_precision_score(y, proba)),
        "recall":    float(recall_score(y, y_pred, zero_division=0)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "fpr":       fpr,
    }


# ─── Main Training Function ───────────────────────────────────────────────────

def train() -> None:
    """
    Full Day-2 training pipeline:

    1. Load config & data splits.
    2. Build Logistic Regression + LightGBM.
    3. Fit both models on train split.
    4. Score on train / val / test → bias-variance analysis.
    5. Log everything to MLflow (nested runs).
    6. Quality gate: warn if any metric is below threshold.
    7. Save best model (by val AUPRC) → model.joblib.
    8. Save metrics report → reports/train_metrics.json.
    """
    cfg = load_config()

    # ── Load splits ──────────────────────────────────────────────────────────
    df_train = load_split("train")
    df_val   = load_split("val")
    df_test  = load_split("test")

    X_train, y_train = split_xy(df_train)
    X_val,   y_val   = split_xy(df_val)
    X_test,  y_test  = split_xy(df_test)

    print(f"Train: {X_train.shape}  |  fraud rate: {y_train.mean():.4%}")
    print(f"Val  : {X_val.shape}")
    print(f"Test : {X_test.shape}")

    # ── Imbalance ratio for LightGBM ─────────────────────────────────────────
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)
    print(f"scale_pos_weight = {scale_pos_weight:.2f}  (neg={n_neg}, pos={n_pos})")

    # ── Build models ─────────────────────────────────────────────────────────
    models: dict[str, Any] = {
        "Logistic Regression": build_logistic(cfg),
        "LightGBM":            build_lightgbm(cfg, scale_pos_weight),
    }

    # ── MLflow experiment ─────────────────────────────────────────────────────
    mlflow.set_experiment("credit-fraud")

    all_results: dict[str, dict[str, Any]] = {}

    best_model_name = ""
    best_val_auprc  = float("-inf")
    best_model      = None

    with mlflow.start_run(
        run_name=cfg["model"].get("run_name", "credit-fraud-day2")
    ) as parent_run:

        # Log shared params
        mlflow.log_param("target_column",    "is_fraud")
        mlflow.log_param("candidates",       ",".join(models.keys()))
        mlflow.log_param("scale_pos_weight", scale_pos_weight)
        mlflow.log_param("n_features",       X_train.shape[1])
        mlflow.log_param("train_size",       len(X_train))
        mlflow.log_param("fraud_rate",       float(y_train.mean()))

        for model_name, model in models.items():
            with mlflow.start_run(run_name=model_name, nested=True) as child_run:
                print(f"\n── Training {model_name} ──────────────────────────")

                mlflow.log_param("model_type", model_name)

                # Fit
                model.fit(X_train, y_train)

                # Score all splits
                results: dict[str, dict[str, float]] = {
                    "train": score(model, X_train, y_train),
                    "val":   score(model, X_val,   y_val),
                    "test":  score(model, X_test,  y_test),
                }
                all_results[model_name] = results

                # Log per-split metrics
                for split_name, metrics in results.items():
                    for metric_name, metric_val in metrics.items():
                        mlflow.log_metric(f"{split_name}_{metric_name}", metric_val)

                # Bias-Variance gaps (key for Day-2 requirement)
                train_auprc = results["train"]["auprc"]
                val_auprc   = results["val"]["auprc"]
                test_auprc  = results["test"]["auprc"]

                bv_gap = abs(train_auprc - val_auprc)
                tt_gap = abs(train_auprc - test_auprc)

                mlflow.log_metric("bias_variance_gap_auprc", bv_gap)
                mlflow.log_metric("train_test_gap_auprc",    tt_gap)

                print(f"  Train AUPRC : {train_auprc:.4f}")
                print(f"  Val   AUPRC : {val_auprc:.4f}  (BV gap: {bv_gap:.4f})")
                print(f"  Test  AUPRC : {test_auprc:.4f}  (TT gap: {tt_gap:.4f})")
                print(f"  ✅ Logged → run_id: {child_run.info.run_id}")

                # Track best model (by val AUPRC)
                if val_auprc > best_val_auprc:
                    best_val_auprc  = val_auprc
                    best_model_name = model_name
                    best_model      = model

        if best_model is None:
            raise RuntimeError("No model was trained. Check your data splits.")

        # Log best model summary to parent run
        mlflow.log_param("best_model",    best_model_name)
        mlflow.log_metric("best_val_auprc", best_val_auprc)

        # ── Quality Gate ─────────────────────────────────────────────────────
        thresholds = cfg.get("thresholds", {})
        best_test  = all_results[best_model_name]["test"]

        print(f"\n── Quality Gate ({best_model_name} on test) ──────────────")
        gate_map = {
            "auprc":     thresholds.get("min_auprc",   0.85),
            "roc_auc":   thresholds.get("min_auc",     0.90),
            "recall":    thresholds.get("min_recall",  0.92),
            "precision": thresholds.get("min_precision", 0.80),
        }
        for metric, threshold in gate_map.items():
            actual = best_test[metric]
            status = "✅" if actual >= threshold else "⚠️ BELOW THRESHOLD"
            print(f"  {metric:10s}: {actual:.4f}  (min={threshold})  {status}")

        # ── Save model ───────────────────────────────────────────────────────
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, MODEL_PATH)
        mlflow.log_artifact(str(MODEL_PATH))
        print(f"\nSaved model to: {MODEL_PATH}")

        # ── Save metrics report ───────────────────────────────────────────────
        report: dict[str, Any] = {
            "best_model":    best_model_name,
            "best_val_auprc": best_val_auprc,
            "results":       {},
        }
        for model_name, splits in all_results.items():
            train_a = splits["train"]["auprc"]
            val_a   = splits["val"]["auprc"]
            test_a  = splits["test"]["auprc"]
            report["results"][model_name] = {
                **splits,
                "bias_variance_gap_auprc": abs(train_a - val_a),
                "train_test_gap_auprc":    abs(train_a - test_a),
            }

        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with REPORT_PATH.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        mlflow.log_artifact(str(REPORT_PATH))
        print(f"Saved metrics to: {REPORT_PATH}")

        print(f"\n🏆 Best model: {best_model_name} (val_auprc={best_val_auprc:.4f})")
        print(f"📋 Parent run ID: {parent_run.info.run_id}")


if __name__ == "__main__":
    train()
