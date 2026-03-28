from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from codes.ml.common.paths import ensure_dirs, find_project_root


def _load_yaml(path: Path, default: dict) -> dict:
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate_model() -> None:
    project_root = find_project_root()
    base_cfg = _load_yaml(project_root / "configs" / "base.yaml", default={})
    inference_cfg = _load_yaml(project_root / "configs" / "inference.yaml", default={})

    target_col = str(base_cfg.get("target", "Class"))
    processed_dir = project_root / str(base_cfg.get("data", {}).get("processed_dir", "data/processed"))

    model_rel_path = inference_cfg.get("model_path", "models/trained/latest.joblib")
    threshold = float(inference_cfg.get("threshold", 0.5))
    feature_columns = inference_cfg.get("feature_columns", None)

    model_path = project_root / str(model_rel_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")

    model = joblib.load(model_path)
    test_df = pd.read_parquet(processed_dir / "test.parquet")

    if feature_columns:
        X_test = test_df[feature_columns]
    else:
        X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    y_score = model.predict_proba(X_test)[:, 1]
    y_pred = (y_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    metrics = {
        "auc_roc": float(roc_auc_score(y_test, y_score)),
        "auprc": float(average_precision_score(y_test, y_score)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "fpr": float(fpr),
        "threshold": float(threshold),
        "model_path": str(model_path.relative_to(project_root)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }

    out_dir = project_root / "reports" / "metrics"
    ensure_dirs(out_dir)

    with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Evaluation complete")
    print(json.dumps(metrics, indent=2))


def main() -> None:
    evaluate_model()


if __name__ == "__main__":
    main()
