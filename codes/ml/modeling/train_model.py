from __future__ import annotations

import json
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from codes.ml.common.paths import ensure_dirs, find_project_root


def _mlflow_available() -> bool:
    return all(
        hasattr(mlflow, attr)
        for attr in [
            "set_tracking_uri",
            "set_registry_uri",
            "set_experiment",
            "start_run",
            "log_param",
            "log_metric",
            "log_metrics",
            "log_artifact",
            "log_dict",
        ]
    )


def _load_yaml(path: Path, default: dict) -> dict:
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _select_threshold(
    y_true: pd.Series,
    y_score: pd.Series,
    min_recall: float,
    policy: str,
) -> tuple[float, dict]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    best_threshold = 0.5
    best_precision = 0.0
    best_recall = 0.0
    best_f1 = -1.0

    for idx, thr in enumerate(thresholds):
        p = float(precision[idx])
        r = float(recall[idx])
        f1 = 0.0 if (p + r) == 0 else (2 * p * r / (p + r))

        if policy == "maximize_precision_at_recall":
            if r >= min_recall and p > best_precision:
                best_precision = p
                best_recall = r
                best_f1 = f1
                best_threshold = float(thr)
        elif policy == "maximize_precision":
            if p > best_precision:
                best_precision = p
                best_recall = r
                best_f1 = f1
                best_threshold = float(thr)
        else:
            # balanced_f1 default: maximize F1 with precision as tie-breaker.
            if f1 > best_f1 or (f1 == best_f1 and p > best_precision):
                best_precision = p
                best_recall = r
                best_f1 = f1
                best_threshold = float(thr)

    if policy == "maximize_precision_at_recall" and best_precision == 0.0:
        # If recall floor is impossible, fallback to balanced F1.
        return _select_threshold(y_true, y_score, min_recall=min_recall, policy="balanced_f1")

    detail = {
        "selected_threshold": float(best_threshold),
        "selected_precision": float(best_precision),
        "selected_recall": float(best_recall),
        "selected_f1": float(best_f1),
        "threshold_policy": policy,
        "min_recall_constraint": float(min_recall),
    }
    return best_threshold, detail


def _compute_metrics(y_true: pd.Series, y_score: pd.Series, threshold: float) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    return {
        "auc_roc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "fpr": fpr,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def _build_candidates(seed: int, training_cfg: dict) -> dict:
    model_cfg = training_cfg.get("model", {})
    logistic_cfg = model_cfg.get("logistic", {})
    rf_cfg = model_cfg.get("random_forest", {})

    candidates = {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight=model_cfg.get("class_weight", "balanced"),
                        C=float(logistic_cfg.get("C", 1.0)),
                        max_iter=int(logistic_cfg.get("max_iter", 1200)),
                        solver=str(logistic_cfg.get("solver", "lbfgs")),
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=int(rf_cfg.get("n_estimators", 300)),
                        max_depth=rf_cfg.get("max_depth", None),
                        min_samples_leaf=int(rf_cfg.get("min_samples_leaf", 1)),
                        class_weight=str(rf_cfg.get("class_weight", "balanced_subsample")),
                        n_jobs=-1,
                        random_state=seed,
                    ),
                )
            ]
        ),
    }

    enabled = set(training_cfg.get("candidates", ["logistic_regression", "random_forest"]))
    return {k: v for k, v in candidates.items() if k in enabled}


def _build_baseline(seed: int) -> tuple:
    """Build a simple baseline model using logistic regression with default parameters."""
    baseline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(random_state=seed, max_iter=1000, solver="lbfgs")),
        ]
    )
    return "baseline_logistic", baseline


def _train_baseline() -> None:
    """Train and log baseline model to MLflow for comparison."""
    project_root = find_project_root()
    base_cfg = _load_yaml(project_root / "configs" / "base.yaml", default={})
    training_cfg = _load_yaml(project_root / "configs" / "training.yaml", default={})
    inference_cfg = _load_yaml(project_root / "configs" / "inference.yaml", default={})

    seed = int(base_cfg.get("seed", 42))
    target_col = str(base_cfg.get("target", "Class"))
    drop_columns = set(training_cfg.get("features", {}).get("drop_columns", ["Time"]))

    processed_dir = project_root / str(base_cfg.get("data", {}).get("processed_dir", "data/processed"))
    train_df = pd.read_parquet(processed_dir / "train.parquet")
    val_df = pd.read_parquet(processed_dir / "val.parquet")
    test_df = pd.read_parquet(processed_dir / "test.parquet")

    feature_columns = [c for c in train_df.columns if c != target_col and c not in drop_columns]

    X_train = train_df[feature_columns]
    y_train = train_df[target_col]
    X_val = val_df[feature_columns]
    y_val = val_df[target_col]
    X_test = test_df[feature_columns]
    y_test = test_df[target_col]

    baseline_name, baseline_model = _build_baseline(seed=seed)

    mlflow_enabled = _mlflow_available()
    mlflow_db_path = project_root / "mlflow.db"
    tracking_uri = f"sqlite:///{mlflow_db_path.as_posix()}"
    if mlflow_enabled:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_registry_uri(tracking_uri)
        mlflow.set_experiment("credit-fraud")

    run_ctx = mlflow.start_run(run_name="week1-baseline-logreg") if mlflow_enabled else nullcontext()
    with run_ctx as run:
        if mlflow_enabled:
            mlflow.log_param("seed", seed)
            mlflow.log_param("feature_count", len(feature_columns))
            mlflow.log_param("model_type", baseline_name)
            mlflow.log_param("threshold", 0.5)
            mlflow.log_param("threshold_tuning", "none")

        # Train on combined train+val for baseline
        train_val_df = pd.concat([train_df, val_df], ignore_index=True)
        X_train_val = train_val_df[feature_columns]
        y_train_val = train_val_df[target_col]
        baseline_model.fit(X_train_val, y_train_val)

        # Evaluate on test set with default 0.5 threshold
        test_score = baseline_model.predict_proba(X_test)[:, 1]
        baseline_threshold = 0.5
        test_metrics = _compute_metrics(y_test, test_score, baseline_threshold)

        metric_map = {
            "auc_roc": test_metrics["auc_roc"],
            "auprc": test_metrics["auprc"],
            "precision": test_metrics["precision"],
            "recall": test_metrics["recall"],
            "f1": test_metrics["f1"],
            "accuracy": test_metrics["accuracy"],
            "fpr": test_metrics["fpr"],
            "pass_min_auc": 1.0 if test_metrics["auc_roc"] >= 0.90 else 0.0,
            "pass_min_auprc": 1.0 if test_metrics["auprc"] >= 0.85 else 0.0,
            "pass_min_recall": 1.0 if test_metrics["recall"] >= 0.92 else 0.0,
            "pass_min_precision": 1.0 if test_metrics["precision"] >= 0.80 else 0.0,
        }

        if mlflow_enabled:
            mlflow.log_metrics({k: float(v) for k, v in metric_map.items()})

        model_dir = project_root / "models" / "trained"
        reports_dir = project_root / "reports" / "metrics"
        ensure_dirs(model_dir, reports_dir)

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        baseline_model_path = model_dir / f"{baseline_name}_{stamp}.joblib"
        joblib.dump(baseline_model, baseline_model_path)

        if mlflow_enabled:
            mlflow.log_artifact(str(baseline_model_path), artifact_path="model_export")

        baseline_info = {
            "model_name": baseline_name,
            "threshold": float(baseline_threshold),
            "feature_columns": feature_columns,
            "model_path": str(baseline_model_path.relative_to(project_root)),
            "selected_at": datetime.now(timezone.utc).isoformat(),
        }

        summary_payload = {
            "model_type": baseline_name,
            "test_metrics": test_metrics,
            "model_info": baseline_info,
            "tracking_uri": tracking_uri,
            "run_id": run.info.run_id if mlflow_enabled and run is not None else "mlflow-disabled",
            "mlflow_enabled": mlflow_enabled,
        }

        with open(reports_dir / "baseline_metrics.json", "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2)

        if mlflow_enabled:
            mlflow.log_dict(summary_payload, "reports/baseline_metrics.json")

    print("Baseline training complete")
    print(json.dumps({"model": baseline_name, **metric_map}, indent=2))


def train_models() -> None:
    project_root = find_project_root()
    base_cfg = _load_yaml(project_root / "configs" / "base.yaml", default={})
    training_cfg = _load_yaml(project_root / "configs" / "training.yaml", default={})
    inference_cfg = _load_yaml(project_root / "configs" / "inference.yaml", default={})

    seed = int(base_cfg.get("seed", 42))
    target_col = str(base_cfg.get("target", "Class"))

    drop_columns = set(training_cfg.get("features", {}).get("drop_columns", ["Time"]))
    threshold_cfg = training_cfg.get("thresholds", {})
    min_recall = float(threshold_cfg.get("min_recall", 0.92))
    threshold_policy = str(threshold_cfg.get("policy", "balanced_f1"))

    processed_dir = project_root / str(base_cfg.get("data", {}).get("processed_dir", "data/processed"))
    train_df = pd.read_parquet(processed_dir / "train.parquet")
    val_df = pd.read_parquet(processed_dir / "val.parquet")
    test_df = pd.read_parquet(processed_dir / "test.parquet")

    feature_columns = [c for c in train_df.columns if c != target_col and c not in drop_columns]

    X_train = train_df[feature_columns]
    y_train = train_df[target_col]
    X_val = val_df[feature_columns]
    y_val = val_df[target_col]
    X_test = test_df[feature_columns]
    y_test = test_df[target_col]

    candidates = _build_candidates(seed=seed, training_cfg=training_cfg)
    if not candidates:
        raise ValueError("No candidate models enabled in configs/training.yaml")

    mlflow_enabled = _mlflow_available()
    mlflow_db_path = project_root / "mlflow.db"
    tracking_uri = f"sqlite:///{mlflow_db_path.as_posix()}"
    if mlflow_enabled:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_registry_uri(tracking_uri)
        mlflow.set_experiment("credit-fraud")

    best_name = None
    best_model = None
    best_threshold = 0.5
    best_val_summary = None

    run_ctx = mlflow.start_run(run_name="improved-threshold-tuned-models") if mlflow_enabled else nullcontext()
    with run_ctx as run:
        if mlflow_enabled:
            mlflow.log_param("seed", seed)
            mlflow.log_param("feature_count", len(feature_columns))
            mlflow.log_param("min_recall_constraint", min_recall)
            mlflow.log_param("threshold_policy", threshold_policy)

        candidate_rows = []
        for name, model in candidates.items():
            model.fit(X_train, y_train)
            val_score = model.predict_proba(X_val)[:, 1]
            threshold, threshold_info = _select_threshold(
                y_val,
                val_score,
                min_recall=min_recall,
                policy=threshold_policy,
            )
            val_metrics = _compute_metrics(y_val, val_score, threshold)

            candidate_rows.append(
                {
                    "name": name,
                    "threshold": threshold,
                    "val_precision": val_metrics["precision"],
                    "val_recall": val_metrics["recall"],
                    "val_auprc": val_metrics["auprc"],
                    "val_f1": val_metrics["f1"],
                }
            )

            if mlflow_enabled:
                mlflow.log_metric(f"{name}_val_precision", float(val_metrics["precision"]))
                mlflow.log_metric(f"{name}_val_recall", float(val_metrics["recall"]))
                mlflow.log_metric(f"{name}_val_auprc", float(val_metrics["auprc"]))
                mlflow.log_metric(f"{name}_val_f1", float(val_metrics["f1"]))
                mlflow.log_metric(f"{name}_val_threshold", float(threshold))

            if best_val_summary is None:
                best_name = name
                best_model = model
                best_threshold = threshold
                best_val_summary = {**val_metrics, **threshold_info}
                continue

            if threshold_policy == "maximize_precision":
                if val_metrics["precision"] > best_val_summary["precision"]:
                    best_name = name
                    best_model = model
                    best_threshold = threshold
                    best_val_summary = {**val_metrics, **threshold_info}
            elif threshold_policy == "maximize_precision_at_recall":
                if (
                    val_metrics["precision"] > best_val_summary["precision"]
                    or (
                        val_metrics["precision"] == best_val_summary["precision"]
                        and val_metrics["recall"] > best_val_summary["recall"]
                    )
                ):
                    best_name = name
                    best_model = model
                    best_threshold = threshold
                    best_val_summary = {**val_metrics, **threshold_info}
            else:
                if (
                    val_metrics["f1"] > best_val_summary["f1"]
                    or (
                        val_metrics["f1"] == best_val_summary["f1"]
                        and val_metrics["precision"] > best_val_summary["precision"]
                    )
                ):
                    best_name = name
                    best_model = model
                    best_threshold = threshold
                    best_val_summary = {**val_metrics, **threshold_info}

        # Refit the winner on combined train+val for stronger final model.
        train_val_df = pd.concat([train_df, val_df], ignore_index=True)
        X_train_val = train_val_df[feature_columns]
        y_train_val = train_val_df[target_col]
        best_model.fit(X_train_val, y_train_val)

        test_score = best_model.predict_proba(X_test)[:, 1]
        test_metrics = _compute_metrics(y_test, test_score, best_threshold)

        metric_map = {
            "auc_roc": test_metrics["auc_roc"],
            "auprc": test_metrics["auprc"],
            "precision": test_metrics["precision"],
            "recall": test_metrics["recall"],
            "f1": test_metrics["f1"],
            "accuracy": test_metrics["accuracy"],
            "fpr": test_metrics["fpr"],
            "selected_threshold": best_threshold,
        }
        if mlflow_enabled:
            mlflow.log_metrics({k: float(v) for k, v in metric_map.items()})
            mlflow.log_param("selected_model", best_name)

        model_dir = project_root / "models" / "trained"
        reports_dir = project_root / "reports" / "metrics"
        ensure_dirs(model_dir, reports_dir)

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        versioned_model_path = model_dir / f"{best_name}_{stamp}.joblib"
        latest_model_path = model_dir / "latest.joblib"
        legacy_root_path = project_root / "model.joblib"

        joblib.dump(best_model, versioned_model_path)
        joblib.dump(best_model, latest_model_path)
        joblib.dump(best_model, legacy_root_path)

        if mlflow_enabled:
            mlflow.log_artifact(str(versioned_model_path), artifact_path="model_export")
            mlflow.log_artifact(str(latest_model_path), artifact_path="model_export")

        model_info = {
            "model_name": best_name,
            "threshold": float(best_threshold),
            "feature_columns": feature_columns,
            "model_path": str(latest_model_path.relative_to(project_root)),
            "selected_at": datetime.now(timezone.utc).isoformat(),
        }
        inference_cfg.update(model_info)
        with open(project_root / "configs" / "inference.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(inference_cfg, f, sort_keys=False)

        summary_payload = {
            "candidate_results": candidate_rows,
            "best_validation": best_val_summary,
            "test_metrics": test_metrics,
            "model_info": model_info,
            "tracking_uri": tracking_uri,
            "run_id": run.info.run_id if mlflow_enabled and run is not None else "mlflow-disabled",
            "mlflow_enabled": mlflow_enabled,
        }

        with open(reports_dir / "training_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2)

        if mlflow_enabled:
            mlflow.log_dict(summary_payload, "reports/training_summary.json")

    print("Improved training complete")
    print(json.dumps({"selected_model": best_name, **metric_map}, indent=2))


def main() -> None:
    # Train baseline model first for comparison
    _train_baseline()
    # Then train improved models with threshold tuning
    train_models()


if __name__ == "__main__":
    main()
