from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from fraud_detection.data.schema import (
    INDEX_COLUMN,
    SOURCE_COLUMNS,
    SOURCE_INDEX_COLUMN,
    TARGET_COLUMN,
    build_read_dtypes,
    validate_dataset,
)
from fraud_detection.data.features import engineer_all_features
from fraud_detection.data.pipeline import create_train_val_test_splits, load_splits
from fraud_detection.modeling.evaluate import analyze_bias_variance, compute_metrics, evaluate_model
from fraud_detection.modeling.train import prepare_features, train_lightgbm, train_logistic_regression
from fraud_detection.utils.paths import ensure_dirs, find_project_root

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None

DEFAULT_DATA_CONFIG: dict[str, Any] = {
    "train_csv": "fraudTrain.csv",
    "test_csv": "fraudTest.csv",
    "processed_dir": "data/processed",
    "reports_dir": "reports/metrics",
    "summary_name": "dataset_summary.json",
    "sample_rows": None,
}

DEFAULT_TRAIN_CONFIG: dict[str, Any] = {
    "experiment": {
        "name": "fraud-detector",
        "tracking_uri": "./mlruns",
        "registry_uri": "./mlruns",
        "registered_model_name": "fraud-detector",
    },
    "model": {
        "run_name": "credit-fraud-day2",
        "logistic": {
            "C": 1.0,
            "max_iter": 1000,
            "solver": "lbfgs",
            "class_weight": "balanced",
            "random_state": 42,
        },
        "lightgbm": {
            "n_estimators": 400,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 1.0,
            "reg_alpha": 0.0,
            "min_child_samples": 50,
            "objective": "binary",
            "metric": "auc",
            "random_state": 42,
            "n_jobs": -1,
        },
    },
    "evaluation": {
        "threshold": 0.5,
        "variance_threshold": 0.05,
        "overfitting_threshold": 0.15,
    },
}

DEFAULT_MONITORING_CONFIG: dict[str, Any] = {
    "reference_path": "data/processed/reference.parquet",
    "current_path": "data/processed/current.parquet",
    "report_path": "reports/drift/drift_report.json",
    "numeric_threshold": 0.05,
    "target_threshold": 0.05,
}


def load_data_config() -> dict[str, Any]:
    config_path = find_project_root() / "configs" / "data.yaml"
    data_cfg = copy.deepcopy(DEFAULT_DATA_CONFIG)
    if not config_path.exists() or yaml is None:
        return data_cfg

    with open(config_path, encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    data_cfg.update(payload.get("data", {}))
    return data_cfg


def load_train_config() -> dict[str, Any]:
    config_path = find_project_root() / "configs" / "train.yaml"
    train_cfg = copy.deepcopy(DEFAULT_TRAIN_CONFIG)
    if not config_path.exists() or yaml is None:
        return train_cfg

    with open(config_path, encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    train_cfg.update({k: v for k, v in payload.items() if k in train_cfg})
    for section in ["experiment", "model", "evaluation"]:
        if section in payload:
            train_cfg.setdefault(section, {}).update(payload.get(section, {}))
    return train_cfg


def load_monitoring_config() -> dict[str, Any]:
    config_path = find_project_root() / "configs" / "monitoring.yaml"
    monitoring_cfg = copy.deepcopy(DEFAULT_MONITORING_CONFIG)
    if not config_path.exists() or yaml is None:
        return monitoring_cfg

    with open(config_path, encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    monitoring_cfg.update(payload.get("monitoring", {}))
    return monitoring_cfg


def read_dataset(csv_path: Path, sample_rows: int | None = None) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset does not exist: {csv_path}")

    read_kwargs: dict[str, Any] = {
        "dtype": build_read_dtypes(),
        "usecols": SOURCE_COLUMNS,
    }
    if sample_rows is not None:
        read_kwargs["nrows"] = sample_rows

    frame = pd.read_csv(csv_path, **read_kwargs)
    renamed = frame.rename(columns={SOURCE_INDEX_COLUMN: INDEX_COLUMN})
    return validate_dataset(renamed)


def summarize_dataset(frame: pd.DataFrame) -> dict[str, Any]:
    timestamps = pd.to_datetime(
        frame["trans_date_trans_time"],
        format="%Y-%m-%d %H:%M:%S",
        errors="raise",
    )
    return {
        "rows": int(len(frame)),
        "fraud_rows": int(frame[TARGET_COLUMN].sum()),
        "fraud_rate": float(frame[TARGET_COLUMN].mean()),
        "amount": {
            "min": float(frame["amt"].min()),
            "mean": float(frame["amt"].mean()),
            "max": float(frame["amt"].max()),
        },
        "time_range": [
            timestamps.min().isoformat(),
            timestamps.max().isoformat(),
        ],
        "unique_cards": int(frame["cc_num"].nunique()),
        "unique_merchants": int(frame["merchant"].nunique()),
        "unique_categories": int(frame["category"].nunique()),
    }


def _load_raw_sources(project_root: Path, data_cfg: dict[str, Any]) -> pd.DataFrame:
    source_paths = [
        project_root / str(data_cfg.get("train_csv", "fraudTrain.csv")),
        project_root / str(data_cfg.get("test_csv", "fraudTest.csv")),
    ]

    frames: list[pd.DataFrame] = []
    for csv_path in source_paths:
        frame = read_dataset(csv_path=csv_path, sample_rows=None if data_cfg.get("sample_rows") is None else int(data_cfg.get("sample_rows")))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def prepare_pipeline_datasets(sample_rows: int | None = None) -> dict[str, Any]:
    project_root = find_project_root()
    data_cfg = load_data_config()
    if sample_rows is not None:
        data_cfg["sample_rows"] = sample_rows

    processed_dir = project_root / str(data_cfg.get("processed_dir", "data/processed"))
    reports_dir = project_root / str(data_cfg.get("reports_dir", "reports/metrics"))
    summary_name = str(data_cfg.get("summary_name", "day1_data_summary.json"))
    ensure_dirs(processed_dir, reports_dir)

    combined = _load_raw_sources(project_root, data_cfg)
    combined = validate_dataset(combined).copy()
    engineered = engineer_all_features(combined)
    train_df, val_df, test_df = create_train_val_test_splits(engineered)

    outputs = {
        "train": processed_dir / "train.parquet",
        "val": processed_dir / "val.parquet",
        "test": processed_dir / "test.parquet",
    }

    for name, frame in [("train", train_df), ("val", val_df), ("test", test_df)]:
        frame.to_parquet(outputs[name], index=False)

    summary = {
        "day": 1,
        "stage": "Data Preparation & Feature Engineering",
        "raw_rows": int(len(combined)),
        "original_features": int(len(combined.columns) - 1),
        "engineered_features": int(engineered.shape[1]),
        "splits": {
            "train": {"rows": int(len(train_df)), "fraud_rate": float(train_df[TARGET_COLUMN].mean())},
            "val": {"rows": int(len(val_df)), "fraud_rate": float(val_df[TARGET_COLUMN].mean())},
            "test": {"rows": int(len(test_df)), "fraud_rate": float(test_df[TARGET_COLUMN].mean())},
        },
        "outputs": {name: str(path.relative_to(project_root)) for name, path in outputs.items()},
    }

    summary_path = reports_dir / summary_name
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    return summary


def _load_processed_splits(project_root: Path, data_cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    processed_dir = project_root / str(data_cfg.get("processed_dir", "data/processed"))
    return load_splits(processed_dir)


def train_pipeline() -> dict[str, Any]:
    project_root = find_project_root()
    data_cfg = load_data_config()
    train_cfg = load_train_config()

    train_df, val_df, test_df = _load_processed_splits(project_root, data_cfg)
    X_train, y_train = prepare_features(train_df, target_col=str(data_cfg.get("target_column", TARGET_COLUMN)))
    X_val, y_val = prepare_features(val_df, target_col=str(data_cfg.get("target_column", TARGET_COLUMN)))
    X_test, y_test = prepare_features(test_df, target_col=str(data_cfg.get("target_column", TARGET_COLUMN)))

    logistic_params = train_cfg["model"]["logistic"]
    lightgbm_params = train_cfg["model"]["lightgbm"]
    logistic_model = train_logistic_regression(X_train, y_train, logistic_params)
    lightgbm_model = train_lightgbm(X_train, y_train, lightgbm_params)

    models = {"Logistic Regression": logistic_model, "LightGBM": lightgbm_model}
    split_bundle = {"train": (X_train, y_train), "val": (X_val, y_val), "test": (X_test, y_test)}

    all_results: dict[str, dict[str, dict[str, float]]] = {}
    best_model_name = ""
    best_val_auprc = float("-inf")
    best_model = None
    best_run_id = None

    tracking_uri = train_cfg["experiment"].get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(train_cfg["experiment"].get("name", "fraud-detector"))

    with mlflow.start_run(run_name=train_cfg["model"].get("run_name", "credit-fraud-day2")) as run:
        mlflow.log_param("target_column", str(data_cfg.get("target_column", TARGET_COLUMN)))
        mlflow.log_param("feature_count", int(X_train.shape[1]))
        mlflow.log_param("train_rows", int(len(X_train)))

        for model_name, model in models.items():
            split_results: dict[str, dict[str, float]] = {}
            for split_name, (X, y) in split_bundle.items():
                probabilities = model.predict_proba(X)[:, 1]
                split_results[split_name] = compute_metrics(y, probabilities, threshold=0.5)
                for metric_name, metric_value in split_results[split_name].items():
                    mlflow.log_metric(f"{model_name.lower().replace(' ', '_')}_{split_name}_{metric_name}", metric_value)

            all_results[model_name] = split_results
            val_auprc = split_results["val"]["auprc"]
            if val_auprc > best_val_auprc:
                best_val_auprc = val_auprc
                best_model_name = model_name
                best_model = model

        mlflow.log_param("best_model", best_model_name)
        mlflow.log_metric("best_val_auprc", best_val_auprc)

        # Log plain validation metrics for promotion comparison across runs.
        best_val_metrics = all_results[best_model_name]["val"]
        mlflow.log_metric("val_recall", best_val_metrics["recall"])
        mlflow.log_metric("val_precision", best_val_metrics["precision"])
        mlflow.log_metric("val_auprc", best_val_metrics["auprc"])

        model_path = project_root / "models" / "trained" / "best_model.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, model_path)
        mlflow.log_artifact(str(model_path), artifact_path="artifacts")
        mlflow.sklearn.log_model(best_model, artifact_path="model")
        best_run_id = run.info.run_id

    metrics_dir = project_root / "reports" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "train_metrics.json"
    legacy_metrics_path = project_root / "reports" / "train_metrics.json"
    report = {
        "best_model": best_model_name,
        "best_val_auprc": best_val_auprc,
        "best_run_id": best_run_id,
        "results": all_results,
    }
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    # Backward-compatible path for existing notebooks.
    with open(legacy_metrics_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))
    return report


def evaluate_pipeline() -> dict[str, Any]:
    project_root = find_project_root()
    data_cfg = load_data_config()

    train_df, val_df, test_df = _load_processed_splits(project_root, data_cfg)
    model_path = project_root / "models" / "trained" / "best_model.joblib"
    model = joblib.load(model_path)

    X_train, y_train = prepare_features(train_df, target_col=str(data_cfg.get("target_column", TARGET_COLUMN)))
    X_val, y_val = prepare_features(val_df, target_col=str(data_cfg.get("target_column", TARGET_COLUMN)))
    X_test, y_test = prepare_features(test_df, target_col=str(data_cfg.get("target_column", TARGET_COLUMN)))

    results = evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test)
    bias_variance = analyze_bias_variance(results, metric="auprc")

    report = {
        "model_path": str(model_path),
        "metrics": results,
        "bias_variance": bias_variance,
    }

    metrics_path = project_root / "reports" / "metrics" / "test_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))
    return report


def promote_pipeline() -> dict[str, Any]:
    from fraud_detection.monitoring.promotion import ModelPromoter, save_promotion_report

    project_root = find_project_root()
    train_cfg = load_train_config()

    train_metrics_path = project_root / "reports" / "metrics" / "train_metrics.json"
    evaluate_metrics_path = project_root / "reports" / "metrics" / "test_metrics.json"
    with open(train_metrics_path, encoding="utf-8") as handle:
        train_metrics = json.load(handle)
    with open(evaluate_metrics_path, encoding="utf-8") as handle:
        evaluate_metrics = json.load(handle)

    promoter = ModelPromoter(
        tracking_uri=train_cfg["experiment"].get("tracking_uri"),
        registry_uri=train_cfg["experiment"].get("registry_uri"),
    )
    challenger_metrics = train_metrics["results"][train_metrics["best_model"]]["val"]
    champion_metrics = promoter.load_champion_metrics(train_cfg["experiment"].get("registered_model_name", "fraud-detector"))

    should_promote, reason = promoter.should_promote(
        challenger_metrics=challenger_metrics,
        champion_metrics=champion_metrics,
        min_precision_ratio=0.95,
    )

    promotion_report = {
        "challenger_run_id": train_metrics.get("best_run_id"),
        "champion_metrics": champion_metrics,
        "challenger_metrics": challenger_metrics,
        "promoted": should_promote,
        "reason": reason,
        "evaluation_metrics": evaluate_metrics,
    }

    if should_promote:
        promoted_version = promoter.promote_model(
            run_id=str(train_metrics.get("best_run_id")),
            model_name=train_cfg["experiment"].get("registered_model_name", "fraud-detector"),
            stage="Production",
        )
        promotion_report["promotion"] = promoted_version

    report_path = project_root / "models" / "registry" / "last_promotion.json"
    save_promotion_report(
        path=report_path,
        challenger_run_id=str(train_metrics.get("best_run_id")),
        champion_run_id=promotion_report.get("promotion", {}).get("version"),
        promoted=should_promote,
        reason=reason,
        challenger_metrics=challenger_metrics,
        champion_metrics=champion_metrics,
    )
    print(json.dumps(promotion_report, indent=2))
    return promotion_report


def _resolve_existing_path(project_root: Path, configured_path: str, fallback_path: str) -> Path:
    configured = project_root / configured_path
    if configured.exists():
        return configured
    fallback = project_root / fallback_path
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"Neither configured path '{configured}' nor fallback '{fallback}' exists"
    )


def drift_pipeline() -> dict[str, Any]:
    from fraud_detection.monitoring.drift import (
        detect_feature_drift,
        detect_target_drift,
        summarize_drift,
    )

    project_root = find_project_root()
    monitoring_cfg = load_monitoring_config()

    reference_path = _resolve_existing_path(
        project_root,
        str(monitoring_cfg.get("reference_path", "data/processed/reference.parquet")),
        "data/processed/train.parquet",
    )
    current_path = _resolve_existing_path(
        project_root,
        str(monitoring_cfg.get("current_path", "data/processed/current.parquet")),
        "data/processed/test.parquet",
    )

    reference_df = pd.read_parquet(reference_path)
    current_df = pd.read_parquet(current_path)

    numeric_threshold = float(
        monitoring_cfg.get(
            "numeric_threshold",
            monitoring_cfg.get("psi_alert_threshold", 0.05),
        )
    )
    target_threshold = float(monitoring_cfg.get("target_threshold", 0.05))

    feature_results = detect_feature_drift(
        reference_df,
        current_df,
        numeric_threshold=numeric_threshold,
    )
    summary = summarize_drift(feature_results)

    target_results: dict[str, Any] | None = None
    if TARGET_COLUMN in reference_df.columns and TARGET_COLUMN in current_df.columns:
        target_results = detect_target_drift(
            reference_df[TARGET_COLUMN],
            current_df[TARGET_COLUMN],
            threshold=target_threshold,
        )

    report = {
        "reference_path": str(reference_path.relative_to(project_root)),
        "current_path": str(current_path.relative_to(project_root)),
        "summary": summary,
        "target": target_results,
        "features": feature_results,
    }

    report_path = project_root / str(
        monitoring_cfg.get("report_path", "reports/drift/drift_report.json")
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))
    return report


def prepare_datasets(sample_rows: int | None = None) -> dict[str, Any]:
    project_root = find_project_root()
    data_cfg = load_data_config()
    effective_sample_rows = sample_rows or data_cfg.get("sample_rows")
    row_limit = int(effective_sample_rows) if effective_sample_rows else None

    processed_dir = project_root / str(data_cfg.get("processed_dir", "data/processed"))
    reports_dir = project_root / str(data_cfg.get("reports_dir", "reports/metrics"))
    summary_name = str(data_cfg.get("summary_name", "dataset_summary.json"))

    ensure_dirs(processed_dir, reports_dir)

    source_paths = {
        "fraudTrain": project_root / str(data_cfg.get("train_csv", "fraudTrain.csv")),
        "fraudTest": project_root / str(data_cfg.get("test_csv", "fraudTest.csv")),
    }

    summary: dict[str, Any] = {
        "target_column": TARGET_COLUMN,
        "sample_rows": row_limit,
        "datasets": {},
    }

    for dataset_name, csv_path in source_paths.items():
        frame = read_dataset(csv_path=csv_path, sample_rows=row_limit)
        output_path = processed_dir / f"{dataset_name}.csv"
        frame.to_csv(output_path, index=False)
        summary["datasets"][dataset_name] = {
            "source_csv": str(csv_path.relative_to(project_root)),
            "processed_csv": str(output_path.relative_to(project_root)),
            **summarize_dataset(frame),
        }

    summary_path = reports_dir / summary_name
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    return summary


def show_paths() -> dict[str, str]:
    project_root = find_project_root()
    data_cfg = load_data_config()
    paths = {
        "train_csv": str((project_root / str(data_cfg["train_csv"])).resolve()),
        "test_csv": str((project_root / str(data_cfg["test_csv"])).resolve()),
        "processed_dir": str((project_root / str(data_cfg["processed_dir"])).resolve()),
        "reports_dir": str((project_root / str(data_cfg["reports_dir"])).resolve()),
    }
    print(json.dumps(paths, indent=2))
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the fraud detection MLOps pipeline."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help="Validate fraudTrain.csv and fraudTest.csv and save cleaned CSV outputs",
    )
    prepare.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="Optional row cap for a smaller validation run",
    )

    subparsers.add_parser("paths", help="Show resolved dataset and output paths")

    subparsers.add_parser("train", help="Train models, log to MLflow, and save the best model")
    subparsers.add_parser("evaluate", help="Evaluate the saved model on train/val/test splits")
    subparsers.add_parser("promote", help="Compare challenger vs champion and promote if approved")
    subparsers.add_parser("drift", help="Run feature/target drift monitoring and write report")
    tune = subparsers.add_parser("tune", help="Run Day 4 hyperparameter tuning and save tuning artifacts")
    tune.add_argument("--n-trials", type=int, default=15, help="Optuna trials per model")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "prepare":
        prepare_pipeline_datasets(sample_rows=args.sample_rows)
    elif args.command == "train":
        train_pipeline()
    elif args.command == "evaluate":
        evaluate_pipeline()
    elif args.command == "promote":
        promote_pipeline()
    elif args.command == "drift":
        drift_pipeline()
    elif args.command == "tune":
        from fraud_detection.modeling.fine_tune import run_fine_tuning_pipeline

        run_fine_tuning_pipeline(n_trials=args.n_trials)
    elif args.command == "paths":
        show_paths()
    else:
        parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
