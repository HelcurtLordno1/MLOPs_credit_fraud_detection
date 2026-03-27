import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
MLFLOW_DB_URI = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_DB_URI)
    mlflow.set_registry_uri(MLFLOW_DB_URI)
    mlflow.set_experiment("credit-fraud")

    with open(PROJECT_ROOT / "configs" / "training.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    train = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "train.parquet")
    test = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "test.parquet")

    X_train = train.drop("Class", axis=1)
    y_train = train["Class"]
    X_test = test.drop("Class", axis=1)
    y_test = test["Class"]

    model_cfg = config.get("model", {})
    class_weight = model_cfg.get("class_weight", "balanced")
    logistic_cfg = model_cfg.get("logistic", {})
    thresholds = config.get("thresholds", {})

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight=class_weight,
                    C=float(logistic_cfg.get("C", 1.0)),
                    max_iter=int(logistic_cfg.get("max_iter", 1000)),
                    solver=str(logistic_cfg.get("solver", "lbfgs")),
                    random_state=42,
                ),
            ),
        ]
    )

    with mlflow.start_run(run_name="week1-baseline-logreg") as run:
        mlflow.set_tag("project", "credit-fraud-detection")
        mlflow.set_tag("stage", "week1-day4-5")
        mlflow.set_tag("pipeline_step", "model_training")

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("class_weight", class_weight)
        mlflow.log_param("logistic_C", float(logistic_cfg.get("C", 1.0)))
        mlflow.log_param("logistic_max_iter", int(logistic_cfg.get("max_iter", 1000)))
        mlflow.log_param("logistic_solver", str(logistic_cfg.get("solver", "lbfgs")))
        mlflow.log_param("train_rows", int(len(train)))
        mlflow.log_param("test_rows", int(len(test)))
        mlflow.log_param("train_fraud_ratio", float(y_train.mean()))
        mlflow.log_param("test_fraud_ratio", float(y_test.mean()))

        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_score = pipe.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_score)
        auprc = average_precision_score(y_test, y_score)
        recall = recall_score(y_test, y_pred, zero_division=0)
        precision = precision_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        accuracy = accuracy_score(y_test, y_pred)

        metrics = {
            "auc_roc": float(auc),
            "auprc": float(auprc),
            "recall": float(recall),
            "precision": float(precision),
            "f1": float(f1),
            "accuracy": float(accuracy),
        }
        mlflow.log_metrics(metrics)

        # Threshold pass/fail values are logged as 1.0 or 0.0 for dashboard filtering.
        mlflow.log_metric("pass_min_auc", float(auc >= float(thresholds.get("min_auc", 0.90))))
        mlflow.log_metric("pass_min_auprc", float(auprc >= float(thresholds.get("min_auprc", 0.85))))
        mlflow.log_metric("pass_min_recall", float(recall >= float(thresholds.get("min_recall", 0.92))))
        mlflow.log_metric("pass_min_precision", float(precision >= float(thresholds.get("min_precision", 0.80))))

        model_path = PROJECT_ROOT / "model.joblib"
        joblib.dump(pipe, model_path)
        mlflow.log_artifact(str(model_path), artifact_path="model_export")

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        conf = confusion_matrix(y_test, y_pred)
        mlflow.log_dict(report, "reports/classification_report.json")
        mlflow.log_dict(
            {
                "tn": int(conf[0, 0]),
                "fp": int(conf[0, 1]),
                "fn": int(conf[1, 0]),
                "tp": int(conf[1, 1]),
            },
            "reports/confusion_matrix.json",
        )
        mlflow.log_text(json.dumps(thresholds, indent=2), "reports/thresholds.json")
        mlflow.log_artifact(str(PROJECT_ROOT / "configs" / "training.yaml"), artifact_path="config")

        mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="model",
            input_example=X_train.head(1),
        )

        model_uri = f"runs:/{run.info.run_id}/model"
        registered = mlflow.register_model(model_uri=model_uri, name="credit-fraud-model")
        mlflow.set_tag("registered_model_name", "credit-fraud-model")
        mlflow.set_tag("registered_model_version", str(registered.version))

    print("Model trained and richly logged to MLflow!")
    print(f"Tracking URI: {MLFLOW_DB_URI}")
    print(f"AUC-ROC: {auc:.4f}")
    print(f"AUPRC: {auprc:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")


if __name__ == "__main__":
    main()
