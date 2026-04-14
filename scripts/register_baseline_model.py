from __future__ import annotations

from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    tracking_uri = f"sqlite:///{(root / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    client = MlflowClient()

    experiment = mlflow.get_experiment_by_name("credit-fraud")
    if experiment is None:
        raise RuntimeError("Experiment 'credit-fraud' not found")

    runs = client.search_runs(
        [experiment.experiment_id],
        "tags.mlflow.runName = 'week1-baseline-logreg'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("No baseline run found with name 'week1-baseline-logreg'")

    run = runs[0]
    run_id = run.info.run_id

    artifacts = client.list_artifacts(run_id, "model_export")
    if not artifacts:
        raise RuntimeError("No model artifacts found under model_export for baseline run")

    baseline_artifact = None
    for artifact in artifacts:
        file_name = artifact.path.split("/")[-1]
        if file_name.startswith("baseline_logistic_") and file_name.endswith(".joblib"):
            baseline_artifact = artifact.path
            break

    if baseline_artifact is None:
        baseline_artifact = artifacts[0].path

    model_name = "credit-fraud-baseline"
    try:
        client.create_registered_model(model_name)
    except Exception:
        # Registered model may already exist.
        pass

    source = f"runs:/{run_id}/{baseline_artifact}"
    model_version = client.create_model_version(name=model_name, source=source, run_id=run_id)
    client.set_registered_model_alias(model_name, "baseline", model_version.version)

    print(
        {
            "tracking_uri": tracking_uri,
            "model_name": model_name,
            "alias": "baseline",
            "version": model_version.version,
            "run_id": run_id,
            "source": source,
        }
    )


if __name__ == "__main__":
    main()
