import subprocess
import sys
import time
from pathlib import Path

import mlflow


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
MLFLOW_DB_URI = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_DB_URI)
    mlflow.set_registry_uri(MLFLOW_DB_URI)
    mlflow.set_experiment("credit-fraud")

    start = time.time()
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-v"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    duration = time.time() - start

    with mlflow.start_run(run_name="week1-pytest-validation"):
        mlflow.set_tag("pipeline_step", "tests")
        mlflow.set_tag("stage", "week1-day6-7")
        mlflow.log_metric("tests_exit_code", float(result.returncode))
        mlflow.log_metric("tests_duration_sec", float(duration))
        mlflow.log_metric("tests_passed", float(result.returncode == 0))
        mlflow.log_text(result.stdout, "tests/pytest_stdout.txt")
        mlflow.log_text(result.stderr, "tests/pytest_stderr.txt")

    print("Pytest run logged to MLflow")
    print(f"Tracking URI: {MLFLOW_DB_URI}")
    print(f"Exit code: {result.returncode}")


if __name__ == "__main__":
    main()
