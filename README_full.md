# MLOps Credit Fraud Detection (Comprehensive Runbook)

This document is an up-to-date, practical guide for running the project end-to-end in its current state.
It includes exact paths, command order, what each step does, expected outputs, and common mistakes.

## 1. Project Goal

Real-time fraud detection for credit card transactions using a reproducible MLOps workflow.

Why this matters:
- Fraud losses are high and detection must balance recall, precision, latency, and false positive rate.

## 2. Team Roles

| Role | Responsibilities | Bonus / Extra Features |
| :--- | :--- | :--- |
| Person 1 | Lead and Documentation | Streamlit App Integration |
| Person 2 | Data Management and DVC | SHAP Model Interpretability |
| Person 3 | Modeling and MLflow | Optuna Hyperparameter Tuning |
| Person 4 | CI/CD and Deployment | Cloud Deployment |
| Person 5 | Monitoring and Alerts | Slack/Email Notifications |
| Person 6 | Presentation and Video | Animated GIF Creation |

## 3. Success Metrics

Technical targets:
- AUPRC >= 0.85
- Recall >= 0.92
- Precision >= 0.80

Operational targets:
- Latency < 50 ms
- FPR < 5%
- Drift alert < 5 minutes

## 4. Dataset

- Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Raw file expected at: `data/raw/creditcard.csv`
- DVC metadata file: `data/raw/creditcard.csv.dvc`
- Current dataset size in this repo context: 284,807 rows (highly imbalanced)

## 5. Current State Summary (Latest)

What is currently implemented and validated:
- DVC pipeline with stages `prepare -> train -> evaluate`.
- Training includes:
	- Baseline model: `baseline_logistic`.
	- Main candidates: `logistic_regression`, `random_forest` with threshold tuning.
- MLflow tracking to local SQLite DB: `mlflow.db`.
- Baseline registration utility script: `scripts/register_baseline_model.py`.
- Test suite is runnable and passing locally.

Recent verified commands in this workspace:
- `python -m src.ml.train` ran successfully.
- `python -m dvc repro` ran successfully.
- `python -m pytest -q` passed.
- `python -m mlflow ui ...` served at `http://127.0.0.1:5000`.

## 6. Updated Folder Structure (Important Paths)

```text
Project_mlops_credit_fraud_detection/
|-- .dvc/
|-- .github/
|-- .venv/
|-- codes/
|   |-- ml/
|   |   |-- common/
|   |   |   `-- paths.py
|   |   |-- data/
|   |   |   `-- prepare_data.py
|   |   `-- modeling/
|   |       |-- train_model.py
|   |       `-- evaluate_model.py
|   `-- __init__.py
|-- configs/
|   |-- base.yaml
|   |-- training.yaml
|   `-- inference.yaml
|-- data/
|   |-- raw/
|   |   |-- creditcard.csv
|   |   `-- creditcard.csv.dvc
|   `-- processed/
|       |-- train.parquet
|       |-- val.parquet
|       |-- test.parquet
|       |-- reference.parquet
|       `-- current.parquet
|-- docs/
|   |-- how_to_run.md
|   |-- reproducibility_proof.md
|   `-- screenshots/
|-- infra/
|   `-- prometheus/
|-- models/
|   `-- trained/
|       |-- latest.joblib
|       |-- baseline_logistic_*.joblib
|       `-- random_forest_*.joblib
|-- reports/
|   |-- data_quality_report.json
|   |-- dataset_fingerprint.json
|   `-- metrics/
|       |-- baseline_metrics.json
|       |-- training_summary.json
|       `-- test_metrics.json
|-- scripts/
|   |-- check_mlflow_status.py
|   |-- log_tests_mlflow.py
|   `-- register_baseline_model.py
|-- src/
|   |-- app/
|   |-- database/
|   |-- ml/
|   |   |-- data.py
|   |   |-- train.py
|   |   `-- evaluate.py
|   `-- monitoring/
|-- tests/
|   `-- test_model.py
|-- dvc.yaml
|-- dvc.lock
|-- mlflow.db
|-- model.joblib
|-- README.md
|-- README_hieu.md
`-- requirements.txt
```

## 7. Absolute Path Safety (Avoid Wrong Folder Errors)

If commands fail with import/path errors, it is often because the terminal is not in project root.

Use this exact root path on your machine:

```powershell
$PROJECT_ROOT = "D:\Desktop_informations\SGK năm 3\SGK kì 2 năm 3\MLOPs\Project_mlops_credit_fraud_detection"
Set-Location "$PROJECT_ROOT"
```

Confirm location before running anything:

```powershell
Get-Location
```

Expected output should end with:
`...\Project_mlops_credit_fraud_detection`

## 8. Full Run Guide (Step-by-Step with Explanations)

### Step 1: Create and activate virtual environment

Purpose:
- Isolate dependencies and avoid conflicts with global Python/Conda.

Commands (Windows PowerShell):

```powershell
Set-Location "$PROJECT_ROOT"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 2: Install dependencies

Purpose:
- Install required packages for DVC, MLflow, modeling, and tests.

```powershell
Set-Location "$PROJECT_ROOT"
pip install -r requirements.txt
```

### Step 3: Ensure dataset file exists

Purpose:
- The data pipeline requires this exact file.

Required file path:
- `data/raw/creditcard.csv`

Quick check:

```powershell
Test-Path "$PROJECT_ROOT\data\raw\creditcard.csv"
```

Expected result:
- `True`

### Step 4: Run DVC pipeline (prepare -> train -> evaluate)

Purpose:
- Reproduce the full artifact workflow with data prep, training, and evaluation.

Command:

```powershell
Set-Location "$PROJECT_ROOT"
python -m dvc repro
```

What this does internally (from `dvc.yaml`):
- `prepare`:
	- Command: `.\.venv\Scripts\python.exe -m codes.ml.data.prepare_data`
	- Outputs processed parquet files and data reports.
- `train`:
	- Command: `.\.venv\Scripts\python.exe -m codes.ml.modeling.train_model`
	- Trains baseline + improved models and writes training metrics.
- `evaluate`:
	- Command: `.\.venv\Scripts\python.exe -m codes.ml.modeling.evaluate_model`
	- Writes final test metrics report.

Verify pipeline state:

```powershell
python -m dvc status
```

Expected:
- `Data and pipelines are up to date.`

### Step 5: Run training directly (requested member flow)

Purpose:
- Run training command explicitly outside DVC for local iteration.

```powershell
Set-Location "$PROJECT_ROOT"
python -m src.ml.train
```

Expected training runs in MLflow:
- `week1-baseline-logreg`
- `improved-threshold-tuned-models`

Expected model artifacts:
- `models/trained/baseline_logistic_*.joblib`
- `models/trained/latest.joblib`
- `model.joblib`

### Step 6: Run evaluation directly (optional but recommended)

Purpose:
- Recompute and verify test metrics from current `latest.joblib`.

```powershell
Set-Location "$PROJECT_ROOT"
python -m src.ml.evaluate
```

Expected report:
- `reports/metrics/test_metrics.json`

### Step 7: Run tests

Purpose:
- Validate expected behavior after changes.

```powershell
Set-Location "$PROJECT_ROOT"
python -m pytest -q
```

Expected:
- All tests pass.

### Step 8: Open MLflow UI

Purpose:
- Inspect experiments, runs, parameters, metrics, and artifacts.

Command:

```powershell
Set-Location "$PROJECT_ROOT"
python -m mlflow ui --backend-store-uri "sqlite:///D:/Desktop_informations/SGK năm 3/SGK kì 2 năm 3/MLOPs/Project_mlops_credit_fraud_detection/mlflow.db" --host 127.0.0.1 --port 5000
```

Open browser:
- http://127.0.0.1:5000

Notes:
- On Windows, MLflow may show a warning about job backend support. This does not block run tracking or UI viewing.
- Use the `credit-fraud` experiment in the UI.

### Step 9: Register baseline model in MLflow Registry (Person 3)

Purpose:
- Keep a stable baseline reference model in registry for comparison against improved models.

Command:

```powershell
Set-Location "$PROJECT_ROOT"
python scripts/register_baseline_model.py
```

Expected registry output:
- Model name: `credit-fraud-baseline`
- Alias: `baseline`
- Version: incremented model version

### Step 10: (Optional) Log test run to MLflow

Purpose:
- Track test quality history in MLflow alongside model runs.

```powershell
Set-Location "$PROJECT_ROOT"
python scripts/log_tests_mlflow.py
```

## 9. Minimum Command Set (Copy/Paste)

```powershell
$PROJECT_ROOT = "D:\Desktop_informations\SGK năm 3\SGK kì 2 năm 3\MLOPs\Project_mlops_credit_fraud_detection"
Set-Location "$PROJECT_ROOT"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m dvc repro
python -m src.ml.train
python -m pytest -q
python scripts/register_baseline_model.py
python -m mlflow ui --backend-store-uri "sqlite:///D:/Desktop_informations/SGK năm 3/SGK kì 2 năm 3/MLOPs/Project_mlops_credit_fraud_detection/mlflow.db" --host 127.0.0.1 --port 5000
```

## 10. Common Errors and Fixes

1. Error: `ModuleNotFoundError: No module named 'codes'`
- Cause: running from wrong directory or running file path directly.
- Fix:
	- `Set-Location "$PROJECT_ROOT"`
	- Use module form, for example `python -m src.ml.train`.

2. Error: `dvc is not recognized`
- Cause: shell cannot find global dvc executable.
- Fix:
	- Use `python -m dvc repro`.

3. Error: missing `creditcard.csv`
- Cause: dataset not placed in expected path.
- Fix:
	- Put file at `data/raw/creditcard.csv`.

4. MLflow UI opens but runs look empty
- Cause: wrong experiment selected.
- Fix:
	- Switch experiment to `credit-fraud`.

## 11. Week 1 Delivery Checklist

- Run training locally (`python -m src.ml.train`).
- Open MLflow UI (`http://127.0.0.1:5000`).
- Take screenshot(s) of runs for submission evidence.
- Person 3 registers baseline model (`python scripts/register_baseline_model.py`).
- Keep README sections synchronized with current commands and file paths.

