# Fraud Detection MLOps

End-to-end fraud detection project for the PaySim credit transaction dataset. This repository was built from scratch in `E:\MLops\fraud-detection` and includes data preparation, model training, evaluation, promotion logic, API serving, a Streamlit operations UI, Docker Compose services, MLflow tracking, DVC, and GitHub Actions.

## What this project contains

- Temporal data validation and splitting with Pandera
- Shared train/serve feature engineering
- Logistic regression baseline and LightGBM candidate
- Optuna tuning for LightGBM
- AUPRC-based model selection
- Recall-first threshold tuning with FPR and precision guardrails
- MLflow experiment tracking and registry hooks
- FastAPI prediction service with Prometheus metrics
- Streamlit dashboard for prediction, drift, and training status
- DVC pipeline with Google Drive remote storage
- GitHub Actions for CI, retraining, and image publishing

## Prerequisites

Before running anything locally, make sure the machine has:

- Python `3.11` recommended
- `pip`
- `git`
- `docker` and `docker compose`
- `dvc` with Google Drive support, installed through `pip install -e .[dev]`

## Project setup

1. Clone the repository and move into it.

```bash
git clone <your-repo-url>
cd fraud-detection
```

2. Create and activate a virtual environment.

```bash
python -m venv .venv
.venv\Scripts\activate
```

3. Install the project and development dependencies.

```bash
pip install --upgrade pip
pip install -e .[dev]
```

4. Copy the example environment file if you want to customize ports or local service settings.

```bash
copy .env.example .env
```

## Dataset and DVC

The raw dataset is expected at:

```text
data/raw/paysim_fraud.zip
```

You have two options:

1. Pull it from DVC:

```bash
dvc pull data/raw/paysim_fraud.zip.dvc
```

2. Or place the dataset archive manually at `data/raw/paysim_fraud.zip`.

### Google Drive remote

The configured DVC remote is:

```text
gdrive://1FV_0iHdu6jZiXNKGc1ltYsWPLu-L15Mh
```

For local development:

- `dvc pull` can use the standard browser-based Google Drive login flow.
- If your current machine already has working DVC Google Drive auth, nothing else is required.

For automation:

- `GDRIVE_CREDENTIALS_DATA` can be provided as the full service-account JSON string.
- The shared Drive folder must be accessible to that service account.
- If `GDRIVE_CREDENTIALS_DATA` is not provided, the retrain workflow assumes the self-hosted runner already has valid Google Drive auth configured.

## Run the pipeline locally

### Option 1: step by step

Run each stage explicitly:

```bash
python -m fraud_detection.cli prepare
python -m fraud_detection.cli train
python -m fraud_detection.cli evaluate
python -m fraud_detection.cli monitor
python -m fraud_detection.cli promote
```

What each command does:

- `prepare`: validates the raw dataset, engineers split-ready data, and writes train/validation/test/reference/current parquet files
- `train`: trains the baseline and candidate models, selects the best model, tunes the serving threshold, and writes the bundle and candidate manifest
- `evaluate`: evaluates the selected candidate on the holdout split and writes test metrics and PR curve artifacts
- `monitor`: compares reference and current slices and writes drift output
- `promote`: applies promotion gates and writes champion metadata plus the production bundle

### Option 2: run the DVC pipeline

```bash
dvc repro
```

The DVC stages are:

- `prepare`
- `train`
- `evaluate`
- `monitor`
- `promote`

### Smoke run on a smaller sample

Useful for quick checks before a full run:

```bash
python -m fraud_detection.cli prepare --sample-rows 300000
python -m fraud_detection.cli train --sample-rows 120000
python -m fraud_detection.cli evaluate
python -m fraud_detection.cli monitor
python -m fraud_detection.cli promote
```

## Start the local services

Bring up the full local stack:

```bash
docker compose up -d --build
```

Stop it:

```bash
docker compose down
```

Useful services:

- API health: `http://localhost:8000/api/v1/health`
- API docs: `http://localhost:8000/docs`
- Streamlit UI: `http://localhost:8501`
- MLflow: `http://localhost:5000`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`

To inspect logs:

```bash
docker compose logs -f api
docker compose logs -f streamlit
docker compose logs -f mlflow
```

## Prediction APIs

### Single prediction

Endpoint:

```text
POST /api/v1/predict
```

Required input fields:

- `step`
- `type`
- `amount`
- `nameOrig`
- `oldbalanceOrg`
- `newbalanceOrig`
- `nameDest`
- `oldbalanceDest`
- `newbalanceDest`

Example payload:

```json
{
  "step": 1,
  "type": "TRANSFER",
  "amount": 5000.0,
  "nameOrig": "C123456789",
  "oldbalanceOrg": 5000.0,
  "newbalanceOrig": 0.0,
  "nameDest": "C987654321",
  "oldbalanceDest": 0.0,
  "newbalanceDest": 5000.0
}
```

### Batch prediction

Endpoint:

```text
POST /api/v1/predict/batch
```

The Streamlit UI also supports CSV-based batch scoring with the same columns.

## Main output artifacts

After a successful run, the key files are:

- `models/trained/model_bundle.joblib`
- `models/trained/production_model.joblib`
- `models/registry/candidate.json`
- `models/registry/champion.json`
- `models/registry/last_promotion.json`
- `reports/metrics/train_metrics.json`
- `reports/metrics/test_metrics.json`
- `reports/drift/drift_report.json`
- `reports/figures/pr_curve.png`

## GitHub Actions

This repo includes:

- `ci.yml`: lint, mypy, tests, smoke training, and Docker build validation
- `retrain.yml`: scheduled or manual retraining on a self-hosted runner
- `cd.yml`: GHCR image build and push on tags or manual dispatch

### Secrets and runner requirements

For GitHub Actions, configure:

- `GDRIVE_CREDENTIALS_DATA` if the retrain runner should use a Google service account

The retrain workflow expects:

- a self-hosted runner with Python and system dependencies available
- access to the DVC Google Drive remote

The CD workflow expects:

- GitHub Packages permissions to push to `ghcr.io`

## Notes

- The dataset is highly imbalanced, so model selection is based on validation AUPRC instead of accuracy.
- Production thresholding is recall-first and constrained by minimum precision and maximum FPR settings in [train.yaml](E:/MLops/fraud-detection/configs/train.yaml).
- Docker Compose is the intended deployment path for this project size.
