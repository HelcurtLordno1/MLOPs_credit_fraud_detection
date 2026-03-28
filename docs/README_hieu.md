# MLOps Credit Fraud Detection (Project Snapshot)

Real-time machine learning pipeline for detecting fraudulent credit card transactions with a reproducible MLOps workflow.

## Team Roles

| Role | Responsibilities | Bonus / Extra Features |
| :--- | :--- | :--- |
| Person 1 | Lead and Documentation | Streamlit App Integration |
| Person 2 | Data Management and DVC | SHAP Model Interpretability |
| Person 3 | Modeling and MLflow | Optuna Hyperparameter Tuning |
| Person 4 | CI/CD and Deployment | Cloud Deployment |
| Person 5 | Monitoring and Alerts | Slack/Email Notifications |
| Person 6 | Presentation and Video | Animated GIF Creation |

## Problem Definition

- Use case: Real-time fraud detection for banking systems.
- Motivation: Financial institutions lose around $32B annually due to credit card fraud.

## Success Metrics

Technical targets:
- AUPRC >= 0.85
- Recall >= 0.92
- Precision >= 0.80

Operational targets:
- Latency < 50 ms
- FPR < 5%
- Drift alert < 5 minutes

## Current Progress (As of Week 1)

Completed:
- Day 0 repository setup and branch workflow
- Day 1 project structure and baseline README
- Day 2-3 data and DVC pipeline
- Day 4-5 baseline modeling files (`configs/training.yaml`, `src/ml/train.py`)
- Day 6-7 minimum test setup (`tests/test_model.py`)
- Reproducibility document (`docs/reproducibility_proof.md`)

Validated by execution in this workspace:
- `dvc repro`: success
- `python -m src.ml.train`: success
- `mlflow ui --host 127.0.0.1 --port 5000`: starts successfully
- `pytest -v`: passed (1 test)

Most recent baseline training output (local run):
- AUC-ROC: 0.9721
- AUPRC: 0.7190
- Recall: 0.9184
- Precision: 0.0610

Interpretation:
- Recall is near target, but precision/AUPRC are below target and need improvement in later modeling iterations.

## Dataset

- Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- File: `data/raw/creditcard.csv`
- Size in this project: 284,807 rows, highly imbalanced target class

## Folder Structure

```text
MLOPs_credit_fraud_detection/
|-- .github/workflows/
|-- configs/
|-- data/
|   |-- raw/
|   `-- processed/
|-- docker/
|-- docs/screenshots/
|-- infra/prometheus/
|-- reports/
|-- scripts/
|-- src/
|   |-- app/
|   |-- database/
|   |-- ml/
|   `-- monitoring/
|-- tests/
|-- .gitignore
|-- dvc.yaml
|-- README.md
|-- README_hieu.md
|-- requirements.txt
|-- Workflow_mlops.md
`-- how_to_run.md
```

## Key Commands

```bash
pip install -r requirements.txt
python -m dvc repro
python -m src.ml.train
python -m pytest -v
python -m mlflow ui --host 127.0.0.1 --port 5000
```

## Pending Items to Fully Close Week 1

- Register baseline model in MLflow Model Registry as `baseline`
- Collect screenshot evidence from all members
- Confirm all Week 1 PR reviews and tagging discipline (`week1-complete`)

