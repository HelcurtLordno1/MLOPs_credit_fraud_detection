# Project MLOps Credit Fraud Detection

## Team Roles
- Person 1: Lead + Documentation + Streamlit bonus
- Person 2: Data + DVC + SHAP bonus
- Person 3: Modeling + MLflow + Optuna bonus
- Person 4: CI/CD + Deployment + Cloud deployment bonus
- Person 5: Monitoring + Alerts + Slack/email bonus
- Person 6: Presentation + Video + Animated GIF bonus

## Problem Definition
- Use case: Real-time fraud detection for banks.
- Motivation: Banks lose around $32B yearly to fraud.

## Success Metrics
Technical metrics:
- AUPRC >= 0.85
- Recall >= 0.92
- Precision >= 0.80

Business/operational metrics:
- Latency < 50 ms
- FPR < 5%
- Drift alert < 5 minutes

## Dataset
- Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- File: `creditcard.csv`

## Quick Start
1. Install dependencies.
2. Put `creditcard.csv` into `data/raw/`.
3. Run `dvc init`.
4. Run `dvc repro` to generate processed datasets.
5. Run `python -m src.ml.train` to train the baseline model.
6. Run `mlflow ui` and open http://127.0.0.1:5000 to inspect runs.

---

## Modeling *(Person 3 - Week 1, Day 4-5)*

### Overview
Current baseline uses Logistic Regression with `class_weight=balanced` and logs runs to MLflow.

### Training Script
```bash
python -m src.ml.train
```

Config file: `configs/training.yaml`

### MLflow Experiment Tracking
- Experiment name: `credit-fraud`
- Tracked metrics: `auc_roc`, `auprc`, `recall`, `precision`
- Tracked params: `model_type`, `class_weight`

```bash
mlflow ui --host 127.0.0.1 --port 5000
```

### Artifacts
- Trained model: `model.joblib`
- MLflow run metadata: `mlruns/`

## Reproducibility *(Week 1, Day 6-7)*

Reference document: `docs/reproducibility_proof.md`

Run order:
1. `pip install -r requirements.txt`
2. `python -m dvc repro`
3. `python -m src.ml.train`
4. `python -m pytest -v`
