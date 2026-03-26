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

## Modeling *(Person 3 — Week 1, Day 4-5)*

### Overview
The modeling pipeline targets real-time fraud detection on the ULB Credit Card dataset (284,807 transactions, 0.172% fraud). The severe class imbalance is addressed through `class_weight=balanced` (LogReg) and `scale_pos_weight` (XGBoost).

### Training Script
```bash
# Week 1 — Logistic Regression baseline
python -m src.ml.train --model logistic

# Week 2 — XGBoost (no tuning)
python -m src.ml.train --model xgb

# Week 2 — XGBoost + Optuna hyperparameter search (20 trials)
python -m src.ml.train --model xgb --tune
```

Config file: `configs/training.yaml`

### Model Versions

| Version | Algorithm | AUPRC | Recall | Precision | Registered As |
|---------|-----------|-------|--------|-----------|---------------|
| v1 (baseline) | LogisticRegression (balanced) | ~0.70 | ~0.88 | ~0.65 | `credit-fraud-model / baseline` |
| v2 (target) | XGBClassifier + Optuna | ≥ 0.85 | ≥ 0.92 | ≥ 0.80 | `credit-fraud-model / candidate` |

### MLflow Experiment Tracking
- **Experiment name:** `credit-fraud`
- **Tracked metrics:** `auc_roc`, `auprc`, `recall`, `precision`, `fpr`
- **Tracked params:** `model_type`, `tuned`, all Optuna best params
- **Model registry:** `credit-fraud-model` — stages: `None → Staging → Production`

```bash
# Launch MLflow UI (local)
mlflow ui --host 127.0.0.1 --port 5000
```

### Quality Gate (enforced in train.py)
All metrics are checked against thresholds defined in `configs/training.yaml`.  
A ⚠️ warning is printed for any metric that falls below the target.

| Metric | Threshold |
|--------|-----------|
| AUC-ROC | ≥ 0.90 |
| AUPRC | ≥ 0.85 |
| Recall | ≥ 0.92 |
| Precision | ≥ 0.80 |

### Artifacts
- Trained model: `model.joblib` (also logged to MLflow)
- MLflow run metadata: `.mlruns/` (auto-generated, git-ignored)

### Promotion Workflow (Week 2)
See `scripts/promote_model.py` — promotes the best run from `Staging` to `Production` in the MLflow registry after metric gate passes.
