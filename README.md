# 🚀 End-to-End MLOps Fraud Detection System
## Credit Card Fraud Detection with Feature Engineering, Model Training & Promotion

## 🤝 Team Roles Summary

| Day | Role | Member | Core Responsibility |
|-----|------|--------|---------------------|
| 1 | Data Engineer | Member 1 | DVC data pipeline setup, feature engineering, train/val/test split outputs |
| 2 | ML Engineer | Member 2 | Model training, MLflow logging, imbalance-aware baselines and comparisons |
| 3 | MLOps Engineer | Member 3 | Evaluation gates, promotion policy, drift monitoring and reporting |
| 4 | Interface Engineer | Member 4 | API schema migration and frontend alignment for Kaggle transaction fields |
| 5 | Automation Architect | Member 5 | Container/runtime automation and deployment extension planning |
| 6 | QA & Documentation Lead | Member 6 | Reproducibility validation, final documentation, release readiness |

---

## 📌 Project Explanation (From Project Plan)

This repository is designed as a role-based MLOps delivery flow where each day adds one production layer on top of the previous day.

1. Day 0 establishes a clean repository baseline and DVC-first data ownership.
2. Day 1 builds the data foundation: schema-aware preprocessing, engineered features, and deterministic splits.
3. Day 2 trains fraud models with class-imbalance handling and logs experiments to MLflow.
4. Day 3 validates model quality, applies champion/challenger promotion rules, and produces drift analysis artifacts.
5. Day 4+ extends the system toward API contracts, frontend usability, deployment automation, and final quality assurance.

This structure ensures teammates can work in parallel while still sharing one reproducible, audited pipeline.

---

## 📊 Data Characteristics

- **Dataset type**: Kaggle-style credit card transaction fraud detection.
- **Total records**: approximately 1.11M transactions.
- **Fraud prevalence**: approximately 0.58% (highly imbalanced binary target).
- **Target column**: `is_fraud`.
- **Engineering scope**: 23 engineered features spanning temporal behavior, customer velocity, merchant risk, category trends, geographic distance, and amount anomalies.
- **Split strategy**: stratified train/validation/test split (60/10/30) to preserve minority-class ratio.

Imbalance handling in training:
- Logistic Regression uses `class_weight='balanced'`.
- LightGBM uses positive-class reweighting (`scale_pos_weight`).

---

### Verified Current State (2026-04-14)
- Full DVC graph executes through: `prepare -> train -> evaluate -> promote -> drift -> tune`.
- CLI command surface includes: `prepare`, `paths`, `train`, `evaluate`, `promote`, `drift`, `tune`.
- Day 1-3 + monitoring artifacts are present:
   - `reports/metrics/day1_data_summary.json`
   - `reports/metrics/train_metrics.json`
   - `reports/metrics/test_metrics.json`
   - `models/registry/last_promotion.json`
   - `reports/drift/drift_report.json`
- Test suite status at hardening checkpoint: `9 passed`.

---

## 🔁 Git + DVC Team Collaboration (No Need to Re-Run From Scratch)

Use this path when you want to continue from the latest shared pipeline state instead of recomputing all stages locally.

### Step 1: Clone and move to the correct branch
```bash
git clone https://github.com/HelcurtLordno1/MLOPs_credit_fraud_detection.git
cd MLOPs_credit_fraud_detection
git checkout mlops_final
```

### Step 2: Create and activate virtual environment
```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

### Step 3: Install project dependencies
Recommended for this repository (uses `pyproject.toml`):
```powershell
pip install -e .[dev]
```

If your team provides a `requirements.txt`, use:
```powershell
pip install -r requirements.txt
```

### Step 4: Pull DVC-tracked data/artifacts from remote
```powershell
python -m dvc pull
```

### Step 5: Confirm paths and environment
```powershell
$env:PYTHONPATH='src'
python -m fraud_detection.cli paths
python -m dvc status
```

### Step 6: Continue from current graph state
```powershell
python -m dvc repro drift
python -m dvc repro tune
```

### What this gives your team
- Git tracks pipeline code and metadata (`dvc.yaml`, `dvc.lock`, `configs/`, `src/`).
- DVC tracks large artifacts (processed parquet files, trained models, reports) by hash.
- `dvc pull` lets any teammate reproduce the same working state quickly.
- `dvc repro` reruns only impacted stages when inputs/configuration change.

---

## 📋 Quick Start (Beginner-Friendly, Full Setup)

This section is for first-time local setup and avoids confusion with the collaboration flow above.

### A. Prerequisites checklist
- Python 3.9+
- Git
- Enough disk space for dataset + model artifacts
- At least ~4 GB RAM for comfortable preprocessing

### B. Open project folder
```bash
cd MLOPs_credit_fraud_detection
```

### C. Ensure data files exist
Place these in project root (or ensure equivalent DVC data is available):
```text
fraudTrain.csv
fraudTest.csv
```

### D. Create environment and install dependencies
```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e .[dev]
```

### E. Set Python path and verify CLI
```powershell
$env:PYTHONPATH='src'
python -m fraud_detection.cli paths
```

### F. Choose one execution mode

1. **Cached collaboration mode (fastest)**
```powershell
python -m dvc pull
python -m dvc repro drift
python -m dvc repro tune
```

2. **Fresh recompute mode (from pipeline inputs)**
```powershell
python -m dvc repro prepare
python -m dvc repro train
python -m dvc repro evaluate
python -m dvc repro promote
python -m dvc repro drift
python -m dvc repro tune
```

### G. Validate outputs after run
Check these files:
- `data/processed/train.parquet`
- `data/processed/val.parquet`
- `data/processed/test.parquet`
- `reports/metrics/day1_data_summary.json`
- `reports/metrics/train_metrics.json`
- `reports/metrics/test_metrics.json`
- `models/registry/last_promotion.json`
- `reports/drift/drift_report.json`

---

## 📅 Three-Day Workflow

### 🔷 Day 1: Data Pipeline & Feature Engineering
**Role**: Data Engineer | **Primary notebook**: `notebook/Day1_Data_Pipeline.ipynb`

**Objectives**:
1. Load and validate raw CSV transactions.
2. Engineer core features for fraud signal strength:
   - Temporal: hour/day/weekend/month features.
   - Customer behavior: count, average amount, variance, historical fraud tendency.
   - Merchant behavior: historical fraud tendency and transaction profile.
   - Category aggregates.
   - Distance-based risk features.
   - Amount anomaly features.
3. Build stratified train/validation/test splits (60/10/30).
4. Persist processed parquet datasets and summary metrics.

**Expected Day 1 artifacts**:
- `data/processed/train.parquet`
- `data/processed/val.parquet`
- `data/processed/test.parquet`
- `reports/metrics/day1_data_summary.json`

---

### 🔶 Day 2: Model Training & Bias-Variance Analysis
**Role**: ML Engineer | **Primary notebook**: `notebook/Day2_Model_Training_BiasVariance.ipynb`

**Objectives**:
1. Train baseline and boosted models on Day 1 outputs.
2. Handle class imbalance correctly for both model families.
3. Evaluate with AUPRC-first strategy and supporting metrics (AUC, recall, precision, F1).
4. Analyze bias-variance behavior across train/validation/test splits.
5. Log experiments and parameters to MLflow.

**Expected Day 2 artifacts**:
- `models/trained/best_model.joblib`
- `reports/metrics/train_metrics.json`
- `mlruns/` experiment tracking records

---

### 🔹 Day 3: Evaluation, Promotion, and Monitoring
**Role**: MLOps Engineer | **Primary notebook**: `notebook/Day3_Evaluation_Promotion.ipynb`

**Objectives**:
1. Evaluate challenger model on all splits.
2. Apply champion/challenger promotion policy.
3. Generate promotion decision output.
4. Run drift checks to support monitoring readiness.

**Promotion criteria**:
```text
IF (challenger_auprc >= champion_auprc) AND
   (challenger_recall >= champion_recall) AND
   (challenger_precision >= champion_precision * 0.95)
THEN promote challenger
ELSE keep champion and request iteration
```

**Expected Day 3 artifacts**:
- `reports/metrics/test_metrics.json`
- `models/registry/last_promotion.json`
- `reports/drift/drift_report.json`

---

## 📚 Key Concepts & Terminology

- **Feature Engineering**: Transforming raw transactions into stronger model inputs.
- **Train/Val/Test Split**: 60/10/30 with stratification for severe class imbalance.
- **AUPRC**: Primary metric for fraud tasks where positive class is rare.
- **Bias vs Variance**: Diagnosing underfitting vs overfitting using split-level gaps.
- **Champion/Challenger**: Production model compared against newly trained candidate.
- **DVC Reproducibility**: Stage-based reruns with dependency tracking and artifact versioning.

---

## 📝 Configuration Files

### `configs/data.yaml`
- Input/output data paths.
- Feature-engineering controls.
- Data validation expectations.

### `configs/train.yaml`
- Model hyperparameters.
- Training/evaluation thresholds.
- Experiment tracking options.

### `configs/monitoring.yaml`
- Drift thresholds.
- Monitoring report controls.

### `configs/serve.yaml`
- Serving/API runtime configuration.

---

## ⚠️ Important Notes

1. Always activate `.venv` before running `python`, `pytest`, or `dvc` commands.
2. Keep `PYTHONPATH='src'` when using local CLI module execution.
3. If using shared team artifacts, run `python -m dvc pull` first.
4. If dependencies conflict, recreate environment and reinstall from clean state.
5. When changing code/config, run targeted DVC stages instead of full reruns.

---

## 👥 Citation

Created as part of an academic MLOps team project.

Team lead planning reference: `project_plan.md`.

---

*Last updated: 2026-04-14*
*Status: ✅ Days 1-3 backbone + DVC monitoring/tuning integration verified*
