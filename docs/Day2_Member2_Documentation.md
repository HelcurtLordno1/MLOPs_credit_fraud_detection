# Day 2 — Member 2: The Model Trainer

**Member:** Duc  
**Role:** ML Engineer (Training & Experimentation)  
**Branch:** `day2-model-training`

---

## 1. Task Overview

After Member 1 (Manh) set up the project structure and uploaded the raw data, the team agreed that feature engineering should be handled together with model development so the modeling owner can directly validate which features are useful.

Therefore, Member 2 (Duc) handled both modeling and data preparation responsibilities:

1. Perform feature engineering from the raw files and produce clean processed data (`data/processed/*.parquet`).
2. Train at least two models (Logistic Regression baseline + LightGBM).
3. Run bias-variance analysis and log all results to MLflow.

---

## 2. Feature Engineering (Added Scope Based on Member 1 Feedback)

Member 2 created `src/ml/data.py` to process `data/fraudTrain.csv` and `data/fraudTest.csv` into standardized outputs:

| File | Purpose |
|------|---------|
| `data/processed/train.parquet` | Training split |
| `data/processed/val.parquet` | Validation split (during training) |
| `data/processed/test.parquet` | Test split (final evaluation) |

### Features Created by Member 2 (`src/ml/data.py`)

The following meaningful features are extracted from raw transaction data:

#### 2.1 Temporal Features
```python
# Parse transaction time
df['trans_date_trans_time'] -> pd.to_datetime()  # fallback: unix_time

# Derived features:
hour_of_day   # Hour in day (0-23)
day_of_week   # Day index (0=Mon, 6=Sun)
is_weekend    # 1 for Saturday/Sunday, 0 otherwise
```
**Reasoning:** Fraud often appears at unusual times (late night, weekends).

#### 2.2 Customer Velocity Features
```python
# Group by cc_num (credit card number)
customer_avg_amt
customer_txn_count
customer_txn_count_last_24h
```
**Reasoning:** Fraud activity can involve abnormal spending and suspicious transaction frequency.

#### 2.3 Merchant Risk Feature
```python
# Group by merchant
merchant_fraud_rate  # Computed from is_fraud
```
**Reasoning:** Some merchants historically show higher fraud concentration.

Implementation note:
- `merchant_fraud_rate` is fit on train data only, then mapped to val/test.
- Train split uses leave-one-out style computation to reduce target leakage.

#### 2.4 Frequency Encoding
```python
# Convert high-cardinality categorical columns to frequency values
category_freq
# (similarly for other string columns)
```
**Reasoning:** Logistic Regression handles normalized frequency-encoded features better than large one-hot vectors for high-cardinality fields.

---

## 3. Files Produced by Member 2

```text
MLOPs_credit_fraud_detection/
|
|-- src/ml/
|   |-- data.py                     <- Feature engineering script (DVC prepare)
|   `-- train.py                    <- Main training script (DVC train)
|
|-- notebooks/
|   |-- build_notebook.py           <- Notebook generation script
|   `-- Day2_Model_Training_BiasVariance.ipynb
|
|-- configs/
|   `-- training.yaml               <- Hyperparameters + quality thresholds
|
|-- reports/
|   |-- train_metrics.json          <- Full metric report
|   |-- bias_variance_analysis.png
|   |-- pr_roc_curves.png
|   |-- confusion_matrices.png
|   |-- feature_importance.png
|   |-- all_metrics_comparison.png
|   `-- target_distribution.png
|
`-- model.joblib                    <- Saved best model (LightGBM)
```

---

## 4. `src/ml/train.py` Details

This is the main Python script called by DVC when running `dvc repro train`.

### 4.1 End-to-End Flow

```text
Load config (training.yaml)
        ->
Load data splits (train/val/test.parquet)
        ->
Split X (features) / y (is_fraud)
        ->
Compute scale_pos_weight = n_neg / n_pos
        ->
Build 2 models: Logistic Regression + LightGBM
        ->
Train both models on train split
        ->
Evaluate on train / val / test (auprc, roc_auc, recall, precision, fpr)
        ->
Compute Bias-Variance Gap (|train_auprc - val_auprc|)
        ->
Log to MLflow (parent run + nested child run per model)
        ->
Quality Gate check against thresholds in training.yaml
        ->
Save best model -> model.joblib
Save metrics -> reports/train_metrics.json
```

### 4.2 Function-Level Explanation

#### `load_config()`
```python
def load_config() -> dict:
    with CONFIG_PATH.open("r") as f:
        return yaml.safe_load(f)
```
Reads `configs/training.yaml`, which centralizes model parameters and quality thresholds. This allows tuning without code edits.

#### `load_split(split)`
```python
def load_split(split: str) -> pd.DataFrame:
    path = PROCESSED_DIR / (split + ".parquet")
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset split: {path}")
    return pd.read_parquet(path)
```
Loads `.parquet` files (train/val/test) and raises a clear error if prepare stage was not executed.

#### `split_xy(df)`
```python
def split_xy(df):
    y = df["is_fraud"].astype(int)
    X = df.drop(columns=["is_fraud"]).select_dtypes(include=[np.number])
    return X, y
```
- Target column: `is_fraud` (0 = legitimate, 1 = fraud)
- Features: numeric columns only (including engineered features)

Important: old datasets may use `isFraud`, but this project is fully aligned to `is_fraud`.

#### `build_logistic(cfg)`
```python
def build_logistic(cfg):
    lr_cfg = cfg["model"].get("logistic", {})
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=lr_cfg.get("C", 1.0),
            max_iter=lr_cfg.get("max_iter", 1000),
            solver="lbfgs",
            class_weight="balanced",
        )),
    ])
```

Why `class_weight='balanced'`?
- Fraud ratio is extremely low (~0.58%).
- Without reweighting, the model can optimize trivial majority predictions.

Why `StandardScaler`?
- Logistic Regression is sensitive to feature scale.

#### `build_lightgbm(cfg, scale_pos_weight)`
```python
def build_lightgbm(cfg, scale_pos_weight):
    lgb_cfg = cfg["model"].get("lightgbm", {})
    return lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        min_child_samples=50,
        scale_pos_weight=scale_pos_weight,
    )
```

`scale_pos_weight` formula:
```python
scale_pos_weight = n_negatives / n_positives
```

Why LightGBM?
1. Strong support for imbalanced data through `scale_pos_weight`.
2. Efficient training and strong non-linear modeling capacity.
3. Works well with engineered numeric and frequency-encoded features.

#### `score(model, X, y)`
```python
def score(model, X, y) -> dict:
    proba = model.predict_proba(X)[:, 1]
    y_pred = (proba >= 0.5).astype(int)
    return {
        "roc_auc": roc_auc_score(y, proba),
        "auprc": average_precision_score(y, proba),
        "recall": recall_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "fpr": fp / (fp + tn),
    }
```

Metric meaning:

| Metric | Formula | Meaning |
|--------|---------|---------|
| AUPRC | Area under PR curve | Primary metric for imbalance |
| ROC-AUC | Area under ROC curve | Global ranking/separability quality |
| Recall | TP / (TP + FN) | Fraction of fraud captured |
| Precision | TP / (TP + FP) | Fraud alert accuracy |
| FPR | FP / (FP + TN) | Legitimate transactions flagged as fraud |

Why AUPRC over accuracy?
- With heavy imbalance, accuracy can be misleading.
- AUPRC focuses directly on minority-class detection quality.

#### `train()`
The main orchestrator uses MLflow nested runs:

```text
MLflow Experiment: "credit-fraud"
`-- Parent Run: "credit-fraud-day2"
    |-- Params: target_column, candidates, scale_pos_weight, ...
    |-- Child Run: "LogisticRegression"
    |   |-- Metrics: train_auprc, val_auprc, test_auprc, ...
    |   `-- Metrics: bias_variance_gap_auprc, train_test_gap_auprc
    `-- Child Run: "LightGBM"
        |-- Metrics: train_auprc, val_auprc, test_auprc, ...
        `-- Metrics: bias_variance_gap_auprc, train_test_gap_auprc
```

Nested runs make model-to-model comparison easy in MLflow UI and provide clean handoff metadata for Day 3 promotion logic.

---

## 5. Bias-Variance Analysis

This is a required Day 2 deliverable. Each model is evaluated on three splits to detect underfitting/overfitting:

```text
Bias-Variance Gap = |Train AUPRC - Val AUPRC|
Train-Test Gap    = |Train AUPRC - Test AUPRC|
```

### Actual Results (from `reports/train_metrics.json`)

#### Logistic Regression (Baseline)
| Split | AUPRC | ROC-AUC | Recall | Precision |
|-------|-------|---------|--------|-----------|
| Train | 0.2275 | 0.9002 | 0.779 | 0.0703 |
| Val   | 0.2357 | 0.8947 | 0.737 | 0.1259 |
| Test  | 0.1432 | 0.8770 | 0.469 | 0.2319 |

- Bias-Variance Gap (AUPRC): `0.0082` (very low, stable)
- Interpretation: underfitting. Recall is acceptable, but precision is very low.

#### LightGBM (Main Model)
| Split | AUPRC | ROC-AUC | Recall | Precision |
|-------|-------|---------|--------|-----------|
| Train | 0.9986 | 0.99999 | 1.000 | 0.749 |
| Val   | 0.9359 | 0.9978 | 0.923 | 0.732 |
| Test  | 0.7929 | 0.9904 | 0.759 | 0.625 |

- Bias-Variance Gap (AUPRC): `0.0627` (moderate)
- Train-Test Gap (AUPRC): `0.2057` (improved vs previous run, still noticeable)
- Interpretation: performance is stronger and generalization improved after leakage-safe feature handling, but further tuning is still recommended.

### Bias-Variance Trade-off Illustration

```text
                High Bias (Underfitting)    Low Bias
Low Variance    Logistic Regression         [Ideal Model]
                train~val~test~0.22

High Variance   [Worst case]                LightGBM
                                            train=0.999
                                            val=0.936
                                            test=0.793
```

---

## 6. `configs/training.yaml` Configuration

```yaml
model:
  run_name: credit-fraud-day2
  candidates:
    - logistic
    - lightgbm

  logistic:
    C: 1.0
    max_iter: 1000
    solver: lbfgs
    class_weight: balanced

  lightgbm:
    n_estimators: 400
    learning_rate: 0.05
    num_leaves: 63
    subsample: 0.9
    colsample_bytree: 0.9
    reg_lambda: 1.0
    min_child_samples: 50

thresholds:
  min_auc: 0.90
  min_auprc: 0.85
  min_recall: 0.92
  min_precision: 0.80
```

---

## 7. DVC Pipeline Integration

`dvc.yaml` defines `train` as dependent on outputs from `prepare`:

```yaml
stages:
  train:
    cmd: ./.venv/bin/python -m src.ml.train
    deps:
      - src/ml/train.py
      - configs/training.yaml
      - data/processed/train.parquet
      - data/processed/val.parquet
      - data/processed/test.parquet
    outs:
      - model.joblib
      - reports/train_metrics.json
```

How to run:
```bash
dvc pull
dvc repro train
# or full pipeline
dvc repro
```

---

## 8. Interactive Notebook

`notebooks/Day2_Model_Training_BiasVariance.ipynb` is the interactive version of `train.py`, generated from `notebooks/build_notebook.py`.

Notebook contents:
1. Environment setup and imports
2. Config and data loading
3. Class imbalance analysis
4. Model training (LogReg + LightGBM)
5. Bias-variance comparison
6. PR and ROC curves
7. Confusion matrices
8. Classification reports
9. Feature importance (LightGBM)
10. MLflow logging
11. Save model and metrics
12. Final summary

Run notebook:
```bash
cd MLOPs_credit_fraud_detection
jupyter lab notebooks/Day2_Model_Training_BiasVariance.ipynb
```

Or regenerate from script:
```bash
python notebooks/build_notebook.py
```

---

## 9. Deliverables for Member 3

After Day 2, Member 3 (Day 3 - Quality Gatekeeper) needs:

1. MLflow parent `run_id` (from terminal output or MLflow UI)
2. `model.joblib` (trained LightGBM model)
3. `reports/train_metrics.json` (full metrics for champion/challenger comparison)

```bash
mlflow ui --host 127.0.0.1 --port 5000
```

---

## 10. Completion Summary

| Item | Status |
|------|--------|
| Feature engineering (`src/ml/data.py`) | Completed |
| Model 1: Logistic Regression baseline | Completed |
| Model 2: LightGBM main model | Completed |
| Class imbalance handling | `class_weight='balanced'` + `scale_pos_weight` |
| Bias-variance analysis | Logged train/val/test gaps in MLflow |
| MLflow experiment tracking | Nested runs, params, metrics, artifacts |
| Quality gate checks | Thresholds from `configs/training.yaml` |
| DVC integration (`dvc repro train`) | Compatible with `src/ml/train.py` |
| Interactive notebook | `Day2_Model_Training_BiasVariance.ipynb` |
| Config file | `configs/training.yaml` |

---

*This document was prepared by Member 2 (Duc) - Day 2: The Model Trainer.*
