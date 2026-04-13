# 🚀 End-to-End MLOps Fraud Detection System
## Credit Card Fraud Detection with Feature Engineering, Model Training & Promotion

This is a **complete MLOps project** implementing Days 1-3 of a collaborative data science workflow:
- **Day 1**: Data Pipeline & Feature Engineering (Member 1 - Data Engineer)
- **Day 2**: Model Training & Bias-Variance Analysis (Member 2 - ML Engineer)
- **Day 3**: Model Evaluation & Promotion Strategy (Member 3 - MLOps Engineer)

### Verified Current State (2026-04-14)
- Full DVC graph executes through: `prepare -> train -> evaluate -> promote -> drift`.
- CLI command surface includes: `prepare`, `paths`, `train`, `evaluate`, `promote`, `drift`.
- Day 1-3 + monitoring artifacts are present:
   - `reports/metrics/day1_data_summary.json`
   - `reports/metrics/train_metrics.json`
   - `reports/metrics/test_metrics.json`
   - `models/registry/last_promotion.json`
   - `reports/drift/drift_report.json`
- Test suite status at hardening checkpoint: `9 passed`.

### Day 4+ Handoff Notes
- Day 1-3 MLOps backbone is production-style and reproducible via DVC.
- Drift monitoring is now automated in pipeline execution (not manual-only).
- Day 4 work should focus on API schema and input contract migration for Kaggle fields.
- Day 5-6 can build directly on current DVC + MLflow lifecycle with container/CI-CD extensions.

---

## 📋 Quick Start

### Prerequisites
- Python 3.9+
- Git  
- Virtual environment (venv, conda, etc.)

### Setup Instructions

1. **Navigate to project**
```bash
cd MLOPs_credit_fraud_detection
```

2. **Ensure dataset files are in project root**
```
fraudTrain.csv     # Training dataset (~555K rows)
fraudTest.csv      # Test dataset (~555K rows)
```

3. **Create virtual environment and install dependencies**
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -e .[dev]
```

4. **Verify setup**
```powershell
$env:PYTHONPATH='src'
python -m fraud_detection.cli paths
```

5. **Run the full Day 1-3 MLOps pipeline (including monitoring)**
```powershell
python -m dvc repro drift
```

---

## 📅 Three-Day Workflow

### 🔷 Day 1: Data Pipeline & Feature Engineering
**Role**: Data Engineer | **File**: `notebook/Day1_Data_Pipeline.ipynb`

**Tasks**:
1. Load and validate raw CSV data (fraudTrain.csv, fraudTest.csv)
2. Engineer 23 features across 6 categories:
   - **Temporal**: hour_of_day, day_of_week, is_weekend, month, day_of_month
   - **Customer Velocity**: customer_txn_count, customer_avg_amt, customer_std_amt, customer_fraud_rate
   - **Merchant Risk**: merchant_fraud_rate, merchant_txn_count, merchant_avg_amt, merchant_std_amt
   - **Category**: category_fraud_rate, category_txn_count, category_avg_amt
   - **Distance**: distance_km, is_distant_txn
   - **Amount**: amt_zscore, amt_is_outlier, amt_ratio_to_customer_avg
3. Create train/val/test splits with stratification (60/10/30)
4. Save processed data as parquet files
5. Generate dataset summary report

**Run Day 1**:
```bash
jupyter notebook notebook/Day1_Data_Pipeline.ipynb
# Or use VS Code Jupyter extension
```

**Expected Outputs**:
- `data/processed/train.parquet`
- `data/processed/val.parquet`
- `data/processed/test.parquet`
- `reports/metrics/day1_data_summary.json`
- `reports/day1_class_distribution.png`

---

### 🔶 Day 2: Model Training & Bias-Variance Analysis
**Role**: ML Engineer | **File**: `notebook/Day2_Model_Training_BiasVariance.ipynb`

**Tasks**:
1. Load processed data splits from Day 1
2. Train two models:
   - **Logistic Regression (Baseline)** with class_weight='balanced'
   - **LightGBM** with scale_pos_weight for imbalance handling
3. Evaluate on train/val/test splits using:
   - AUPRC (best for imbalanced data)
   - AUC-ROC, Recall, Precision, F1
4. Perform comprehensive bias-variance analysis:
   - Train-val gap (underfitting indicator)
   - Train-test gap (generalization)
   - Issue diagnosis and recommendations
5. Log all experiments to MLflow
6. Save best model and metrics

**Run Day 2**:
```bash
jupyter notebook notebook/Day2_Model_Training_BiasVariance.ipynb
```

**Key Concepts**:
- **AUPRC** (Area Under Precision-Recall Curve): Optimal for fraud detection (~0.58% fraud rate)
- **Bias**: Model too simple, underfitting (high train & test error)
- **Variance**: Model too complex, overfitting (small train error, large test error)
- **Good Fit**: Small gap between train and validation, reasonable test performance

**Expected Outputs**:
- `models/trained/best_model.joblib`
- `reports/train_metrics.json`
- `reports/target_distribution.png`
- `reports/bias_variance_analysis.png`
- `reports/pr_roc_curves.png`
- `reports/confusion_matrices.png`
- `reports/feature_importance.png`
- `mlruns/` (MLflow experiment tracking)

---

### 🔹 Day 3: Model Evaluation & Promotion Strategy
**Role**: MLOps Engineer | **File**: `notebook/Day3_Evaluation_Promotion.ipynb`

**Tasks**:
1. Load trained model from Day 2
2. Comprehensive evaluation:
   - Metrics on train/val/test splits
   - Detailed classification report
   - Confusion matrix analysis
3. Bias-variance final assessment
4. Implement champion vs challenger promotion logic:
   - **Promotion Rules**:
     - Challenger AUPRC ≥ Champion AUPRC
     - Challenger Recall ≥ Champion Recall
     - Challenger Precision ≥ 95% of Champion Precision
5. Drift detection analysis:
   - Feature distribution drift (Kolmogorov-Smirnov test)
   - Target distribution drift (fraud rate changes)
6. Generate promotion decision report
7. Create monitoring strategy

**Run Day 3**:
```bash
jupyter notebook notebook/Day3_Evaluation_Promotion.ipynb
```

**Run Day 3 via reproducible pipeline commands**:
```powershell
python -m dvc repro evaluate
python -m dvc repro promote
python -m dvc repro drift
```

**Promotion Decision Criteria**:
```
IF (challenger_auprc >= champion_auprc) AND
   (challenger_recall >= champion_recall) AND
   (challenger_precision >= champion_precision * 0.95)
THEN: PROMOTE to Production
ELSE: Hold for review or request model improvements
```

**Expected Outputs**:
- `reports/day3_confusion_matrix.png`
- `reports/day3_evaluation_summary.png`
- `reports/metrics/test_metrics.json`
- `models/registry/last_promotion.json`
- `reports/drift/drift_report.json`
- MLflow Model Registry entries

---

## 📁 Project Structure

```
fraud-detection/
│
├── notebook/
│   ├── Day1_Data_Pipeline.ipynb           # ✅ Data preparation & feature engineering
│   ├── Day2_Model_Training_BiasVariance.ipynb  # ✅ Model training & analysis
│   └── Day3_Evaluation_Promotion.ipynb    # ✅ Evaluation & promotion strategy
│
├── src/fraud_detection/
│   ├── data/
│   │   ├── features.py                    # Feature engineering functions
│   │   ├── pipeline.py                    # Data pipeline orchestration
│   │   └── schema.py                      # Data validation schemas
│   ├── modeling/
│   │   ├── train.py                       # Model training functions
│   │   └── evaluate.py                    # Model evaluation metrics
│   ├── monitoring/
│   │   ├── drift.py                       # Drift detection
│   │   └── promotion.py                   # Model promotion logic
│   ├── utils/
│   │   ├── paths.py                       # Path utilities
│   │   └── mlflow_utils.py                # MLflow helpers
│   ├── cli.py                             # Command-line interface
│   └── config.py                          # Configuration management
│
├── configs/
│   ├── data.yaml                          # Data configuration
│   ├── train.yaml                         # Training configuration
│   ├── monitoring.yaml                    # Monitoring configuration
│   └── serve.yaml                         # API serving configuration
│
├── data/
│   ├── processed/
│   │   ├── train.parquet                  # Processed training data
│   │   ├── val.parquet                    # Processed validation data
│   │   └── test.parquet                   # Processed test data
│   ├── interim/                           # Temporary data
│   └── monitoring/                        # Monitoring data
│
├── models/
│   ├── trained/
│   │   └── best_model.joblib              # Best trained model
│   └── registry/                          # Model registry
│
├── reports/
│   ├── metrics/
│   │   ├── day1_data_summary.json         # Data statistics
│   │   ├── train_metrics.json             # Training metrics
│   │   └── test_metrics.json              # Evaluation report
│   ├── drift/
│   │   └── drift_report.json              # Automated drift report
│   └── figures/
│       ├── day1_class_distribution.png
│       ├── bias_variance_analysis.png
│       ├── pr_roc_curves.png
│       └── ...
│
├── tests/                                 # Unit tests
├── Dockerfile                             # Container image
├── docker-compose.yaml                    # Multi-container orchestration
├── dvc.yaml                               # DVC pipeline
├── pyproject.toml                         # Project dependencies
└── README.md                              # This file
```

## 🔧 Feature Engineering Details

The pipeline creates **23 engineered features** from raw transaction data:

### Temporal Features (5)
- `hour_of_day`: Transaction hour (0-23)
- `day_of_week`: Day of week (0-6, Monday-Sunday)
- `is_weekend`: Binary weekend flag
- `month`: Month of transaction (1-12)
- `day_of_month`: Day of month (1-31)

### Customer Velocity Features (4)
- `customer_txn_count`: Total transactions per customer
- `customer_avg_amt`: Average transaction amount
- `customer_std_amt`: Standard deviation of amounts
- `customer_fraud_rate`: Historical fraud rate for customer

### Merchant Risk Features (4)
- `merchant_fraud_rate`: Historical fraud rate at merchant
- `merchant_txn_count`: Total transactions at merchant
- `merchant_avg_amt`: Average amount at merchant
- `merchant_std_amt`: Standard deviation at merchant

### Category Features (3)
- `category_fraud_rate`: Fraud rate for transaction category
- `category_txn_count`: Total transactions in category
- `category_avg_amt`: Average amount in category

### Distance Features (2)
- `distance_km`: Distance between cardholder and merchant (Haversine formula)
- `is_distant_txn`: Flag for transactions >100km away

### Amount Anomaly Features (3)
- `amt_zscore`: Z-score relative to customer average
- `amt_is_outlier`: Flag for unusual amounts (|z-score| > 3)
- `amt_ratio_to_customer_avg`: Ratio to customer historical average

---

## 📊 Data Characteristics

- **Total Records**: ~1.11M transactions
- **Fraud Cases**: ~13K (~0.58% fraud rate - highly imbalanced)
- **Features**: 23 engineered + raw numeric features
- **Time Period**: Spanning several months
- **Geography**: US-based credit card transactions

### Class Imbalance Handling
Both models address the severe imbalance:
- **Logistic Regression**: `class_weight='balanced'` (automatically weights fraud 172x higher)
- **LightGBM**: `scale_pos_weight=172` (explicit positive class weighting)

---

## 📈 Performance Metrics

### Primary Metric: AUPRC
Why AUPRC for fraud detection?
- ROC curve is misleading with imbalanced data (high TNR makes curve look good)
- PR curve focuses on positive class (fraud), which is what we care about
- AUPRC = area under Precision-Recall curve
- Value range: 0-1 (higher is better)
- Threshold-independent evaluation

### Supporting Metrics
- **Recall (True Positive Rate)**: "Of actual frauds, how many did we catch?" → Minimize missed fraud
- **Precision**: "Of predicted frauds, how many are correct?" → Minimize false alarms
- **F1-Score**: Harmonic mean of Recall and Precision → Balanced evaluation

---

## 🔄 MLflow Integration

The project uses **MLflow** for experiment tracking:

View experiments:
```bash
mlflow ui
# Open browser to http://localhost:5000
```

Features tracked:
- Model parameters (learning_rate, n_estimators, etc.)
- Metrics (auprc, recall, precision, etc.) for train/val/test
- Model artifacts (pickled models, plots)
- Feature importance
- Bias-variance gaps

---

## 🚨 Bias-Variance Analysis Interpretation

| Diagnosis | Symptoms | Solution |
|-----------|----------|----------|
| **Good Fit** | Small train-val gap (~0.01-0.05) | Ready for production |
| **Underfitting (High Bias)** | Low train performance, model too simple | Increase model complexity, add features |
| **Overfitting (High Variance)** | Large train-val gap (>0.15), memorizing data | Add regularization, more data, feature reduction |
| **Moderate Variance** | Gap between 0.05-0.15 | Monitor validation performance, consider ensemble |

---

## 🎯 Next Steps (Days 4-6)

After completing Days 1-3, the next phases would be:

- **Day 4** (API & Frontend): Expose predictions via FastAPI, Streamlit UI
- **Day 5** (Infrastructure): Containerize with Docker, deploy to Kubernetes
- **Day 6** (Validation & Docs): End-to-end testing, team documentation

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_data_pipeline.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

---

## 📚 Key Concepts & Terminology

- **Feature Engineering**: Creating new features from raw data to improve model performance
- **Train/Val/Test Split**: 60% training, 10% validation (tuning), 30% test (evaluation) with stratification
- **Stratification**: Maintaining class distribution across splits (important for imbalanced data)
- **Bias**: Inability to capture true relationship (high training error)
- **Variance**: Sensitivity to training data variations (overfitting)
- **AUPRC**: Optimal metric for imbalanced classification (fraud detection)
- **Champion/Challenger**: Production model vs. new candidate model with promotion rules

---

## 🤝 Team Roles Summary

| Day | Role | Member | Focus |
|-----|------|--------|-------|
| 1 | Data Engineer | Member 1 | Feature engineering, data pipeline, splits |
| 2 | ML Engineer | Member 2 | Model training, experiments, bias-variance |
| 3 | MLOps Engineer | Member 3 | Evaluation, promotion logic, monitoring |

---

## 📝 Configuration Files

### `configs/data.yaml`
- Input/output paths
- Feature engineering settings
- Data validation rules

### `configs/train.yaml`
- Model hyperparameters
- Evaluation thresholds
- MLflow settings

### `configs/monitoring.yaml`
- Drift detection thresholds
- Alert configurations

---

## ⚠️ Important Notes

1. **Dataset Files**: Ensure `fraudTrain.csv` and `fraudTest.csv` are in the project root
2. **Python Path**: Set `PYTHONPATH='src'` when running CLI commands
3. **Virtual Environment**: Always activate before running commands
4. **Memory Requirements**: Feature engineering on full dataset requires ~4GB RAM
5. **Runtime**: Full pipeline may take 10-15 minutes depending on hardware

---

## 📞 Troubleshooting

### "Module not found" errors
```bash
$env:PYTHONPATH='src'
# Re-run command
```

### "Cannot load parquet files"
```bash
# Install pyarrow backend
pip install pyarrow
```

### "Data file not found"
- Verify `fraudTrain.csv` and `fraudTest.csv` are in project root
- Check file names (case-sensitive on some systems)

### "Jupyter kernel failure"
- Reinstall or update jupyter: `pip install --upgrade jupyter`
- Python interpreter mismatch: ensure using venv Python

---

## 📄 License

MIT License

---

## 👥 Citation

Created as part of academic MLOps project (Team Lead: Project Planner)

---

*Last updated: 2026-04-13*
*Status: ✅ Days 1-3 Complete | Ready for Days 4-6 Implementation*
