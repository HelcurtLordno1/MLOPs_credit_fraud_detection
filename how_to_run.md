# How To Run (Current Workflow State)

This guide covers exactly how to run the project at the current stage (through Week 1 Day 6-7), including DVC, model training, MLflow, and tests.

## 1) Prerequisites

- Python 3.10+ (project currently runs with Python 3.11 in this workspace)
- Git
- DVC (installed from `requirements.txt`)
- Dataset file: `data/raw/creditcard.csv`

## 2) Clone and Enter Project

```bash
git clone https://github.com/HelcurtLordno1/MLOPs_credit_fraud_detection.git
cd MLOPs_credit_fraud_detection
```

If you need a separate working branch:

```bash
git checkout -b your-branch-name
```

## 3) Create Virtual Environment and Install Dependencies

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 4) DVC Setup and Data Pipeline

### 4.1 What DVC does in this project

- Git tracks metadata files (`dvc.yaml`, `dvc.lock`, `*.dvc`), not large raw data.
- DVC tracks versions of `data/raw/creditcard.csv` and generated artifacts.
- `dvc repro` runs pipeline stages from `dvc.yaml` and updates `dvc.lock`.

Current DVC stage:
- `prepare` stage in `dvc.yaml`
- Runs: `python src/ml/data.py`
- Produces:
  - `data/processed/train.parquet`
  - `data/processed/test.parquet`
  - `data/processed/reference.parquet`

### 4.2 Run DVC pipeline

```bash
python -m dvc repro
```

Optional (if your team uses configured remote storage):

```bash
python -m dvc pull
python -m dvc push
```

## 5) Run Baseline Model Training (Week 1 Day 4-5)

Training script:
- `src/ml/train.py`

Config file:
- `configs/training.yaml`

Run training:

```bash
python -m src.ml.train
```

Expected outputs:
- `model.joblib`
- New run under MLflow experiment `credit-fraud`
- Console metrics (AUC-ROC, AUPRC, Recall, Precision)

## 6) Launch MLflow UI

```bash
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Open in browser:
- http://127.0.0.1:5000

Important UI note:
- In the top-left experiment selector, choose `credit-fraud` (not only `Default`).
- Your training and test runs are logged under `credit-fraud`.

Note for Windows:
- MLflow may show a warning about job execution backend support on Windows. This does not block experiment tracking or UI usage.

## 7) Run Tests (Week 1 Day 6-7)

```bash
python -m pytest -v
```

Current expected result:
- 1 passing test in `tests/test_model.py`

Optional but recommended (log test results into MLflow UI):

```bash
python scripts/log_tests_mlflow.py
```

This creates a run named `week1-pytest-validation` under `credit-fraud` and logs:
- `tests_exit_code`
- `tests_duration_sec`
- `tests_passed`
- full pytest stdout/stderr as artifacts

## 8) Reproducibility Document

Reference file:
- `docs/reproducibility_proof.md`

It records:
- Python version guidance
- Commands to reproduce data and training artifacts
- Expected artifact list

## 9) Recommended Team Run Order

1. `pip install -r requirements.txt`
2. `python -m dvc repro`
3. `python -m src.ml.train`
4. `python scripts/log_tests_mlflow.py`
5. `python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000`

## 10) Troubleshooting

- If `dvc repro` fails due to missing data:
  - Ensure `data/raw/creditcard.csv` exists.
- If MLflow does not open:
  - Check terminal for port conflicts and retry on another port.
- If model training fails due to missing packages:
  - Re-run `pip install -r requirements.txt` in the active `.venv`.
- If tests are not found:
  - Ensure you run commands from project root.
