# Reproducibility Proof

## Python Version
- Python 3.10+ recommended

## Commands to Reproduce
1. Install dependencies:
   - `pip install -r requirements.txt`
2. Prepare data artifacts:
   - `dvc repro`
3. Train baseline model:
   - `python -m src.ml.train`
4. Run tests:
   - `pytest -v`

## Expected Artifacts
- `data/processed/train.parquet`
- `data/processed/test.parquet`
- `data/processed/reference.parquet`
- `model.joblib`
- MLflow run under experiment `credit-fraud`
