# Improvements Needed for Project_mlops_credit_fraud_detection

This document is a direct gap analysis between:
- Source of good practices: `credit-card-fraud-detection`
- Current target project: `Project_mlops_credit_fraud_detection`

Goal: improve model quality (especially precision/recall/AUPRC) and complete the missing MLOps implementation.

## 1) Critical Missing Components (Must Fix First)

### Missing production files (currently only `.gitkeep`)
- `src/app/` has no API implementation.
- `src/monitoring/` has no drift/production monitoring job.
- `src/database/` has no prediction logging model.
- `infra/prometheus/` has no scrape config.
- `.github/workflows/` has no CI pipeline.
- `docker/` has no app/MLflow Dockerfiles.

Impact:
- You cannot serve predictions reliably.
- You cannot observe drift or latency in production.
- You cannot pass a complete MLOps rubric.

Required additions:
- Add API with `/health`, `/predict`, `/metrics`, `/model_info`, `/reload`.
- Add Prometheus metric instrumentation (request count, prediction count, latency histogram).
- Add monitoring scripts for Evidently drift reports from reference/current data and production logs.
- Add CI workflow (pytest, style checks, docker build).
- Add Docker Compose stack for app + MLflow + Prometheus + Grafana.

## 2) Data Pipeline Gaps and Improvements

Current status in your project:
- `src/ml/data.py` only does train/test split + reference sample.
- `dvc.yaml` only has `prepare` stage.

Missing techniques from the stronger repo and from robust fraud workflows:
- No `current.parquet` output for drift baseline comparison.
- No data quality assertions (nulls, duplicate rows, expected schema, class labels).
- No explicit preprocessing policy for `Time` and `Amount`.
- No stage separation (`prepare -> train -> evaluate`) in DVC.

Needed improvements:
1. Upgrade `src/ml/data.py` into a function-based script with:
	- deterministic seed,
	- schema validation,
	- null/duplicate checks,
	- train/test/reference/current outputs,
	- saved summary stats artifact.
2. Update `dvc.yaml` to include stages:
	- `prepare`
	- `train`
	- `evaluate`
3. Add `data_quality_report.json` artifact (row counts, fraud ratio, missing values).

## 3) Modeling Gaps and Improvements for Better Precision/Accuracy

Current status in your project:
- Only one baseline Logistic Regression pipeline.
- Default threshold of model `.predict()` (implicit 0.5).
- Metrics logged, but no threshold optimization pipeline.
- No calibrated probabilities, no cross-validation selection, no challenger model.

Priority modeling improvements:

1. Threshold optimization (highest impact for fraud precision/recall tradeoff)
- Compute precision-recall curve on validation set.
- Select threshold by objective, for example:
  - max F1 for class 1, or
  - max precision under constraint recall >= 0.92, or
  - minimum expected business cost.
- Log selected threshold to MLflow and store in `configs/inference.yaml`.

2. Add validation split and cross-validation
- Use stratified train/val split or StratifiedKFold.
- Tune threshold on validation only, then report final test metrics.
- Avoid using test set for threshold search.

3. Add challenger models
- Keep Logistic Regression as baseline.
- Add at least one tree-based challenger (XGBoost/LightGBM/RandomForest).
- Tune hyperparameters (manual grid or Optuna).

4. Improve class imbalance handling
- Keep `class_weight="balanced"` baseline.
- Compare with:
  - tuned class weights,
  - SMOTE/undersampling pipelines,
  - `scale_pos_weight` (for XGBoost).
- Decide based on AUPRC and precision/recall at operating threshold.

5. Probability calibration
- Add Platt or isotonic calibration for models with poorly calibrated probabilities.
- Track Brier score and reliability curve.

6. Better metric suite
- Continue tracking `auc_roc`, `auprc`, `precision`, `recall`, `f1`, `accuracy`.
- Add:
  - confusion matrix at chosen threshold,
  - false positive rate,
  - precision@k (optional),
  - business-cost metric.

## 4) MLflow and Model Lifecycle Gaps

Current status:
- Training logs metrics and registers model.
- Missing full promote/validate lifecycle scripts used in stronger implementation.

Needed improvements:
1. Add `scripts/validate_model.py`:
	- load candidate model version,
	- evaluate on test data,
	- enforce minimum gates (AUC/AUPRC/Recall/Precision),
	- compare against current production alias if available.
2. Add `scripts/promote_model.py`:
	- alias-based promotion (`production`, `challenger`),
	- optional API reload call.
3. Add `scripts/promote_and_restart.py`:
	- set alias + restart app container automatically.
4. Add a simple model registry policy:
	- only promote if validation passes,
	- keep rollback alias to previous version.

## 5) Serving/API Gaps

Current status:
- No API implementation in `src/app`.

Needed improvements:
1. Add request schema validation (fixed-length features and type checks).
2. Load model from MLflow alias first; fallback to local artifact.
3. Expose operational endpoints:
	- `/health`
	- `/predict`
	- `/metrics`
	- `/model_info`
	- `/reload`
4. Persist prediction logs (feature subset, score, threshold, output, timestamp).

## 6) Monitoring and Drift Gaps

Current status:
- No monitoring code, no Prometheus config, no Grafana assets.

Needed improvements:
1. Add `src/monitoring/drift_job.py`:
	- compare `reference.parquet` vs `current.parquet` (or production batch),
	- generate HTML report into `reports/`.
2. Add `src/monitoring/production_monitor.py`:
	- pull recent inference records,
	- compute drift and quality metrics on rolling windows.
3. Add `infra/prometheus/prometheus.yml` with scrape target `app:8000/metrics`.
4. Export and commit one Grafana dashboard JSON.
5. Add alert policy:
	- drift detected,
	- latency > target,
	- low precision/recall trend.

## 7) Testing Gaps

Current status:
- `tests/test_model.py` only checks that a config file exists.

Needed tests:
1. Data tests:
	- class column exists,
	- no unexpected nulls,
	- fraud class present in both train/test splits.
2. Model tests:
	- model file exists after training,
	- prediction shape and range,
	- minimum quality gate assertions.
3. API tests:
	- `/health`, `/predict`, `/metrics`, `/model_info`, invalid payload cases.
4. Promotion validation tests:
	- candidate model gate checks,
	- alias switch behavior.

## 8) CI/CD and Reproducibility Gaps

Current status:
- No CI workflow file.

Needed improvements:
1. Add `.github/workflows/ci.yaml` pipeline with:
	- dependency install,
	- `pytest`,
	- style/lint checks,
	- docker build checks.
2. Add a simple CD step (optional in week 3):
	- build/push app image,
	- deploy compose/k8s target.
3. Ensure all teammates can reproduce:
	- `dvc pull && dvc repro`,
	- `python -m src.ml.train`,
	- `docker compose up`.

## 9) Configuration and Project Hygiene Gaps

Current status:
- Only `configs/training.yaml` exists.

Needed config files:
- `configs/base.yaml` (common paths/seeds)
- `configs/inference.yaml` (threshold + model alias)
- `configs/monitoring.yaml` (drift thresholds/window sizes)

Additional hygiene tasks:
- Pin and clean `requirements.txt` (currently appears to have encoding/noise issues).
- Add `pytest.ini` for stable test discovery.
- Add `README` sections for full runbook and troubleshooting.

## 10) Prioritized Action Plan (Execution Order)

Week 1 (must complete):
1. Data pipeline hardening (`data.py`, `dvc.yaml`, quality report).
2. Modeling improvements (threshold tuning + validation split + richer metrics).
3. Real test suite (data/model/API basics).

Week 2 (must complete):
1. API implementation + Docker.
2. CI workflow and passing checks.
3. MLflow validation/promote scripts.

Week 3 (must complete):
1. Monitoring stack (Evidently + Prometheus + Grafana).
2. Controlled drift simulation + retrain + promote demo.
3. Final evidence package (reports/screenshots/video).

## 11) Minimum Acceptance Gates for Your Project

Use these as hard go/no-go gates:
- AUPRC >= 0.85
- Recall >= 0.92
- Precision >= 0.80
- FPR < 0.05 at selected threshold
- API p95 latency < 50 ms (local baseline)
- Drift report generated and reviewable
- CI green on push/PR

If these gates are not met, do not promote model alias to `production`.

