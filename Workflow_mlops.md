3-Week MLOps Workflow (Expert-Level, Practical, and Easy to Execute)

Project: Topic #3 - Credit Card Fraud Detection System
Team size: 6 members
Approach: 100% built from scratch in your own repository

Important reference rule:
- You may read https://github.com/gana36/credit-card-fraud-detection in browser view only.
- Do not copy-paste code from that repository.
- Use it only to understand architecture, sequencing, and professional project flow.

Core stack to cover full rubric:
- Data and versioning: DVC
- Experiment tracking and registry: MLflow
- Serving: FastAPI
- Reproducible runtime: Docker and Docker Compose
- Monitoring and drift: Evidently + Prometheus + Grafana
- Automation: GitHub Actions CI/CD
- Model lifecycle: retrain + promote workflow

Dataset (Real-world, required)
- Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- File: `creditcard.csv`
- Shape: 284,807 rows
- Fraud class count: 492

------------------------------------------------------------
1) Team Roles and Ownership (Keep exactly these roles)
------------------------------------------------------------

Person 1: Lead + Documentation + Streamlit bonus
Person 2: Data + DVC + SHAP bonus
Person 3: Modeling + MLflow + Optuna bonus
Person 4: CI/CD + Deployment + Cloud deployment bonus
Person 5: Monitoring + Alerts + Slack/email bonus
Person 6: Presentation + Video + Animated GIF bonus

Collaboration rule:
- Every critical milestone must be reproducible on all 6 laptops.
- Every person contributes both code and evidence (screenshots, logs, PR review, or report section).

------------------------------------------------------------
2) Success Metrics (Technical + Business)
------------------------------------------------------------

Technical goals:
- AUPRC >= 0.85
- Recall >= 0.92
- Precision >= 0.80

Business and operations goals:
- Inference latency < 50 ms (single request, local container baseline)
- False Positive Rate < 5%
- Drift alert visibility < 5 minutes from monitoring job execution

Definition of "done" for final demo:
- Full repo history from Day 0 to final tag
- Automated CI runs green on push/PR
- API serving and metrics stack run with Docker Compose
- Drift report and dashboard screenshots available
- Retrain/promote path documented and runnable

------------------------------------------------------------
3) Day 0 (30 minutes): Repository Bootstrap
------------------------------------------------------------

Owner: Person 1

Steps:
1. Create GitHub repository: `MLOPs_credit_fraud_detection` (Public).
2. Add initial files:
   - `README.md` (title + all 6 names + roles)
   - `.gitignore` (GitHub Python template)
   - `.env.example`
3. Invite Persons 2-6 as collaborators.
4. Everyone clones and switches to setup branch:

```bash
git clone https://github.com/HelcurtLordno1/MLOPs_credit_fraud_detection.git
cd MLOPs_credit_fraud_detection
git checkout -b setup
```

Checkpoint evidence:
- Screenshot of collaborators list
- Screenshot of branch and initial commit

------------------------------------------------------------
4) Week 1: Foundations (PDF Weeks 7-10)
------------------------------------------------------------

Day 1 - Project Structure + Problem Definition
Owner: Person 1

Create this exact structure:

```text
Project_mlops_credit_fraud_detection/
|-- src/
|   |-- ml/
|   |-- app/
|   |-- database/
|   `-- monitoring/
|-- configs/
|-- data/raw/
|-- data/processed/
|-- scripts/
|-- tests/
|-- reports/
|-- infra/
|   `-- prometheus/
|-- docker/
|-- .github/workflows/
|-- docs/screenshots/
|-- requirements.txt
|-- dvc.yaml
`-- README.md
```

Update `README.md` top section with:
- Use case: Real-time fraud detection for banks
- Motivation: Banks lose ~$32B yearly
- Success metrics (technical + business, above)
- Team member names + roles

Team action:

```bash
git add README.md
git commit -m "Person1: Initial structure + problem definition"
git push origin setup
```

Person 1 opens PR: "Setup project structure".
All members review and approve.

------------------------------------------------------------
Day 2-3 - Data Pipeline + DVC (15% Data rubric)
Owner: Person 2
------------------------------------------------------------

1. Put `creditcard.csv` into `data/raw/`.

2. Create `requirements.txt` with this baseline:

```txt
pandas
numpy
scikit-learn
mlflow
fastapi
uvicorn
evidently
prometheus-client
pydantic
pyyaml
python-dotenv
joblib
dvc
pytest
black
flake8
```

3. Initialize DVC:

```bash
dvc init
```

4. Create `dvc.yaml`:

```yaml
stages:
  prepare:
    cmd: python src/ml/data.py
    deps:
      - data/raw/creditcard.csv
    outs:
      - data/processed/train.parquet
      - data/processed/test.parquet
      - data/processed/reference.parquet
```

5. Create `src/ml/data.py`:

```python
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/raw/creditcard.csv")
train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["Class"])

train.to_parquet("data/processed/train.parquet")
test.to_parquet("data/processed/test.parquet")

# Reference sample for future drift detection
reference = train.sample(10000, random_state=42)
reference.to_parquet("data/processed/reference.parquet")

print("Data prepared!")
```

6. Run pipeline:

```bash
dvc add data/raw/creditcard.csv
dvc repro
dvc push
```

7. Every teammate runs `dvc pull` and `dvc repro` locally.

8. EDA evidence from all members:
- Create an individual notebook in project root.
- Plot class imbalance.
- Save figure as `reports/eda.png`.

9. Commit DVC-managed metadata only:

```bash
git add data/raw/.gitignore data/.dvc dvc.yaml dvc.lock
git commit -m "Person2: DVC data versioning + prepared Parquet"
```

README update by Person 2:
- Kaggle dataset link
- Class imbalance (0.172% fraud)
- DVC commands used
- Data quality note (missing values check)

------------------------------------------------------------
Day 4-5 - Modeling + MLflow (15% Modeling rubric)
Owner: Person 3
------------------------------------------------------------

1. Create `configs/training.yaml`:

```yaml
model:
  type: LogisticRegression
  class_weight: balanced
preprocessing:
  scaler: StandardScaler
thresholds:
  min_auc: 0.90
```

2. Create `src/ml/train.py`:

```python
import mlflow
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import yaml

mlflow.set_experiment("credit-fraud")

with mlflow.start_run():
    config = yaml.safe_load(open("configs/training.yaml"))
    train = pd.read_parquet("data/processed/train.parquet")
    X = train.drop("Class", axis=1)
    y = train["Class"]

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced")),
        ]
    )
    pipe.fit(X, y)

    joblib.dump(pipe, "model.joblib")
    mlflow.log_metric("train_auc", 0.95)  # placeholder for first baseline
    mlflow.sklearn.log_model(pipe, "model")

    print("Model trained and logged!")
```

3. Run and validate:

```bash
python -m src.ml.train
mlflow ui
```

4. All members:
- Run training locally.
- Open MLflow at `http://127.0.0.1:5000`.
- Screenshot your run.

5. Person 3 registers model as `baseline` in MLflow Registry and writes Modeling section in `README.md`.

------------------------------------------------------------
Day 6-7 - Tests + Reproducibility (All members)
------------------------------------------------------------

1. Add `tests/test_model.py` with one passing test.
2. Run:

```bash
pytest -v
```

3. Person 1 updates reproducibility section in `README.md`.
4. Person 1 creates `docs/reproducibility_proof.md` with:
- Python version used
- Commands to reproduce data + train
- Expected outputs/artifacts

5. Merge all Week 1 PRs and create tag:

```bash
git tag week1-complete
git push origin week1-complete
```

------------------------------------------------------------
5) Week 2: CI/CD + Deployment (25% MLOps pipeline)
------------------------------------------------------------

Day 1-2 - CI with GitHub Actions
Owner: Person 4

Create `.github/workflows/ci.yaml`:

```yaml
name: CI
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt
      - run: pytest
      - run: black --check .
      - run: flake8
      - name: Build Docker
        run: docker build -f docker/Dockerfile.app -t app .
```

Validation:
- Push a small change and verify CI passes (green badge).

Day 3-4 - FastAPI + Docker Compose
Owner: Person 4

Create `src/app/api.py`:

```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load("model.joblib")


class Transaction(BaseModel):
    features: list[float]


@app.post("/predict")
def predict(tx: Transaction):
    pred = model.predict([tx.features])[0]
    return {"fraud_probability": float(pred)}
```

Create `docker/Dockerfile.app`:

```dockerfile
FROM python:3.10
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["uvicorn", "src.app.api:app", "--host", "0.0.0.0"]
```

Create `infra/docker-compose.yaml`:

```yaml
services:
  app:
    build: .
    ports: ["8000:8000"]
  mlflow:
    image: ghcr.io/mlflow/mlflow
    ports: ["5000:5000"]
  postgres:
    image: postgres:14
  prometheus:
    image: prom/prometheus
    volumes: ["./infra/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml"]
  grafana:
    image: grafana/grafana
    ports: ["3000:3000"]
```

Run stack:

```bash
docker compose -f infra/docker-compose.yaml up --build -d
```

Test API:

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"features":[0,1.2,...]}'
```

Day 5-7
- Create `scripts/promote_model.py` (simple model promotion helper).
- Add 1-2 extra experiments in MLflow.
- Merge and tag Week 2.

```bash
git tag week2-complete
git push origin week2-complete
```

------------------------------------------------------------
6) Week 3: Monitoring + Final Delivery
------------------------------------------------------------

Day 1-2 - Monitoring and Drift
Owner: Person 5

1. Create `src/monitoring/drift_job.py` using minimal Evidently report flow.
2. Create `infra/prometheus/prometheus.yml`.
3. Add Grafana dashboard JSON export to repository.
4. Generate evidence:
- Evidently HTML report screenshot
- Live Grafana chart screenshot

Day 3 - Controlled Drift + Retrain + Promote
Owner: Person 5 with Person 3 support

Workflow:
1. Simulate drifted batch.
2. Run drift job and show detection.
3. Retrain model.
4. Promote new candidate model.
5. Re-test API performance and key metrics.

Day 4-7 - Final Communication Package
Owner: Person 6 (with all members)

1. Build 12-slide final deck.
2. Record 5-minute final video.
3. Every member speaks 15-20 seconds minimum.
4. Person 1 completes 4-page final report.
5. Add all individual bonus contributions.

Final release:

```bash
git tag v1.0-final
git push origin v1.0-final
```

------------------------------------------------------------
7) Quality Gate Checklist (Use before final submission)
------------------------------------------------------------

- [ ] Repository history starts from Day 0 setup and shows iterative commits.
- [ ] DVC can reproduce processed datasets from raw file.
- [ ] MLflow has baseline + additional experiments and registered model.
- [ ] API container responds correctly on `/predict`.
- [ ] CI pipeline passes tests, formatting, linting, and Docker build.
- [ ] Monitoring artifacts (Evidently + Prometheus/Grafana) are present.
- [ ] Retrain/promote process is demonstrated and documented.
- [ ] Final report, slides, and video are complete and team-wide.

------------------------------------------------------------
8) Exact Deliverables
------------------------------------------------------------

1. Your own GitHub repository with full project history.
2. Running FastAPI service and Grafana dashboard (localhost or cloud).
3. Final report + slide deck + 5-minute video (all 6 members speak).

This plan is intentionally practical: detailed enough to execute professionally, but simple enough to finish within 3 weeks without overcomplicating the MLOps implementation.

------------------------------------------------------------
9) Gap-Driven Upgrade Plan for Current Repository State
------------------------------------------------------------

This section is based on direct comparison against a stronger working reference repository and your current implementation state.

Current repository reality (important):
- `src/app/`, `src/monitoring/`, `src/database/`, `.github/workflows/`, `infra/prometheus/`, and `docker/` are still placeholders (`.gitkeep`).
- `dvc.yaml` has only `prepare` stage.
- `tests/test_model.py` is only a config-exists check.
- You have baseline training metrics logging, but no deployable API/monitoring lifecycle.

### A) Exact missing work for data quality and reproducibility

1. Upgrade `src/ml/data.py`:
- Add deterministic function-based pipeline.
- Add schema checks (`Class` exists, expected feature columns).
- Add null/duplicate checks and summary logs.
- Export `current.parquet` in addition to `train/test/reference`.

2. Upgrade `dvc.yaml` from 1 stage to 3 stages:
- `prepare` -> data split + quality outputs
- `train` -> train model and log to MLflow
- `evaluate` -> run model evaluation and export metrics report

3. Add reproducibility artifacts:
- data quality JSON report
- dataset fingerprint/checksum metadata
- explicit random seed logging in all scripts

### B) Exact missing work for model quality (accuracy/precision/recall)

1. Add validation split and threshold tuning
- Tune threshold on validation set (not test).
- Select threshold by objective:
  - maximize precision with recall >= 0.92, or
  - maximize F1 for fraud class.
- Save chosen threshold in `configs/inference.yaml`.

2. Add challenger modeling track
- Keep Logistic Regression baseline.
- Add one challenger (XGBoost/LightGBM) and compare on AUPRC + recall + precision.
- Add hyperparameter optimization (Optuna or controlled grid search).

3. Improve imbalance strategy
- Compare class weighting vs resampling techniques.
- For tree model, tune `scale_pos_weight`.

4. Add calibration and business-aware metrics
- Add probability calibration when needed.
- Track false positive rate and cost-weighted error metric.

### C) Exact missing work for serving and model lifecycle

1. Implement API in `src/app/api.py` with:
- `/health`
- `/predict`
- `/metrics`
- `/model_info`
- `/reload`

2. Model loading policy:
- Load from MLflow alias (`production`) first.
- Fallback to local model artifact if registry unavailable.

3. Add lifecycle scripts:
- `scripts/validate_model.py`
- `scripts/promote_model.py`
- `scripts/promote_and_restart.py`

Promotion rule:
- Only promote if candidate passes all defined gates.

### D) Exact missing work for monitoring and alerting

1. Create `src/monitoring/drift_job.py`
- Compare reference vs current data with Evidently.
- Save report to `reports/drift_*.html`.

2. Create `src/monitoring/production_monitor.py`
- Read prediction logs and evaluate drift/performance trend.

3. Create `infra/prometheus/prometheus.yml`
- Scrape `app:8000/metrics` every 5s.

4. Add Grafana dashboard JSON export to repo.

### E) Exact missing work for CI/CD and testing

1. Create `.github/workflows/ci.yaml`
- install dependencies
- run `pytest`
- run formatting/lint checks
- build Docker images

2. Expand tests:
- data validation tests
- model metric gate tests
- API endpoint and schema tests
- prediction range/type tests

3. Add deployment smoke test
- start compose stack in CI (or local scripted validation)
- hit `/health` and one `/predict` request

------------------------------------------------------------
10) Refined Weekly Execution (What to add to original plan)
------------------------------------------------------------

Week 1 add-ons (highest impact on model quality):
1. Complete data quality checks + current.parquet output.
2. Add threshold tuning and validation split logic.
3. Add evaluation script and stricter test suite.

Week 2 add-ons (highest impact on MLOps completeness):
1. Implement API + Prometheus metrics instrumentation.
2. Implement Dockerfiles and full docker compose stack.
3. Implement CI workflow with tests + lint + docker build.
4. Implement validate/promote scripts for model lifecycle.

Week 3 add-ons (highest impact on operations readiness):
1. Implement drift/production monitoring scripts.
2. Run controlled drift simulation and generate report evidence.
3. Perform retrain -> validate -> promote -> reload demo.
4. Capture final dashboard/report evidence for submission.

------------------------------------------------------------
11) New Mandatory Quality Gates (Do not skip)
------------------------------------------------------------

Model gates:
- AUPRC >= 0.85
- Recall >= 0.92
- Precision >= 0.80
- FPR < 0.05 at chosen operating threshold

System gates:
- API p95 latency < 50 ms (local baseline)
- CI green on push and PR
- Drift report generated and archived
- Promotion only from validated model version

Submission gates:
- Reproducible setup on all team machines
- Full evidence package: logs, screenshots, reports, run commands

This section should be treated as required remediation for your current repo state, not optional enhancement.

