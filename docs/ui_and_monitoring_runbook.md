# UI and Monitoring Runbook (FastAPI + Streamlit + Grafana)

This document explains exactly what to open, what to look for, and how to confirm the system is healthy.

## 1. Access Matrix

| Surface | URL (default) | Auth | What you should see |
|---|---|---|---|
| FastAPI Swagger | http://localhost:8000/docs | None | Interactive API docs with prediction and monitoring endpoints |
| FastAPI Health | http://localhost:8000/health | None | JSON with `status`, `model_loaded`, `model_name`, `version` |
| FastAPI Metrics | http://localhost:8000/metrics | None | Prometheus metrics text including prediction counters and latency histogram |
| Streamlit Dashboard | http://localhost:8501 | None | Fraud Detection dashboard with tabs and KPI cards |
| Grafana | http://localhost:3000 | `admin` / `admin` (unless changed) | Grafana home/login and the Fraud API dashboard |
| Prometheus UI | http://localhost:9090 | None | Query console and target status |

Notes:
- If you run alternate ports (for example API `8002`, Streamlit `8502`), replace URLs accordingly.
- In host browser, use `localhost`. Reserve `host.docker.internal` for container-to-host calls.

## 2. FastAPI: What To Check

## 2.1 Liveness and model readiness

Open `GET /health` and confirm:
- `status` is `ok` (or at least `degraded` with a clear reason).
- `model_loaded` is `true` for prediction readiness.
- `model_name` and `version` are non-empty when loaded.

Expected shape:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "...",
  "version": "..."
}
```

## 2.2 Prediction endpoints

Check in Swagger (`/docs`):
- `POST /api/v1/predict`
- `POST /api/v1/predict/batch`
- `GET /api/v1/model`
- `GET /api/v1/drift`

Expected single prediction response fields:

```json
{
  "fraud_probability": 0.1234,
  "is_fraud": false,
  "threshold": 0.5
}
```

## 2.3 Prometheus metrics contract

Open `GET /metrics` and verify these metric families exist:
- `fraud_api_predictions_total{outcome="legit|fraud"}`
- `fraud_api_prediction_latency_seconds_bucket`
- `fraud_api_prediction_latency_seconds_count`
- `fraud_api_model_loaded_total`

Operational meaning:
- Prediction traffic should increase `fraud_api_predictions_total`.
- Latency panel reads from histogram buckets.

## 3. Streamlit: What To Check

Open the dashboard and confirm the following areas:

## 3.1 Sidebar status

- API Status: `Online` when `/health` reports model loaded.
- Model and Version are shown.
- Pipeline Status shows:
  - Best model
  - Validation AUPRC
  - Promotion badge (`Promoted` or `Pending`)

## 3.2 Top KPI cards

Confirm cards render values:
- Val AUPRC
- Val Recall
- Val Precision
- Feature Drift (`features_with_drift / total_features_checked`)

## 3.3 Tabs and expected behavior

- Single Prediction:
  - Fill form or keep defaults.
  - Click `Analyze Transaction`.
  - See result box and probability.
- Batch Scoring:
  - Upload CSV matching transaction schema.
  - Click `Score All Transactions`.
  - See totals and per-row predictions.
- Model Card:
  - Shows active model metadata from API.
  - Includes model comparison table from local metrics.
- Drift Monitor:
  - Shows drift summary, target distribution, and feature-level details.
- Pipeline Status:
  - Shows promotion decision and latest training run summary.

## 4. Grafana: What To Check

## 4.1 Login and dashboard

- Login URL: `http://localhost:3000`
- Credentials (current local setup): `admin` / `admin`
- Open dashboard UID:
  - `http://localhost:3000/d/fraud-api-dashboard/fraud-api-dashboard`

## 4.2 Panel expectations

Panel 1: `Predictions per Outcome`
- Query: `sum(fraud_api_predictions_total) by (outcome)`
- Expected: lines/series for `legit` and optionally `fraud`.

Panel 2: `P95 Prediction Latency`
- Query: `histogram_quantile(0.95, sum(fraud_api_prediction_latency_seconds_bucket) by (le))`
- Expected: non-empty line after traffic is sent.

Recommended dashboard controls:
- Time range: `Last 1 hour`
- Refresh: `5s` or `10s`

## 5. End-to-End Smoke Test (UI + Monitoring)

Run API and Streamlit, then generate traffic by clicking `Analyze Transaction` several times in Streamlit.

Expected chain:
1. Streamlit prediction returns response.
2. FastAPI `fraud_api_predictions_total` increases.
3. Prometheus query returns updated value.
4. Grafana panel updates within one scrape interval.

Useful Prometheus checks:
- `sum(fraud_api_predictions_total) by (outcome)`
- `histogram_quantile(0.95, sum(fraud_api_prediction_latency_seconds_bucket) by (le))`

## 6. Common Issues and Fixes

No data in Grafana:
- Confirm Prometheus target is `UP`.
- Confirm dashboard datasource UID is `prometheus`.
- Confirm API metrics endpoint is reachable at the target configured in Prometheus.
- Expand time range to `Last 1 hour` and refresh.

Browser timeout with `host.docker.internal`:
- Use `localhost` from host browser.
- Keep `host.docker.internal` only for container-to-host routing.

Streamlit cannot predict:
- Confirm `API_URL` for Streamlit points to a live API.
- Confirm `/health` returns `model_loaded: true`.

## 7. File References

- FastAPI app: `src/fraud_detection/api/main.py`
- API schemas: `src/fraud_detection/api/schemas.py`
- Streamlit app: `streamlit_app/app.py`
- Streamlit helpers: `src/fraud_detection/ui_helpers.py`
- Prometheus config: `deployment/monitoring/prometheus.yml`
- Grafana datasource provisioning: `deployment/monitoring/grafana/provisioning/datasources/prometheus.yaml`
- Grafana dashboard: `deployment/monitoring/grafana/dashboards/fraud-api-dashboard.json`
