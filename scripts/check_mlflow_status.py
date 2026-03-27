from pathlib import Path

from mlflow.tracking import MlflowClient


project_root = Path(__file__).resolve().parents[1]
tracking_uri = f"sqlite:///{(project_root / 'mlflow.db').as_posix()}"
client = MlflowClient(tracking_uri=tracking_uri)

exps = [e for e in client.search_experiments() if e.name == "credit-fraud"]
print("tracking_uri=", tracking_uri)
print("experiments_found=", len(exps))

if exps:
    exp = exps[0]
    runs = client.search_runs([exp.experiment_id], order_by=["attributes.start_time DESC"], max_results=10)
    print("latest_runs=", len(runs))
    for run in runs:
        print(run.info.run_name, run.info.status, run.data.metrics)

models = [m for m in client.search_registered_models() if m.name == "credit-fraud-model"]
print("registered_models_found=", len(models))
for m in models:
    latest_versions = [v.version for v in m.latest_versions] if m.latest_versions else []
    print(m.name, latest_versions)
