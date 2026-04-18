🛠️ Global Context & Pre-Configuration (Team Lead - Week 0)
Before Week 10 begins, the Team Lead (or you) must perform the Repository Reset. This ensures the new work does not carry over the old Git history from the baseline project.
Clone & Clean: Clone hoducmanh222/fraud-detection locally. Delete the .git folder.
Initialize New Repo: git init, commit all files as "Initial baseline structure".
Push to Main Repo: Push this clean state to your HelcurtLordno1/MLOPs_credit_fraud_detection repository.
Data Preparation: Download the new dataset (fraudTrain.csv) and place it inside data/raw/. Do not commit this file; DVC will handle it.

📅 Week 10: Member 1 (The Foundation Builder) - ghi thêm README
Role: Data Engineer & Pipeline Architect
Core Task: Establish the DVC pipeline for the new dataset and adapt the data processing layer.
This is the most critical week. If this fails, the rest of the team cannot work. You are creating the "fuel" for the ML engine.
1. DVC Remote Configuration
Action: Initialize DVC (dvc init if not already present). Configure the Google Drive remote.
Command: dvc remote add -d storage gdrive://<your_folder_id>
Credential Handling: You need the GDRIVE_CREDENTIALS_DATA JSON string set as an environment variable or in the DVC config.
Why this matters: This allows all subsequent members to pull the exact version of the data you process.
2. Adapting src/fraud_detection/data/ (Implicit in Pipeline)
While the folder shows api and modeling, the baseline likely has a prepare.py CLI command.
File to Analyze: src/fraud_detection/utils/cli.py (Look for the prepare function entry point).
Target Script: You need to locate the script that generates data/processed/train.parquet. In the baseline, this is likely a function called prepare_data() triggered by CLI.
Feature Engineering Mandate (Crucial for New Dataset):
Temporal Features: Parse trans_date_trans_time (or unix_time). Create hour_of_day, day_of_week, and is_weekend.
Customer Velocity: Group by cc_num to calculate customer_avg_amt, customer_txn_count_last_24h.
Merchant Risk: Group by merchant to calculate merchant_fraud_rate (using target is_fraud).
Categorical Encoding: Use category and city as features. Decide on Target Encoding or Frequency Encoding (Logistic Regression likes Frequency Encoding better than One-Hot for high cardinality).
3. Running the First Pipeline Stage
Command: dvc repro prepare
Verification: Check data/processed/ for train.parquet, val.parquet, test.parquet.
Commit & Push:
bash
git add dvc.lock data/.gitignore data/raw/.gitignore
git commit -m "feat: DVC pipeline setup and data preparation for new dataset"
dvc push
git push

📅 Week 11: Member 2 (The Model Trainer)
Role: ML Engineer (Training & Experimentation)
Core Task: Modify src/fraud_detection/modeling/train.py to work with the new features and log experiments to MLflow.
You are taking the data prepared by Member 1 and building the intelligence of the system.
1. Environment Check
Action: dvc pull (to get the processed data from Day 1).
Note: Ensure MLflow tracking server is accessible (if using a remote one) or just use local file storage for now.
2. Deep Dive: src/fraud_detection/modeling/train.py
Modify Feature Loading: Ensure the script reads the exact column names outputted by Member 1's prepare stage.
Target Column: The new dataset uses is_fraud (0/1) instead of PaySim's isFraud. Change every occurrence of isFraud to is_fraud in the code.
Model Selection Strategy:
Keep the Logistic Regression baseline (fast, explainable).
Keep the LightGBM classifier (handles imbalance well).
Important: The new dataset is highly imbalanced (~0.5% fraud). Ensure the code uses scale_pos_weight in LightGBM or class_weight='balanced' in Logistic Regression.
3. MLflow Integration Check
File: Look for mlflow.log_param and mlflow.log_metric calls inside train.py.
Action: Run the training script manually once to ensure it doesn't crash.
Command: python -m fraud_detection.utils.cli train (or similar, check pyproject.toml for script entry points).
DVC Execution: Once manual run succeeds, run it via DVC: dvc repro train.
4. Deliverable
Commit: Updated src/fraud_detection/modeling/train.py.
Data: Updated dvc.lock (reflecting the new model artifact).
MLflow Run ID: Note the run ID of the newly trained model for the next member.

📅 Week 12: Member 3 (The Quality Gatekeeper)
Role: MLOps Engineer (Evaluation & Promotion)
Core Task: Adapt evaluation metrics and implement the promotion logic in monitoring/promotion.py.
You are the bridge between a "trained model" and a "deployed model." You decide if the model is safe to use.
1. Adapt src/fraud_detection/modeling/evaluate.py
Metric Adjustment: Credit card fraud detection cares about Recall (catching fraud) and Precision (not annoying customers). The baseline uses AUPRC (Area Under Precision-Recall Curve). This is perfect—do not change the metric logic, just ensure the column names match the new data (is_fraud).
2. Implement Promotion Logic: src/fraud_detection/monitoring/promotion.py
Goal: Compare the "Champion" model (currently in production) vs the "Challenger" (just trained by Member 2).
Logic to Implement:
Load Challenger metrics from MLflow (using the Run ID from Day 2).
Load Champion metrics from MLflow Model Registry (if exists).
Rule: If Challenger Recall >= Champion Recall AND Challenger Precision >= (Champion Precision * 0.95), PROMOTE.
Action: Register the Challenger as the new "Production" stage model in MLflow.
Code Modification: Look for mlflow.register_model() and client.transition_model_version_stage().
3. DVC Pipeline Update
File: dvc.yaml
Action: Ensure the evaluate stage depends on the train output. Ensure the promote stage depends on evaluate output.
Run: dvc repro evaluate -> dvc repro promote.

📅 Week 13: Member 4 (The Interface Engineer)
Role: API & Frontend Developer
Core Task: Fix the FastAPI schema and ensure the Streamlit UI works with the new data structure.
Members 1-3 built the engine. You are building the steering wheel and dashboard.
1. API Schema Adaptation: src/fraud_detection/api/schema.py
The Problem: The baseline Transaction Pydantic model expects PaySim fields (e.g., oldbalanceOrg, newbalanceOrig). The new dataset has cc_num, merchant, category, amt, city, job, dob.
Action: Rewrite the Transaction class to match the new dataset's required input fields. Do not include the target is_fraud; that is what the API predicts.
Example:
python
class Transaction(BaseModel):
    cc_num: str
    merchant: str
    category: str
    amt: float
    first: str
    last: str
    gender: str
    street: str
    city: str
    state: str
    zip: int
    lat: float
    long: float
    city_pop: int
    job: str
    dob: str
    trans_num: str
    unix_time: int
    merch_lat: float
    merch_long: float
2. Pipeline Alignment: src/fraud_detection/api/pipeline.py
Function: preprocess_input()
Action: This function must transform the raw API JSON into the exact DataFrame format that the model expects (which was defined by Member 1's feature engineering). This is the most common point of failure.
Verification: Write a small test script to hit the /predict endpoint locally with a sample JSON from the test set. Ensure it returns {"fraud_probability": 0.XXX}.
3. Streamlit UI Check: src/streamlit_app/app.py
Action: Update the input fields in the Streamlit sidebar to match the new Transaction schema. You might need to add text inputs for merchant, category, job, etc.

📅 Week 14: Member 5 (The Automation Architect)
Role: DevOps/MLOps Infrastructure Architect
Core Task: Containerize the full stack, create Kubernetes deployment manifests, and automate retraining & rollout.
Goal: The system is no longer just Docker Compose—it's a cloud‑native, Kubernetes‑ready ML application.
1. Docker Image Refinement & Registry Push
File: Dockerfile (Root level)
Action: Ensure the Docker image build is optimized (multi‑stage builds, .dockerignore).
Registry Setup: Create a public repository on Docker Hub (e.g., teamname/fraud-api).
Test Build & Push:
bash
docker build -t teamname/fraud-api:latest .
docker push teamname/fraud-api:latest
2. Kubernetes Manifest Creation (The New Core Work)
Directory: Create a new folder k8s/ at the root of the repository.
Files to Write:
api-deployment.yaml
Defines 3 replicas of the FastAPI container. Specifies environment variables (e.g., MLFLOW_TRACKING_URI pointing to the internal MLflow service). Uses a livenessProbe and readinessProbe on the /health endpoint.
api-service.yaml
Type: LoadBalancer (if using a cloud cluster) or NodePort (for local Minikube). Exposes port 8000.
mlflow-deployment.yaml
Runs the MLflow tracking server. Mounts a PersistentVolumeClaim to ensure model registry and artifacts survive pod restarts.
mlflow-service.yaml
Type: ClusterIP (internal only, accessed by the API and Streamlit pods).
streamlit-deployment.yaml
Optional: Deploys the Streamlit dashboard for monitoring.
3. Automating Kubernetes Rollouts in CI/CD
File: .github/workflows/retrain.yml
New Steps to Add:
Install kubectl in the GitHub Actions runner.
Authenticate to the Kubernetes cluster (using KUBECONFIG stored as a GitHub Secret).
Update Image Tag: After retraining produces a new model, build a new Docker image tagged with the MLflow run ID or commit SHA.
Apply Manifests: kubectl apply -f k8s/
Rollout Restart: kubectl rollout restart deployment/fraud-api to force pods to pull the new image.
4. Local Testing with Minikube (Verification)
Action: Before pushing changes, Member 5 must test the manifests locally using Minikube or Kind.
Commands:
bash
minikube start
kubectl apply -f k8s/
minikube service fraud-api-service --url
Validation: Send a test request to the exposed URL. Verify the prediction works.
5. Monitoring Script Alignment
File: src/fraud_detection/monitoring/drift.py
Action: Ensure the drift detection script can run as a Kubernetes CronJob in the future (or just note it in the README). No immediate code changes required, but adapt the logic to output logs in JSON format for easier ingestion by cloud monitoring tools (e.g., Stackdriver, CloudWatch).
Deliverables for Member 5:
Optimized Dockerfile and pushed image to a registry.
Complete set of Kubernetes YAML files in k8s/ directory.
Updated .github/workflows/retrain.yml with Kubernetes deployment steps.
Evidence of successful deployment on a local Minikube cluster (screenshot in PR).

📅 Week 15: Member 6 (The Quality Assurance & Documentation Lead)
Role: Project Manager & Technical Writer
Core Task: End-to-end validation, reproducibility check, and final presentation preparation.
You are the final set of eyes. If you can run it from scratch, the teacher can run it from scratch.
1. The "Nuke and Pave" Test (Critical for Academic Integrity)
Action: Delete your local fraud-detection folder entirely.
Clone: git clone <your-repo> from scratch.
Execute:
pip install -e . (using pyproject.toml).
dvc pull (Pulls the exact data and model from Google Drive).
docker-compose up (Starts the whole system).
Validation: Open browser. Send a request. Get a prediction.
2. Final Report & README Polish
File: README.md
Updates:
Change dataset description to "Kaggle Credit Card Fraud Detection".
Add a Team Contribution Table mapping each member to their day's task.
Add a diagram of the DVC pipeline (dvc dag output).
3. Final Commit & Release
Action: Tag the repo as v1.0.0.
Push: Ensure all changes (including the updated dvc.lock from the retraining action) are merged to main.
