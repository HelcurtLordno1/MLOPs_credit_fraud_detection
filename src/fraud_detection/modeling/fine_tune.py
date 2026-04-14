"""Hyperparameter tuning and fine-tuning for fraud detection models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fraud_detection.data.pipeline import load_splits
from fraud_detection.data.schema import TARGET_COLUMN
from fraud_detection.modeling.evaluate import compute_metrics
from fraud_detection.modeling.train import prepare_features, train_lightgbm, train_logistic_regression
from fraud_detection.utils.paths import find_project_root


class ModelFinetuner:
    """Fine-tune models using Optuna for hyperparameter optimization."""

    def __init__(self, X_train: pd.DataFrame, y_train: pd.Series,
                 X_val: pd.DataFrame, y_val: pd.Series,
                 random_state: int = 42):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.random_state = random_state
        self.best_params = {}
        self.best_scores = {}

    def objective_logistic(self, trial: optuna.Trial) -> float:
        """Objective function for Logistic Regression hyperparameter optimization."""
        C = trial.suggest_float('C', 0.001, 100.0, log=True)
        max_iter = trial.suggest_int('max_iter', 500, 3000, step=100)
        
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                C=C,
                max_iter=max_iter,
                solver='lbfgs',
                class_weight='balanced',
                random_state=self.random_state,
            ))
        ])
        
        model.fit(self.X_train, self.y_train)
        y_pred_proba = model.predict_proba(self.X_val)[:, 1]
        
        auprc = average_precision_score(self.y_val, y_pred_proba)
        return auprc

    def objective_lightgbm(self, trial: optuna.Trial) -> float:
        """Objective function for LightGBM hyperparameter optimization."""
        n_estimators = trial.suggest_int('n_estimators', 100, 800, step=50)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
        num_leaves = trial.suggest_int('num_leaves', 20, 127)
        subsample = trial.suggest_float('subsample', 0.5, 1.0)
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
        reg_lambda = trial.suggest_float('reg_lambda', 0.0, 5.0)
        reg_alpha = trial.suggest_float('reg_alpha', 0.0, 5.0)
        min_child_samples = trial.suggest_int('min_child_samples', 10, 100, step=5)
        
        neg_count = int((self.y_train == 0).sum())
        pos_count = int((self.y_train == 1).sum())
        scale_pos_weight = neg_count / max(pos_count, 1)
        
        model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            min_child_samples=min_child_samples,
            scale_pos_weight=scale_pos_weight,
            objective='binary',
            metric='auc',
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1,
        )
        
        model.fit(self.X_train, self.y_train)
        y_pred_proba = model.predict_proba(self.X_val)[:, 1]
        
        auprc = average_precision_score(self.y_val, y_pred_proba)
        return auprc

    def optimize_logistic_regression(self, n_trials: int = 50) -> dict[str, Any]:
        """Optimize Logistic Regression hyperparameters."""
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        
        study.optimize(self.objective_logistic, n_trials=n_trials, show_progress_bar=True)
        
        best_trial = study.best_trial
        self.best_params['logistic'] = best_trial.params
        
        # Train final model with best params
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                C=best_trial.params['C'],
                max_iter=int(best_trial.params['max_iter']),
                solver='lbfgs',
                class_weight='balanced',
                random_state=self.random_state,
            ))
        ])
        
        model.fit(self.X_train, self.y_train)
        y_pred_proba = model.predict_proba(self.X_val)[:, 1]
        
        self.best_scores['logistic'] = {
            'val_auprc': average_precision_score(self.y_val, y_pred_proba),
            'val_recall': recall_score(self.y_val, (y_pred_proba > 0.5).astype(int)),
            'val_precision': precision_score(self.y_val, (y_pred_proba > 0.5).astype(int)),
            'val_auc_roc': roc_auc_score(self.y_val, y_pred_proba),
        }
        
        return {
            'best_params': self.best_params['logistic'],
            'best_auprc': best_trial.value,
            'best_scores': self.best_scores['logistic'],
            'n_trials': len(study.trials),
        }

    def optimize_lightgbm(self, n_trials: int = 50) -> dict[str, Any]:
        """Optimize LightGBM hyperparameters."""
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        
        study.optimize(self.objective_lightgbm, n_trials=n_trials, show_progress_bar=True)
        
        best_trial = study.best_trial
        self.best_params['lightgbm'] = best_trial.params
        
        # Train final model with best params
        neg_count = int((self.y_train == 0).sum())
        pos_count = int((self.y_train == 1).sum())
        scale_pos_weight = neg_count / max(pos_count, 1)
        
        model = lgb.LGBMClassifier(
            n_estimators=int(best_trial.params['n_estimators']),
            learning_rate=best_trial.params['learning_rate'],
            num_leaves=int(best_trial.params['num_leaves']),
            subsample=best_trial.params['subsample'],
            colsample_bytree=best_trial.params['colsample_bytree'],
            reg_lambda=best_trial.params['reg_lambda'],
            reg_alpha=best_trial.params['reg_alpha'],
            min_child_samples=int(best_trial.params['min_child_samples']),
            scale_pos_weight=scale_pos_weight,
            objective='binary',
            metric='auc',
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1,
        )
        
        model.fit(self.X_train, self.y_train)
        y_pred_proba = model.predict_proba(self.X_val)[:, 1]
        
        self.best_scores['lightgbm'] = {
            'val_auprc': average_precision_score(self.y_val, y_pred_proba),
            'val_recall': recall_score(self.y_val, (y_pred_proba > 0.5).astype(int)),
            'val_precision': precision_score(self.y_val, (y_pred_proba > 0.5).astype(int)),
            'val_auc_roc': roc_auc_score(self.y_val, y_pred_proba),
        }
        
        return {
            'best_params': self.best_params['lightgbm'],
            'best_auprc': best_trial.value,
            'best_scores': self.best_scores['lightgbm'],
            'n_trials': len(study.trials),
        }

    def compare_with_original(self, original_params: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Compare fine-tuned parameters with original parameters."""
        comparison = {
            'logistic_regression': {
                'original': original_params.get('logistic', {}),
                'finetuned': self.best_params.get('logistic', {}),
                'same': original_params.get('logistic', {}) == self.best_params.get('logistic', {}),
                'original_scores': {},
                'finetuned_scores': self.best_scores.get('logistic', {}),
            },
            'lightgbm': {
                'original': original_params.get('lightgbm', {}),
                'finetuned': self.best_params.get('lightgbm', {}),
                'same': original_params.get('lightgbm', {}) == self.best_params.get('lightgbm', {}),
                'original_scores': {},
                'finetuned_scores': self.best_scores.get('lightgbm', {}),
            }
        }
        
        return comparison


def _load_train_config(project_root: Path) -> dict[str, Any]:
    config_path = project_root / "configs" / "train.yaml"
    if not config_path.exists():
        return {}

    try:
        import yaml
    except ImportError:
        return {}

    with open(config_path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _evaluate_on_splits(model: Any, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, dict[str, float]]:
    return {
        "train": compute_metrics(y_train, model.predict_proba(X_train)[:, 1]),
        "val": compute_metrics(y_val, model.predict_proba(X_val)[:, 1]),
        "test": compute_metrics(y_test, model.predict_proba(X_test)[:, 1]),
    }


def run_fine_tuning_pipeline(n_trials: int = 15) -> dict[str, Any]:
    """Run Day 4 fine-tuning pipeline and save artifacts for DVC."""
    project_root = find_project_root()
    processed_dir = project_root / "data" / "processed"
    output_dir = project_root / "reports" / "hyperparameter_tuning"
    model_dir = project_root / "models" / "trained"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df = load_splits(processed_dir)
    X_train, y_train = prepare_features(train_df, target_col=TARGET_COLUMN)
    X_val, y_val = prepare_features(val_df, target_col=TARGET_COLUMN)
    X_test, y_test = prepare_features(test_df, target_col=TARGET_COLUMN)

    print(f"Loaded splits: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

    tuner = ModelFinetuner(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
    lr_tuned = tuner.optimize_logistic_regression(n_trials=n_trials)
    lgbm_tuned = tuner.optimize_lightgbm(n_trials=n_trials)

    # Retrain tuned models on train+val for fair test evaluation.
    X_train_val = pd.concat([X_train, X_val], axis=0)
    y_train_val = pd.concat([y_train, y_val], axis=0)

    lr_model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=lr_tuned["best_params"]["C"],
            max_iter=int(lr_tuned["best_params"]["max_iter"]),
            solver="lbfgs",
            class_weight="balanced",
            random_state=42,
        )),
    ])
    lr_model.fit(X_train_val, y_train_val)

    neg_count = int((y_train_val == 0).sum())
    pos_count = int((y_train_val == 1).sum())
    scale_pos_weight = neg_count / max(pos_count, 1)
    lgbm_model = lgb.LGBMClassifier(
        n_estimators=int(lgbm_tuned["best_params"]["n_estimators"]),
        learning_rate=float(lgbm_tuned["best_params"]["learning_rate"]),
        num_leaves=int(lgbm_tuned["best_params"]["num_leaves"]),
        subsample=float(lgbm_tuned["best_params"]["subsample"]),
        colsample_bytree=float(lgbm_tuned["best_params"]["colsample_bytree"]),
        reg_lambda=float(lgbm_tuned["best_params"]["reg_lambda"]),
        reg_alpha=float(lgbm_tuned["best_params"]["reg_alpha"]),
        min_child_samples=int(lgbm_tuned["best_params"]["min_child_samples"]),
        scale_pos_weight=scale_pos_weight,
        objective="binary",
        metric="auc",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    lgbm_model.fit(X_train_val, y_train_val)

    train_cfg = _load_train_config(project_root)
    logistic_cfg = train_cfg.get("model", {}).get("logistic", {})
    lightgbm_cfg = train_cfg.get("model", {}).get("lightgbm", {})

    baseline_lr = train_logistic_regression(X_train, y_train, logistic_cfg)
    baseline_lgbm = train_lightgbm(X_train, y_train, lightgbm_cfg)

    lr_baseline_metrics = _evaluate_on_splits(baseline_lr, X_train, y_train, X_val, y_val, X_test, y_test)
    lgbm_baseline_metrics = _evaluate_on_splits(baseline_lgbm, X_train, y_train, X_val, y_val, X_test, y_test)
    lr_tuned_metrics = _evaluate_on_splits(lr_model, X_train, y_train, X_val, y_val, X_test, y_test)
    lgbm_tuned_metrics = _evaluate_on_splits(lgbm_model, X_train, y_train, X_val, y_val, X_test, y_test)

    lr_payload = {
        "model_type": "LogisticRegression",
        "n_trials": int(n_trials),
        "best_params": lr_tuned["best_params"],
        "baseline_metrics": lr_baseline_metrics,
        "tuned_metrics": lr_tuned_metrics,
        "test_auprc_improvement": float(lr_tuned_metrics["test"]["auprc"] - lr_baseline_metrics["test"]["auprc"]),
    }
    lgbm_payload = {
        "model_type": "LightGBM",
        "n_trials": int(n_trials),
        "best_params": lgbm_tuned["best_params"],
        "baseline_metrics": lgbm_baseline_metrics,
        "tuned_metrics": lgbm_tuned_metrics,
        "test_auprc_improvement": float(lgbm_tuned_metrics["test"]["auprc"] - lgbm_baseline_metrics["test"]["auprc"]),
    }

    with open(output_dir / "lr_tuning_results.json", "w", encoding="utf-8") as handle:
        json.dump(lr_payload, handle, indent=2)
    with open(output_dir / "lgbm_tuning_results.json", "w", encoding="utf-8") as handle:
        json.dump(lgbm_payload, handle, indent=2)

    joblib.dump(lr_model, model_dir / "best_lr_tuned_model.joblib")
    joblib.dump(lgbm_model, model_dir / "best_lgbm_tuned_model.joblib")

    best_model_name = "LGBM Tuned" if lgbm_tuned_metrics["test"]["auprc"] >= lr_tuned_metrics["test"]["auprc"] else "LR Tuned"
    best_model_auprc = lgbm_tuned_metrics["test"]["auprc"] if best_model_name == "LGBM Tuned" else lr_tuned_metrics["test"]["auprc"]
    summary_text = (
        "HYPERPARAMETER TUNING SUMMARY\n"
        f"Trials per model: {n_trials}\n"
        f"LR test AUPRC: baseline={lr_baseline_metrics['test']['auprc']:.4f}, tuned={lr_tuned_metrics['test']['auprc']:.4f}\n"
        f"LGBM test AUPRC: baseline={lgbm_baseline_metrics['test']['auprc']:.4f}, tuned={lgbm_tuned_metrics['test']['auprc']:.4f}\n"
        f"Best tuned model: {best_model_name} (AUPRC={best_model_auprc:.4f})\n"
    )
    with open(output_dir / "tuning_summary_report.txt", "w", encoding="utf-8") as handle:
        handle.write(summary_text)

    result = {
        "best_model": best_model_name,
        "best_test_auprc": float(best_model_auprc),
        "output_dir": str(output_dir.relative_to(project_root)),
    }
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Day 4 fine-tuning pipeline")
    parser.add_argument("--n-trials", type=int, default=15, help="Optuna trials per model")
    args = parser.parse_args()
    run_fine_tuning_pipeline(n_trials=args.n_trials)


if __name__ == "__main__":
    main()
