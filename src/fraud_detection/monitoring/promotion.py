"""Model promotion and lifecycle management."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import mlflow

try:
    from mlflow.tracking import MlflowClient
except ImportError:
    from mlflow import MlflowClient


class ModelPromoter:
    """Manage model promotion between stages (challenger vs champion)."""

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        registry_uri: Optional[str] = None,
    ):
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        if registry_uri:
            mlflow.set_registry_uri(registry_uri)
        self.client = MlflowClient()

    def load_run_metrics(self, run_id: str) -> dict[str, float]:
        run = self.client.get_run(run_id)
        return {key: float(value) for key, value in run.data.metrics.items()}

    @staticmethod
    def _normalize_for_promotion(metrics: dict[str, float]) -> dict[str, float]:
        normalized = dict(metrics)
        if "val_recall" in normalized and "val_precision" in normalized:
            return normalized

        candidates: dict[str, dict[str, float]] = {}
        for key, value in metrics.items():
            if key.endswith("_val_auprc"):
                model_key = key[: -len("_val_auprc")]
                if model_key == "best":
                    continue
                candidates.setdefault(model_key, {})["val_auprc"] = value
            elif key.endswith("_val_recall"):
                model_key = key[: -len("_val_recall")]
                if model_key == "best":
                    continue
                candidates.setdefault(model_key, {})["val_recall"] = value
            elif key.endswith("_val_precision"):
                model_key = key[: -len("_val_precision")]
                if model_key == "best":
                    continue
                candidates.setdefault(model_key, {})["val_precision"] = value

        if not candidates:
            return normalized

        complete = {
            mk: vals
            for mk, vals in candidates.items()
            if "val_auprc" in vals and "val_recall" in vals and "val_precision" in vals
        }
        selection_pool = complete or candidates
        best_model = max(selection_pool, key=lambda mk: selection_pool[mk].get("val_auprc", 0.0))
        best = selection_pool[best_model]
        normalized.setdefault("val_auprc", best.get("val_auprc", 0.0))
        normalized.setdefault("val_recall", best.get("val_recall", 0.0))
        normalized.setdefault("val_precision", best.get("val_precision", 0.0))
        return normalized

    def load_champion_metrics(
        self,
        model_name: str,
        stage: str = "Production",
    ) -> Optional[dict[str, float]]:
        try:
            versions = self.client.get_latest_versions(model_name, stages=[stage])
        except Exception:
            return None
        if not versions:
            return None

        champion_version = versions[0]
        if not getattr(champion_version, "run_id", None):
            return None
        return self._normalize_for_promotion(self.load_run_metrics(champion_version.run_id))

    def promote_model(
        self,
        run_id: str,
        model_name: str,
        stage: str = "Production",
    ) -> dict[str, Any]:
        model_uri = f"runs:/{run_id}/model"
        try:
            mv = mlflow.register_model(model_uri, model_name)
        except mlflow.exceptions.MlflowException:
            versions = self.client.get_latest_versions(model_name)
            mv = versions[0] if versions else None

        if mv:
            self.client.transition_model_version_stage(
                model_name,
                mv.version,
                stage,
                archive_existing_versions=(stage == "Production"),
            )

        return {"model_name": model_name, "version": mv.version if mv else None, "stage": stage}

    def should_promote(
        self,
        challenger_metrics: dict[str, float],
        champion_metrics: Optional[dict[str, float]] = None,
        min_precision_ratio: float = 0.95,
    ) -> tuple[bool, str]:
        if champion_metrics is None:
            return True, "No champion model, promoting as first production model"

        challenger_recall = challenger_metrics.get(
            "val_recall", challenger_metrics.get("recall", 0.0)
        )
        champion_recall = champion_metrics.get("val_recall", champion_metrics.get("recall", 0.0))
        challenger_precision = challenger_metrics.get(
            "val_precision", challenger_metrics.get("precision", 0.0)
        )
        champion_precision = champion_metrics.get(
            "val_precision", champion_metrics.get("precision", 0.0)
        )

        if challenger_recall < champion_recall:
            return (
                False,
                f"Challenger Recall ({challenger_recall:.4f}) < Champion ({champion_recall:.4f})",
            )

        min_precision = champion_precision * min_precision_ratio
        if challenger_precision < min_precision:
            return (
                False,
                f"Challenger Precision ({challenger_precision:.4f}) < "
                f"Min required ({min_precision:.4f})",
            )

        return (
            True,
            f"Recall: {challenger_recall:.4f} >= {champion_recall:.4f} | "
            f"Precision: {challenger_precision:.4f} >= {min_precision:.4f}",
        )


def load_metrics_from_file(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def save_promotion_report(
    path: Path,
    challenger_run_id: str,
    champion_run_id: Optional[str],
    promoted: bool,
    reason: str,
    challenger_metrics: dict[str, Any],
    champion_metrics: Optional[dict[str, Any]] = None,
) -> None:
    report = {
        "challenger_run_id": challenger_run_id,
        "champion_run_id": champion_run_id,
        "promoted": promoted,
        "reason": reason,
        "challenger_metrics": challenger_metrics,
        "champion_metrics": champion_metrics,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
