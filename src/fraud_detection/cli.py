from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import pandas as pd

from fraud_detection.data.schema import (
    INDEX_COLUMN,
    SOURCE_COLUMNS,
    SOURCE_INDEX_COLUMN,
    TARGET_COLUMN,
    build_read_dtypes,
    validate_dataset,
)
from fraud_detection.utils.paths import ensure_dirs, find_project_root

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None

DEFAULT_DATA_CONFIG: dict[str, Any] = {
    "train_csv": "fraudTrain.csv",
    "test_csv": "fraudTest.csv",
    "processed_dir": "data/processed",
    "reports_dir": "reports/metrics",
    "summary_name": "dataset_summary.json",
    "sample_rows": None,
}


def load_data_config() -> dict[str, Any]:
    config_path = find_project_root() / "configs" / "data.yaml"
    data_cfg = copy.deepcopy(DEFAULT_DATA_CONFIG)
    if not config_path.exists() or yaml is None:
        return data_cfg

    with open(config_path, encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    data_cfg.update(payload.get("data", {}))
    return data_cfg


def read_dataset(csv_path: Path, sample_rows: int | None = None) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset does not exist: {csv_path}")

    read_kwargs: dict[str, Any] = {
        "dtype": build_read_dtypes(),
        "usecols": SOURCE_COLUMNS,
    }
    if sample_rows is not None:
        read_kwargs["nrows"] = sample_rows

    frame = pd.read_csv(csv_path, **read_kwargs)
    renamed = frame.rename(columns={SOURCE_INDEX_COLUMN: INDEX_COLUMN})
    return validate_dataset(renamed)


def summarize_dataset(frame: pd.DataFrame) -> dict[str, Any]:
    timestamps = pd.to_datetime(
        frame["trans_date_trans_time"],
        format="%Y-%m-%d %H:%M:%S",
        errors="raise",
    )
    return {
        "rows": int(len(frame)),
        "fraud_rows": int(frame[TARGET_COLUMN].sum()),
        "fraud_rate": float(frame[TARGET_COLUMN].mean()),
        "amount": {
            "min": float(frame["amt"].min()),
            "mean": float(frame["amt"].mean()),
            "max": float(frame["amt"].max()),
        },
        "time_range": [
            timestamps.min().isoformat(),
            timestamps.max().isoformat(),
        ],
        "unique_cards": int(frame["cc_num"].nunique()),
        "unique_merchants": int(frame["merchant"].nunique()),
        "unique_categories": int(frame["category"].nunique()),
    }


def prepare_datasets(sample_rows: int | None = None) -> dict[str, Any]:
    project_root = find_project_root()
    data_cfg = load_data_config()
    effective_sample_rows = sample_rows or data_cfg.get("sample_rows")
    row_limit = int(effective_sample_rows) if effective_sample_rows else None

    processed_dir = project_root / str(data_cfg.get("processed_dir", "data/processed"))
    reports_dir = project_root / str(data_cfg.get("reports_dir", "reports/metrics"))
    summary_name = str(data_cfg.get("summary_name", "dataset_summary.json"))

    ensure_dirs(processed_dir, reports_dir)

    source_paths = {
        "fraudTrain": project_root / str(data_cfg.get("train_csv", "fraudTrain.csv")),
        "fraudTest": project_root / str(data_cfg.get("test_csv", "fraudTest.csv")),
    }

    summary: dict[str, Any] = {
        "target_column": TARGET_COLUMN,
        "sample_rows": row_limit,
        "datasets": {},
    }

    for dataset_name, csv_path in source_paths.items():
        frame = read_dataset(csv_path=csv_path, sample_rows=row_limit)
        output_path = processed_dir / f"{dataset_name}.csv"
        frame.to_csv(output_path, index=False)
        summary["datasets"][dataset_name] = {
            "source_csv": str(csv_path.relative_to(project_root)),
            "processed_csv": str(output_path.relative_to(project_root)),
            **summarize_dataset(frame),
        }

    summary_path = reports_dir / summary_name
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    return summary


def show_paths() -> dict[str, str]:
    project_root = find_project_root()
    data_cfg = load_data_config()
    paths = {
        "train_csv": str((project_root / str(data_cfg["train_csv"])).resolve()),
        "test_csv": str((project_root / str(data_cfg["test_csv"])).resolve()),
        "processed_dir": str((project_root / str(data_cfg["processed_dir"])).resolve()),
        "reports_dir": str((project_root / str(data_cfg["reports_dir"])).resolve()),
    }
    print(json.dumps(paths, indent=2))
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate fraudTrain.csv and fraudTest.csv and write cleaned outputs."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help="Validate fraudTrain.csv and fraudTest.csv and save cleaned CSV outputs",
    )
    prepare.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="Optional row cap for a smaller validation run",
    )

    subparsers.add_parser("paths", help="Show resolved dataset and output paths")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "prepare":
        prepare_datasets(sample_rows=args.sample_rows)
    elif args.command == "paths":
        show_paths()
    else:
        parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
