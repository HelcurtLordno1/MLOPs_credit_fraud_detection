from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from codes.ml.common.paths import ensure_dirs, find_project_root


SEED_DEFAULT = 42


def _load_base_config(project_root: Path) -> dict:
    cfg_path = project_root / "configs" / "base.yaml"
    if not cfg_path.exists():
        return {
            "seed": SEED_DEFAULT,
            "data": {
                "raw_csv": "data/raw/creditcard.csv",
                "processed_dir": "data/processed",
                "test_size": 0.2,
                "val_size_from_train": 0.25,
                "reference_sample_size": 10000,
                "current_sample_size": 10000,
            },
            "target": "Class",
        }

    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _file_md5(file_path: Path) -> str:
    digest = hashlib.md5()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _validate_schema(df: pd.DataFrame, target_col: str) -> None:
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")
    if set(df[target_col].unique()) - {0, 1}:
        raise ValueError("Target column contains labels outside {0, 1}.")


def _clean_data(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, dict]:
    initial_rows = len(df)
    duplicate_rows = int(df.duplicated().sum())
    null_total = int(df.isnull().sum().sum())

    if duplicate_rows > 0:
        df = df.drop_duplicates().reset_index(drop=True)

    numeric_cols = [col for col in df.columns if col != target_col and pd.api.types.is_numeric_dtype(df[col])]
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # In case target has nulls from bad imports, drop those rows.
    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    report = {
        "initial_rows": int(initial_rows),
        "rows_after_cleaning": int(len(df)),
        "duplicate_rows_removed": int(duplicate_rows),
        "null_values_before": int(null_total),
        "null_values_after": int(df.isnull().sum().sum()),
    }
    return df, report


def prepare_data() -> None:
    project_root = find_project_root()
    cfg = _load_base_config(project_root)

    seed = int(cfg.get("seed", SEED_DEFAULT))
    target_col = str(cfg.get("target", "Class"))
    data_cfg = cfg.get("data", {})

    raw_csv = project_root / str(data_cfg.get("raw_csv", "data/raw/creditcard.csv"))
    processed_dir = project_root / str(data_cfg.get("processed_dir", "data/processed"))
    reports_dir = project_root / "reports"

    ensure_dirs(processed_dir, reports_dir)

    if not raw_csv.exists():
        raise FileNotFoundError(f"Raw dataset not found: {raw_csv}")

    df = pd.read_csv(raw_csv)
    _validate_schema(df, target_col)
    df, clean_report = _clean_data(df, target_col)

    train_val, test = train_test_split(
        df,
        test_size=float(data_cfg.get("test_size", 0.2)),
        random_state=seed,
        stratify=df[target_col],
    )

    train, val = train_test_split(
        train_val,
        test_size=float(data_cfg.get("val_size_from_train", 0.25)),
        random_state=seed,
        stratify=train_val[target_col],
    )

    reference_size = min(len(train), int(data_cfg.get("reference_sample_size", 10000)))
    current_size = min(len(test), int(data_cfg.get("current_sample_size", 10000)))

    reference = train.sample(reference_size, random_state=seed)
    current = test.sample(current_size, random_state=seed)

    train.to_parquet(processed_dir / "train.parquet", index=False)
    val.to_parquet(processed_dir / "val.parquet", index=False)
    test.to_parquet(processed_dir / "test.parquet", index=False)
    reference.to_parquet(processed_dir / "reference.parquet", index=False)
    current.to_parquet(processed_dir / "current.parquet", index=False)

    data_quality = {
        "seed": seed,
        "target_col": target_col,
        "fraud_ratio": {
            "full": float(df[target_col].mean()),
            "train": float(train[target_col].mean()),
            "val": float(val[target_col].mean()),
            "test": float(test[target_col].mean()),
        },
        "row_counts": {
            "full": int(len(df)),
            "train": int(len(train)),
            "val": int(len(val)),
            "test": int(len(test)),
            "reference": int(len(reference)),
            "current": int(len(current)),
        },
        "cleaning": clean_report,
        "columns": list(df.columns),
    }

    fingerprint = {
        "raw_csv_path": str(raw_csv),
        "raw_csv_md5": _file_md5(raw_csv),
    }

    with open(reports_dir / "data_quality_report.json", "w", encoding="utf-8") as f:
        json.dump(data_quality, f, indent=2)

    with open(reports_dir / "dataset_fingerprint.json", "w", encoding="utf-8") as f:
        json.dump(fingerprint, f, indent=2)

    print("Data preparation complete")
    print(json.dumps(data_quality["row_counts"], indent=2))


def main() -> None:
    prepare_data()


if __name__ == "__main__":
    main()
