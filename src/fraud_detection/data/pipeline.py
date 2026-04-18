"""Data processing pipeline for fraud detection."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from fraud_detection.data.features import engineer_all_features
from fraud_detection.data.schema import TARGET_COLUMN


def create_train_val_test_splits(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets with stratification by fraud label.

    Args:
        df: Input dataframe
        test_size: Proportion for test set (of total data)
        val_size: Proportion for validation set (of remaining after test split)
        random_state: Random seed for reproducibility
        stratify: Whether to stratify by is_fraud column

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    stratify_col = df[TARGET_COLUMN] if stratify else None

    # First split: separate test set
    train_temp, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col,
    )

    # Second split: separate validation from training
    stratify_temp = train_temp[TARGET_COLUMN] if stratify else None
    train_df, val_df = train_test_split(
        train_temp,
        test_size=val_size / (1 - test_size),  # Adjust for size of remaining data
        random_state=random_state,
        stratify=stratify_temp,
    )

    return train_df, val_df, test_df


def prepare_data(
    input_path: Path,
    output_dir: Path,
    test_size: float = 0.2,
    val_size: float = 0.1,
    apply_feature_engineering: bool = True,
) -> dict[str, Path]:
    """
    Load, engineer features, and split data into parquet files.

    Args:
        input_path: Path to input CSV file
        output_dir: Directory to write parquet files
        test_size: Proportion for test set
        val_size: Proportion for validation set
        apply_feature_engineering: Whether to engineer features

    Returns:
        Dictionary mapping split names to output paths
    """
    # Load data
    df = pd.read_csv(input_path)
    print(f"✓ Loaded {len(df):,} rows from {input_path.name}")

    # Apply feature engineering
    if apply_feature_engineering:
        print("→ Applying feature engineering...")
        df = engineer_all_features(df)
        print(f"✓ Created {df.shape[1] - len(pd.read_csv(input_path).columns)} new features")

    # Create splits
    print(f"→ Creating train/val/test splits (test={test_size}, val={val_size})...")
    train_df, val_df, test_df = create_train_val_test_splits(
        df, test_size=test_size, val_size=val_size
    )

    # Write outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        output_path = output_dir / f"{split_name}.parquet"
        split_df.to_parquet(output_path, index=False)
        outputs[split_name] = output_path

        fraud_count = split_df[TARGET_COLUMN].sum()
        fraud_pct = (fraud_count / len(split_df)) * 100
        print(
            f"✓ {split_name.capitalize():5s}: {len(split_df):>8,} rows, "
            f"fraud: {fraud_count:>7,} ({fraud_pct:.2f}%)"
        )

    return outputs


def load_splits(
    processed_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load train, val, test splits from directory.

    Args:
        processed_dir: Directory containing parquet files

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    train_df = pd.read_parquet(processed_dir / "train.parquet")
    val_df = pd.read_parquet(processed_dir / "val.parquet")
    test_df = pd.read_parquet(processed_dir / "test.parquet")

    return train_df, val_df, test_df
