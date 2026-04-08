"""Data quality and pipeline validation tests.

These tests verify that the data preparation pipeline produces valid outputs
and that the processed data meets quality expectations.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _skip_if_missing(path: Path) -> pd.DataFrame:
    """Load parquet or skip test if file is missing."""
    if not path.exists():
        pytest.skip(f"File not found: {path.name}. Run data prep first.")
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Data existence tests
# ---------------------------------------------------------------------------


class TestDataFilesExist:
    """Verify all expected processed data files are present."""

    @pytest.mark.parametrize(
        "filename",
        [
            "train.parquet",
            "val.parquet",
            "test.parquet",
            "reference.parquet",
            "current.parquet",
        ],
    )
    def test_processed_file_exists(self, filename):
        path = PROCESSED_DIR / filename
        if not path.exists():
            pytest.skip(f"{filename} not found. Run `dvc repro` first.")
        assert path.stat().st_size > 0, f"{filename} is empty"


# ---------------------------------------------------------------------------
# Schema validation tests
# ---------------------------------------------------------------------------


class TestDataSchema:
    """Verify schema consistency across all data splits."""

    def test_target_column_in_train(self):
        df = _skip_if_missing(PROCESSED_DIR / "train.parquet")
        assert "Class" in df.columns

    def test_target_column_in_test(self):
        df = _skip_if_missing(PROCESSED_DIR / "test.parquet")
        assert "Class" in df.columns

    def test_target_column_in_val(self):
        df = _skip_if_missing(PROCESSED_DIR / "val.parquet")
        assert "Class" in df.columns

    def test_target_values_binary(self):
        """Target column should only contain 0 and 1."""
        df = _skip_if_missing(PROCESSED_DIR / "train.parquet")
        assert set(df["Class"].unique()).issubset({0, 1})

    def test_columns_consistent_across_splits(self):
        """All splits should have the same columns."""
        train = _skip_if_missing(PROCESSED_DIR / "train.parquet")
        test = _skip_if_missing(PROCESSED_DIR / "test.parquet")
        val = _skip_if_missing(PROCESSED_DIR / "val.parquet")

        assert list(train.columns) == list(test.columns)
        assert list(train.columns) == list(val.columns)

    def test_expected_feature_columns_present(self):
        """V1-V28 and Amount columns should be present."""
        df = _skip_if_missing(PROCESSED_DIR / "train.parquet")
        expected = [f"V{i}" for i in range(1, 29)] + ["Amount"]
        for col in expected:
            assert col in df.columns, f"Missing expected column: {col}"


# ---------------------------------------------------------------------------
# Data quality tests
# ---------------------------------------------------------------------------


class TestDataQuality:
    """Verify data quality expectations."""

    def test_no_null_values_in_features(self):
        """Feature columns should have no null values after cleaning."""
        df = _skip_if_missing(PROCESSED_DIR / "train.parquet")
        feature_cols = [c for c in df.columns if c != "Class"]
        null_counts = df[feature_cols].isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]
        assert len(cols_with_nulls) == 0, f"Columns with nulls: {cols_with_nulls.to_dict()}"

    def test_no_null_target(self):
        """Target column should have no null values."""
        df = _skip_if_missing(PROCESSED_DIR / "train.parquet")
        assert df["Class"].isnull().sum() == 0

    def test_fraud_present_in_train(self):
        """Training set should contain fraud examples."""
        df = _skip_if_missing(PROCESSED_DIR / "train.parquet")
        assert df["Class"].sum() > 0, "No fraud cases in training data"

    def test_fraud_present_in_test(self):
        """Test set should contain fraud examples."""
        df = _skip_if_missing(PROCESSED_DIR / "test.parquet")
        assert df["Class"].sum() > 0, "No fraud cases in test data"

    def test_train_larger_than_test(self):
        """Training set should be larger than test set."""
        train = _skip_if_missing(PROCESSED_DIR / "train.parquet")
        test = _skip_if_missing(PROCESSED_DIR / "test.parquet")
        assert len(train) > len(test)

    def test_reference_data_has_rows(self):
        """Reference data for drift detection should not be empty."""
        df = _skip_if_missing(PROCESSED_DIR / "reference.parquet")
        assert len(df) > 0


# ---------------------------------------------------------------------------
# Data quality report tests
# ---------------------------------------------------------------------------


class TestDataQualityReport:
    """Verify the data quality report artifact."""

    def test_report_exists(self):
        report_path = PROJECT_ROOT / "reports" / "data_quality_report.json"
        if not report_path.exists():
            pytest.skip("Data quality report not found. Run `dvc repro` first.")
        assert report_path.stat().st_size > 0

    def test_fingerprint_exists(self):
        fp_path = PROJECT_ROOT / "reports" / "dataset_fingerprint.json"
        if not fp_path.exists():
            pytest.skip("Dataset fingerprint not found. Run `dvc repro` first.")
        assert fp_path.stat().st_size > 0
