from __future__ import annotations

from typing import Any

import pandas as pd

SOURCE_INDEX_COLUMN = "Unnamed: 0"
INDEX_COLUMN = "row_id"
TARGET_COLUMN = "is_fraud"

SOURCE_COLUMNS = [
    SOURCE_INDEX_COLUMN,
    "trans_date_trans_time",
    "cc_num",
    "merchant",
    "category",
    "amt",
    "first",
    "last",
    "gender",
    "street",
    "city",
    "state",
    "zip",
    "lat",
    "long",
    "city_pop",
    "job",
    "dob",
    "trans_num",
    "unix_time",
    "merch_lat",
    "merch_long",
    TARGET_COLUMN,
]

DATASET_COLUMNS = [
    INDEX_COLUMN,
    "trans_date_trans_time",
    "cc_num",
    "merchant",
    "category",
    "amt",
    "first",
    "last",
    "gender",
    "street",
    "city",
    "state",
    "zip",
    "lat",
    "long",
    "city_pop",
    "job",
    "dob",
    "trans_num",
    "unix_time",
    "merch_lat",
    "merch_long",
    TARGET_COLUMN,
]


def build_read_dtypes() -> dict[str, Any]:
    return {
        SOURCE_INDEX_COLUMN: "int64",
        "trans_date_trans_time": "string",
        "cc_num": "string",
        "merchant": "string",
        "category": "string",
        "amt": "float64",
        "first": "string",
        "last": "string",
        "gender": "string",
        "street": "string",
        "city": "string",
        "state": "string",
        "zip": "string",
        "lat": "float64",
        "long": "float64",
        "city_pop": "int64",
        "job": "string",
        "dob": "string",
        "trans_num": "string",
        "unix_time": "int64",
        "merch_lat": "float64",
        "merch_long": "float64",
        TARGET_COLUMN: "int8",
    }


def _validate_non_empty_strings(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        invalid = frame[column].astype("string").str.fullmatch(r".+", na=False)
        if not bool(invalid.all()):
            raise ValueError(f"Column '{column}' contains empty values")


def _validate_pattern(frame: pd.DataFrame, column: str, pattern: str) -> None:
    matches = frame[column].astype("string").str.fullmatch(pattern, na=False)
    if not bool(matches.all()):
        raise ValueError(f"Column '{column}' contains invalid values")


def _validate_range(frame: pd.DataFrame, column: str, minimum: float, maximum: float) -> None:
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any() or bool(((values < minimum) | (values > maximum)).any()):
        raise ValueError(f"Column '{column}' is outside the allowed range")


def validate_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    actual_columns = list(frame.columns)
    if actual_columns != DATASET_COLUMNS:
        raise ValueError(
            f"Dataset columns do not match the expected schema. "
            f"Expected {DATASET_COLUMNS}, got {actual_columns}."
        )

    _validate_non_empty_strings(
        frame,
        ["merchant", "category", "first", "last", "street", "city", "job"],
    )
    _validate_pattern(frame, "trans_date_trans_time", r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")
    _validate_pattern(frame, "cc_num", r"\d+")
    _validate_pattern(frame, "state", r"[A-Z]{2}")
    _validate_pattern(frame, "zip", r"\d{4,5}")
    _validate_pattern(frame, "dob", r"\d{4}-\d{2}-\d{2}")
    _validate_pattern(frame, "trans_num", r"[0-9a-f]{32}")

    if not bool(frame[INDEX_COLUMN].ge(0).all()):
        raise ValueError(f"Column '{INDEX_COLUMN}' must be non-negative")
    if not bool(frame["gender"].isin(["F", "M"]).all()):
        raise ValueError("Column 'gender' must contain only 'F' or 'M'")
    if not bool(frame["city_pop"].ge(0).all()):
        raise ValueError("Column 'city_pop' must be non-negative")
    if not bool(frame["unix_time"].ge(0).all()):
        raise ValueError("Column 'unix_time' must be non-negative")
    if not bool(frame["amt"].ge(0).all()):
        raise ValueError("Column 'amt' must be non-negative")
    if not bool(frame[TARGET_COLUMN].isin([0, 1]).all()):
        raise ValueError(f"Column '{TARGET_COLUMN}' must contain only 0 or 1")

    _validate_range(frame, "lat", -90.0, 90.0)
    _validate_range(frame, "merch_lat", -90.0, 90.0)
    _validate_range(frame, "long", -180.0, 180.0)
    _validate_range(frame, "merch_long", -180.0, 180.0)

    pd.to_datetime(frame["trans_date_trans_time"], format="%Y-%m-%d %H:%M:%S", errors="raise")
    pd.to_datetime(frame["dob"], format="%Y-%m-%d", errors="raise")
    return frame
