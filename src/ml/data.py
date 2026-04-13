"""
Day 2 — Member 2: The Model Trainer (taking over Feature Engineering)
==========================================
Data preparation pipeline: raw CSV → feature-engineered parquet splits.

Features engineered:
  Temporal   : hour_of_day, day_of_week, is_weekend
  Customer   : customer_avg_amt, customer_txn_count
  Merchant   : merchant_fraud_rate
  Frequency  : category_freq, city_freq, merchant_freq, job_freq,
               state_freq, gender_freq

Output splits (in data/processed/):
  train.parquet  — 70% of fraudTrain.csv (after feature engineering)
  val.parquet    — 15%
  test.parquet   — 15% + full fraudTest.csv rows

Usage:
    python -m src.ml.data       # direct
    dvc repro prepare           # via DVC pipeline
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# ─── Paths ───────────────────────────────────────────────────────────────────
DATA_DIR      = Path("data")
PROCESSED_DIR = Path("data/processed")


# ─── Feature Engineering ─────────────────────────────────────────────────────

def _extract_event_ts(df: pd.DataFrame) -> pd.Series:
    if "trans_date_trans_time" in df.columns:
        ts = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")
    elif "unix_time" in df.columns:
        ts = pd.to_datetime(df["unix_time"], unit="s", errors="coerce")
    else:
        ts = pd.Series(pd.NaT, index=df.index)
    return ts.fillna(pd.Timestamp("1970-01-01"))


def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ts = _extract_event_ts(out)
    out["hour_of_day"] = ts.dt.hour.astype(int)
    out["day_of_week"] = ts.dt.dayofweek.astype(int)
    out["is_weekend"] = out["day_of_week"].isin([5, 6]).astype(int)
    out["event_ts"] = ts
    return out


def _add_customer_24h_count(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "cc_num" not in out.columns:
        out["customer_txn_count_last_24h"] = 1.0
        return out

    order = out.sort_values(["cc_num", "event_ts"]).index
    ordered = out.loc[order, ["cc_num", "event_ts"]].copy()
    ordered["txn"] = 1.0
    rolling = (
        ordered.groupby("cc_num")
        .rolling("24h", on="event_ts")["txn"]
        .sum()
        .reset_index(level=0, drop=True)
    )
    out.loc[order, "customer_txn_count_last_24h"] = rolling.values
    return out


def _build_frequency_maps(train_df: pd.DataFrame, cols: list[str]) -> dict[str, pd.Series]:
    maps: dict[str, pd.Series] = {}
    n = max(len(train_df), 1)
    for col in cols:
        if col in train_df.columns:
            maps[col] = train_df[col].value_counts(dropna=False) / n
    return maps


def _transform_with_maps(
    df: pd.DataFrame,
    *,
    freq_maps: dict[str, pd.Series],
    customer_avg_map: pd.Series,
    customer_count_map: pd.Series,
    merchant_risk_map: pd.Series,
    global_avg_amt: float,
    global_txn_count: float,
    global_merchant_risk: float,
    is_train: bool,
) -> pd.DataFrame:
    out = _add_temporal_features(df)
    out = _add_customer_24h_count(out)

    if "cc_num" in out.columns:
        out["customer_avg_amt"] = out["cc_num"].map(customer_avg_map).fillna(global_avg_amt)
        out["customer_txn_count"] = out["cc_num"].map(customer_count_map).fillna(global_txn_count)
    else:
        out["customer_avg_amt"] = global_avg_amt
        out["customer_txn_count"] = global_txn_count

    if "merchant" in out.columns:
        if is_train and "is_fraud" in out.columns:
            merchant_sum = out.groupby("merchant")["is_fraud"].transform("sum")
            merchant_count = out.groupby("merchant")["is_fraud"].transform("count")
            loo_rate = (merchant_sum - out["is_fraud"]) / (merchant_count - 1)
            out["merchant_fraud_rate"] = loo_rate.where(merchant_count > 1, global_merchant_risk)
        else:
            out["merchant_fraud_rate"] = out["merchant"].map(merchant_risk_map).fillna(global_merchant_risk)
    else:
        out["merchant_fraud_rate"] = global_merchant_risk

    for col, fmap in freq_maps.items():
        if col in out.columns:
            out[f"{col}_freq"] = out[col].map(fmap).fillna(0.0)

    out = out.drop(columns=["event_ts"], errors="ignore")
    out = out.select_dtypes(include=["number"])
    return out


# ─── Main Preparation Function ────────────────────────────────────────────────

def prepare_data() -> None:
    """
    Full Day-1 data preparation pipeline:

    1. Load fraudTrain.csv (for train/val split) and fraudTest.csv (for test).
    2. Validate that is_fraud column exists.
    3. Apply feature engineering via add_features().
    4. Split train into train (70%) and val (15%) subsets.
       fraudTest.csv becomes the test split.
    5. Save splits as parquet files in data/processed/.

    Raises:
        FileNotFoundError: If raw CSV files are not found in data/.
        ValueError: If is_fraud column is missing from the dataset.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load raw data ─────────────────────────────────────────────────────────
    train_csv = DATA_DIR / "fraudTrain.csv"
    test_csv  = DATA_DIR / "fraudTest.csv"

    df_train_raw = pd.read_csv(train_csv)
    df_test_raw  = pd.read_csv(test_csv)

    # ── Validate target column ────────────────────────────────────────────────
    if "is_fraud" not in df_train_raw.columns:
        raise ValueError("Missing required target column: is_fraud")

    # Split raw data first so all maps are fitted on train-only data.
    df_train_raw_split, df_val_raw_split = train_test_split(
        df_train_raw,
        test_size=0.15,
        random_state=42,
        stratify=df_train_raw["is_fraud"],
    )

    freq_cols = ["category", "city", "merchant", "job", "state", "gender"]
    freq_maps = _build_frequency_maps(df_train_raw_split, freq_cols)

    if "cc_num" in df_train_raw_split.columns and "amt" in df_train_raw_split.columns:
        customer_avg_map = df_train_raw_split.groupby("cc_num")["amt"].mean()
        customer_count_map = df_train_raw_split.groupby("cc_num")["amt"].count().astype(float)
        global_avg_amt = float(df_train_raw_split["amt"].mean())
    else:
        customer_avg_map = pd.Series(dtype=float)
        customer_count_map = pd.Series(dtype=float)
        global_avg_amt = 0.0

    global_txn_count = 1.0
    if "merchant" in df_train_raw_split.columns:
        merchant_risk_map = df_train_raw_split.groupby("merchant")["is_fraud"].mean()
    else:
        merchant_risk_map = pd.Series(dtype=float)

    global_merchant_risk = float(df_train_raw_split["is_fraud"].mean())

    # ── Feature engineering with train-fitted mappings ───────────────────────
    train_df = _transform_with_maps(
        df_train_raw_split,
        freq_maps=freq_maps,
        customer_avg_map=customer_avg_map,
        customer_count_map=customer_count_map,
        merchant_risk_map=merchant_risk_map,
        global_avg_amt=global_avg_amt,
        global_txn_count=global_txn_count,
        global_merchant_risk=global_merchant_risk,
        is_train=True,
    )

    val_df = _transform_with_maps(
        df_val_raw_split,
        freq_maps=freq_maps,
        customer_avg_map=customer_avg_map,
        customer_count_map=customer_count_map,
        merchant_risk_map=merchant_risk_map,
        global_avg_amt=global_avg_amt,
        global_txn_count=global_txn_count,
        global_merchant_risk=global_merchant_risk,
        is_train=False,
    )

    df_test_fe = _transform_with_maps(
        df_test_raw,
        freq_maps=freq_maps,
        customer_avg_map=customer_avg_map,
        customer_count_map=customer_count_map,
        merchant_risk_map=merchant_risk_map,
        global_avg_amt=global_avg_amt,
        global_txn_count=global_txn_count,
        global_merchant_risk=global_merchant_risk,
        is_train=False,
    )

    # ── Save parquet splits ───────────────────────────────────────────────────
    train_df.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
    val_df.to_parquet(PROCESSED_DIR   / "val.parquet",   index=False)
    df_test_fe.to_parquet(PROCESSED_DIR / "test.parquet", index=False)

    print("Prepared datasets:")
    print(f"  train={len(train_df)}")
    print(f"  val  ={len(val_df)}")
    print(f"  test ={len(df_test_fe)}")


if __name__ == "__main__":
    prepare_data()
