"""Feature engineering for fraud detection."""

from __future__ import annotations

import numpy as np
import pandas as pd


def engineer_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create temporal features from transaction timestamp.

    Features:
    - hour_of_day: Hour of transaction
    - day_of_week: Day of week (0=Monday, 6=Sunday)
    - is_weekend: Binary flag for weekend transactions
    - month: Month of transaction
    - day_of_month: Day of month
    """
    df = df.copy()

    # Parse transaction datetime
    ts = pd.to_datetime(df["trans_date_trans_time"], format="%Y-%m-%d %H:%M:%S")

    df["hour_of_day"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    df["month"] = ts.dt.month
    df["day_of_month"] = ts.dt.day

    return df


def engineer_customer_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create customer velocity features based on credit card.

    Features:
    - customer_txn_count: Total transactions per customer
    - customer_avg_amt: Average transaction amount per customer
    - customer_std_amt: Standard deviation of transaction amount
    - customer_fraud_rate: Historical fraud rate for customer
    """
    df = df.copy()

    # Group by credit card number
    cc_stats = (
        df.groupby("cc_num")
        .agg(
            {
                "cc_num": "count",
                "amt": ["mean", "std", "sum"],
                "is_fraud": ["sum", "mean"],
            }
        )
        .reset_index()
    )

    cc_stats.columns = [
        "cc_num",
        "customer_txn_count",
        "customer_avg_amt",
        "customer_std_amt",
        "customer_total_amt",
        "customer_fraud_count",
        "customer_fraud_rate",
    ]

    cc_stats["customer_fraud_rate"] = cc_stats["customer_fraud_rate"].fillna(0)
    cc_stats["customer_std_amt"] = cc_stats["customer_std_amt"].fillna(0)

    # Merge back to original dataframe
    df = df.merge(
        cc_stats[
            [
                "cc_num",
                "customer_txn_count",
                "customer_avg_amt",
                "customer_std_amt",
                "customer_fraud_rate",
            ]
        ],
        on="cc_num",
        how="left",
    )

    return df


def engineer_merchant_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create merchant risk features.

    Features:
    - merchant_fraud_rate: Historical fraud rate at merchant
    - merchant_txn_count: Total transactions at merchant
    - merchant_avg_amt: Average transaction amount at merchant
    - merchant_std_amt: Standard deviation of transaction amount
    """
    df = df.copy()

    # Group by merchant
    merchant_stats = (
        df.groupby("merchant")
        .agg(
            {
                "merchant": "count",
                "amt": ["mean", "std"],
                "is_fraud": ["sum", "mean"],
            }
        )
        .reset_index()
    )

    merchant_stats.columns = [
        "merchant",
        "merchant_txn_count",
        "merchant_avg_amt",
        "merchant_std_amt",
        "merchant_fraud_count",
        "merchant_fraud_rate",
    ]

    merchant_stats["merchant_fraud_rate"] = merchant_stats["merchant_fraud_rate"].fillna(0)
    merchant_stats["merchant_std_amt"] = merchant_stats["merchant_std_amt"].fillna(0)

    # Merge back to original dataframe
    df = df.merge(
        merchant_stats[
            [
                "merchant",
                "merchant_fraud_rate",
                "merchant_txn_count",
                "merchant_avg_amt",
                "merchant_std_amt",
            ]
        ],
        on="merchant",
        how="left",
    )

    return df


def engineer_category_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create category-based features.

    Features:
    - category_fraud_rate: Historical fraud rate for transaction category
    - category_txn_count: Total transactions in category
    - category_avg_amt: Average transaction amount in category
    """
    df = df.copy()

    # Group by category
    cat_stats = (
        df.groupby("category")
        .agg(
            {
                "category": "count",
                "amt": "mean",
                "is_fraud": ["sum", "mean"],
            }
        )
        .reset_index()
    )

    cat_stats.columns = [
        "category",
        "category_txn_count",
        "category_avg_amt",
        "category_fraud_count",
        "category_fraud_rate",
    ]

    cat_stats["category_fraud_rate"] = cat_stats["category_fraud_rate"].fillna(0)

    # Merge back to original dataframe
    df = df.merge(
        cat_stats[["category", "category_fraud_rate", "category_txn_count", "category_avg_amt"]],
        on="category",
        how="left",
    )

    return df


def engineer_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate distance between customer home and merchant location.

    Uses Haversine formula for great-circle distance.
    """
    df = df.copy()

    def haversine(lat1, lon1, lat2, lon2):
        """Calculate great circle distance in miles."""
        R = 3959  # Earth radius in miles

        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))

        return R * c

    df["distance_km"] = (
        haversine(
            df["lat"].values, df["long"].values, df["merch_lat"].values, df["merch_long"].values
        )
        * 1.60934
    )  # Convert miles to km

    # Distance can be zero for online transactions - flag them
    df["is_distant_txn"] = (df["distance_km"] > 100).astype(int)

    return df


def engineer_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create amount-based anomaly features.

    Features:
    - amt_zscore: Z-score of transaction amount relative to customer average
    - amt_is_outlier: Flag for unusual transaction amount
    - amt_ratio_to_customer_avg: Ratio of transaction amount to customer average
    """
    df = df.copy()

    # Z-score relative to customer average
    df["amt_zscore"] = (df["amt"] - df["customer_avg_amt"]) / (df["customer_std_amt"] + 1)
    df["amt_zscore"] = df["amt_zscore"].fillna(0)

    # Flag outliers (z-score > 3 or < -3)
    df["amt_is_outlier"] = (df["amt_zscore"].abs() > 3).astype(int)

    # Ratio to customer average
    df["amt_ratio_to_customer_avg"] = (df["amt"] / (df["customer_avg_amt"] + 1)).clip(upper=100)

    return df


def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering transformations.

    Args:
        df: Input dataframe with raw features

    Returns:
        DataFrame with engineered features
    """
    # Apply features in order
    df = engineer_temporal_features(df)
    df = engineer_customer_velocity_features(df)
    df = engineer_merchant_risk_features(df)
    df = engineer_category_features(df)
    df = engineer_distance_features(df)
    df = engineer_amount_features(df)

    return df


def get_feature_names() -> dict[str, list[str]]:
    """
    Get categorized feature names for reference.

    Returns:
        Dictionary mapping feature categories to feature names
    """
    return {
        "temporal": ["hour_of_day", "day_of_week", "is_weekend", "month", "day_of_month"],
        "customer_velocity": [
            "customer_txn_count",
            "customer_avg_amt",
            "customer_std_amt",
            "customer_fraud_rate",
        ],
        "merchant_risk": [
            "merchant_fraud_rate",
            "merchant_txn_count",
            "merchant_avg_amt",
            "merchant_std_amt",
        ],
        "category": ["category_fraud_rate", "category_txn_count", "category_avg_amt"],
        "distance": ["distance_km", "is_distant_txn"],
        "amount": ["amt_zscore", "amt_is_outlier", "amt_ratio_to_customer_avg"],
        "raw_numeric": ["amt", "lat", "long", "city_pop", "merch_lat", "merch_long", "unix_time"],
    }
