from __future__ import annotations

import pandas as pd

from fraud_detection.data.features import engineer_all_features, get_feature_names


def test_engineer_all_features_outputs_expected_columns() -> None:
    frame = pd.DataFrame(
        [
            {
                "row_id": 1,
                "trans_date_trans_time": "2019-01-01 00:00:00",
                "cc_num": "1111222233334444",
                "merchant": "fraud_M1",
                "category": "food_dining",
                "amt": 1000.0,
                "first": "A",
                "last": "B",
                "gender": "F",
                "street": "1 Main St",
                "city": "Austin",
                "state": "TX",
                "zip": "78701",
                "lat": 30.2672,
                "long": -97.7431,
                "city_pop": 950000,
                "job": "Engineer",
                "dob": "1990-01-01",
                "trans_num": "0123456789abcdef0123456789abcdef",
                "unix_time": 1546300800,
                "merch_lat": 30.30,
                "merch_long": -97.70,
                "is_fraud": 0,
            },
            {
                "row_id": 2,
                "trans_date_trans_time": "2019-01-01 01:00:00",
                "cc_num": "1111222233334444",
                "merchant": "fraud_M2",
                "category": "shopping_net",
                "amt": 10.0,
                "first": "A",
                "last": "B",
                "gender": "F",
                "street": "1 Main St",
                "city": "Austin",
                "state": "TX",
                "zip": "78701",
                "lat": 30.2672,
                "long": -97.7431,
                "city_pop": 950000,
                "job": "Engineer",
                "dob": "1990-01-01",
                "trans_num": "fedcba9876543210fedcba9876543210",
                "unix_time": 1546304400,
                "merch_lat": 30.31,
                "merch_long": -97.69,
                "is_fraud": 1,
            },
        ]
    )

    transformed = engineer_all_features(frame)
    feature_groups = get_feature_names()

    assert transformed.shape[0] == 2
    for column in feature_groups["temporal"]:
        assert column in transformed.columns
    for column in feature_groups["customer_velocity"]:
        assert column in transformed.columns
    for column in feature_groups["merchant_risk"]:
        assert column in transformed.columns
    assert transformed["customer_txn_count"].iloc[0] >= 1
