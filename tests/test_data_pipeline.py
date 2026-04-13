from __future__ import annotations

import pandas as pd

from fraud_detection.data.pipeline import create_train_val_test_splits


def test_create_train_val_test_splits_stratified() -> None:
    rows = []
    for idx in range(300):
        rows.append(
            {
                "feature": float(idx),
                "is_fraud": 1 if idx % 20 == 0 else 0,
            }
        )
    frame = pd.DataFrame(rows)

    train_df, val_df, test_df = create_train_val_test_splits(frame, stratify=True)

    assert len(train_df) + len(val_df) + len(test_df) == len(frame)
    assert train_df["is_fraud"].sum() > 0
    assert val_df["is_fraud"].sum() > 0
    assert test_df["is_fraud"].sum() > 0
