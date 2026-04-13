from __future__ import annotations

import pandas as pd

from fraud_detection.monitoring.drift import (
    detect_feature_drift,
    detect_target_drift,
    summarize_drift,
)


def test_drift_functions_return_expected_shapes() -> None:
    reference = pd.DataFrame(
        {
            "amt": [10.0, 12.0, 14.0, 16.0],
            "city_pop": [100, 200, 300, 400],
            "is_fraud": [0, 0, 1, 0],
        }
    )
    current = pd.DataFrame(
        {
            "amt": [11.0, 13.0, 15.0, 17.0],
            "city_pop": [110, 210, 310, 410],
            "is_fraud": [0, 1, 0, 0],
        }
    )

    feature_results = detect_feature_drift(reference, current)
    target_result = detect_target_drift(reference["is_fraud"], current["is_fraud"])
    summary = summarize_drift(feature_results)

    assert "amt" in feature_results
    assert "city_pop" in feature_results
    assert "overall_drift_detected" in summary
    assert "p_value" in target_result
