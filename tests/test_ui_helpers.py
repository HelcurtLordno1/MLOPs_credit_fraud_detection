"""Tests for ui_helpers module."""
from __future__ import annotations

from fraud_detection.ui_helpers import (
    default_transaction_payload,
    load_local_status,
    parse_batch_csv,
)


def test_default_transaction_payload_has_required_fields() -> None:
    payload = default_transaction_payload()
    required = [
        "cc_num", "merchant", "category", "amt", "first", "last",
        "gender", "street", "city", "state", "zip", "lat", "long",
        "city_pop", "job", "dob", "trans_num", "unix_time",
        "merch_lat", "merch_long",
    ]
    for field in required:
        assert field in payload, f"Missing field: {field}"


def test_default_payload_types() -> None:
    payload = default_transaction_payload()
    assert isinstance(payload["amt"], (int, float))
    assert isinstance(payload["unix_time"], int)
    assert isinstance(payload["lat"], float)
    assert isinstance(payload["cc_num"], str)


def test_load_local_status_returns_dict() -> None:
    status = load_local_status()
    assert isinstance(status, dict)


def test_parse_batch_csv() -> None:
    csv_content = b"cc_num,amt,category\n1234,10.5,food\n5678,20.0,travel\n"
    df = parse_batch_csv(csv_content)
    assert len(df) == 2
    assert "cc_num" in df.columns
    assert "amt" in df.columns
