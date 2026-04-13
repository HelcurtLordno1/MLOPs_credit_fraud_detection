from __future__ import annotations

import fraud_detection.api.main as api_main
import fraud_detection.api.service as api_service


def test_api_modules_are_marked_for_day4_handoff() -> None:
    assert "intentionally left empty" in (api_main.__doc__ or "").lower()
    assert "intentionally left empty" in (api_service.__doc__ or "").lower()
