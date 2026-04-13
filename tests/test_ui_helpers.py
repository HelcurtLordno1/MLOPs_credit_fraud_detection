from __future__ import annotations

import fraud_detection.ui_helpers as ui_helpers


def test_ui_helpers_module_is_marked_for_day4_handoff() -> None:
    assert "intentionally left empty" in (ui_helpers.__doc__ or "").lower()
