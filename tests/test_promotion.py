from __future__ import annotations

import json
from pathlib import Path

from fraud_detection.monitoring.promotion import ModelPromoter, save_promotion_report


def test_should_promote_rule_matches_day3_requirement() -> None:
    challenger = {"recall": 0.90, "precision": 0.40}
    champion = {"recall": 0.88, "precision": 0.42}

    promoter = ModelPromoter()
    promote, _ = promoter.should_promote(challenger, champion, min_precision_ratio=0.95)

    assert promote is True


def test_save_promotion_report_writes_json(tmp_path: Path) -> None:
    report_path = tmp_path / "models" / "registry" / "last_promotion.json"
    save_promotion_report(
        path=report_path,
        challenger_run_id="run-1",
        champion_run_id="run-0",
        promoted=True,
        reason="rule matched",
        challenger_metrics={"recall": 0.9, "precision": 0.4},
        champion_metrics={"recall": 0.88, "precision": 0.42},
    )

    with open(report_path, encoding="utf-8") as handle:
        payload = json.load(handle)

    assert payload["promoted"] is True
    assert payload["challenger_run_id"] == "run-1"
