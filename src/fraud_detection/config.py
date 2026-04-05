from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from fraud_detection.utils.paths import find_project_root


def load_yaml(relative_path: str, default: Dict[str, Any] | None = None) -> Dict[str, Any]:
    path = find_project_root() / relative_path
    if not path.exists():
        return default.copy() if default is not None else {}
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def save_yaml(relative_path: str, payload: Dict[str, Any]) -> Path:
    path = find_project_root() / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return path

