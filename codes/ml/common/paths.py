from __future__ import annotations

from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path(__file__).resolve()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "dvc.yaml").exists() and (candidate / "configs").exists():
            return candidate
    raise RuntimeError("Could not locate project root from current path.")


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
