from __future__ import annotations

from pathlib import Path
import tomllib
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_toml(path: str | Path) -> dict[str, Any]:
    """Load a TOML file, resolving relative paths from the project root."""
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    with candidate.open("rb") as handle:
        return tomllib.load(handle)


def load_study_config(path: str | Path = "configs/study.toml") -> dict[str, Any]:
    return load_toml(path)


def project_path(value: str | Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate

