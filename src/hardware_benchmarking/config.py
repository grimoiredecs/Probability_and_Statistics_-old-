"""Configuration loading for the active benchmarking pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DomainSettings:
    domain_key: str
    raw_filepath: str
    target_col: str
    min_r2: float
    max_rmse: float


@dataclass(frozen=True)
class Settings:
    domains: dict[str, DomainSettings]
    directories: dict[str, str]
    strategies: list[str]
    evaluation: dict[str, Any]
    objectives: dict[str, Any]


def load_settings(path: str | Path = "config/settings.yaml") -> Settings:
    with Path(path).open(encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}
    domains = {
        name: DomainSettings(domain_key=value["domain_key"], raw_filepath=value["raw_filepath"],
                            target_col=value["target_col"], min_r2=float(value["min_r2"]),
                            max_rmse=float(value["max_rmse"]))
        for name, value in raw["pipeline"].items()
    }
    return Settings(
        domains=domains,
        directories=raw["directories"],
        strategies=raw["strategies"],
        evaluation=raw.get("evaluation", {"cv_folds": 5, "random_state": 42, "selection_metric": "CV_R2_Mean"}),
        objectives=raw.get("objectives", {}),
    )
