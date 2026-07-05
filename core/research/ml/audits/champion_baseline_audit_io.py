from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml


def _meta_output_dir(config: dict[str, Any]) -> Path:
    ml_config = config.get("ml", {})
    return Path(
        ml_config.get(
            "output_dir",
            Path(config.get("reports", {}).get("ml_dir", "reports/ml"))
            / "regime_transformer_meta_ensemble_v1",
        )
    )


def _expanded_dataset_path(config: dict[str, Any]) -> Path:
    ml_config = config.get("ml", {})
    return Path(
        ml_config.get(
            "expanded_rebalance_dataset_path",
            Path(config.get("cache", {}).get("ml_dir", "cache/ml"))
            / "expanded_rebalance_dataset.csv",
        )
    )


def _meta_dataset_path(config: dict[str, Any]) -> Path:
    ml_config = config.get("ml", {})
    return Path(
        ml_config.get(
            "meta_dataset_path",
            Path(config.get("cache", {}).get("ml_dir", "cache/ml"))
            / "meta_ensemble_dataset.csv",
        )
    )


def _champion_config_path(config: dict[str, Any]) -> Path:
    return Path(
        str(
            config.get("research", {})
            .get("dual_momentum", {})
            .get(
                "champion_config_path",
                "configs/champions/ranked_top5_monthly_exposure90_v1.yaml",
            )
        )
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return payload if isinstance(payload, dict) else {}
