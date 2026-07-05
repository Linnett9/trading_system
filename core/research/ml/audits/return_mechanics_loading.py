from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _artifact_paths(config: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    ml_config = config.get("ml", {})
    report_dir = Path(config.get("reports", {}).get("ml_dir", output_dir.parent))
    cache_dir = Path(config.get("cache", {}).get("ml_dir", "cache/ml"))
    return {
        "trading_research_leaderboard": output_dir / "trading_research_leaderboard.md",
        "trading_research_leaderboard_csv": output_dir / "trading_research_leaderboard.csv",
        "allocation_comparison": output_dir / "allocation_policy_comparison.json",
        "allocation_leaderboard": output_dir / "allocation_policy_leaderboard.md",
        "allocation_diagnostics": output_dir / "allocation_policy_diagnostics.json",
        "grid_search": output_dir / "allocation_policy_grid_search.json",
        "optimizer_results": output_dir / "allocation_optimizer_results.json",
        "optimizer_report": output_dir / "allocation_optimizer_report.md",
        "selected_optimizer_exposure_path_json": (
            output_dir / "selected_optimizer_exposure_path.json"
        ),
        "selected_optimizer_exposure_path_csv": (
            output_dir / "selected_optimizer_exposure_path.csv"
        ),
        "shadow_overlay": output_dir / "allocation_shadow_overlay.json",
        "meta_audit": output_dir / "meta_dataset_audit.json",
        "meta_dataset": Path(
            ml_config.get("meta_dataset_path", cache_dir / "meta_ensemble_dataset.csv")
        ),
        "meta_auxiliary_predictions": output_dir / "meta_auxiliary_predictions.csv",
        "expanded_dataset": Path(
            ml_config.get(
                "expanded_rebalance_dataset_path",
                cache_dir / "expanded_rebalance_dataset.csv",
            )
        ),
        "expanded_audit": Path(
            ml_config.get(
                "expanded_rebalance_audit_path",
                report_dir / "expanded_rebalance_dataset_audit.json",
            )
        ),
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return payload if isinstance(payload, dict) else {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
