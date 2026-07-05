from __future__ import annotations

from pathlib import Path
from typing import Any


def _normalized_expansion_config(config: dict[str, Any]) -> dict[str, Any]:
    settings = config.get("settings")
    if not isinstance(settings, list) or not settings:
        settings = [
            {
                "name": "current_strict_non_overlap",
                "description": "Current adjusted canonical non-overlap selection.",
                "spacing": "strict_non_overlap",
                "minimum_gap_days": 0,
                "enforce_non_overlap": True,
            },
            {
                "name": "monthly_leakage_safe",
                "description": "First valid adjusted period each month, then remove overlaps.",
                "spacing": "monthly",
                "minimum_gap_days": 0,
                "enforce_non_overlap": True,
            },
            {
                "name": "quarterly_leakage_safe",
                "description": "First valid adjusted period each quarter, then remove overlaps.",
                "spacing": "quarterly",
                "minimum_gap_days": 0,
                "enforce_non_overlap": True,
            },
            {
                "name": "all_valid_min_gap_0",
                "description": "All valid adjusted periods filtered by label-window non-overlap.",
                "spacing": "all_valid_min_gap",
                "minimum_gap_days": 0,
                "enforce_non_overlap": True,
            },
        ]
    normalized = []
    for item in settings:
        if not isinstance(item, dict):
            continue
        normalized.append({
            "name": str(item.get("name") or item.get("spacing") or "setting"),
            "description": str(item.get("description") or ""),
            "spacing": str(item.get("spacing") or "monthly"),
            "minimum_gap_days": int(item.get("minimum_gap_days", 0)),
            "enforce_non_overlap": bool(item.get("enforce_non_overlap", True)),
        })
    return {"settings": normalized}
def _validation_config(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("ml", {}).get("benchmark_relative_validation", {}) or {})
def _expansion_config(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("ml", {}).get("independent_period_expansion", {}) or {})
def _output_dir(config: dict[str, Any]) -> Path:
    return Path(
        config.get("ml", {}).get(
            "output_dir",
            "reports/ml/regime_transformer_meta_ensemble_v1",
        )
    )
