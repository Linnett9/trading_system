from __future__ import annotations

from pathlib import Path
from typing import Any

from core.research.framework.config import StockLevelResearchConfig


def _artifact_status(
    config: dict[str, Any],
    settings: StockLevelResearchConfig,
) -> dict[str, Any]:
    source_paths = _artifact_source_paths(config, settings.output_dir)
    existing = settings.base_artifact_path
    existing_mtime = existing.stat().st_mtime if existing.exists() else None
    newest_source = max(
        (path.stat().st_mtime for path in source_paths if path.exists()),
        default=None,
    )
    refresh_required = (
        not existing.exists()
        or (
            newest_source is not None
            and existing_mtime is not None
            and newest_source > existing_mtime
        )
    )
    return {
        "path": str(existing),
        "refresh_required": refresh_required,
        "source_paths": [str(path) for path in source_paths],
    }

def _artifact_source_paths(config: dict[str, Any], output_dir: Path) -> list[Path]:
    ml = dict(config.get("ml", {}) or {})
    cache = dict(config.get("cache", {}) or {})
    return [
        Path(
            ml.get(
                "expanded_rebalance_dataset_path",
                Path(cache.get("ml_dir", "cache/ml")) / "expanded_rebalance_dataset.csv",
            )
        ),
        output_dir / "meta_auxiliary_predictions.csv",
    ]

def _with_ml_overrides(config: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    updated = dict(config)
    if "output_dir" in overrides:
        overrides["stock_alpha_output_dir_override"] = True
    updated["ml"] = {**dict(config.get("ml", {}) or {}), **overrides}
    return updated
