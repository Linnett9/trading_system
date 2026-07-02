from __future__ import annotations

from pathlib import Path
from typing import Any

from core.research.ml.audits.adjusted_data_types import DEFAULT_INSPECT_SYMBOLS


def _normalize_comparison_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "stooq_parquet_dir": str(
            config.get("stooq_parquet_dir", "data/processed/stooq_parquet")
        ),
        "adjusted_source_name": str(
            config.get("adjusted_source_name", "local_adjusted_price_csv")
        ),
        "adjusted_data_dir": str(
            config.get("adjusted_data_dir", "data/reference/adjusted_prices")
        ),
        "adjusted_combined_path": config.get("adjusted_combined_path"),
        "inspect_symbols": [
            str(symbol).upper()
            for symbol in config.get("inspect_symbols", DEFAULT_INSPECT_SYMBOLS)
        ],
        "suspicious_daily_return_abs": float(
            config.get("suspicious_daily_return_abs", 0.50)
        ),
        "split_ratio_tolerance": float(config.get("split_ratio_tolerance", 0.08)),
    }


def _comparison_config(config: dict[str, Any]) -> dict[str, Any]:
    ml_config = config.get("ml", {})
    source = dict(ml_config.get("adjusted_data_source", {}) or {})
    audit = dict(ml_config.get("data_adjustment_audit", {}) or {})
    source.setdefault(
        "stooq_parquet_dir",
        ml_config.get("stooq_parquet_dir", "data/processed/stooq_parquet"),
    )
    for key in ("inspect_symbols", "suspicious_daily_return_abs", "split_ratio_tolerance"):
        if key in audit:
            source[key] = audit[key]
    return _normalize_comparison_config(source)


def _validation_config(config: dict[str, Any]) -> dict[str, Any]:
    ml_config = config.get("ml", {})
    validation = dict(ml_config.get("benchmark_relative_validation", {}) or {})
    validation["adjusted_replay"] = dict(
        ml_config.get("adjusted_replay", {}) or {}
    )
    return validation


def _output_dir(config: dict[str, Any]) -> Path:
    ml_config = config.get("ml", {})
    return Path(
        ml_config.get(
            "output_dir",
            Path(config.get("reports", {}).get("ml_dir", "reports/ml"))
            / "regime_transformer_meta_ensemble_v1",
        )
    )
