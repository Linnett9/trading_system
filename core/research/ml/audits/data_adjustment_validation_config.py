from __future__ import annotations

from pathlib import Path
from typing import Any

from core.research.ml.audits.data_adjustment_validation_types import (
    DEFAULT_INSPECT_SYMBOLS,
)


def _normalize_audit_config(config: dict[str, Any]) -> dict[str, Any]:
    validation = dict(config)
    acceptable = validation.get("acceptable_adjusted_price_statuses")
    if acceptable is None:
        acceptable = {
            "known_adjusted",
            "appears_adjusted",
            "raw_adjusted_identical",
        }
    return {
        "stooq_parquet_dir": str(
            validation.get("stooq_parquet_dir", "data/processed/stooq_parquet")
        ),
        "inspect_symbols": [
            str(symbol).upper()
            for symbol in validation.get("inspect_symbols", DEFAULT_INSPECT_SYMBOLS)
        ],
        "suspicious_daily_return_abs": float(
            validation.get("suspicious_daily_return_abs", 0.50)
        ),
        "impossible_daily_return_abs": float(
            validation.get("impossible_daily_return_abs", 4.0)
        ),
        "large_symbol_period_return_abs": float(
            validation.get("large_symbol_period_return_abs", 1.0)
        ),
        "split_ratio_tolerance": float(validation.get("split_ratio_tolerance", 0.08)),
        "acceptable_adjusted_price_statuses": {
            str(status) for status in acceptable
        },
        "allow_unknown_adjusted_price_status": bool(
            validation.get("allow_unknown_adjusted_price_status", False)
        ),
    }


def _audit_config(config: dict[str, Any]) -> dict[str, Any]:
    ml_config = config.get("ml", {})
    audit = dict(ml_config.get("data_adjustment_audit", {}) or {})
    validation = dict(ml_config.get("benchmark_relative_validation", {}) or {})
    audit.setdefault(
        "stooq_parquet_dir",
        ml_config.get("stooq_parquet_dir", "data/processed/stooq_parquet"),
    )
    for key in (
        "acceptable_adjusted_price_statuses",
        "allow_unknown_adjusted_price_status",
    ):
        if key in validation:
            audit[key] = validation[key]
    return _normalize_audit_config(audit)


def _validation_config(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("ml", {}).get("benchmark_relative_validation", {}) or {})


def _output_dir(config: dict[str, Any]) -> Path:
    ml_config = config.get("ml", {})
    return Path(
        ml_config.get(
            "output_dir",
            Path(config.get("reports", {}).get("ml_dir", "reports/ml"))
            / "regime_transformer_meta_ensemble_v1",
        )
    )
