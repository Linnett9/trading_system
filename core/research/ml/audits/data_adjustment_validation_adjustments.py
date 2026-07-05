from __future__ import annotations

from typing import Any

from core.research.ml.audits.data_adjustment_validation_config import _normalize_audit_config
from core.research.ml.audits.data_adjustment_validation_dependencies import (
    _candidate_suspicious_dependencies,
    _daily_row_dependencies,
    _period_anomaly_dependencies,
    _period_anomaly_rows,
    _suspicious_rebalance_dates_from_daily_rows,
    _unique_dependency_rows,
)
from core.research.ml.audits.data_adjustment_validation_detection import detect_split_like_jumps
from core.research.ml.audits.data_adjustment_validation_flags import (
    _adjustment_red_flags,
    _symbols_to_audit,
)
from core.research.ml.audits.data_adjustment_validation_price_rows import (
    _normalized_price_rows,
    _split_like_factor,
)
from core.research.ml.audits.data_adjustment_validation_status import (
    _adjusted_status_acceptable,
    _overall_adjusted_status,
    _raw_adjusted_comparison,
    _symbol_adjusted_status,
    _symbol_adjustment_report,
)
from core.research.ml.audits.data_adjustment_validation_types import (
    DEFAULT_INSPECT_SYMBOLS,
    RESEARCH_METADATA,
)


def build_data_adjustment_audit(
    *,
    symbol_rows_by_symbol: dict[str, list[dict[str, Any]]],
    canonical_replay: dict[str, Any],
    champion_audit: dict[str, Any],
    audit_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = _normalize_audit_config(audit_config or {})
    symbol_reports = []
    suspicious_rows = []
    for symbol in sorted(symbol_rows_by_symbol):
        rows = symbol_rows_by_symbol.get(symbol, [])
        report = _symbol_adjustment_report(symbol, rows, config)
        symbol_reports.append(report)
        suspicious_rows.extend(report["suspicious_rows"])
    period_anomalies = _period_anomaly_rows(champion_audit, config)
    suspicious_rebalance_dates = sorted({
        *(
            str(row.get("rebalance_date"))
            for row in period_anomalies
            if row.get("rebalance_date")
        ),
        *_suspicious_rebalance_dates_from_daily_rows(
            canonical_replay,
            suspicious_rows,
        ),
    })
    dependencies = _candidate_suspicious_dependencies(
        canonical_replay,
        suspicious_rows=suspicious_rows,
        period_anomalies=period_anomalies,
    )
    adjusted_status = _overall_adjusted_status(symbol_reports)
    acceptable = _adjusted_status_acceptable(adjusted_status, config)
    return {
        "mode": "stooq_data_adjustment_audit_research_only",
        "data_source": "local Stooq parquet research data",
        "data_path": config["stooq_parquet_dir"],
        "inspect_symbols": sorted(symbol_rows_by_symbol),
        "required_inspection_symbols": list(DEFAULT_INSPECT_SYMBOLS),
        "price_column_used": "close",
        "adjusted_price_status": adjusted_status,
        "adjusted_status": adjusted_status,
        "promotion_gate": {
            "adjusted_price_status_acceptable": acceptable,
            "acceptable_statuses": sorted(config["acceptable_adjusted_price_statuses"]),
            "allow_unknown_adjusted_price_status": config[
                "allow_unknown_adjusted_price_status"
            ],
        },
        "thresholds": {
            "suspicious_daily_return_abs": config["suspicious_daily_return_abs"],
            "impossible_daily_return_abs": config["impossible_daily_return_abs"],
            "large_symbol_period_return_abs": config[
                "large_symbol_period_return_abs"
            ],
            "split_ratio_tolerance": config["split_ratio_tolerance"],
        },
        "symbol_count": len(symbol_reports),
        "suspicious_row_count": len(suspicious_rows),
        "period_anomaly_count": len(period_anomalies),
        "suspicious_rebalance_dates": suspicious_rebalance_dates,
        "suspicious_symbols": sorted({
            str(row.get("symbol"))
            for row in [*suspicious_rows, *period_anomalies]
            if row.get("symbol")
        }),
        "candidate_dependencies": dependencies,
        "symbols": symbol_reports,
        "suspicious_rows": suspicious_rows,
        "period_anomalies": period_anomalies,
        "red_flags": _adjustment_red_flags(
            adjusted_status,
            acceptable,
            suspicious_rows,
            period_anomalies,
            dependencies,
        ),
        **RESEARCH_METADATA,
    }
