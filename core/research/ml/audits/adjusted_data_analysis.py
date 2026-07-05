from __future__ import annotations

from typing import Any

from core.research.ml.audits.adjusted_data_config import _normalize_comparison_config
from core.research.ml.audits.adjusted_data_loading import (
    _adjusted_close_by_date,
    _date,
    _number,
    _raw_close_by_date,
)
from core.research.ml.audits.adjusted_data_types import (
    COMMON_SPLIT_FACTORS,
    RESEARCH_METADATA,
)
from infrastructure.data.adjusted_price_csv_data_feed import AdjustedPricePoint


def build_adjusted_data_comparison(
    *,
    raw_rows_by_symbol: dict[str, list[dict[str, Any]]],
    adjusted_rows_by_symbol: dict[str, list[AdjustedPricePoint | dict[str, Any]]],
    canonical_replay: dict[str, Any],
    comparison_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = _normalize_comparison_config(comparison_config or {})
    rows = []
    symbol_reports = []
    for symbol in sorted(raw_rows_by_symbol):
        raw_by_date = _raw_close_by_date(raw_rows_by_symbol.get(symbol, []))
        adjusted_by_date = _adjusted_close_by_date(
            adjusted_rows_by_symbol.get(symbol, [])
        )
        symbol_rows = _comparison_rows_for_symbol(
            symbol,
            raw_by_date,
            adjusted_by_date,
            config,
        )
        rows.extend(symbol_rows)
        symbol_reports.append(_symbol_report(symbol, raw_by_date, adjusted_by_date, symbol_rows))
    dependencies = _candidate_distortion_dependencies(canonical_replay, rows)
    required = set(config["inspect_symbols"])
    required_available = [
        row for row in symbol_reports
        if row["symbol"] in required and row["adjusted_source_available"]
    ]
    source_status = (
        "available"
        if len(required_available) == len(required)
        else "partial"
        if required_available
        else "missing"
    )
    acceptable = source_status == "available"
    anomaly_survival = _anomaly_survival_by_symbol(symbol_reports, required)
    return {
        "mode": "adjusted_data_comparison_research_only",
        "raw_source": {
            "name": "stooq_parquet_close",
            "path": config["stooq_parquet_dir"],
            "preserved_separately": True,
        },
        "adjusted_source": {
            "name": config["adjusted_source_name"],
            "data_dir": config["adjusted_data_dir"],
            "combined_path": config.get("adjusted_combined_path"),
            "available_status": source_status,
            "acceptable": acceptable,
        },
        "promotion_gate": {
            "adjusted_source_available_and_acceptable": acceptable,
        },
        "inspect_symbols": list(config["inspect_symbols"]),
        "symbol_count": len(symbol_reports),
        "comparison_row_count": len(rows),
        "split_like_distortion_count": sum(
            bool(row.get("split_like_distortion")) for row in rows
        ),
        "candidate_dependencies": dependencies,
        "anomaly_survival_by_symbol": anomaly_survival,
        "symbols": symbol_reports,
        "rows": rows,
        "red_flags": _comparison_red_flags(source_status, rows, dependencies),
        **RESEARCH_METADATA,
    }


def detect_split_like_adjustment_ratio(
    previous_ratio: float | None,
    current_ratio: float | None,
    *,
    tolerance: float = 0.08,
) -> float | None:
    if previous_ratio is None or current_ratio is None:
        return None
    if previous_ratio <= 0.0 or current_ratio <= 0.0:
        return None
    ratio_change = current_ratio / previous_ratio
    return _split_like_factor(ratio_change, tolerance)


def _comparison_rows_for_symbol(
    symbol: str,
    raw_by_date: dict[str, float],
    adjusted_by_date: dict[str, float],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    previous: dict[str, Any] | None = None
    for day in sorted(set(raw_by_date) | set(adjusted_by_date)):
        raw_close = raw_by_date.get(day)
        adjusted_close = adjusted_by_date.get(day)
        adjustment_ratio = (
            adjusted_close / raw_close
            if raw_close is not None and adjusted_close is not None and raw_close > 0
            else None
        )
        raw_daily = _daily_return(previous, "raw_close", raw_close)
        adjusted_daily = _daily_return(previous, "adjusted_close", adjusted_close)
        ratio_factor = detect_split_like_adjustment_ratio(
            _number((previous or {}).get("adjustment_ratio")),
            adjustment_ratio,
            tolerance=config["split_ratio_tolerance"],
        )
        raw_factor = _split_like_return_factor(
            raw_daily,
            config["split_ratio_tolerance"],
        )
        adjusted_factor = _split_like_return_factor(
            adjusted_daily,
            config["split_ratio_tolerance"],
        )
        raw_suspicious = _is_suspicious_return(
            raw_daily,
            config["suspicious_daily_return_abs"],
        )
        adjusted_suspicious = _is_suspicious_return(
            adjusted_daily,
            config["suspicious_daily_return_abs"],
        )
        split_like_distortion = bool(
            raw_suspicious
            and not adjusted_suspicious
            and (ratio_factor is not None or raw_factor is not None)
        )
        row = {
            "symbol": symbol.upper(),
            "date": day,
            "raw_close": raw_close,
            "adjusted_close": adjusted_close,
            "adjustment_ratio": adjustment_ratio,
            "raw_daily_return": raw_daily,
            "adjusted_daily_return": adjusted_daily,
            "raw_split_like_factor": raw_factor,
            "adjusted_split_like_factor": adjusted_factor,
            "adjustment_ratio_split_like_factor": ratio_factor,
            "raw_suspicious_jump": raw_suspicious,
            "adjusted_suspicious_jump": adjusted_suspicious,
            "split_like_distortion": split_like_distortion,
            "anomaly_survives_adjustment": bool(
                raw_suspicious and (adjusted_suspicious or adjusted_close is None)
            ),
            **RESEARCH_METADATA,
        }
        rows.append(row)
        previous = row
    return rows


def _symbol_report(
    symbol: str,
    raw_by_date: dict[str, float],
    adjusted_by_date: dict[str, float],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    comparable = [
        row for row in rows
        if row.get("raw_close") is not None and row.get("adjusted_close") is not None
    ]
    return {
        "symbol": symbol.upper(),
        "raw_source_available": bool(raw_by_date),
        "adjusted_source_available": bool(adjusted_by_date),
        "raw_row_count": len(raw_by_date),
        "adjusted_row_count": len(adjusted_by_date),
        "comparable_row_count": len(comparable),
        "first_comparable_date": comparable[0]["date"] if comparable else None,
        "last_comparable_date": comparable[-1]["date"] if comparable else None,
        "split_like_distortion_count": sum(
            bool(row.get("split_like_distortion")) for row in rows
        ),
        "anomaly_survives_adjustment_count": sum(
            bool(row.get("anomaly_survives_adjustment")) for row in rows
        ),
        **RESEARCH_METADATA,
    }


def _candidate_distortion_dependencies(
    canonical_replay: dict[str, Any],
    comparison_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    distortions = [
        row for row in comparison_rows
        if row.get("split_like_distortion")
    ]
    output = {}
    for name, candidate in canonical_replay.get("candidates", {}).items():
        dependencies = []
        for row in candidate.get("rows", []) or []:
            dependencies.extend(_row_distortion_dependencies(row, distortions))
        unique = _unique_dependencies(dependencies)
        output[str(name)] = {
            "candidate_name": str(name),
            "raw_adjusted_distortion_dependency_count": len(unique),
            "distortion_rebalance_dates": sorted({
                str(row.get("rebalance_date"))
                for row in unique
                if row.get("rebalance_date")
            }),
            "distortion_symbols": sorted({
                str(row.get("symbol"))
                for row in unique
                if row.get("symbol")
            }),
            "dependencies": unique[:100],
            **RESEARCH_METADATA,
        }
    return output


def _row_distortion_dependencies(
    period: dict[str, Any],
    distortions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    start = _date(period.get("rebalance_date"))
    end = _date(period.get("outcome_end_date")) or start
    symbols = {str(symbol).upper() for symbol in period.get("selected_symbols", [])}
    if start is None or end is None:
        return []
    rows = []
    for distortion in distortions:
        event_date = _date(distortion.get("date"))
        symbol = str(distortion.get("symbol", "")).upper()
        if event_date is None or symbol not in symbols:
            continue
        if start <= event_date <= end:
            rows.append({
                "rebalance_date": period.get("rebalance_date"),
                "outcome_end_date": period.get("outcome_end_date"),
                "symbol": symbol,
                "event_date": distortion.get("date"),
                "raw_daily_return": distortion.get("raw_daily_return"),
                "adjusted_daily_return": distortion.get("adjusted_daily_return"),
                "adjustment_ratio": distortion.get("adjustment_ratio"),
            })
    return rows


def _anomaly_survival_by_symbol(
    symbol_reports: list[dict[str, Any]],
    required_symbols: set[str],
) -> dict[str, dict[str, Any]]:
    by_symbol = {row["symbol"]: row for row in symbol_reports}
    output = {}
    for symbol in sorted(required_symbols):
        row = by_symbol.get(symbol, {})
        adjusted_available = bool(row.get("adjusted_source_available", False))
        survives = (
            not adjusted_available
            or int(row.get("anomaly_survives_adjustment_count") or 0) > 0
        )
        output[symbol] = {
            "symbol": symbol,
            "adjusted_source_available": adjusted_available,
            "raw_adjusted_distortion_count": int(
                row.get("split_like_distortion_count") or 0
            ),
            "anomaly_survives_adjustment_count": int(
                row.get("anomaly_survives_adjustment_count") or 0
            ),
            "anomaly_survives_adjusted_comparison": survives,
        }
    return output


def _comparison_red_flags(
    source_status: str,
    rows: list[dict[str, Any]],
    dependencies: dict[str, dict[str, Any]],
) -> list[str]:
    flags = []
    if source_status != "available":
        flags.append("adjusted_source_missing_or_partial")
    if any(row.get("split_like_distortion") for row in rows):
        flags.append("raw_adjusted_split_like_distortions_present")
    if any(
        int(row.get("raw_adjusted_distortion_dependency_count") or 0) > 0
        for row in dependencies.values()
    ):
        flags.append("candidate_depends_on_raw_adjusted_distortion")
    return sorted(set(flags))


def _adjusted_closes_from_comparison(
    comparison: dict[str, Any],
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for row in comparison.get("rows", []) or []:
        adjusted = _number(row.get("adjusted_close"))
        if adjusted is None:
            continue
        symbol = str(row.get("symbol", "")).upper()
        day = str(row.get("date", ""))
        if not symbol or not day:
            continue
        output.setdefault(symbol, {})[day] = adjusted
    return output


def _daily_return(
    previous: dict[str, Any] | None,
    key: str,
    current: float | None,
) -> float | None:
    prior = _number((previous or {}).get(key))
    if prior is None or current is None or prior <= 0:
        return None
    return current / prior - 1.0


def _split_like_return_factor(
    daily_return: float | None,
    tolerance: float,
) -> float | None:
    if daily_return is None:
        return None
    return _split_like_factor(1.0 + daily_return, tolerance)


def _split_like_factor(ratio: float, tolerance: float) -> float | None:
    if ratio <= 0.0:
        return None
    for factor in COMMON_SPLIT_FACTORS:
        inverse = 1.0 / factor
        if abs(ratio - factor) / factor <= tolerance:
            return factor
        if abs(ratio - inverse) / inverse <= tolerance:
            return factor
    return None


def _is_suspicious_return(value: float | None, threshold: float) -> bool:
    return value is not None and abs(value) >= threshold


def _unique_dependencies(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    output = []
    for row in rows:
        key = (row.get("rebalance_date"), row.get("symbol"), row.get("event_date"))
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output


def _symbols_to_compare(
    canonical_replay: dict[str, Any],
    config: dict[str, Any],
) -> list[str]:
    symbols = {str(symbol).upper() for symbol in config["inspect_symbols"]}
    symbols.update({"SPY", "QQQ"})
    for candidate in canonical_replay.get("candidates", {}).values():
        for row in candidate.get("rows", []) or []:
            symbols.update(
                str(symbol).upper()
                for symbol in row.get("selected_symbols", []) or []
            )
    return sorted(symbol for symbol in symbols if symbol)
