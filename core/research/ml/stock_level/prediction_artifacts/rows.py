from __future__ import annotations

import math
from typing import Any

from core.research.ml.stock_level.prediction_artifacts.math import (
    _average_dollar_volume,
    _trailing_drawdown,
    _trailing_liquidity_score,
    _trailing_return,
    _trailing_volatility,
)
from core.research.ml.stock_level.prediction_artifacts.targets import (
    _actual_targets,
    _add_cross_sectional_targets,
)
from core.research.ml.stock_level.prediction_artifacts.types import (
    ACTUAL_COLUMNS,
    BASELINE_PREDICTION_COLUMNS,
    CONTEXT_COLUMNS,
    PREDICTION_COLUMNS,
    RESEARCH_METADATA,
    TARGET_TYPES,
)


def build_stock_level_prediction_artifacts(
    *,
    expanded_rows: list[dict[str, str]],
    artifact_rows: list[dict[str, str]],
    universe_symbols: list[str],
    closes_by_symbol: dict[str, dict[str, dict[str, float]]],
    sector_by_symbol: dict[str, str] | None = None,
    market_symbol: str = "SPY",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sector_by_symbol = sector_by_symbol or {}
    dates = _artifact_dates(artifact_rows) or _expanded_dates(expanded_rows)
    context_by_date = _context_by_date(expanded_rows)
    artifact_by_date_symbol = _artifact_by_date_symbol(artifact_rows)
    symbols = sorted({symbol.upper() for symbol in universe_symbols if symbol})
    prepared_symbol_data = {
        symbol: _prepare_symbol_data(closes_by_symbol.get(symbol, {}))
        for symbol in symbols
    }
    market_symbol = market_symbol.upper()
    market_data = _prepare_symbol_data(closes_by_symbol.get(market_symbol, {}))
    rows: list[dict[str, Any]] = []
    for date in dates:
        context = context_by_date.get(date, {})
        for symbol in symbols:
            source = artifact_by_date_symbol.get((date, symbol), {})
            symbol_data = prepared_symbol_data.get(symbol, {})
            row = {
                "rebalance_date": date,
                "symbol": symbol,
                "sector": sector_by_symbol.get(symbol, ""),
                "average_dollar_volume_21d": _average_dollar_volume(
                    symbol_data.get("dollar_volume_dates", []),
                    symbol_data.get("dollar_volume_values", []),
                    date,
                    lookback=21,
                ),
                "average_dollar_volume_63d": _average_dollar_volume(
                    symbol_data.get("dollar_volume_dates", []),
                    symbol_data.get("dollar_volume_values", []),
                    date,
                    lookback=63,
                ),
                "source": (
                    "stock_level_prediction_artifact"
                    if source
                    else "stock_level_actuals_from_reference_prices"
                ),
                "source_feature_id": source.get("feature_id", ""),
                "source_model_type": source.get("model_type", ""),
                "source_split": source.get("split", ""),
                "source_dataset_hash": source.get("dataset_hash", ""),
                "true_stock_level_row": True,
            }
            for column in PREDICTION_COLUMNS:
                row[column] = source.get(column, "")
            row.update(_baseline_predictions(symbol_data, date))
            row.update(_actual_targets(symbol_data, date, market_data=market_data))
            for column in CONTEXT_COLUMNS:
                row[column] = context.get(column, "")
            rows.append(row)
    _add_cross_sectional_targets(rows)
    audit = _audit(rows, symbols, dates, artifact_rows)
    audit["market_residual_label_generation"] = {
        "market_symbol": market_symbol,
        "market_symbol_loaded": bool(market_data.get("close_dates")),
        "market_symbol_is_tradable_candidate": market_symbol in symbols,
        "computed_before_dev_symbol_filtering": True,
    }
    return rows, audit


def _artifact_by_date_symbol(
    artifact_rows: list[dict[str, str]],
) -> dict[tuple[str, str], dict[str, str]]:
    output = {}
    for row in artifact_rows:
        symbol = str(row.get("symbol", "")).upper()
        date = str(row.get("rebalance_date") or row.get("date") or "")
        if not symbol or not date:
            continue
        output[(date, symbol)] = row
    return output


def _prepare_symbol_data(
    symbol_data: dict[str, dict[str, float]],
) -> dict[str, Any]:
    close = dict(symbol_data.get("close", {}))
    dollar_volume = dict(symbol_data.get("dollar_volume", {}))
    close_dates = sorted(close)
    dollar_volume_dates = sorted(dollar_volume)
    return {
        "close": close,
        "dollar_volume": dollar_volume,
        "close_dates": close_dates,
        "close_values": [close[date] for date in close_dates],
        "dollar_volume_dates": dollar_volume_dates,
        "dollar_volume_values": [
            dollar_volume[date] for date in dollar_volume_dates
        ],
    }


def _baseline_predictions(
    symbol_data: dict[str, Any],
    rebalance_date: str,
) -> dict[str, Any]:
    close_dates = symbol_data.get("close_dates", [])
    close_values = symbol_data.get("close_values", [])
    dollar_volume_dates = symbol_data.get("dollar_volume_dates", [])
    dollar_volume_values = symbol_data.get("dollar_volume_values", [])
    momentum_20 = _trailing_return(close_dates, close_values, rebalance_date, lookback=20)
    momentum_60 = _trailing_return(close_dates, close_values, rebalance_date, lookback=60)
    momentum_120 = _trailing_return(close_dates, close_values, rebalance_date, lookback=120)
    volatility_20 = _trailing_volatility(close_dates, close_values, rebalance_date, lookback=20)
    drawdown_60 = _trailing_drawdown(close_dates, close_values, rebalance_date, lookback=60)
    liquidity = _trailing_liquidity_score(
        dollar_volume_dates,
        dollar_volume_values,
        rebalance_date,
        lookback=63,
    )
    risk = max(
        abs(volatility_20) if volatility_20 != "" else 0.0,
        abs(drawdown_60) if drawdown_60 != "" else 0.0,
        1e-6,
    )
    return {
        "predicted_momentum_20d": momentum_20,
        "predicted_momentum_60d": momentum_60,
        "predicted_momentum_120d": momentum_120,
        "predicted_volatility_20d": volatility_20,
        "predicted_drawdown_60d": drawdown_60,
        "predicted_liquidity_score": liquidity,
        "predicted_risk_adjusted_momentum": (
            momentum_60 / risk if momentum_60 != "" else ""
        ),
    }

def _audit(
    rows: list[dict[str, Any]],
    symbols: list[str],
    dates: list[str],
    artifact_rows: list[dict[str, str]],
) -> dict[str, Any]:
    missing_predictions = {
        column: sum(row.get(column) in (None, "") for row in rows)
        for column in PREDICTION_COLUMNS
    }
    populated_predictions = {
        column: len(rows) - missing_predictions[column]
        for column in PREDICTION_COLUMNS
    }
    missing_actuals = {
        column: sum(row.get(column) in (None, "") for row in rows)
        for column in ACTUAL_COLUMNS
    }
    target_audit = {
        column: {
            "target_type": target_type,
            "available": any(row.get(column) not in (None, "") for row in rows),
            "missing_values": missing_actuals[column],
            "date_coverage": len({row["rebalance_date"] for row in rows if row.get(column) not in (None, "")}),
            "symbol_coverage": len({row["symbol"] for row in rows if row.get(column) not in (None, "")}),
        }
        for column, target_type in TARGET_TYPES.items()
    }
    artifact_symbol_rows = sum(1 for row in artifact_rows if row.get("symbol"))
    return {
        "mode": "stock_level_prediction_artifacts_research_only",
        "purpose": (
            "Create one row per symbol per rebalance_date for Phase 2A "
            "cross-sectional ranking research without replacing existing "
            "artifact-level prediction files."
        ),
        "root_cause_artifact_level_limitation": (
            "Existing prediction_artifacts.csv rows are keyed by feature_id/"
            "variant_id and have blank symbol values; they predict strategy/"
            "variant outcomes rather than individual security outcomes."
        ),
        "row_count": len(rows),
        "symbol_count": len(symbols),
        "rebalance_date_count": len(dates),
        "date_range": [dates[0], dates[-1]] if dates else None,
        "average_symbols_per_date": (len(rows) / len(dates)) if dates else 0.0,
        "missing_prediction_counts": missing_predictions,
        "populated_prediction_counts": populated_predictions,
        "missing_actual_target_counts": missing_actuals,
        "target_audit": target_audit,
        "artifact_rows_with_symbol_predictions": artifact_symbol_rows,
        "true_stock_level_rows": bool(rows),
        "usable_for_stock_level_ranking": (
            bool(rows)
            and any(populated_predictions[column] > 0 for column in BASELINE_PREDICTION_COLUMNS)
        ),
        "suitable_for_true_stock_level_ranking_diagnostics": (
            bool(rows)
            and any(populated_predictions[column] > 0 for column in BASELINE_PREDICTION_COLUMNS)
        ),
        "suitability_reason": (
            "stock-level rows include point-in-time baseline forecast signals"
            if any(populated_predictions[column] > 0 for column in BASELINE_PREDICTION_COLUMNS)
            else (
                "stock-level rows and actual targets are present, but current saved "
                "model artifacts do not contain symbol-level predictions"
            )
        ),
        "existing_artifact_level_files_preserved": True,
        "prediction_fields_are_explicitly_missing": True,
        "leakage_safety_note": (
            "Actual targets are computed from prices after rebalance_date. "
            "They are evaluation fields only, not prediction inputs."
        ),
        "leakage_safety_notes": [
            "Baseline forecast columns use only price and volume observations strictly before rebalance_date.",
            "Actual target columns use post-rebalance prices and are evaluation fields only.",
            "Existing artifact-level prediction files are preserved and not overwritten.",
        ],
        **RESEARCH_METADATA,
    }


def _context_by_date(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    output = {}
    for row in rows:
        date = row.get("rebalance_date") or row.get("feature_date")
        if not date or date in output:
            continue
        output[date] = {column: row.get(column, "") for column in CONTEXT_COLUMNS}
    return output


def _artifact_dates(rows: list[dict[str, str]]) -> list[str]:
    return sorted({
        str(row.get("rebalance_date") or row.get("date") or "")
        for row in rows
        if row.get("rebalance_date") or row.get("date")
    })


def _expanded_dates(rows: list[dict[str, str]]) -> list[str]:
    return sorted({
        str(row.get("rebalance_date") or row.get("feature_date") or "")
        for row in rows
        if row.get("rebalance_date") or row.get("feature_date")
    })
