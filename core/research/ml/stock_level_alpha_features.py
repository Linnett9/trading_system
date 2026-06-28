from __future__ import annotations

import csv
import json
import math
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


RESEARCH_METADATA = {
    "research_only": True,
    "trading_impact": "none",
    "production_validated": False,
    "promotion_thresholds_changed": False,
}
NOTICE = "Research only. Trading impact: none. Production validated: false."
ENGINEERED_FEATURE_COLUMNS = (
    "momentum_250d",
    "momentum_acceleration",
    "momentum_persistence",
    "momentum_consistency",
    "relative_momentum_vs_spy",
    "relative_momentum_vs_sector",
    "momentum_percentile",
    "distance_from_52_week_high",
    "drawdown_recovery_days",
    "rolling_max_drawdown_120d",
    "ulcer_index",
    "downside_deviation",
    "volatility_percentile",
    "volatility_trend",
    "volatility_regime",
    "ATR_percentile",
    "sector_relative_strength",
    "industry_relative_strength",
)
FEATURE_DEFINITIONS = {
    "momentum_250d": "Trailing 250-observation return using prices strictly before rebalance.",
    "momentum_acceleration": "OLS slope of 20d, 60d, and 120d momentum versus horizon.",
    "momentum_persistence": "Fraction of the latest 120 trailing 20d return windows that are positive.",
    "momentum_consistency": "R-squared of a linear trend fitted to 120 log closing prices.",
    "relative_momentum_vs_spy": "Stock 120d momentum minus SPY 120d momentum on the same date.",
    "relative_momentum_vs_sector": "Stock 120d momentum minus its sector cross-sectional mean.",
    "momentum_percentile": "Cross-sectional percentile of 120d momentum on each rebalance date.",
    "distance_from_52_week_high": "Latest close divided by the prior 252-observation high, minus one.",
    "drawdown_recovery_days": "Trading observations since the latest prior 252-observation high; zero at a high.",
    "rolling_max_drawdown_120d": "Worst peak-to-trough drawdown inside the prior 120 observations.",
    "ulcer_index": "Root mean square percentage drawdown over the prior 120 observations.",
    "downside_deviation": "Root mean square of negative daily returns over the prior 60 observations.",
    "volatility_percentile": "Percentile of current 20d volatility versus its prior 252 observations.",
    "volatility_trend": "Current 20d volatility divided by 60d volatility, minus one.",
    "volatility_regime": "Numeric volatility bucket: 0 low, 1 normal, 2 high.",
    "ATR_percentile": "Percentile of normalized ATR(14) versus its prior 252 observations.",
    "sector_relative_strength": "Within-sector percentile of 120d momentum on each rebalance date.",
    "industry_relative_strength": "Within-industry percentile of 120d momentum when industry metadata exists.",
}


@dataclass(frozen=True)
class StockLevelAlphaFeaturePaths:
    enriched_csv_path: Path
    audit_csv_path: Path
    audit_json_path: Path
    audit_markdown_path: Path


def write_stock_level_alpha_features(
    config: dict[str, Any],
) -> StockLevelAlphaFeaturePaths:
    ml_config = config.get("ml", {})
    output_dir = _output_dir(config)
    source_path = Path(
        ml_config.get(
            "stock_level_base_prediction_artifacts_path",
            output_dir / "stock_level_prediction_artifacts.csv",
        )
    )
    if not source_path.exists():
        raise FileNotFoundError(f"Base stock-level artifact not found: {source_path}")
    rows = _read_csv(source_path)
    symbols = sorted({str(row.get("symbol", "")).upper() for row in rows if row.get("symbol")})
    spy_symbol = str(ml_config.get("stock_ranker_spy_symbol", "SPY")).upper()
    price_histories = _load_price_histories(
        Path(ml_config.get("stooq_parquet_dir", "data/processed/stooq_parquet")),
        sorted({*symbols, spy_symbol}),
    )
    enriched_rows, audit = build_stock_level_alpha_features(
        rows,
        price_histories,
        spy_symbol=spy_symbol,
        source_path=str(source_path),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = StockLevelAlphaFeaturePaths(
        enriched_csv_path=output_dir / "stock_level_prediction_artifacts_enriched.csv",
        audit_csv_path=output_dir / "stock_level_alpha_feature_audit.csv",
        audit_json_path=output_dir / "stock_level_alpha_feature_audit.json",
        audit_markdown_path=output_dir / "stock_level_alpha_feature_audit.md",
    )
    _write_enriched_csv(paths.enriched_csv_path, rows, enriched_rows)
    _write_audit_csv(paths.audit_csv_path, audit["features"])
    paths.audit_json_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    paths.audit_markdown_path.write_text(_markdown(audit), encoding="utf-8")
    return paths


def build_stock_level_alpha_features(
    rows: list[dict[str, Any]],
    price_histories: dict[str, list[dict[str, Any]]],
    *,
    spy_symbol: str = "SPY",
    source_path: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prepared_histories = {
        symbol.upper(): _prepare_history(history)
        for symbol, history in price_histories.items()
    }
    spy_history = prepared_histories.get(spy_symbol.upper(), [])
    enriched_rows: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        symbol = str(row.get("symbol", "")).upper()
        rebalance_date = str(row.get("rebalance_date", ""))
        history = _history_before(prepared_histories.get(symbol, []), rebalance_date)
        spy_before = _history_before(spy_history, rebalance_date)
        row.update(_time_series_features(history, spy_before))
        enriched_rows.append(row)
    _add_cross_sectional_features(enriched_rows)
    audit = _audit(rows, enriched_rows, prepared_histories, source_path)
    return enriched_rows, audit


def _time_series_features(
    history: list[dict[str, float | str]],
    spy_history: list[dict[str, float | str]],
) -> dict[str, Any]:
    closes = [float(row["close"]) for row in history]
    spy_closes = [float(row["close"]) for row in spy_history]
    momentum_20 = _trailing_return(closes, 20)
    momentum_60 = _trailing_return(closes, 60)
    momentum_120 = _trailing_return(closes, 120)
    momentum_250 = _trailing_return(closes, 250)
    spy_momentum_120 = _trailing_return(spy_closes, 120)
    volatility_20 = _volatility(closes, 20)
    volatility_60 = _volatility(closes, 60)
    volatility_percentile = _volatility_percentile(closes)
    return {
        "momentum_250d": momentum_250,
        "momentum_acceleration": _slope(
            (20.0, 60.0, 120.0),
            (momentum_20, momentum_60, momentum_120),
        ),
        "momentum_persistence": _momentum_persistence(closes),
        "momentum_consistency": _trend_r_squared(closes, 120),
        "relative_momentum_vs_spy": _difference(momentum_120, spy_momentum_120),
        "relative_momentum_vs_sector": "",
        "momentum_percentile": "",
        "distance_from_52_week_high": _distance_from_high(closes, 252),
        "drawdown_recovery_days": _drawdown_recovery_days(closes, 252),
        "rolling_max_drawdown_120d": _max_drawdown(closes, 120),
        "ulcer_index": _ulcer_index(closes, 120),
        "downside_deviation": _downside_deviation(closes, 60),
        "volatility_percentile": volatility_percentile,
        "volatility_trend": _ratio_minus_one(volatility_20, volatility_60),
        "volatility_regime": _volatility_regime(volatility_percentile),
        "ATR_percentile": _atr_percentile(history),
        "sector_relative_strength": "",
        "industry_relative_strength": "",
    }


def _add_cross_sectional_features(rows: list[dict[str, Any]]) -> None:
    by_date: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_date.setdefault(str(row.get("rebalance_date", "")), []).append(row)
    for date_rows in by_date.values():
        momentum = {
            id(row): _number(row.get("predicted_momentum_120d"))
            for row in date_rows
        }
        global_values = [value for value in momentum.values() if value is not None]
        for row in date_rows:
            value = momentum[id(row)]
            row["momentum_percentile"] = _percentile_rank(value, global_values)
        _add_group_relative_features(
            date_rows,
            momentum,
            group_column="sector",
            difference_column="relative_momentum_vs_sector",
            percentile_column="sector_relative_strength",
        )
        _add_group_relative_features(
            date_rows,
            momentum,
            group_column="industry",
            difference_column=None,
            percentile_column="industry_relative_strength",
        )


def _add_group_relative_features(
    rows: list[dict[str, Any]],
    momentum: dict[int, float | None],
    *,
    group_column: str,
    difference_column: str | None,
    percentile_column: str,
) -> None:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        group = str(row.get(group_column, "")).strip()
        value = momentum[id(row)]
        if group and value is not None:
            grouped.setdefault(group, []).append(value)
    for row in rows:
        group = str(row.get(group_column, "")).strip()
        value = momentum[id(row)]
        values = grouped.get(group, [])
        if not group or value is None or not values:
            if difference_column:
                row[difference_column] = ""
            row[percentile_column] = ""
            continue
        if difference_column:
            row[difference_column] = value - mean(values)
        row[percentile_column] = _percentile_rank(value, values)


def _prepare_history(history: list[dict[str, Any]]) -> list[dict[str, float | str]]:
    prepared = []
    for row in history:
        date = str(row.get("date") or row.get("timestamp") or "")[:10]
        close = _number(row.get("close"))
        if not date or close is None or close <= 0.0:
            continue
        prepared.append(
            {
                "date": date,
                "close": close,
                "high": _number(row.get("high")) or close,
                "low": _number(row.get("low")) or close,
            }
        )
    return sorted(prepared, key=lambda row: str(row["date"]))


def _history_before(
    history: list[dict[str, float | str]],
    rebalance_date: str,
) -> list[dict[str, float | str]]:
    dates = [str(row["date"]) for row in history]
    return history[: bisect_left(dates, rebalance_date)]


def _trailing_return(values: list[float], lookback: int) -> float | str:
    if len(values) <= lookback or values[-lookback - 1] <= 0.0:
        return ""
    return values[-1] / values[-lookback - 1] - 1.0


def _momentum_persistence(values: list[float]) -> float | str:
    if len(values) < 140:
        return ""
    endpoints = range(len(values) - 120, len(values))
    outcomes = [values[index] / values[index - 20] - 1.0 for index in endpoints]
    return mean(1.0 if value > 0.0 else 0.0 for value in outcomes)


def _trend_r_squared(values: list[float], lookback: int) -> float | str:
    if len(values) < lookback:
        return ""
    sample = [math.log(value) for value in values[-lookback:] if value > 0.0]
    if len(sample) != lookback:
        return ""
    x = list(range(lookback))
    x_mean = mean(x)
    y_mean = mean(sample)
    denominator = sum((value - x_mean) ** 2 for value in x)
    slope = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, sample)) / denominator
    fitted = [y_mean + slope * (value - x_mean) for value in x]
    total = sum((value - y_mean) ** 2 for value in sample)
    residual = sum((value - estimate) ** 2 for value, estimate in zip(sample, fitted))
    return 1.0 - residual / total if total > 0.0 else 1.0


def _distance_from_high(values: list[float], lookback: int) -> float | str:
    if len(values) < lookback:
        return ""
    high = max(values[-lookback:])
    return values[-1] / high - 1.0 if high > 0.0 else ""


def _drawdown_recovery_days(values: list[float], lookback: int) -> int | str:
    if len(values) < lookback:
        return ""
    sample = values[-lookback:]
    high = max(sample)
    latest_high_index = max(index for index, value in enumerate(sample) if value == high)
    return len(sample) - 1 - latest_high_index


def _max_drawdown(values: list[float], lookback: int) -> float | str:
    if len(values) < lookback:
        return ""
    peak = values[-lookback]
    worst = 0.0
    for value in values[-lookback:]:
        peak = max(peak, value)
        worst = min(worst, value / peak - 1.0)
    return worst


def _ulcer_index(values: list[float], lookback: int) -> float | str:
    if len(values) < lookback:
        return ""
    peak = values[-lookback]
    squared = []
    for value in values[-lookback:]:
        peak = max(peak, value)
        squared.append((value / peak - 1.0) ** 2)
    return math.sqrt(mean(squared))


def _downside_deviation(values: list[float], lookback: int) -> float | str:
    if len(values) <= lookback:
        return ""
    sample = values[-lookback - 1 :]
    returns = [sample[index] / sample[index - 1] - 1.0 for index in range(1, len(sample))]
    return math.sqrt(mean(min(value, 0.0) ** 2 for value in returns))


def _volatility(values: list[float], lookback: int) -> float | str:
    if len(values) <= lookback:
        return ""
    sample = values[-lookback - 1 :]
    returns = [sample[index] / sample[index - 1] - 1.0 for index in range(1, len(sample))]
    return pstdev(returns) if len(returns) > 1 else ""


def _volatility_percentile(values: list[float]) -> float | str:
    if len(values) < 272:
        return ""
    series = []
    for end in range(len(values) - 252, len(values)):
        sample = values[end - 20 : end + 1]
        returns = [sample[index] / sample[index - 1] - 1.0 for index in range(1, len(sample))]
        series.append(pstdev(returns))
    return _percentile_rank(series[-1], series)


def _atr_percentile(history: list[dict[str, float | str]]) -> float | str:
    if len(history) < 266:
        return ""
    true_ranges = []
    for index, row in enumerate(history):
        high = float(row["high"])
        low = float(row["low"])
        previous_close = float(history[index - 1]["close"]) if index else float(row["close"])
        true_ranges.append(max(high - low, abs(high - previous_close), abs(low - previous_close)))
    normalized_atr = []
    for end in range(len(history) - 252, len(history)):
        if end < 13:
            continue
        atr = mean(true_ranges[end - 13 : end + 1])
        close = float(history[end]["close"])
        normalized_atr.append(atr / close if close > 0.0 else 0.0)
    return _percentile_rank(normalized_atr[-1], normalized_atr) if normalized_atr else ""


def _slope(x: tuple[float, ...], y: tuple[Any, ...]) -> float | str:
    numbers = [_number(value) for value in y]
    if any(value is None for value in numbers):
        return ""
    x_mean = mean(x)
    y_mean = mean(float(value) for value in numbers if value is not None)
    denominator = sum((value - x_mean) ** 2 for value in x)
    return sum(
        (a - x_mean) * (float(b) - y_mean)
        for a, b in zip(x, numbers)
        if b is not None
    ) / denominator


def _percentile_rank(value: float | None, values: list[float]) -> float | str:
    if value is None or not values:
        return ""
    less = sum(item < value for item in values)
    equal = sum(item == value for item in values)
    return (less + 0.5 * equal) / len(values)


def _ratio_minus_one(left: Any, right: Any) -> float | str:
    left_number = _number(left)
    right_number = _number(right)
    if left_number is None or right_number is None or right_number <= 0.0:
        return ""
    return left_number / right_number - 1.0


def _difference(left: Any, right: Any) -> float | str:
    left_number = _number(left)
    right_number = _number(right)
    return left_number - right_number if left_number is not None and right_number is not None else ""


def _volatility_regime(percentile: Any) -> int | str:
    value = _number(percentile)
    if value is None:
        return ""
    return 0 if value < 1.0 / 3.0 else 2 if value > 2.0 / 3.0 else 1


def _number(value: Any) -> float | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _audit(
    source_rows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    histories: dict[str, list[dict[str, float | str]]],
    source_path: str | None,
) -> dict[str, Any]:
    features = []
    for feature in ENGINEERED_FEATURE_COLUMNS:
        populated = sum(_number(row.get(feature)) is not None for row in rows)
        features.append(
            {
                "feature": feature,
                "definition": FEATURE_DEFINITIONS[feature],
                "populated_count": populated,
                "missing_count": len(rows) - populated,
                "availability_rate": populated / len(rows) if rows else 0.0,
            }
        )
    source_columns = list(source_rows[0]) if source_rows else []
    return {
        "mode": "stock_level_alpha_features_research_only",
        "source_path": source_path,
        "output_policy": "Write a sibling enriched artifact; never overwrite the source CSV.",
        "row_count": len(rows),
        "source_column_count": len(source_columns),
        "engineered_feature_count": len(ENGINEERED_FEATURE_COLUMNS),
        "source_columns_preserved": all(
            all(row.get(column) == source.get(column) for column in source_columns)
            for source, row in zip(source_rows, rows)
        ),
        "unique_symbol_date_rows": len(
            {(row.get("rebalance_date"), row.get("symbol")) for row in rows}
        )
        == len(rows),
        "price_history_symbol_count": sum(bool(history) for history in histories.values()),
        "industry_metadata_available": any(str(row.get("industry", "")).strip() for row in rows),
        "features": features,
        **RESEARCH_METADATA,
    }


def _load_price_histories(
    parquet_dir: Path,
    symbols: list[str],
) -> dict[str, list[dict[str, Any]]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Stock-level alpha feature generation requires pyarrow. "
            "Install project requirements before running this research command."
        ) from exc
    output = {}
    for symbol in symbols:
        path = parquet_dir / f"{symbol}.parquet"
        if not path.exists():
            output[symbol] = []
            continue
        table = pq.read_table(path, columns=["timestamp", "high", "low", "close"])
        data = table.to_pydict()
        output[symbol] = [
            {
                "date": value.date().isoformat() if hasattr(value, "date") else str(value)[:10],
                "high": high,
                "low": low,
                "close": close,
            }
            for value, high, low, close in zip(
                data["timestamp"], data["high"], data["low"], data["close"]
            )
        ]
    return output


def _output_dir(config: dict[str, Any]) -> Path:
    return Path(config.get("ml", {}).get("output_dir", "reports/ml"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_enriched_csv(
    path: Path,
    source_rows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> None:
    source_columns = list(source_rows[0]) if source_rows else []
    fieldnames = [*source_columns, *ENGINEERED_FEATURE_COLUMNS]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_audit_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "feature",
        "definition",
        "populated_count",
        "missing_count",
        "availability_rate",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Stock-Level Alpha Feature Audit",
        "",
        NOTICE,
        "",
        f"- Rows: {audit['row_count']}",
        f"- Engineered features: {audit['engineered_feature_count']}",
        f"- Source columns preserved: {audit['source_columns_preserved']}",
        f"- Unique symbol/date rows: {audit['unique_symbol_date_rows']}",
        f"- Industry metadata available: {audit['industry_metadata_available']}",
        "- Promotion thresholds changed: false",
        "",
        "| Feature | Populated | Missing | Availability | Definition |",
        "|---|---:|---:|---:|---|",
    ]
    for row in audit["features"]:
        lines.append(
            f"| {row['feature']} | {row['populated_count']} | {row['missing_count']} | "
            f"{row['availability_rate']:.4f} | {row['definition']} |"
        )
    lines.append("")
    return "\n".join(lines)
