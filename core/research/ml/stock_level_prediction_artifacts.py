from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import yaml

from core.research.ml.sector_reference import load_sector_by_symbol
from core.research.framework.data import CsvRowRepository
from core.research.framework.reporting import ResearchArtifactWriter


RESEARCH_METADATA = {
    "research_only": True,
    "trading_impact": "none",
    "production_validated": False,
}
NOTICE = "Research only. Trading impact: none. Production validated: false."
PREDICTION_COLUMNS = (
    "predicted_probability",
    "predicted_forward_return_5d",
    "predicted_forward_return_10d",
    "predicted_future_volatility",
    "predicted_future_drawdown",
    "predicted_max_adverse_excursion",
    "predicted_momentum_20d",
    "predicted_momentum_60d",
    "predicted_momentum_120d",
    "predicted_volatility_20d",
    "predicted_drawdown_60d",
    "predicted_liquidity_score",
    "predicted_risk_adjusted_momentum",
)
BASELINE_PREDICTION_COLUMNS = (
    "predicted_momentum_20d",
    "predicted_momentum_60d",
    "predicted_momentum_120d",
    "predicted_volatility_20d",
    "predicted_drawdown_60d",
    "predicted_liquidity_score",
    "predicted_risk_adjusted_momentum",
)
ACTUAL_COLUMNS = (
    "actual_forward_return_5d",
    "actual_forward_return_10d",
    "actual_future_volatility",
    "actual_future_drawdown",
    "actual_max_adverse_excursion",
)
CONTEXT_COLUMNS = (
    "breadth_above_sma_200",
    "spy_realized_volatility_21d",
    "spy_realized_volatility_63d",
    "spy_max_drawdown_63d",
    "spy_max_drawdown_126d",
)


@dataclass(frozen=True)
class StockLevelPredictionArtifactsPaths:
    csv_path: Path
    json_path: Path
    markdown_path: Path


def write_stock_level_prediction_artifacts(
    config: dict[str, Any],
) -> StockLevelPredictionArtifactsPaths:
    output_dir = _output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    expanded_rows = _read_csv(_expanded_dataset_path(config))
    meta_rows = _read_csv(output_dir / "meta_auxiliary_predictions.csv")
    sector_by_symbol = load_sector_by_symbol(
        config.get("ml", {}).get("sector_reference_path"),
        inline_mapping=dict(config.get("ml", {}).get("sector_by_symbol", {})),
    )
    rows, audit = build_stock_level_prediction_artifacts(
        expanded_rows=expanded_rows,
        artifact_rows=meta_rows,
        universe_symbols=_universe_symbols(config),
        closes_by_symbol=_load_closes_by_symbol(config),
        sector_by_symbol=sector_by_symbol,
    )
    paths = StockLevelPredictionArtifactsPaths(
        csv_path=output_dir / "stock_level_prediction_artifacts.csv",
        json_path=output_dir / "stock_level_prediction_artifacts.json",
        markdown_path=output_dir / "stock_level_prediction_artifacts.md",
    )
    _write_csv(paths.csv_path, rows)
    writer = ResearchArtifactWriter()
    writer.write_json(paths.json_path, audit)
    writer.write_markdown(paths.markdown_path, _markdown(audit))
    return paths


def build_stock_level_prediction_artifacts(
    *,
    expanded_rows: list[dict[str, str]],
    artifact_rows: list[dict[str, str]],
    universe_symbols: list[str],
    closes_by_symbol: dict[str, dict[str, dict[str, float]]],
    sector_by_symbol: dict[str, str] | None = None,
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
            row.update(_actual_targets(symbol_data, date))
            for column in CONTEXT_COLUMNS:
                row[column] = context.get(column, "")
            rows.append(row)
    audit = _audit(rows, symbols, dates, artifact_rows)
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


def _history_values_before(
    dates: list[str],
    values: list[float],
    rebalance_date: str,
) -> list[float]:
    index = bisect_left(dates, rebalance_date)
    return values[:index]


def _trailing_return(
    dates: list[str],
    values: list[float],
    rebalance_date: str,
    *,
    lookback: int,
) -> float | str:
    history = _history_values_before(dates, values, rebalance_date)
    if len(history) <= lookback:
        return ""
    end = history[-1]
    start = history[-lookback - 1]
    return (end / start) - 1.0 if start > 0.0 else ""


def _trailing_volatility(
    dates: list[str],
    values: list[float],
    rebalance_date: str,
    *,
    lookback: int,
) -> float | str:
    history = _history_values_before(dates, values, rebalance_date)
    if len(history) <= lookback:
        return ""
    prices = history[-lookback - 1 :]
    returns = [
        (prices[index] / prices[index - 1]) - 1.0
        for index in range(1, len(prices))
        if prices[index - 1] > 0.0
    ]
    return pstdev(returns) if len(returns) > 1 else ""


def _trailing_drawdown(
    dates: list[str],
    values: list[float],
    rebalance_date: str,
    *,
    lookback: int,
) -> float | str:
    history = _history_values_before(dates, values, rebalance_date)
    if len(history) < lookback:
        return ""
    values = history[-lookback:]
    peak = values[0]
    worst = 0.0
    for value in values:
        peak = max(peak, value)
        worst = min(worst, (value / peak) - 1.0 if peak > 0.0 else 0.0)
    return worst


def _trailing_liquidity_score(
    dates: list[str],
    values_by_index: list[float],
    rebalance_date: str,
    *,
    lookback: int,
) -> float | str:
    history = _history_values_before(dates, values_by_index, rebalance_date)
    if not history:
        return ""
    values = history[-lookback:]
    return math.log1p(mean(values)) if values else ""


def _actual_targets(symbol_data: dict[str, Any], rebalance_date: str) -> dict[str, Any]:
    closes_by_date = symbol_data.get("close", {})
    dates = symbol_data.get("close_dates", [])
    if rebalance_date not in closes_by_date:
        return {column: "" for column in ACTUAL_COLUMNS}
    index = dates.index(rebalance_date)
    start = closes_by_date[rebalance_date]
    if start <= 0.0:
        return {column: "" for column in ACTUAL_COLUMNS}
    forward = []
    for horizon in (5, 10):
        end_index = index + horizon
        if end_index < len(dates):
            forward.append((horizon, closes_by_date[dates[end_index]] / start - 1.0))
        else:
            forward.append((horizon, ""))
    future_prices = [
        closes_by_date[date]
        for date in dates[index + 1 : index + 11]
        if closes_by_date[date] > 0.0
    ]
    returns = [
        (future_prices[i] / future_prices[i - 1]) - 1.0
        for i in range(1, len(future_prices))
        if future_prices[i - 1] > 0.0
    ]
    drawdowns = [(price / start) - 1.0 for price in future_prices]
    return {
        "actual_forward_return_5d": forward[0][1],
        "actual_forward_return_10d": forward[1][1],
        "actual_future_volatility": pstdev(returns) if len(returns) > 1 else "",
        "actual_future_drawdown": min(drawdowns) if drawdowns else "",
        "actual_max_adverse_excursion": min(drawdowns) if drawdowns else "",
    }


def _average_dollar_volume(
    dates: list[str],
    values_by_index: list[float],
    rebalance_date: str,
    *,
    lookback: int,
) -> float | str:
    index = bisect_left(dates, rebalance_date)
    if index >= len(dates) or dates[index] != rebalance_date:
        return ""
    values = [
        value
        for value in values_by_index[max(0, index - lookback + 1) : index + 1]
        if value > 0.0
    ]
    return mean(values) if values else ""


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


def _universe_symbols(config: dict[str, Any]) -> list[str]:
    ml_config = config.get("ml", {})
    expanded_config = ml_config.get("expanded_rebalance_dataset", {}) or {}
    paths = expanded_config.get("universe_paths") or [
        "data/reference/universes/current_32.yaml"
    ]
    symbols: list[str] = []
    for raw_path in paths:
        path = Path(str(raw_path))
        if not path.exists():
            continue
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        symbols.extend(str(symbol).upper() for symbol in payload.get("symbols", []))
    max_symbols = expanded_config.get("max_symbols")
    unique = list(dict.fromkeys(symbols))
    return unique[: int(max_symbols)] if max_symbols else unique


def _load_closes_by_symbol(config: dict[str, Any]) -> dict[str, dict[str, dict[str, float]]]:
    parquet_dir = Path(
        str(config.get("ml", {}).get("stooq_parquet_dir", "data/processed/stooq_parquet"))
    )
    closes = {}
    for symbol in _universe_symbols(config):
        path = parquet_dir / f"{symbol.upper()}.parquet"
        if path.exists():
            closes[symbol.upper()] = _read_parquet_closes(path)
    return closes


def _read_parquet_closes(path: Path) -> dict[str, dict[str, float]]:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return {}
    table = pq.read_table(path, columns=["timestamp", "close", "volume"])
    data = table.to_pydict()
    closes = {}
    dollar_volume = {}
    for value, close, volume in zip(data["timestamp"], data["close"], data["volume"]):
        if close is None or not math.isfinite(float(close)):
            continue
        date = value.date().isoformat() if hasattr(value, "date") else str(value)[:10]
        closes[date] = float(close)
        if volume is not None and math.isfinite(float(volume)):
            dollar_volume[date] = float(close) * float(volume)
    return {"close": closes, "dollar_volume": dollar_volume}


def _output_dir(config: dict[str, Any]) -> Path:
    return Path(
        config.get("ml", {}).get(
            "output_dir",
            Path(config.get("reports", {}).get("ml_dir", "reports/ml"))
            / "regime_transformer_meta_ensemble_v1",
        )
    )


def _expanded_dataset_path(config: dict[str, Any]) -> Path:
    return Path(
        config.get("ml", {}).get(
            "expanded_rebalance_dataset_path",
            Path(config.get("cache", {}).get("ml_dir", "cache/ml"))
            / "expanded_rebalance_dataset.csv",
        )
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    return CsvRowRepository().read(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "rebalance_date",
        "symbol",
        "sector",
        "average_dollar_volume_21d",
        "average_dollar_volume_63d",
        *PREDICTION_COLUMNS,
        *ACTUAL_COLUMNS,
        *CONTEXT_COLUMNS,
        "source",
        "source_feature_id",
        "source_model_type",
        "source_split",
        "source_dataset_hash",
        "true_stock_level_row",
    ]
    normalized = [
        {name: row.get(name, "") for name in fieldnames}
        for row in rows
    ]
    ResearchArtifactWriter().write_csv(path, normalized, fieldnames=fieldnames)


def _markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Stock-Level Prediction Artifacts",
        "",
        NOTICE,
        "",
        f"- Rows: {audit['row_count']}",
        f"- Symbols: {audit['symbol_count']}",
        f"- Rebalance dates: {audit['rebalance_date_count']}",
        f"- Date range: {audit['date_range']}",
        f"- Average symbols per date: {audit['average_symbols_per_date']:.2f}",
        f"- True stock-level rows: {audit['true_stock_level_rows']}",
        f"- Usable for stock-level ranking: {audit['usable_for_stock_level_ranking']}",
        f"- Suitable for true stock-level ranking diagnostics: {audit['suitable_for_true_stock_level_ranking_diagnostics']}",
        f"- Suitability reason: {audit['suitability_reason']}",
        "",
        "## Populated Predictions",
        "",
    ]
    for column, count in audit["populated_prediction_counts"].items():
        lines.append(f"- {column}: {count}")
    lines.extend([
        "",
        "## Missing Predictions",
        "",
    ])
    for column, count in audit["missing_prediction_counts"].items():
        lines.append(f"- {column}: {count}")
    lines.extend(["", "## Missing Actual Targets", ""])
    for column, count in audit["missing_actual_target_counts"].items():
        lines.append(f"- {column}: {count}")
    lines.extend([
        "",
        "## Root Cause",
        "",
        audit["root_cause_artifact_level_limitation"],
        "",
    ])
    return "\n".join(lines)
