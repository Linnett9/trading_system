from __future__ import annotations

import time
from bisect import bisect_left
from concurrent.futures import ProcessPoolExecutor
from statistics import mean
from typing import Any

from core.research.ml.stock_level.stock_level_alpha_features_audit import _audit
from core.research.ml.stock_level.stock_level_alpha_features_math import (
    _atr_percentile,
    _difference,
    _distance_from_high,
    _downside_deviation,
    _drawdown_recovery_days,
    _max_drawdown,
    _momentum_persistence,
    _number,
    _percentile_rank,
    _ratio_minus_one,
    _slope,
    _trailing_return,
    _trend_r_squared,
    _ulcer_index,
    _volatility,
    _volatility_percentile,
    _volatility_regime,
)


def build_stock_level_alpha_features(
    rows: list[dict[str, Any]],
    price_histories: dict[str, list[dict[str, Any]]],
    *,
    spy_symbol: str = "SPY",
    source_path: str | None = None,
    n_jobs: int = 1,
    executor_cls: type | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if n_jobs < 1:
        raise ValueError("stock_alpha_feature_n_jobs must be at least one")
    started = time.perf_counter()
    prepared_histories = {
        symbol.upper(): _prepare_history(history)
        for symbol, history in price_histories.items()
    }
    spy_history = prepared_histories.get(spy_symbol.upper(), [])
    enriched_rows = _build_symbol_level_features(
        rows,
        prepared_histories,
        spy_history,
        n_jobs=n_jobs,
        executor_cls=executor_cls or ProcessPoolExecutor,
    )
    enriched_rows.sort(
        key=lambda row: (
            str(row.get("rebalance_date", "")),
            str(row.get("symbol", "")).upper(),
        )
    )
    _add_cross_sectional_features(enriched_rows)
    audit = _audit(rows, enriched_rows, prepared_histories, source_path, n_jobs)
    symbol_count = len({str(row.get("symbol", "")).upper() for row in rows if row.get("symbol")})
    audit["parallelism"].update({"requested_workers": n_jobs, "effective_workers": min(n_jobs, symbol_count), "symbol_count": symbol_count, "elapsed_seconds": time.perf_counter() - started})
    return enriched_rows, audit
def _build_symbol_level_features(
    rows: list[dict[str, Any]],
    histories: dict[str, list[dict[str, float | str]]],
    spy_history: list[dict[str, float | str]],
    *,
    n_jobs: int,
    executor_cls: type,
) -> list[dict[str, Any]]:
    rows_by_symbol: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_symbol.setdefault(str(row.get("symbol", "")).upper(), []).append(row)
    symbols = sorted(rows_by_symbol)
    tasks = [
        (
            rows_by_symbol[symbol],
            histories.get(symbol, []),
            spy_history,
        )
        for symbol in symbols
    ]
    if n_jobs == 1 or len(tasks) <= 1:
        return [
            row
            for task in tasks
            for row in _build_symbol_rows(task)
        ]
    max_workers = min(n_jobs, len(tasks))
    with executor_cls(max_workers=max_workers) as executor:
        return [
            row
            for symbol_rows in executor.map(_build_symbol_rows, tasks)
            for row in symbol_rows
        ]
def _build_symbol_rows(
    task: tuple[
        list[dict[str, Any]],
        list[dict[str, float | str]],
        list[dict[str, float | str]],
    ],
) -> list[dict[str, Any]]:
    rows, history, spy_history = task
    output = []
    for source in rows:
        row = dict(source)
        rebalance_date = str(row.get("rebalance_date", ""))
        history_before = _history_before(history, rebalance_date)
        spy_before = _history_before(spy_history, rebalance_date)
        row.update(_time_series_features(history_before, spy_before))
        output.append(row)
    return output
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
