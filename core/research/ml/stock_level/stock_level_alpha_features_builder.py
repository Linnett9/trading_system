from __future__ import annotations

import time
from bisect import bisect_left
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from statistics import mean, median, pstdev
from typing import Any

from core.research.ml.reference.market_information_availability_authority import (
    MarketInformationAvailabilityAuthority,
    any_timestamp_session_policy,
    daily_price_feature_event,
)
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
from core.research.ml.stock_level.stock_level_alpha_features_types import (
    BREADTH_CONTRACT_VERSION,
    INDUSTRY_MAPPING_CONTRACT_VERSION,
    MARKET_CONTEXT_CONTRACT_VERSION,
)
from core.research.ml.stock_level.selector_lineage import (
    merge_enrichment_preserving_base,
)


DAILY_PRICE_AVAILABILITY_AUTHORITY = MarketInformationAvailabilityAuthority(
    session_policy=any_timestamp_session_policy("daily_price_feature_cutoff_decision_timestamp_policy_v1")
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
    phase_timings: list[dict[str, Any]] = []
    phase_started, phase_start_ts = _phase_start()
    prepared_histories = {
        symbol.upper(): _prepare_history(history)
        for symbol, history in price_histories.items()
    }
    _record_phase(
        phase_timings,
        "symbol-data preparation",
        phase_started,
        phase_start_ts,
        requested_workers=1,
        effective_workers=1,
        task_count=len(price_histories),
    )
    spy_history = prepared_histories.get(spy_symbol.upper(), [])
    phase_started, phase_start_ts = _phase_start()
    enriched_rows = _build_symbol_level_features(
        rows,
        prepared_histories,
        spy_history,
        n_jobs=n_jobs,
        executor_cls=executor_cls or ProcessPoolExecutor,
    )
    symbol_count = len({str(row.get("symbol", "")).upper() for row in rows if row.get("symbol")})
    _record_phase(
        phase_timings,
        "symbol-task execution",
        phase_started,
        phase_start_ts,
        requested_workers=n_jobs,
        effective_workers=min(n_jobs, symbol_count) if symbol_count else 1,
        task_count=symbol_count,
        execution_mode="serial" if n_jobs == 1 or symbol_count <= 1 else "process_pool",
    )
    phase_started, phase_start_ts = _phase_start()
    enriched_rows.sort(
        key=lambda row: (
            str(row.get("rebalance_date", "")),
            str(row.get("symbol", "")).upper(),
        )
    )
    _record_phase(
        phase_timings,
        "deterministic sorting",
        phase_started,
        phase_start_ts,
        requested_workers=1,
        effective_workers=1,
        task_count=len(enriched_rows),
    )
    phase_started, phase_start_ts = _phase_start()
    _add_cross_sectional_features(enriched_rows)
    enriched_rows = merge_enrichment_preserving_base(rows, enriched_rows)
    _record_phase(
        phase_timings,
        "cross-sectional calculation",
        phase_started,
        phase_start_ts,
        requested_workers=1,
        effective_workers=1,
        task_count=len({str(row.get("rebalance_date", "")) for row in enriched_rows}),
    )
    audit = _audit(rows, enriched_rows, prepared_histories, source_path, n_jobs)
    audit["parallelism"].update({"requested_workers": n_jobs, "effective_workers": min(n_jobs, symbol_count), "symbol_count": symbol_count, "elapsed_seconds": time.perf_counter() - started})
    audit["phase_timings"] = phase_timings
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
        row.update(_daily_price_availability_metadata(history_before, source))
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
    market_momentum_20 = _trailing_return(spy_closes, 20)
    market_momentum_60 = _trailing_return(spy_closes, 60)
    market_momentum_120 = _trailing_return(spy_closes, 120)
    market_volatility_20 = _volatility(spy_closes, 20)
    market_distance_200 = _distance_from_average(spy_closes, 200)
    market_context_source_date = spy_history[-1]["date"] if spy_history else ""
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
        "market_momentum_20d": market_momentum_20,
        "market_momentum_60d": market_momentum_60,
        "market_momentum_120d": market_momentum_120,
        "market_volatility_20d": market_volatility_20,
        "market_drawdown_60d": _max_drawdown(spy_closes, 60),
        "market_distance_from_200d_average": market_distance_200,
        "market_trend_state": _market_trend_state(market_distance_200, market_momentum_60),
        "market_volatility_percentile": _volatility_percentile(spy_closes),
        "market_context_source_date": market_context_source_date,
        "market_context_availability_timestamp": market_context_source_date,
        "market_context_status": "available" if market_context_source_date else "missing_benchmark_history",
        "market_context_contract_identity": MARKET_CONTEXT_CONTRACT_VERSION,
        "_stock_above_200d_average": _above_average(closes, 200),
        "breadth_positive_momentum_20d": "",
        "breadth_positive_momentum_60d": "",
        "breadth_above_long_term_trend": "",
        "breadth_cross_sectional_median_return": "",
        "breadth_return_dispersion": "",
        "breadth_advance_decline_ratio": "",
        "breadth_coverage": "",
        "breadth_eligible_symbol_count": "",
        "breadth_observed_symbol_count": "",
        "breadth_contract_identity": BREADTH_CONTRACT_VERSION,
        "industry_id": "",
        "industry_mapping_identity": INDUSTRY_MAPPING_CONTRACT_VERSION,
        "industry_peer_count": "",
        "industry_mapping_available": "",
        "industry_relative_status": "",
        "relative_momentum_vs_industry": "",
        "industry_momentum_percentile": "",
        "_stock_momentum_20d": momentum_20,
        "_stock_momentum_60d": momentum_60,
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
            difference_column="relative_momentum_vs_industry",
            percentile_column="industry_relative_strength",
            extra_percentile_column="industry_momentum_percentile",
        )
        _add_breadth_features(date_rows)
def _add_group_relative_features(
    rows: list[dict[str, Any]],
    momentum: dict[int, float | None],
    *,
    group_column: str,
    difference_column: str | None,
    percentile_column: str,
    extra_percentile_column: str | None = None,
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
        if group_column == "industry":
            row["industry_id"] = group
            row["industry_mapping_available"] = 1.0 if group else 0.0
            row["industry_peer_count"] = len(values)
            row["industry_mapping_identity"] = INDUSTRY_MAPPING_CONTRACT_VERSION
        if not group or value is None or not values:
            if difference_column:
                row[difference_column] = ""
            row[percentile_column] = ""
            if extra_percentile_column:
                row[extra_percentile_column] = ""
            if group_column == "industry":
                row["industry_relative_status"] = "missing_industry_or_momentum"
            continue
        if difference_column:
            row[difference_column] = value - mean(values)
        row[percentile_column] = _percentile_rank(value, values)
        if extra_percentile_column:
            row[extra_percentile_column] = _percentile_rank(value, values)
        if group_column == "industry":
            row["industry_relative_status"] = "available" if len(values) >= 2 else "single_peer_neutral"


def _add_breadth_features(rows: list[dict[str, Any]]) -> None:
    eligible = len(rows)
    momentum_20 = [
        _number(row.get("predicted_momentum_20d"))
        if _number(row.get("predicted_momentum_20d")) is not None
        else _number(row.get("_stock_momentum_20d"))
        for row in rows
    ]
    momentum_60 = [
        _number(row.get("predicted_momentum_60d"))
        if _number(row.get("predicted_momentum_60d")) is not None
        else _number(row.get("_stock_momentum_60d"))
        for row in rows
    ]
    above_trend = [_number(row.get("_stock_above_200d_average")) for row in rows]
    observed_20 = [value for value in momentum_20 if value is not None]
    observed_60 = [value for value in momentum_60 if value is not None]
    observed_trend = [value for value in above_trend if value is not None]
    positives = sum(value > 0.0 for value in observed_20)
    negatives = sum(value < 0.0 for value in observed_20)
    breadth = {
        "breadth_positive_momentum_20d": _fraction_positive(observed_20),
        "breadth_positive_momentum_60d": _fraction_positive(observed_60),
        "breadth_above_long_term_trend": mean(observed_trend) if observed_trend else "",
        "breadth_cross_sectional_median_return": median(observed_20) if observed_20 else "",
        "breadth_return_dispersion": pstdev(observed_20) if len(observed_20) >= 2 else 0.0 if observed_20 else "",
        "breadth_advance_decline_ratio": positives / negatives if negatives else float(positives) if positives else 0.0,
        "breadth_coverage": len(observed_20) / eligible if eligible else "",
        "breadth_eligible_symbol_count": eligible,
        "breadth_observed_symbol_count": len(observed_20),
        "breadth_contract_identity": BREADTH_CONTRACT_VERSION,
    }
    for row in rows:
        row.update(breadth)
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


def _daily_price_availability_metadata(
    history: list[dict[str, float | str]],
    source: dict[str, Any],
) -> dict[str, Any]:
    rebalance_date = str(source.get("rebalance_date", ""))
    decision = source.get("decision_timestamp") or f"{rebalance_date}T00:00:00Z"
    observation = history[-1]["date"] if history else None
    result = DAILY_PRICE_AVAILABILITY_AUTHORITY.evaluate(
        daily_price_feature_event(
            observation_timestamp=observation,
            decision_timestamp=decision,
            source_version="stock_level_daily_price_feature_cutoff_v1",
        )
    )
    return {
        "daily_price_availability_status": result["status"],
        "daily_price_earliest_permitted_use": result["earliest_permitted_use"] or "",
        "daily_price_availability_authority_version": result["authority_version"],
    }


def _distance_from_average(values: list[float], lookback: int) -> float | str:
    if len(values) < lookback:
        return ""
    average = mean(values[-lookback:])
    return values[-1] / average - 1.0 if average else ""


def _above_average(values: list[float], lookback: int) -> float | str:
    distance = _distance_from_average(values, lookback)
    return 1.0 if isinstance(distance, float) and distance > 0.0 else 0.0 if isinstance(distance, float) else ""


def _market_trend_state(distance_from_average: Any, momentum_60: Any) -> float | str:
    distance = _number(distance_from_average)
    momentum = _number(momentum_60)
    if distance is None or momentum is None:
        return ""
    return 1.0 if distance > 0.0 and momentum > 0.0 else 0.0


def _fraction_positive(values: list[float]) -> float | str:
    return sum(value > 0.0 for value in values) / len(values) if values else ""


def _phase_start() -> tuple[float, str]:
    return time.perf_counter(), datetime.now(timezone.utc).isoformat()


def _record_phase(
    timings: list[dict[str, Any]],
    phase_name: str,
    started: float,
    start_timestamp: str,
    *,
    requested_workers: int,
    effective_workers: int,
    task_count: int | None = None,
    execution_mode: str = "serial",
) -> None:
    timings.append(
        {
            "phase_name": phase_name,
            "start_timestamp": start_timestamp,
            "end_timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": max(0.0, time.perf_counter() - started),
            "requested_workers": requested_workers,
            "effective_workers": effective_workers,
            "task_count": task_count,
            "execution_mode": execution_mode,
        }
    )
