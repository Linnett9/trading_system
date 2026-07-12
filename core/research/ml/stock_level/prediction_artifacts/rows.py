from __future__ import annotations

import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from typing import Any

from core.research.ml.runtime_parallelism import apply_worker_thread_environment
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
from core.research.ml.stock_level.prediction_artifacts.decision_grid import (
    resolve_decision_grid,
)
from core.research.ml.stock_level.prediction_artifacts.types import (
    ACTUAL_COLUMNS,
    BASELINE_PREDICTION_COLUMNS,
    DECISION_CONTEXT_COLUMNS,
    CONTEXT_COLUMNS,
    PREDICTION_COLUMNS,
    RESEARCH_METADATA,
    TARGET_PROVENANCE_COLUMNS,
    TARGET_PROVENANCE_CONTRACT_VERSION,
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
    dataset_workers: int = 1,
    inner_thread_limit: int = 1,
    decision_grid_frequency: str = "source",
    decision_grid_start_date: str | None = None,
    decision_grid_end_date: str | None = None,
    decision_grid_max_sessions: int | None = None,
    decision_grid_min_history_sessions: int = 1,
    executor_cls: type | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if dataset_workers < 1:
        raise ValueError("stock_level_dataset_workers must be at least one")
    if inner_thread_limit < 1:
        raise ValueError("stock_level_dataset_inner_threads must be at least one")
    started = time.perf_counter()
    phase_timings: list[dict[str, Any]] = []
    sector_by_symbol = sector_by_symbol or {}
    phase_started, phase_start_ts = _phase_start()
    artifact_by_date_symbol = _artifact_by_date_symbol(artifact_rows)
    symbols = sorted({symbol.upper() for symbol in universe_symbols if symbol})
    _record_phase(
        phase_timings,
        "context alignment",
        phase_started,
        phase_start_ts,
        requested_workers=1,
        effective_workers=1,
        task_count=len(artifact_rows),
    )
    phase_started, phase_start_ts = _phase_start()
    prepared_symbol_data = {
        symbol: _prepare_symbol_data(closes_by_symbol.get(symbol, {}))
        for symbol in symbols
    }
    market_symbol = market_symbol.upper()
    market_data = _prepare_symbol_data(closes_by_symbol.get(market_symbol, {}))
    _record_phase(
        phase_timings,
        "symbol-data preparation",
        phase_started,
        phase_start_ts,
        requested_workers=1,
        effective_workers=1,
        task_count=len(symbols),
    )
    phase_started, phase_start_ts = _phase_start()
    decision_grid = resolve_decision_grid(
        expanded_rows=expanded_rows,
        artifact_rows=artifact_rows,
        symbols=symbols,
        prepared_symbol_data=prepared_symbol_data,
        market_data=market_data,
        frequency=decision_grid_frequency,
        start_date=decision_grid_start_date,
        end_date=decision_grid_end_date,
        max_sessions=decision_grid_max_sessions,
        min_history_sessions=decision_grid_min_history_sessions,
    )
    _record_phase(
        phase_timings,
        "daily-grid construction",
        phase_started,
        phase_start_ts,
        requested_workers=1,
        effective_workers=1,
        task_count=len(symbols),
    )
    dates = decision_grid.dates
    rows, parallelism = _build_dataset_symbol_rows(
        symbols=symbols,
        dates=dates,
        row_metadata_by_date=decision_grid.row_metadata_by_date,
        artifact_by_date_symbol=artifact_by_date_symbol,
        prepared_symbol_data=prepared_symbol_data,
        market_symbol=market_symbol,
        market_data=market_data,
        sector_by_symbol=sector_by_symbol,
        dataset_workers=dataset_workers,
        inner_thread_limit=inner_thread_limit,
        executor_cls=executor_cls or ProcessPoolExecutor,
        phase_timings=phase_timings,
    )
    phase_started, phase_start_ts = _phase_start()
    _add_cross_sectional_targets(rows)
    _record_phase(
        phase_timings,
        "cross-sectional calculation",
        phase_started,
        phase_start_ts,
        requested_workers=1,
        effective_workers=1,
        task_count=len(dates),
    )
    phase_started, phase_start_ts = _phase_start()
    audit = _audit(rows, symbols, dates, artifact_rows)
    _record_phase(
        phase_timings,
        "base validation",
        phase_started,
        phase_start_ts,
        requested_workers=1,
        effective_workers=1,
        task_count=len(rows),
    )
    audit["decision_grid"] = decision_grid.audit
    audit.update(decision_grid.audit)
    audit["dataset_parallelism"] = {
        **parallelism,
            "elapsed_seconds": time.perf_counter() - started,
    }
    audit["phase_timings"] = phase_timings
    audit["market_residual_label_generation"] = {
        "market_symbol": market_symbol,
        "benchmark_symbol": market_symbol,
        "benchmark_return_column": "actual_benchmark_return_10d",
        "benchmark_return_horizon_trading_days": 10,
        "benchmark_return_convention": "simple_close_to_close",
        "market_symbol_loaded": bool(market_data.get("close_dates")),
        "market_symbol_is_tradable_candidate": market_symbol in symbols,
        "computed_before_dev_symbol_filtering": True,
    }
    return rows, audit


def _build_dataset_symbol_rows(
    *,
    symbols: list[str],
    dates: list[str],
    row_metadata_by_date: dict[str, dict[str, Any]],
    artifact_by_date_symbol: dict[tuple[str, str], dict[str, str]],
    prepared_symbol_data: dict[str, dict[str, Any]],
    market_symbol: str,
    market_data: dict[str, Any],
    sector_by_symbol: dict[str, str],
    dataset_workers: int,
    inner_thread_limit: int,
    executor_cls: type,
    phase_timings: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    phase_started, phase_start_ts = _phase_start()
    tasks = [
        {
            "symbol": symbol,
            "dates": dates,
            "row_metadata_by_date": row_metadata_by_date,
            "artifact_by_date_symbol": {
                date: artifact_by_date_symbol.get((date, symbol), {})
                for date in dates
            },
            "symbol_data": prepared_symbol_data.get(symbol, {}),
            "market_symbol": market_symbol,
            "market_data": market_data,
            "sector": sector_by_symbol.get(symbol, ""),
            "inner_thread_limit": inner_thread_limit,
        }
        for symbol in symbols
    ]
    task_count = len(tasks)
    effective_workers = min(dataset_workers, task_count) if task_count else 1
    execution_mode = "serial" if effective_workers <= 1 else "process_pool"
    _record_phase(
        phase_timings,
        "symbol-task dispatch",
        phase_started,
        phase_start_ts,
        requested_workers=dataset_workers,
        effective_workers=effective_workers,
        task_count=task_count,
        execution_mode=execution_mode,
    )
    metadata = {
        "parallelism_owner": "stock_level_prediction_artifacts_symbol_tasks",
        "requested_workers": dataset_workers,
        "effective_workers": effective_workers,
        "task_count": task_count,
        "completed_task_count": 0,
        "failed_task_count": 0,
        "inner_thread_limit": inner_thread_limit,
        "nested_parallelism_prevented": inner_thread_limit == 1,
        "worker_execution_mode": execution_mode,
        "worker_process_ids": [],
        "task_symbols": list(symbols),
    }
    if not tasks:
        metadata["completed_task_count"] = 0
        return [], metadata
    results: list[dict[str, Any]] = []
    try:
        phase_started, phase_start_ts = _phase_start()
        if effective_workers <= 1:
            results = [_build_dataset_symbol_task(task) for task in tasks]
        else:
            with executor_cls(max_workers=effective_workers) as executor:
                results = list(executor.map(_build_dataset_symbol_task, tasks))
        _record_phase(
            phase_timings,
            "symbol-task execution",
            phase_started,
            phase_start_ts,
            requested_workers=dataset_workers,
            effective_workers=effective_workers,
            task_count=task_count,
            execution_mode=execution_mode,
        )
    except Exception:
        metadata["failed_task_count"] = task_count - len(results)
        raise
    phase_started, phase_start_ts = _phase_start()
    metadata["completed_task_count"] = len(results)
    metadata["worker_process_ids"] = sorted({
        int(result["worker_process_id"]) for result in results if result.get("worker_process_id")
    })
    _record_phase(
        phase_timings,
        "worker result collection",
        phase_started,
        phase_start_ts,
        requested_workers=dataset_workers,
        effective_workers=effective_workers,
        task_count=len(results),
        execution_mode="coordinator",
    )
    phase_started, phase_start_ts = _phase_start()
    rows = _validate_and_merge_symbol_results(
        results,
        expected_symbols=symbols,
        expected_dates=dates,
    )
    _record_phase(
        phase_timings,
        "deterministic sorting",
        phase_started,
        phase_start_ts,
        requested_workers=1,
        effective_workers=1,
        task_count=len(rows),
    )
    return rows, metadata


def _build_dataset_symbol_task(task: dict[str, Any]) -> dict[str, Any]:
    apply_worker_thread_environment(int(task.get("inner_thread_limit", 1)))
    symbol = str(task["symbol"]).upper()
    symbol_data = task.get("symbol_data", {})
    market_data = task.get("market_data", {})
    dates = list(task.get("dates", []))
    rows = []
    for date in dates:
        metadata = task.get("row_metadata_by_date", {}).get(date, {})
        source = task.get("artifact_by_date_symbol", {}).get(date, {})
        row = {
            "rebalance_date": date,
            "symbol": symbol,
            "sector": task.get("sector", ""),
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
            "benchmark_symbol": task.get("market_symbol", "SPY"),
            "true_stock_level_row": True,
        }
        for column in PREDICTION_COLUMNS:
            row[column] = source.get(column, "")
        row.update(_baseline_predictions(symbol_data, date))
        row.update(
            _actual_targets(
                symbol_data,
                date,
                market_data=market_data,
                decision_dates=dates,
                decision_metadata={
                    **metadata,
                    "decision_frequency": metadata.get("decision_frequency", "daily" if metadata else "source"),
                    "target_horizon_trading_days": 10,
                    "overlapping_targets": metadata.get("overlapping_targets", True),
                    "required_purge_horizon_trading_days": 10,
                },
            )
        )
        for column in CONTEXT_COLUMNS:
            row[column] = metadata.get(column, "")
        for column in DECISION_CONTEXT_COLUMNS:
            row[column] = metadata.get(column, "")
        rows.append(row)
    return {
        "symbol": symbol,
        "rows": rows,
        "row_count": len(rows),
        "worker_process_id": os.getpid(),
    }


def _validate_and_merge_symbol_results(
    results: list[dict[str, Any]],
    *,
    expected_symbols: list[str],
    expected_dates: list[str],
) -> list[dict[str, Any]]:
    expected_symbol_set = set(expected_symbols)
    result_symbols = [str(result.get("symbol", "")).upper() for result in results]
    duplicate_symbols = sorted({
        symbol for symbol in result_symbols if result_symbols.count(symbol) > 1
    })
    missing_symbols = sorted(expected_symbol_set - set(result_symbols))
    extra_symbols = sorted(set(result_symbols) - expected_symbol_set)
    if duplicate_symbols or missing_symbols or extra_symbols:
        raise ValueError(
            "Invalid stock-level dataset worker results: "
            f"duplicates={duplicate_symbols} missing={missing_symbols} extra={extra_symbols}"
        )
    rows = [
        row
        for result in results
        for row in result.get("rows", [])
    ]
    expected_keys = {
        (date, symbol)
        for date in expected_dates
        for symbol in expected_symbols
    }
    keys = [
        (str(row.get("rebalance_date", "")), str(row.get("symbol", "")).upper())
        for row in rows
    ]
    duplicate_keys = sorted({key for key in keys if keys.count(key) > 1})
    missing_keys = sorted(expected_keys - set(keys))
    extra_keys = sorted(set(keys) - expected_keys)
    if duplicate_keys or missing_keys or extra_keys:
        raise ValueError(
            "Invalid stock-level dataset worker row keys: "
            f"duplicate_keys={duplicate_keys[:5]} missing_keys={missing_keys[:5]} extra_keys={extra_keys[:5]}"
        )
    rows.sort(
        key=lambda row: (
            str(row.get("decision_timestamp") or row.get("rebalance_date", "")),
            str(row.get("symbol", "")).upper(),
        )
    )
    return rows


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
        "close_index_by_date": {date: index for index, date in enumerate(close_dates)},
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
    provenance_complete = sum(
        1
        for row in rows
        if all(str(row.get(column, "")).strip() for column in TARGET_PROVENANCE_COLUMNS)
    )
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
        "target_provenance_contract_version": TARGET_PROVENANCE_CONTRACT_VERSION,
        "target_provenance_audit": {
            "complete_rows": provenance_complete,
            "missing_rows": len(rows) - provenance_complete,
            "required_columns": list(TARGET_PROVENANCE_COLUMNS),
        },
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
