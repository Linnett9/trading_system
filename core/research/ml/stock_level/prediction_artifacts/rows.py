from __future__ import annotations

import math
import os
import json
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
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
    task_timeout_seconds: float | None = None,
    progress_interval_seconds: float = 30.0,
    diagnostics_path: Path | None = None,
    diagnostic_run_id: str | None = None,
    partition_dir: Path | None = None,
    resume_partitions: bool = True,
    partition_only: bool = False,
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
        task_timeout_seconds=task_timeout_seconds,
        progress_interval_seconds=progress_interval_seconds,
        diagnostics_path=diagnostics_path,
        diagnostic_run_id=diagnostic_run_id,
        partition_dir=partition_dir,
        resume_partitions=resume_partitions,
        partition_only=partition_only,
        phase_timings=phase_timings,
    )
    if partition_only:
        audit = {
            "row_count": sum(
                int(value)
                for value in parallelism.get("partition_row_counts", {}).values()
            ),
            "symbol_count": len(symbols),
            "rebalance_date_count": len(dates),
            "streaming_partition_consolidation": True,
        }
    else:
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
    task_timeout_seconds: float | None,
    progress_interval_seconds: float,
    diagnostics_path: Path | None,
    diagnostic_run_id: str | None,
    partition_dir: Path | None,
    resume_partitions: bool,
    partition_only: bool,
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
            "diagnostics_path": str(diagnostics_path) if diagnostics_path else "",
            "diagnostic_run_id": diagnostic_run_id or "",
            "partition_dir": str(partition_dir) if partition_dir else "",
            "resume_partitions": resume_partitions,
            "retain_rows": not partition_only,
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
        "task_timeout_seconds": task_timeout_seconds,
        "progress_interval_seconds": progress_interval_seconds,
        "last_progress_completed_task_count": 0,
        "last_progress_elapsed_seconds": 0.0,
        "diagnostics_path": str(diagnostics_path) if diagnostics_path else "",
        "diagnostic_run_id": diagnostic_run_id,
        "partition_dir": str(partition_dir) if partition_dir else "",
        "resume_partitions": bool(resume_partitions),
        "reused_partition_count": 0,
        "written_partition_count": 0,
    }
    if not tasks:
        metadata["completed_task_count"] = 0
        return [], metadata
    results: list[dict[str, Any]] = []
    pending_tasks = tasks
    if partition_dir is not None and resume_partitions:
        reused_results, pending_tasks = _load_reusable_partitions(
            tasks,
            partition_dir=partition_dir,
            expected_dates=dates,
            diagnostics_path=diagnostics_path,
            diagnostic_run_id=diagnostic_run_id,
            retain_rows=not partition_only,
        )
        results.extend(reused_results)
        metadata["reused_partition_count"] = len(reused_results)
    try:
        phase_started, phase_start_ts = _phase_start()
        for task in pending_tasks:
            _write_task_event(
                diagnostics_path,
                "dispatched",
                symbol=str(task["symbol"]),
                date_count=len(dates),
                rows_read=len(task.get("symbol_data", {}).get("close_dates", [])),
                worker_pid=None,
                diagnostic_run_id=diagnostic_run_id,
            )
        if not pending_tasks:
            pass
        elif effective_workers <= 1:
            results.extend([_build_dataset_symbol_task(task) for task in pending_tasks])
        else:
            print(
                "[stock-alpha] stock artifact symbol tasks dispatched "
                f"tasks={len(pending_tasks)} workers={effective_workers} "
                f"timeout_seconds={task_timeout_seconds or 'none'}",
                flush=True,
            )
            executor = executor_cls(max_workers=effective_workers)
            try:
                if not hasattr(executor, "submit"):
                    results.extend(list(executor.map(_build_dataset_symbol_task, pending_tasks)))
                else:
                    results.extend(_collect_symbol_task_futures(
                        executor,
                        pending_tasks,
                        task_timeout_seconds=task_timeout_seconds,
                        progress_interval_seconds=progress_interval_seconds,
                        metadata=metadata,
                        diagnostics_path=diagnostics_path,
                        diagnostic_run_id=diagnostic_run_id,
                    ))
                print(
                    "[stock-alpha] all symbol futures resolved "
                    f"successful_task_count={len(results)} "
                    f"failed_task_count={metadata['failed_task_count']} "
                    f"cancelled_task_count={metadata.get('cancelled_task_count', 0)}",
                    flush=True,
                )
            finally:
                print("[stock-alpha] executor shutdown started", flush=True)
                if hasattr(executor, "shutdown"):
                    executor.shutdown(wait=True, cancel_futures=True)
                elif hasattr(executor, "__exit__"):
                    executor.__exit__(None, None, None)
                print("[stock-alpha] executor shutdown completed", flush=True)
        _record_phase(
            phase_timings,
            "symbol-task execution",
            phase_started,
            phase_start_ts,
            requested_workers=dataset_workers,
            effective_workers=effective_workers,
            task_count=len(pending_tasks),
            execution_mode=execution_mode,
        )
    except Exception:
        metadata["failed_task_count"] = task_count - len(results)
        raise
    phase_started, phase_start_ts = _phase_start()
    metadata["completed_task_count"] = len(results)
    metadata["written_partition_count"] = len([result for result in results if result.get("partition_written")])
    metadata["partition_row_counts"] = {
        str(result.get("symbol", "")).upper(): int(result.get("row_count", 0))
        for result in results
    }
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
    if partition_only:
        result_symbols = [str(result.get("symbol", "")).upper() for result in results]
        if len(result_symbols) != len(set(result_symbols)) or set(result_symbols) != set(symbols):
            raise ValueError("partition-only worker result population mismatch")
        results.clear()
        pending_tasks.clear()
        tasks.clear()
        return [], metadata
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


def _collect_symbol_task_futures(
    executor: Any,
    tasks: list[dict[str, Any]],
    *,
    task_timeout_seconds: float | None,
    progress_interval_seconds: float,
    metadata: dict[str, Any],
    diagnostics_path: Path | None,
    diagnostic_run_id: str | None,
) -> list[dict[str, Any]]:
    futures = {
        executor.submit(_build_dataset_symbol_task, task): str(task["symbol"]).upper()
        for task in tasks
    }
    pending = set(futures)
    completed: list[dict[str, Any]] = []
    started = time.perf_counter()
    next_progress = started + max(1.0, progress_interval_seconds)
    while pending:
        now = time.perf_counter()
        if task_timeout_seconds is not None and now - started > task_timeout_seconds:
            for future in pending:
                future.cancel()
            pending_symbols = sorted(futures[future] for future in pending)
            metadata["failed_task_count"] = len(pending)
            metadata["timed_out_symbols"] = pending_symbols
            for symbol in pending_symbols:
                _write_task_event(
                    diagnostics_path,
                    "timeout",
                    symbol=symbol,
                    exception_summary=f"pending after {task_timeout_seconds:.1f}s",
                    diagnostic_run_id=diagnostic_run_id,
                )
            raise TimeoutError(
                "stock-level dataset symbol tasks timed out after "
                f"{task_timeout_seconds:.1f}s; completed={len(completed)} "
                f"pending={len(pending)} pending_symbols={pending_symbols[:10]}"
            )
        done, pending = wait(
            pending,
            timeout=max(0.1, min(1.0, next_progress - now)),
            return_when=FIRST_COMPLETED,
        )
        for future in done:
            symbol = futures[future]
            if future.cancelled():
                metadata["cancelled_task_count"] = (
                    int(metadata.get("cancelled_task_count", 0)) + 1
                )
                _write_task_event(
                    diagnostics_path,
                    "cancelled",
                    symbol=symbol,
                    diagnostic_run_id=diagnostic_run_id,
                )
                continue
            try:
                completed.append(future.result())
            except Exception as exc:
                metadata["failed_task_count"] = (
                    int(metadata.get("failed_task_count", 0)) + 1
                )
                _write_task_event(
                    diagnostics_path,
                    "failed",
                    symbol=symbol,
                    exception_summary=f"{type(exc).__name__}: {exc}",
                    diagnostic_run_id=diagnostic_run_id,
                )
                raise
        if time.perf_counter() >= next_progress:
            elapsed = time.perf_counter() - started
            metadata["last_progress_completed_task_count"] = len(completed)
            metadata["last_progress_elapsed_seconds"] = elapsed
            print(
                "[stock-alpha] stock artifact symbol task heartbeat "
                f"completed={len(completed)}/{len(tasks)} "
                f"pending={len(pending)} elapsed={elapsed:.1f}s",
                flush=True,
            )
            next_progress = time.perf_counter() + max(1.0, progress_interval_seconds)
    if metadata.get("cancelled_task_count"):
        raise RuntimeError(
            "stock-level dataset symbol task cancellation blocks consolidation"
        )
    return completed


def _build_dataset_symbol_task(task: dict[str, Any]) -> dict[str, Any]:
    apply_worker_thread_environment(int(task.get("inner_thread_limit", 1)))
    task_started = time.perf_counter()
    symbol = str(task["symbol"]).upper()
    symbol_data = task.get("symbol_data", {})
    market_data = task.get("market_data", {})
    dates = list(task.get("dates", []))
    diagnostics_path = Path(str(task.get("diagnostics_path"))) if task.get("diagnostics_path") else None
    diagnostic_run_id = str(task.get("diagnostic_run_id") or "")
    partition_dir = Path(str(task.get("partition_dir"))) if task.get("partition_dir") else None
    _write_task_event(
        diagnostics_path,
        "started",
        symbol=symbol,
        worker_pid=os.getpid(),
        rows_read=len(symbol_data.get("close_dates", [])),
        date_count=len(dates),
        memory_estimate_bytes=_task_memory_estimate_bytes(task),
        diagnostic_run_id=diagnostic_run_id,
    )
    rows = []
    try:
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
    except Exception as exc:
        _write_task_event(
            diagnostics_path,
            "failed",
            symbol=symbol,
            worker_pid=os.getpid(),
            seconds_elapsed=max(0.0, time.perf_counter() - task_started),
            exception_summary=f"{type(exc).__name__}: {exc}",
            diagnostic_run_id=diagnostic_run_id,
        )
        raise
    seconds = max(0.0, time.perf_counter() - task_started)
    partition_path = ""
    partition_written = False
    if partition_dir is not None:
        partition_path = str(_write_symbol_partition(
            partition_dir,
            symbol,
            rows,
            dates,
            diagnostic_run_id=diagnostic_run_id,
        ))
        partition_written = True
    _write_task_event(
        diagnostics_path,
        "completed",
        symbol=symbol,
        worker_pid=os.getpid(),
        rows_read=len(symbol_data.get("close_dates", [])),
        rows_emitted=len(rows),
        date_count=len(dates),
        seconds_elapsed=seconds,
        output_size_estimate_bytes=_rows_size_estimate_bytes(rows),
        partition_path=partition_path,
        diagnostic_run_id=diagnostic_run_id,
    )
    return {
        "symbol": symbol,
        "rows": rows if bool(task.get("retain_rows", True)) else [],
        "row_count": len(rows),
        "worker_process_id": os.getpid(),
        "seconds_elapsed": seconds,
        "partition_path": partition_path,
        "partition_written": partition_written,
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


PARTITION_SCHEMA_VERSION = "stock_level_symbol_partition_v1"


def _load_reusable_partitions(
    tasks: list[dict[str, Any]],
    *,
    partition_dir: Path,
    expected_dates: list[str],
    diagnostics_path: Path | None,
    diagnostic_run_id: str | None,
    retain_rows: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reused = []
    pending = []
    for task in tasks:
        symbol = str(task["symbol"]).upper()
        path = _symbol_partition_path(partition_dir, symbol)
        if not path.exists():
            pending.append(task)
            continue
        result = _load_symbol_partition(path, symbol=symbol, expected_dates=expected_dates)
        if not retain_rows:
            result["rows"] = []
        reused.append(result)
        _write_task_event(
            diagnostics_path,
            "partition_reused",
            symbol=symbol,
            rows_emitted=result["row_count"],
            date_count=len(expected_dates),
            partition_path=str(path),
            diagnostic_run_id=diagnostic_run_id,
        )
    return reused, pending


def _symbol_partition_path(partition_dir: Path, symbol: str) -> Path:
    safe_symbol = symbol.upper().replace("/", "_").replace("\\", "_")
    return partition_dir / f"{safe_symbol}.json"


def _write_symbol_partition(
    partition_dir: Path,
    symbol: str,
    rows: list[dict[str, Any]],
    expected_dates: list[str],
    *,
    diagnostic_run_id: str = "",
) -> Path:
    path = _symbol_partition_path(partition_dir, symbol)
    payload = _symbol_partition_payload(
        symbol,
        rows,
        expected_dates,
        diagnostic_run_id=diagnostic_run_id,
    )
    if path.exists():
        existing = _load_symbol_partition(path, symbol=symbol, expected_dates=expected_dates)
        if _rows_hash(existing["rows"]) != payload["rows_sha256"]:
            raise ValueError(f"Refusing to overwrite incompatible stock-level symbol partition: {path}")
        return path
    partition_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True, default=str), encoding="utf-8")
    os.replace(tmp, path)
    return path


def _symbol_partition_payload(
    symbol: str,
    rows: list[dict[str, Any]],
    expected_dates: list[str],
    *,
    diagnostic_run_id: str = "",
) -> dict[str, Any]:
    _validate_partition_rows(symbol, rows, expected_dates)
    rows_sha = _rows_hash(rows)
    payload = {
        "schema_version": PARTITION_SCHEMA_VERSION,
        "symbol": symbol.upper(),
        "row_count": len(rows),
        "expected_date_count": len(expected_dates),
        "rows_sha256": rows_sha,
        "diagnostic_run_id": diagnostic_run_id,
        "dataset_identity": _hash_json({
            "source_dataset_hashes": sorted({
                str(row.get("source_dataset_hash"))
                for row in rows
                if row.get("source_dataset_hash")
            }),
            "decision_dates": expected_dates,
        }),
        "feature_schema_identity": _hash_json(
            sorted({key for row in rows for key in row})
        ),
        "target_contract_identity": _hash_json({
            "target_provenance_contract_version": sorted({
                str(row.get("target_provenance_contract_version"))
                for row in rows
                if row.get("target_provenance_contract_version")
            }),
            "target_horizons": sorted({
                str(row.get("target_horizon_trading_days"))
                for row in rows
                if row.get("target_horizon_trading_days") not in (None, "")
            }),
        }),
        "decision_date_panel_identity": _hash_json(expected_dates),
        "rows": rows,
    }
    payload["partition_identity"] = _hash_json({key: value for key, value in payload.items() if key != "rows"})
    return payload


def _load_symbol_partition(path: Path, *, symbol: str, expected_dates: list[str]) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Corrupt stock-level symbol partition: {path}") from exc
    if payload.get("schema_version") != PARTITION_SCHEMA_VERSION:
        raise ValueError(f"Incompatible stock-level symbol partition schema: {path}")
    if str(payload.get("symbol", "")).upper() != symbol.upper():
        raise ValueError(f"Stock-level symbol partition symbol mismatch: {path}")
    rows = list(payload.get("rows") or [])
    _validate_partition_rows(symbol, rows, expected_dates)
    if payload.get("row_count") != len(rows):
        raise ValueError(f"Stock-level symbol partition row count mismatch: {path}")
    if payload.get("rows_sha256") != _rows_hash(rows):
        raise ValueError(f"Stock-level symbol partition checksum mismatch: {path}")
    return {
        "symbol": symbol.upper(),
        "rows": rows,
        "row_count": len(rows),
        "worker_process_id": 0,
        "partition_path": str(path),
        "partition_reused": True,
    }


def _validate_partition_rows(symbol: str, rows: list[dict[str, Any]], expected_dates: list[str]) -> None:
    expected_keys = {(date, symbol.upper()) for date in expected_dates}
    keys = [
        (str(row.get("rebalance_date", "")), str(row.get("symbol", "")).upper())
        for row in rows
    ]
    duplicate_keys = sorted({key for key in keys if keys.count(key) > 1})
    missing_keys = sorted(expected_keys - set(keys))
    extra_keys = sorted(set(keys) - expected_keys)
    if duplicate_keys or missing_keys or extra_keys:
        raise ValueError(
            "Invalid stock-level symbol partition rows: "
            f"symbol={symbol} duplicate_keys={duplicate_keys[:5]} "
            f"missing_keys={missing_keys[:5]} extra_keys={extra_keys[:5]}"
        )


def _rows_hash(rows: list[dict[str, Any]]) -> str:
    return _hash_json(rows)


def _hash_json(payload: Any) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _write_task_event(
    diagnostics_path: Path | None,
    event_type: str,
    *,
    symbol: str,
    worker_pid: int | None = None,
    rows_read: int | None = None,
    rows_emitted: int | None = None,
    date_count: int | None = None,
    seconds_elapsed: float | None = None,
    memory_estimate_bytes: int | None = None,
    output_size_estimate_bytes: int | None = None,
    partition_path: str | None = None,
    exception_summary: str | None = None,
    diagnostic_run_id: str | None = None,
) -> None:
    if diagnostics_path is None:
        return
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "diagnostic_run_id": diagnostic_run_id,
        "event_type": event_type,
        "symbol": symbol,
        "pid": worker_pid,
        "rows_read": rows_read,
        "rows_emitted": rows_emitted,
        "date_count": date_count,
        "seconds_elapsed": seconds_elapsed,
        "memory_estimate_bytes": memory_estimate_bytes,
        "output_size_estimate_bytes": output_size_estimate_bytes,
        "partition_path": partition_path,
        "exception_summary": exception_summary,
    }
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(payload, sort_keys=True, default=str) + "\n").encode("utf-8")
    with diagnostics_path.open("a+b") as handle:
        _lock_file(handle)
        try:
            handle.seek(0, os.SEEK_END)
            handle.write(line)
            handle.flush()
        finally:
            _unlock_file(handle)


def _lock_file(handle: Any) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _unlock_file(handle: Any) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _task_memory_estimate_bytes(task: dict[str, Any]) -> int:
    symbol_data = task.get("symbol_data", {})
    market_data = task.get("market_data", {})
    symbol_points = len(symbol_data.get("close_dates", [])) + len(symbol_data.get("dollar_volume_dates", []))
    market_points = len(market_data.get("close_dates", [])) + len(market_data.get("dollar_volume_dates", []))
    return (symbol_points + market_points) * 32


def _rows_size_estimate_bytes(rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    return sum(
        sum(len(str(key)) + len(str(value)) for key, value in row.items())
        for row in rows
    )


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
