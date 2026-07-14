from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.research.framework.config import StockLevelResearchConfig
from core.research.framework.reporting import ResearchArtifactWriter
from core.research.ml.runtime_parallelism import apply_worker_thread_environment
from core.research.ml.sector_reference import load_sector_by_symbol
from core.research.ml.stock_level.stock_alpha_paths import stock_alpha_report_metadata
from core.research.ml.stock_level.prediction_artifacts.io import (
    _markdown,
)
from core.research.ml.stock_level.prediction_artifacts.math import (
    _average_dollar_volume,
    _forward_return,
    _history_values_before,
    _trailing_drawdown,
    _trailing_liquidity_score,
    _trailing_return,
    _trailing_volatility,
)
from core.research.ml.stock_level.prediction_artifacts.rows import (
    _artifact_by_date_symbol,
    _artifact_dates,
    _audit,
    _baseline_predictions,
    _context_by_date,
    _expanded_dates,
    _prepare_symbol_data,
    build_stock_level_prediction_artifacts,
)
from core.research.ml.stock_level.prediction_artifacts.sources import (
    _expanded_dataset_path,
    _load_closes_by_symbol,
    _output_dir,
    _read_csv,
    _read_parquet_closes,
    _universe_symbols,
)
from core.research.ml.stock_level.prediction_artifacts.targets import (
    _actual_targets,
    _add_cross_sectional_targets,
)
from core.research.ml.stock_level.prediction_artifacts.types import (
    ACTUAL_COLUMNS,
    BASELINE_PREDICTION_COLUMNS,
    CONTEXT_COLUMNS,
    NOTICE,
    PREDICTION_COLUMNS,
    RESEARCH_METADATA,
    TARGET_TYPES,
    StockLevelPredictionArtifactsPaths,
)
from core.research.ml.stock_level.stock_level_artifact_io import (
    canonical_artifact_path,
    write_stock_level_artifact,
)

LOGGER = logging.getLogger("research")


def write_stock_level_prediction_artifacts(
    config: dict[str, Any],
) -> StockLevelPredictionArtifactsPaths:
    phase_timings: list[dict[str, Any]] = []
    phase_started, phase_start_ts = _phase_start()
    settings = StockLevelResearchConfig.from_mapping(config)
    apply_worker_thread_environment(settings.dataset_inner_threads)
    output_dir = _output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    expanded_path = _expanded_dataset_path(config)
    meta_path = output_dir / "meta_auxiliary_predictions.csv"
    expanded_rows = _read_csv(expanded_path)
    meta_rows = _read_csv(meta_path)
    sector_by_symbol = load_sector_by_symbol(
        config.get("ml", {}).get("sector_reference_path"),
        inline_mapping=dict(config.get("ml", {}).get("sector_by_symbol", {})),
    )
    universe_symbols = _universe_symbols(config)
    diagnostic_path = output_dir / "stock_artifact_symbol_tasks.jsonl"
    partition_dir = Path(
        str(
            config.get("ml", {}).get(
                "stock_level_dataset_partition_dir",
                output_dir / "stock_artifact_symbol_partitions",
            )
        )
    )
    resume_partitions = bool(config.get("ml", {}).get("stock_level_dataset_resume_partitions", True))
    diagnostic_run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    print(
        "[stock-alpha] stock artifact sources loaded "
        f"expanded_rows={len(expanded_rows)} meta_rows={len(meta_rows)} "
        f"universe_symbols={len(universe_symbols)} output_dir={output_dir}",
        flush=True,
    )
    LOGGER.info(
        "stock_artifact_sources_loaded",
        extra={
            "expanded_rows": len(expanded_rows),
            "meta_rows": len(meta_rows),
            "universe_symbols": len(universe_symbols),
            "output_dir": str(output_dir),
        },
    )
    _record_phase(
        phase_timings,
        "configuration and source loading",
        phase_started,
        phase_start_ts,
        task_count=len(expanded_rows) + len(meta_rows),
    )
    phase_started, phase_start_ts = _phase_start()
    closes_by_symbol = _load_closes_by_symbol(config)
    preflight = _preflight_payload(
        config,
        output_dir=output_dir,
        expanded_path=expanded_path,
        meta_path=meta_path,
        universe_symbols=universe_symbols,
        expanded_rows=expanded_rows,
        meta_rows=meta_rows,
        closes_by_symbol=closes_by_symbol,
        diagnostic_path=diagnostic_path,
        partition_dir=partition_dir,
        resume_partitions=resume_partitions,
        settings=settings,
    )
    print(
        "[stock-alpha] stock artifact preflight "
        f"config={preflight['config_path']} output_dir={preflight['output_dir']} "
        f"symbols={preflight['symbol_count']} expected_tasks={preflight['expected_task_count']} "
        f"workers={preflight['requested_workers']} price_files_size_mb={preflight['price_files_total_size_mb']:.1f} "
        f"expanded_rows={preflight['expanded_rows']} expanded_date_range={preflight['expanded_date_range']} "
        f"diagnostics={diagnostic_path} partitions={partition_dir} diagnostic_run_id={diagnostic_run_id}",
        flush=True,
    )
    print(
        "[stock-alpha] stock artifact prices loaded "
        f"symbols_with_prices={len(closes_by_symbol)}",
        flush=True,
    )
    LOGGER.info(
        "stock_artifact_prices_loaded",
        extra={"symbols_with_prices": len(closes_by_symbol)},
    )
    _record_phase(
        phase_timings,
        "price-history loading",
        phase_started,
        phase_start_ts,
        task_count=len(closes_by_symbol),
    )
    rows, audit = build_stock_level_prediction_artifacts(
        expanded_rows=expanded_rows,
        artifact_rows=meta_rows,
        universe_symbols=universe_symbols,
        closes_by_symbol=closes_by_symbol,
        sector_by_symbol=sector_by_symbol,
        market_symbol=str(config.get("ml", {}).get("stock_ranker_market_symbol", "SPY")),
        dataset_workers=settings.dataset_workers,
        inner_thread_limit=settings.dataset_inner_threads,
        decision_grid_frequency=str(
            config.get("ml", {}).get("stock_level_decision_frequency", "source")
        ),
        decision_grid_start_date=config.get("ml", {}).get("stock_level_decision_start_date"),
        decision_grid_end_date=config.get("ml", {}).get("stock_level_decision_end_date"),
        decision_grid_max_sessions=config.get("ml", {}).get("stock_level_decision_max_sessions"),
        decision_grid_min_history_sessions=int(
            config.get("ml", {}).get(
                "stock_level_decision_min_history_sessions",
                config.get("ml", {}).get("feature_lookback_days", 1),
            )
        ),
        task_timeout_seconds=_optional_float(
            config.get("ml", {}).get("stock_level_dataset_task_timeout_seconds")
        ),
        progress_interval_seconds=float(
            config.get("ml", {}).get(
                "stock_level_dataset_progress_interval_seconds",
                30.0,
            )
        ),
        diagnostics_path=diagnostic_path,
        diagnostic_run_id=diagnostic_run_id,
        partition_dir=partition_dir,
        resume_partitions=resume_partitions,
    )
    audit["phase_timings"] = [*phase_timings, *audit.get("phase_timings", [])]
    audit["stock_artifact_preflight"] = preflight
    paths = StockLevelPredictionArtifactsPaths(
        parquet_path=canonical_artifact_path(output_dir, "stock_level_prediction_artifacts", config),
        json_path=output_dir / "stock_level_prediction_artifacts.json",
        markdown_path=output_dir / "stock_level_prediction_artifacts.md",
        sample_csv_path=output_dir / "stock_level_prediction_artifacts_sample.csv",
    )
    fieldnames = list(rows[0]) if rows else ["rebalance_date", "symbol"]
    identity = write_stock_level_artifact(
        paths.parquet_path,
        rows,
        fieldnames=fieldnames,
        config=config,
        inspection_sample_path=paths.sample_csv_path,
        phase_timings=audit["phase_timings"],
        write_phase_name="base Parquet writing",
        validation_phase_name="base validation",
        hash_phase_name="logical-content hashing",
    )
    audit.update(
        stock_alpha_report_metadata(
            config,
            output_dir,
            generated_artifact_paths=[
                paths.parquet_path,
                paths.json_path,
                paths.markdown_path,
                paths.sample_csv_path,
            ],
        )
    )
    audit["canonical_artifact"] = identity
    audit["artifact_format"] = identity["artifact_format"]
    audit["artifact_path"] = identity["resolved_artifact_path"]
    audit["schema_fingerprint"] = identity["schema_fingerprint"]
    audit["artifact_sha256"] = identity["sha256"]
    audit["logical_content_sha256"] = identity["logical_content_sha256"]
    audit["target_contract_version"] = identity.get("target_contract_version")
    audit["benchmark_contract_version"] = identity.get("benchmark_contract_version")
    writer = ResearchArtifactWriter()
    writer.write_json(paths.json_path, audit)
    writer.write_markdown(paths.markdown_path, _markdown(audit))
    return paths


def _phase_start() -> tuple[float, str]:
    return time.perf_counter(), datetime.now(timezone.utc).isoformat()


def _record_phase(
    timings: list[dict[str, Any]],
    phase_name: str,
    started: float,
    start_timestamp: str,
    *,
    requested_workers: int = 1,
    effective_workers: int = 1,
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


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _preflight_payload(
    config: dict[str, Any],
    *,
    output_dir: Path,
    expanded_path: Path,
    meta_path: Path,
    universe_symbols: list[str],
    expanded_rows: list[dict[str, str]],
    meta_rows: list[dict[str, str]],
    closes_by_symbol: dict[str, dict[str, dict[str, float]]],
    diagnostic_path: Path,
    partition_dir: Path,
    resume_partitions: bool,
    settings: StockLevelResearchConfig,
) -> dict[str, Any]:
    price_file_sizes = _price_file_sizes(config, universe_symbols)
    expanded_dates = [
        str(row.get("rebalance_date") or row.get("feature_date") or "")
        for row in expanded_rows
        if row.get("rebalance_date") or row.get("feature_date")
    ]
    available_ram_mb = _available_ram_mb()
    return {
        "config_path": str(config.get("config_path", "")),
        "output_dir": str(output_dir),
        "expanded_dataset_path": str(expanded_path),
        "expanded_dataset_exists": expanded_path.exists(),
        "expanded_dataset_size_bytes": expanded_path.stat().st_size if expanded_path.exists() else 0,
        "expanded_rows": len(expanded_rows),
        "expanded_date_range": [min(expanded_dates), max(expanded_dates)] if expanded_dates else None,
        "meta_auxiliary_predictions_path": str(meta_path),
        "meta_auxiliary_predictions_exists": meta_path.exists(),
        "meta_auxiliary_predictions_size_bytes": meta_path.stat().st_size if meta_path.exists() else 0,
        "meta_rows": len(meta_rows),
        "symbol_count": len(universe_symbols),
        "symbols_preview": universe_symbols[:10],
        "expected_task_count": len(universe_symbols),
        "requested_workers": settings.dataset_workers,
        "inner_thread_limit": settings.dataset_inner_threads,
        "decision_frequency": str(config.get("ml", {}).get("stock_level_decision_frequency", "source")),
        "decision_max_sessions": config.get("ml", {}).get("stock_level_decision_max_sessions"),
        "decision_min_history_sessions": config.get("ml", {}).get("stock_level_decision_min_history_sessions"),
        "price_symbols_loaded": len(closes_by_symbol),
        "price_files_total_size_bytes": sum(price_file_sizes.values()),
        "price_files_total_size_mb": sum(price_file_sizes.values()) / (1024 * 1024),
        "price_file_missing_symbols": sorted(set(universe_symbols) - set(price_file_sizes)),
        "available_ram_mb": available_ram_mb,
        "partition_dir": str(partition_dir),
        "resume_partitions": resume_partitions,
        "existing_partition_count": len(list(partition_dir.glob("*.json"))) if partition_dir.exists() else 0,
        "expected_output_paths": {
            "parquet_path": str(canonical_artifact_path(output_dir, "stock_level_prediction_artifacts", config)),
            "json_path": str(output_dir / "stock_level_prediction_artifacts.json"),
            "markdown_path": str(output_dir / "stock_level_prediction_artifacts.md"),
            "diagnostics_jsonl_path": str(diagnostic_path),
        },
    }


def _price_file_sizes(config: dict[str, Any], symbols: list[str]) -> dict[str, int]:
    ml = config.get("ml", {})
    parquet_dir = Path(str(ml.get("stooq_parquet_dir", ml.get("parquet_dir", "data/processed/stooq_parquet"))))
    sizes = {}
    for symbol in symbols:
        flat_path = parquet_dir / f"{symbol.upper()}.parquet"
        nested_path = parquet_dir / symbol.upper() / "1Day" / "bars.parquet"
        path = flat_path if flat_path.exists() else nested_path
        if path.exists():
            sizes[symbol.upper()] = path.stat().st_size
    return sizes


def _available_ram_mb() -> float | None:
    if os.name != "nt":
        return None
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return status.ullAvailPhys / (1024 * 1024)
    except Exception:
        return None
