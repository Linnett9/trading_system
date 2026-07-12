from __future__ import annotations

import time
from datetime import datetime, timezone
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


def write_stock_level_prediction_artifacts(
    config: dict[str, Any],
) -> StockLevelPredictionArtifactsPaths:
    phase_timings: list[dict[str, Any]] = []
    phase_started, phase_start_ts = _phase_start()
    settings = StockLevelResearchConfig.from_mapping(config)
    apply_worker_thread_environment(settings.dataset_inner_threads)
    output_dir = _output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    expanded_rows = _read_csv(_expanded_dataset_path(config))
    meta_rows = _read_csv(output_dir / "meta_auxiliary_predictions.csv")
    sector_by_symbol = load_sector_by_symbol(
        config.get("ml", {}).get("sector_reference_path"),
        inline_mapping=dict(config.get("ml", {}).get("sector_by_symbol", {})),
    )
    universe_symbols = _universe_symbols(config)
    _record_phase(
        phase_timings,
        "configuration and source loading",
        phase_started,
        phase_start_ts,
        task_count=len(expanded_rows) + len(meta_rows),
    )
    phase_started, phase_start_ts = _phase_start()
    closes_by_symbol = _load_closes_by_symbol(config)
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
    )
    audit["phase_timings"] = [*phase_timings, *audit.get("phase_timings", [])]
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
