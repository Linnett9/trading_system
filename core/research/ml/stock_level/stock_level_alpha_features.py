from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from core.research.framework.config import StockLevelResearchConfig
from core.research.framework.logging import ResearchStageLogger
from core.research.framework.registry import FeatureRegistry
from core.research.framework.reporting import ResearchArtifactWriter
from core.research.ml.runtime_parallelism import apply_stock_alpha_worker_caps
from core.research.ml.stock_level.stock_alpha_paths import stock_alpha_report_metadata
from core.research.ml.stock_level.stock_level_artifact_io import canonical_artifact_path
from core.research.ml.stock_level.stock_alpha_run_profile import apply_stock_alpha_run_profile
from core.research.ml.stock_level.stock_level_alpha_features_audit import (
    _audit,
    alpha_feature_registry,
)
from core.research.ml.stock_level.stock_level_alpha_features_builder import (
    _add_cross_sectional_features,
    _add_group_relative_features,
    _build_symbol_level_features,
    _build_symbol_rows,
    _history_before,
    _prepare_history,
    _time_series_features,
    build_stock_level_alpha_features,
)
from core.research.ml.stock_level.stock_level_alpha_features_io import (
    _load_price_histories,
    _markdown,
    _output_dir,
    _read_csv,
    _write_audit_csv,
    _write_enriched_csv,
)
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
    ENGINEERED_FEATURE_COLUMNS,
    FEATURE_DEFINITIONS,
    NOTICE,
    RESEARCH_METADATA,
    StockLevelAlphaFeaturePaths,
)


def write_stock_level_alpha_features(
    config: dict[str, Any],
) -> StockLevelAlphaFeaturePaths:
    phase_timings: list[dict[str, Any]] = []
    settings = StockLevelResearchConfig.from_mapping(config)
    apply_stock_alpha_worker_caps(config)
    output_dir = settings.output_dir
    source_path = settings.base_artifact_path
    if not source_path.exists():
        raise FileNotFoundError(f"Base stock-level artifact not found: {source_path}")
    logger = ResearchStageLogger("stock_level_alpha_features")
    with logger.stage("loading"):
        phase_started, phase_start_ts = _phase_start()
        rows = _read_csv(source_path)
        rows, run_profile = apply_stock_alpha_run_profile(rows, settings)
        _record_phase(
            phase_timings,
            "alpha-feature loading",
            phase_started,
            phase_start_ts,
            task_count=len(rows),
        )
    symbols = sorted({str(row.get("symbol", "")).upper() for row in rows if row.get("symbol")})
    spy_symbol = settings.spy_symbol
    with logger.stage("feature_generation"):
        phase_started, phase_start_ts = _phase_start()
        price_histories = _load_price_histories(
            settings.parquet_dir,
            sorted({*symbols, spy_symbol}),
        )
        _record_phase(
            phase_timings,
            "price-history loading",
            phase_started,
            phase_start_ts,
            task_count=len(price_histories),
        )
        phase_started, phase_start_ts = _phase_start()
        enriched_rows, audit = build_stock_level_alpha_features(
            rows,
            price_histories,
            spy_symbol=spy_symbol,
            source_path=str(source_path),
            n_jobs=settings.alpha_feature_n_jobs,
        )
        _record_phase(
            phase_timings,
            "alpha-feature calculation",
            phase_started,
            phase_start_ts,
            requested_workers=settings.alpha_feature_n_jobs,
            effective_workers=audit.get("parallelism", {}).get("effective_workers", 1),
            task_count=len(symbols),
            execution_mode=audit.get("parallelism", {}).get("partition", "symbol"),
        )
        audit["phase_timings"] = [
            *phase_timings,
            *audit.get("phase_timings", []),
        ]
        audit.update(run_profile)
        audit.update(stock_alpha_report_metadata(config, output_dir, source_artifact_path=source_path))

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = StockLevelAlphaFeaturePaths(
        enriched_parquet_path=canonical_artifact_path(output_dir, "stock_level_prediction_artifacts_enriched", config),
        audit_csv_path=output_dir / "stock_level_alpha_feature_audit.csv",
        audit_json_path=output_dir / "stock_level_alpha_feature_audit.json",
        audit_markdown_path=output_dir / "stock_level_alpha_feature_audit.md",
        enriched_sample_csv_path=output_dir / "stock_level_prediction_artifacts_enriched_sample.csv",
    )
    with logger.stage("report_generation"):
        artifact_identity = _write_enriched_csv(
            paths.enriched_parquet_path,
            rows,
            enriched_rows,
            config=config,
            sample_path=paths.enriched_sample_csv_path,
            phase_timings=audit["phase_timings"],
            write_phase_name="enriched Parquet writing",
            validation_phase_name="enriched validation",
            hash_phase_name="logical-content hashing",
        )
        _write_audit_csv(paths.audit_csv_path, audit["features"])
        writer = ResearchArtifactWriter()
        if artifact_identity is not None:
            audit["canonical_artifact"] = artifact_identity
            audit["artifact_format"] = artifact_identity["artifact_format"]
            audit["artifact_path"] = artifact_identity["resolved_artifact_path"]
            audit["artifact_sha256"] = artifact_identity["sha256"]
            audit["logical_content_sha256"] = artifact_identity["logical_content_sha256"]
            audit["schema_fingerprint"] = artifact_identity["schema_fingerprint"]
            audit["target_contract_version"] = artifact_identity.get("target_contract_version")
            audit["benchmark_contract_version"] = "stock_level_benchmark_return_10d_v1"
        writer.write_json(paths.audit_json_path, audit)
        writer.write_markdown(paths.audit_markdown_path, _markdown(audit))
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
