from __future__ import annotations

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
    rows, audit = build_stock_level_prediction_artifacts(
        expanded_rows=expanded_rows,
        artifact_rows=meta_rows,
        universe_symbols=_universe_symbols(config),
        closes_by_symbol=_load_closes_by_symbol(config),
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
