from __future__ import annotations

from typing import Any

from core.research.framework.config import StockLevelResearchConfig
from core.research.framework.data import CsvRowRepository
from core.research.framework.logging import ResearchStageLogger
from core.research.framework.registry import FeatureRegistry
from core.research.framework.reporting import ResearchArtifactWriter
from core.research.ml.runtime_parallelism import apply_stock_alpha_worker_caps
from core.research.ml.stock_level.stock_alpha_paths import stock_alpha_report_metadata
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
    settings = StockLevelResearchConfig.from_mapping(config)
    apply_stock_alpha_worker_caps(config)
    output_dir = settings.output_dir
    source_path = settings.base_artifact_path
    if not source_path.exists():
        raise FileNotFoundError(f"Base stock-level artifact not found: {source_path}")
    logger = ResearchStageLogger("stock_level_alpha_features")
    repository = CsvRowRepository()
    with logger.stage("loading"):
        rows = repository.read(source_path)
        rows, run_profile = apply_stock_alpha_run_profile(rows, settings)
    symbols = sorted({str(row.get("symbol", "")).upper() for row in rows if row.get("symbol")})
    spy_symbol = settings.spy_symbol
    with logger.stage("feature_generation"):
        price_histories = _load_price_histories(
            settings.parquet_dir,
            sorted({*symbols, spy_symbol}),
        )
        enriched_rows, audit = build_stock_level_alpha_features(
            rows,
            price_histories,
            spy_symbol=spy_symbol,
            source_path=str(source_path),
            n_jobs=settings.alpha_feature_n_jobs,
        )
        audit.update(run_profile)
        audit.update(stock_alpha_report_metadata(config, output_dir, source_artifact_path=source_path))

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = StockLevelAlphaFeaturePaths(
        enriched_csv_path=output_dir / "stock_level_prediction_artifacts_enriched.csv",
        audit_csv_path=output_dir / "stock_level_alpha_feature_audit.csv",
        audit_json_path=output_dir / "stock_level_alpha_feature_audit.json",
        audit_markdown_path=output_dir / "stock_level_alpha_feature_audit.md",
    )
    with logger.stage("report_generation"):
        _write_enriched_csv(paths.enriched_csv_path, rows, enriched_rows)
        _write_audit_csv(paths.audit_csv_path, audit["features"])
        writer = ResearchArtifactWriter()
        writer.write_json(paths.audit_json_path, audit)
        writer.write_markdown(paths.audit_markdown_path, _markdown(audit))
    return paths
