from __future__ import annotations

from typing import Any

from core.research.framework.reporting import ResearchArtifactWriter
from core.research.ml.data.sector_reference import load_sector_by_symbol
from core.research.ml.stock_level.stock_alpha_paths import stock_alpha_report_metadata
from core.research.ml.stock_level.prediction_artifacts.io import (
    _markdown,
    _write_csv,
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
        market_symbol=str(config.get("ml", {}).get("stock_ranker_market_symbol", "SPY")),
    )
    paths = StockLevelPredictionArtifactsPaths(
        csv_path=output_dir / "stock_level_prediction_artifacts.csv",
        json_path=output_dir / "stock_level_prediction_artifacts.json",
        markdown_path=output_dir / "stock_level_prediction_artifacts.md",
    )
    audit.update(stock_alpha_report_metadata(config, output_dir, generated_artifact_paths=[paths.csv_path, paths.json_path, paths.markdown_path]))
    _write_csv(paths.csv_path, rows)
    writer = ResearchArtifactWriter()
    writer.write_json(paths.json_path, audit)
    writer.write_markdown(paths.markdown_path, _markdown(audit))
    return paths
