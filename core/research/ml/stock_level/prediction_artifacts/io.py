from __future__ import annotations

from typing import Any

from core.research.framework.reporting import ResearchArtifactWriter
from core.research.ml.stock_level.prediction_artifacts.types import (
    ACTUAL_COLUMNS,
    CONTEXT_COLUMNS,
    NOTICE,
    PREDICTION_COLUMNS,
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "rebalance_date",
        "symbol",
        "sector",
        "average_dollar_volume_21d",
        "average_dollar_volume_63d",
        *PREDICTION_COLUMNS,
        *ACTUAL_COLUMNS,
        *CONTEXT_COLUMNS,
        "source",
        "source_feature_id",
        "source_model_type",
        "source_split",
        "source_dataset_hash",
        "true_stock_level_row",
    ]
    normalized = [
        {name: row.get(name, "") for name in fieldnames}
        for row in rows
    ]
    ResearchArtifactWriter().write_csv(path, normalized, fieldnames=fieldnames)


def _markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Stock-Level Prediction Artifacts",
        "",
        NOTICE,
        "",
        f"- Rows: {audit['row_count']}",
        f"- Symbols: {audit['symbol_count']}",
        f"- Rebalance dates: {audit['rebalance_date_count']}",
        f"- Date range: {audit['date_range']}",
        f"- Average symbols per date: {audit['average_symbols_per_date']:.2f}",
        f"- True stock-level rows: {audit['true_stock_level_rows']}",
        f"- Usable for stock-level ranking: {audit['usable_for_stock_level_ranking']}",
        f"- Suitable for true stock-level ranking diagnostics: {audit['suitable_for_true_stock_level_ranking_diagnostics']}",
        f"- Suitability reason: {audit['suitability_reason']}",
        "",
        "## Populated Predictions",
        "",
    ]
    for column, count in audit["populated_prediction_counts"].items():
        lines.append(f"- {column}: {count}")
    lines.extend([
        "",
        "## Missing Predictions",
        "",
    ])
    for column, count in audit["missing_prediction_counts"].items():
        lines.append(f"- {column}: {count}")
    lines.extend(["", "## Missing Actual Targets", ""])
    for column, count in audit["missing_actual_target_counts"].items():
        lines.append(f"- {column}: {count}")
    lines.extend([
        "",
        "## Root Cause",
        "",
        audit["root_cause_artifact_level_limitation"],
        "",
    ])
    return "\n".join(lines)
