from __future__ import annotations

from pathlib import Path
from typing import Any

from core.research.framework.config import StockLevelResearchConfig
from core.research.framework.data import CsvRowRepository
from core.research.framework.reporting import ResearchArtifactWriter
from core.research.ml.stock_level.stock_level_alpha_features_types import ENGINEERED_FEATURE_COLUMNS, NOTICE


def _load_price_histories(
    parquet_dir: Path,
    symbols: list[str],
) -> dict[str, list[dict[str, Any]]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Stock-level alpha feature generation requires pyarrow. "
            "Install project requirements before running this research command."
        ) from exc
    output = {}
    for symbol in symbols:
        path = parquet_dir / f"{symbol}.parquet"
        if not path.exists():
            output[symbol] = []
            continue
        table = pq.read_table(path, columns=["timestamp", "high", "low", "close"])
        data = table.to_pydict()
        output[symbol] = [
            {
                "date": value.date().isoformat() if hasattr(value, "date") else str(value)[:10],
                "high": high,
                "low": low,
                "close": close,
            }
            for value, high, low, close in zip(
                data["timestamp"], data["high"], data["low"], data["close"]
            )
        ]
    return output
def _output_dir(config: dict[str, Any]) -> Path:
    return StockLevelResearchConfig.from_mapping(config).output_dir
def _read_csv(path: Path) -> list[dict[str, str]]:
    return CsvRowRepository().read(path)
def _write_enriched_csv(
    path: Path,
    source_rows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> None:
    source_columns = list(source_rows[0]) if source_rows else []
    fieldnames = [*source_columns, *ENGINEERED_FEATURE_COLUMNS]
    ResearchArtifactWriter().write_csv(path, rows, fieldnames=fieldnames)
def _write_audit_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "feature",
        "definition",
        "populated_count",
        "missing_count",
        "availability_rate",
    ]
    ResearchArtifactWriter().write_csv(path, rows, fieldnames=fieldnames)
def _markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Stock-Level Alpha Feature Audit",
        "",
        NOTICE,
        "",
        f"- Rows: {audit['row_count']}",
        f"- Engineered features: {audit['engineered_feature_count']}",
        f"- Source columns preserved: {audit['source_columns_preserved']}",
        f"- Unique symbol/date rows: {audit['unique_symbol_date_rows']}",
        f"- Industry metadata available: {audit['industry_metadata_available']}",
        f"- Alpha feature workers: {audit['parallelism']['stock_alpha_feature_n_jobs']}",
        f"- Parallel partition: {audit['parallelism']['partition']}",
        "- Promotion thresholds changed: false",
        "",
        "| Feature | Populated | Missing | Availability | Definition |",
        "|---|---:|---:|---:|---|",
    ]
    for row in audit["features"]:
        lines.append(
            f"| {row['feature']} | {row['populated_count']} | {row['missing_count']} | "
            f"{row['availability_rate']:.4f} | {row['definition']} |"
        )
    lines.append("")
    return "\n".join(lines)
