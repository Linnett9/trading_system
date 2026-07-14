from __future__ import annotations

from pathlib import Path
from typing import Any

from core.research.framework.config import StockLevelResearchConfig
from core.research.framework.data import CsvRowRepository
from core.research.framework.reporting import ResearchArtifactWriter
from core.research.ml.stock_level.stock_level_artifact_io import (
    read_stock_level_artifact,
    write_stock_level_artifact,
)
from core.research.ml.stock_level.stock_level_alpha_features_types import (
    ENGINEERED_FEATURE_COLUMNS,
    ENRICHMENT_METADATA_COLUMNS,
    NOTICE,
)


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
        canonical_files = sorted((parquet_dir / f"symbol={symbol}").glob("year=*/bars.parquet"))
        if canonical_files:
            rows = []
            for canonical_path in canonical_files:
                table = pq.read_table(canonical_path, columns=["session_date", "model_high", "model_low", "model_close", "selector_eligible"])
                for row in table.to_pylist():
                    if not row.get("selector_eligible"):
                        continue
                    rows.append(
                        {
                            "date": str(row["session_date"])[:10],
                            "high": row["model_high"],
                            "low": row["model_low"],
                            "close": row["model_close"],
                        }
                    )
            output[symbol] = sorted(rows, key=lambda row: row["date"])
            continue
        path = parquet_dir / f"{symbol}.parquet"
        if not path.exists():
            structured_path = parquet_dir / symbol.upper() / "1Day" / "bars.parquet"
            path = structured_path
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
    if path.suffix.lower() == ".parquet":
        return read_stock_level_artifact(
            path,
            required_columns={"rebalance_date", "symbol"},
        )
    return CsvRowRepository().read(path)


def _write_enriched_csv(
    path: Path,
    source_rows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    *,
    config: dict[str, Any] | None = None,
    sample_path: Path | None = None,
    phase_timings: list[dict[str, Any]] | None = None,
    write_phase_name: str | None = None,
    validation_phase_name: str | None = None,
    hash_phase_name: str | None = None,
) -> dict[str, Any] | None:
    source_columns = list(source_rows[0]) if source_rows else []
    fieldnames = [*source_columns, *ENGINEERED_FEATURE_COLUMNS, *ENRICHMENT_METADATA_COLUMNS]
    if path.suffix.lower() == ".parquet":
        return write_stock_level_artifact(
            path,
            rows,
            fieldnames=fieldnames,
            config=config or {"ml": {"stock_level_artifact_format": "parquet"}},
            inspection_sample_path=sample_path,
            phase_timings=phase_timings,
            write_phase_name=write_phase_name,
            validation_phase_name=validation_phase_name,
            hash_phase_name=hash_phase_name,
        )
    ResearchArtifactWriter().write_csv(path, rows, fieldnames=fieldnames)
    return None
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
