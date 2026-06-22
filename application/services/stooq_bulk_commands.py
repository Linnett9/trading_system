from __future__ import annotations

import json
from pathlib import Path

from infrastructure.data.stooq_bulk_importer import StooqBulkImporter


def run_stooq_bulk_import(config: dict) -> None:
    ml_config = config.get("ml", {})
    research_config = config.get("research", {})
    symbols = research_config.get(
        "stooq_import_symbols",
        config.get("backtest", {}).get("symbols", []),
    )
    if not symbols:
        raise ValueError("No symbols configured for Stooq bulk import")
    importer = StooqBulkImporter(
        extracted_dir=str(ml_config.get(
            "stooq_bulk_extracted_dir", "data/raw/stooq_bulk/extracted"
        )),
        parquet_dir=str(ml_config.get(
            "stooq_parquet_dir", "data/processed/stooq_parquet"
        )),
        zip_path=str(ml_config.get(
            "stooq_bulk_zip_path", "data/raw/stooq_bulk/us_daily_ascii.zip"
        )),
        minimum_history_years=int(ml_config.get("minimum_history_years", 9)),
        history_tolerance_days=int(
            ml_config.get("history_coverage_tolerance_days", 10)
        ),
    )
    results = importer.import_symbols([str(symbol).upper() for symbol in symbols])
    report_path = Path(ml_config.get("output_dir", "reports/ml")) / "stooq_bulk_import.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({
        "source": "stooq_bulk",
        "symbol_count": len(results),
        "symbols": [result.__dict__ for result in results],
        "research_only": True,
        "trading_impact": "none",
    }, indent=2), encoding="utf-8")
    print("\nSTOOQ BULK IMPORT")
    print(f"Imported symbols: {len(results)}")
    print(f"Parquet directory: {ml_config.get('stooq_parquet_dir')}")
    print(f"Report: {report_path}")
