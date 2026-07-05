from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from infrastructure.data.market_parquet import (
    MarketParquetImporter,
    migrate_legacy_daily_parquet,
    normalize_timeframe,
)


def run_market_parquet_import(
    config: dict[str, Any],
    *,
    symbols: list[str] | None = None,
    timeframes: list[str] | None = None,
) -> None:
    ml_config = config.get("ml", {})
    market_config = ml_config.get("market_data", {})
    raw_root = Path(
        market_config.get(
            "raw_root",
            "~/Downloads",
        )
    ).expanduser()
    output_root = Path(market_config.get("processed_root", "data/processed"))
    timezone_name = str(market_config.get("timezone", "UTC"))
    configured_timeframes = {
        normalize_timeframe(timeframe): dict(timeframe_config or {})
        for timeframe, timeframe_config in market_config.get("timeframes", {}).items()
    }
    requested_timeframes = [
        normalize_timeframe(value)
        for value in (
            timeframes
            or market_config.get("enabled_timeframes")
            or list(configured_timeframes)
            or ["1Day"]
        )
    ]
    missing_dirs = []
    for timeframe in requested_timeframes:
        timeframe_config = configured_timeframes.get(timeframe, {})
        raw_dir = Path(timeframe_config.get("raw_dir", raw_root / timeframe)).expanduser()
        if timeframe != "1Day" and not raw_dir.is_dir():
            missing_dirs.append(f"{timeframe}: {raw_dir}")
    if missing_dirs:
        raise FileNotFoundError(
            "Raw market data directories are required for requested intraday "
            "timeframes: "
            + "; ".join(missing_dirs)
        )

    results = []
    for timeframe in requested_timeframes:
        timeframe_config = configured_timeframes.get(timeframe, {})
        raw_dir = Path(timeframe_config.get("raw_dir", raw_root / timeframe)).expanduser()
        if timeframe == "1Day" and not raw_dir.exists():
            results.extend(
                migrate_legacy_daily_parquet(
                    legacy_dir=ml_config.get(
                        "stooq_parquet_dir",
                        "data/processed/stooq_parquet",
                    ),
                    output_root=output_root,
                    symbols=symbols,
                    resume=bool(market_config.get("resume_import", True)),
                )
            )
            continue
        importer = MarketParquetImporter(
            raw_dir=raw_dir,
            output_root=output_root,
            timezone_name=timezone_name,
        )
        results.extend(
            importer.import_timeframe(
                timeframe,
                symbols=symbols,
                resume=bool(market_config.get("resume_import", True)),
            )
        )

    report_path = Path(
        market_config.get(
            "import_report_path",
            Path(ml_config.get("output_dir", "reports/ml")) / "market_parquet_import.json",
        )
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "source": "market_parquet_import",
                "processed_root": str(output_root),
                "timestamp_timezone": "UTC",
                "timestamp_semantics": "bar_close",
                "timeframes": requested_timeframes,
                "imported_count": sum(not result.skipped_existing for result in results),
                "skipped_existing_count": sum(result.skipped_existing for result in results),
                "symbols": [asdict(result) for result in results],
                "research_only": True,
                "trading_impact": "none",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nMARKET PARQUET IMPORT")
    print(f"Timeframes: {', '.join(requested_timeframes)}")
    print(f"Processed root: {output_root}")
    print(f"Imported files: {sum(not result.skipped_existing for result in results)}")
    print(f"Skipped existing files: {sum(result.skipped_existing for result in results)}")
    print(f"Report: {report_path}")
