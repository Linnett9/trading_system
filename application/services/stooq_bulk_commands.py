from __future__ import annotations

from pathlib import Path

from infrastructure.data.stooq_bulk_importer import (
    StooqBulkImporter,
    StooqRawSymbolCandidate,
)


def run_stooq_bulk_import(
    config: dict,
    symbols: list[str] | None = None,
    top: int | None = None,
    all_raw: bool = False,
    asset_class: str = "all",
    min_rows: int | None = None,
    exclude_warrants_units_rights: bool = False,
) -> None:
    ml_config = config.get("ml", {})
    research_config = config.get("research", {})
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
    raw_candidates = _select_raw_candidates(
        importer,
        top=top,
        all_raw=all_raw,
        asset_class=asset_class,
        min_rows=min_rows,
        exclude_warrants_units_rights=exclude_warrants_units_rights,
    )
    if raw_candidates is not None:
        resolved_symbols = [candidate.symbol for candidate in raw_candidates]
    else:
        resolved_symbols = symbols or research_config.get(
            "stooq_import_symbols",
            config.get("backtest", {}).get("symbols", []),
        )
    if not resolved_symbols:
        raise ValueError("No symbols configured for Stooq bulk import")
    report_path = Path(ml_config.get("output_dir", "reports/ml")) / "stooq_bulk_import.json"
    manifest = importer.import_symbols_with_manifest(
        [str(symbol).upper() for symbol in resolved_symbols],
        manifest_path=report_path,
        resume=bool(ml_config.get("resume_stooq_bulk_import", True)),
    )
    print("\nSTOOQ BULK IMPORT")
    if raw_candidates is not None:
        print(f"Raw Stooq candidates selected: {len(raw_candidates)}")
        print(f"Raw asset class filter: {asset_class}")
        print(f"Raw min rows filter: {min_rows or 0}")
        print(
            "Exclude warrants/units/rights: "
            f"{exclude_warrants_units_rights}"
        )
        if top is not None:
            print(f"Raw top limit: {top}")
    print(f"Imported symbols: {len(manifest.imported)}")
    print(f"Skipped existing symbols: {len(manifest.skipped_existing)}")
    print(f"Missing symbols: {len(manifest.missing_symbols)}")
    print(f"Failed symbols: {len(manifest.failed_symbols)}")
    print(f"Symbols with missing-data gaps: {len(manifest.missing_data_symbols)}")
    print(f"Parquet directory: {ml_config.get('stooq_parquet_dir')}")
    print(f"Report: {report_path}")
    for item in manifest.missing_symbols:
        print(f"Missing {item['symbol']}: {item['reason']}")
    for item in manifest.failed_symbols:
        print(f"Failed {item['symbol']}: {item['reason']}")


def _select_raw_candidates(
    importer: StooqBulkImporter,
    top: int | None,
    all_raw: bool,
    asset_class: str,
    min_rows: int | None,
    exclude_warrants_units_rights: bool,
) -> list[StooqRawSymbolCandidate] | None:
    raw_selection_requested = (
        all_raw
        or top is not None
        or asset_class != "all"
        or min_rows is not None
        or exclude_warrants_units_rights
    )
    if not raw_selection_requested:
        return None
    return importer.select_raw_symbols(
        top=top,
        asset_class=asset_class,
        min_rows=min_rows or 0,
        exclude_warrants_units_rights=exclude_warrants_units_rights,
    )
