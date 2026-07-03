from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


DEFAULT_ADJUSTED_PRICE_DIR = Path("data/reference/adjusted_prices")
DEFAULT_REPORTS_ROOT = Path("reports")
DEFAULT_OUTPUT_PATH = (
    DEFAULT_REPORTS_ROOT
    / "ml"
    / "benchmark"
    / "regime_transformer_meta_ensemble_v1"
    / "missing_adjusted_price_symbols_for_news_transformer.json"
)
PRICE_IMPORTER_CANDIDATES = (
    "infrastructure/data/yahoo_adjusted_price_importer.py",
    "application/services/adjusted_price_commands.py",
)
FORWARD_LABEL_TRADING_DAYS = 20
PRICE_DATE_COLUMNS = ("date", "Date", "timestamp")
PRICE_CLOSE_COLUMNS = ("adj_close", "adjusted_close", "Adj Close", "AdjClose", "close", "Close")


@dataclass(frozen=True)
class PriceDirectorySummary:
    symbols: list[str]
    date_min: str
    date_max: str
    file_count: int
    required_columns_present: bool


def build_missing_price_symbol_plan(
    *,
    event_dataset_path: str | Path,
    adjusted_price_dir: str | Path = DEFAULT_ADJUSTED_PRICE_DIR,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    reports_root: str | Path = DEFAULT_REPORTS_ROOT,
    project_root: str | Path = ".",
) -> dict[str, Any]:
    output = Path(output_path)
    reports_root_path = Path(reports_root)
    if not _is_under_reports(output, reports_root_path):
        raise ValueError("output_path must be under reports/")

    events = _event_summary(Path(event_dataset_path))
    prices = _price_directory_summary(Path(adjusted_price_dir))
    event_symbols = events["symbols"]
    price_symbols = prices.symbols
    covered = sorted(set(event_symbols) & set(price_symbols))
    missing = sorted(set(event_symbols) - set(price_symbols))
    importers = _candidate_price_importers(Path(project_root))
    recommended_next_step = (
        "run_existing_adjusted_price_importer_for_missing_news_symbols_after_explicit_download_approval"
        if importers
        else "implement_adjusted_price_backfill_for_missing_news_symbols"
    )

    return {
        "event_symbol_count": len(event_symbols),
        "covered_price_symbol_count": len(covered),
        "missing_price_symbol_count": len(missing),
        "missing_price_symbols": missing,
        "covered_price_symbols": covered,
        "existing_price_date_min": prices.date_min,
        "existing_price_date_max": prices.date_max,
        "required_history_start": events["date_min"],
        "required_history_end": events["date_max"],
        "required_forward_label_trading_days": FORWARD_LABEL_TRADING_DAYS,
        "adjusted_price_dir": str(adjusted_price_dir),
        "adjusted_price_file_count": prices.file_count,
        "adjusted_price_required_columns_present": prices.required_columns_present,
        "candidate_price_importers_found": importers,
        "recommended_next_step": recommended_next_step,
        "trading_impact": "none_report_only",
    }


def write_missing_price_symbol_plan(
    *,
    event_dataset_path: str | Path,
    adjusted_price_dir: str | Path = DEFAULT_ADJUSTED_PRICE_DIR,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    reports_root: str | Path = DEFAULT_REPORTS_ROOT,
    project_root: str | Path = ".",
) -> dict[str, Any]:
    report = build_missing_price_symbol_plan(
        event_dataset_path=event_dataset_path,
        adjusted_price_dir=adjusted_price_dir,
        output_path=output_path,
        reports_root=reports_root,
        project_root=project_root,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _event_summary(path: Path) -> dict[str, Any]:
    symbols: set[str] = set()
    dates: list[str] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            symbol = str(row.get("symbol", "")).strip().upper()
            if symbol:
                symbols.add(symbol)
            date_value = _event_date(row)
            if date_value:
                dates.append(date_value)
    return {
        "symbols": sorted(symbols),
        "date_min": min(dates, default=""),
        "date_max": max(dates, default=""),
    }


def _event_date(row: dict[str, str]) -> str:
    for column in ("available_at_timestamp", "event_timestamp", "date"):
        value = str(row.get(column, "")).strip()
        if value:
            return value[:10]
    return ""


def _price_directory_summary(path: Path) -> PriceDirectorySummary:
    if path.is_dir():
        files = sorted(path.glob("*.csv"))
    elif path.exists():
        files = [path]
    else:
        files = []

    symbols: set[str] = set()
    dates: list[str] = []
    required_columns_present = bool(files)
    for file_path in files:
        summary = _price_file_summary(file_path)
        if summary["symbol"]:
            symbols.add(summary["symbol"])
        dates.extend(summary["dates"])
        required_columns_present = (
            required_columns_present
            and summary["has_date"]
            and summary["has_close_or_adjusted_close"]
        )
    return PriceDirectorySummary(
        symbols=sorted(symbols),
        date_min=min(dates, default=""),
        date_max=max(dates, default=""),
        file_count=len(files),
        required_columns_present=required_columns_present,
    )


def _price_file_summary(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        symbol = path.stem.upper()
        dates: list[str] = []
        for row in reader:
            row_symbol = str(row.get("symbol") or row.get("Symbol") or "").strip().upper()
            if row_symbol:
                symbol = row_symbol
            date_value = _first_text(row, PRICE_DATE_COLUMNS)
            if date_value:
                dates.append(date_value[:10])
    return {
        "symbol": symbol,
        "dates": dates,
        "has_date": bool(fieldnames & set(PRICE_DATE_COLUMNS)),
        "has_close_or_adjusted_close": bool(fieldnames & set(PRICE_CLOSE_COLUMNS)),
    }


def _first_text(row: dict[str, str], columns: Sequence[str]) -> str:
    for column in columns:
        value = str(row.get(column, "")).strip()
        if value:
            return value
    return ""


def _candidate_price_importers(project_root: Path) -> list[str]:
    return [
        candidate
        for candidate in PRICE_IMPORTER_CANDIDATES
        if (project_root / candidate).exists()
    ]


def _is_under_reports(path: Path, reports_root: Path) -> bool:
    resolved_path = path.resolve()
    resolved_reports = reports_root.resolve()
    return resolved_path == resolved_reports or resolved_reports in resolved_path.parents


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create a report-only plan for missing news-transformer adjusted prices."
    )
    parser.add_argument("--event-dataset", required=True)
    parser.add_argument("--adjusted-price-dir", default=str(DEFAULT_ADJUSTED_PRICE_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--reports-root", default=str(DEFAULT_REPORTS_ROOT))
    args = parser.parse_args(argv)

    report = write_missing_price_symbol_plan(
        event_dataset_path=args.event_dataset,
        adjusted_price_dir=args.adjusted_price_dir,
        output_path=args.output,
        reports_root=args.reports_root,
    )
    summary = {
        "event_symbol_count": report["event_symbol_count"],
        "covered_price_symbol_count": report["covered_price_symbol_count"],
        "missing_price_symbol_count": report["missing_price_symbol_count"],
        "first_30_missing_price_symbols": report["missing_price_symbols"][:30],
        "existing_price_date_min": report["existing_price_date_min"],
        "existing_price_date_max": report["existing_price_date_max"],
        "recommended_next_step": report["recommended_next_step"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
