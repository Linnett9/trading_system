from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_CANDIDATES = ("data/reference/adjusted_prices",)
REQUIRED_PRICE_COLUMNS = {"date", "close_or_adjusted_close"}


def build_stock_alpha_news_price_source_audit(
    *,
    event_dataset_path: str | Path,
    candidate_paths: Sequence[str | Path] = DEFAULT_CANDIDATES,
) -> dict[str, Any]:
    news_symbols = _event_symbols(Path(event_dataset_path))
    candidates = [
        _candidate_summary(Path(path), news_symbols)
        for path in candidate_paths
    ]
    viable = [
        candidate for candidate in candidates
        if candidate["required_columns_present"] and candidate["symbol_coverage_count"] == len(news_symbols)
    ]
    return {
        "canonical_price_source_found": bool(viable),
        "candidate_price_sources": candidates,
        "required_columns_present": bool(viable),
        "missing_columns": [] if viable else ["full news-symbol coverage from adjusted close source"],
        "symbol_coverage_count": max((candidate["symbol_coverage_count"] for candidate in candidates), default=0),
        "missing_news_symbols": min(
            (candidate["missing_news_symbols"] for candidate in candidates),
            key=len,
            default=news_symbols,
        ),
        "date_min": min((candidate["date_min"] for candidate in candidates if candidate["date_min"]), default=""),
        "date_max": max((candidate["date_max"] for candidate in candidates if candidate["date_max"]), default=""),
        "recommended_next_step": (
            "attach_price_return_labels"
            if viable
            else "extend_adjusted_price_source_to_full_news_symbol_universe"
        ),
    }


def _candidate_summary(path: Path, news_symbols: Sequence[str]) -> dict[str, Any]:
    if path.is_dir():
        files = sorted(path.glob("*.csv"))
    elif path.exists():
        files = [path]
    else:
        files = []
    symbols: set[str] = set()
    date_min = ""
    date_max = ""
    missing_columns: set[str] = set()
    for file_path in files:
        summary = _price_file_summary(file_path)
        if summary["symbol"]:
            symbols.add(summary["symbol"])
        if not summary["has_date"]:
            missing_columns.add("date")
        if not summary["has_close_or_adjusted_close"]:
            missing_columns.add("close_or_adjusted_close")
        if summary["date_min"]:
            date_min = min([value for value in (date_min, summary["date_min"]) if value])
            date_max = max(date_max, summary["date_max"])
    covered = sorted(set(news_symbols) & symbols)
    return {
        "path": str(path),
        "exists": path.exists(),
        "file_count": len(files),
        "required_columns_present": bool(files) and not missing_columns,
        "missing_columns": sorted(missing_columns),
        "symbol_coverage_count": len(covered),
        "covered_news_symbols": covered,
        "missing_news_symbols": sorted(set(news_symbols) - symbols),
        "date_min": date_min,
        "date_max": date_max,
        "adjusted_close_preferred": True,
    }


def _price_file_summary(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = set(reader.fieldnames or [])
            symbol = path.stem.upper()
            dates = []
            for row in reader:
                symbol = str(row.get("symbol") or row.get("Symbol") or symbol).strip().upper()
                value = str(row.get("date") or row.get("Date") or row.get("timestamp") or "")[:10]
                if value:
                    dates.append(value)
    except OSError:
        fieldnames = set()
        symbol = ""
        dates = []
    return {
        "symbol": symbol,
        "has_date": bool(fieldnames & {"date", "Date", "timestamp"}),
        "has_close_or_adjusted_close": bool(fieldnames & {"adjusted_close", "adj_close", "Adj Close", "AdjClose", "close", "Close"}),
        "date_min": min(dates, default=""),
        "date_max": max(dates, default=""),
    }


def _event_symbols(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        return sorted({
            str(row.get("symbol", "")).strip().upper()
            for row in csv.DictReader(handle)
            if str(row.get("symbol", "")).strip()
        })


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit local candidate price sources for stock-alpha news labels.")
    parser.add_argument("--event-dataset", required=True)
    parser.add_argument("--candidate-path", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    report = build_stock_alpha_news_price_source_audit(
        event_dataset_path=args.event_dataset,
        candidate_paths=args.candidate_path or DEFAULT_CANDIDATES,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
