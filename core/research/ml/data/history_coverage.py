from __future__ import annotations

import json
from datetime import date
from pathlib import Path


def assess_history_coverage(
    candles_by_symbol: dict[str, list[object]],
    required_years: int,
    tolerance_days: int = 10,
    source_metadata: dict[str, dict] | None = None,
) -> dict:
    source_metadata = source_metadata or {}
    symbol_ranges = {}
    for symbol, candles in sorted(candles_by_symbol.items()):
        dates = sorted(
            candle.timestamp.date() for candle in candles if candle.close > 0
        )
        symbol_ranges[symbol] = {
            "first_date": dates[0].isoformat() if dates else None,
            "last_date": dates[-1].isoformat() if dates else None,
            "bar_count": len(dates),
            "request": source_metadata.get(symbol, {}),
        }

    available = [value for value in symbol_ranges.values() if value["first_date"]]
    common_start = max(date.fromisoformat(value["first_date"]) for value in available) if available else None
    common_end = min(date.fromisoformat(value["last_date"]) for value in available) if available else None
    available_days = (common_end - common_start).days if common_start and common_end else 0
    required_days = required_years * 365
    return {
        "required_years": required_years,
        "required_calendar_days": required_days,
        "tolerance_days": tolerance_days,
        "common_start_date": common_start.isoformat() if common_start else None,
        "common_end_date": common_end.isoformat() if common_end else None,
        "available_calendar_days": available_days,
        "coverage_sufficient": (
            len(available) == len(symbol_ranges)
            and available_days >= required_days - tolerance_days
        ),
        "symbols": symbol_ranges,
    }


def write_history_coverage_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
