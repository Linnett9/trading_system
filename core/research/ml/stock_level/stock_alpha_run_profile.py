from __future__ import annotations

from typing import Any

from core.research.framework.config import StockLevelResearchConfig


def apply_stock_alpha_run_profile(
    rows: list[dict[str, Any]], settings: StockLevelResearchConfig
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return a stable research subset; benchmark/full inputs are unchanged."""
    if settings.run_size != "dev":
        return rows, _counts(rows, settings.run_size)
    dates = sorted({str(row.get("rebalance_date", "")) for row in rows if row.get("rebalance_date")})
    selected_dates = (
        dates[-settings.dev_max_dates :]
        if settings.dev_recent_dates_only
        else dates[: settings.dev_max_dates]
    )
    allowed_dates = set(selected_dates)
    candidates = [row for row in rows if str(row.get("rebalance_date", "")) in allowed_dates]
    symbols = sorted({str(row.get("symbol", "")).upper() for row in candidates if row.get("symbol")})
    allowed_symbols = set(symbols[: settings.dev_max_symbols])
    subset = [
        row for row in candidates
        if str(row.get("symbol", "")).upper() in allowed_symbols
    ]
    subset.sort(key=lambda row: (str(row.get("rebalance_date", "")), str(row.get("symbol", "")).upper()))
    return subset, _counts(subset, settings.run_size)


def _counts(rows: list[dict[str, Any]], run_size: str) -> dict[str, Any]:
    return {
        "run_size": run_size,
        "effective_row_count": len(rows),
        "effective_date_count": len({row.get("rebalance_date") for row in rows if row.get("rebalance_date")}),
        "effective_symbol_count": len({str(row.get("symbol", "")).upper() for row in rows if row.get("symbol")}),
    }
