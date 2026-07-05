from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import date, datetime
import json
from pathlib import Path
from statistics import mean
from typing import Any


@dataclass(frozen=True)
class SymbolInventory:
    symbol: str
    path: str
    first_date: str | None
    last_date: str | None
    bar_count: int
    available_calendar_days: int
    history_years: float
    missing_or_gap_estimate: int | None
    zero_volume_days: int | None
    latest_close: float | None
    average_volume_63d: float | None
    average_dollar_volume_63d: float | None
    average_dollar_volume_252d: float | None
    passes_min_history_years: bool
    passes_latest_date_check: bool
    passes_liquidity_check: bool
    included_reason: str
    excluded_reason: str

    @property
    def included(self) -> bool:
        return (
            self.passes_min_history_years
            and self.passes_latest_date_check
            and self.passes_liquidity_check
        )


def build_data_inventory(
    parquet_dir: str | Path = "data/processed/stooq_parquet",
    output_dir: str | Path = "reports/ml",
    min_history_years: int = 9,
    max_latest_gap_days: int = 14,
    min_average_dollar_volume_252d: float = 50_000_000,
    as_of_date: date | None = None,
) -> list[SymbolInventory]:
    parquet_path = Path(parquet_dir)
    output_path = Path(output_dir)
    inventories = [
        inspect_symbol_file(
            path,
            min_history_years=min_history_years,
            max_latest_gap_days=max_latest_gap_days,
            min_average_dollar_volume_252d=min_average_dollar_volume_252d,
            as_of_date=as_of_date,
        )
        for path in sorted(parquet_path.glob("*.parquet"))
    ]
    write_inventory_reports(inventories, output_path, parquet_path)
    return inventories


def inspect_symbol_file(
    path: Path,
    min_history_years: int = 9,
    max_latest_gap_days: int = 14,
    min_average_dollar_volume_252d: float = 50_000_000,
    as_of_date: date | None = None,
) -> SymbolInventory:
    rows = _read_parquet_rows(path)
    symbol = path.stem.upper()
    dates = _date_values(rows)
    closes = _numeric_column(rows, ("close", "Close"))
    volumes = _numeric_column(rows, ("volume", "Volume"))

    bar_count = len(rows)
    first_date = dates[0] if dates else None
    last_date = dates[-1] if dates else None
    available_days = (last_date - first_date).days if first_date and last_date else 0
    history_years = available_days / 365.25 if available_days else 0.0
    latest_gap_days = (
        ((as_of_date or date.today()) - last_date).days if last_date else max_latest_gap_days + 1
    )
    expected_trading_days = int(available_days * 5 / 7) if available_days else 0
    gap_estimate = max(0, expected_trading_days - bar_count) if dates else None

    latest_close = closes[-1] if closes else None
    average_volume_63d = _tail_average(volumes, 63)
    average_dollar_volume_63d = _tail_average_product(closes, volumes, 63)
    average_dollar_volume_252d = _tail_average_product(closes, volumes, 252)
    zero_volume_days = (
        sum(1 for volume in volumes if volume == 0) if volumes else None
    )

    passes_history = history_years >= float(min_history_years)
    passes_latest = latest_gap_days <= int(max_latest_gap_days)
    passes_liquidity = (
        average_dollar_volume_252d is not None
        and average_dollar_volume_252d >= float(min_average_dollar_volume_252d)
    )
    excluded = _excluded_reason(
        passes_history,
        passes_latest,
        passes_liquidity,
        has_liquidity=average_dollar_volume_252d is not None,
    )
    return SymbolInventory(
        symbol=symbol,
        path=str(path),
        first_date=first_date.isoformat() if first_date else None,
        last_date=last_date.isoformat() if last_date else None,
        bar_count=bar_count,
        available_calendar_days=available_days,
        history_years=round(history_years, 4),
        missing_or_gap_estimate=gap_estimate,
        zero_volume_days=zero_volume_days,
        latest_close=latest_close,
        average_volume_63d=average_volume_63d,
        average_dollar_volume_63d=average_dollar_volume_63d,
        average_dollar_volume_252d=average_dollar_volume_252d,
        passes_min_history_years=passes_history,
        passes_latest_date_check=passes_latest,
        passes_liquidity_check=passes_liquidity,
        included_reason="passes_all_filters" if not excluded else "",
        excluded_reason=excluded,
    )


def write_inventory_reports(
    inventories: list[SymbolInventory],
    output_dir: Path,
    parquet_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": "stooq_parquet_inventory",
        "parquet_dir": str(parquet_dir),
        "symbol_count": len(inventories),
        "included_count": sum(1 for item in inventories if item.included),
        "symbols": [asdict(item) | {"included": item.included} for item in inventories],
        "research_only": True,
        "trading_impact": "none",
    }
    (output_dir / "data_inventory.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    csv_path = output_dir / "symbol_coverage.csv"
    fieldnames = list(asdict(inventories[0]).keys()) + ["included"] if inventories else [
        "symbol",
        "included",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in inventories:
            writer.writerow(asdict(item) | {"included": item.included})


def _read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "ML data inventory requires pyarrow. Install dependencies with: "
            "python -m pip install -r requirements.txt"
        ) from exc
    table = pq.read_table(path)
    columns = table.to_pydict()
    names = list(columns)
    return [
        {name: columns[name][index] for name in names}
        for index in range(table.num_rows)
    ]


def _date_values(rows: list[dict[str, Any]]) -> list[date]:
    dates = []
    for index, row in enumerate(rows):
        value = _first_present(row, ("timestamp", "Timestamp", "date", "Date"))
        if value is None:
            value = _first_present(row, ("__index_level_0__", "index"))
        parsed = _to_date(value)
        if parsed is not None:
            dates.append(parsed)
    return sorted(dates)


def _numeric_column(rows: list[dict[str, Any]], names: tuple[str, ...]) -> list[float]:
    values = []
    for row in rows:
        value = _first_present(row, names)
        if value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def _first_present(row: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in row:
            return row[name]
    return None


def _to_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value)
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _tail_average(values: list[float], count: int) -> float | None:
    if not values:
        return None
    return float(mean(values[-count:]))


def _tail_average_product(
    closes: list[float],
    volumes: list[float],
    count: int,
) -> float | None:
    if not closes or not volumes:
        return None
    pairs = list(zip(closes, volumes))[-count:]
    if not pairs:
        return None
    return float(mean(close * volume for close, volume in pairs))


def _excluded_reason(
    passes_history: bool,
    passes_latest: bool,
    passes_liquidity: bool,
    has_liquidity: bool,
) -> str:
    reasons = []
    if not passes_history:
        reasons.append("insufficient_history")
    if not passes_latest:
        reasons.append("stale_latest_date")
    if not has_liquidity:
        reasons.append("missing_liquidity_data")
    elif not passes_liquidity:
        reasons.append("insufficient_liquidity")
    return ", ".join(reasons)
