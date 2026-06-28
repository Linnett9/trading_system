from __future__ import annotations

import csv
import json
import math
import ssl
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlencode
from urllib.request import Request, urlopen


YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
SOURCE_NAME = "yahoo_finance_chart_adjusted_close"


class YahooChartClient(Protocol):
    def fetch_chart(
        self,
        symbol: str,
        start: date,
        end: date,
    ) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class AdjustedPriceImportResult:
    symbol: str
    output_path: str
    row_count: int
    first_date: str | None
    last_date: str | None
    source: str
    adjusted_ohlc: bool


@dataclass(frozen=True)
class AdjustedPriceImportManifest:
    source: str
    output_dir: str
    download_date: str
    requested_symbols: list[str]
    requested_symbol_count: int
    imported_symbol_count: int
    failed_symbol_count: int
    symbols: list[dict[str, Any]]
    failed_symbols: list[dict[str, str]]
    research_only: bool = True
    trading_impact: str = "none"


class YahooFinanceChartClient:
    """Small Yahoo Finance chart API client for research reference data."""

    def fetch_chart(
        self,
        symbol: str,
        start: date,
        end: date,
    ) -> dict[str, Any]:
        period1 = int(
            datetime.combine(
                start,
                datetime.min.time(),
                tzinfo=timezone.utc,
            ).timestamp()
        )
        # Yahoo period2 is exclusive; add one day so the configured end date is included.
        period2 = int(
            datetime.combine(
                end + timedelta(days=1),
                datetime.min.time(),
                tzinfo=timezone.utc,
            ).timestamp()
        )
        query = urlencode({
            "period1": period1,
            "period2": period2,
            "interval": "1d",
            "events": "history",
            "includeAdjustedClose": "true",
        })
        url = f"{YAHOO_CHART_URL.format(symbol=symbol.upper())}?{query}"
        request = Request(
            url,
            headers={"User-Agent": "trading-system-research-adjusted-price-audit"},
        )
        with urlopen(request, timeout=30, context=_ssl_context()) as response:
            return json.loads(response.read().decode("utf-8"))


class YahooAdjustedPriceImporter:
    """Download Yahoo adjusted prices into local research-only CSV references."""

    def __init__(
        self,
        output_dir: str = "data/reference/adjusted_prices",
        *,
        client: YahooChartClient | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.client = client or YahooFinanceChartClient()

    def import_symbols(
        self,
        symbols: list[str],
        *,
        start: date,
        end: date,
        manifest_path: Path | None = None,
    ) -> AdjustedPriceImportManifest:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        imported: list[AdjustedPriceImportResult] = []
        failed: list[dict[str, str]] = []
        requested_symbols = sorted({str(item).upper() for item in symbols})
        for symbol in requested_symbols:
            try:
                rows = self._download_rows(symbol, start, end)
                if not rows:
                    raise RuntimeError("Yahoo returned no adjusted rows")
                output_path = self.output_dir / f"{symbol}.csv"
                self._write_csv(output_path, rows)
                imported.append(
                    AdjustedPriceImportResult(
                        symbol=symbol,
                        output_path=str(output_path),
                        row_count=len(rows),
                        first_date=rows[0]["date"],
                        last_date=rows[-1]["date"],
                        source=SOURCE_NAME,
                        adjusted_ohlc=True,
                    )
                )
            except Exception as exc:
                failed.append({"symbol": symbol, "reason": str(exc)})
        manifest = AdjustedPriceImportManifest(
            source=SOURCE_NAME,
            output_dir=str(self.output_dir),
            download_date=datetime.now(timezone.utc).date().isoformat(),
            requested_symbols=requested_symbols,
            requested_symbol_count=len(requested_symbols),
            imported_symbol_count=len(imported),
            failed_symbol_count=len(failed),
            symbols=[result.__dict__ for result in imported],
            failed_symbols=failed,
        )
        self.write_manifest(
            manifest_path or self.output_dir / "manifest.json",
            manifest,
        )
        return manifest

    def write_manifest(
        self,
        path: Path,
        manifest: AdjustedPriceImportManifest,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest.__dict__, indent=2), encoding="utf-8")

    def _download_rows(
        self,
        symbol: str,
        start: date,
        end: date,
    ) -> list[dict[str, Any]]:
        payload = self.client.fetch_chart(symbol, start, end)
        result = _chart_result(payload, symbol)
        timestamps = result.get("timestamp") or []
        quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
        adjclose = ((result.get("indicators") or {}).get("adjclose") or [{}])[0]
        rows = []
        for index, timestamp in enumerate(timestamps):
            raw_close = _item(quote.get("close"), index)
            adjusted_close = _item(adjclose.get("adjclose"), index)
            if raw_close is None or adjusted_close is None or raw_close <= 0:
                continue
            ratio = adjusted_close / raw_close
            row = {
                "symbol": symbol,
                "date": datetime.fromtimestamp(
                    int(timestamp),
                    tz=timezone.utc,
                ).date().isoformat(),
                "open": _scaled(_item(quote.get("open"), index), ratio),
                "high": _scaled(_item(quote.get("high"), index), ratio),
                "low": _scaled(_item(quote.get("low"), index), ratio),
                "close": adjusted_close,
                "adj_close": adjusted_close,
                "volume": _item(quote.get("volume"), index) or 0,
            }
            if _valid_row(row):
                rows.append(row)
        return sorted(rows, key=lambda row: row["date"])

    def _write_csv(self, path: Path, rows: list[dict[str, Any]]) -> None:
        fieldnames = [
            "symbol",
            "date",
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "volume",
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(
                {name: row.get(name) for name in fieldnames} for row in rows
            )


def _chart_result(payload: dict[str, Any], symbol: str) -> dict[str, Any]:
    chart = payload.get("chart", {})
    error = chart.get("error")
    if error:
        raise RuntimeError(f"Yahoo chart error for {symbol}: {error}")
    results = chart.get("result") or []
    if not results:
        raise RuntimeError(f"Yahoo chart returned no result for {symbol}")
    return results[0]


def _item(values: Any, index: int) -> float | None:
    if not isinstance(values, list) or index >= len(values):
        return None
    value = values[index]
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _scaled(value: float | None, ratio: float) -> float | None:
    return value * ratio if value is not None else None


def _valid_row(row: dict[str, Any]) -> bool:
    prices = [row.get("open"), row.get("high"), row.get("low"), row.get("close")]
    return all(_is_positive(value) for value in prices) and float(row.get("volume") or 0) >= 0


def _is_positive(value: Any) -> bool:
    try:
        return float(value) > 0.0
    except (TypeError, ValueError):
        return False


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())
